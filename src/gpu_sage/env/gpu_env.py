"""Gymnasium environment for RL-based GPU scheduling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gpu_sage.core.cluster import Cluster
from gpu_sage.core.models import Job
from gpu_sage.core.simulator import Simulator
from gpu_sage.schedulers.fcfs import FCFSScheduler
from gpu_sage.workloads.generator import SyntheticWorkload, WorkloadConfig


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the shaped scheduling reward."""

    throughput: float = 2.0
    waiting: float = 0.02
    utilization: float = 0.25
    fragmentation: float = 0.25
    invalid_action: float = 1.0
    idle: float = 0.01


class GPUSchedulingEnv(gym.Env):
    """Event-driven single-agent GPU scheduling environment.

    Each RL step either launches one waiting job or chooses NOOP, which advances
    the simulator to the next event. The agent never directly manipulates GPUs;
    the environment owns deterministic resource allocation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory_gb: float = 80.0,
        max_jobs: int = 16,
        episode_jobs: int = 100,
        workload_config: WorkloadConfig | None = None,
        reward_config: RewardConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")

        self.num_gpus = num_gpus
        self.gpu_memory_gb = gpu_memory_gb
        self.max_jobs = max_jobs
        self.episode_jobs = episode_jobs
        self.workload_config = workload_config or WorkloadConfig()
        self.reward_config = reward_config or RewardConfig()
        self._seed = seed

        self.action_space = spaces.Discrete(max_jobs + 1)  # final action = NOOP
        self.observation_space = spaces.Dict(
            {
                "cluster": spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32),
                "gpus": spaces.Box(0.0, 1.0, shape=(num_gpus, 3), dtype=np.float32),
                "jobs": spaces.Box(0.0, 1.0, shape=(max_jobs, 8), dtype=np.float32),
                "job_mask": spaces.MultiBinary(max_jobs),
            }
        )

        self.rng = np.random.default_rng(seed)
        self.sim: Simulator | None = None
        self.jobs: list[Job] = []
        self.candidate_ids: list[int | None] = [None] * max_jobs
        self.last_decision_time = 0.0
        self._episode_done = False

    @property
    def noop_action(self) -> int:
        return self.max_jobs

    def _generate_jobs(self) -> list[Job]:
        # Derive a fresh integer seed from the episode RNG so repeated resets are reproducible.
        workload_seed = int(self.rng.integers(0, 2**32 - 1))
        generator = SyntheticWorkload(self.workload_config, seed=workload_seed)
        return generator.generate(self.episode_jobs)

    def _ensure_event_ready(self) -> None:
        """Advance through events until the agent has a decision to make."""
        assert self.sim is not None
        while not self.sim.waiting_jobs and len(self.sim.event_queue) > 0:
            self.sim.step_until_next_event(schedule=False)
        # Process completions/arrivals at exactly the same timestamp together.
        while len(self.sim.event_queue) > 0 and self.sim.event_queue.peek() is not None and self.sim.event_queue.peek().time == self.sim.current_time:
            self.sim.step_until_next_event(schedule=False)

    def _candidate_jobs(self) -> list[Job]:
        assert self.sim is not None
        waiting = list(self.sim.waiting_jobs.values())
        # Stable deterministic ranking keeps the action meaning consistent.
        waiting.sort(key=lambda j: (-j.priority, j.arrival_time, j.job_id))
        return waiting[: self.max_jobs]

    def _fragmentation(self) -> float:
        assert self.sim is not None
        free = self.sim.cluster.free_gpu_count
        if free == 0:
            return 0.0
        # For now, count compatible capacity for each waiting job. Lower is worse.
        feasible_counts = [len(self.sim.cluster.feasible_gpu_ids(job)) for job in self.sim.waiting_jobs.values()]
        if not feasible_counts:
            return 0.0
        max_allocatable = max(feasible_counts)
        return float(max(0.0, 1.0 - min(max_allocatable, free) / free))

    def _observation(self) -> dict[str, np.ndarray]:
        assert self.sim is not None
        candidates = self._candidate_jobs()
        self.candidate_ids = [job.job_id for job in candidates] + [None] * (self.max_jobs - len(candidates))

        time_norm = min(self.sim.current_time / max(1.0, self.workload_config.max_duration * 10), 1.0)
        # Use actual job count for eval mode (fixed workload may differ from episode_jobs)
        total_jobs = len(self.jobs) if self.jobs else self.episode_jobs
        cluster = np.array(
            [
                time_norm,
                self.sim.cluster.free_gpu_count / self.num_gpus,
                self.sim.cluster.utilization(self.sim.running_jobs),
                min(len(self.sim.waiting_jobs) / max(self.max_jobs, 1), 1.0),
                min(len(self.sim.running_jobs) / self.num_gpus, 1.0),
                min(len(self.sim.completed_jobs) / max(total_jobs, 1), 1.0),
            ],
            dtype=np.float32,
        )

        gpus = np.zeros((self.num_gpus, 3), dtype=np.float32)
        for gpu in self.sim.cluster.gpus:
            gpus[gpu.gpu_id] = [
                1.0 if gpu.is_free else 0.0,
                0.0 if gpu.is_free else 1.0,
                min(gpu.memory_gb / 80.0, 1.0),
            ]

        jobs = np.zeros((self.max_jobs, 8), dtype=np.float32)
        mask = np.zeros(self.max_jobs, dtype=np.int8)
        for idx, job in enumerate(candidates):
            mask[idx] = 1
            compatible = self.sim.cluster.feasible_gpu_ids(job)
            jobs[idx] = [
                min(job.gpu_count / self.num_gpus, 1.0),
                min(job.gpu_memory_gb / 80.0, 1.0),
                min(job.duration / max(self.workload_config.max_duration, 1), 1.0),
                min(job.priority / max(self.workload_config.max_priority, 1), 1.0),
                min(max(0.0, self.sim.current_time - job.arrival_time) / max(self.workload_config.max_duration, 1), 1.0),
                1.0 if job.preemptible else 0.0,
                min(len(compatible) / self.num_gpus, 1.0),
                1.0 if self.sim.cluster.can_allocate(job) else 0.0,
            ]

        return {"cluster": cluster, "gpus": gpus, "jobs": jobs, "job_mask": mask}

    def action_mask(self) -> np.ndarray:
        """Return valid-action mask for MaskablePPO/similar algorithms."""
        assert self.sim is not None
        candidates = self._candidate_jobs()
        mask = np.zeros(self.max_jobs + 1, dtype=bool)
        for idx, job in enumerate(candidates):
            mask[idx] = self.sim.cluster.can_allocate(job)
        # NOOP is always valid while there are future events.
        mask[self.noop_action] = len(self.sim.event_queue) > 0
        return mask

    def action_masks(self) -> np.ndarray:
        """MaskablePPO-compatible action mask method."""
        return self.action_mask()

    def _reward(
        self,
        previous_time: float,
        previous_gpu_time: float,
        completed_before: int,
        invalid: bool,
    ) -> float:
        assert self.sim is not None
        cfg = self.reward_config
        elapsed = self.sim.current_time - previous_time

        # GPU utilization over the actual transition interval. This avoids
        # incorrectly giving zero utilization when a completion event occurs
        # exactly at the end of a busy interval.
        gpu_time_delta = self.sim.gpu_time_used - previous_gpu_time
        capacity = elapsed * self.num_gpus
        interval_utilization = gpu_time_delta / capacity if capacity > 0 else 0.0

        reward = cfg.utilization * interval_utilization * elapsed
        reward -= cfg.waiting * sum(
            max(0.0, self.sim.current_time - j.arrival_time)
            for j in self.sim.waiting_jobs.values()
        )
        reward -= cfg.fragmentation * self._fragmentation() * elapsed
        if elapsed > 0 and interval_utilization == 0.0:
            reward -= cfg.idle * elapsed

        completed_delta = len(self.sim.completed_jobs) - completed_before
        reward += cfg.throughput * completed_delta
        if invalid:
            reward -= cfg.invalid_action
        return float(reward)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self._seed is not None:
            self.rng = np.random.default_rng(self._seed)

        # Evaluation mode: accept pre-generated workload so PPO sees the *same* W0 as baselines.
        # Training mode: generate stochastic workload as before.
        if options is not None and "fixed_jobs" in options and options["fixed_jobs"] is not None:
            self.jobs = deepcopy(options["fixed_jobs"])
        elif options is not None and "workload" in options and options["workload"] is not None:
            self.jobs = deepcopy(options["workload"])
        else:
            self.jobs = self._generate_jobs()
        self.sim = Simulator(Cluster.homogeneous(self.num_gpus, self.gpu_memory_gb), FCFSScheduler())
        self.sim.load_jobs(deepcopy(self.jobs))
        self.last_decision_time = 0.0
        self._episode_done = False
        self._ensure_event_ready()
        obs = self._observation()
        info = {"action_mask": self.action_mask(), "time": self.sim.current_time}
        return obs, info

    def step(self, action: int):
        assert self.sim is not None
        if self._episode_done:
            raise RuntimeError("Episode is done; call reset() before step().")

        previous_time = self.sim.current_time
        previous_gpu_time = self.sim.gpu_time_used
        completed_before = len(self.sim.completed_jobs)
        invalid = False
        launched: list[int] = []

        if action == self.noop_action:
            if len(self.sim.event_queue) == 0:
                self._episode_done = True
            else:
                self.sim.step_until_next_event(schedule=False)
                self._ensure_event_ready()
        elif 0 <= action < self.max_jobs:
            candidates = self._candidate_jobs()
            if action >= len(candidates) or not self.sim.cluster.can_allocate(candidates[action]):
                invalid = True
            else:
                job = candidates[action]
                self.sim.schedule_job(job.job_id)
                launched.append(job.job_id)
                # One decision per action. The next step may schedule another job at the same timestamp.
                if not self.sim.waiting_jobs:
                    self._ensure_event_ready()
        else:
            invalid = True

        terminated = len(self.sim.event_queue) == 0 and not self.sim.waiting_jobs and not self.sim.running_jobs
        truncated = False
        self._episode_done = terminated or truncated
        reward = self._reward(previous_time, previous_gpu_time, completed_before, invalid)
        obs = self._observation()
        info = {
            "action_mask": self.action_mask() if not self._episode_done else np.ones(self.max_jobs + 1, dtype=bool),
            "time": self.sim.current_time,
            "launched_jobs": launched,
            "invalid_action": invalid,
            "waiting_jobs": len(self.sim.waiting_jobs),
            "running_jobs": len(self.sim.running_jobs),
            "completed_jobs": len(self.sim.completed_jobs),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        assert self.sim is not None
        print(self.sim.snapshot())
