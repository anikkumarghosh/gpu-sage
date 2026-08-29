"""Rigorous RL-vs-baseline benchmark runner — now with PPO integration.

Critical correctness property: for each seed/scenario the *identical* workload is
evaluated by every scheduler from a fresh simulator state. No scheduler may
mutate the workload seen by another.

Architecture:

    Workload (seeded) -> Simulator -> Scheduler interface
                                  ├── FCFS
                                  ├── SJF
                                  ├── Priority
                                  ├── Best-Fit
                                  └── PPO (via env-based deterministic inference)
                        -> Metrics -> Benchmark aggregation (mean±std)
                        -> Per-job records (for P50/P95/P99, fairness, starvation)
                        -> PPO decision logs (for dashboard)

The simulator remains the single source of truth for time/arrival/completion/GPU allocation.
PPO evaluation uses deterministic inference (MaskablePPO.predict(deterministic=True)).
Training vs Evaluation are separated: training generates stochastic workloads,
evaluation receives the *same* fixed workload W0 as baselines.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Allow duplicate OpenMP for torch on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from gpu_sage.core.cluster import Cluster
from gpu_sage.core.simulator import Simulator
from gpu_sage.evaluation.metrics import Metrics, compute_metrics
from gpu_sage.schedulers.fcfs import FCFSScheduler
from gpu_sage.schedulers.heuristics import BestFitScheduler, PriorityScheduler, SJFScheduler, TopologyBestFitScheduler
from gpu_sage.schedulers.rl import RandomScheduler, RLScheduler
from gpu_sage.workloads.generator import SCENARIO_NAMES, generate_workload, get_scenario_config

# ---------------------------------------------------------------------------
# Scheduler registry — all schedulers share the common Simulator interface.
# ---------------------------------------------------------------------------

SCHEDULER_FACTORIES: dict[str, Callable[[], object]] = {
    "FCFS": lambda: FCFSScheduler(),
    "SJF": lambda: SJFScheduler(),
    "Priority": lambda: PriorityScheduler(),
    "BestFit": lambda: BestFitScheduler(),
    "TopologyBestFit": lambda: TopologyBestFitScheduler(),
    "Random": lambda: RandomScheduler(seed=0),
    "PPO": lambda: RLScheduler(),
}

DEFAULT_SCHEDULERS: list[str] = ["FCFS", "SJF", "Priority", "BestFit"]


def available_schedulers() -> list[str]:
    return list(SCHEDULER_FACTORIES.keys())


def make_scheduler(name: str):
    if name not in SCHEDULER_FACTORIES:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(SCHEDULER_FACTORIES)}")
    return SCHEDULER_FACTORIES[name]()


# ---------------------------------------------------------------------------
# Per-job record helper — preserves full job-level data for later analysis.
# ---------------------------------------------------------------------------

def _cluster_for_scenario(scenario: str, num_gpus: int, gpu_memory_gb: float, cluster=None, cluster_config_path=None):
    """Resolve cluster for a scenario, preserving homogeneous default.

    - If cluster is supplied, use it.
    - If cluster_config_path supplied, load heterogeneous yaml.
    - If scenario is heterogeneous/topology_sensitive/mixed_ml, default to heterogeneous 8GPU.
    - Else homogeneous.
    """
    if cluster is not None:
        return cluster
    if cluster_config_path is not None:
        from pathlib import Path
        import yaml

        data = yaml.safe_load(Path(cluster_config_path).read_text()) or {}
        c = data.get("cluster", data)
        specs = c.get("gpus")
        if specs:
            from gpu_sage.core.cluster import Cluster
            from gpu_sage.core.topology import Topology

            cl = Cluster.heterogeneous(specs)
            topo_cfg = c.get("topology", {})
            ttype = str(topo_cfg.get("type", "two_group"))
            if ttype == "two_group":
                cl.topology = Topology.two_group(cl.total_gpus, group_size=int(topo_cfg.get("group_size", 4)))
            elif ttype == "fully_connected":
                cl.topology = Topology.fully_connected_nvlink(cl.total_gpus)
            return cl
    if scenario in ("heterogeneous", "topology_sensitive", "mixed_ml"):
        # Default heterogeneous 8GPU matching configs/heterogeneous_8gpu.yaml
        from gpu_sage.core.cluster import Cluster

        return Cluster.heterogeneous(
            [
                {"gpu_type": "A100_80GB", "memory_gb": 80},
                {"gpu_type": "A100_80GB", "memory_gb": 80},
                {"gpu_type": "A100_40GB", "memory_gb": 40},
                {"gpu_type": "A100_40GB", "memory_gb": 40},
                {"gpu_type": "V100_32GB", "memory_gb": 32},
                {"gpu_type": "V100_32GB", "memory_gb": 32},
                {"gpu_type": "T4_16GB", "memory_gb": 16},
                {"gpu_type": "T4_16GB", "memory_gb": 16},
            ]
        )
    from gpu_sage.core.cluster import Cluster

    return Cluster.homogeneous(num_gpus, memory_gb=gpu_memory_gb)


def job_to_record(
    job,
    scheduler: str,
    scenario: str,
    seed: int,
) -> dict:
    return {
        "job_id": job.job_id,
        "arrival_time": job.arrival_time,
        "start_time": job.start_time,
        "completion_time": job.completion_time,
        "waiting_time": job.waiting_time,
        "turnaround_time": job.turnaround_time,
        "gpu_requirement": job.gpu_count,
        "memory_requirement": job.gpu_memory_gb,
        "execution_time": job.duration,
        "priority": job.priority,
        "scheduler": scheduler,
        "scenario": scenario,
        "seed": seed,
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "assigned_gpus": ",".join(map(str, job.assigned_gpus)) if job.assigned_gpus else "",
        "placement_penalty": float(getattr(job, "placement_penalty", 1.0)),
        "communication_cost": float(getattr(job, "communication_cost", 0.0)),
        "topology_sensitive": bool(getattr(job, "topology_sensitive", False)),
        "preferred_gpu_type": str(getattr(job, "preferred_gpu_type", "") or ""),
        "required_gpu_type": str(getattr(job, "required_gpu_type", "") or ""),
    }


# ---------------------------------------------------------------------------
# PPO evaluation via Gymnasium env — deterministic, same workload as baselines
# ---------------------------------------------------------------------------

def evaluate_ppo_fixed_workload(
    jobs: list,
    model_path: str | Path | None = None,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    max_jobs: int = 16,
    workload_config=None,
    reward_config=None,
    seed: int = 0,
    cluster=None,
    cluster_config_path: str | Path | None = None,
    heterogeneous_obs: bool = False,
) -> tuple[Metrics, list[dict], list[dict], dict]:
    """Evaluate PPO on a fixed workload via env loop (deterministic).

    Args:
        jobs: Fixed workload (already generated via generate_workload).
        model_path: Path to MaskablePPO zip. If None or not found, falls back to deterministic heuristic.
        num_gpus: Cluster size.
        gpu_memory_gb: Per-GPU memory.
        max_jobs: Env max_jobs (action space size -1).
        workload_config: WorkloadConfig for observation normalization (scenario preset).
        seed: Seed for env (not used for generation when fixed_jobs given).

    Returns:
        (Metrics, per_job_records, decision_logs, stats)
        stats contains: scheduling_decisions, noop_actions, invalid_actions, completed_jobs,
                        action_distribution, avg_reward, total_reward, steps
    """
    from copy import deepcopy

    # Late import to avoid loading torch unless needed
    # Use fixed workload mode: env.reset(options={"fixed_jobs": jobs})
    from gpu_sage.env.gpu_env import GPUSchedulingEnv

    if workload_config is None:
        try:
            # Try to infer from first job? fallback
            from gpu_sage.workloads.generator import WorkloadConfig
            workload_config = WorkloadConfig()
        except Exception:
            workload_config = None

    # Resolve cluster for PPO env (heterogeneous if scenario requires it)
    # If caller passed explicit cluster, use it; else infer from scenario.
    ppo_cluster = cluster
    if ppo_cluster is None and cluster_config_path is not None:
        from pathlib import Path as _P

        import yaml

        data = yaml.safe_load(_P(cluster_config_path).read_text()) or {}
        c = data.get("cluster", data)
        specs = c.get("gpus")
        if specs:
            from gpu_sage.core.cluster import Cluster as _Cl
            from gpu_sage.core.topology import Topology as _Tp

            ppo_cluster = _Cl.heterogeneous(specs)
            topo_cfg = c.get("topology", {})
            ttype = str(topo_cfg.get("type", "two_group"))
            if ttype == "two_group":
                ppo_cluster.topology = _Tp.two_group(ppo_cluster.total_gpus, group_size=int(topo_cfg.get("group_size", 4)))
            elif ttype == "fully_connected":
                ppo_cluster.topology = _Tp.fully_connected_nvlink(ppo_cluster.total_gpus)
    # Auto-hetero for new scenarios if no explicit cluster
    if ppo_cluster is None and workload_config is not None:
        # If scenario's workload suggests hetero, create hetero cluster
        # Detect via topology_sensitive_fraction >0
        if getattr(workload_config, "topology_sensitive_fraction", 0) > 0:
            ppo_cluster = _cluster_for_scenario(
                getattr(workload_config, "scenario", "") or "heterogeneous", num_gpus, gpu_memory_gb
            )
            # If still homogeneous but scenario is hetero-like, keep hetero
            if ppo_cluster and len(set(g.gpu_type for g in ppo_cluster.gpus)) == 1:
                # force heterogeneous for hetero scenarios
                if workload_config.topology_sensitive_fraction > 0.1:
                    from gpu_sage.core.cluster import Cluster as _Cl2

                    ppo_cluster = _Cl2.heterogeneous(
                        [
                            {"gpu_type": "A100_80GB", "memory_gb": 80},
                            {"gpu_type": "A100_80GB", "memory_gb": 80},
                            {"gpu_type": "A100_40GB", "memory_gb": 40},
                            {"gpu_type": "A100_40GB", "memory_gb": 40},
                            {"gpu_type": "V100_32GB", "memory_gb": 32},
                            {"gpu_type": "V100_32GB", "memory_gb": 32},
                            {"gpu_type": "T4_16GB", "memory_gb": 16},
                            {"gpu_type": "T4_16GB", "memory_gb": 16},
                        ]
                    )
    env = GPUSchedulingEnv(
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        max_jobs=max_jobs,
        episode_jobs=len(jobs),
        workload_config=workload_config,
        reward_config=reward_config,
        seed=seed,
        cluster=ppo_cluster,
        heterogeneous_obs=heterogeneous_obs,
    )

    model = None
    if model_path is not None:
        mp = Path(model_path)
        # Support directory containing final_model.zip or direct file
        candidates = [mp, mp / "final_model.zip", mp / "final_model", mp / "best" / "best_model.zip"]
        found = None
        for c in candidates:
            if c.exists() and c.is_file():
                found = c
                break
            # sb3 saves without .zip extension handling, check for .zip existence
            if Path(str(c) + ".zip").exists():
                found = Path(str(c) + ".zip")
                break
        if found is None and mp.exists():
            found = mp
        if found is not None and found.exists():
            try:
                from sb3_contrib import MaskablePPO
                model = MaskablePPO.load(str(found))
            except Exception as e:
                # Fallback if load fails
                print(f"[PPO eval] Warning: could not load model {found}: {e}, using heuristic fallback")
                model = None

    # Deepcopy for isolation
    fixed = deepcopy(jobs)
    obs, info = env.reset(options={"fixed_jobs": fixed})
    decision_logs: list[dict] = []
    total_reward = 0.0
    steps = 0
    noop_count = 0
    invalid_count = 0
    scheduling_decisions = 0
    action_dist: dict[int, int] = {}

    # For deterministic fallback when no model: pick first feasible
    terminated = False
    truncated = False
    while not (terminated or truncated):
        # Action mask from env
        mask = info.get("action_mask", env.action_mask()) if isinstance(info, dict) else env.action_mask()
        # Ensure mask is numpy bool
        import numpy as np
        mask_arr = np.array(mask, dtype=bool)
        # Choose action
        if model is not None:
            try:
                action, _ = model.predict(obs, action_masks=mask_arr, deterministic=True)
                action = int(action)
            except Exception as e:
                # Fallback to first feasible on predict failure
                feasible = np.where(mask_arr)[0]
                action = int(feasible[0]) if len(feasible) > 0 else int(env.noop_action)
        else:
            # Deterministic fallback: first feasible (not NOOP) else NOOP
            feasible = np.where(mask_arr)[0]
            # Prefer feasible jobs over NOOP; NOOP is last index
            # Filter out NOOP if there's feasible job
            job_feasible = [a for a in feasible if a != env.noop_action]
            if job_feasible:
                action = int(job_feasible[0])
            elif mask_arr[env.noop_action]:
                action = int(env.noop_action)
            else:
                # Should not happen; pick NOOP
                action = int(env.noop_action)

        # Log before step
        free_gpus = env.sim.cluster.free_gpu_count if env.sim else 0
        util = env.sim.cluster.utilization(env.sim.running_jobs) if env.sim else 0.0
        queue_len = len(env.sim.waiting_jobs) if env.sim else 0
        # Candidate job id for this action (for logging)
        sel_job_id = None
        if action < env.max_jobs:
            candidates = env._candidate_jobs()  # type: ignore
            if action < len(candidates):
                sel_job_id = candidates[action].job_id

        action_dist[action] = action_dist.get(action, 0) + 1
        if action == env.noop_action:
            noop_count += 1
        else:
            # Will be validated as scheduling_decisions if valid
            pass

        obs_next, reward, terminated, truncated, info_next = env.step(action)
        total_reward += float(reward)
        steps += 1
        if info_next.get("invalid_action"):
            invalid_count += 1
        else:
            if action != env.noop_action:
                scheduling_decisions += 1

        # Include reward components if available
        comps = info_next.get("reward_components", {})
        decision_logs.append(
            {
                "step": steps,
                "simulation_time": float(env.sim.current_time) if env.sim else 0.0,
                "queue_length": int(queue_len),
                "free_gpus": int(free_gpus),
                "gpu_utilization": float(util),
                "action": int(action),
                "selected_job_id": sel_job_id,
                "reward": float(reward),
                "throughput_reward": float(comps.get("throughput_reward", 0)),
                "waiting_penalty": float(comps.get("waiting_penalty", 0)),
                "utilization_reward": float(comps.get("utilization_reward", 0)),
                "fragmentation_penalty": float(comps.get("fragmentation_penalty", 0)),
                "idle_penalty": float(comps.get("idle_penalty", 0)),
                "invalid_penalty": float(comps.get("invalid_penalty", 0)),
                "invalid_action": bool(info_next.get("invalid_action", False)),
                "waiting_jobs": int(info_next.get("waiting_jobs", 0)),
                "running_jobs": int(info_next.get("running_jobs", 0)),
                "completed_jobs": int(info_next.get("completed_jobs", 0)),
            }
        )

        obs, info = obs_next, info_next
        # Safety break to avoid infinite loop
        if steps > 10000:
            break

    # Collect final jobs from simulator store
    final_jobs = []
    if env.sim is not None:
        # Use job store which has all jobs with final times
        final_jobs = list(env.sim._job_store.values())
    else:
        final_jobs = fixed

    # Compute metrics from final jobs
    # Need gpu_time_used and simulated_time from env.sim
    gpu_time_used = env.sim.gpu_time_used if env.sim else 0.0
    sim_time = env.sim.current_time if env.sim else 0.0
    total_gpus = env.sim.cluster.total_gpus if env.sim and env.sim.cluster else num_gpus
    metrics = compute_metrics(
        final_jobs,
        total_gpus=total_gpus,
        simulated_time=sim_time,
        gpu_time_used=gpu_time_used,
        scheduling_decisions=scheduling_decisions,
        invalid_scheduling_attempts=invalid_count,
        per_gpu_memory_gb=gpu_memory_gb,
    )
    per_job_records = [job_to_record(j, scheduler="PPO", scenario="unknown", seed=seed) for j in final_jobs]
    # Fix scenario/seed in per_job after; caller will override scenario field
    stats = {
        "scheduling_decisions": int(scheduling_decisions),
        "noop_actions": int(noop_count),
        "invalid_actions": int(invalid_count),
        "completed_jobs": int(metrics.completed_jobs),
        "action_distribution": action_dist,
        "total_reward": float(total_reward),
        "avg_reward": float(total_reward / max(steps, 1)),
        "steps": int(steps),
    }
    return metrics, per_job_records, decision_logs, stats


# ---------------------------------------------------------------------------
# Single-seed evaluation — identical workload for every scheduler.
# ---------------------------------------------------------------------------

def run_single_seed(
    scenario: str,
    seed: int,
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    ppo_model_path: str | Path | None = None,
    workload_config=None,
    cluster=None,
    cluster_config_path: str | Path | None = None,
    heterogeneous_obs: bool = False,
) -> dict[str, Metrics]:
    """Run one seed/scenario against all schedulers on the SAME workload.

    For heuristic schedulers, uses Simulator+Scheduler.select loop.
    For PPO, uses env-based deterministic evaluation on the *same* base workload.

    Returns:
        Dict scheduler_name -> Metrics
    """
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    base_jobs = generate_workload(scenario=scenario, seed=seed, count=num_jobs)
    if workload_config is None:
        try:
            workload_config = get_scenario_config(scenario)
        except Exception:
            workload_config = None
    results: dict[str, Metrics] = {}
    # Resolve cluster once per seed/scenario so all schedulers share same cluster+W0
    base_cluster = _cluster_for_scenario(scenario, num_gpus, gpu_memory_gb, cluster=cluster, cluster_config_path=cluster_config_path)
    for name in schedulers:
        if name == "PPO":
            # PPO path via env — same base_jobs
            metrics, _, _, _ = evaluate_ppo_fixed_workload(
                jobs=copy.deepcopy(base_jobs),
                model_path=ppo_model_path,
                num_gpus=num_gpus,
                gpu_memory_gb=gpu_memory_gb,
                workload_config=workload_config,
                seed=seed,
                cluster=base_cluster,
                cluster_config_path=cluster_config_path,
                heterogeneous_obs=heterogeneous_obs,
            )
            # Fix per-job scenario/seed fields already handled in metrics generation
            results[name] = metrics
        else:
            scheduler = make_scheduler(name)
            jobs_copy = copy.deepcopy(base_jobs)
            # Deepcopy cluster so per-scheduler runs are isolated
            import copy as _cp

            cl = _cp.deepcopy(base_cluster)
            sim = Simulator(cluster=cl, scheduler=scheduler)
            sim.load_jobs(jobs_copy)
            sim.run()
            metrics = compute_metrics(
                jobs_copy,
                total_gpus=cl.total_gpus,
                simulated_time=sim.current_time,
                gpu_time_used=sim.gpu_time_used,
                scheduling_decisions=len(sim.completed_jobs),
                invalid_scheduling_attempts=0,
                per_gpu_memory_gb=max(g.memory_gb for g in cl.gpus),
            )
            results[name] = metrics
    return results


def run_single_seed_detailed(
    scenario: str,
    seed: int,
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    ppo_model_path: str | Path | None = None,
    workload_config=None,
) -> tuple[dict[str, Metrics], dict[str, list[dict]]]:
    """Like run_single_seed but also returns per-job records (backward compatible).

    For PPO with detailed logs, use run_single_seed_detailed_with_logs().
    """
    metrics_map, per_job_map, _ = run_single_seed_detailed_with_logs(
        scenario=scenario,
        seed=seed,
        schedulers=schedulers,
        num_jobs=num_jobs,
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        ppo_model_path=ppo_model_path,
        workload_config=workload_config,
    )
    return metrics_map, per_job_map


def run_single_seed_detailed_with_logs(
    scenario: str,
    seed: int,
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    ppo_model_path: str | Path | None = None,
    workload_config=None,
    cluster=None,
    cluster_config_path: str | Path | None = None,
    heterogeneous_obs: bool = False,
) -> tuple[dict[str, Metrics], dict[str, list[dict]], dict[str, list[dict]]]:
    """Like run_single_seed but also returns per-job records and PPO decision logs.

    Returns:
        (metrics_dict, per_job_dict, ppo_logs_dict)
        per_job_dict[scheduler] = list of per-job record dicts
        ppo_logs_dict["PPO"] = list of decision logs if PPO was run, else empty
    """
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    base_jobs = generate_workload(scenario=scenario, seed=seed, count=num_jobs)
    if workload_config is None:
        try:
            workload_config = get_scenario_config(scenario)
        except Exception:
            workload_config = None
    metrics_map: dict[str, Metrics] = {}
    per_job_map: dict[str, list[dict]] = {}
    ppo_logs: dict[str, list[dict]] = {}
    base_cluster = _cluster_for_scenario(scenario, num_gpus, gpu_memory_gb, cluster=cluster, cluster_config_path=cluster_config_path)
    for name in schedulers:
        if name == "PPO":
            metrics, per_job_recs, decision_logs, stats = evaluate_ppo_fixed_workload(
                jobs=copy.deepcopy(base_jobs),
                model_path=ppo_model_path,
                num_gpus=num_gpus,
                gpu_memory_gb=gpu_memory_gb,
                workload_config=workload_config,
                seed=seed,
                cluster=base_cluster,
                cluster_config_path=cluster_config_path,
                heterogeneous_obs=heterogeneous_obs,
            )
            # Update per-job scenario/seed correctly
            for rec in per_job_recs:
                rec["scenario"] = scenario
                rec["seed"] = seed
            metrics_map[name] = metrics
            per_job_map[name] = per_job_recs
            ppo_logs[name] = decision_logs
        else:
            scheduler = make_scheduler(name)
            jobs_copy = copy.deepcopy(base_jobs)
            import copy as _cp

            cl = _cp.deepcopy(base_cluster)
            sim = Simulator(cluster=cl, scheduler=scheduler)
            sim.load_jobs(jobs_copy)
            sim.run()
            metrics = compute_metrics(
                jobs_copy,
                total_gpus=cl.total_gpus,
                simulated_time=sim.current_time,
                gpu_time_used=sim.gpu_time_used,
                scheduling_decisions=len(sim.completed_jobs),
                invalid_scheduling_attempts=0,
                per_gpu_memory_gb=max(g.memory_gb for g in cl.gpus),
            )
            metrics_map[name] = metrics
            per_job_map[name] = [job_to_record(j, scheduler=name, scenario=scenario, seed=seed) for j in jobs_copy]
    return metrics_map, per_job_map, ppo_logs


def run_benchmark(
    scenario: str,
    seeds: list[int],
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    ppo_model_path: str | Path | None = None,
) -> dict[int, dict[str, Metrics]]:
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {SCENARIO_NAMES}")
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    all_results: dict[int, dict[str, Metrics]] = {}
    for seed in seeds:
        all_results[seed] = run_single_seed(
            scenario=scenario,
            seed=seed,
            schedulers=schedulers,
            num_jobs=num_jobs,
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            ppo_model_path=ppo_model_path,
        )
    return all_results


def run_benchmark_detailed(
    scenario: str,
    seeds: list[int],
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    ppo_model_path: str | Path | None = None,
) -> tuple[dict[int, dict[str, Metrics]], dict[int, dict[str, list[dict]]]]:
    """Backward compatible: returns (metrics, per_job) only."""
    metrics, per_job, _ = run_benchmark_detailed_with_logs(
        scenario=scenario,
        seeds=seeds,
        schedulers=schedulers,
        num_jobs=num_jobs,
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        ppo_model_path=ppo_model_path,
    )
    return metrics, per_job


def run_benchmark_detailed_with_logs(
    scenario: str,
    seeds: list[int],
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    ppo_model_path: str | Path | None = None,
) -> tuple[dict[int, dict[str, Metrics]], dict[int, dict[str, list[dict]]], dict[int, dict[str, list[dict]]]]:
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {SCENARIO_NAMES}")
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    all_metrics: dict[int, dict[str, Metrics]] = {}
    all_per_job: dict[int, dict[str, list[dict]]] = {}
    all_logs: dict[int, dict[str, list[dict]]] = {}
    for seed in seeds:
        m, p, l = run_single_seed_detailed_with_logs(
            scenario=scenario,
            seed=seed,
            schedulers=schedulers,
            num_jobs=num_jobs,
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            ppo_model_path=ppo_model_path,
        )
        all_metrics[seed] = m
        all_per_job[seed] = p
        all_logs[seed] = l
    return all_metrics, all_per_job, all_logs


def aggregate_results(
    results: dict[int, dict[str, Metrics]],
) -> pd.DataFrame:
    schedulers = set()
    for seed_map in results.values():
        schedulers.update(seed_map.keys())
    schedulers = sorted(schedulers)

    rows = []
    metric_keys = None
    for sched in schedulers:
        dicts = [results[seed][sched].as_dict() for seed in results if sched in results[seed]]
        if not dicts:
            continue
        if metric_keys is None:
            metric_keys = list(dicts[0].keys())
        row: dict[str, float | str | int] = {"scheduler": sched, "num_seeds": len(dicts)}
        for k in metric_keys:
            vals = [d[k] for d in dicts if isinstance(d[k], (int, float))]
            if vals:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
            else:
                row[f"{k}_mean"] = dicts[0][k]
                row[f"{k}_std"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def per_seed_dataframe(results: dict[int, dict[str, Metrics]]) -> pd.DataFrame:
    rows = []
    for seed, sched_map in results.items():
        for sched, metrics in sched_map.items():
            d = metrics.as_dict()
            d["seed"] = seed
            d["scheduler"] = sched
            rows.append(d)
    return pd.DataFrame(rows)


def per_job_dataframe(per_job_results: dict[int, dict[str, list[dict]]]) -> pd.DataFrame:
    rows: list[dict] = []
    for seed, sched_map in per_job_results.items():
        for sched, records in sched_map.items():
            rows.extend(records)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Persistence — CSV + JSON under artifacts/benchmarks/
# ---------------------------------------------------------------------------

def save_benchmark(
    scenario: str,
    results: dict[int, dict[str, Metrics]],
    out_dir: str | Path = "artifacts/benchmarks",
    *,
    num_gpus: int | None = None,
    num_jobs: int | None = None,
    per_job_results: dict[int, dict[str, list[dict]]] | None = None,
    ppo_logs: dict[int, dict[str, list[dict]]] | None = None,
    workload_config: dict | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tidy = per_seed_dataframe(results)
    agg = aggregate_results(results)

    tidy["scenario"] = scenario
    agg["scenario"] = scenario
    if num_gpus is not None:
        tidy["num_gpus"] = num_gpus
        agg["num_gpus"] = num_gpus
    if num_jobs is not None:
        tidy["num_jobs"] = num_jobs
        agg["num_jobs"] = num_jobs
    if workload_config is not None:
        tidy["workload_config"] = json.dumps(workload_config)
        agg["workload_config"] = json.dumps(workload_config)

    p_tidy = out / f"{scenario}.csv"
    p_agg = out / f"{scenario}_agg.csv"
    p_json = out / f"{scenario}.json"

    tidy.to_csv(p_tidy, index=False)
    agg.to_csv(p_agg, index=False)

    if workload_config is None:
        try:
            cfg = get_scenario_config(scenario)
            workload_config = asdict(cfg)
        except Exception:
            workload_config = {}

    serializable = {str(seed): {sched: m.as_dict() for sched, m in sched_map.items()} for seed, sched_map in results.items()}
    meta = {
        "scenario": scenario,
        "seeds": list(results.keys()),
        "num_gpus": num_gpus,
        "num_jobs": num_jobs,
        "workload_config": workload_config,
        "schedulers": sorted({s for sm in results.values() for s in sm}),
        "results": serializable,
    }
    p_json.write_text(json.dumps(meta, indent=2))

    out_paths: dict[str, Path] = {"tidy": p_tidy, "agg": p_agg, "json": p_json, "tidy_df": tidy, "agg_df": agg}  # type: ignore

    if per_job_results is not None:
        job_df = per_job_dataframe(per_job_results)
        if not job_df.empty:
            p_jobs_csv = out / f"{scenario}_jobs.csv"
            p_jobs_json = out / f"{scenario}_jobs.json"
            job_df.to_csv(p_jobs_csv, index=False)
            p_jobs_json.write_text(json.dumps({"scenario": scenario, "jobs": job_df.to_dict(orient="records")}, indent=2))
            out_paths["jobs_csv"] = p_jobs_csv  # type: ignore
            out_paths["jobs_json"] = p_jobs_json  # type: ignore
            out_paths["jobs_df"] = job_df  # type: ignore

    if ppo_logs is not None:
        # Flatten PPO logs across seeds
        flat_logs: list[dict] = []
        for seed, sched_map in ppo_logs.items():
            for sched, logs in sched_map.items():
                for entry in logs:
                    e = dict(entry)
                    e["seed"] = seed
                    e["scheduler"] = sched
                    e["scenario"] = scenario
                    flat_logs.append(e)
        if flat_logs:
            import pandas as pd
            log_df = pd.DataFrame(flat_logs)
            p_log_csv = out / f"{scenario}_ppo_decisions.csv"
            p_log_json = out / f"{scenario}_ppo_decisions.json"
            log_df.to_csv(p_log_csv, index=False)
            p_log_json.write_text(json.dumps({"scenario": scenario, "decisions": flat_logs}, indent=2))
            out_paths["ppo_log_csv"] = p_log_csv  # type: ignore
            out_paths["ppo_log_json"] = p_log_json  # type: ignore

    return out_paths  # type: ignore[return-value]


def run_and_save(
    scenario: str,
    seeds: list[int],
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    out_dir: str | Path = "artifacts/benchmarks",
    ppo_model_path: str | Path | None = None,
) -> dict[str, Path]:
    metrics_results, per_job_results, ppo_logs = run_benchmark_detailed_with_logs(
        scenario=scenario,
        seeds=seeds,
        schedulers=schedulers,
        num_jobs=num_jobs,
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
        ppo_model_path=ppo_model_path,
    )
    try:
        cfg = get_scenario_config(scenario)
        wc = asdict(cfg)
    except Exception:
        wc = {}
    return save_benchmark(
        scenario=scenario,
        results=metrics_results,
        out_dir=out_dir,
        num_gpus=num_gpus,
        num_jobs=num_jobs,
        per_job_results=per_job_results,
        ppo_logs=ppo_logs if ppo_logs else None,
        workload_config=wc,
    )
