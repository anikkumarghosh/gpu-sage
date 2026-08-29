"""Discrete-event GPU cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from gpu_sage.core.cluster import Cluster
from gpu_sage.core.events import EventQueue
from gpu_sage.core.models import Job, JobStatus
from gpu_sage.schedulers.base import Scheduler

ARRIVAL = "job_arrival"
COMPLETION = "job_completion"


@dataclass
class Simulator:
    """Runs the cluster forward through job arrivals and completions."""

    cluster: Cluster
    scheduler: Scheduler
    event_queue: EventQueue = field(default_factory=EventQueue)
    current_time: float = 0.0
    waiting_jobs: dict[int, Job] = field(default_factory=dict)
    running_jobs: dict[int, Job] = field(default_factory=dict)
    completed_jobs: dict[int, Job] = field(default_factory=dict)
    gpu_time_used: float = 0.0
    _last_metric_time: float = 0.0
    _job_store: dict[int, Job] = field(default_factory=dict, init=False, repr=False)

    def reset(self, jobs: Iterable[Job]) -> None:
        self.event_queue = EventQueue()
        self.current_time = 0.0
        self.waiting_jobs.clear()
        self.running_jobs.clear()
        self.completed_jobs.clear()
        self.gpu_time_used = 0.0
        self._last_metric_time = 0.0
        self._job_store.clear()
        for gpu in self.cluster.gpus:
            gpu.allocated_job_id = None

        jobs = list(jobs)
        self._job_store.update({job.job_id: job for job in jobs})
        for job in jobs:
            job.status = JobStatus.WAITING
            job.start_time = None
            job.completion_time = None
            job.assigned_gpus = []
            self.event_queue.push(job.arrival_time, ARRIVAL, job.job_id)

    def load_jobs(self, jobs: Iterable[Job]) -> None:
        self.reset(jobs)

    def get_job(self, job_id: int) -> Job:
        return self._job_store[job_id]

    def feasible_waiting_jobs(self) -> list[Job]:
        return [job for job in self.waiting_jobs.values() if self.cluster.can_allocate(job)]

    def schedule_job(self, job_id: int) -> Job:
        """Launch exactly one waiting job at the current simulation time."""
        if job_id not in self.waiting_jobs:
            raise ValueError(f"Job {job_id} is not waiting")
        job = self.waiting_jobs[job_id]
        if not self.cluster.can_allocate(job):
            raise ValueError(f"Job {job_id} is not currently feasible")

        job.assigned_gpus = self.cluster.allocate(job)
        job.status = JobStatus.RUNNING
        job.start_time = self.current_time
        self.waiting_jobs.pop(job.job_id)
        self.running_jobs[job.job_id] = job
        self.event_queue.push(self.current_time + job.duration, COMPLETION, job.job_id)
        return job

    def try_schedule_with(self, selector: Callable[[list[Job], list[Job]], Job | None]) -> list[int]:
        """Keep launching jobs using a selector until it returns None."""
        launched: list[int] = []
        while True:
            selected = selector(list(self.waiting_jobs.values()), self.feasible_waiting_jobs())
            if selected is None:
                break
            self.schedule_job(selected.job_id)
            launched.append(selected.job_id)
        return launched

    def _advance_clock(self, new_time: float) -> float:
        delta = new_time - self.current_time
        if delta < 0:
            raise RuntimeError("Event time moved backwards")
        self.gpu_time_used += sum(job.gpu_count for job in self.running_jobs.values()) * delta
        self.current_time = new_time
        return delta

    def step_until_next_event(self, schedule: bool = True) -> dict:
        """Process one event and optionally run the configured heuristic scheduler."""
        if len(self.event_queue) == 0:
            return {"event": None, "time": self.current_time, "launched": []}

        event = self.event_queue.pop()
        self._advance_clock(event.time)
        job = self._job_store[event.job_id]

        if event.event_type == ARRIVAL:
            self.waiting_jobs[job.job_id] = job
        elif event.event_type == COMPLETION:
            # A future extension may add cancellation/preemption events.
            if job.status == JobStatus.RUNNING:
                self.cluster.release(job)
                job.status = JobStatus.COMPLETED
                job.completion_time = self.current_time
                self.running_jobs.pop(job.job_id, None)
                self.completed_jobs[job.job_id] = job
        else:
            raise ValueError(f"Unknown event type: {event.event_type}")

        launched = self.try_schedule_with(self.scheduler.select) if schedule else []
        return {
            "event": event.event_type,
            "job_id": event.job_id,
            "time": self.current_time,
            "launched": launched,
        }

    def run(self) -> list[dict]:
        history: list[dict] = []
        while len(self.event_queue) > 0:
            history.append(self.step_until_next_event(schedule=True))
        return history

    def snapshot(self) -> dict:
        return {
            "time": self.current_time,
            "free_gpus": self.cluster.free_gpu_count,
            "total_gpus": self.cluster.total_gpus,
            "utilization": self.cluster.utilization(self.running_jobs),
            "waiting": [job.job_id for job in self.waiting_jobs.values()],
            "running": [job.job_id for job in self.running_jobs.values()],
            "completed": [job.job_id for job in self.completed_jobs.values()],
        }
