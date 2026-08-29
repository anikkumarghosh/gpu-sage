"""Classical scheduling baselines."""

from __future__ import annotations

from gpu_sage.core.models import Job
from gpu_sage.schedulers.base import Scheduler


class SJFScheduler(Scheduler):
    """Shortest-job-first among currently feasible jobs."""

    def select(self, waiting_jobs: list[Job], feasible_jobs: list[Job]) -> Job | None:
        if not feasible_jobs:
            return None
        return min(feasible_jobs, key=lambda j: (j.duration, j.arrival_time, j.job_id))


class PriorityScheduler(Scheduler):
    """Highest priority first; ties go to the oldest job."""

    def select(self, waiting_jobs: list[Job], feasible_jobs: list[Job]) -> Job | None:
        if not feasible_jobs:
            return None
        return min(feasible_jobs, key=lambda j: (-j.priority, j.arrival_time, j.job_id))


class BestFitScheduler(Scheduler):
    """Prefer the smallest feasible job to reduce stranded GPU capacity."""

    def select(self, waiting_jobs: list[Job], feasible_jobs: list[Job]) -> Job | None:
        if not feasible_jobs:
            return None
        return min(feasible_jobs, key=lambda j: (j.gpu_count, j.duration, j.arrival_time, j.job_id))


class PriorityAgingScheduler(Scheduler):
    """Priority scheduling with a linear waiting-time aging term."""

    def __init__(self, aging_rate: float = 0.01) -> None:
        if aging_rate < 0:
            raise ValueError("aging_rate must be non-negative")
        self.aging_rate = aging_rate

    def select(self, waiting_jobs: list[Job], feasible_jobs: list[Job]) -> Job | None:
        if not feasible_jobs:
            return None
        # Simulator time is not passed into this interface, so waiting_time here is
        # intentionally represented by an externally updated attribute when used.
        # For the first benchmark round we keep the scheduler deterministic by using
        # a stable approximation based on job age encoded by arrival time.
        return max(
            feasible_jobs,
            key=lambda j: (j.priority + self.aging_rate * (-j.arrival_time), -j.arrival_time, -j.job_id),
        )
