"""Classical scheduling baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpu_sage.core.models import Job
from gpu_sage.schedulers.base import Scheduler

if TYPE_CHECKING:
    from gpu_sage.core.cluster import Cluster


class SJFScheduler(Scheduler):
    """Shortest-job-first among currently feasible jobs."""

    def select(
        self,
        waiting_jobs: list[Job],
        feasible_jobs: list[Job],
        cluster: "Cluster | None" = None,
        current_time: float = 0.0,
    ) -> Job | None:
        if not feasible_jobs:
            return None
        return min(feasible_jobs, key=lambda j: (j.duration, j.arrival_time, j.job_id))


class PriorityScheduler(Scheduler):
    """Highest priority first; ties go to the oldest job."""

    def select(
        self,
        waiting_jobs: list[Job],
        feasible_jobs: list[Job],
        cluster: "Cluster | None" = None,
        current_time: float = 0.0,
    ) -> Job | None:
        if not feasible_jobs:
            return None
        return min(feasible_jobs, key=lambda j: (-j.priority, j.arrival_time, j.job_id))


class BestFitScheduler(Scheduler):
    """Prefer the smallest feasible job to reduce stranded GPU capacity."""

    def select(
        self,
        waiting_jobs: list[Job],
        feasible_jobs: list[Job],
        cluster: "Cluster | None" = None,
        current_time: float = 0.0,
    ) -> Job | None:
        if not feasible_jobs:
            return None
        return min(feasible_jobs, key=lambda j: (j.gpu_count, j.duration, j.arrival_time, j.job_id))


class TopologyBestFitScheduler(Scheduler):
    """Topology-aware placement heuristic.

    Picks the feasible job whose *best* feasible GPU set minimizes
    communication_cost. Among equal-cost candidates, prefers shorter duration
    then arrival_time. The simulator still owns actual placement; this
    scheduler only scores jobs by their cheapest feasible topology.
    """

    def select(
        self,
        waiting_jobs: list[Job],
        feasible_jobs: list[Job],
        cluster: "Cluster | None" = None,
        current_time: float = 0.0,
    ) -> Job | None:
        if not feasible_jobs:
            return None
        if cluster is None or cluster.topology is None:
            # Fall back to BestFit behavior if no topology
            return min(feasible_jobs, key=lambda j: (j.gpu_count, j.duration, j.arrival_time, j.job_id))

        def _score(job: Job) -> tuple:
            best = cluster.best_feasible_set(job, strategy="compact")
            cost = cluster.topology.communication_cost(best) if best else 9e9
            # Prefer lower cost; tie-break by shorter, older, lower id
            return (cost, job.duration, job.arrival_time, job.job_id)

        return min(feasible_jobs, key=_score)


class PriorityAgingScheduler(Scheduler):
    """Priority scheduling with a linear waiting-time aging term."""

    def __init__(self, aging_rate: float = 0.01) -> None:
        if aging_rate < 0:
            raise ValueError("aging_rate must be non-negative")
        self.aging_rate = aging_rate

    def select(
        self,
        waiting_jobs: list[Job],
        feasible_jobs: list[Job],
        cluster: "Cluster | None" = None,
        current_time: float = 0.0,
    ) -> Job | None:
        if not feasible_jobs:
            return None
        # Use current_time to compute true waiting time for aging when available.
        def _score(j: Job) -> float:
            wait = max(0.0, current_time - j.arrival_time) if current_time else 0.0
            return j.priority + self.aging_rate * wait

        return max(
            feasible_jobs,
            key=lambda j: (_score(j), -j.arrival_time, -j.job_id),
        )
