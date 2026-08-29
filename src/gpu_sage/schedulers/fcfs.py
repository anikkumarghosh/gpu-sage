"""First-Come, First-Served baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpu_sage.core.models import Job
from gpu_sage.schedulers.base import Scheduler

if TYPE_CHECKING:
    from gpu_sage.core.cluster import Cluster


class FCFSScheduler(Scheduler):
    """FCFS selects the earliest-arriving feasible job."""

    def select(
        self,
        waiting_jobs: list[Job],
        feasible_jobs: list[Job],
        cluster: "Cluster | None" = None,
        current_time: float = 0.0,
    ) -> Job | None:
        if not feasible_jobs:
            return None
        return min(feasible_jobs, key=lambda job: (job.arrival_time, job.job_id))
