"""First-Come, First-Served baseline."""

from gpu_sage.core.models import Job
from gpu_sage.schedulers.base import Scheduler


class FCFSScheduler(Scheduler):
    def select(self, waiting_jobs: list[Job], feasible_jobs: list[Job]) -> Job | None:
        if not feasible_jobs:
            return None
        return min(feasible_jobs, key=lambda job: (job.arrival_time, job.job_id))
