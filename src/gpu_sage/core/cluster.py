"""GPU cluster resource management."""

from __future__ import annotations

from dataclasses import dataclass

from .models import GPU, Job


@dataclass
class Cluster:
    """Homogeneous or heterogeneous collection of GPUs."""

    gpus: list[GPU]

    @classmethod
    def homogeneous(cls, num_gpus: int, memory_gb: float = 80.0, gpu_type: str = "A100") -> "Cluster":
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        return cls([GPU(i, memory_gb, gpu_type) for i in range(num_gpus)])

    @property
    def total_gpus(self) -> int:
        return len(self.gpus)

    @property
    def free_gpus(self) -> list[GPU]:
        return [gpu for gpu in self.gpus if gpu.is_free]

    @property
    def free_gpu_count(self) -> int:
        return len(self.free_gpus)

    def feasible_gpu_ids(self, job: Job) -> list[int]:
        candidates = [gpu for gpu in self.gpus if gpu.is_free and gpu.memory_gb >= job.gpu_memory_gb]
        return [gpu.gpu_id for gpu in candidates]

    def can_allocate(self, job: Job) -> bool:
        return len(self.feasible_gpu_ids(job)) >= job.gpu_count

    def allocate(self, job: Job) -> list[int]:
        candidates = self.feasible_gpu_ids(job)
        if len(candidates) < job.gpu_count:
            raise ValueError(f"Job {job.job_id} cannot be allocated: insufficient compatible GPUs")

        assigned = candidates[: job.gpu_count]
        for gpu_id in assigned:
            self.gpus[gpu_id].allocated_job_id = job.job_id
        return assigned

    def release(self, job: Job) -> None:
        for gpu in self.gpus:
            if gpu.allocated_job_id == job.job_id:
                gpu.allocated_job_id = None

    def utilization(self, running_jobs: dict[int, Job]) -> float:
        """GPU-count utilization in [0, 1]."""
        if self.total_gpus == 0:
            return 0.0
        used = sum(job.gpu_count for job in running_jobs.values())
        return used / self.total_gpus
