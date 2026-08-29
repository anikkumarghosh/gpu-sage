"""Synthetic workload generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from gpu_sage.core.models import Job


@dataclass
class WorkloadConfig:
    arrival_rate: float = 0.15
    min_gpus: int = 1
    max_gpus: int = 4
    min_memory_gb: float = 8.0
    max_memory_gb: float = 64.0
    min_duration: float = 20.0
    max_duration: float = 200.0
    min_priority: int = 1
    max_priority: int = 5


class SyntheticWorkload:
    """Poisson arrivals + bounded job properties."""

    def __init__(self, config: WorkloadConfig, seed: int = 0) -> None:
        if config.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate(self, count: int) -> list[Job]:
        if count <= 0:
            return []

        inter_arrivals = self.rng.exponential(1.0 / self.config.arrival_rate, size=count)
        arrivals = np.cumsum(inter_arrivals)
        jobs: list[Job] = []

        for job_id, arrival_time in enumerate(arrivals):
            jobs.append(
                Job(
                    job_id=job_id,
                    arrival_time=float(arrival_time),
                    gpu_count=int(self.rng.integers(self.config.min_gpus, self.config.max_gpus + 1)),
                    gpu_memory_gb=float(self.rng.uniform(self.config.min_memory_gb, self.config.max_memory_gb)),
                    duration=float(self.rng.uniform(self.config.min_duration, self.config.max_duration)),
                    priority=int(self.rng.integers(self.config.min_priority, self.config.max_priority + 1)),
                )
            )
        return jobs
