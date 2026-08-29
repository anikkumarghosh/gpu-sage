"""Metrics for comparing scheduling policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from gpu_sage.core.models import Job


@dataclass(frozen=True)
class Metrics:
    completed_jobs: int
    total_jobs: int
    completion_rate: float
    average_waiting_time: float
    p50_waiting_time: float
    p95_waiting_time: float
    average_turnaround_time: float
    p50_turnaround_time: float
    p95_turnaround_time: float
    throughput_jobs_per_time: float
    gpu_utilization: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def compute_metrics(jobs: list[Job], total_gpus: int, simulated_time: float, gpu_time_used: float) -> Metrics:
    completed = [j for j in jobs if j.completion_time is not None and j.start_time is not None]
    waits = [float(j.waiting_time) for j in completed if j.waiting_time is not None]
    turnarounds = [float(j.turnaround_time) for j in completed if j.turnaround_time is not None]

    utilization = gpu_time_used / (total_gpus * simulated_time) if total_gpus > 0 and simulated_time > 0 else 0.0

    return Metrics(
        completed_jobs=len(completed),
        total_jobs=len(jobs),
        completion_rate=len(completed) / len(jobs) if jobs else 0.0,
        average_waiting_time=float(np.mean(waits)) if waits else 0.0,
        p50_waiting_time=_percentile(waits, 50),
        p95_waiting_time=_percentile(waits, 95),
        average_turnaround_time=float(np.mean(turnarounds)) if turnarounds else 0.0,
        p50_turnaround_time=_percentile(turnarounds, 50),
        p95_turnaround_time=_percentile(turnarounds, 95),
        throughput_jobs_per_time=len(completed) / simulated_time if simulated_time > 0 else 0.0,
        gpu_utilization=utilization,
    )
