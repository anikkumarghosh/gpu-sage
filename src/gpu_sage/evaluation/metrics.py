"""Metrics for comparing scheduling policies — mathematically documented."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from gpu_sage.core.models import Job


@dataclass(frozen=True)
class Metrics:
    """Rigorous scheduler comparison metrics.

    Definitions:

    * ``completed_jobs`` — jobs that reached ``COMPLETED`` state.
    * ``throughput`` — completed / simulated_time (jobs per time unit).
    * ``average_waiting_time`` — mean(start - arrival) over completed.
    * ``median_waiting_time`` — 50th percentile of waiting times (=p50).
    * ``p95_waiting_time`` — 95th percentile using ``numpy.percentile``.
    * ``average_turnaround_time`` — mean(completion - arrival) = JCT mean.
    * ``median_jct`` — 50th percentile of turnaround/JCT.
    * ``p95_jct`` — 95th percentile of JCT.
    * ``gpu_utilization`` — gpu_time_used / (total_gpus * simulated_time) in [0,1].
    * ``gpu_idle_time`` — fraction of idle GPU-time = 1 - utilization, also in [0,1];
      absolute idle gpu-seconds = total_gpus*simulated_time - gpu_time_used.
    * ``gpu_idle_fraction`` — alias for gpu_idle_time (fraction).
    * ``resource_allocation_efficiency`` — same as utilization for homogeneous
      cluster; measures fraction of available GPU-seconds that performed useful work.
    * ``rejected_jobs`` — jobs that can never run (gpu_count > total_gpus or
      gpu_memory > per-GPU memory).
    * ``infeasible_jobs`` — same as rejected for this simulator.
    * ``scheduling_decisions`` — number of successful schedule_job calls.
    * ``invalid_scheduling_attempts`` — infeasible selections attempted (heuristics 0,
      RL may be >0).
    * ``jains_fairness_index`` — Jain's fairness: (sum x_i)^2 / (n * sum x_i^2)
      computed on waiting times of completed jobs, in [1/n, 1] ⊆ [0,1]; 1 = perfectly fair.
    """

    completed_jobs: int
    total_jobs: int
    completion_rate: float
    average_waiting_time: float
    median_waiting_time: float
    p50_waiting_time: float
    p95_waiting_time: float
    average_turnaround_time: float
    median_jct: float
    p50_turnaround_time: float
    p95_turnaround_time: float
    p95_jct: float
    throughput_jobs_per_time: float
    throughput: float
    gpu_utilization: float
    gpu_idle_time: float
    gpu_idle_fraction: float
    resource_allocation_efficiency: float
    rejected_jobs: int
    infeasible_jobs: int
    scheduling_decisions: int
    invalid_scheduling_attempts: int
    jains_fairness_index: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    """Numpy percentile; returns 0.0 for empty list."""
    return float(np.percentile(values, percentile)) if values else 0.0


def _jains_index(values: list[float]) -> float:
    """Jain's fairness index ∈ [0,1]. 1 = perfectly fair.

    Formula: J = (sum x_i)^2 / (n * sum x_i^2)
    For empty or all-zero values, returns 1.0 (vacuously fair) when n<=1 else 0.0/1.0 handling.
    Bounded in [1/n, 1] for positive values, and in [0,1] generally.
    """
    if not values:
        return 1.0
    arr = np.array(values, dtype=float)
    # Use absolute waiting times; waiting times are non-negative, so index meaningful.
    # If all zero (no waiting), fairness is perfect.
    if np.all(arr == 0):
        return 1.0
    n = len(arr)
    sum_x = float(np.sum(arr))
    sum_x2 = float(np.sum(arr**2))
    if sum_x2 == 0:
        return 1.0
    j = (sum_x**2) / (n * sum_x2)
    # Clamp to [0,1] for numerical safety.
    return float(max(0.0, min(1.0, j)))


def compute_metrics(
    jobs: list[Job],
    total_gpus: int,
    simulated_time: float,
    gpu_time_used: float,
    *,
    scheduling_decisions: int | None = None,
    invalid_scheduling_attempts: int = 0,
    per_gpu_memory_gb: float | None = None,
) -> Metrics:
    """Compute comprehensive metrics for a finished simulation.

    Args:
        jobs: All jobs from the workload (with start/completion times filled by simulator).
        total_gpus: Cluster size.
        simulated_time: Simulator current_time after ``run()``.
        gpu_time_used: Integrated gpu-seconds of running jobs.
        scheduling_decisions: Number of successful scheduling decisions. Defaults to
            ``completed_jobs`` + still-running if not provided.
        invalid_scheduling_attempts: Number of infeasible selections (RL may produce >0).
        per_gpu_memory_gb: Per-GPU memory to detect rejected jobs due to memory.
            If None, only gpu_count is checked.

    Returns:
        Metrics dataclass with mathematically correct fields.

    Notes:
        * GPU utilization = gpu_time_used / (total_gpus * simulated_time) ∈ [0,1].
        * GPU idle fraction = 1 - utilization.
        * Resource allocation efficiency = utilization (for homogeneous cluster).
        * Throughput = completed / simulated_time.
        * P95 uses linear interpolation via ``numpy.percentile`` (default).
        * Jain's index on waiting times ∈ [0,1].
    """
    completed = [j for j in jobs if j.completion_time is not None and j.start_time is not None]
    waits = [float(j.waiting_time) for j in completed if j.waiting_time is not None]
    turnarounds = [float(j.turnaround_time) for j in completed if j.turnaround_time is not None]

    utilization = gpu_time_used / (total_gpus * simulated_time) if total_gpus > 0 and simulated_time > 0 else 0.0
    # Clamp to [0,1] to handle floating point / logic edge cases.
    utilization = float(max(0.0, min(1.0, utilization)))
    idle_fraction = 1.0 - utilization
    idle_fraction = float(max(0.0, min(1.0, idle_fraction)))

    # Rejected / infeasible: jobs that can never be allocated.
    rejected = 0
    for j in jobs:
        if j.gpu_count > total_gpus:
            rejected += 1
        elif per_gpu_memory_gb is not None and j.gpu_memory_gb > per_gpu_memory_gb:
            rejected += 1

    if scheduling_decisions is None:
        scheduling_decisions = len(completed)

    jain = _jains_index(waits)

    avg_wait = float(np.mean(waits)) if waits else 0.0
    avg_turn = float(np.mean(turnarounds)) if turnarounds else 0.0
    p50_wait = _percentile(waits, 50)
    p95_wait = _percentile(waits, 95)
    p50_turn = _percentile(turnarounds, 50)
    p95_turn = _percentile(turnarounds, 95)
    throughput = len(completed) / simulated_time if simulated_time > 0 else 0.0

    return Metrics(
        completed_jobs=len(completed),
        total_jobs=len(jobs),
        completion_rate=len(completed) / len(jobs) if jobs else 0.0,
        average_waiting_time=avg_wait,
        median_waiting_time=p50_wait,
        p50_waiting_time=p50_wait,
        p95_waiting_time=p95_wait,
        average_turnaround_time=avg_turn,
        median_jct=p50_turn,
        p50_turnaround_time=p50_turn,
        p95_turnaround_time=p95_turn,
        p95_jct=p95_turn,
        throughput_jobs_per_time=throughput,
        throughput=throughput,
        gpu_utilization=utilization,
        gpu_idle_time=idle_fraction,
        gpu_idle_fraction=idle_fraction,
        resource_allocation_efficiency=utilization,
        rejected_jobs=rejected,
        infeasible_jobs=rejected,
        scheduling_decisions=int(scheduling_decisions),
        invalid_scheduling_attempts=int(invalid_scheduling_attempts),
        jains_fairness_index=jain,
    )
