"""Compare classical scheduling heuristics on the same workload."""

from __future__ import annotations

import pandas as pd

from gpu_sage.core.cluster import Cluster
from gpu_sage.core.simulator import Simulator
from gpu_sage.evaluation.metrics import compute_metrics
from gpu_sage.schedulers.fcfs import FCFSScheduler
from gpu_sage.schedulers.heuristics import BestFitScheduler, PriorityScheduler, SJFScheduler
from gpu_sage.workloads.generator import SyntheticWorkload, WorkloadConfig


def run_policy(name, scheduler, jobs):
    # Clone jobs so one policy cannot mutate another policy's episode.
    from copy import deepcopy
    local_jobs = deepcopy(jobs)
    sim = Simulator(Cluster.homogeneous(8, memory_gb=80), scheduler)
    sim.load_jobs(local_jobs)
    sim.run()
    metrics = compute_metrics(local_jobs, sim.cluster.total_gpus, sim.current_time, sim.gpu_time_used)
    return {"scheduler": name, **metrics.as_dict()}


def main() -> None:
    jobs = SyntheticWorkload(
        WorkloadConfig(
            arrival_rate=0.08,
            min_gpus=1,
            max_gpus=4,
            min_duration=20,
            max_duration=180,
        ),
        seed=7,
    ).generate(100)

    policies = [
        ("FCFS", FCFSScheduler()),
        ("SJF", SJFScheduler()),
        ("Priority", PriorityScheduler()),
        ("BestFit", BestFitScheduler()),
    ]
    results = pd.DataFrame([run_policy(*policy, jobs) for policy in policies])
    cols = [
        "scheduler",
        "average_waiting_time",
        "p95_waiting_time",
        "average_turnaround_time",
        "p95_turnaround_time",
        "throughput_jobs_per_time",
        "gpu_utilization",
    ]
    print(results[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
