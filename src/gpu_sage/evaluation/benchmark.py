"""Rigorous RL-vs-baseline benchmark runner.

Critical correctness property: for each seed/scenario the *identical* workload is
evaluated by every scheduler from a fresh simulator state. No scheduler may
mutate the workload seen by another.

Architecture:

    Workload (seeded) -> Simulator -> Scheduler interface
                                  ├── FCFS
                                  ├── SJF
                                  ├── Priority
                                  ├── Best-Fit
                                  └── PPO (RLScheduler)
                        -> Metrics -> Benchmark aggregation (mean±std)
                        -> Per-job records (for P50/P95/P99, fairness, starvation)

The simulator owns time advancement, arrivals, completions, allocation/release.
Schedulers only return a Job or NOOP.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from gpu_sage.core.cluster import Cluster
from gpu_sage.core.simulator import Simulator
from gpu_sage.evaluation.metrics import Metrics, compute_metrics
from gpu_sage.schedulers.fcfs import FCFSScheduler
from gpu_sage.schedulers.heuristics import BestFitScheduler, PriorityScheduler, SJFScheduler
from gpu_sage.schedulers.rl import RandomScheduler, RLScheduler
from gpu_sage.workloads.generator import SCENARIO_NAMES, generate_workload, get_scenario_config

# ---------------------------------------------------------------------------
# Scheduler registry — all schedulers share the common Simulator interface.
# ---------------------------------------------------------------------------

SCHEDULER_FACTORIES: dict[str, Callable[[], object]] = {
    "FCFS": lambda: FCFSScheduler(),
    "SJF": lambda: SJFScheduler(),
    "Priority": lambda: PriorityScheduler(),
    "BestFit": lambda: BestFitScheduler(),
    "Random": lambda: RandomScheduler(seed=0),
    "PPO": lambda: RLScheduler(),
}

DEFAULT_SCHEDULERS: list[str] = ["FCFS", "SJF", "Priority", "BestFit"]


def available_schedulers() -> list[str]:
    return list(SCHEDULER_FACTORIES.keys())


def make_scheduler(name: str):
    if name not in SCHEDULER_FACTORIES:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(SCHEDULER_FACTORIES)}")
    return SCHEDULER_FACTORIES[name]()


# ---------------------------------------------------------------------------
# Per-job record helper — preserves full job-level data for later analysis.
# ---------------------------------------------------------------------------

def job_to_record(
    job,
    scheduler: str,
    scenario: str,
    seed: int,
) -> dict:
    """Convert a Job to a per-job record with all required fields.

    Contains:
        job_id, arrival_time, start_time, completion_time,
        waiting_time, turnaround_time,
        gpu_requirement, memory_requirement, execution_time,
        priority, scheduler, scenario, seed, status
    """
    return {
        "job_id": job.job_id,
        "arrival_time": job.arrival_time,
        "start_time": job.start_time,
        "completion_time": job.completion_time,
        "waiting_time": job.waiting_time,
        "turnaround_time": job.turnaround_time,
        "gpu_requirement": job.gpu_count,
        "memory_requirement": job.gpu_memory_gb,
        "execution_time": job.duration,
        "priority": job.priority,
        "scheduler": scheduler,
        "scenario": scenario,
        "seed": seed,
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "assigned_gpus": ",".join(map(str, job.assigned_gpus)) if job.assigned_gpus else "",
    }


# ---------------------------------------------------------------------------
# Single-seed evaluation — identical workload for every scheduler.
# ---------------------------------------------------------------------------

def run_single_seed(
    scenario: str,
    seed: int,
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
) -> dict[str, Metrics]:
    """Run one seed/scenario against all schedulers on the SAME workload.

    Steps:
        1. Generate one workload using fixed seed.
        2. Deepcopy that workload for each scheduler.
        3. Run each scheduler from fresh Simulator state.
        4. Collect Metrics.

    Returns:
        Dict scheduler_name -> Metrics
    """
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    base_jobs = generate_workload(scenario=scenario, seed=seed, count=num_jobs)
    results: dict[str, Metrics] = {}
    for name in schedulers:
        scheduler = make_scheduler(name)
        jobs_copy = copy.deepcopy(base_jobs)
        cluster = Cluster.homogeneous(num_gpus, memory_gb=gpu_memory_gb)
        sim = Simulator(cluster=cluster, scheduler=scheduler)
        sim.load_jobs(jobs_copy)
        sim.run()
        metrics = compute_metrics(
            jobs_copy,
            total_gpus=cluster.total_gpus,
            simulated_time=sim.current_time,
            gpu_time_used=sim.gpu_time_used,
            scheduling_decisions=len(sim.completed_jobs),
            invalid_scheduling_attempts=0,
            per_gpu_memory_gb=gpu_memory_gb,
        )
        results[name] = metrics
    return results


def run_single_seed_detailed(
    scenario: str,
    seed: int,
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
) -> tuple[dict[str, Metrics], dict[str, list[dict]]]:
    """Like run_single_seed but also returns per-job records.

    Returns:
        (metrics_dict, per_job_dict) where per_job_dict[scheduler] = list of per-job record dicts
    """
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    base_jobs = generate_workload(scenario=scenario, seed=seed, count=num_jobs)
    metrics_map: dict[str, Metrics] = {}
    per_job_map: dict[str, list[dict]] = {}
    for name in schedulers:
        scheduler = make_scheduler(name)
        jobs_copy = copy.deepcopy(base_jobs)
        cluster = Cluster.homogeneous(num_gpus, memory_gb=gpu_memory_gb)
        sim = Simulator(cluster=cluster, scheduler=scheduler)
        sim.load_jobs(jobs_copy)
        sim.run()
        metrics = compute_metrics(
            jobs_copy,
            total_gpus=cluster.total_gpus,
            simulated_time=sim.current_time,
            gpu_time_used=sim.gpu_time_used,
            scheduling_decisions=len(sim.completed_jobs),
            invalid_scheduling_attempts=0,
            per_gpu_memory_gb=gpu_memory_gb,
        )
        metrics_map[name] = metrics
        per_job_map[name] = [job_to_record(j, scheduler=name, scenario=scenario, seed=seed) for j in jobs_copy]
    return metrics_map, per_job_map


def run_benchmark(
    scenario: str,
    seeds: list[int],
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
) -> dict[int, dict[str, Metrics]]:
    """Multi-seed benchmark: scenario × seeds × schedulers.

    Returns:
        Dict seed -> (Dict scheduler -> Metrics)
    """
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {SCENARIO_NAMES}")
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    all_results: dict[int, dict[str, Metrics]] = {}
    for seed in seeds:
        all_results[seed] = run_single_seed(
            scenario=scenario,
            seed=seed,
            schedulers=schedulers,
            num_jobs=num_jobs,
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
        )
    return all_results


def run_benchmark_detailed(
    scenario: str,
    seeds: list[int],
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
) -> tuple[dict[int, dict[str, Metrics]], dict[int, dict[str, list[dict]]]]:
    """Multi-seed benchmark with per-job records.

    Returns:
        (metrics_results, per_job_results)
        metrics_results: Dict seed -> Dict scheduler -> Metrics
        per_job_results: Dict seed -> Dict scheduler -> List[per-job record dict]
    """
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {SCENARIO_NAMES}")
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    all_metrics: dict[int, dict[str, Metrics]] = {}
    all_per_job: dict[int, dict[str, list[dict]]] = {}
    for seed in seeds:
        m, p = run_single_seed_detailed(
            scenario=scenario,
            seed=seed,
            schedulers=schedulers,
            num_jobs=num_jobs,
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
        )
        all_metrics[seed] = m
        all_per_job[seed] = p
    return all_metrics, all_per_job


def aggregate_results(
    results: dict[int, dict[str, Metrics]],
) -> pd.DataFrame:
    """Aggregate multi-seed results into mean±std per scheduler/metric.

    Returns a DataFrame with one row per scheduler and columns like
    ``avg_wait_mean``, ``avg_wait_std``, etc.
    """
    schedulers = set()
    for seed_map in results.values():
        schedulers.update(seed_map.keys())
    schedulers = sorted(schedulers)

    rows = []
    metric_keys = None
    for sched in schedulers:
        dicts = [results[seed][sched].as_dict() for seed in results if sched in results[seed]]
        if not dicts:
            continue
        if metric_keys is None:
            metric_keys = list(dicts[0].keys())
        row: dict[str, float | str | int] = {"scheduler": sched, "num_seeds": len(dicts)}
        for k in metric_keys:
            vals = [d[k] for d in dicts if isinstance(d[k], (int, float))]
            if vals:
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
            else:
                row[f"{k}_mean"] = dicts[0][k]
                row[f"{k}_std"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def per_seed_dataframe(results: dict[int, dict[str, Metrics]]) -> pd.DataFrame:
    """Flatten seed × scheduler results into a tidy DataFrame (one row per seed-scheduler)."""
    rows = []
    for seed, sched_map in results.items():
        for sched, metrics in sched_map.items():
            d = metrics.as_dict()
            d["seed"] = seed
            d["scheduler"] = sched
            rows.append(d)
    return pd.DataFrame(rows)


def per_job_dataframe(per_job_results: dict[int, dict[str, list[dict]]]) -> pd.DataFrame:
    """Flatten per-job records across seeds/schedulers into a DataFrame."""
    rows: list[dict] = []
    for seed, sched_map in per_job_results.items():
        for sched, records in sched_map.items():
            rows.extend(records)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Persistence — CSV + JSON under artifacts/benchmarks/
# ---------------------------------------------------------------------------

def save_benchmark(
    scenario: str,
    results: dict[int, dict[str, Metrics]],
    out_dir: str | Path = "artifacts/benchmarks",
    *,
    num_gpus: int | None = None,
    num_jobs: int | None = None,
    per_job_results: dict[int, dict[str, list[dict]]] | None = None,
    workload_config: dict | None = None,
) -> dict[str, Path]:
    """Save machine-readable results.

    Writes:
        {out_dir}/{scenario}.csv       — per-seed tidy metrics
        {out_dir}/{scenario}_agg.csv   — mean±std aggregation
        {out_dir}/{scenario}.json      — full per-seed metrics + config metadata
        {out_dir}/{scenario}_jobs.csv  — per-job records (if available)
        {out_dir}/{scenario}_jobs.json — per-job records as JSON (if available)

    Args:
        scenario: Scenario name.
        results: Metrics results from run_benchmark.
        out_dir: Output directory.
        num_gpus: Cluster size for repro metadata.
        num_jobs: Jobs per workload for repro metadata.
        per_job_results: Optional per-job records from run_benchmark_detailed.
        workload_config: Optional workload config dict for repro metadata.

    Returns:
        Dict of written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tidy = per_seed_dataframe(results)
    agg = aggregate_results(results)

    # Enrich with repro metadata
    tidy["scenario"] = scenario
    agg["scenario"] = scenario
    if num_gpus is not None:
        tidy["num_gpus"] = num_gpus
        agg["num_gpus"] = num_gpus
    if num_jobs is not None:
        tidy["num_jobs"] = num_jobs
        agg["num_jobs"] = num_jobs
    if workload_config is not None:
        # Store as JSON string in CSV for traceability
        tidy["workload_config"] = json.dumps(workload_config)
        agg["workload_config"] = json.dumps(workload_config)

    p_tidy = out / f"{scenario}.csv"
    p_agg = out / f"{scenario}_agg.csv"
    p_json = out / f"{scenario}.json"

    tidy.to_csv(p_tidy, index=False)
    agg.to_csv(p_agg, index=False)

    # Resolve workload config for JSON if not provided
    if workload_config is None:
        try:
            cfg = get_scenario_config(scenario)
            workload_config = asdict(cfg)
        except Exception:
            workload_config = {}

    serializable = {str(seed): {sched: m.as_dict() for sched, m in sched_map.items()} for seed, sched_map in results.items()}
    meta = {
        "scenario": scenario,
        "seeds": list(results.keys()),
        "num_gpus": num_gpus,
        "num_jobs": num_jobs,
        "workload_config": workload_config,
        "schedulers": sorted({s for sm in results.values() for s in sm}),
        "results": serializable,
    }
    p_json.write_text(json.dumps(meta, indent=2))

    out_paths: dict[str, Path] = {"tidy": p_tidy, "agg": p_agg, "json": p_json, "tidy_df": tidy, "agg_df": agg}  # type: ignore

    # Per-job artifacts
    if per_job_results is not None:
        job_df = per_job_dataframe(per_job_results)
        if not job_df.empty:
            p_jobs_csv = out / f"{scenario}_jobs.csv"
            p_jobs_json = out / f"{scenario}_jobs.json"
            job_df.to_csv(p_jobs_csv, index=False)
            # JSON per-job: list of records
            p_jobs_json.write_text(json.dumps({"scenario": scenario, "jobs": job_df.to_dict(orient="records")}, indent=2))
            out_paths["jobs_csv"] = p_jobs_csv  # type: ignore
            out_paths["jobs_json"] = p_jobs_json  # type: ignore
            out_paths["jobs_df"] = job_df  # type: ignore

    return out_paths  # type: ignore[return-value]


def run_and_save(
    scenario: str,
    seeds: list[int],
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    out_dir: str | Path = "artifacts/benchmarks",
) -> dict[str, Path]:
    """Convenience: run multi-seed benchmark (with per-job) and save to disk."""
    metrics_results, per_job_results = run_benchmark_detailed(
        scenario=scenario,
        seeds=seeds,
        schedulers=schedulers,
        num_jobs=num_jobs,
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
    )
    # Capture workload config for repro
    try:
        cfg = get_scenario_config(scenario)
        wc = asdict(cfg)
    except Exception:
        wc = {}
    return save_benchmark(
        scenario=scenario,
        results=metrics_results,
        out_dir=out_dir,
        num_gpus=num_gpus,
        num_jobs=num_jobs,
        per_job_results=per_job_results,
        workload_config=wc,
    )
