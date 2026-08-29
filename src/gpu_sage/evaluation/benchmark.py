"""Rigorous RL-vs-baseline benchmark runner.

Critical correctness property: for each seed/scenario the *identical* workload is
evaluated by every scheduler from a fresh simulator state. No scheduler may
mutate the workload seen by another.

Architecture:

    Workload (seeded) → Simulator → Scheduler interface
                                  ├── FCFS
                                  ├── SJF
                                  ├── Priority
                                  ├── Best-Fit
                                  └── PPO (RLScheduler)
                        → Metrics → Benchmark aggregation (mean±std)

The simulator owns time advancement, arrivals, completions, allocation/release.
Schedulers only return a Job or NOOP.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
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
        Dict scheduler_name → Metrics
    """
    if schedulers is None:
        schedulers = DEFAULT_SCHEDULERS
    # 1. Single workload generation — critical for fair comparison.
    base_jobs = generate_workload(scenario=scenario, seed=seed, count=num_jobs)
    # Keep a fingerprint to verify determinism in tests.
    results: dict[str, Metrics] = {}
    for name in schedulers:
        scheduler = make_scheduler(name)
        # 2. Isolate workload per scheduler.
        jobs_copy = copy.deepcopy(base_jobs)
        cluster = Cluster.homogeneous(num_gpus, memory_gb=gpu_memory_gb)
        sim = Simulator(cluster=cluster, scheduler=scheduler)
        sim.load_jobs(jobs_copy)
        history = sim.run()
        # 3. Metrics — simulator owns resource accounting.
        # Count scheduling decisions as number of launch events (completed + exceptional)
        # For this benchmark, scheduling_decisions = completed_jobs (no preemption).
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
        Dict seed → (Dict scheduler → Metrics)
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


def aggregate_results(
    results: dict[int, dict[str, Metrics]],
) -> pd.DataFrame:
    """Aggregate multi-seed results into mean±std per scheduler/metric.

    Returns a DataFrame with one row per scheduler and columns like
    ``avg_wait_mean``, ``avg_wait_std``, etc.
    """
    # Collect per-scheduler metric dicts across seeds.
    schedulers = set()
    for seed_map in results.values():
        schedulers.update(seed_map.keys())
    schedulers = sorted(schedulers)

    # Gather per-scheduler list of metric dicts.
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
        # Also store per-seed raw values for transparency (as JSON string)
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


# ---------------------------------------------------------------------------
# Persistence — CSV + JSON under artifacts/benchmarks/
# ---------------------------------------------------------------------------

def save_benchmark(
    scenario: str,
    results: dict[int, dict[str, Metrics]],
    out_dir: str | Path = "artifacts/benchmarks",
) -> dict[str, Path]:
    """Save machine-readable results.

    Writes:
        {out_dir}/{scenario}.csv       — per-seed tidy results
        {out_dir}/{scenario}_agg.csv   — mean±std aggregation
        {out_dir}/{scenario}.json      — full per-seed metrics as JSON
        {out_dir}/summary.csv          — appended summary across scenarios (if called multiple times)

    Returns:
        Dict of written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tidy = per_seed_dataframe(results)
    agg = aggregate_results(results)

    # Ensure scenario column for summary
    tidy["scenario"] = scenario
    agg["scenario"] = scenario

    p_tidy = out / f"{scenario}.csv"
    p_agg = out / f"{scenario}_agg.csv"
    p_json = out / f"{scenario}.json"

    tidy.to_csv(p_tidy, index=False)
    agg.to_csv(p_agg, index=False)

    # JSON: seed → scheduler → metrics dict
    serializable = {str(seed): {sched: m.as_dict() for sched, m in sched_map.items()} for seed, sched_map in results.items()}
    p_json.write_text(json.dumps({"scenario": scenario, "seeds": list(results.keys()), "results": serializable}, indent=2))

    # Append to summary.csv (tidy across all scenarios if desired by caller)
    # We write a per-scenario summary row per scheduler into summary.csv;
    # caller that loops over scenarios should concatenate agg DataFrames.
    return {"tidy": p_tidy, "agg": p_agg, "json": p_json, "tidy_df": tidy, "agg_df": agg}  # type: ignore[return-value]


def run_and_save(
    scenario: str,
    seeds: list[int],
    schedulers: list[str] | None = None,
    num_jobs: int = 100,
    num_gpus: int = 8,
    gpu_memory_gb: float = 80.0,
    out_dir: str | Path = "artifacts/benchmarks",
) -> dict[str, Path]:
    """Convenience: run multi-seed benchmark and save to disk."""
    results = run_benchmark(
        scenario=scenario,
        seeds=seeds,
        schedulers=schedulers,
        num_jobs=num_jobs,
        num_gpus=num_gpus,
        gpu_memory_gb=gpu_memory_gb,
    )
    return save_benchmark(scenario=scenario, results=results, out_dir=out_dir)
