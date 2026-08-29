"""Unified benchmark: FCFS / SJF / Priority / Best-Fit / PPO on SAME workloads.

One command → same workload → all 5 schedulers → same metrics → comparison table.

Usage:
    python scripts/benchmark_all.py --scenario balanced --seeds 0 1 2 3 4 --ppo-model artifacts/ppo/models/final_model.zip
    python scripts/benchmark_all.py --all --seeds 0 1 2 --ppo-model artifacts/ppo/final_model.zip --jobs 50
    python scripts/benchmark_all.py --scenario balanced --seeds 0 1 2   # without PPO (baselines only)

This reuses the same rigorous benchmark framework as scripts/benchmark_baselines.py,
but adds PPO via deterministic env-based evaluation on the identical W0 per seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpu_sage.evaluation.benchmark import DEFAULT_SCHEDULERS, available_schedulers, run_and_save
from gpu_sage.workloads.generator import SCENARIO_NAMES


def format_mean_std(mean: float, std: float, multi: bool) -> str:
    if multi:
        return f"{mean:.2f} +/- {std:.2f}"
    return f"{mean:.2f}"


def print_comparison(scenario: str, agg: pd.DataFrame, seeds: list[int]) -> None:
    multi = len(seeds) > 1
    cols = [
        ("Scheduler", "scheduler"),
        ("Avg Wait", "average_waiting_time"),
        ("P95 Wait", "p95_waiting_time"),
        ("Avg JCT", "average_turnaround_time"),
        ("P95 JCT", "p95_turnaround_time"),
        ("Utilization", "gpu_utilization"),
        ("Throughput", "throughput_jobs_per_time"),
    ]
    header = " | ".join(f"{name:>14}" for name, _ in cols)
    sep = "-+-".join("-" * 14 for _ in cols)
    print(f"\nScenario: {scenario}  |  seeds={seeds}  |  n={len(seeds)}")
    print(header)
    print(sep)
    for _, row in agg.iterrows():
        parts = []
        for disp, key in cols:
            if key == "scheduler":
                parts.append(f"{row['scheduler']:>14}")
            else:
                mean = row.get(f"{key}_mean", row.get(key, 0.0))
                std = row.get(f"{key}_std", 0.0)
                parts.append(f"{format_mean_std(float(mean), float(std), multi):>14}")
        print(" | ".join(parts))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified PPO vs baselines benchmark")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--scenario", type=str, choices=SCENARIO_NAMES, help="Single scenario")
    g.add_argument("--all", action="store_true", help="All scenarios")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds")
    parser.add_argument("--jobs", "--num-jobs", dest="num_jobs", type=int, default=100, help="Jobs per workload")
    parser.add_argument("--gpus", "--num-gpus", dest="num_gpus", type=int, default=8, help="Cluster size")
    parser.add_argument("--ppo-model", type=Path, default=None, help="Path to PPO model (.zip) to include PPO scheduler")
    parser.add_argument("--schedulers", type=str, nargs="+", default=None, help=f"Schedulers to compare. Available: {available_schedulers()}")
    parser.add_argument("--out", "--output", dest="out", type=Path, default=Path("artifacts/benchmarks"), help="Output dir")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        args.scenario = "balanced"

    # Determine scheduler list
    if args.schedulers is not None:
        schedulers = args.schedulers
    else:
        schedulers = list(DEFAULT_SCHEDULERS)
        if args.ppo_model is not None:
            schedulers.append("PPO")

    avail = set(available_schedulers())
    for s in schedulers:
        if s not in avail:
            parser.error(f"Unknown scheduler '{s}'. Available: {sorted(avail)}")
    if "PPO" in schedulers and args.ppo_model is None:
        print("[WARN] PPO scheduler requested but no --ppo-model provided; PPO will use heuristic fallback (still deterministic)")

    scenarios = SCENARIO_NAMES if args.all else [args.scenario]  # type: ignore

    args.out.mkdir(parents=True, exist_ok=True)

    all_aggs: list[pd.DataFrame] = []
    for scen in scenarios:
        saved = run_and_save(
            scenario=scen,
            seeds=args.seeds,
            schedulers=schedulers,
            num_jobs=args.num_jobs,
            num_gpus=args.num_gpus,
            out_dir=args.out,
            ppo_model_path=args.ppo_model,
        )
        agg = saved["agg_df"]  # type: ignore
        all_aggs.append(agg)
        print_comparison(scen, agg, args.seeds)
        base = f"  -> saved {saved['tidy']} , {saved['agg']} , {saved['json']}"
        if "jobs_csv" in saved:
            base += f" , {saved['jobs_csv']}"
        if "ppo_log_csv" in saved:
            base += f" , {saved['ppo_log_csv']} (PPO decisions)"
        print(base)

    if len(all_aggs) > 1:
        summary = pd.concat(all_aggs, ignore_index=True)
        summary_path = args.out / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
