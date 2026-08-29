"""Compare classical scheduling heuristics on the same workload — rigorous benchmark.

Usage:
    python scripts/benchmark_baselines.py --scenario balanced --seeds 0 1 2 3 4 5 6 7 8 9
    python scripts/benchmark_baselines.py --all --seeds 0 1 2 3 4 5 6 7 8 9
    python scripts/benchmark_baselines.py --scenario gpu_heavy --seeds 0 --num-jobs 50

The benchmark guarantees identical workloads per seed/scenario across all schedulers.
Results are saved to artifacts/benchmarks/ as CSV + JSON and a comparison table
is printed with mean +/- std for multi-seed runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gpu_sage.evaluation.benchmark import (
    DEFAULT_SCHEDULERS,
    available_schedulers,
    run_and_save,
)
from gpu_sage.workloads.generator import SCENARIO_NAMES


def format_mean_std(mean: float, std: float, multi: bool) -> str:
    if multi:
        return f"{mean:.2f} +/- {std:.2f}"
    return f"{mean:.2f}"


def print_comparison(
    scenario: str,
    agg: pd.DataFrame,
    seeds: list[int],
) -> None:
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
    parser = argparse.ArgumentParser(description="Rigorous benchmark for GPU-Sage schedulers")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--scenario", type=str, choices=SCENARIO_NAMES, help="Single scenario to benchmark")
    g.add_argument("--all", action="store_true", help="Benchmark all scenarios")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="List of seeds, e.g. --seeds 0 1 2 3 4")
    # Support both --num-jobs and --jobs (and --num_jobs) for compatibility
    parser.add_argument("--num-jobs", "--jobs", dest="num_jobs", type=int, default=100, help="Jobs per workload")
    parser.add_argument("--num-gpus", "--gpus", dest="num_gpus", type=int, default=8, help="Cluster size")
    parser.add_argument(
        "--schedulers",
        type=str,
        nargs="+",
        default=None,
        help=f"Schedulers to compare. Available: {available_schedulers()}. Default: {DEFAULT_SCHEDULERS}",
    )
    # Support both --out and --output
    parser.add_argument("--out", "--output", dest="out", type=str, default="artifacts/benchmarks", help="Output directory")
    args = parser.parse_args()

    if not args.scenario and not args.all:
        args.scenario = "balanced"

    schedulers = args.schedulers if args.schedulers is not None else DEFAULT_SCHEDULERS
    avail = set(available_schedulers())
    for s in schedulers:
        if s not in avail:
            parser.error(f"Unknown scheduler '{s}'. Available: {sorted(avail)}")

    scenarios: list[str]
    if args.all:
        scenarios = SCENARIO_NAMES
    else:
        scenarios = [args.scenario]  # type: ignore[list-item]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_aggs: list[pd.DataFrame] = []
    for scen in scenarios:
        saved = run_and_save(
            scenario=scen,
            seeds=args.seeds,
            schedulers=schedulers,
            num_jobs=args.num_jobs,
            num_gpus=args.num_gpus,
            out_dir=out_dir,
        )
        agg = saved["agg_df"]  # type: ignore[typeddict-item]
        all_aggs.append(agg)
        print_comparison(scen, agg, args.seeds)
        # Print saved paths including per-job artifacts
        base_msg = f"  -> saved {saved['tidy']} , {saved['agg']} , {saved['json']}"
        if "jobs_csv" in saved:
            base_msg += f" , {saved['jobs_csv']}"
        print(base_msg)

    if len(all_aggs) > 1:
        summary = pd.concat(all_aggs, ignore_index=True)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print("\n=== Summary (mean utilization & avg wait across seeds) ===")
        for scen in scenarios:
            agg = [a for a in all_aggs if (a["scenario"] == scen).any()][0]
            print_comparison(scen, agg, args.seeds)


if __name__ == "__main__":
    main()
