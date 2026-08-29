"""Evaluate a trained PPO model on a fixed workload — same workload as baselines.

Usage:
    python scripts/benchmark_ppo.py --model artifacts/ppo/models/final_model.zip --scenario balanced --seed 0
    python scripts/benchmark_ppo.py --model artifacts/ppo/final_model.zip --scenario balanced --seed 0 --jobs 50

The PPO evaluation uses deterministic inference and the SAME workload W0 that
baselines see, guaranteeing fair comparison.

For example, seed 0 generates W0 once, then FCFS/SJF/Priority/BestFit/PPO all
evaluate on that identical W0.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from gpu_sage.evaluation.benchmark import evaluate_ppo_fixed_workload, run_single_seed
from gpu_sage.evaluation.metrics import compute_metrics
from gpu_sage.workloads.generator import SCENARIO_NAMES, generate_workload, get_scenario_config


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO deterministic evaluation on fixed workload")
    parser.add_argument("--model", type=Path, required=True, help="Path to MaskablePPO model (.zip) or directory")
    parser.add_argument("--scenario", type=str, choices=SCENARIO_NAMES, default="balanced", help="Workload scenario")
    parser.add_argument("--seed", type=int, default=0, help="Workload seed (W0)")
    parser.add_argument("--jobs", "--num-jobs", dest="num_jobs", type=int, default=50, help="Number of jobs")
    parser.add_argument("--gpus", "--num-gpus", dest="num_gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--out", "--output", dest="out", type=Path, default=Path("artifacts/benchmarks"), help="Output dir for logs")
    args = parser.parse_args()

    workload_cfg = get_scenario_config(args.scenario)

    # 1. Generate canonical workload W0 once
    base_jobs = generate_workload(scenario=args.scenario, seed=args.seed, count=args.num_jobs)
    print(f"Generated workload W0: scenario={args.scenario} seed={args.seed} jobs={len(base_jobs)}")

    # 2. Evaluate PPO on W0 via deterministic env loop
    metrics, per_job, decision_logs, stats = evaluate_ppo_fixed_workload(
        jobs=copy.deepcopy(base_jobs),
        model_path=args.model,
        num_gpus=args.num_gpus,
        workload_config=workload_cfg,
        seed=args.seed,
    )
    print("\n=== PPO Evaluation (deterministic) ===")
    print(f"Model: {args.model}")
    print(f"Completed: {metrics.completed_jobs}/{metrics.total_jobs}  Throughput: {metrics.throughput:.4f}")
    print(f"Avg Wait: {metrics.average_waiting_time:.2f}  P95 Wait: {metrics.p95_waiting_time:.2f}")
    print(f"Avg JCT: {metrics.average_turnaround_time:.2f}  P95 JCT: {metrics.p95_turnaround_time:.2f}")
    print(f"Utilization: {metrics.gpu_utilization:.3f}  Idle: {metrics.gpu_idle_time:.3f}")
    print(f"Scheduling decisions: {stats['scheduling_decisions']}  NOOP: {stats['noop_actions']}  Invalid: {stats['invalid_actions']}")
    print(f"Avg Reward: {stats['avg_reward']:.3f}  Steps: {stats['steps']}")
    print(f"Action dist: {stats['action_distribution']}")

    # 3. Also run FCFS on same W0 for direct comparison (identical workload check)
    fcfs_metrics = run_single_seed(
        scenario=args.scenario, seed=args.seed, schedulers=["FCFS"], num_jobs=args.num_jobs, num_gpus=args.num_gpus
    )["FCFS"]
    print("\n=== FCFS on SAME W0 (for comparison) ===")
    print(f"Avg Wait: {fcfs_metrics.average_waiting_time:.2f}  P95 Wait: {fcfs_metrics.p95_waiting_time:.2f}")
    print(f"Avg JCT: {fcfs_metrics.average_turnaround_time:.2f}  Utilization: {fcfs_metrics.gpu_utilization:.3f}")

    # Verify identical workload job IDs
    ppo_job_ids = sorted(r["job_id"] for r in per_job)
    base_ids = sorted(j.job_id for j in base_jobs)
    assert ppo_job_ids == base_ids, "PPO and baseline must see same job IDs"
    print(f"\nVerified identical workload: job IDs {base_ids[:5]}... (n={len(base_ids)})")

    # 4. Save per-decision log
    args.out.mkdir(parents=True, exist_ok=True)
    import json
    import pandas as pd

    if decision_logs:
        log_df = pd.DataFrame(decision_logs)
        log_path = args.out / f"{args.scenario}_seed{args.seed}_ppo_decisions.csv"
        log_df.to_csv(log_path, index=False)
        print(f"\nPer-decision log saved to {log_path} (rows={len(log_df)})")
        # Also save JSON with stats
        meta_path = args.out / f"{args.scenario}_seed{args.seed}_ppo_eval.json"
        meta_path.write_text(
            json.dumps(
                {
                    "scenario": args.scenario,
                    "seed": args.seed,
                    "model": str(args.model),
                    "metrics": metrics.as_dict(),
                    "stats": {k: (dict(v) if isinstance(v, dict) else v) for k, v in stats.items()},
                    "per_job": per_job[:3],  # sample
                },
                indent=2,
            )
        )
        print(f"Eval meta saved to {meta_path}")

    print("\nPPO evaluation complete — deterministic, same workload as baselines.")


if __name__ == "__main__":
    main()
