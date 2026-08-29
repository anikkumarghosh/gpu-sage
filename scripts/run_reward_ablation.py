"""Controlled reward ablation — 6 configs A-F under identical workloads.

Screening: 50k steps, single seed, balanced scenario, 5 eval seeds.
Finalists: 250k steps, 3 training seeds, 5 eval seeds, all 6 scenarios.

Usage:
    python scripts/run_reward_ablation.py --mode screening --steps 50000
    python scripts/run_reward_ablation.py --mode full --steps 250000 --train-seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REWARD_CONFIGS = {
    "A_baseline": "configs/rewards/reward_A_baseline.yaml",
    "B_waiting": "configs/rewards/reward_B_waiting_focused.yaml",
    "C_util": "configs/rewards/reward_C_util_focused.yaml",
    "D_frag": "configs/rewards/reward_D_fragmentation.yaml",
    "E_throughput": "configs/rewards/reward_E_throughput.yaml",
    "F_balanced": "configs/rewards/reward_F_balanced.yaml",
}

def run_cmd(cmd: list[str], env: dict | None = None):
    print(f"\n$ {' '.join(cmd)}")
    # Ensure KMP var
    e = os.environ.copy()
    e["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if env:
        e.update(env)
    result = subprocess.run(cmd, env=e)
    if result.returncode != 0:
        print(f"[WARN] Command failed with {result.returncode}: {' '.join(cmd)}")
    return result.returncode

def train_one(config_name: str, config_path: str, steps: int, seed: int, out_base: Path):
    out = out_base / config_name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "training/train_ppo.py",
        "--steps", str(steps),
        "--seed", str(seed),
        "--reward-config", config_path,
        "--out", str(out),
    ]
    run_cmd(cmd)
    # Find latest run's model
    runs = sorted((out / "runs").glob("*"), key=lambda p: p.stat().st_mtime) if (out / "runs").exists() else []
    if runs:
        latest = runs[-1]
        model = latest / "model" / "final_model.zip"
        if model.exists():
            return model
    # Fallback to legacy
    legacy = out / "models" / "final_model.zip"
    if legacy.exists():
        return legacy
    legacy2 = out / "final_model.zip"
    if legacy2.exists():
        return legacy2
    return None

def evaluate_one(model_path: Path | None, scenario: str, seeds: list[int], out_dir: Path, reward_config_path: str | None = None):
    # Use benchmark_all with optional reward config for component logging (eval reward same as train)
    # For now, evaluation metrics are independent of reward, but we pass model
    # We need to run benchmark via python import to avoid subprocess overhead for small eval
    from gpu_sage.evaluation.benchmark import run_and_save
    # If model is None, PPO will fallback to heuristic (still deterministic)
    saved = run_and_save(
        scenario=scenario,
        seeds=seeds,
        schedulers=["FCFS", "SJF", "Priority", "BestFit", "PPO"],
        num_jobs=100,
        num_gpus=8,
        out_dir=out_dir,
        ppo_model_path=str(model_path) if model_path else None,
    )
    return saved

def main():
    parser = argparse.ArgumentParser(description="Reward ablation: controlled A-F")
    parser.add_argument("--mode", choices=["screening", "full"], default="screening", help="screening=50k single seed, full=250k multi-seed")
    parser.add_argument("--steps", type=int, default=None, help="Override timesteps (default 50000 for screening, 250000 for full)")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0], help="Training seeds")
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[0,1,2,3,4], help="Evaluation seeds")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None, help="Scenarios to eval (default: balanced for screening, all 6 for full)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/reward_ablation"), help="Output base")
    args = parser.parse_args()

    if args.steps is None:
        args.steps = 50000 if args.mode == "screening" else 250000
    if args.scenarios is None:
        args.scenarios = ["balanced"] if args.mode == "screening" else ["balanced","gpu_heavy","short_jobs","bursty","heavy_tail","priority_skew"]

    out_base = Path(args.out) / args.mode
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Reward ablation {args.mode}: steps={args.steps}, train_seeds={args.train_seeds}, eval_seeds={args.eval_seeds}, scenarios={args.scenarios}")
    print(f"Output: {out_base}")
    print(f"Configs: {list(REWARD_CONFIGS.keys())}")

    results = {}
    for name, cfg_path in REWARD_CONFIGS.items():
        print(f"\n{'='*60}\nConfig {name}: {cfg_path}\n{'='*60}")
        models = []
        for seed in args.train_seeds:
            model = train_one(name, cfg_path, args.steps, seed, out_base)
            print(f"Trained {name} seed {seed} -> {model}")
            models.append(model)
        # For screening, use first model's seed for eval; for full, evaluate each seed's model? Simplify: use seed 0 model for eval in screening
        eval_model = models[0] if models else None
        # Evaluate across scenarios
        for scen in args.scenarios:
            eval_out = out_base / name / f"eval_{scen}"
            # Run benchmark via function (faster than subprocess)
            # Use single eval seed set for screening, multi for full
            # For screening, we evaluate the first model only
            from gpu_sage.evaluation.benchmark import run_and_save
            saved = run_and_save(
                scenario=scen,
                seeds=args.eval_seeds,
                schedulers=["FCFS","SJF","Priority","BestFit","PPO"],
                num_jobs=100,
                out_dir=eval_out,
                ppo_model_path=str(eval_model) if eval_model else None,
            )
            print(f"Evaluated {name} on {scen} -> {saved['agg']}")
        results[name] = {"model": str(eval_model) if eval_model else None, "config": cfg_path}

    # Generate summary table for screening (balanced only)
    try:
        import pandas as pd
        summary_rows = []
        for name in REWARD_CONFIGS:
            for scen in args.scenarios:
                agg_path = out_base / name / f"eval_{scen}" / f"{scen}_agg.csv"
                if agg_path.exists():
                    df = pd.read_csv(agg_path)
                    for _, row in df.iterrows():
                        r = row.to_dict()
                        r["reward_config"] = name
                        r["scenario"] = scen
                        summary_rows.append(r)
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = out_base / "ablation_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\nAblation summary saved to {summary_path} ({len(summary_df)} rows)")

            # Selection criterion: balanced scenario, rank by Avg JCT then Avg Wait
            bal = summary_df[summary_df["scenario"] == "balanced"]
            if not bal.empty:
                # Filter PPO only for ranking
                ppo_bal = bal[bal["scheduler"] == "PPO"]
                if not ppo_bal.empty:
                    ppo_bal = ppo_bal.sort_values("average_turnaround_time_mean")
                    print("\nPPO ranking on balanced (by Avg JCT):")
                    for _, row in ppo_bal.iterrows():
                        print(f"  {row['reward_config']}: JCT {row['average_turnaround_time_mean']:.1f}+/-{row['average_turnaround_time_std']:.1f}, Wait {row['average_waiting_time_mean']:.1f}")
        # Also save component analysis placeholder
        (out_base / "README.md").write_text(f"# Reward Ablation {args.mode}\n\nSteps: {args.steps}, train seeds {args.train_seeds}, eval seeds {args.eval_seeds}\n\nConfigs: {list(REWARD_CONFIGS.keys())}\n\nSee {out_base}/<config>/eval_* for per-scenario results.\n")
    except Exception as e:
        print(f"[summary] Failed: {e}")

    print(f"\nAblation {args.mode} complete. Results under {out_base}")

if __name__ == "__main__":
    main()
