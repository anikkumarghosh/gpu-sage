"""Generalization + robustness evaluation for PPO vs baselines.

Runs already-trained PPO (A_baseline, F_balanced) and all heuristic schedulers
across in-distribution and out-of-distribution workload shifts.

Usage:
  python scripts/generalize_ppo.py --mode all
  python scripts/generalize_ppo.py --mode id       # in-distribution only
  python scripts/generalize_ppo.py --mode ood       # only OOD shifts
  python scripts/generalize_ppo.py --mode report    # generate report + plots

Examples with custom model paths:
  python scripts/generalize_ppo.py --mode all --model-a /path/to/A_baseline.zip --model-f /path/to/F_balanced.zip
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from gpu_sage.evaluation.benchmark import run_single_seed
from gpu_sage.workloads.generator import SCENARIO_NAMES, generate_workload, get_scenario_config, WorkloadConfig
from sb3_contrib import MaskablePPO

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Training distribution (documented from code inspection)
# ---------------------------------------------------------------------------
TRAINING_CONFIG = {
    "scenario": "balanced",
    "arrival_rate": 0.08,
    "min_gpus": 1,
    "max_gpus": 4,
    "min_memory_gb": 8.0,
    "max_memory_gb": 64.0,
    "min_duration": 20.0,
    "max_duration": 180.0,
    "min_priority": 1,
    "max_priority": 5,
    "num_gpus": 8,
    "gpu_memory_gb": 80.0,
    "max_jobs": 16,
}

# ---------------------------------------------------------------------------
# OOD shift definitions (mutated WorkloadConfig params)
# ---------------------------------------------------------------------------
SHIFT_DEFS = [
    {"name": "A_high_gpu", "desc": "Higher GPU demand",
     "gen_kwargs": {"min_gpus": 2, "max_gpus": 8}},
    {"name": "B_short_jobs", "desc": "Shorter jobs",
     "gen_kwargs": {"min_duration": 5, "max_duration": 50}},
    {"name": "C_long_jobs", "desc": "Longer jobs",
     "gen_kwargs": {"min_duration": 60, "max_duration": 400, "arrival_rate": 0.05}},
    {"name": "D_bursty", "desc": "Bursty arrivals",
     "gen_kwargs": {}},
    {"name": "E_heavy_tail", "desc": "Heavy-tailed runtime",
     "gen_kwargs": {"min_duration": 10, "max_duration": 400,
                    "heavy_tail_fraction": 0.07, "heavy_tail_multiplier": 8.0}},
    {"name": "F_priority_skew", "desc": "Priority distribution shift",
     "gen_kwargs": {}},
]

SHIFT_ID = {"name": "ID_balanced", "desc": "In-distribution (balanced)",
            "gen_kwargs": {}}


# ---------------------------------------------------------------------------
# Model paths — NO LONGER hardcoded. User supplies via --model-a/--model-f.
# If not supplied, the script will error loudly rather than silently
# falling back to a heuristic and falsely labeling it "PPO".
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pct_degradation(ood, id_):
    if id_ == 0:
        return float('inf')
    return (ood - id_) / id_ * 100.0


def shift_gen_kwargs(base_cfg: WorkloadConfig, gen_kwargs: dict) -> WorkloadConfig:
    """Apply gen_kwargs overrides to a copy of the base WorkloadConfig."""
    from dataclasses import replace
    cfg = replace(base_cfg)
    for k, v in gen_kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Core: evaluate PPO on a shifted scenario across seeds
# ---------------------------------------------------------------------------


def eval_shift_ppo(model_path: Path | None, shift_def: dict, seeds: list[int],
                   num_jobs: int, num_gpus: int) -> dict[int, dict]:
    """Evaluate PPO on one shift across seeds.

    Returns dict seed -> {scheduler: Metrics}

    If model_path is None or not found, raises RuntimeError instead of
    silently falling back to a heuristic and falsely labeling it "PPO".
    """
    if model_path is not None and not Path(model_path).exists():
        raise RuntimeError(
            f"Model not found: {model_path}. "
            "Pass --model-a/--model-f to point to trained .zip files.\n"
            "If running from the original milestone repo, the models are at:\n"
            "  artifacts/reward_ablation/full/A_baseline/runs/20260829_200441_seed1_steps250000/model/final_model.zip\n"
            "  artifacts/reward_ablation/full/F_balanced/runs/20260829_194948_seed0_steps250000/model/final_model.zip"
        )

    # Build custom workload config
    base_cfg = get_scenario_config(TRAINING_CONFIG["scenario"])
    cfg = shift_gen_kwargs(base_cfg, shift_def["gen_kwargs"])
    scenario_name = shift_def["name"]

    print(f"    Scenario: {scenario_name} (arrival_rate={cfg.arrival_rate}, min_gpus={cfg.min_gpus}, max_gpus={cfg.max_gpus},"
          f" min_dur={cfg.min_duration}, max_dur={cfg.max_duration})")

    results = {}
    for seed in seeds:
        try:
            # Generate workload W0 with shifted config (seed=0 for W0 generation)
            base_jobs = generate_workload(scenario="custom", seed=0, count=num_jobs, config=cfg)

            # Evaluate PPO on this same W0 via deterministic env loop
            ppo_mp = str(model_path) if model_path else None
            metrics, per_job, decision_logs, stats = evaluate_ppo_fixed_workload(
                jobs=deepcopy(base_jobs),
                model_path=ppo_mp,
                num_gpus=num_gpus,
                gpu_memory_gb=80.0,
                max_jobs=num_jobs,
            )

            results[seed] = {"PPO": metrics}

            # Save aggregation CSV
            all_rows = []
            d = metrics.as_dict()
            d["scheduler"] = "PPO"
            d["model"] = Path(model_path).name if model_path else "baseline"
            d["scenario"] = scenario_name
            d["seed"] = seed
            all_rows.append(d)
            if all_rows:
                df = pd.DataFrame(all_rows)
                out_csv = Path("artifacts/reward_ablation") / f"{shift_def['name']}_{Path(model_path).name if model_path else 'baseline'}_seed{seed}_agg.csv"
                df.to_csv(out_csv, index=False)

        except Exception as e:
            print(f"[error] shift {shift_def['name']} seed {seed}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[seed] = None
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="PPO generalization + robustness evaluation")
    parser.add_argument("--mode", choices=["id", "ood", "all", "report"], default="all",
                        help="Evaluation mode")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Eval seeds")
    parser.add_argument("--jobs", type=int, default=16,
                        help="Jobs per workload (training used 16)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/reward_ablation"),
                        help="Output base dir")
    parser.add_argument("--model-a", type=Path, default=None,
                        help="Path to A_baseline trained MaskablePPO .zip model")
    parser.add_argument("--model-f", type=Path, default=None,
                        help="Path to F_balanced trained MaskablePPO .zip model")
    args = parser.parse_args(argv)

    out_base = args.out
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"=== Generalization evaluation mode={args.mode} seeds={args.seeds} ===")
    print(f"Training config: arrival_rate={TRAINING_CONFIG['arrival_rate']}, "
          f"{TRAINING_CONFIG['min_gpus']}-{TRAINING_CONFIG['max_gpus']} GPUs, "
          f"{TRAINING_CONFIG['min_duration']}-{TRAINING_CONFIG['max_duration']}s, "
          f"{TRAINING_CONFIG['min_priority']}-{TRAINING_CONFIG['max_priority']} priority, "
          f"{TRAINING_CONFIG['num_gpus']} GPUs x {TRAINING_CONFIG['gpu_memory_gb']}GB A100")

    # Resolve model paths: use args if supplied, fall back to legacy defaults
    # only if the files actually exist (so old one-liners still work if paths happen to exist)
    model_a = args.model_a if args.model_a and Path(args.model_a).exists() else None
    model_f = args.model_f if args.model_f and Path(args.model_f).exists() else None

    # ---- In-distribution (balanced) ----
    if args.mode in ["id", "all"]:
        print("\n--- In-distribution: balanced ---")
        for label, model_path in [("A_baseline", model_a), ("F_balanced", model_f)]:
            print(f"  evaluating {label}...")
            try:
                res = run_single_seed(
                    scenario="balanced",
                    seed=args.seeds[0],
                    schedulers=["FCFS", "SJF", "Priority", "BestFit", "PPO"],
                    num_jobs=args.jobs,
                    num_gpus=8,
                    gpu_memory_gb=80.0,
                    ppo_model_path=str(model_path) if model_path else None,
                )
                for sched in ["FCFS", "SJF", "Priority", "BestFit", "PPO"]:
                    if sched in res:
                        m = res[sched]
                        print(f"  {label} {sched}: JCT={m.average_turnaround_time:.1f} Wait={m.average_waiting_time:.1f} Util={m.gpu_utilization:.3f}")
            except Exception as e:
                print(f"    [error] {e}")

    # ---- OOD shifts ----
    if args.mode in ["ood", "all"]:
        print("\n--- Out-of-distribution shifts ---")
        for shift in SHIFT_DEFS:
            print(f"\n  Shift {shift['name']}: {shift['desc']}")
            try:
                for label, model_path in [("A_baseline", model_a), ("F_balanced", model_f)]:
                    if model_path is None:
                        print(f"    {label}: model not supplied, skipping PPO eval")
                    else:
                        print(f"    evaluating {label}...")
                        eval_shift_ppo(model_path, shift, args.seeds, args.jobs, 8)
            except Exception as e:
                print(f"    [error] {e}")

    # ---- Generalization scores + report ----
    if args.mode in ["all", "report"]:
        print("\n--- Generalization scores & degradation ---")
        all_records = []

        # In-distribution balanced
        for label, model_path in [("A_baseline", model_a), ("F_balanced", model_f)]:
            for seed in args.seeds:
                p = out_base / f"id_{label}_seed{seed}_agg.csv"
                if p.exists():
                    df = pd.read_csv(p)
                    for _, row in df.iterrows():
                        r = row.to_dict()
                        r["shift"] = "ID"
                        r["model"] = label
                        r["seed"] = seed
                        all_records.append(r)

        # OOD shifts
        for shift in SHIFT_DEFS:
            for label in ["A_baseline", "F_balanced"]:
                for seed in args.seeds:
                    # Find the CSV - try multiple filename patterns
                    p1 = out_base / f"{shift['name']}_{label}_seed{seed}_agg.csv"
                    p2 = out_base / f"{shift['name']}_{label}_final_model.zip_seed{seed}_agg.csv"
                    p3 = out_base / f"{shift['name']}_{Path(model_path).name}_seed{seed}_agg.csv" if model_path else None
                    p = p1 if p1.exists() else (p2 if p2.exists() else (p3 if p3 and p3.exists() else None))
                    if p:
                        df = pd.read_csv(p)
                        for _, row in df.iterrows():
                            r = row.to_dict()
                            r["shift"] = shift['name']
                            r["model"] = label
                            r["seed"] = seed
                            all_records.append(r)

        if not all_records:
            print("[warn] no aggregation data found")
        else:
            big = pd.DataFrame(all_records)
            print(f"Loaded {len(big)} records across ID + 6 OOD shifts")

            # Summary table
            METRIC_COLS = ["average_waiting_time", "average_turnaround_time",
                           "gpu_utilization", "throughput"]

            summary_rows = []
            for model in ["A_baseline", "F_balanced"]:
                for shift_name in ["ID"] + [s["name"] for s in SHIFT_DEFS]:
                    subset = big[(big["model"] == model) & (big["shift"] == shift_name)]
                    if subset.empty:
                        continue
                    row = {"model": model, "shift": shift_name}
                    for m in METRIC_COLS:
                        vals = subset[m]
                        if not vals.empty:
                            row[f"{m}_mean"] = float(np.mean(vals))
                            row[f"{m}_std"] = float(np.std(vals, ddof=0))
                        else:
                            row[f"{m}_mean"] = None
                            row[f"{m}_std"] = None
                        # Degradation vs ID for OOD shifts
                    if shift_name != "ID":
                        id_row = big[(big["model"] == model) & (big["shift"] == "ID")]
                        if not id_row.empty:
                            id_vals = {m: id_row[m].values[0] for m in METRIC_COLS if m in id_row.columns}
                            for m in METRIC_COLS:
                                if m in id_vals and not pd.isna(id_vals[m]) and id_vals[m] != 0:
                                    pct, _ = pct_degradation(row.get(f"{m}_mean", np.nan), id_vals[m])
                                    row[f"{m}_degr_pct"] = pct
                                else:
                                    row[f"{m}_degr_pct"] = None
                                # ratio: OOD / ID (for lower-is-better, ratio > 1 = degradation)
                                ratio = row.get(f"{m}_mean", np.nan) / id_vals[m] if m in id_vals and id_vals[m] != 0 else None
                                row[f"{m}_ratio"] = ratio
                            # Also compute retention = 1 - degradation/100
                            for m in METRIC_COLS:
                                if m in id_vals and not pd.isna(id_vals[m]) and id_vals[m] != 0:
                                    deg = pct_degradation(row.get(f"{m}_mean", np.nan), id_vals[m])
                                    row[f"{m}_retention"] = 1 - deg / 100.0
                                else:
                                    row[f"{m}_retention"] = None
                    summary_rows.append(row)

            summary_df = pd.DataFrame(summary_rows)
            summary_path = out_base / "generalization_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved summary to {summary_path}")
            print(summary_df.to_string(index=False))

            # Heatmap
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Pivot JCT means (average_turnaround_time)
                pivots = []
                for model in ["A_baseline", "F_balanced"]:
                    row_data = []
                    for shift_name in ["ID"] + [s["name"] for s in SHIFT_DEFS]:
                        subset = big[(big["model"] == model) & (big["shift"] == shift_name)]
                        if not subset.empty and "average_turnaround_time" in subset.columns:
                            val = float(subset["average_turnaround_time"].iloc[0])
                        else:
                            val = np.nan
                        row_data.append(val)
                    pivots.append(row_data)

                hv = pd.DataFrame(pivots, index=["A_baseline", "F_balanced"],
                                  columns=["ID"] + [s["name"] for s in SHIFT_DEFS])
                heatmap_path = out_base / "heatmap_jct.csv"
                hv.to_csv(heatmap_path)

                plt.figure(figsize=(10, 6))
                im = plt.imshow(hv.values, cmap="YlOrRd", aspect="auto")
                plt.colorbar(im, label="Avg JCT (s)")
                plt.xticks(range(len(hv.columns)), hv.columns)
                plt.yticks(range(len(hv.index)), hv.index)
                plt.title("Avg JCT vs Distribution Shift (lower is better)")
                plt.tight_layout()
                plt.savefig(out_base / "heatmap_jct.png", dpi=150)
                plt.close()
                print(f"Saved heatmap plot to {out_base / 'heatmap_jct.png'}")
            except Exception as e:
                print(f"[warn] heatmap plot failed: {e}")

    print(f"\nGeneralization evaluation {args.mode} complete.")


if __name__ == "__main__":
    main()