"""Generate reward ablation analysis: tables, plots, component and starvation analysis."""
import pathlib, json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = pathlib.Path("artifacts/reward_ablation")
screen_dir = base / "screening"
full_base = base / "full"

# Screening summary
print("=== Screening (50k, balanced, 5 eval seeds) ===")
screen_summary = screen_dir / "ablation_summary.csv"
if screen_summary.exists():
    df = pd.read_csv(screen_summary)
    # Filter to PPO only for ranking
    ppo = df[(df["scheduler"]=="PPO") & (df["scenario"]=="balanced")]
    print(ppo[["reward_config","average_waiting_time_mean","p95_waiting_time_mean","average_turnaround_time_mean","gpu_utilization_mean"]].to_string(index=False))
    # Save table
    ppo.to_csv(screen_dir / "ppo_ranking_balanced.csv", index=False)

# For full evaluation, we have two finalists: A_baseline and F_balanced with 5 seeds each
# Load their eval_5seeds
for name in ["A_baseline", "F_balanced"]:
    eval_dir = full_base / name / "eval_5seeds"
    if not eval_dir.exists():
        continue
    # Generate comparison plots for this config (already done via benchmark, but ensure)
    from gpu_sage.evaluation.plots import generate_comparison_plots
    out = eval_dir / "plots"
    generate_comparison_plots(eval_dir, out)
    print(f"Plots for {name} -> {out}")

# Combined reward vs performance table (balanced, PPO only across configs)
print("\n=== Reward vs Performance (balanced, PPO) ===")
rows = []
for name in ["A_baseline","B_waiting","C_util","D_frag","E_throughput","F_balanced"]:
    # Screening eval for balanced
    p = screen_dir / name / "eval_balanced" / "balanced_agg.csv"
    if p.exists():
        df = pd.read_csv(p)
        r = df[df["scheduler"]=="PPO"].iloc[0]
        rows.append({
            "Reward": name,
            "Avg JCT": f"{r['average_turnaround_time_mean']:.1f}+/-{r['average_turnaround_time_std']:.1f}",
            "P95 JCT": f"{r['p95_turnaround_time_mean']:.1f}+/-{r['p95_turnaround_time_std']:.1f}",
            "Avg Wait": f"{r['average_waiting_time_mean']:.1f}+/-{r['average_waiting_time_std']:.1f}",
            "P95 Wait": f"{r['p95_waiting_time_mean']:.1f}+/-{r['p95_waiting_time_std']:.1f}",
            "Util": f"{r['gpu_utilization_mean']:.3f}+/-{r['gpu_utilization_std']:.3f}",
            "Throughput": f"{r['throughput_mean']:.4f}" if 'throughput_mean' in r else f"{r.get('throughput_jobs_per_time_mean',0):.4f}",
        })
import pandas as pd
if rows:
    pdf = pd.DataFrame(rows)
    print(pdf.to_string(index=False))
    pdf.to_csv(base / "reward_vs_performance_balanced.csv", index=False)

# Reward component contribution (from decision logs)
print("\n=== Reward Component Contribution (avg per step) ===")
for name in ["A_baseline","F_balanced"]:
    # Find latest full run's decision logs for balanced
    eval_dir = full_base / name / "eval_5seeds"
    # Decision logs are per scenario, we need to aggregate
    import glob
    logs = list(eval_dir.glob("*_ppo_decisions.csv"))
    if not logs:
        continue
    dfs = []
    for lf in logs:
        try:
            df = pd.read_csv(lf)
            dfs.append(df)
        except: pass
    if dfs:
        all_logs = pd.concat(dfs)
        # Average per component
        comps = ["throughput_reward","waiting_penalty","utilization_reward","fragmentation_penalty","idle_penalty","invalid_penalty"]
        print(f"\n{name}:")
        for c in comps:
            if c in all_logs.columns:
                print(f"  {c}: {all_logs[c].mean():.3f} (sum {all_logs[c].sum():.1f})")
        # Total
        if "reward" in all_logs.columns:
            print(f"  total_reward mean: {all_logs['reward'].mean():.3f}")

# Starvation analysis: per GPU requirement
print("\n=== Starvation by GPU requirement (balanced, PPO) ===")
for name in ["A_baseline","F_balanced"]:
    # Find per-job file for balanced
    jf = None
    for cand in [full_base / name / "eval_5seeds" / "balanced_jobs.csv", base / "screening" / name / "eval_balanced" / "balanced_jobs.csv"]:
        if cand.exists():
            jf = cand
            break
    if jf and jf.exists():
        df = pd.read_csv(jf)
        # Filter PPO only
        ppo_jobs = df[df["scheduler"]=="PPO"]
        if not ppo_jobs.empty:
            print(f"\n{name} (n={len(ppo_jobs)}):")
            for gpu in sorted(ppo_jobs["gpu_requirement"].unique()):
                sub = ppo_jobs[ppo_jobs["gpu_requirement"]==gpu]
                print(f"  GPU {int(gpu)}: n={len(sub)}, avg_wait={sub['waiting_time'].mean():.1f}, P95={np.percentile(sub['waiting_time'].dropna(),95):.1f}, max={sub['waiting_time'].max():.1f}")

# GPU-heavy failure focused
print("\n=== GPU-heavy failure analysis ===")
for name in ["A_baseline","F_balanced"]:
    jf = full_base / name / "eval_5seeds" / "gpu_heavy_jobs.csv"
    if not jf.exists():
        jf = screen_dir / name / "eval_balanced" / "balanced_jobs.csv"
    if jf.exists():
        df = pd.read_csv(jf)
        ppo = df[df["scheduler"]=="PPO"]
        # Check large job avoidance
        large = ppo[ppo["gpu_requirement"]>=4]
        print(f"{name} gpu_heavy large jobs: {len(large)}/{len(ppo)} ({len(large)/max(len(ppo),1)*100:.1f}%)")

# Plots for reward vs performance
try:
    # Plot reward config vs Avg JCT
    if 'pdf' in locals():
        plt.figure(figsize=(10,6))
        # Need numeric values for plot, reload with means
        comps = []
        for name in ["A_baseline","B_waiting","C_util","D_frag","E_throughput","F_balanced"]:
            p = screen_dir / name / "eval_balanced" / "balanced_agg.csv"
            if p.exists():
                df = pd.read_csv(p)
                r = df[df["scheduler"]=="PPO"].iloc[0]
                comps.append((name, r["average_turnaround_time_mean"], r["average_turnaround_time_std"]))
        if comps:
            names, means, stds = zip(*comps)
            plt.bar(names, means, yerr=stds, capsize=4)
            plt.ylabel("Avg JCT")
            plt.title("Reward Config vs Avg JCT (balanced, 5 seeds, 50k screening)")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(base / "reward_vs_jct.png", dpi=150)
            plt.close()
            print(f"Saved {base / 'reward_vs_jct.png'}")
except Exception as e:
    print(f"Plot failed: {e}")

print("\nAnalysis complete.")
