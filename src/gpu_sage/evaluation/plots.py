"""Presentation-quality comparison plots for PPO vs baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "average_waiting_time": "Avg Waiting Time (s)",
    "p95_waiting_time": "P95 Waiting Time (s)",
    "average_turnaround_time": "Avg JCT (s)",
    "p95_turnaround_time": "P95 JCT (s)",
    "gpu_utilization": "GPU Utilization",
    "throughput_jobs_per_time": "Throughput (jobs/s)",
    "throughput": "Throughput (jobs/s)",
}

def generate_comparison_plots(
    benchmark_dir: Path | str = "artifacts/benchmarks",
    out_dir: Path | str | None = None,
    scenarios: list[str] | None = None,
):
    """Generate 5 comparison plots with error bars from aggregated benchmark CSVs.

    Reads {benchmark_dir}/{scenario}_agg.csv for each scenario, aggregates,
    and saves PNGs to out_dir.

    Args:
        benchmark_dir: Directory containing {scenario}_agg.csv
        out_dir: Where to save plots (defaults to benchmark_dir/plots)
        scenarios: List of scenario names; if None, auto-discovers from benchmark_dir
    """
    benchmark_dir = Path(benchmark_dir)
    if out_dir is None:
        out_dir = benchmark_dir / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover scenarios if not given
    if scenarios is None:
        scenarios = sorted([p.stem.replace("_agg", "") for p in benchmark_dir.glob("*_agg.csv")])

    if not scenarios:
        print(f"[plots] No scenarios found in {benchmark_dir}")
        return

    # Load all agg data
    all_data = {}
    for scen in scenarios:
        p = benchmark_dir / f"{scen}_agg.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                all_data[scen] = df
            except Exception as e:
                print(f"[plots] Could not read {p}: {e}")

    if not all_data:
        print("[plots] No data loaded")
        return

    # Collect scheduler order (consistent)
    sched_set = set()
    for df in all_data.values():
        sched_set.update(df["scheduler"].tolist())
    # Preferred order
    order = ["FCFS", "SJF", "Priority", "BestFit", "PPO", "Random"]
    schedulers = [s for s in order if s in sched_set] + sorted(sched_set - set(order))

    # For each metric, create a grouped bar chart: scenarios on x, schedulers as bars
    metrics_to_plot = ["average_waiting_time", "p95_waiting_time", "average_turnaround_time", "gpu_utilization", "throughput_jobs_per_time"]
    # Filter to available metrics
    available_metrics = []
    for m in metrics_to_plot:
        # Check if any df has this metric
        if any(f"{m}_mean" in df.columns for df in all_data.values()):
            available_metrics.append(m)

    for metric in available_metrics:
        ylabel = METRICS.get(metric, metric)
        plt.figure(figsize=(14, 6))
        x = np.arange(len(scenarios))
        width = 0.15
        # Color map
        colors = {"FCFS": "#1f77b4", "SJF": "#ff7f0e", "Priority": "#2ca02c", "BestFit": "#d62728", "PPO": "#9467bd", "Random": "#8c564b"}
        for i, sched in enumerate(schedulers):
            means = []
            stds = []
            for scen in scenarios:
                df = all_data[scen]
                row = df[df["scheduler"] == sched]
                if not row.empty and f"{metric}_mean" in row.columns:
                    means.append(float(row.iloc[0][f"{metric}_mean"]))
                    stds.append(float(row.iloc[0][f"{metric}_std"]) if f"{metric}_std" in row.columns else 0.0)
                else:
                    means.append(0)
                    stds.append(0)
            # Bar
            offset = (i - len(schedulers)/2 + 0.5) * width
            plt.bar(x + offset, means, width, yerr=stds, capsize=4, label=sched, color=colors.get(sched, None), edgecolor="black", linewidth=0.5)
        plt.xticks(x, scenarios, rotation=15)
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Scenario (mean ± std, 5 seeds)" if len(next(iter(all_data.values()))) > 0 else ylabel)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fname = metric.replace("_jobs_per_time", "").replace("average_", "avg_").replace("turnaround", "jct") + ".png"
        # More descriptive names
        name_map = {
            "average_waiting_time": "avg_waiting_time.png",
            "p95_waiting_time": "p95_waiting_time.png",
            "average_turnaround_time": "avg_jct.png",
            "gpu_utilization": "gpu_utilization.png",
            "throughput_jobs_per_time": "throughput.png",
        }
        fname = name_map.get(metric, f"{metric}.png")
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()
        print(f"[plots] Saved {out_dir / fname}")

    print(f"[plots] All comparison plots saved to {out_dir}")

def generate_training_plots_from_eval(run_dir: Path | str):
    """Fallback: ensure training plots exist, already handled by train_ppo."""
    pass
