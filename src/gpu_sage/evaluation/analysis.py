"""Pathological behavior analysis for PPO vs baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def analyze_pathologies(
    benchmark_dir: Path | str = "artifacts/benchmarks",
    out_path: Path | str | None = None,
) -> str:
    """Analyze starvation, large-job, fragmentation, NOOP, priority.

    Reads per-job and per-decision logs from benchmark_dir and returns
    a markdown report string; also saves to out_path if given.
    """
    benchmark_dir = Path(benchmark_dir)
    # Try to find latest run summary or just analyze available CSVs
    report_lines = ["# GPU-Sage PPO Pathological Analysis", ""]
    report_lines.append(f"Benchmark dir: `{benchmark_dir}`")
    report_lines.append("")

    # Find all _jobs.csv
    job_files = list(benchmark_dir.glob("*_jobs.csv"))
    if not job_files:
        report_lines.append("No per-job files found — run benchmark first.")
        return "\n".join(report_lines)

    # Load all jobs
    try:
        # Use the most recent scenario's jobs as example, but aggregate all
        dfs = []
        for jf in job_files:
            try:
                df = pd.read_csv(jf)
                dfs.append(df)
            except Exception:
                pass
        if not dfs:
            report_lines.append("Could not load job data.")
            return "\n".join(report_lines)
        all_jobs = pd.concat(dfs, ignore_index=True)
        report_lines.append(f"Total per-job records: {len(all_jobs)}")
        report_lines.append("")

        # Starvation: max waiting vs median
        report_lines.append("## Starvation")
        for sched in sorted(all_jobs["scheduler"].unique()):
            sub = all_jobs[all_jobs["scheduler"] == sched]
            waits = sub["waiting_time"].dropna()
            if len(waits) == 0:
                continue
            report_lines.append(
                f"- **{sched}**: median={waits.median():.1f}s, P95={np.percentile(waits,95):.1f}s, max={waits.max():.1f}s, "
                f"max/median={waits.max()/max(waits.median(),1):.1f}x"
            )
            # Check if max > 3* P95 => severe starvation
            if waits.max() > 3 * np.percentile(waits, 95):
                report_lines.append(f"  - WARN: {sched} shows severe starvation (max >> P95)")
        report_lines.append("")

        # Large-job starvation
        report_lines.append("## Large-job Starvation (GPU count)")
        for sched in sorted(all_jobs["scheduler"].unique()):
            sub = all_jobs[all_jobs["scheduler"] == sched]
            # Completed vs total? Use status
            completed = sub[sub["status"] == "completed"]
            if len(completed) == 0:
                continue
            # Compare avg GPU requirement of completed jobs
            avg_gpu = completed["gpu_requirement"].mean()
            # Check distribution: % of 4-8 GPU jobs completed
            large = completed[completed["gpu_requirement"] >= 4]
            large_pct = len(large) / max(len(completed), 1) * 100
            report_lines.append(f"- **{sched}**: avg GPU={avg_gpu:.2f}, large (>=4 GPUs)={large_pct:.1f}% of completed")
            # Compare PPO vs baselines: if PPO has significantly lower large_pct, it avoids large jobs
        # Quick check for PPO avoidance
        if "PPO" in all_jobs["scheduler"].unique():
            ppo_large_pct = len(all_jobs[(all_jobs["scheduler"] == "PPO") & (all_jobs["status"] == "completed") & (all_jobs["gpu_requirement"] >= 4)]) / max(len(all_jobs[(all_jobs["scheduler"] == "PPO") & (all_jobs["status"] == "completed")]),1) *100
            sjf_large_pct = None
            if "SJF" in all_jobs["scheduler"].unique():
                sjf_large_pct = len(all_jobs[(all_jobs["scheduler"] == "SJF") & (all_jobs["status"] == "completed") & (all_jobs["gpu_requirement"] >= 4)]) / max(len(all_jobs[(all_jobs["scheduler"] == "SJF") & (all_jobs["status"] == "completed")]),1) *100
                if sjf_large_pct and ppo_large_pct < sjf_large_pct - 5:
                    report_lines.append(f"  - PPO large-job rate {ppo_large_pct:.1f}% < SJF {sjf_large_pct:.1f}% -- possible large-job avoidance")
        report_lines.append("")

        # Fragmentation: via utilization
        report_lines.append("## Fragmentation / Utilization")
        # Read agg CSVs for utilization
        agg_files = list(benchmark_dir.glob("*_agg.csv"))
        for af in agg_files:
            try:
                df = pd.read_csv(af)
                scen = af.stem.replace("_agg", "")
                report_lines.append(f"**{scen}**:")
                for _, row in df.iterrows():
                    report_lines.append(f"  - {row['scheduler']}: util={row.get('gpu_utilization_mean', row.get('gpu_utilization', 0)):.3f} ± {row.get('gpu_utilization_std',0):.3f}")
            except Exception:
                pass
        report_lines.append("")

        # NOOP abuse
        report_lines.append("## NOOP Abuse")
        ppo_logs = list(benchmark_dir.glob("*_ppo_decisions.csv"))
        if ppo_logs:
            for lf in ppo_logs:
                try:
                    df = pd.read_csv(lf)
                    total = len(df)
                    noop = len(df[df["action"] == df["action"].max()]) if "action" in df.columns else 0
                    # PPO NOOP is max_jobs (16), but logs have action column
                    # Use free_gpus and queue_length to check unnecessary NOOPs
                    # For now just report noop rate
                    if total > 0:
                        rate = noop / total * 100 if total else 0
                        report_lines.append(f"- **{lf.stem}**: steps={total}, NOOP={noop} ({rate:.1f}%)")
                        if rate > 50:
                            report_lines.append("  - WARN: High NOOP rate -- possible NOOP abuse")
                except Exception as e:
                    report_lines.append(f"- Could not analyze {lf.name}: {e}")
        else:
            report_lines.append("No PPO decision logs found — PPO NOOP analysis skipped (run with --ppo-model)")
        report_lines.append("")

        # Priority behavior
        report_lines.append("## Priority Behavior")
        for sched in sorted(all_jobs["scheduler"].unique()):
            sub = all_jobs[all_jobs["scheduler"] == sched]
            if "priority" not in sub.columns or "waiting_time" not in sub.columns:
                continue
            # Avg waiting per priority
            pri_groups = sub.groupby("priority")["waiting_time"].mean()
            report_lines.append(f"- **{sched}**:")
            for pri, avg_w in pri_groups.items():
                report_lines.append(f"  - P{int(pri)}: avg wait {avg_w:.1f}s (n={len(sub[sub['priority']==pri])})")
            # Check if high priority has lower wait than low (expected for Priority scheduler)
            if 1 in pri_groups and 5 in pri_groups:
                if pri_groups[5] < pri_groups[1]:
                    report_lines.append(f"  - {sched} respects priority (P5 faster than P1)")
                else:
                    report_lines.append(f"  - {sched} does not prioritize P5 over P1")
        report_lines.append("")

        report_lines.append("---")
        report_lines.append("*Generated from per-job and decision logs; see `artifacts/benchmarks/` for raw data.*")

    except Exception as e:
        report_lines.append(f"Analysis failed: {e}")

    report = "\n".join(report_lines)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(report, encoding="utf-8")
    return report
