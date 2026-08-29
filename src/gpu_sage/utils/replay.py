"""Lightweight replay helper for showcase GIF/video prep.

Instead of recording full training, this renders evaluation replay frames
from PPO decision logs. Each frame is an ASCII snapshot suitable for
terminal capture or conversion to GIF via an external tool.

Example frame:
    GPU 0  ████████████████░░░░
    GPU 1  ██████░░░░░░░░░░░░░░
    Queue: [J42] [J51] [J53]
    PPO Decision: -> J51
    Utilization: 87.4%  Completed: 37  Time: 1842s
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_ascii_frame(
    gpu_states: List[str],
    queue: List[str],
    decision: str | None,
    utilization: float,
    completed: int,
    sim_time: float,
    width: int = 20,
) -> str:
    lines = []
    for i, s in enumerate(gpu_states):
        # s is like "busy" or "idle" or GPU util bar
        bar = "█" * int(utilization * width) + "░" * (width - int(utilization * width))
        lines.append(f"GPU {i}  {bar}")
    lines.append("")
    lines.append("Queue:")
    if queue:
        lines.append(" ".join(f"[{j}]" for j in queue[:8]))
    else:
        lines.append("(empty)")
    lines.append("")
    lines.append(f"PPO Decision: -> {decision}" if decision else "PPO Decision: NOOP")
    lines.append("")
    lines.append(f"Utilization: {utilization*100:.1f}%")
    lines.append(f"Completed:   {completed}")
    lines.append(f"Simulation:  {sim_time:.0f}s")
    return "\n".join(lines)


def render_frames_from_logs(logs: List[Dict], out_dir: Path, max_frames: int = 20):
    """Render first `max_frames` decision logs as ASCII text files for GIF prep."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(logs[:max_frames]):
        txt = render_ascii_frame(
            gpu_states=[f"GPU {j}: {'busy' if j % 2 == 0 else 'idle'}" for j in range(4)],
            queue=[f"J{entry.get('selected_job_id', '?')}"] if entry.get("selected_job_id") else [],
            decision=f"J{entry.get('selected_job_id')}" if entry.get("selected_job_id") else None,
            utilization=float(entry.get("gpu_utilization", 0)),
            completed=int(entry.get("completed_jobs", 0)),
            sim_time=float(entry.get("simulation_time", 0)),
        )
        (out_dir / f"frame_{i:03d}.txt").write_text(txt)
    return out_dir


def plot_decision_timeline(logs: List[Dict], out_path: Path):
    """Generate a compact utilization/queue timeline plot from decision logs."""
    if not logs:
        return
    import pandas as pd

    df = pd.DataFrame(logs)
    plt.figure(figsize=(10, 5))
    if "gpu_utilization" in df.columns:
        plt.plot(df["simulation_time"], df["gpu_utilization"], label="Utilization", linewidth=2)
    if "queue_length" in df.columns:
        # Normalize queue for display
        q = df["queue_length"] / max(df["queue_length"].max(), 1)
        plt.plot(df["simulation_time"], q, label="Queue (norm)", alpha=0.6)
    plt.title("PPO Evaluation Replay — Utilization & Queue over Simulation Time")
    plt.xlabel("Simulation Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
