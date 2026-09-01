"""GPU-Sage Interactive Dashboard — Reinforcement Learning for GPU Cluster Scheduling.

Launch with:
    streamlit run app.py

or from the project root:
    cd gpu-sage && streamlit run app.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

# ---------------------------------------------------------------------------
# Project imports (reuse existing modules — no recreation)
# ---------------------------------------------------------------------------
from gpu_sage.evaluation.benchmark import run_benchmark_detailed, aggregate_results, per_seed_dataframe, run_single_seed_detailed_with_logs, DEFAULT_SCHEDULERS
from gpu_sage.evaluation.metrics import compute_metrics
from gpu_sage.workloads.generator import SCENARIO_NAMES, generate_workload, get_scenario_config, WorkloadConfig
from gpu_sage.core.cluster import Cluster
from gpu_sage.core.simulator import Simulator
from gpu_sage.core.topology import Topology
from gpu_sage.schedulers.fcfs import FCFSScheduler
from gpu_sage.schedulers.heuristics import BestFitScheduler, PriorityScheduler, SJFScheduler, TopologyBestFitScheduler
from gpu_sage.env.gpu_env import GPUSchedulingEnv
from sb3_contrib import MaskablePPO

# Allow duplicate OpenMP libs on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------------------------------------------------------------------------
# Page config & layout constants
# ---------------------------------------------------------------------------
PAGE_TITLE = "GPU-Sage — RL for GPU Cluster Scheduling"
PAGE_ICON = "🖥️"
LAYOUT = "wide"

# ---------------------------------------------------------------------------
# Helper: load once @ session state (avoid re-computation)
# ---------------------------------------------------------------------------

def _init_session():
    """Initialize session state with pre-loaded data."""
    if "df_agg" not in st.session_state:
        # Load all benchmark aggregation CSVs
        agg_dir = Path("artifacts/benchmarks")
        frames = []
        for csv_file in agg_dir.glob("*_agg.csv"):
            df = pd.read_csv(csv_file)
            df["scenario"] = csv_file.stem.replace("_agg", "")
            frames.append(df)
        if frames:
            st.session_state.df_agg = pd.concat(frames, ignore_index=True)
        else:
            st.session_state.df_agg = pd.DataFrame()

    if "models_loaded" not in st.session_state:
        # Load PPO models — cache under session
        models_dir = Path("artifacts/ppo")
        model_paths = []
        if models_dir.exists():
            # Primary model
            primary = models_dir / "final_model.zip"
            if primary.exists():
                model_paths.append(("PPO (primary)", primary))
            # Checkpoints
            for cp in sorted(models_dir.glob("checkpoints/ppo_*.zip")):
                model_paths.append((f"Checkpoint: {cp.stem}", cp))
            # Best model
            best = models_dir / "best_model.zip"
            if best.exists() and best not in [Path(p) for _, p in model_paths]:
                model_paths.append(("Best model", best))
        st.session_state.models_info = model_paths

    if "seed_selector" not in st.session_state:
        st.session_state.seed_selector = 0

    if "scheduler_names" not in st.session_state:
        st.session_state.scheduler_names = [
            "PPO",
            "FCFS",
            "SJF",
            "Priority",
            "BestFit",
        ]

    if "selected_scheduler" not in st.session_state:
        st.session_state.selected_scheduler = "PPO"

    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = "balanced"

    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = None

    if "ppo_model" not in st.session_state:
        st.session_state.ppo_model = None

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = None  # dict with live state

    if "live_state" not in st.session_state:
        # ONE authoritative live state — single source of truth for ALL dashboard elements
        st.session_state.live_state = {
            "simulation_time": 0.0,
            "gpu_count": 8,
            "waiting_jobs": [],
            "running_jobs": [],
            "completed_jobs": [],
            "selected_job": None,
            "selected_action": None,
            "reward_components": {
                "throughput_reward": 0.0,
                "waiting_penalty": 0.0,
                "utilization_reward": 0.0,
                "fragmentation_penalty": 0.0,
                "idle_penalty": 0.0,
            },
            "current_utilization": 0.0,
            "configured_gpu_count": 8,
        }


_init_session()

# ---------------------------------------------------------------------------
# Load PPO model lazily
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading PPO model...")
def load_ppo_model(path: Path) -> MaskablePPO | None:
    try:
        return MaskablePPO.load(str(path))
    except Exception as e:
        st.warning(f"Could not load PPO model: {e}")
        return None

# ---------------------------------------------------------------------------
# Core: run benchmark for given scenario + schedulers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Running benchmark evaluation...")
def run_benchmark_cached(
    scenario: str,
    schedulers: list[str],
    num_jobs: int,
    num_gpus: int,
    seeds: list[int],
) -> dict[int, dict[str, object]]:
    """Run benchmark using existing engine; cache by scenario+schedulers."""
    metrics, _per_job = run_benchmark_detailed(
        scenario=scenario,
        seeds=seeds,
        schedulers=schedulers,
        num_jobs=num_jobs,
        num_gpus=num_gpus,
    )
    return metrics

# ---------------------------------------------------------------------------
# Render ASCII frame (reuse existing utility)
# ---------------------------------------------------------------------------

def render_ascii_frame(
    gpu_states: list[str],
    queue: list[str],
    decision: str | None,
    utilization: float,
    completed: int,
    sim_time: float,
    width: int = 20,
) -> str:
    """Render a single ASCII frame from the existing replay utility."""
    from gpu_sage.utils.replay import render_ascii_frame as _render

    return _render(gpu_states, queue, decision, utilization, completed, sim_time, width)

# ---------------------------------------------------------------------------
# Plot topology graph
# ---------------------------------------------------------------------------

def plot_topology(cluster: Cluster, highlight_gpus: list[int] | None = None):
    """Plotly topology graph: nodes = GPUs colored by type, edges by link type."""
    n = cluster.total_gpus
    topo = cluster.topology
    if topo is None:
        # Homogeneous: show simple row of GPUs
        fig = go.Figure()
        for i, g in enumerate(cluster.gpus):
            color = "#636EFA" if g.is_free else "#EF553B"
            fig.add_trace(go.Scatter(x=[i], y=[0], mode="markers+text", marker=dict(size=30, color=color), text=f"GPU{i}<br>{g.gpu_type}", textposition="top center", name=f"GPU{i}"))
        fig.update_layout(title="Cluster (homogeneous — no topology penalty)", showlegend=False, height=250, yaxis=dict(visible=False), xaxis=dict(title="GPU ID"))
        return fig
    # Heterogeneous: two-group layout
    pos = {}
    for i in range(n):
        group = i // 4
        idx = i % 4
        pos[i] = (idx, 1 - group)  # group A y=1, group B y=0
    fig = go.Figure()
    # edges
    for i in range(n):
        for j in range(i + 1, n):
            lt = topo.get_link_type(i, j)
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            color = "#00CC96" if lt == "NVLINK" else "#AB63FA"
            width = 3 if lt == "NVLINK" else 1
            dash = "solid" if lt == "NVLINK" else "dot"
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=color, width=width, dash=dash), showlegend=False, hoverinfo="text", text=f"{i}-{j}: {lt}"))
    # nodes
    type_colors = {"A100_80GB": "#636EFA", "A100_40GB": "#19D3F3", "V100_32GB": "#FF6692", "V100": "#FF6692", "T4_16GB": "#FFA15A", "T4": "#FFA15A", "A100": "#636EFA"}
    for i, g in enumerate(cluster.gpus):
        x, y = pos[i]
        base = type_colors.get(g.gpu_type, "#636EFA")
        # highlight if in selected set
        is_highlight = highlight_gpus is not None and i in highlight_gpus
        size = 40 if is_highlight else 30
        line = dict(color="gold", width=3) if is_highlight else dict(color="white", width=1)
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", marker=dict(size=size, color=base, line=line), text=f"GPU{i}<br>{g.gpu_type}<br>{g.memory_gb:.0f}GB", textposition="top center", name=f"GPU{i}"))
    fig.update_layout(title="Cluster Topology (NVLink solid, PCIe dotted; gold border = selected GPUs)", height=350, showlegend=False, yaxis=dict(visible=False), xaxis=dict(title="Group position"))
    return fig

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LIVE STATE MANAGEMENT
# ---------------------------------------------------------------------------

def _build_live_state_from_simulator(
    sim: Simulator | None = None,
    *,
    num_gpus: int = 8,
    scheduler_name: str | None = None,
    seed: int | None = None,
    metrics_obj: object | None = None,
    per_job_records: list[dict] | None = None,
) -> dict:
    """Build the one authoritative LiveState.

    Can be called two ways:
    1. _build_live_state_from_simulator(sim, num_gpus=8) — from a Simulator instance.
    2. _build_live_state_from_simulator(num_gpus=8, scheduler_name="PPO", seed=0, metrics_obj=metrics)
       — from benchmark Metrics + scheduler info (no Simulator needed).
    """
    # --- Path A: from a Simulator instance ---
    if sim is not None:
        free_gpus = sim.cluster.free_gpu_count
        total_gpus = sim.cluster.total_gpus
        utilization = sim.cluster.utilization(sim.running_jobs) if sim.cluster.gpus else 0.0

        # Gather job lists from simulator store
        all_jobs = sim._job_store

        waiting = [j for j in all_jobs.values() if j.status.name == "WAITING"]
        running = [j for j in all_jobs.values() if j.status.name == "RUNNING"]
        completed = [j for j in all_jobs.values() if j.status.name == "COMPLETED"]

        # selected_job / selected_action come from current PPO decision if PPO scheduler
        selected_job = None
        selected_action = None
        reward_components = {
            "throughput_reward": 0.0,
            "waiting_penalty": 0.0,
            "utilization_reward": 0.0,
            "fragmentation_penalty": 0.0,
            "idle_penalty": 0.0,
        }
        current_utilization = utilization

        if scheduler_name is not None and st.session_state.selected_scheduler == "PPO":
            # Try to get the latest PPO decision from decision logs
            ppo_decisions_csv = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_ppo_decisions.csv")
            if ppo_decisions_csv.exists():
                import pandas as pd
                ppo_df = pd.read_csv(ppo_decisions_csv)
                if not ppo_df.empty:
                    latest = ppo_df.iloc[-1]
                    sjid = int(latest.get("selected_job_id", 0))
                    if sjid > 0 and sjid in all_jobs:
                        selected_job = all_jobs[sjid]
                        selected_action = sjid
                        reward_components = {
                            "throughput_reward": float(latest.get("throughput_reward", 0)),
                            "waiting_penalty": float(latest.get("waiting_penalty", 0)),
                            "utilization_reward": float(latest.get("utilization_reward", 0)),
                            "fragmentation_penalty": float(latest.get("fragmentation_penalty", 0)),
                            "idle_penalty": float(latest.get("idle_penalty", 0)),
                        }
                    else:
                        selected_job = None
                        selected_action = None
                else:
                    selected_job = None
                    selected_action = None
            else:
                selected_job = None
                selected_action = None
        else:
            selected_job = None
            selected_action = None

        return {
            "simulation_time": float(sim.current_time),
            "gpu_count": num_gpus,
            "waiting_jobs": [j.job_id for j in waiting],
            "running_jobs": [j.job_id for j in running],
            "completed_jobs": [j.job_id for j in completed],
            "selected_job": selected_job,
            "selected_action": selected_action,
            "reward_components": reward_components,
            "current_utilization": current_utilization,
            "configured_gpu_count": total_gpus,
        }

    # --- Path B: from benchmark Metrics + scheduler info (no Simulator) ---
    # metrics_obj is the Metrics dataclass from benchmark results
    if metrics_obj is not None and scheduler_name is not None and seed is not None:
        import pandas as pd

        # Build LiveState from the Metrics object and decision logs
        # We extract key fields from metrics_obj and the decision log CSV
        current_utilization = float(getattr(metrics_obj, "gpu_utilization", 0.0))

        # Try to get live PPO decision info from the decision log
        ppo_decisions_csv = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_ppo_decisions.csv")
        selected_job = None
        selected_action = None
        reward_components = {
            "throughput_reward": 0.0,
            "waiting_penalty": 0.0,
            "utilization_reward": 0.0,
            "fragmentation_penalty": 0.0,
            "idle_penalty": 0.0,
        }
        sim_time = 0.0
        waiting_jobs: list[int] = []
        running_jobs: list[int] = []
        completed_jobs_list: list[int] = []

        if scheduler_name == "PPO" and ppo_decisions_csv.exists():
            ppo_df = pd.read_csv(ppo_decisions_csv)
            if not ppo_df.empty:
                latest = ppo_df.iloc[-1]
                sim_time = float(latest.get("simulation_time", 0.0))
                sjid = int(latest.get("selected_job_id", 0))
                if sjid > 0:
                    from types import SimpleNamespace
                    selected_job = SimpleNamespace(
                        job_id=sjid,
                        gpu_count=int(latest.get("gpu_count", 1)),
                        priority=int(latest.get("priority", 1)),
                    )
                    selected_action = sjid
                    # Get reward components from the same decision
                    reward_components = {
                        "throughput_reward": float(latest.get("throughput_reward", 0.0)),
                        "waiting_penalty": float(latest.get("waiting_penalty", 0.0)),
                        "utilization_reward": float(latest.get("utilization_reward", 0.0)),
                        "fragmentation_penalty": float(latest.get("fragmentation_penalty", 0.0)),
                        "idle_penalty": float(latest.get("idle_penalty", 0.0)),
                    }
                    # Build job lists from the decision log
                    # (we don't have real job IDs from the log, so use placeholder
                    # lists sized to the logged counts purely for len() display)
                    running_count = int(latest.get("running_jobs", 0))
                    completed_from_log = int(latest.get("completed_jobs", 0))
                    running_jobs = list(range(running_count))
                    completed_jobs_list = list(range(completed_from_log))
                else:
                    selected_job = None
                    selected_action = None
            else:
                selected_job = None
                selected_action = None
                sim_time = 0.0

        # If no PPO decision log, use metrics-based initial state
        if selected_job is None:
            # Build real waiting/running/completed job-id lists from the
            # per-job records produced by this run, instead of a hardcoded
            # placeholder that always showed "Completed Jobs: 1".
            if per_job_records:
                completed_jobs_list = [r["job_id"] for r in per_job_records if r.get("status") == "completed"]
                running_jobs = [r["job_id"] for r in per_job_records if r.get("status") == "running"]
                waiting_jobs = [r["job_id"] for r in per_job_records if r.get("status") not in ("completed", "running")]
                sim_time = max((r.get("completion_time", 0.0) for r in per_job_records), default=0.0)
            else:
                sim_time = 0.0
                completed_jobs_list = []

        return {
            "simulation_time": sim_time,
            "gpu_count": num_gpus if num_gpus else 8,
            "waiting_jobs": waiting_jobs,
            "running_jobs": running_jobs,
            "completed_jobs": completed_jobs_list,
            "selected_job": selected_job,
            "selected_action": selected_action,
            "reward_components": reward_components,
            "current_utilization": current_utilization,
            "configured_gpu_count": num_gpus if num_gpus else 8,
        }

    # --- Fallback: initial state ---
    return {
        "simulation_time": 0.0,
        "gpu_count": num_gpus if num_gpus else 8,
        "waiting_jobs": [],
        "running_jobs": [],
        "completed_jobs": [],
        "selected_job": None,
        "selected_action": None,
        "reward_components": {
            "throughput_reward": 0.0,
            "waiting_penalty": 0.0,
            "utilization_reward": 0.0,
            "fragmentation_penalty": 0.0,
            "idle_penalty": 0.0,
        },
        "current_utilization": 0.0,
        "configured_gpu_count": num_gpus if num_gpus else 8,
    }


def _initialize_live_state_from_scratch(scenario: str, num_gpus: int, scheduler_name: str) -> dict:
    """Create initial LiveState right after Reset / initial launch.

    This ensures: simulation_time=0, completed_jobs=0, queue populated from
    initial workload, no historical values survive.
    """
    from gpu_sage.workloads.generator import generate_workload

    cfg = get_scenario_config(scenario)
    base_jobs = generate_workload(
        scenario=scenario, seed=st.session_state.seed_selector, count=16, config=cfg
    )

    # Create a fresh simulator with the scheduler
    from gpu_sage.schedulers.base import Scheduler
    from gpu_sage.schedulers.fcfs import FCFSScheduler
    from gpu_sage.schedulers.rl import RLScheduler

    if scheduler_name == "PPO":
        scheduler = RLScheduler()
    else:
        from gpu_sage.schedulers.heuristics import BestFitScheduler, PriorityScheduler, SJFScheduler
        scheduler_map = {
            "FCFS": FCFSScheduler(),
            "SJF": SJFScheduler(),
            "Priority": PriorityScheduler(),
            "BestFit": BestFitScheduler(),
        }
        scheduler = scheduler_map.get(scheduler_name, FCFSScheduler())

    cluster = Cluster.homogeneous(num_gpus, 80, "A100")
    sim = Simulator(cluster, scheduler)
    sim.load_jobs(base_jobs)
    # Run until first decision (this is what Start does internally, but we want initial state)
    # For initial state, just capture t=0
    # Actually, let's just create the LiveState directly without running the sim
    # The initial state should have: t=0, no completed, queue = all waiting

    return {
        "simulation_time": 0.0,
        "gpu_count": num_gpus,
        "waiting_jobs": [j.job_id for j in sim.waiting_jobs.values()],
        "running_jobs": [j.job_id for j in sim.running_jobs.values()],
        "completed_jobs": [],  # 0 at t=0
        "selected_job": None,
        "selected_action": None,
        "reward_components": {
            "throughput_reward": 0.0,
            "waiting_penalty": 0.0,
            "utilization_reward": 0.0,
            "fragmentation_penalty": 0.0,
            "idle_penalty": 0.0,
        },
        "current_utilization": 0.0,
        "configured_gpu_count": num_gpus,
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Page: Dashboard — Home / Live Simulation
# ---------------------------------------------------------------------------

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

st.title(f"{PAGE_ICON}  {PAGE_TITLE}")
st.caption("Reinforcement Learning for Adaptive GPU Cluster Scheduling")

# --- Mode indicator ---
if st.session_state.sim_state is not None and st.session_state.get("live_state", {}).get("simulation_time", 0) > 0:
    st.success("MODE: LIVE SIMULATION")
else:
    st.info("MODE: EXPERIMENT RESULTS (charts below)")

# --- Header ---
with st.container():
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown("### GPU-Sage")
    with c2:
        chosen_scenario = st.selectbox(
            "Workload scenario",
            options=SCENARIO_NAMES,
            index=SCENARIO_NAMES.index(st.session_state.selected_scenario)
            if st.session_state.selected_scenario in SCENARIO_NAMES
            else 0,
            key="scenario_select",
        )
        st.session_state.selected_scenario = chosen_scenario
    with c3:
        pass


    # Scheduler selector
    chosen_scheduler = st.selectbox(
        "Scheduler",
        options=st.session_state.scheduler_names,
        index=(
            st.session_state.scheduler_names.index(
                st.session_state.selected_scheduler
            )
            if st.session_state.selected_scheduler in st.session_state.scheduler_names
            else 0
        ),
        key="scheduler_select",
    )
    st.session_state.selected_scheduler = chosen_scheduler

    # PPO model selector (hom vs hetero vs graph) — only when PPO chosen
    scheduler = st.session_state.selected_scheduler
    if scheduler == "PPO":
        if "ppo_model_choice" not in st.session_state:
            st.session_state.ppo_model_choice = "Homogeneous-trained (8xA100)"
        st.session_state.ppo_model_choice = st.selectbox(
            "PPO model",
            options=["Homogeneous-trained (8xA100)", "Heterogeneous-trained (8-hetero)", "Graph-trained (topology-aware)"],
            index=0 if st.session_state.get("ppo_model_choice") == "Homogeneous-trained (8xA100)" else 1 if st.session_state.get("ppo_model_choice") == "Heterogeneous-trained (8-hetero)" else 2,
            key="ppo_model_select",
        )

    # GPU count
    num_gpus = st.slider("GPUs", min_value=4, max_value=16, value=8, key="gpu_slider")

# --- Main area: two columns ---
left_col, right_col = st.columns([2, 1])

# --- LEFT: Live simulation ---
with left_col:
    st.subheader("Live Simulation")

    # Control buttons
    col_btn_start, col_btn_pause, col_btn_step, col_btn_reset = st.columns(4)

    with col_btn_start:
        if st.button("Start", type="primary", use_container_width=True):
            # Generate workload once and start sim
            cfg = get_scenario_config(st.session_state.selected_scenario)
            base_jobs = generate_workload(
                scenario="custom", seed=st.session_state.seed_selector, count=16, config=cfg
            )
            # Resolve PPO model path if PPO selected
            ppo_path = None
            if st.session_state.selected_scheduler == "PPO":
                choice = st.session_state.get("ppo_model_choice", "Homogeneous-trained (8xA100)")

                def _latest_model(glob_pattern: str) -> Path | None:
                    """Find the most recently trained final_model.zip matching a glob."""
                    matches = sorted(
                        Path(".").glob(glob_pattern),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    return matches[0] if matches else None

                if "Heterogeneous" in choice:
                    cand = _latest_model("artifacts/ppo_hetero*/runs/*/model/final_model.zip")
                    ppo_path = str(cand) if cand else None
                    if ppo_path is None:
                        st.warning(
                            "No heterogeneous-trained PPO model found under artifacts/ppo_hetero*/. "
                            "Falling back to heuristic scheduler. Train one with training/train_ppo.py "
                            "using a heterogeneous cluster config first."
                        )
                elif "Graph" in choice:
                    cand = _latest_model("artifacts/ppo_graph*/runs/*/model/final_model.zip")
                    ppo_path = str(cand) if cand else None
                    if ppo_path is None:
                        st.warning(
                            "No graph/topology-trained PPO model found under artifacts/ppo_graph*/. "
                            "Falling back to heuristic scheduler."
                        )
                else:
                    cand = _latest_model("artifacts/ppo/runs/*/model/final_model.zip")
                    ppo_path = str(cand) if cand else None
                    if ppo_path is None:
                        st.warning(
                            "No homogeneous-trained PPO model found under artifacts/ppo/. "
                            "Falling back to heuristic scheduler."
                        )
            # Auto-advance the seed on repeated Start clicks with unchanged
            # settings so results are visibly different each time, instead of
            # silently reproducing an identical run (workload gen is
            # deterministic per scenario+seed).
            run_key = (st.session_state.selected_scenario, st.session_state.selected_scheduler)
            if st.session_state.get("_last_run_key") == run_key:
                st.session_state.seed_selector = (st.session_state.seed_selector + 1) % 1000
            st.session_state["_last_run_key"] = run_key

            from gpu_sage.evaluation.benchmark import run_single_seed_detailed_with_logs, save_benchmark

            # Race ALL schedulers on the exact same generated job stream/seed —
            # this is the actual comparison, not 5 independent overwritten runs.
            st.session_state.simulation_complete = False
            all_schedulers = DEFAULT_SCHEDULERS + ["PPO"]
            metrics_map, per_job_map, ppo_logs = run_single_seed_detailed_with_logs(
                scenario=st.session_state.selected_scenario,
                seed=st.session_state.seed_selector,
                schedulers=all_schedulers,
                num_jobs=16,
                num_gpus=num_gpus,
                ppo_model_path=ppo_path,
            )
            seed = st.session_state.seed_selector
            results = {seed: metrics_map}
            scen = st.session_state.selected_scenario

            # All schedulers ran together this click, so this write is now a
            # complete, correct set of rows for the scenario — no merge needed.
            save_benchmark(
                scenario=scen,
                results=results,
                out_dir="artifacts/benchmarks",
                num_gpus=num_gpus,
                num_jobs=16,
                per_job_results={seed: per_job_map},
                ppo_logs={seed: ppo_logs},
            )

            st.session_state.benchmark_results = results

            # --- Animate the selected scheduler's run: replay jobs arriving,
            # starting, and completing over simulated time using the real
            # per-job timestamps from the run that just happened. ---
            selected_records = per_job_map.get(st.session_state.selected_scheduler, [])
            anim_placeholder = st.empty()
            if selected_records:
                events = []
                for r in selected_records:
                    events.append((r["arrival_time"], "arrive", r["job_id"]))
                    events.append((r["start_time"], "start", r["job_id"]))
                    events.append((r["completion_time"], "complete", r["job_id"]))
                events.sort(key=lambda e: e[0])

                waiting_set, running_set, completed_set = set(), set(), set()
                sleep_per_event = max(0.03, min(0.15, 3.0 / max(len(events), 1)))
                for t, kind, jid in events:
                    if kind == "arrive":
                        waiting_set.add(jid)
                    elif kind == "start":
                        waiting_set.discard(jid)
                        running_set.add(jid)
                    elif kind == "complete":
                        running_set.discard(jid)
                        completed_set.add(jid)
                    with anim_placeholder.container():
                        st.caption(
                            f"**Simulating {st.session_state.selected_scheduler}…**  "
                            f"t={t:.0f}s  |  Waiting: {len(waiting_set)}  |  "
                            f"Running: {len(running_set)}  |  Completed: {len(completed_set)}"
                        )
                        st.progress(min(len(completed_set) / max(len(selected_records), 1), 1.0))
                    time.sleep(sleep_per_event)
            anim_placeholder.empty()
            # Reload aggregate benchmark data so the Comparison Charts panel
            # reflects the run that was just saved (was previously loaded
            # once at session start and never refreshed).
            agg_dir = Path("artifacts/benchmarks")
            frames = []
            for csv_file in agg_dir.glob("*_agg.csv"):
                fdf = pd.read_csv(csv_file)
                fdf["scenario"] = csv_file.stem.replace("_agg", "")
                frames.append(fdf)
            st.session_state.df_agg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            st.session_state.simulation_complete = True
            # Build the ONE authoritative LiveState from the simulator
            # We need to extract live state from the results
            seed = st.session_state.seed_selector
            scheduler_name = st.session_state.selected_scheduler
            if seed in results and scheduler_name in results[seed]:
                metrics_obj = results[seed][scheduler_name]
                # Build LiveState from the metrics + real per-job records for this run
                live = _build_live_state_from_simulator(
                    metrics_obj=metrics_obj,
                    scheduler_name=scheduler_name,
                    seed=seed,
                    num_gpus=num_gpus,
                    per_job_records=per_job_map.get(scheduler_name),
                )
            else:
                live = _initialize_live_state_from_scratch(st.session_state.selected_scenario, num_gpus, scheduler_name)
            st.session_state.sim_state = {
                "results": results,
                "scenario": st.session_state.selected_scenario,
                "scheduler": st.session_state.selected_scheduler,
                "num_gpus": num_gpus,
                "seed": st.session_state.seed_selector,
                "live_state": live,  # Store the authoritative live state
            }
            st.session_state.decision_step_index = 0
            st.rerun()

    with col_btn_pause:
        if st.button("Pause", use_container_width=True):
            st.session_state.sim_paused = True
            st.info("Simulation paused")

    with col_btn_step:
        if st.button("Step", use_container_width=True):
            if st.session_state.sim_state is not None:
                # Advance to the NEXT decision-log row instead of re-running
                # the whole episode and grabbing the last row every time.
                if "decision_step_index" not in st.session_state:
                    st.session_state.decision_step_index = 0
                seed = st.session_state.seed_selector
                scheduler_name = st.session_state.selected_scheduler
                ppo_decisions_csv = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_ppo_decisions.csv")
                if scheduler_name == "PPO" and ppo_decisions_csv.exists():
                    import pandas as pd
                    ppo_df = pd.read_csv(ppo_decisions_csv)
                    if not ppo_df.empty:
                        st.session_state.decision_step_index = min(
                            st.session_state.decision_step_index + 1, len(ppo_df) - 1
                        )
                        row = ppo_df.iloc[st.session_state.decision_step_index]
                        from types import SimpleNamespace
                        
                        def _safe_int(k, default=0):
                            v = row.get(k, default)
                            return int(v) if pd.notna(v) else default
                            
                        def _safe_float(k, default=0.0):
                            v = row.get(k, default)
                            return float(v) if pd.notna(v) else default
                            
                        sjid = _safe_int("selected_job_id", 0)
                        live = st.session_state.sim_state["live_state"]
                        live["simulation_time"] = _safe_float("simulation_time", live["simulation_time"])
                        live["current_utilization"] = _safe_float("gpu_utilization", live["current_utilization"])
                        live["completed_jobs"] = list(range(_safe_int("completed_jobs", 0)))
                        live["running_jobs"] = list(range(_safe_int("running_jobs", 0)))
                        live["selected_job"] = SimpleNamespace(
                            job_id=sjid,
                            gpu_count=_safe_int("gpu_count", 1),
                            priority=_safe_int("priority", 1),
                        ) if sjid > 0 else None
                        live["selected_action"] = sjid if sjid > 0 else None
                        st.session_state.sim_state["live_state"] = live
                st.rerun()
            else:
                st.warning("Start simulation first")

    with col_btn_reset:
        if st.button("Reset", use_container_width=True):
            # Clear ALL state - no historical values may survive
            st.session_state.sim_state = None
            st.session_state.benchmark_results = None
            st.session_state.decision_step_index = 0
            st.session_state.simulation_complete = False
            # Reset live state to initial
            st.session_state.live_state = {
                "simulation_time": 0.0,
                "gpu_count": 8,
                "waiting_jobs": [],
                "running_jobs": [],
                "completed_jobs": [],
                "selected_job": None,
                "selected_action": None,
                "reward_components": {
                    "throughput_reward": 0.0,
                    "waiting_penalty": 0.0,
                    "utilization_reward": 0.0,
                    "fragmentation_penalty": 0.0,
                    "idle_penalty": 0.0,
                },
                "current_utilization": 0.0,
                "configured_gpu_count": 8,
            }
            st.rerun()

    # --- Display simulation results using LiveState ---
    if st.session_state.sim_state is not None:
        results = st.session_state.sim_state["results"]
        seed = st.session_state.seed_selector
        scheduler_name = st.session_state.selected_scheduler
        live = st.session_state.sim_state.get("live_state", {})

        # Pull metrics for the selected scheduler
        if seed in results and scheduler_name in results[seed]:
            metrics = results[seed][scheduler_name]

            # Live metrics cards — ALL from LiveState
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("GPU Utilization", f"{live['current_utilization']:.1%}")
            with m2:
                st.metric("Completed Jobs", len(live["completed_jobs"]))
            with m3:
                st.metric("Avg Wait", f"{metrics.average_waiting_time:.1f}s")
            with m4:
                st.metric("P95 Wait", f"{metrics.p95_waiting_time:.1f}s")

            m5, m6, m7, m8 = st.columns(4)
            with m5:
                st.metric("Avg JCT", f"{metrics.average_turnaround_time:.1f}s")
            with m6:
                st.metric("P95 JCT", f"{metrics.p95_jct:.1f}s")
            with m7:
                st.metric("Throughput", f"{metrics.throughput:.3f}/s")
            with m8:
                st.metric("Fairness (Jain)", f"{metrics.jains_fairness_index:.2f}")

            # Queue display from LiveState
            st.markdown("#### Job Queue")
            waiting = live["waiting_jobs"]
            running = live["running_jobs"]
            completed = live["completed_jobs"]

            st.caption(f"**Queue:** {len(waiting)} waiting  |  **Running:** {len(running)}  |  **Completed:** {len(completed)}")

            # Per-job details from LiveState
            if scheduler_name == "PPO":
                ppo_decisions_csv = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_ppo_decisions.csv")
                if ppo_decisions_csv.exists():
                    import pandas as pd
                    ppo_df = pd.read_csv(ppo_decisions_csv)
                    if not ppo_df.empty:
                        step_idx = st.session_state.get("decision_step_index", -1)
                        latest = ppo_df.iloc[step_idx]
                        def _safe_int(k, default=0):
                            v = latest.get(k, default)
                            return int(v) if pd.notna(v) else default
                            
                        def _safe_float(k, default=0.0):
                            v = latest.get(k, default)
                            return float(v) if pd.notna(v) else default
                            
                        sjid = _safe_int("selected_job_id", 0)
                        running_jobs_count = _safe_int("running_jobs", 0)
                        completed_from_log = _safe_int("completed_jobs", 0)
                        sim_time_from_log = _safe_float("simulation_time", 0)
                        # Show jobs from the decision log
                        st.caption(f"**Simulation Time:** {sim_time_from_log:.0f}s")
                        st.caption(f"**Running Jobs:** {running_jobs_count}  **Completed From Log:** {completed_from_log}")
                    else:
                        st.caption("Simulation Time: 0s  |  Running: 0  |  Completed: 0")
                else:
                    st.caption("Simulation Time: 0s  |  Running: 0  |  Completed: 0")
            else:
                # Heuristic scheduler - show from LiveState
                sim_time_live = live["simulation_time"]
                st.caption(f"**Simulation Time:** {sim_time_live:.0f}s  |  Running: {len(running)}  |  Completed: {len(completed)}")

            # PPO decision panel (if PPO selected)
            if scheduler_name == "PPO":
                st.markdown("#### PPO Decision Panel")
                if live["selected_job"] is not None:
                    sjid = live["selected_job"].job_id if live["selected_job"] else 0
                    gr = live["selected_job"].gpu_count if live["selected_job"] else 1
                    pri = live["selected_job"].priority if live["selected_job"] else 1
                    wt = 0.0  # Will get from decision log
                    free_gpus = len(live["waiting_jobs"])  # approximate

                    # Get latest decision log info for display
                    ppo_decisions_csv = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_ppo_decisions.csv")
                    if ppo_decisions_csv.exists():
                        ppo_df = pd.read_csv(ppo_decisions_csv)
                        if not ppo_df.empty:
                            step_idx = st.session_state.get("decision_step_index", -1)
                            latest = ppo_df.iloc[step_idx]
                            def _si(k, default=0):
                                v = latest.get(k, default); return int(v) if pd.notna(v) else default
                            def _sf(k, default=0.0):
                                v = latest.get(k, default); return float(v) if pd.notna(v) else default
                            sjid = _si("selected_job_id", sjid)
                            gr = _si("gpu_count", gr)
                            pri = _si("priority", pri)
                            wt = _sf("waiting_time", 0.0)
                            free_gpus = _si("free_gpus", free_gpus)
                            action = _si("action", 0)
                            reward = _sf("reward", 0.0)
                            throughput_reward = _sf("throughput_reward", 0.0)
                            waiting_penalty = _sf("waiting_penalty", 0.0)
                            utilization_reward = _sf("utilization_reward", 0.0)
                            fragmentation_penalty = _sf("fragmentation_penalty", 0.0)
                            idle_penalty = _sf("idle_penalty", 0.0)
                        else:
                            sjid, gr, pri, wt = 0, 1, 1, 0.0
                            free_gpus = 0
                            reward = 0.0
                            throughput_reward = 0.0
                            waiting_penalty = 0.0
                            utilization_reward = 0.0
                            fragmentation_penalty = 0.0
                            idle_penalty = 0.0
                    else:
                        sjid, gr, pri, wt = 0, 1, 1, 0.0
                        free_gpus = 0
                        reward = 0.0
                        throughput_reward = 0.0
                        waiting_penalty = 0.0
                        utilization_reward = 0.0
                        fragmentation_penalty = 0.0
                        idle_penalty = 0.0

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.info(f"**Selected Job:**\n#{int(sjid)}")
                    with c2:
                        st.info(f"**GPU Requirement:**\n{int(gr)}")
                    with c3:
                        st.info(f"**Priority:**\n{int(pri)}")
                    with c4:
                        st.info(f"**Waiting Time:**\n{float(wt):.0f}s")

                    st.caption(f"**Available GPUs:** {int(free_gpus)}")
                    st.caption(f"**Action:** SELECT JOB {int(sjid)}")

                    # Reward components from the same decision transition
                    st.markdown("**Reward Components:**")
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.caption(f"Throughput: {float(throughput_reward):.2f}")
                    with r2:
                        st.caption(f"Waiting: {float(waiting_penalty):.2f}")
                    with r3:
                        st.caption(f"Utilization: {float(utilization_reward):.2f}")

                    r4, r5 = st.columns(2)
                    with r4:
                        st.caption(f"Fragmentation: {float(fragmentation_penalty):.2f}")
                    with r5:
                        st.caption(f"Idle: {float(idle_penalty):.2f}")

                    # Consistency check
                    if live["simulation_time"] == 0 and len(live["completed_jobs"]) > 0:
                        st.warning(
                            "State consistency: simulation_time=0 but completed_jobs>0. "
                            "Data may be mixed from different sources."
                        )
                else:
                    st.info("No PPO decision at current state")

            # ASCII snapshot — MUST derive from LiveState only
            st.markdown("#### ASCII Snapshot")
            util = live["current_utilization"]
            n_gpus = live["configured_gpu_count"]
            queue_jobs = []
            decision = "NOOP"
            sim_time = live["simulation_time"]
            completed_count = len(live["completed_jobs"])

            if scheduler_name == "PPO" and live["selected_job"] is not None:
                ppo_decisions_csv = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_ppo_decisions.csv")
                if ppo_decisions_csv.exists():
                    import pandas as pd
                    ppo_df = pd.read_csv(ppo_decisions_csv)
                    if not ppo_df.empty:
                        step_idx = st.session_state.get("decision_step_index", -1)
                        latest = ppo_df.iloc[step_idx]
                        sim_time = float(latest.get("simulation_time", sim_time))
                        decision = f"J{int(latest.get('selected_job_id', 0))}" if latest.get("selected_job_id") is not None else "NOOP"
                        running_from_log = int(latest.get("running_jobs", 0))
                        completed_from_log = int(latest.get("completed_jobs", 0))
                        # Use the log's completed count but note it's from the decision
                        completed_count = completed_from_log
                        # Build queue from log
                        queue_jobs = []  # will be populated below

            # Compute real per-GPU busy/free status from the jobs CSV instead of
            # smearing one cluster-wide average across every GPU identically.
            busy_gpu_ids: set[int] = set()
            jobs_csv_path = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_jobs.csv")
            if jobs_csv_path.exists():
                try:
                    import pandas as pd
                    jdf = pd.read_csv(jobs_csv_path)
                    active = jdf[
                        (jdf["start_time"] <= sim_time) & (jdf["completion_time"] > sim_time)
                    ]
                    for ag in active["assigned_gpus"].dropna():
                        busy_gpu_ids.update(int(x) for x in str(ag).split(",") if x.strip().isdigit())
                except Exception:
                    pass

            gpu_states = [
                f"GPU {i}: {'█' * 20 if i in busy_gpu_ids else '░' * 20}"
                for i in range(n_gpus)
            ]
            # Build queue from live state
            for jid in live["waiting_jobs"][:4]:
                queue_jobs.append(f"J{jid}")

            frame = render_ascii_frame(
                gpu_states=gpu_states,
                queue=queue_jobs,
                decision=decision,
                utilization=util,
                completed=completed_count,
                sim_time=sim_time,
                width=20,
            )
            st.code(frame, language=None, line_numbers=False)
        else:
            # EXPERIMENT mode: no live simulation
            st.code(
                render_ascii_frame(
                    gpu_states=[f"GPU {i}: ███████████████████░" for i in range(min(8, num_gpus))],
                    queue=[],
                    decision="NOOP",
                    utilization=0.0,
                    completed=0,
                    sim_time=0.0,
                    width=20,
                ),
                language=None,
                line_numbers=False,
            )

        # Cluster topology view - from current state
        st.markdown("#### Cluster Topology View")
        try:
            if st.session_state.selected_scenario in ("heterogeneous", "topology_sensitive", "mixed_ml"):
                topo_cluster = Cluster.heterogeneous(
                    [
                        {"gpu_type": "A100_80GB", "memory_gb": 80},
                        {"gpu_type": "A100_80GB", "memory_gb": 80},
                        {"gpu_type": "A100_40GB", "memory_gb": 40},
                        {"gpu_type": "A100_40GB", "memory_gb": 40},
                        {"gpu_type": "V100_32GB", "memory_gb": 32},
                        {"gpu_type": "V100_32GB", "memory_gb": 32},
                        {"gpu_type": "T4_16GB", "memory_gb": 16},
                        {"gpu_type": "T4_16GB", "memory_gb": 16},
                    ]
                )
                topo_cluster.topology = Topology.two_group(8, 4)
            else:
                topo_cluster = Cluster.homogeneous(num_gpus, 80, "A100")
            # Highlight GPUs for current state
            highlight = None
            if scheduler_name == "PPO" and live["selected_job"] is not None:
                try:
                    jobs_csv2 = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_jobs.csv")
                    if jobs_csv2.exists():
                        import pandas as pd
                        jdf2 = pd.read_csv(jobs_csv2)
                        sel = live["selected_job"].job_id if live["selected_job"] else 0
                        row = jdf2[jdf2["job_id"] == sel]
                        if not row.empty and "assigned_gpus" in row.columns:
                            ag = str(row.iloc[0]["assigned_gpus"])
                            if ag:
                                highlight = [int(x) for x in ag.split(",") if x.strip().isdigit()]
                except Exception:
                    pass
            fig_topo = plot_topology(topo_cluster, highlight_gpus=highlight)
            st.plotly_chart(fig_topo, use_container_width=True)
            # Placement explanation from current state
            if scheduler_name == "PPO" and live["selected_job"] is not None:
                try:
                    jobs_all = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_jobs.csv")
                    if jobs_all.exists():
                        import pandas as pd
                        jdf_all = pd.read_csv(jobs_all)
                        sel_id = live["selected_job"].job_id if live["selected_job"] else 0
                        jrow = jdf_all[jdf_all["job_id"] == sel_id]
                        if not jrow.empty:
                            r = jrow.iloc[0]

                            def _fmt(val, default="N/A", precision=1):
                                if pd.isna(val) or val is None or val == "?" or str(val).strip() == "":
                                    return default
                                try:
                                    fval = float(val)
                                    if precision == 0:
                                        return str(int(round(fval)))
                                    return f"{fval:.{precision}f}"
                                except (ValueError, TypeError):
                                    return str(val)

                            gpu_req = _fmt(r.get("gpu_requirement"), precision=0)
                            mem_req = _fmt(r.get("memory_requirement"), precision=1)
                            topo_sens = _fmt(r.get("topology_sensitive"))
                            assigned_gpus = _fmt(r.get("assigned_gpus"))
                            place_pen = _fmt(r.get("placement_penalty"), precision=2)
                            comm_cost = _fmt(r.get("communication_cost"), precision=2)

                            mem_str = f"{mem_req}GB" if mem_req != "N/A" else "N/A"
                            st.caption(f"Selected Job: #{sel_id} | GPUs={gpu_req} | Mem={mem_str} | Topology Sensitive: {topo_sens}")
                            st.caption(f"Assigned GPUs: {assigned_gpus} | Placement Penalty: {place_pen} | Communication Cost: {comm_cost}")
                        else:
                            st.caption("Placement details: job not found in current state records")
                except Exception as e:
                    st.caption(f"Placement details unavailable: {e}")
            else:
                st.caption("Placement details: PPO scheduler not selected or no selected job")
        except Exception as e:
            st.caption(f"Topology view unavailable: {e}")

    # --- STATE CONSISTENCY DEBUG (development only) ---
    with st.expander("State Consistency Debug", expanded=False):
        live = st.session_state.sim_state.get("live_state", {}) if st.session_state.sim_state else {}
        sim_time = live.get("simulation_time", 0.0)
        completed = len(live.get("completed_jobs", []))
        waiting = len(live.get("waiting_jobs", []))
        running = len(live.get("running_jobs", []))
        selected = live.get("selected_job", None)
        selected_action = live.get("selected_action", None)
        util = live.get("current_utilization", 0.0)
        gpu_count = live.get("configured_gpu_count", None)
        print_parts = [
            f"simulation_time: {sim_time}",
            f"completed: {completed}",
            f"waiting: {waiting}",
            f"running: {running}",
            f"selected_job: {selected}",
            f"selected_action: {selected_action}",
            f"utilization: {util}",
            f"gpu_count: {gpu_count}",
        ]
        for p in print_parts:
            st.text(p)
        # Simple consistency checks
        checks = []
        if sim_time == 0 and completed > 0:
            checks.append("⚠️ WARNING: simulation_time=0 but completed>0")
        if sim_time > 0 and completed >= 0:
            checks.append("✓ simulation_time>0 and completed>=0")
        if gpu_count != 8:
            checks.append(f"⚠️ GPU count={gpu_count} differs from configured 8")
        for c in checks:
            st.info(c)

# --- RIGHT: Comparison / Charts ---
with right_col:
    st.subheader("Comparison Charts")

    if not st.session_state.get("simulation_complete", False):
        st.info("Click **Start** to race all schedulers on the same job stream — results appear here once it finishes.")
    elif st.session_state.df_agg.empty:
        st.info("No benchmark data loaded — click **Start** to run a simulation.")
    else:
        # --- Chart 1: Average Waiting Time ---
        st.markdown("##### Average Waiting Time")
        fig1 = px.bar(
            st.session_state.df_agg,
            x="scheduler",
            y="average_waiting_time_mean",
            color="scenario",
            title="Average Waiting Time by Scheduler & Scenario",
            labels={"average_waiting_time_mean": "Wait (s)", "scheduler": "Scheduler"},
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Chart 2: P95 Waiting Time ---
        st.markdown("##### P95 Waiting Time")
        fig2 = px.bar(
            st.session_state.df_agg,
            x="scheduler",
            y="p95_waiting_time_mean",
            color="scenario",
            title="P95 Waiting Time by Scheduler & Scenario",
            labels={"p95_waiting_time_mean": "P95 Wait (s)", "scheduler": "Scheduler"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --- Chart 3: Average JCT ---
        st.markdown("##### Average JCT")
        fig3 = px.bar(
            st.session_state.df_agg,
            x="scheduler",
            y="average_turnaround_time_mean",
            color="scenario",
            title="Average JCT by Scheduler & Scenario",
            labels={"average_turnaround_time_mean": "JCT (s)", "scheduler": "Scheduler"},
        )
        st.plotly_chart(fig3, use_container_width=True)

        # --- Chart 4: GPU Utilization ---
        st.markdown("##### GPU Utilization")
        fig4 = px.bar(
            st.session_state.df_agg,
            x="scheduler",
            y="gpu_utilization_mean",
            color="scenario",
            title="GPU Utilization by Scheduler & Scenario",
            labels={"gpu_utilization_mean": "Utilization", "scheduler": "Scheduler"},
        )
        st.plotly_chart(fig4, use_container_width=True)

        # --- Chart 5: Throughput ---
        st.markdown("##### Throughput")
        fig5 = px.bar(
            st.session_state.df_agg,
            x="scheduler",
            y="throughput_mean",
            color="scenario",
            title="Throughput by Scheduler & Scenario",
            labels={"throughput_mean": "Jobs/s", "scheduler": "Scheduler"},
        )
        st.plotly_chart(fig5, use_container_width=True)

        # --- Chart 6: Fairness ---
        st.markdown("##### Fairness (Jain's Index)")
        fig6 = px.bar(
            st.session_state.df_agg,
            x="scheduler",
            y="jains_fairness_index_mean",
            color="scenario",
            title="Fairness by Scheduler & Scenario",
            labels={
                "jains_fairness_index_mean": "Jain's Index",
                "scheduler": "Scheduler",
            },
        )
        st.plotly_chart(fig6, use_container_width=True)

        # --- OOD/Generalization section ---
        st.markdown("---")
        st.subheader("OOD / Generalization (from experiment)")
        # Load the generalization summary we generated earlier
        gen_csv = Path("artifacts/reward_ablation/generalization_summary.csv")
        if gen_csv.exists():
            gdf = pd.read_csv(gen_csv)
            fig_g = px.bar(
                gdf,
                x="shift",
                y="average_turnaround_time_mean",
                color="model",
                title="JCT vs Distribution Shift (lower is better)",
                labels={"average_turnaround_time_mean": "Avg JCT (s)", "shift": "Distribution Shift", "model": "Model"},
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(
                "A_baseline: +4.8% avg degradation vs ID; F_balanced: -5.6% avg degradation (improvement)."
            )
        else:
            st.info("Run `python scripts/generalize_ppo.py --mode report` to generate generalization data.")

# --- Footer ---
st.divider()
st.caption(
    "GPU-Sage — RL for Adaptive GPU Cluster Scheduling | "
    "github.com/anikkumarghosh/gpu-sage"
)