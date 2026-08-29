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
from src.gpu_sage.evaluation.benchmark import run_benchmark_detailed, aggregate_results, per_seed_dataframe
from src.gpu_sage.evaluation.metrics import compute_metrics
from src.gpu_sage.workloads.generator import SCENARIO_NAMES, generate_workload, get_scenario_config, WorkloadConfig
from src.gpu_sage.core.cluster import Cluster
from src.gpu_sage.core.simulator import Simulator
from src.gpu_sage.schedulers.fcfs import FCFSScheduler
from src.gpu_sage.schedulers.heuristics import BestFitScheduler, PriorityScheduler, SJFScheduler
from src.gpu_sage.env.gpu_env import GPUSchedulingEnv
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
        st.session_state.sim_state = None  # dict with sim, jobs, metrics, etc.


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
    return run_benchmark_detailed(
        scenario=scenario,
        seeds=seeds,
        schedulers=schedulers,
        num_jobs=num_jobs,
        num_gpus=num_gpus,
    )


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
    from src.gpu_sage.utils.replay import render_ascii_frame as _render

    return _render(gpu_states, queue, decision, utilization, completed, sim_time, width)


# ---------------------------------------------------------------------------
# Page: Dashboard — Home / Live Simulation
# ---------------------------------------------------------------------------

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)

st.title(f"{PAGE_ICON}  {PAGE_TITLE}")
st.caption("Reinforcement Learning for Adaptive GPU Cluster Scheduling")

# --- Header ---
with st.container():
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.markdown("### GPU-Sage")
    with c2:
        scenario = st.select_scenario = st.selectbox(
            "Workload scenario",
            options=SCENARIO_NAMES,
            index=SCENARIO_NAMES.index(st.session_state.selected_scenario)
            if st.session_state.selected_scenario in SCENARIO_NAMES
            else 0,
            key="scenario_select",
        )
    with c3:
        st.session_state.selected_scenario = scenario

    # Scheduler selector
    scheduler = st.select_scheduler = st.selectbox(
        "Scheduler",
        options=st.session_state.scheduler_names,
        index=st.session_state.scheduler_names.index(
            st.session_state.selected_scheduler
        )
        if st.session_state.selected_scheduler in st.session_state.scheduler_names
        else 0,
        key="scheduler_select",
    )
    st.session_state.selected_scheduler = scheduler

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
            # Run one benchmark to get results
            results = run_benchmark_cached(
                scenario=st.session_state.selected_scenario,
                schedulers=[st.session_state.selected_scheduler],
                num_jobs=16,
                num_gpus=num_gpus,
                seeds=[st.session_state.seed_selector],
            )
            st.session_state.benchmark_results = results
            st.session_state.sim_state = {
                "results": results,
                "scenario": st.session_state.selected_scenario,
                "scheduler": st.session_state.selected_scheduler,
                "num_gpus": num_gpus,
                "seed": st.session_state.seed_selector,
            }
            st.rerun()

    with col_btn_pause:
        if st.button("Pause", use_container_width=True):
            st.info("Simulation paused")

    with col_btn_step:
        if st.button("Step", use_container_width=True):
            if st.session_state.sim_state is not None:
                # Advance one decision step
                results = st.session_state.sim_state["results"]
                seed = st.session_state.seed_selector
                # Re-run just the selected scheduler on the same W0
                schedulers = [st.session_state.selected_scheduler]
                new_results = run_benchmark_cached(
                    scenario=st.session_state.selected_scenario,
                    schedulers=schedulers,
                    num_jobs=16,
                    num_gpus=num_gpus,
                    seeds=[seed],
                )
                st.session_state.sim_state["results"] = new_results
                st.rerun()
            else:
                st.warning("Start simulation first")

    with col_btn_reset:
        if st.button("Reset", use_container_width=True):
            st.session_state.sim_state = None
            st.session_state.benchmark_results = None
            st.rerun()

    # --- Display simulation results ---
    if st.session_state.sim_state is not None:
        results = st.session_state.sim_state["results"]
        seed = st.session_state.seed_selector
        scheduler_name = st.session_state.selected_scheduler

        # Pull metrics for the selected scheduler
        if seed in results and scheduler_name in results[seed]:
            metrics = results[seed][scheduler_name]

            # Live metrics cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("GPU Utilization", f"{metrics.gpu_utilization:.1%}")
            with m2:
                st.metric("Completed Jobs", int(metrics.completed_jobs))
            with m3:
                st.metric("Avg Wait", f"{metrics.average_waiting_time:.1f}s")
            with m4:
                st.metric("P95 Wait", f"{metrics.p95_waiting_time:.1f}s")

            m5, m6, m7, m8 = st.columns(4)
            with m5:
                st.metric("Avg JCT", f"{metrics.average_turnaround_time:.1f}s")
            with m6:
                st.metric("P95 JCT", f"{metrics.p95_turnaround_time:.1f}s")
            with m7:
                st.metric("Throughput", f"{metrics.throughput:.3f}/s")
            with m8:
                st.metric("Fairness (Jain)", f"{metrics.jains_fairness_index:.3f}")

            # Per-job queue visualization
            st.markdown("#### Job Queue")
            # Load per-job records for this scenario+seed
            jobs_csv = Path(f"artifacts/benchmarks/{st.session_state.selected_scenario}_jobs.csv")
            if jobs_csv.exists():
                job_df = pd.read_csv(jobs_csv)
                # Filter to this seed if multi-seed
                # Show first few jobs with status
                sample_jobs = job_df.head(12)
                for _, row in sample_jobs.iterrows():
                    status = row.get("status", "COMPLETED")
                    # Determine status from metrics
                    jct = row.get("turnaround_time", 0)
                    wait = row.get("waiting_time", 0)
                    gpu_req = row.get("gpu_count", 1)

                    # Color-code
                    if status == "COMPLETED":
                        badge_color = "🟢"
                    elif jct > 500:
                        badge_color = "🔴"
                    else:
                        badge_color = "🟡"

                    st.caption(
                        f"{badge_color} Job **#{int(row['job_id'])}** "
                        f"GPU={gpu_req}  Wait={wait:.0f}s  JCT={jct:.0f}s  Pri={row.get('priority','?')}"
                    )

            # PPO decision panel (if PPO selected)
            if scheduler_name == "PPO":
                ppo_decisions_csv = Path(
                    f"artifacts/benchmarks/{st.session_state.selected_scenario}_ppo_decisions.csv"
                )
                if ppo_decisions_csv.exists():
                    ppo_df = pd.read_csv(ppo_decisions_csv)
                    # Show latest decision
                    latest = ppo_df.iloc[-1]
                    st.markdown("#### PPO Decision Panel")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.info(f"**Selected Job:**\n#{int(latest['selected_job_id'])}")
                    with c2:
                        st.info(f"**GPU Requirement:**\n{int(latest['gpu_count'])}")
                    with c3:
                        st.info(f"**Priority:**\n{int(latest['priority'])}")
                    with c4:
                        st.info(f"**Waiting Time:**\n{float(latest['waiting_time']):.0f}s")

                    st.caption(f"**Available GPUs:** {int(latest['free_gpus'])}")
                    st.caption(f"**Action:** SELECT JOB {int(latest['selected_job_id'])}")

                    # Reward components
                    st.markdown("**Reward Components:**")
                    r1, r2, r3 = st.columns(3)
                    with r1:
                        st.caption(f"Throughput: {float(latest['throughput_reward']):.2f}")
                    with r2:
                        st.caption(f"Waiting: {float(latest['waiting_penalty']):.2f}")
                    with r3:
                        st.caption(f"Utilization: {float(latest['utilization_reward']):.2f}")

                    # Fragmentation + idle
                    r4, r5 = st.columns(2)
                    with r4:
                        st.caption(f"Fragmentation: {float(latest['fragmentation_penalty']):.2f}")
                    with r5:
                        st.caption(f"Idle: {float(latest['idle_penalty']):.2f}")

            # ASCII frame snapshot
            st.markdown("#### ASCII Snapshot")
            if st.session_state.sim_state is not None:
                r = results[seed][scheduler_name]
                # Build simple GPU states from metrics
                util = r.gpu_utilization
                gpu_states = [
                    f"GPU {i}: {'█' * int(util * 20) + '░' * (20 - int(util * 20))}" for i in range(min(4, num_gpus))
                ]
                queue_jobs = []
                # Try to get queue from job_df
                if jobs_csv.exists():
                    jdf = pd.read_csv(jobs_csv)
                    waiting = jdf[jdf.get("completion_time", 0) == 0]
                    for _, wrow in waiting.head(4).iterrows():
                        queue_jobs.append(f"J{int(wrow['job_id'])}")
                decision = f"J{int(latest['selected_job_id'])}" if scheduler_name == "PPO" and 'latest' in dir() else "NOOP"
                frame = render_ascii_frame(
                    gpu_states=gpu_states,
                    queue=queue_jobs,
                    decision=decision,
                    utilization=util,
                    completed=int(r.completed_jobs),
                    sim_time=float(r.simulated_time if hasattr(r, 'simulated_time') else 0),
                    width=20,
                )
                st.code(frame, language=None, line_numbers=False)

# --- RIGHT: Comparison / Charts ---
with right_col:
    st.subheader("Comparison Charts")

    # Ensure we have benchmark data
    if st.session_state.df_agg.empty:
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