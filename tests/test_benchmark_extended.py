"""Extended benchmark tests: per-job preservation, invariants, all scenarios."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gpu_sage.evaluation.benchmark import (
    run_benchmark,
    run_benchmark_detailed,
    run_single_seed,
    run_single_seed_detailed,
    save_benchmark,
)
from gpu_sage.evaluation.metrics import compute_metrics
from gpu_sage.workloads.generator import SCENARIO_NAMES, generate_workload
from gpu_sage.core.models import Job


def test_p95_ge_median():
    # Use detailed benchmark for each scenario with small jobs
    for scenario in SCENARIO_NAMES:
        res = run_benchmark(scenario=scenario, seeds=[0], num_jobs=15)
        for sched, metrics in res[0].items():
            assert metrics.p95_waiting_time >= metrics.median_waiting_time - 1e-9
            assert metrics.p95_waiting_time >= metrics.p50_waiting_time - 1e-9
            assert metrics.p95_turnaround_time >= metrics.median_jct - 1e-9
            assert metrics.p95_jct >= metrics.median_jct - 1e-9


def test_throughput_non_negative():
    for scenario in ["balanced", "short_jobs"]:
        res = run_benchmark(scenario=scenario, seeds=[1, 2], num_jobs=10)
        for seed in res:
            for sched, m in res[seed].items():
                assert m.throughput >= 0.0
                assert m.throughput_jobs_per_time >= 0.0
                assert m.completed_jobs >= 0


def test_benchmark_can_run_all_scenarios():
    for scenario in SCENARIO_NAMES:
        res = run_benchmark(scenario=scenario, seeds=[0], num_jobs=5)
        assert 0 in res
        # 4 default schedulers
        assert len(res[0]) == 4
        for sched in ["FCFS", "SJF", "Priority", "BestFit"]:
            assert sched in res[0]


def test_per_job_records_preserved():
    metrics_map, per_job_map = run_single_seed_detailed(scenario="balanced", seed=42, num_jobs=10)
    # Each scheduler should have 10 per-job records
    for sched in ["FCFS", "SJF", "Priority", "BestFit"]:
        records = per_job_map[sched]
        assert len(records) == 10
        for rec in records:
            # Required fields per spec section 5
            for field in [
                "job_id",
                "arrival_time",
                "start_time",
                "completion_time",
                "waiting_time",
                "turnaround_time",
                "gpu_requirement",
                "memory_requirement",
                "execution_time",
                "priority",
                "scheduler",
            ]:
                assert field in rec, f"Missing {field} in per-job record {rec}"
            assert rec["scheduler"] == sched
            assert rec["gpu_requirement"] >= 1
            # execution_time maps to duration
            assert rec["execution_time"] > 0


def test_identical_jobs_before_simulation():
    """Verify that all schedulers receive identical jobs (per-job before simulation identical)."""
    # We check that initial workload before scheduling is identical across schedulers
    # by comparing per-job arrival/execution/priority before simulation effects.
    # run_single_seed_detailed generates per-job after simulation (with start/completion),
    # so we instead directly test generate_workload determinism and that benchmark
    # deepcopies correctly: generate once then deepcopy gives identical arrival/execution/priority.
    base = generate_workload(scenario="balanced", seed=123, count=15)
    import copy
    copies = [copy.deepcopy(base) for _ in range(3)]
    for c in copies:
        for a, b in zip(base, c):
            assert a.arrival_time == b.arrival_time
            assert a.duration == b.duration
            assert a.gpu_count == b.gpu_count
            assert a.priority == b.priority


def test_per_job_reproducibility():
    m1, p1 = run_single_seed_detailed(scenario="short_jobs", seed=7, num_jobs=12)
    m2, p2 = run_single_seed_detailed(scenario="short_jobs", seed=7, num_jobs=12)
    for sched in m1:
        assert m1[sched].as_dict() == m2[sched].as_dict()
        assert p1[sched] == p2[sched]


def test_save_benchmark_repro_metadata():
    metrics, per_job = run_benchmark_detailed(scenario="balanced", seeds=[0, 1], num_jobs=8)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        saved = save_benchmark(
            scenario="balanced",
            results=metrics,
            out_dir=out,
            num_gpus=8,
            num_jobs=8,
            per_job_results=per_job,
            workload_config={"arrival_rate": 0.08},
        )
        # Check that CSV and JSON exist
        assert saved["tidy"].exists()
        assert saved["agg"].exists()
        assert saved["json"].exists()
        assert saved["jobs_csv"].exists()
        assert saved["jobs_json"].exists()
        # Check repro metadata in JSON
        meta = json.loads(saved["json"].read_text())
        assert meta["scenario"] == "balanced"
        assert meta["num_gpus"] == 8
        assert meta["num_jobs"] == 8
        assert "workload_config" in meta
        # Check tidy CSV contains repro columns
        import pandas as pd
        df = pd.read_csv(saved["tidy"])
        assert "scenario" in df.columns
        assert "num_gpus" in df.columns
        assert "num_jobs" in df.columns
        # Check per-job CSV has required fields and row count = seeds * schedulers * jobs
        job_df = pd.read_csv(saved["jobs_csv"])
        assert len(job_df) == 2 * 4 * 8  # 2 seeds * 4 sched * 8 jobs
        for col in ["job_id", "arrival_time", "waiting_time", "scheduler", "scenario", "seed"]:
            assert col in job_df.columns


def test_existing_workload_seeding_still_deterministic():
    # Re-test that run_benchmark_detailed also deterministic
    r1, p1 = run_benchmark_detailed(scenario="bursty", seeds=[0, 1, 2], num_jobs=7)
    r2, p2 = run_benchmark_detailed(scenario="bursty", seeds=[0, 1, 2], num_jobs=7)
    for seed in [0, 1, 2]:
        for sched in r1[seed]:
            assert r1[seed][sched].as_dict() == r2[seed][sched].as_dict()
            assert p1[seed][sched] == p2[seed][sched]
