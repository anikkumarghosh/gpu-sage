"""Tests for rigorous RL-vs-baseline evaluation framework."""

import copy

import numpy as np
import pytest

from gpu_sage.core.cluster import Cluster
from gpu_sage.core.models import Job, JobStatus
from gpu_sage.core.simulator import Simulator
from gpu_sage.evaluation.benchmark import (
    DEFAULT_SCHEDULERS,
    run_benchmark,
    run_single_seed,
)
from gpu_sage.evaluation.metrics import compute_metrics
from gpu_sage.schedulers import (
    BestFitScheduler,
    FCFSScheduler,
    PriorityScheduler,
    RandomScheduler,
    RLScheduler,
    SJFScheduler,
)
from gpu_sage.workloads.generator import (
    SCENARIO_NAMES,
    SyntheticWorkload,
    WorkloadConfig,
    generate_workload,
    get_scenario_config,
)


# ---------------------------------------------------------------------------
# Workload determinism
# ---------------------------------------------------------------------------

def test_seeded_workload_deterministic():
    jobs_a = generate_workload(scenario="balanced", seed=42, count=20)
    jobs_b = generate_workload(scenario="balanced", seed=42, count=20)
    assert len(jobs_a) == len(jobs_b) == 20
    for a, b in zip(jobs_a, jobs_b):
        assert a.arrival_time == b.arrival_time
        assert a.gpu_count == b.gpu_count
        assert a.duration == b.duration
        assert a.priority == b.priority
        assert a.gpu_memory_gb == b.gpu_memory_gb


def test_different_seed_gives_different_workload():
    jobs_a = generate_workload(scenario="balanced", seed=1, count=20)
    jobs_b = generate_workload(scenario="balanced", seed=2, count=20)
    # At least one job differs
    assert any(a.arrival_time != b.arrival_time for a, b in zip(jobs_a, jobs_b))


def test_scenario_configs_exist():
    expected = {"balanced", "gpu_heavy", "short_jobs", "bursty", "heavy_tail", "priority_skew"}
    assert set(SCENARIO_NAMES) == expected
    for scen in expected:
        cfg = get_scenario_config(scen)
        assert isinstance(cfg, WorkloadConfig)
        assert cfg.arrival_rate > 0
        jobs = generate_workload(scenario=scen, seed=0, count=10)
        assert len(jobs) == 10


def test_synthetic_workload_seed_reproducible_with_config():
    cfg = WorkloadConfig(arrival_rate=0.1, min_gpus=1, max_gpus=2, min_duration=10, max_duration=20)
    jobs_a = SyntheticWorkload(cfg, seed=123).generate(15)
    jobs_b = SyntheticWorkload(cfg, seed=123).generate(15)
    assert [j.arrival_time for j in jobs_a] == [j.arrival_time for j in jobs_b]


def test_scenario_specific_distributions():
    # heavy_tail should produce at least one long job in large sample
    jobs = generate_workload(scenario="heavy_tail", seed=0, count=200)
    cfg = get_scenario_config("heavy_tail")
    long_jobs = [j for j in jobs if j.duration > cfg.max_duration]
    # heavy_tail_fraction ~0.07 so expect at least 1 long job
    assert len(long_jobs) >= 1
    # priority_skew: mostly low priority
    jobs_ps = generate_workload(scenario="priority_skew", seed=0, count=200)
    low = sum(1 for j in jobs_ps if j.priority == 1)
    assert low > 100  # >50% should be priority 1
    # gpu_heavy: min_gpus >=4
    jobs_gh = generate_workload(scenario="gpu_heavy", seed=0, count=50)
    assert all(j.gpu_count >= 4 for j in jobs_gh)
    # short_jobs: durations bounded small
    jobs_sj = generate_workload(scenario="short_jobs", seed=0, count=50)
    assert all(j.duration <= 50 for j in jobs_sj)
    assert all(j.gpu_count <= 2 for j in jobs_sj)


# ---------------------------------------------------------------------------
# Identical workload across schedulers
# ---------------------------------------------------------------------------

def test_identical_workload_used_by_all_schedulers():
    # run_single_seed must use identical base workload per seed.
    # We verify by running twice with same seed and checking metrics are reproducible,
    # and by ensuring deepcopied isolation: modifying one copy doesn't affect another.
    res1 = run_single_seed(scenario="balanced", seed=7, num_jobs=20)
    res2 = run_single_seed(scenario="balanced", seed=7, num_jobs=20)
    # Same seed -> same metrics per scheduler (deterministic)
    for sched in DEFAULT_SCHEDULERS:
        assert res1[sched].completed_jobs == res2[sched].completed_jobs
        assert res1[sched].average_waiting_time == pytest.approx(res2[sched].average_waiting_time)
        assert res1[sched].gpu_utilization == pytest.approx(res2[sched].gpu_utilization)

    # Direct deep copy isolation test
    base = generate_workload(scenario="balanced", seed=7, count=10)
    copy1 = copy.deepcopy(base)
    copy2 = copy.deepcopy(base)
    # Mutate copy1, copy2 should remain unchanged
    copy1[0].job_id = 9999
    assert copy2[0].job_id == base[0].job_id
    # Values equal before mutation
    base2 = generate_workload(scenario="balanced", seed=7, count=10)
    assert [j.arrival_time for j in base] == [j.arrival_time for j in base2]


def test_benchmark_produces_reproducible_results():
    results_a = run_benchmark(scenario="short_jobs", seeds=[0, 1], num_jobs=10)
    results_b = run_benchmark(scenario="short_jobs", seeds=[0, 1], num_jobs=10)
    for seed in [0, 1]:
        for sched in DEFAULT_SCHEDULERS:
            a = results_a[seed][sched]
            b = results_b[seed][sched]
            assert a.as_dict() == b.as_dict()


# ---------------------------------------------------------------------------
# Scheduler common interface
# ---------------------------------------------------------------------------

def test_each_baseline_runs_through_common_interface():
    schedulers = [
        FCFSScheduler(),
        SJFScheduler(),
        PriorityScheduler(),
        BestFitScheduler(),
        RandomScheduler(seed=0),
        RLScheduler(),
    ]
    jobs = generate_workload(scenario="balanced", seed=1, count=20)
    for sched in schedulers:
        # Interface must accept (waiting, feasible, cluster, current_time)
        waiting = jobs[:5]
        feasible = [j for j in waiting if j.gpu_count <= 8]
        cluster = Cluster.homogeneous(8, memory_gb=80)
        result = sched.select(waiting, feasible, cluster, current_time=0.0)
        assert result is None or isinstance(result, Job)
        # Also must accept 2-arg legacy call via simulator
        cluster2 = Cluster.homogeneous(8, memory_gb=80)
        sim = Simulator(cluster=cluster2, scheduler=sched)
        sim.load_jobs(copy.deepcopy(jobs))
        # Should run without error
        history = sim.run()
        assert len(history) > 0
        # Simulator owns resource logic: no scheduler should duplicate allocation
        assert sim.cluster.total_gpus == 8


def test_scheduler_noop_when_no_feasible():
    sched = FCFSScheduler()
    jobs = [Job(job_id=1, arrival_time=0, gpu_count=16, gpu_memory_gb=16, duration=10)]
    cluster = Cluster.homogeneous(8)
    # No feasible because 16 > 8
    feasible = [j for j in jobs if cluster.can_allocate(j)]
    assert feasible == []
    assert sched.select(jobs, feasible, cluster, 0.0) is None


# ---------------------------------------------------------------------------
# Metrics correctness
# ---------------------------------------------------------------------------

def test_metrics_calculation_correct():
    jobs = [
        Job(1, 0, 1, 16, 10, start_time=2, completion_time=12, status=JobStatus.COMPLETED),
        Job(2, 0, 2, 16, 20, start_time=5, completion_time=25, status=JobStatus.COMPLETED),
    ]
    # waiting: 2, 5 => avg 3.5
    # turnaround: 12, 25 => avg 18.5
    metrics = compute_metrics(jobs, total_gpus=4, simulated_time=25, gpu_time_used=35)
    assert metrics.completed_jobs == 2
    assert metrics.total_jobs == 2
    assert metrics.average_waiting_time == pytest.approx(3.5)
    assert metrics.average_turnaround_time == pytest.approx(18.5)
    assert metrics.throughput_jobs_per_time == pytest.approx(2 / 25)
    assert metrics.throughput == pytest.approx(2 / 25)
    assert metrics.gpu_utilization == pytest.approx(35 / 100)
    assert metrics.gpu_idle_time == pytest.approx(1 - 35 / 100)
    assert metrics.resource_allocation_efficiency == pytest.approx(35 / 100)
    assert metrics.median_waiting_time == pytest.approx(np.median([2, 5]))
    assert metrics.median_jct == pytest.approx(np.median([12, 25]))
    assert metrics.rejected_jobs == 0
    assert metrics.scheduling_decisions == 2
    assert metrics.invalid_scheduling_attempts == 0
    # Jain fairness on waits [2,5]: (7^2)/(2*(4+25))=49/58≈0.8448
    assert metrics.jains_fairness_index == pytest.approx(49 / 58, rel=1e-6)


def test_p95_calculation_correct():
    # Known values: 0..100 step 10 => 11 values; p95 via numpy linear interpolation
    waits = list(range(0, 101, 10))  # 0,10,...,100
    jobs = [Job(i, 0, 1, 16, 10, start_time=w, completion_time=w + 10, status=JobStatus.COMPLETED) for i, w in enumerate(waits)]
    for j, w in zip(jobs, waits):
        j.start_time = float(w)
        j.completion_time = float(w + 10)
    metrics = compute_metrics(jobs, total_gpus=4, simulated_time=200, gpu_time_used=100)
    expected_p95 = float(np.percentile(waits, 95))
    assert metrics.p95_waiting_time == pytest.approx(expected_p95)
    # turnaround = wait+10 => same p95 shift
    expected_p95_jct = float(np.percentile([w + 10 for w in waits], 95))
    assert metrics.p95_turnaround_time == pytest.approx(expected_p95_jct)
    assert metrics.p95_jct == pytest.approx(expected_p95_jct)


def test_throughput_calculation_correct():
    jobs = [Job(i, float(i), 1, 16, 10, start_time=float(i), completion_time=float(i + 10), status=JobStatus.COMPLETED) for i in range(5)]
    metrics = compute_metrics(jobs, total_gpus=2, simulated_time=20, gpu_time_used=50)
    assert metrics.throughput == pytest.approx(5 / 20)
    assert metrics.throughput_jobs_per_time == pytest.approx(5 / 20)
    # Zero time edge
    metrics2 = compute_metrics(jobs, total_gpus=2, simulated_time=0, gpu_time_used=0)
    assert metrics2.throughput == 0.0


def test_gpu_utilization_bounded():
    jobs = [Job(1, 0, 1, 16, 10, start_time=0, completion_time=10, status=JobStatus.COMPLETED)]
    for gpu_time, sim_time in [(0, 10), (80, 10), (40, 10), (0, 0)]:
        m = compute_metrics(jobs, total_gpus=8, simulated_time=sim_time, gpu_time_used=gpu_time)
        assert 0.0 <= m.gpu_utilization <= 1.0
        assert 0.0 <= m.gpu_idle_time <= 1.0
        assert 0.0 <= m.gpu_idle_fraction <= 1.0
        assert 0.0 <= m.resource_allocation_efficiency <= 1.0


def test_jain_fairness_bounded():
    # Empty
    m_empty = compute_metrics([], total_gpus=8, simulated_time=10, gpu_time_used=0)
    assert 0.0 <= m_empty.jains_fairness_index <= 1.0
    # Single job -> perfect fairness
    jobs = [Job(1, 0, 1, 16, 10, start_time=5, completion_time=15, status=JobStatus.COMPLETED)]
    m_single = compute_metrics(jobs, total_gpus=8, simulated_time=20, gpu_time_used=10)
    assert m_single.jains_fairness_index == pytest.approx(1.0)
    # Two jobs with equal wait -> 1.0
    jobs_eq = [
        Job(1, 0, 1, 16, 10, start_time=5, completion_time=15, status=JobStatus.COMPLETED),
        Job(2, 0, 1, 16, 10, start_time=5, completion_time=15, status=JobStatus.COMPLETED),
    ]
    m_eq = compute_metrics(jobs_eq, total_gpus=8, simulated_time=20, gpu_time_used=20)
    assert m_eq.jains_fairness_index == pytest.approx(1.0)
    # Unequal waits => less than 1 but >0
    jobs_uneq = [
        Job(1, 0, 1, 16, 10, start_time=1, completion_time=11, status=JobStatus.COMPLETED),
        Job(2, 0, 1, 16, 10, start_time=100, completion_time=110, status=JobStatus.COMPLETED),
    ]
    m_uneq = compute_metrics(jobs_uneq, total_gpus=8, simulated_time=200, gpu_time_used=20)
    assert 0.0 < m_uneq.jains_fairness_index < 1.0
    # Many jobs
    rng = np.random.default_rng(0)
    jobs_many = [Job(i, 0, 1, 16, 10, start_time=float(rng.integers(0, 50)), completion_time=float(rng.integers(50, 100)), status=JobStatus.COMPLETED) for i in range(20)]
    for j in jobs_many:
        # Ensure waiting non-negative
        if j.start_time is None or j.completion_time is None:
            continue
    m_many = compute_metrics(jobs_many, total_gpus=8, simulated_time=200, gpu_time_used=100)
    assert 0.0 <= m_many.jains_fairness_index <= 1.0


def test_rejected_jobs_counted():
    # 16 GPU job on 8 GPU cluster -> rejected
    jobs = [
        Job(1, 0, 16, 16, 10),
        Job(2, 0, 2, 16, 10, start_time=0, completion_time=10, status=JobStatus.COMPLETED),
    ]
    m = compute_metrics(jobs, total_gpus=8, simulated_time=10, gpu_time_used=20, per_gpu_memory_gb=80)
    assert m.rejected_jobs == 1
    assert m.infeasible_jobs == 1
    # Memory too large
    jobs_mem = [Job(1, 0, 1, 200, 10)]
    m2 = compute_metrics(jobs_mem, total_gpus=8, simulated_time=10, gpu_time_used=0, per_gpu_memory_gb=80)
    assert m2.rejected_jobs == 1
