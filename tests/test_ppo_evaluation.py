"""PPO integration tests — deterministic evaluation on same workload as baselines."""

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gpu_sage.evaluation.benchmark import (
    DEFAULT_SCHEDULERS,
    evaluate_ppo_fixed_workload,
    run_benchmark,
    run_single_seed,
)
from gpu_sage.workloads.generator import generate_workload


def _has_ppo_model():
    # Look for any trained model in artifacts/ppo
    for p in [Path("artifacts/ppo/models/final_model.zip"), Path("artifacts/ppo/final_model.zip")]:
        if p.exists():
            return p
    return None


def test_ppo_model_can_be_loaded():
    model_path = _has_ppo_model()
    if model_path is None:
        pytest.skip("No trained PPO model found (train with: python training/train_ppo.py --steps 5000)")
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(model_path))
    assert model is not None
    # Check policy exists
    assert hasattr(model, "predict")


def test_ppo_can_run_against_fixed_workload():
    jobs = generate_workload(scenario="balanced", seed=0, count=10)
    # Use heuristic fallback if no model
    model_path = _has_ppo_model()
    metrics, per_job, logs, stats = evaluate_ppo_fixed_workload(
        jobs=copy.deepcopy(jobs), model_path=model_path, num_gpus=8, seed=0
    )
    assert metrics.completed_jobs >= 0
    assert len(per_job) == 10
    assert len(logs) > 0
    assert stats["steps"] > 0
    assert "scheduling_decisions" in stats
    assert "noop_actions" in stats


def test_ppo_evaluation_deterministic():
    jobs = generate_workload(scenario="balanced", seed=42, count=15)
    model_path = _has_ppo_model()
    m1, p1, l1, s1 = evaluate_ppo_fixed_workload(jobs=copy.deepcopy(jobs), model_path=model_path, seed=42)
    m2, p2, l2, s2 = evaluate_ppo_fixed_workload(jobs=copy.deepcopy(jobs), model_path=model_path, seed=42)
    assert m1.as_dict() == m2.as_dict()
    assert p1 == p2
    # Logs should be deterministic too (same actions)
    assert l1 == l2
    assert s1["action_distribution"] == s2["action_distribution"]


def test_ppo_and_fcfs_receive_same_workload():
    jobs = generate_workload(scenario="short_jobs", seed=123, count=12)
    # FCFS metrics via benchmark
    fcfs_metrics = run_single_seed(scenario="short_jobs", seed=123, schedulers=["FCFS"], num_jobs=12)["FCFS"]
    # PPO on same jobs
    model_path = _has_ppo_model()
    ppo_metrics, ppo_per_job, _, _ = evaluate_ppo_fixed_workload(jobs=copy.deepcopy(jobs), model_path=model_path, seed=123)
    # Both should see same job_ids
    fcfs_base_ids = sorted(j.job_id for j in jobs)
    ppo_ids = sorted(r["job_id"] for r in ppo_per_job)
    assert fcfs_base_ids == ppo_ids
    # Total jobs same
    assert ppo_metrics.total_jobs == fcfs_metrics.total_jobs == 12


def test_ppo_produces_valid_actions():
    jobs = generate_workload(scenario="balanced", seed=7, count=10)
    model_path = _has_ppo_model()
    _, _, logs, stats = evaluate_ppo_fixed_workload(jobs=copy.deepcopy(jobs), model_path=model_path, seed=7)
    # All actions should be in range [0, max_jobs] and not invalid beyond count
    for entry in logs:
        assert 0 <= entry["action"] <= 16  # default max_jobs 16
        # If invalid_action, it should be counted
    # Invalid actions should be tracked
    assert stats["invalid_actions"] >= 0
    # At least some scheduling decisions
    assert stats["scheduling_decisions"] + stats["noop_actions"] == stats["steps"]


def test_ppo_evaluation_same_metric_schema_as_heuristics():
    jobs = generate_workload(scenario="balanced", seed=0, count=10)
    # Heuristic
    heur_metrics = run_single_seed(scenario="balanced", seed=0, schedulers=["FCFS"], num_jobs=10)["FCFS"]
    # PPO
    model_path = _has_ppo_model()
    ppo_metrics, _, _, _ = evaluate_ppo_fixed_workload(jobs=copy.deepcopy(jobs), model_path=model_path, seed=0)
    # Schema should match: same keys
    assert set(heur_metrics.as_dict().keys()) == set(ppo_metrics.as_dict().keys())
    # All required metrics present
    for key in ["completed_jobs", "throughput", "average_waiting_time", "median_waiting_time", "p95_waiting_time",
                "average_turnaround_time", "median_jct", "p95_jct", "gpu_utilization", "gpu_idle_time",
                "resource_allocation_efficiency", "rejected_jobs", "scheduling_decisions",
                "jains_fairness_index"]:
        assert key in ppo_metrics.as_dict()


def test_tiny_ppo_training_and_eval():
    """Train a tiny PPO model (2048 steps) and evaluate — no 250k required."""
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from gpu_sage.env.gpu_env import GPUSchedulingEnv
    from gpu_sage.workloads.generator import WorkloadConfig

    def make_tiny_env():
        return GPUSchedulingEnv(
            num_gpus=4,
            gpu_memory_gb=80,
            max_jobs=8,
            episode_jobs=20,
            workload_config=WorkloadConfig(arrival_rate=0.1, min_gpus=1, max_gpus=2, min_duration=10, max_duration=30),
            seed=0,
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_env = DummyVecEnv([make_tiny_env])
        try:
            model = MaskablePPO("MultiInputPolicy", train_env, verbose=0, seed=0, device="cpu")
            model.learn(total_timesteps=2048, progress_bar=False)
            model_path = tmp_path / "tiny_model.zip"
            model.save(str(model_path))
            assert model_path.exists()

            # Evaluate the tiny model on fixed workload
            jobs = generate_workload(scenario="short_jobs", seed=0, count=10)
            metrics, per_job, logs, stats = evaluate_ppo_fixed_workload(
                jobs=jobs, model_path=str(model_path), num_gpus=4, seed=0
            )
            assert metrics.total_jobs == 10
            assert len(logs) > 0
        finally:
            train_env.close()


def test_ppo_per_decision_log_saved():
    jobs = generate_workload(scenario="balanced", seed=0, count=8)
    model_path = _has_ppo_model()
    _, _, logs, _ = evaluate_ppo_fixed_workload(jobs=copy.deepcopy(jobs), model_path=model_path, seed=0)
    for entry in logs:
        for field in ["simulation_time", "queue_length", "selected_job_id", "action", "reward", "free_gpus", "gpu_utilization"]:
            assert field in entry, f"Missing {field} in decision log"

    # Also test that benchmark saving includes ppo logs
    from gpu_sage.evaluation.benchmark import run_benchmark_detailed_with_logs

    with tempfile.TemporaryDirectory() as tmp:
        metrics, per_job, ppo_logs = run_benchmark_detailed_with_logs(
            scenario="balanced", seeds=[0], num_jobs=8, schedulers=["FCFS", "PPO"], ppo_model_path=model_path
        )
        assert "PPO" in metrics[0]
        assert "FCFS" in metrics[0]
        # PPO logs should be present
        assert 0 in ppo_logs and "PPO" in ppo_logs[0]
        assert len(ppo_logs[0]["PPO"]) > 0
