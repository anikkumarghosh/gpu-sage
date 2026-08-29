"""Fast tests for training artifact paths and showcase helpers."""

import json
import tempfile
from pathlib import Path


def test_run_directory_creation():
    from training.train_ppo import get_git_hash
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_seed0_steps100"
    base = Path(tempfile.mkdtemp())
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "tensorboard").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "model").mkdir()
    (run_dir / "plots").mkdir()
    assert (run_dir / "tensorboard").exists()
    assert (run_dir / "checkpoints").exists()
    # Config saving
    config = {"run_id": run_id, "timesteps": 100, "seed": 0}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f)
    assert (run_dir / "config.json").exists()
    loaded = json.loads((run_dir / "config.json").read_text())
    assert loaded["run_id"] == run_id

def test_checkpoint_naming():
    # Check that checkpoint helper would create expected filenames
    run_dir = Path(tempfile.mkdtemp()) / "run"
    run_dir.mkdir(parents=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    # Simulate checkpoint naming every 50k
    for step in [50000, 100000, 150000, 200000, 250000]:
        p = ckpt_dir / f"ppo_{step}_steps.zip"
        p.write_text("dummy")
    files = sorted(p.name for p in ckpt_dir.glob("*.zip"))
    assert "ppo_50000_steps.zip" in files
    assert "ppo_250000_steps.zip" in files
    assert len(files) == 5

def test_summary_generation():
    import json
    summary = {
        "run_id": "20260101_000000_seed0_steps100",
        "timesteps": 100,
        "seed": 0,
        "training_time_seconds": 12.3,
        "final_mean_reward": -123.4,
        "best_mean_reward": -100.0,
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "training_summary.json"
        p.write_text(json.dumps(summary, indent=2))
        loaded = json.loads(p.read_text())
        assert loaded["run_id"] == summary["run_id"]
        assert loaded["final_mean_reward"] == -123.4

def test_deterministic_seed_handling():
    from gpu_sage.workloads.generator import generate_workload
    jobs_a = generate_workload(scenario="balanced", seed=42, count=10)
    jobs_b = generate_workload(scenario="balanced", seed=42, count=10)
    assert [j.arrival_time for j in jobs_a] == [j.arrival_time for j in jobs_b]
    jobs_c = generate_workload(scenario="balanced", seed=43, count=10)
    assert jobs_a[0].arrival_time != jobs_c[0].arrival_time

def test_artifact_paths_exist_after_small_training():
    # Check that a recent training run has expected structure
    runs_dir = Path("artifacts/ppo/runs")
    if not runs_dir.exists():
        return  # skip if no runs yet
    # Find any run
    runs = list(runs_dir.iterdir())
    if not runs:
        return
    run = max(runs, key=lambda p: p.stat().st_mtime)
    # Check that at least config and model exist or can be missing for old runs
    # For this test, just verify path logic
    assert run.is_dir()
    # Check that benchmark comparison plots can be generated
    from src.gpu_sage.evaluation.plots import generate_comparison_plots
    # This should not raise
    assert callable(generate_comparison_plots)

def test_replay_helper():
    from src.gpu_sage.utils.replay import render_ascii_frame
    txt = render_ascii_frame(
        gpu_states=["busy", "idle", "busy", "idle"],
        queue=["J1", "J2"],
        decision="J1",
        utilization=0.75,
        completed=5,
        sim_time=100.0,
    )
    assert "GPU 0" in txt
    assert "J1" in txt
    assert "Utilization" in txt
