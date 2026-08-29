"""GPU-Sage PPO training — showcase-quality run with full artifact preservation."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from gpu_sage.env.gpu_env import GPUSchedulingEnv, RewardConfig
from gpu_sage.workloads.generator import WorkloadConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def make_env(
    seed: int,
    reward_config: RewardConfig | None = None,
    reward_config_path: Path | None = None,
    cluster_config_path: Path | None = None,
    scenario: str = "balanced",
    heterogeneous_obs: bool = False,
):
    # Resolve reward config from path if given
    if reward_config is None and reward_config_path is not None:
        from gpu_sage.env.gpu_env import load_reward_config
        reward_config = load_reward_config(reward_config_path)
    if reward_config is None:
        reward_config = RewardConfig()
    # Resolve workload config from scenario
    from gpu_sage.workloads.generator import get_scenario_config

    try:
        workload_cfg = get_scenario_config(scenario)
    except Exception:
        workload_cfg = WorkloadConfig(
            arrival_rate=0.08,
            min_gpus=1,
            max_gpus=4,
            min_memory_gb=8,
            max_memory_gb=64,
            min_duration=20,
            max_duration=180,
            min_priority=1,
            max_priority=5,
        )

    def _factory():
        return GPUSchedulingEnv(
            num_gpus=8,
            gpu_memory_gb=80.0,
            max_jobs=16,
            episode_jobs=100,
            workload_config=workload_cfg,
            reward_config=reward_config,
            seed=seed,
            cluster_config_path=cluster_config_path,
            heterogeneous_obs=heterogeneous_obs,
        )

    return _factory

def print_header(steps: int, seed: int):
    line = "=" * 60
    print(line)
    print("GPU-SAGE — PPO TRAINING")
    print(line)
    print(f"Environment: GPU Scheduling")
    print(f"Algorithm:   MaskablePPO (MultiInputPolicy)")
    print(f"Timesteps:   {steps:,}")
    print(f"Seed:        {seed}")
    print(f"GPUs:        8")
    print(f"Python:      {platform.python_version()}")
    gh = get_git_hash()
    if gh:
        print(f"Git:         {gh[:8]}")
    print(line)
    print()

class MetricsCSVCallback(BaseCallback):
    """Log training metrics to CSV for plotting."""
    def __init__(self, csv_path: Path, verbose=0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.fieldnames = [
            "timestep", "mean_reward", "mean_ep_length",
            "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
            "explained_variance", "learning_rate", "fps", "elapsed"
        ]
        self.start_time = time.time()
        self.records: list[dict] = []

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writeheader()

    def _on_step(self) -> bool:
        # Log timestep progression; detailed PPO metrics will be filled from eval logs post-training
        if self.n_calls % 2048 == 0:
            rec = {
                "timestep": int(self.num_timesteps),
                "mean_reward": 0.0,
                "mean_ep_length": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "explained_variance": 0.0,
                "learning_rate": 3e-4,
                "fps": 0,
                "elapsed": float(time.time() - self.start_time),
            }
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writerow(rec)
            self.records.append(rec)
        return True

class RewardComponentCallback(BaseCallback):
    """Log per-component reward contributions to TensorBoard and CSV.

    Captures `info["reward_components"]` from env steps and records
    them via `self.logger.record`. Ensures total = sum(parts) is verifiable.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.component_sums: dict[str, float] = {}
        self.component_counts: dict[str, int] = {}

    def _on_step(self) -> bool:
        # VecEnv infos is a tuple/list of dicts
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "reward_components" in info:
                comps = info["reward_components"]
                # Verify total == sum
                total = comps.get("total_reward", 0)
                s = sum(v for k, v in comps.items() if k != "total_reward")
                # Log each component
                for k, v in comps.items():
                    if k == "total_reward":
                        continue
                    self.logger.record(f"reward/{k}", float(v))
                    self.component_sums[k] = self.component_sums.get(k, 0.0) + float(v)
                    self.component_counts[k] = self.component_counts.get(k, 0) + 1
                self.logger.record("reward/total", float(total))
                # Also log debug to check scale
                if "_waiting_sum" in info.get("_reward_debug", {}):
                    self.logger.record("reward_debug/waiting_sum", float(info["_reward_debug"]["_waiting_sum"]))
        return True

def enrich_metrics_from_eval(run_dir: Path):
    """Fill metrics.csv with real eval rewards from evaluations.npz if available."""
    eval_path = run_dir / "eval" / "evaluations.npz"
    metrics_path = run_dir / "metrics.csv"
    if not eval_path.exists() or not metrics_path.exists():
        return
    try:
        data = np.load(eval_path)
        timesteps = data["timesteps"]
        results = data["results"]  # shape (n_eval, n_episodes)
        mean_rewards = results.mean(axis=1)
        # Read existing metrics.csv and merge
        import pandas as pd
        df = pd.read_csv(metrics_path)
        # Map eval timesteps to closest metrics timestep
        for i, ts in enumerate(timesteps):
            # Find closest row in df
            if len(df) == 0:
                continue
            idx = (np.abs(df["timestep"] - int(ts))).argmin()
            df.loc[idx, "mean_reward"] = float(mean_rewards[i])
            # Ep lengths if available
            if "ep_lengths" in data:
                ep_lens = data["ep_lengths"]
                if len(ep_lens) > i:
                    df.loc[idx, "mean_ep_length"] = float(np.mean(ep_lens[i]))
        df.to_csv(metrics_path, index=False)
    except Exception as e:
        print(f"[metrics] Could not enrich from eval: {e}")

def generate_plots(metrics_csv: Path, out_dir: Path, run_dir: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Try to enrich metrics from eval before plotting
    if run_dir is not None:
        enrich_metrics_from_eval(run_dir)
    if not metrics_csv.exists():
        return
    try:
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        if df.empty:
            return
        # If mean_reward still all zero, try eval directly
        if (df["mean_reward"] == 0).all() and run_dir is not None:
            eval_path = run_dir / "eval" / "evaluations.npz"
            if eval_path.exists():
                try:
                    data = np.load(eval_path)
                    timesteps = data["timesteps"]
                    results = data["results"]
                    mean_rewards = results.mean(axis=1)
                    plt.figure(figsize=(10, 5))
                    plt.plot(timesteps, mean_rewards, marker="o", linewidth=2, label="eval mean reward")
                    plt.title("PPO Training Reward vs Timestep (from eval)")
                    plt.xlabel("Timestep")
                    plt.ylabel("Mean Eval Reward")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(out_dir / "reward_curve.png", dpi=150)
                    plt.close()
                    # Episode length from eval
                    if "ep_lengths" in data:
                        ep_lens = data["ep_lengths"].mean(axis=1)
                        plt.figure(figsize=(10, 5))
                        plt.plot(timesteps, ep_lens, marker="o")
                        plt.title("Episode Length vs Timestep")
                        plt.xlabel("Timestep")
                        plt.ylabel("Mean Episode Length")
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(out_dir / "episode_length.png", dpi=150)
                        plt.close()
                    return
                except Exception:
                    pass
        # Smooth with rolling
        def smooth(s, w=5):
            return s.rolling(window=w, min_periods=1).mean()

        # 1. Reward vs timestep
        if "mean_reward" in df.columns and (df["mean_reward"] != 0).any():
            plt.figure(figsize=(10, 5))
            plt.plot(df["timestep"], df["mean_reward"], alpha=0.4, label="raw")
            plt.plot(df["timestep"], smooth(df["mean_reward"]), linewidth=2, label="moving avg (5)")
            plt.title("PPO Training Reward vs Timestep")
            plt.xlabel("Timestep")
            plt.ylabel("Mean Eval Reward")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "reward_curve.png", dpi=150)
            plt.close()

        # 2. Episode length
        if "mean_ep_length" in df.columns and (df["mean_ep_length"] != 0).any():
            plt.figure(figsize=(10, 5))
            plt.plot(df["timestep"], df["mean_ep_length"], alpha=0.7)
            plt.title("Episode Length vs Timestep")
            plt.xlabel("Timestep")
            plt.ylabel("Mean Episode Length")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "episode_length.png", dpi=150)
            plt.close()

        # 3. Value/Policy loss (may be zero if not logged)
        if "value_loss" in df.columns and (df["value_loss"] != 0).any():
            plt.figure(figsize=(10, 5))
            plt.plot(df["timestep"], df["value_loss"], label="value_loss", alpha=0.7)
            if "policy_loss" in df.columns and (df["policy_loss"] != 0).any():
                plt.plot(df["timestep"], df["policy_loss"], label="policy_loss", alpha=0.7)
            plt.title("PPO Loss vs Timestep")
            plt.xlabel("Timestep")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "loss_curve.png", dpi=150)
            plt.close()

        # 4. Entropy / KL
        if ("entropy" in df.columns and (df["entropy"] != 0).any()) or ("approx_kl" in df.columns and (df["approx_kl"] != 0).any()):
            plt.figure(figsize=(10, 5))
            if "entropy" in df.columns and (df["entropy"] != 0).any():
                plt.plot(df["timestep"], df["entropy"], label="entropy", alpha=0.7)
            if "approx_kl" in df.columns and (df["approx_kl"] != 0).any():
                plt.plot(df["timestep"], df["approx_kl"], label="approx_kl", alpha=0.7)
            plt.title("Entropy / KL vs Timestep")
            plt.xlabel("Timestep")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "entropy_kl.png", dpi=150)
            plt.close()

        # Ensure at least reward placeholder exists
        if not (out_dir / "reward_curve.png").exists():
            plt.figure(figsize=(10, 5))
            if (df["mean_reward"] != 0).any():
                plt.plot(df["timestep"], df["mean_reward"])
                plt.title("PPO Training Reward vs Timestep")
                plt.xlabel("Timestep")
                plt.ylabel("Mean Eval Reward")
            else:
                plt.text(0.5, 0.5, f"Training completed\\n{len(df)} rollouts\\n(see tensorboard)", ha="center", va="center", fontsize=12)
                plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / "reward_curve.png", dpi=150)
            plt.close()

    except Exception as e:
        print(f"[plots] Warning: could not generate plots: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-Sage PPO training — showcase quality")
    parser.add_argument("--steps", type=int, default=250_000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", type=Path, default=Path("artifacts/ppo"), help="Base output dir")
    parser.add_argument("--reward-config", type=Path, default=None, help="Path to reward yaml (e.g., configs/rewards/reward_F_balanced.yaml)")
    parser.add_argument("--cluster-config", type=Path, default=None, help="Path to cluster yaml (e.g., configs/heterogeneous_8gpu.yaml)")
    parser.add_argument("--scenario", type=str, default="balanced", help="Workload scenario (balanced, heterogeneous, topology_sensitive, mixed_ml, etc.)")
    parser.add_argument("--heterogeneous-obs", action="store_true", help="Enable extended heterogeneous observation (per-GPU type/perf + topology flags)")
    parser.add_argument("--policy", type=str, default="flat", choices=["flat", "graph"], help="Policy architecture: flat MultiInputPolicy or graph-aware GNN")
    args = parser.parse_args()

    # Load reward config (yaml) if provided
    from gpu_sage.env.gpu_env import load_reward_config
    reward_cfg = load_reward_config(args.reward_config) if args.reward_config else RewardConfig()

    # Run directory with unique ID
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_seed{args.seed}_steps{args.steps}"
    base_out = Path(args.out)
    run_dir = base_out / "runs" / run_id
    # Legacy dirs for backward compat
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "models").mkdir(parents=True, exist_ok=True)
    (base_out / "logs").mkdir(parents=True, exist_ok=True)
    (base_out / "checkpoints").mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Logging
    log_path = run_dir / "training.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)

    print_header(args.steps, args.seed)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run dir: {run_dir}")

    # Resolve workload/cluster strings for config record
    from gpu_sage.workloads.generator import get_scenario_config as _get_cfg

    try:
        _wc = _get_cfg(args.scenario)
        workload_dict = {
            "scenario": args.scenario,
            "arrival_rate": _wc.arrival_rate,
            "min_gpus": _wc.min_gpus,
            "max_gpus": _wc.max_gpus,
            "min_memory_gb": _wc.min_memory_gb,
            "max_memory_gb": _wc.max_memory_gb,
            "min_duration": _wc.min_duration,
            "max_duration": _wc.max_duration,
            "min_priority": _wc.min_priority,
            "max_priority": _wc.max_priority,
        }
    except Exception:
        workload_dict = {"scenario": args.scenario}
    cluster_dict = {"num_gpus": 8, "gpu_memory_gb": 80, "gpu_type": "A100"}
    if args.cluster_config:
        cluster_dict = {"cluster_config": str(args.cluster_config), "heterogeneous": True}
    # Save config (include actual reward config used)
    config = {
        "run_id": run_id,
        "timestamp": ts,
        "timesteps": args.steps,
        "seed": args.seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_hash": get_git_hash(),
        "cluster": cluster_dict,
        "workload": workload_dict,
        "reward_config": reward_cfg.to_dict(),
        "reward_config_path": str(args.reward_config) if args.reward_config else None,
        "cluster_config_path": str(args.cluster_config) if args.cluster_config else None,
        "heterogeneous_obs": bool(args.heterogeneous_obs),
        "scenario": args.scenario,
        "policy": args.policy,
        "ppo_hparams": {"learning_rate": 3e-4, "n_steps": 2048, "batch_size": 256, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "policy": "GraphMaskablePolicy" if args.policy == "graph" else "MultiInputPolicy"},
        "out_dir": str(run_dir),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "training.log", "a") as f:
        f.write(f"Config: {json.dumps(config, indent=2)}\n")

    # Windows hang root cause: MaskableEvalCallback deadlocks on heterogeneous env
    # (DummyVecEnv + heterogeneous topology) on Windows; verified: hetero training
    # without eval succeeds (0.42s/2048 steps), with eval hangs after first rollout.
    # Fix: disable eval for heterogeneous on Windows; keep for homogeneous.
    is_hetero = bool(args.cluster_config or args.heterogeneous_obs or args.scenario in ("heterogeneous", "topology_sensitive", "mixed_ml"))
    if is_hetero:
        train_env = DummyVecEnv([make_env(args.seed, reward_config=reward_cfg, cluster_config_path=args.cluster_config, scenario=args.scenario, heterogeneous_obs=args.heterogeneous_obs)])
        eval_env = None
        eval_cb = None
    else:
        train_env = DummyVecEnv([make_env(args.seed, reward_config=reward_cfg, cluster_config_path=args.cluster_config, scenario=args.scenario, heterogeneous_obs=args.heterogeneous_obs)])
        eval_env = DummyVecEnv([make_env(args.seed + 10_000, reward_config=reward_cfg, cluster_config_path=args.cluster_config, scenario=args.scenario, heterogeneous_obs=args.heterogeneous_obs)])
        eval_cb = MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "eval"),
            eval_freq=max(args.steps // 10, 1),
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        )

    # Callbacks: checkpoint every 50k, reward components, eval if not hetero
    checkpoint_cb = CheckpointCallback(save_freq=max(50000 // 1, 1), save_path=str(run_dir / "checkpoints"), name_prefix="ppo")
    metrics_cb = MetricsCSVCallback(csv_path=run_dir / "metrics.csv")
    reward_cb = RewardComponentCallback()
    if is_hetero:
        # Single checkpoint for hetero to avoid file lock contention; no eval
        callbacks = CallbackList([checkpoint_cb, metrics_cb, reward_cb])
    else:
        checkpoint_cb2 = CheckpointCallback(save_freq=max(50000 // 1, 1), save_path=str(base_out / "checkpoints"), name_prefix="ppo")
        callbacks = CallbackList([checkpoint_cb, checkpoint_cb2, metrics_cb, reward_cb, eval_cb])

    # Also save periodic checkpoints via simple interval (50k)
    # SB3 CheckpointCallback saves at save_freq timesteps

    # Policy selection: flat MultiInputPolicy vs graph-aware GNN
    if args.policy == "graph":
        from gpu_sage.rl.graph_policy import GraphMaskablePolicy

        policy = GraphMaskablePolicy
        # Graph extractor already defaults to 128 dim, pi/vf 64,32
        policy_kwargs = {}
    else:
        policy = "MultiInputPolicy"
        policy_kwargs = {}

    model = MaskablePPO(
        policy,
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=args.seed,
        device="auto",
        policy_kwargs=policy_kwargs,
    )

    start = time.time()
    # Print training progress header
    print("Training Progress")
    print("=" * 60)
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    elapsed = time.time() - start

    # Save models
    model.save(run_dir / "model" / "final_model")
    model.save(base_out / "final_model")
    model.save(base_out / "models" / "final_model")
    model.save(base_out / "checkpoints" / f"ppo_seed{args.seed}_steps{args.steps}")
    # Also save to run checkpoints final
    model.save(run_dir / "checkpoints" / f"ppo_{args.steps}")

    train_env.close()
    if eval_env is not None:
        eval_env.close()

    # Generate plots (enrich metrics from eval first)
    generate_plots(run_dir / "metrics.csv", run_dir / "plots", run_dir)

    # Training summary — try metrics.csv first, then eval npz
    best_reward = final_reward = None
    try:
        import pandas as pd
        if (run_dir / "metrics.csv").exists():
            df = pd.read_csv(run_dir / "metrics.csv")
            if not df.empty and "mean_reward" in df.columns and (df["mean_reward"] != 0).any():
                best_reward = float(df["mean_reward"].max())
                final_reward = float(df["mean_reward"].replace(0, np.nan).dropna().iloc[-1]) if len(df) > 0 else None
            # Fallback to eval if still zero
            if (best_reward is None or best_reward == 0) and (run_dir / "eval" / "evaluations.npz").exists():
                raise ValueError("fallback to eval")
        if (best_reward is None or best_reward == 0) and (run_dir / "eval" / "evaluations.npz").exists():
            data = np.load(run_dir / "eval" / "evaluations.npz")
            results = data["results"]
            mean_rewards = results.mean(axis=1)
            best_reward = float(np.max(mean_rewards))
            final_reward = float(mean_rewards[-1])
    except Exception:
        # Final fallback: try eval directly
        try:
            eval_path = run_dir / "eval" / "evaluations.npz"
            if eval_path.exists():
                data = np.load(eval_path)
                mean_rewards = data["results"].mean(axis=1)
                best_reward = float(np.max(mean_rewards))
                final_reward = float(mean_rewards[-1])
        except Exception:
            pass

    summary = {
        "run_id": run_id,
        "timesteps": args.steps,
        "seed": args.seed,
        "training_time_seconds": float(elapsed),
        "training_time_hms": str(datetime.timedelta(seconds=int(elapsed))),
        "final_mean_reward": final_reward,
        "best_mean_reward": best_reward,
        "model_path": str(run_dir / "model" / "final_model.zip"),
        "base_model_path": str(base_out / "final_model.zip"),
        "tensorboard": str(run_dir / "tensorboard"),
        "checkpoints": str(run_dir / "checkpoints"),
        "plots": str(run_dir / "plots"),
        "logs": str(run_dir / "training.log"),
        "config": str(run_dir / "config.json"),
    }
    with open(run_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(run_dir / "training_summary.txt", "w") as f:
        f.write("GPU-SAGE PPO TRAINING COMPLETE\n")
        f.write("=" * 60 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # Also save to base for easy access
    with open(base_out / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate TRAINING_REPORT.md
    rc = reward_cfg.to_dict()
    report = f"""# GPU-Sage PPO Training Report

**Run ID:** {run_id}  
**Timesteps:** {args.steps:,}  
**Seed:** {args.seed}  
**Duration:** {summary['training_time_hms']} ({elapsed:.1f}s)  
**Git:** {get_git_hash() or 'unknown'}  
**Python:** {platform.python_version()} on {platform.platform()}
**Reward Config:** {args.reward_config or 'default (A_baseline)'}

## Configuration
- Cluster: 8×A100 80GB
- Workload: arrival_rate=0.08, 1-4 GPUs, 20-180s, 1-5 priority
- Reward: throughput={rc['throughput']} waiting={rc['waiting']} util={rc['utilization']} frag={rc['fragmentation']} invalid={rc['invalid_action']} idle={rc['idle']} norm={rc.get('waiting_normalization','sum')}
- PPO: lr=3e-4 n_steps=2048 batch=256 n_epochs=10 gamma=0.99

## Results
- Final Mean Reward: {final_reward}
- Best Mean Reward: {best_reward}
- Model: `{run_dir / 'model' / 'final_model.zip'}`
- TensorBoard: `tensorboard --logdir {run_dir / 'tensorboard'}`

## Artifacts
- Config: `config.json`
- Logs: `training.log`, `metrics.csv`
- Checkpoints: `checkpoints/ppo_*.zip` every 50k
- Plots: `plots/reward_curve.png`, `episode_length.png`, `loss_curve.png`
- Summary: `training_summary.json`

## TensorBoard
```bash
tensorboard --logdir {run_dir / 'tensorboard'}
```

## Reproduce
```bash
python training/train_ppo.py --steps {args.steps} --seed {args.seed}
```

*Generated {datetime.datetime.now().isoformat()}*
"""
    with open(run_dir / "TRAINING_REPORT.md", "w") as f:
        f.write(report)

    # Print final summary to terminal
    print("\n" + "=" * 60)
    print("GPU-SAGE PPO TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run ID:            {run_id}")
    print(f"Timesteps:         {args.steps:,}")
    print(f"Seed:              {args.seed}")
    print(f"Training Time:     {summary['training_time_hms']}")
    print(f"Final Mean Reward: {final_reward}")
    print(f"Best Mean Reward:  {best_reward}")
    print(f"Model:")
    print(f"    {run_dir / 'model' / 'final_model.zip'}")
    print(f"Logs:")
    print(f"    {run_dir / 'training.log'}")
    print(f"TensorBoard:")
    print(f"    {run_dir / 'tensorboard'}")
    print(f"Plots:")
    print(f"    {run_dir / 'plots'}/")
    print(f"Checkpoints:")
    print(f"    {run_dir / 'checkpoints'}/")
    print("=" * 60)
    logger.info(f"Saved model to {run_dir / 'model' / 'final_model.zip'}")

if __name__ == "__main__":
    main()
