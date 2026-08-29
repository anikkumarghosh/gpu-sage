"""Train the first GPU-Sage PPO scheduler."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Windows OpenMP duplicate lib workaround for torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from gpu_sage.env.gpu_env import GPUSchedulingEnv, RewardConfig
from gpu_sage.workloads.generator import WorkloadConfig


def make_env(seed: int):
    def _factory():
        return GPUSchedulingEnv(
            num_gpus=8,
            gpu_memory_gb=80.0,
            max_jobs=16,
            episode_jobs=100,
            workload_config=WorkloadConfig(
                arrival_rate=0.08,
                min_gpus=1,
                max_gpus=4,
                min_memory_gb=8,
                max_memory_gb=64,
                min_duration=20,
                max_duration=180,
                min_priority=1,
                max_priority=5,
            ),
            reward_config=RewardConfig(
                throughput=2.0,
                waiting=0.02,
                utilization=0.25,
                fragmentation=0.25,
                invalid_action=1.0,
                idle=0.01,
            ),
            seed=seed,
        )

    return _factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ppo"))
    args = parser.parse_args()

    # Ensure standard artifact layout: models/, logs/, checkpoints/ + legacy paths
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "models").mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env(args.seed)])
    eval_env = DummyVecEnv([make_env(args.seed + 10_000)])

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(args.out / "best"),
        log_path=str(args.out / "eval"),
        eval_freq=max(args.steps // 10, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = MaskablePPO(
        "MultiInputPolicy",
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
        tensorboard_log=str(args.out / "tensorboard"),
        seed=args.seed,
        device="auto",
    )

    model.learn(total_timesteps=args.steps, callback=eval_callback, progress_bar=True)
    # Save to both legacy and new structured locations
    model.save(args.out / "final_model")
    model.save(args.out / "models" / "final_model")
    # Also save a seed-specific checkpoint for reproducibility
    model.save(args.out / "checkpoints" / f"ppo_seed{args.seed}_steps{args.steps}")

    train_env.close()
    eval_env.close()
    print(f"Saved model to {args.out / 'final_model.zip'} and {args.out / 'models' / 'final_model.zip'}")


if __name__ == "__main__":
    main()
