# GPU-Sage PPO Training Report

**Run ID:** 20260829_184450_seed0_steps250000  
**Timesteps:** 250,000  
**Seed:** 0  
**Duration:** 0:05:06 (306.5s)  
**Git:** 38b7c3260939d47df87df646f0eaa565b04ff29b  
**Python:** 3.13.9 on Windows-11-10.0.26200-SP0

## Configuration
- Cluster: 8×A100 80GB
- Workload: arrival_rate=0.08, 1-4 GPUs, 20-180s, 1-5 priority
- Reward: throughput=2.0 waiting=0.02 util=0.25 frag=0.25 invalid=1.0 idle=0.01
- PPO: lr=3e-4 n_steps=2048 batch=256 n_epochs=10 gamma=0.99

## Results
- Final Mean Reward: -93454.71788823605
- Best Mean Reward: -87060.94024312496
- Model: `artifacts\ppo\runs\20260829_184450_seed0_steps250000\model\final_model.zip`
- TensorBoard: `tensorboard --logdir artifacts\ppo\runs\20260829_184450_seed0_steps250000\tensorboard`

## Artifacts
- Config: `config.json`
- Logs: `training.log`, `metrics.csv`
- Checkpoints: `checkpoints/ppo_*.zip` every 50k
- Plots: `plots/reward_curve.png`, `episode_length.png`, `loss_curve.png`
- Summary: `training_summary.json`

## TensorBoard
```bash
tensorboard --logdir artifacts\ppo\runs\20260829_184450_seed0_steps250000\tensorboard
```

## Reproduce
```bash
python training/train_ppo.py --steps 250000 --seed 0
```

*Generated 2026-08-29T18:49:59.815764*
