# GPU-Sage

An event-driven simulator and reinforcement-learning benchmark for intelligent GPU cluster scheduling.

## Current milestone

**Milestone 1: deterministic simulator core**

- GPU resource model
- ML job model
- FIFO waiting queue
- discrete-event simulation
- deterministic resource allocation/release
- basic FCFS scheduler

RL and Gymnasium integration will be added after the simulator and baseline behavior are validated.

## Run

```bash
python scripts/demo_simulation.py
```


## Milestone 2: Gymnasium + PPO environment

The repository now contains an event-driven Gymnasium environment in `src/gpu_sage/env/gpu_env.py`. The agent chooses one waiting-job candidate per decision or takes a `NOOP` action to advance to the next event. The environment exposes structured observations for the cluster, GPUs, and candidate jobs, plus an action mask for MaskablePPO.

### Install

```bash
pip install -e "./[rl,dev]"
```

### Validate the environment

```bash
python scripts/check_environment.py
```

### Train PPO

```bash
python training/train_ppo.py --steps 250000
```

The first training run uses `MaskablePPO` with `MultiInputPolicy`. Model checkpoints and TensorBoard logs are written under `artifacts/ppo/`.

### Important design choice

A scheduling action does not directly manipulate GPUs. The simulator owns resource allocation. This separation lets the exact same simulation engine evaluate FCFS/SJF/Priority/Best-Fit and the RL policy under identical workloads.
