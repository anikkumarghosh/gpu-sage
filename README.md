# GPU-Sage

An event-driven simulator and reinforcement-learning benchmark for intelligent GPU cluster scheduling.

## Install

```bash
pip install -e "./[rl,dev]"
```

## Quickstart

```bash
python scripts/demo_simulation.py
python scripts/check_environment.py
```

## Baseline vs PPO Evaluation

This milestone adds a scientifically valid **PPO vs FCFS vs SJF vs Priority vs Best-Fit** comparison.

### Schedulers

- **FCFS**, **SJF**, **Priority**, **Best-Fit** — heuristic baselines via `Simulator` + `Scheduler.select`
- **PPO** — `MaskablePPO` (`MultiInputPolicy`) via `GPUSchedulingEnv` with deterministic inference (`deterministic=True`)

All 5 share the **identical workload** per seed/scenario.

### Identical Workloads & Deterministic Evaluation

For each `scenario × seed` the benchmark generates workload **W0 once** (seeded `SyntheticWorkload`), then runs every scheduler on a `deepcopy(W0)` from a fresh simulator/env. PPO does **not** generate its own workload during evaluation — it receives `fixed_jobs=W0` via `env.reset(options={"fixed_jobs": W0})`. Training still generates stochastic workloads; evaluation is locked to W0. PPO inference is deterministic; no learning occurs during evaluation.

### Workload Scenarios

`balanced` (1–4 GPUs, mixed), `gpu_heavy` (4–8 GPUs, long), `short_jobs` (1–2 GPUs, short), `bursty` (burst/quiet periods), `heavy_tail` (many short + few very long), `priority_skew` (mostly low priority + occasional high).

### Metrics (same schema for all schedulers)

`completed_jobs`, `throughput = completed/sim_time`, `average/median/P95 waiting_time = start-arrival`, `average/median/P95 JCT = completion-arrival`, `gpu_utilization = busy_time / (gpus*sim_time)`, `gpu_idle_time = 1-util`, `resource_efficiency = util`, `rejected/infeasible`, `scheduling_decisions`, `invalid_attempts`, `Jain's fairness = (sum x)^2 / (n sum x^2)` on waiting times. Utilization is integrated over time, not snapshot.

Per-job records (`job_id, arrival/start/completion, waiting/turnaround, gpu/memory, duration, priority, scheduler`) and, for PPO, per-decision logs (`simulation_time, queue_length, selected_job_id, action, reward, free_gpus, gpu_utilization`) are saved for later analysis.

### How to Train PPO

```bash
python training/train_ppo.py --steps 250000 --seed 0
# saves to artifacts/ppo/{final_model.zip, models/, checkpoints/, best/, eval/, tensorboard/}
# also supports --out custom_dir
```

Uses `MaskablePPO` with `n_steps=2048`, `batch_size=256`, `gamma=0.99`, etc. Set `KMP_DUPLICATE_LIB_OK=TRUE` on Windows if needed. Large models under `artifacts/ppo/` are not committed.

### How to Run Evaluation

```bash
# PPO alone on same W0
python scripts/benchmark_ppo.py --model artifacts/ppo/models/final_model.zip --scenario balanced --seed 0 --jobs 50

# Unified 5-way comparison (mean ± std)
python scripts/benchmark_all.py --scenario balanced --seeds 0 1 2 3 4 --ppo-model artifacts/ppo/models/final_model.zip
python scripts/benchmark_all.py --all --seeds 0 1 2 --ppo-model artifacts/ppo/models/final_model.zip --jobs 20

# Baselines only (also via benchmark_baselines.py)
python scripts/benchmark_baselines.py --scenario balanced --seeds 0 1 2
python scripts/benchmark_baselines.py --scenario balanced --seeds 0 1 2 --ppo-model artifacts/ppo/models/final_model.zip
```

Results under `artifacts/benchmarks/`:
`balanced.csv` (per-seed metrics), `balanced_agg.csv` (mean±std), `balanced.json` (repro metadata: scheduler/scenario/seed/gpus/jobs/workload_config), `balanced_jobs.csv/json` (per-job), `balanced_ppo_decisions.csv/json` (PPO steps).

### Important Design Choice

A scheduling action never manipulates GPUs directly — the **simulator owns** time, arrivals, completions, allocation/release. This lets the exact same engine evaluate heuristics and PPO under identical workloads.
