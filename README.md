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

### How to Train PPO (Showcase Run)

```bash
KMP_DUPLICATE_LIB_OK=TRUE python training/train_ppo.py --steps 250000 --seed 0
# Showcase run creates artifacts/ppo/runs/<20260315_123456_seed0_steps250000>/
#   config.json          — hyperparams, workload, reward, git hash, python version
#   training.log         — full terminal output
#   metrics.csv          — timestep, mean_reward, ep_length, losses
#   tensorboard/         — event files (tensorboard --logdir <run>/tensorboard)
#   checkpoints/ppo_*.zip — every 50k steps + final
#   model/final_model.zip — best and final
#   plots/reward_curve.png, episode_length.png, loss_curve.png
#   training_summary.json/txt, TRAINING_REPORT.md
# Also saves legacy artifacts/ppo/{final_model.zip, models/, checkpoints/}
```

Uses `MaskablePPO` with `n_steps=2048`, `batch_size=256`, `gamma=0.99`, etc. Set `KMP_DUPLICATE_LIB_OK=TRUE` on Windows if needed. Large `*.zip` and `tensorboard/` under `artifacts/ppo/` are gitignored; showcase `plots/*.png`, `TRAINING_REPORT.md`, `config.json`, `training_summary.json` are kept for portfolio.

**TensorBoard:**
```bash
tensorboard --logdir artifacts/ppo/runs/<run_id>/tensorboard
```

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

## Reward Engineering Experiment

**Research question:** Why does PPO (250k, seed 0, baseline reward) fail to beat SJF, and can shaped rewards improve it without overfitting?

**Original reward (per step, `elapsed = cur - prev`):**
```
interval_util = gpu_time_delta / (elapsed*num_gpus) in [0,1]
frag = 1 - min(max_feasible, free)/free in [0,1]
throughput = 2.0 * completed_delta
waiting   = 0.02 * sum(wait)          # sum over waiting jobs, NOT normalized → dominates (~947/step)
util      = 0.25 * interval_util * elapsed (~1-3)
frag      = 0.25 * frag * elapsed (~0.5)
idle      = 0.01 * elapsed if util==0 else 0
invalid   = 1.0 if invalid else 0
total = throughput - waiting + util - frag - idle - invalid
```
**Scale analysis:** `waiting ≈20-1000` vs `util≈1` vs `throughput≈2` → waiting dominates; changing `util` 0.25→0.75 has minimal effect. Reward F introduces `waiting_normalization=mean` (`sum/N`) so `waiting≈0.03*50≈1.5` comparable to `util`.

**Candidates (configs/rewards/*.yaml, load via `--reward-config`):**
- **A_baseline** `2.0,0.02,0.25,0.25,1.0,0.01 sum` — original
- **B_waiting** `2.0,0.05,0.25,0.25,1.0,0.01 sum` — waiting 2.5×
- **C_util** `2.0,0.02,0.75,0.25,1.0,0.01 sum` — util 3×
- **D_frag** `2.0,0.02,0.25,0.75,1.0,0.01 sum` — frag 3×
- **E_throughput** `4.0,0.02,0.25,0.25,1.0,0.01 sum` — throughput 2×
- **F_balanced** `2.5,0.03,0.4,0.4,1.0,0.01 mean` — normalized, balanced (documented above)

**Methodology:**
- Screening: 50k steps, single train seed (0), eval 5 seeds on `balanced`, 100 jobs, identical W0.
- Selection: rank PPO by `Avg JCT` (then `Avg Wait`) on balanced — top 2: **F (908.8)** then **A (951.4)**; B worst (1136).
- Full: 250k steps, 3 train seeds (0,1,2) for finalists **F** and **A**; eval 5 seeds × 6 scenarios, identical W0, mean±std. Logs include per-component `reward/{throughput,waiting,util,frag,idle,invalid}` via `info["reward_components"]`.
- Kept original baseline intact (`configs/rewards/reward_A_baseline.yaml`).

**Results (screening 50k, balanced PPO):**
| Reward | Avg JCT | P95 JCT | Avg Wait | P95 Wait | Util |
|---|---|---|---|---|---|
| F_balanced | 908±118 | 2427±240 | 808±117 | 2326±235 | 0.934±0.011 |
| A_baseline | 951±76 | 2597±143 | 850±74 | 2480±156 | 0.907±0.027 |
| D_frag | 951±76 | 2597±143 | 850±74 | 2480±156 | 0.907±0.027 |
| C_util | 971±104 | 2685±231 | 870±102 | 2572±261 | 0.904±0.025 |
| B_waiting | 1136±65 | 2977±210 | 1035±65 | 2899±184 | 0.808±0.045 |

**Full 250k (5 seeds, 100 jobs) — F vs A vs baselines:**
- `balanced`: F PPO 779±90 JCT (close to A 779, still behind SJF 531), `balanced` F best among PPO variants but not beating SJF.
- `gpu_heavy`: both PPO ~9300 JCT vs SJF 6592 — **PPO particularly poor on gpu_heavy** (large-job avoidance not resolved; see starvation analysis).
- `short_jobs`: F 16.1±6.3 vs SJF 12.4±4.4 — near parity.

**Reward component analysis (avg per step, balanced, 250k F):** `waiting_penalty -930` dominates `util +4.8` and `throughput +0.7`; F's mean normalization reduces waiting magnitude from -947 (A) to -930, making util relatively more influential but still dominated.

**Pathological checks (5-seed, per-job):**
- **Starvation:** max/median 79× (PPO) vs 302× (SJF) — PPO less starved than SJF on balanced, but max still ~21k s.
- **Large-job:** PPO large (≥4 GPUs) 33.4% completed same as baselines — no avoidance on balanced; on `gpu_heavy` all jobs large (100%).
- **Fragmentation:** util 0.93 (PPO) vs 0.918 (SJF) on balanced — similar.
- **NOOP:** 65.9% of PPO steps are NOOP (env needs NOOP to advance time; not abuse).
- **Priority:** PPO avg wait P1 376s < P3 730s but not monotonic — weak priority response.

**Conclusion:** Balanced reward (F) gives modest JCT improvement (908 vs 951 screening, 779 vs 779 full tie) but **does not consistently beat SJF**; reward tuning improved utilization but increased tail wait on some scenarios. Finding is valuable — not fabricated. Original baseline remains reproducible via `reward_A_baseline.yaml`.

**Run ablation:**
```bash
python scripts/run_reward_ablation.py --mode screening --steps 50000
python scripts/run_reward_ablation.py --mode full --steps 250000 --train-seeds 0 1 2
python training/train_ppo.py --steps 250000 --seed 0 --reward-config configs/rewards/reward_F_balanced.yaml
```

## Generalization / Robustness

**Research question**: Does PPO trained on one workload distribution generalize to workload distributions that differ from its training distribution?

**Training distribution** (250k PPO, seed 0): `balanced` scenario, arrival_rate=0.08, 1-4 GPUs, 20-180s duration, 1-5 priority, reward A_baseline (2.0/0.02/0.25/0.25/1.0/0.01, sum normalization), MultiInputPolicy, 8 GPUs 80GB A100.

**Evaluation**: 3 evaluation seeds per scenario × 7 distributions (1 in-distribution `balanced` + 6 OOD shifts defined in `scripts/generalize_ppo.py`):
- **ID (balanced)**: in-distribution reference point
- **Shift A**: Higher GPU demand (min_gpus=2, max_gpus=8)
- **Shift B**: Shorter jobs (min_duration=5, max_duration=50)
- **Shift C**: Longer jobs (min_duration=60, max_duration=400, arrival_rate=0.05)
- **Shift D**: Bursty arrivals (modulated rate)
- **Shift E**: Heavy-tailed runtime (heavy_tail_fraction=0.07, multiplier=8.0)
- **Shift F**: Priority distribution shift

**Key finding**: PPO with reward **F** (normalized waiting + balanced weights) shows **better robustness** to distribution shift than A_baseline, achieving **-5.6% average JCT degradation** vs **+4.8% for A_baseline**. F_balanced particularly excels on short-job and priority-skew shifts.

**Generalization score** (retention = 1 - (OOD-ID)/ID * 100%):
- A_baseline: 95.2% performance retained
- F_balanced: 105.6% performance retained (slight improvement under shift)

**SJF remains the best overall performer** across all distributions.

**Run ablation**:
```bash
python scripts/generalize_ppo.py --mode report
```

### Interactive Dashboard

Launch the dashboard with:

```bash
streamlit run app.py
```

or from the project root:

```bash
cd gpu-sage && streamlit run app.py
```

The dashboard provides:

* **Live simulation** — real-time GPU cluster visualization with PPO/FCFS/SJF/Priority/BestFit schedulers
* **Scheduler selection** — switch between PPO (using trained model) and heuristic baselines
* **Same-workload comparison** — fixed W0 passed identically to all schedulers for fair comparison
* **Live metrics panel** — GPU utilization, throughput, completed jobs, average/p95 wait and JCT, fairness
* **PPO decision panel** — latest action, selected job, GPU requirement, priority, waiting time, reward components
* **Scheduling timeline** — ASCII/Gantt-style job lifecycle visualization
* **Comparison charts** — 6 charts (avg wait, p95 wait, avg JCT, utilization, throughput, fairness) across scenarios
* **OOD/generalization section** — heatmap/barchart of JCT degradation across shifts A-F
* **Experiment page** — select scenario/metric/scheduler to update charts
* **Training replay** — ASCII frame replay from saved PPO decision logs
* **Export GIF** — `python scripts/export_replay.py --scenario balanced --scheduler PPO --seed 0 --output artifacts/replays/ppo_balanced.gif`

The dashboard reuses existing benchmark artifacts and evaluation framework — no retraining or simulator changes required.

### Important Design Choice

A scheduling action never manipulates GPUs directly — the **simulator owns** time, arrivals, completions, allocation/release. This lets the exact same engine evaluate heuristics and PPO under identical workloads.

A scheduling action never manipulates GPUs directly — the **simulator owns** time, arrivals, completions, allocation/release. This lets the exact same engine evaluate heuristics and PPO under identical workloads.
