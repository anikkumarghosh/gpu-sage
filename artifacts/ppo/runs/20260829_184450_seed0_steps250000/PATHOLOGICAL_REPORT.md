# GPU-Sage PPO Pathological Analysis

Benchmark dir: `artifacts\benchmarks`

Total per-job records: 15025

## Starvation
- **BestFit**: median=67.0s, P95=12358.3s, max=21879.8s, max/median=326.7x
- **FCFS**: median=294.4s, P95=12830.0s, max=19940.9s, max/median=67.7x
- **PPO**: median=268.1s, P95=13749.5s, max=21304.6s, max/median=79.5x
- **Priority**: median=153.6s, P95=12427.5s, max=19860.5s, max/median=129.3x
- **SJF**: median=71.1s, P95=9733.5s, max=21486.1s, max/median=302.2x

## Large-job Starvation (GPU count)
- **BestFit**: avg GPU=2.91, large (>=4 GPUs)=33.4% of completed
- **FCFS**: avg GPU=2.91, large (>=4 GPUs)=33.4% of completed
- **PPO**: avg GPU=2.91, large (>=4 GPUs)=33.4% of completed
- **Priority**: avg GPU=2.91, large (>=4 GPUs)=33.4% of completed
- **SJF**: avg GPU=2.91, large (>=4 GPUs)=33.4% of completed

## Fragmentation / Utilization
**balanced**:
  - BestFit: util=0.909 ± 0.006
  - FCFS: util=0.934 ± 0.010
  - PPO: util=0.930 ± 0.014
  - Priority: util=0.931 ± 0.022
  - SJF: util=0.918 ± 0.016
**bursty**:
  - BestFit: util=0.844 ± 0.050
  - FCFS: util=0.883 ± 0.066
  - PPO: util=0.871 ± 0.060
  - Priority: util=0.863 ± 0.054
  - SJF: util=0.859 ± 0.052
**gpu_heavy**:
  - BestFit: util=0.824 ± 0.012
  - FCFS: util=0.829 ± 0.012
  - PPO: util=0.817 ± 0.014
  - Priority: util=0.830 ± 0.012
  - SJF: util=0.829 ± 0.012
**heavy_tail**:
  - BestFit: util=0.836 ± 0.030
  - FCFS: util=0.889 ± 0.058
  - PPO: util=0.884 ± 0.037
  - Priority: util=0.896 ± 0.017
  - SJF: util=0.848 ± 0.042
**priority_skew**:
  - BestFit: util=0.905 ± 0.013
  - FCFS: util=0.938 ± 0.014
  - PPO: util=0.944 ± 0.002
  - Priority: util=0.934 ± 0.024
  - SJF: util=0.922 ± 0.014
**short_jobs**:
  - BestFit: util=0.867 ± 0.026
  - FCFS: util=0.877 ± 0.036
  - PPO: util=0.881 ± 0.032
  - Priority: util=0.874 ± 0.028
  - SJF: util=0.867 ± 0.028

## NOOP Abuse
- **balanced_ppo_decisions**: steps=1466, NOOP=966 (65.9%)
  - WARN: High NOOP rate -- possible NOOP abuse
- **balanced_seed0_ppo_decisions**: steps=54, NOOP=34 (63.0%)
  - WARN: High NOOP rate -- possible NOOP abuse
- **bursty_ppo_decisions**: steps=1412, NOOP=912 (64.6%)
  - WARN: High NOOP rate -- possible NOOP abuse
- **gpu_heavy_ppo_decisions**: steps=1485, NOOP=985 (66.3%)
  - WARN: High NOOP rate -- possible NOOP abuse
- **heavy_tail_ppo_decisions**: steps=1468, NOOP=968 (65.9%)
  - WARN: High NOOP rate -- possible NOOP abuse
- **priority_skew_ppo_decisions**: steps=1467, NOOP=967 (65.9%)
  - WARN: High NOOP rate -- possible NOOP abuse
- **short_jobs_ppo_decisions**: steps=1244, NOOP=744 (59.8%)
  - WARN: High NOOP rate -- possible NOOP abuse

## Priority Behavior
- **BestFit**:
  - P1: avg wait 1545.1s (n=787)
  - P2: avg wait 1718.4s (n=632)
  - P3: avg wait 2124.9s (n=525)
  - P4: avg wait 1835.2s (n=539)
  - P5: avg wait 1978.8s (n=517)
  - BestFit does not prioritize P5 over P1
- **FCFS**:
  - P1: avg wait 1704.3s (n=787)
  - P2: avg wait 1955.8s (n=632)
  - P3: avg wait 2326.8s (n=525)
  - P4: avg wait 2061.8s (n=539)
  - P5: avg wait 2297.2s (n=517)
  - FCFS does not prioritize P5 over P1
- **PPO**:
  - P1: avg wait 3018.7s (n=787)
  - P2: avg wait 2602.8s (n=632)
  - P3: avg wait 1947.1s (n=525)
  - P4: avg wait 1021.3s (n=539)
  - P5: avg wait 1678.2s (n=517)
  - PPO respects priority (P5 faster than P1)
- **Priority**:
  - P1: avg wait 2704.8s (n=787)
  - P2: avg wait 2481.5s (n=632)
  - P3: avg wait 2136.9s (n=525)
  - P4: avg wait 1226.1s (n=539)
  - P5: avg wait 618.9s (n=517)
  - Priority respects priority (P5 faster than P1)
- **SJF**:
  - P1: avg wait 1273.0s (n=787)
  - P2: avg wait 1411.2s (n=632)
  - P3: avg wait 1811.5s (n=525)
  - P4: avg wait 1434.2s (n=539)
  - P5: avg wait 1513.8s (n=517)
  - SJF does not prioritize P5 over P1

---
*Generated from per-job and decision logs; see `artifacts/benchmarks/` for raw data.*