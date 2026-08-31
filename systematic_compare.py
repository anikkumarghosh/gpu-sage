"""Systematic comparison: Graph PPO vs Flat PPO on all scenarios."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sb3_contrib import MaskablePPO
from gpu_sage.env.gpu_env import GPUSchedulingEnv
from gpu_sage.workloads.generator import get_scenario_config
import numpy as np
import json
from pathlib import Path
import subprocess
import sys

SCENARIOS = ['topology_sensitive', 'heterogeneous', 'mixed_ml', 'balanced']
SEED = 0
N_STEPS_TRAIN = 128
N_STEPS_EVAL = 200

def train_and_evaluate(scenario, policy, model_name_suffix=""):
    """Train a PPO model and evaluate on fixed workload."""
    
    # Train
    print(f"  Training {policy} on {scenario}...", end=" ")
    train_result = subprocess.run(
        [sys.executable, "training/train_ppo.py",
         "--steps", str(N_STEPS_TRAIN),
         "--seed", str(SEED),
         "--policy", policy,
         "--scenario", scenario,
         "--cluster-config", "configs/heterogeneous_8gpu.yaml",
         "--heterogeneous-obs" if policy == "graph" else "",
         "--out", f"artifacts/ppo/runs/{model_name_suffix}_{scenario}_seed{SEED}_steps{N_STEPS_TRAIN}"],
        cwd="C:/Users/ANIK KUMAR/Downloads/gpu-sage-milestone2/gpu-sage",
        capture_output=True, text=True, timeout=120000
    )
    print(train_result.stdout[-200:] if len(train_result.stdout) > 200 else train_result.stdout)
    if train_result.returncode != 0:
        print(f"Training failed: {train_result.stderr}")
        return None
    
    # Find the model
    run_dir = Path(f"artifacts/ppo/runs/{model_name_suffix}_{scenario}_seed{SEED}_steps{N_STEPS_TRAIN}")
    model_path = run_dir / "model" / "final_model.zip" if run_dir.exists() else None
    
    if not model_path or not model_path.exists():
        print(f"Model not found at {model_path}")
        return None
    
    # Evaluate
    print(f"  Evaluating {policy} on {scenario}...", end=" ")
    env = GPUSchedulingEnv(
        num_gpus=8,
        gpu_memory_gb=80.0,
        max_jobs=16,
        episode_jobs=20,
        workload_config=get_scenario_config(scenario),
        reward_config=None,
        seed=SEED,
        cluster_config_path='configs/heterogeneous_8gpu.yaml',
        heterogeneous_obs=True,
    )
    
    obs, info = env.reset(seed=SEED)
    model = MaskablePPO.load(str(model_path))
    
    completed = 0
    waiting = 0
    total_reward = 0.0
    steps = 0
    
    for step in range(N_STEPS_EVAL):
        action, _states = model.predict(obs, action_masks=info['action_mask'], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        completed = info['completed_jobs']
        waiting = info['waiting_jobs']
        
        if terminated or truncated:
            break
    
    env.close()
    
    avg_jct = env.sim.current_time / completed if completed > 0 else None
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'completed': completed,
        'waiting': waiting,
        'avg_jct': avg_jct,
        'final_time': env.sim.current_time if completed > 0 else None,
    }

results = {}

print("=" * 70)
print("SYSTEMATIC COMPARISON: Graph PPO vs Flat PPO")
print("=" * 70)

for scenario in SCENARIOS:
    print(f"\n--- Scenario: {scenario} ---")
    
    # Graph PPO
    print(f"  Graph PPO:")
    graph_result = train_and_evaluate(scenario, "graph", f"graph")
    
    # Flat PPO
    print(f"  Flat PPO:")
    flat_result = train_and_evaluate(scenario, "flat", f"flat")
    
    results[scenario] = {
        'graph': graph_result,
        'flat': flat_result,
    }
    
    g = graph_result
    f = flat_result
    print(f"    Graph:  JCT={g['avg_jct']:.2f} (completed {g['completed']}/{N_STEPS_EVAL} steps), reward={g['total_reward']:.1f}" if g else "    Graph: FAILED")
    print(f"    Flat:   JCT={f['avg_jct']:.2f} (completed {f['completed']}/{N_STEPS_EVAL} steps), reward={f['total_reward']:.1f}" if f else "    Flat: FAILED")

# Save results
with open('comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '=' * 70)
print("RESULTS SAVED TO comparison_results.json")
print('=' * 70)

# Print summary
print("\n\nSUMMARY:")
for scenario in SCENARIOS:
    r = results[scenario]
    g = r['graph']
    f = r['flat']
    print(f"\n{scenario}:")
    if g:
        print(f"  Graph PPO:  JCT={g['avg_jct']:.2f} (completed {g['completed']}/{N_STEPS_EVAL} steps), reward={g['total_reward']:.1f}")
    if f:
        print(f"  Flat PPO:   JCT={f['avg_jct']:.2f} (completed {f['completed']}/{N_STEPS_EVAL} steps), reward={f['total_reward']:.1f}")