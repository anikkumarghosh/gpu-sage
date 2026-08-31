"""Evaluate all graph/flat PPO models on all scenarios."""
import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sb3_contrib import MaskablePPO
from gpu_sage.env.gpu_env import GPUSchedulingEnv
from gpu_sage.workloads.generator import get_scenario_config
import numpy as np
from pathlib import Path

SCENARIOS = ['topology_sensitive', 'heterogeneous', 'mixed_ml', 'balanced']

def find_model(run_id, scenario):
    """Find model file for a given run_id and scenario."""
    model_path = Path(f'artifacts/ppo/runs/{run_id}/model/final_model.zip')
    return model_path if model_path.exists() else None

def evaluate_model(model_path, scenario):
    """Evaluate a PPO model and return metrics."""
    env = GPUSchedulingEnv(
        num_gpus=8,
        gpu_memory_gb=80.0,
        max_jobs=16,
        episode_jobs=20,
        workload_config=get_scenario_config(scenario),
        reward_config=None,
        seed=0,
        cluster_config_path='configs/heterogeneous_8gpu.yaml',
        heterogeneous_obs=True,
    )

    obs, info = env.reset(seed=0)
    model = MaskablePPO.load(str(model_path))
    
    completed = 0
    waiting = 0
    total_reward = 0.0
    steps = 0
    
    for step in range(200):
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

# Known run IDs from our training runs (with --heterogeneous-obs)
RUN_IDS = {
    'topology_sensitive': {'graph': '20260830_000940_seed0_steps128', 'flat': '20260830_001000_seed0_steps128'},
    'heterogeneous': {'graph': '20260830_001036_seed0_steps128', 'flat': '20260830_001058_seed0_steps128'},
    'mixed_ml': {'graph': '20260830_001115_seed0_steps128', 'flat': '20260830_001135_seed0_steps128'},
    'balanced': {'graph': '20260830_001153_seed0_steps128', 'flat': '20260830_001214_seed0_steps128'},
}

results = {}

print("=" * 70)
print("EVALUATION: Graph PPO vs Flat PPO on all scenarios")
print("=" * 70)

for scenario in SCENARIOS:
    print(f"\n--- Scenario: {scenario} ---")
    
    run_ids = RUN_IDS[scenario]
    
    for policy, run_id in run_ids.items():
        model_path = find_model(run_id, scenario)
        if model_path is None:
            print(f"  {policy} model not found for run {run_id}")
            continue
            
        print(f"  {policy} model: {model_path}")
        
        eval_result = evaluate_model(str(model_path), scenario)
        results.setdefault(scenario, {})[policy] = eval_result
        
        print(f"    JCT={eval_result['avg_jct']:.2f} (completed {eval_result['completed']}/200 steps), reward={eval_result['total_reward']:.1f}")

# Save results
with open('comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '=' * 70)
print("RESULTS SAVED TO comparison_results.json")
print('=' * 70)

# Print summary table
print("\n\nFINAL COMPARISON TABLE:")
print(f"{'Scenario':<20} {'Method':<12} {'JCT':>8} {'Completed':>10} {'Reward':>12}")
print("-" * 62)
for scenario in SCENARIOS:
    r = results.get(scenario, {})
    if 'graph' in r:
        print(f"{scenario:<20} {'Graph PPO':<12} {r['graph']['avg_jct']:>7.2f} {r['graph']['completed']:>10} {r['graph']['total_reward']:>12.1f}")
    if 'flat' in r:
        print(f"{'':<20} {'Flat PPO':<12} {r['flat']['avg_jct']:>7.2f} {r['flat']['completed']:>10} {r['flat']['total_reward']:>12.1f}")