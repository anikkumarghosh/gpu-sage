"""Compare flat PPO vs graph PPO on topology-sensitive workload - with metrics."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sb3_contrib import MaskablePPO
from gpu_sage.env.gpu_env import GPUSchedulingEnv
from gpu_sage.workloads.generator import get_scenario_config
import numpy as np

def run_model(model_path, label, seed=0):
    """Run a PPO model and return key metrics."""
    env = GPUSchedulingEnv(
        num_gpus=8,
        gpu_memory_gb=80.0,
        max_jobs=16,
        episode_jobs=20,
        workload_config=get_scenario_config('topology_sensitive'),
        reward_config=None,
        seed=seed,
        cluster_config_path='configs/heterogeneous_8gpu.yaml',
        heterogeneous_obs=True,
    )

    obs, info = env.reset(seed=seed)
    model = MaskablePPO.load(model_path)
    
    completed = 0
    waiting = 0
    total_reward = 0.0
    steps = 0
    
    for step in range(200):  # More steps to see completion
        action, _states = model.predict(obs, action_masks=info['action_mask'], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        completed = info['completed_jobs']
        waiting = info['waiting_jobs']
        
        if terminated or truncated:
            break
    
    env.close()
    
    # Calculate JCT-like metric: avg time to complete jobs
    # Use simulator's completed jobs timing
    avg_jct = None
    if completed > 0:
        # Approximate: current time / completed gives rough JCT
        avg_jct = env.sim.current_time / completed if hasattr(env, 'sim') and env.sim else None
    
    return {
        'label': label,
        'model_path': model_path,
        'total_reward': total_reward,
        'steps': steps,
        'completed': completed,
        'waiting': waiting,
        'avg_jct': avg_jct,
        'final_time': env.sim.current_time if hasattr(env, 'sim') and env.sim else None,
    }

# Graph PPO
graph_result = run_model(
    'artifacts/ppo/runs/20260829_235327_seed0_steps128/model/final_model.zip',
    'Graph PPO', seed=0
)

# Flat PPO
flat_result = run_model(
    'artifacts/ppo/runs/20260829_235449_seed0_steps128/model/final_model.zip',
    'Flat PPO', seed=0
)

print()
print("=" * 60)
print("COMPARISON RESULTS (topology_sensitive, seed 0)")
print("=" * 60)
for key in ['label', 'total_reward', 'steps', 'completed', 'waiting', 'avg_jct', 'final_time']:
    g = str(graph_result.get(key, ''))
    f = str(flat_result.get(key, ''))
    print(f"{key:20s}: Graph={g} | Flat={f}")
print()
print("Graph PPO completed:", graph_result['completed'], "jobs in", graph_result['steps'], "steps")
print("Flat PPO completed:", flat_result['completed'], "jobs in", flat_result['steps'], "steps")