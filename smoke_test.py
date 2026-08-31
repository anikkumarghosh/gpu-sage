"""Minimal graph policy smoke test with heterogeneous_obs."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sb3_contrib import MaskablePPO
from gpu_sage.env.gpu_env import GPUSchedulingEnv
from gpu_sage.workloads.generator import get_scenario_config
import numpy as np

# Create env WITH heterogeneous_obs=True
env = GPUSchedulingEnv(
    num_gpus=8,
    gpu_memory_gb=80.0,
    max_jobs=16,
    episode_jobs=20,
    workload_config=get_scenario_config('topology_sensitive'),
    reward_config=None,
    seed=0,
    cluster_config_path='configs/heterogeneous_8gpu.yaml',
    heterogeneous_obs=True,
)

obs, info = env.reset(seed=0)
print('Env observation space keys:', list(env.observation_space.spaces.keys()))

# Load the new graph model (trained with --heterogeneous-obs)
model = MaskablePPO.load('artifacts/ppo/runs/20260829_235327_seed0_steps128/model/final_model.zip')
print('Model features_extractor_class:', model.policy.features_extractor_class)
print('Model observation space keys:', list(model.policy.observation_space.spaces.keys()))
print('Model cluster dim:', model.policy.observation_space.spaces['cluster'].shape)
print('Model gpu dim:', model.policy.observation_space.spaces['gpus'].shape)
print('Model jobs dim:', model.policy.observation_space.spaces['jobs'].shape)

# Test predict
action, _states = model.predict(obs, action_masks=info['action_mask'], deterministic=True)
print(f'Predict action: {action}')

# Step
obs2, reward, terminated, truncated, info2 = env.step(action)
print(f'Step reward: {reward:.2f}, terminated: {terminated}')

print(f'Completed: {info2["completed_jobs"]}, Waiting: {info2["waiting_jobs"]}')
env.close()
print('Prediction test successful!')