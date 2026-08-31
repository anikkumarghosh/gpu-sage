import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sb3_contrib import MaskablePPO
from gpu_sage.rl.graph_no_topo import GraphFeaturesExtractorNoTopo, GraphMaskablePolicyNoTopo
from gpu_sage.env.gpu_env import GPUSchedulingEnv
from gpu_sage.workloads.generator import get_scenario_config
import numpy as np

# Create env
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

# Create model with proper policy_kwargs
policy_kwargs = dict(
    features_extractor_class=GraphFeaturesExtractorNoTopo,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[64, 32], vf=[64, 32]),
)

model = MaskablePPO(
    GraphMaskablePolicyNoTopo,
    env,
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
    tensorboard_log='output_test',
    seed=0,
    device='auto',
    policy_kwargs=policy_kwargs,
)

# Train for 128 steps
model.learn(total_timesteps=128, progress_bar=True)
model.save('artifacts/ppo/runs/no_topo_seed0_steps128/model/final_model.zip')
print('Model saved!')

# Test prediction
obs_test, info_test = env.reset(seed=0)
action, _states = model.predict(obs_test, action_masks=info_test['action_mask'], deterministic=True)
print(f'Predicted action: {action}')

obs2, reward, terminated, truncated, info2 = env.step(action)
print(f'Step reward: {reward:.2f}, terminated: {terminated}')
print(f'Completed: {info2["completed_jobs"]}, Waiting: {info2["waiting_jobs"]}')

env.close()
model.close()
print('NoTopo training and evaluation complete!')