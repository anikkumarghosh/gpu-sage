"""Test topology generalization: graph PPO on different topology config."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sb3_contrib import MaskablePPO
from gpu_sage.env.gpu_env import GPUSchedulingEnv
from gpu_sage.workloads.generator import get_scenario_config
import numpy as np
from gpu_sage.core.cluster import Cluster
from gpu_sage.core.models import GPU
from gpu_sage.core.topology import Topology

# Load the graph model trained on topology_sensitive (two_group topology)
graph_model = MaskablePPO.load('artifacts/ppo/runs/20260830_000940_seed0_steps128/model/final_model.zip')
print(f'Model: {type(graph_model.policy).__name__}')
print(f'Features extractor: {type(graph_model.policy.features_extractor_class).__name__}')

# Create cluster with same GPUs but DIFFERENT topology
# Training topology: two_group (groups of 4 with NVLink within, PCIe across)
# Testing topology: fully_connected (all NVLink, no across-group penalty)

cluster = Cluster.homogeneous(8, 80.0)  # 8 GPUs, all A100 80GB

# Set fully_connected NVLink topology (different from training two_group)
cluster.topology = Topology.fully_connected_nvlink(8)
cluster.placement_alpha = 0.6
cluster.placement_cap = 2.5

# Create env with this cluster
env = GPUSchedulingEnv(
    num_gpus=8,
    gpu_memory_gb=80.0,
    max_jobs=16,
    episode_jobs=20,
    workload_config=get_scenario_config('topology_sensitive'),
    reward_config=None,
    seed=0,
    cluster=None,  # we pass cluster directly
    heterogeneous_obs=True,
)

# Override the sim's cluster
obs, info = env.reset(seed=0)
env.sim.cluster = cluster

# Re-observation with new topology
obs = env._observation()
print(f'Observation cluster shape: {obs["cluster"].shape}')
print(f'Observation gpus shape: {obs["gpus"].shape}')

# The GNN adjacency is fixed from training (two_group), but let's see if inference works
try:
    action, _states = graph_model.predict(obs, action_masks=info['action_mask'], deterministic=True)
    print(f'Predicted action: {action}')
    
    # Step
    obs2, reward, terminated, truncated, info2 = env.step(action)
    print(f'Step reward: {reward:.2f}, terminated: {terminated}')
    print(f'Completed: {info2["completed_jobs"]}, Waiting: {info2["waiting_jobs"]}')
    
    print()
    print('Topology generalization test: model ran on fully_connected topology '
          '(trained on two_group)')
    
except Exception as e:
    print(f'Error during prediction: {e}')
    import traceback
    traceback.print_exc()

env.close()
print('Topology generalization test complete!')