from gpu_sage.env.gpu_env import GPUSchedulingEnv
from gpu_sage.workloads.generator import get_scenario_config
import numpy as np

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
print('Observation keys and shapes:')
for k, v in obs.items():
    print(f'  {k}: shape={v.shape}')

# Check topo-related features
print()
print('Job features - last 2 elements are topo_flag and pref_id:')
print(f'  jobs[-2] (topo_flag) sample: {obs["jobs"][-2]}')
print(f'  jobs[-1] (pref_id) sample: {obs["jobs"][-1]}')

print()
print('Cluster features - last 2 elements are avg_perf and hetero_flag:')
print(f'  cluster[-2] (avg_perf) sample: {obs["cluster"][-2]}')
print(f'  hetero_flag present: {obs["cluster"][-1] if len(obs["cluster"]) > 7 else "check shape"}')

env.close()