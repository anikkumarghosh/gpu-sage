import numpy as np
from gymnasium.utils.env_checker import check_env

from gpu_sage.env.gpu_env import GPUSchedulingEnv


def test_env_reset_and_step() -> None:
    env = GPUSchedulingEnv(num_gpus=4, max_jobs=8, episode_jobs=20, seed=123)
    obs, info = env.reset(seed=123)

    assert set(obs) == {"cluster", "gpus", "jobs", "job_mask"}
    assert obs["gpus"].shape == (4, 3)
    assert obs["jobs"].shape == (8, 8)
    assert info["action_mask"].shape == (9,)

    mask = info["action_mask"]
    action = int(np.flatnonzero(mask)[0])
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    assert next_obs["cluster"].shape == (6,)
    assert isinstance(float(reward), float)
    assert not (terminated and truncated)
    assert next_info["action_mask"].shape == (9,)


def test_env_check_env() -> None:
    env = GPUSchedulingEnv(num_gpus=4, max_jobs=8, episode_jobs=10, seed=7)
    check_env(env, skip_render_check=True)
