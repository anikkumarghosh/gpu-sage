"""Smoke-test the Gymnasium environment and action masking."""

from gymnasium.utils.env_checker import check_env

from gpu_sage.env import GPUSchedulingEnv


def main() -> None:
    env = GPUSchedulingEnv(num_gpus=4, max_jobs=8, episode_jobs=20, seed=123)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=123)
    print("Environment OK")
    print("Observation shapes:", {k: v.shape for k, v in obs.items()})
    print("Action mask:", info["action_mask"].astype(int).tolist())
    print("NOOP action:", env.noop_action)
    env.close()


if __name__ == "__main__":
    main()
