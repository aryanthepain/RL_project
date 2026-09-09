from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
from gymnasium.spaces import Box


# Standard benchmark hyperparameters from ICML 2020 paper (Table 2)
# drop_top = 2 for Humanoid, drop_top = 5 for other continuous locomotion
BENCHMARK_ENVS: Dict[str, Dict[str, Any]] = {
    "HalfCheetah-v4": {"drop_top": 5, "total_timesteps": 1_000_000},
    "HalfCheetah-v5": {"drop_top": 5, "total_timesteps": 1_000_000},
    "Hopper-v4": {"drop_top": 5, "total_timesteps": 1_000_000},
    "Hopper-v5": {"drop_top": 5, "total_timesteps": 1_000_000},
    "Walker2d-v4": {"drop_top": 5, "total_timesteps": 1_000_000},
    "Walker2d-v5": {"drop_top": 5, "total_timesteps": 1_000_000},
    "Ant-v4": {"drop_top": 5, "total_timesteps": 3_000_000},
    "Ant-v5": {"drop_top": 5, "total_timesteps": 3_000_000},
    "Humanoid-v4": {"drop_top": 2, "total_timesteps": 3_000_000},
    "Humanoid-v5": {"drop_top": 2, "total_timesteps": 3_000_000},
}


def get_env_metadata(env_id: str) -> Dict[str, Any]:
    """Retrieve metadata and recommended TQC hyperparameters for an environment."""
    if env_id in BENCHMARK_ENVS:
        return BENCHMARK_ENVS[env_id].copy()

    # Dynamic fallback based on environment name
    env_lower = env_id.lower()
    if "humanoid" in env_lower:
        return {"drop_top": 2, "total_timesteps": 3_000_000}
    return {"drop_top": 5, "total_timesteps": 1_000_000}


def make_env(env_id: str, seed: Optional[int] = None) -> gym.Env:
    """Create and configure a Gymnasium MuJoCo benchmark environment with seed control.

    Args:
        env_id: Standard Gymnasium environment identifier (e.g. 'HalfCheetah-v4').
        seed: Optional integer seed for action space and environment dynamics.

    Returns:
        Configured Gymnasium environment.
    """
    try:
        env = gym.make(env_id)
    except gym.error.NamespaceNotFound:
        raise
    except gym.error.DeprecatedEnv as e:
        # If deprecated, attempt fallback to newer version if suggested
        raise e

    if not isinstance(env.action_space, Box):
        raise ValueError(
            f"Environment {env_id} action space must be continuous Box, got {type(env.action_space)}"
        )

    if seed is not None:
        env.action_space.seed(seed)
        env.reset(seed=seed)

    return env


def get_env_dims(env: gym.Env) -> Tuple[int, int]:
    """Extract observation (state) dimension and continuous action dimension."""
    if not isinstance(env.observation_space, Box):
        raise ValueError("Environment observation space must be continuous Box.")
    if not isinstance(env.action_space, Box):
        raise ValueError("Environment action space must be continuous Box.")

    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    return state_dim, action_dim
