from typing import Tuple
import gymnasium as gym
import numpy as np
import torch
from .agent import TQCAgent


def evaluate_policy(
    agent: TQCAgent,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """Evaluate agent policy for a fixed number of episodes with deterministic action selection.

    Args:
        agent: Trained TQCAgent.
        env: Evaluation Gymnasium environment.
        n_episodes: Number of evaluation episodes (default 10).
        deterministic: If True, uses mean of squashed Gaussian policy (tanh(mu)).

    Returns:
        Tuple of (mean_return, std_return).
    """
    returns = []

    with torch.no_grad():
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_return = 0.0
            done = False

            while not done:
                action = agent.select_action(state, deterministic=deterministic)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                done = terminated or truncated
                state = next_state

            returns.append(episode_return)

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    return mean_return, std_return
