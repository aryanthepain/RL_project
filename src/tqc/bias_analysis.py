import argparse
import json
import os
from typing import Any, Dict, List, Optional
import gymnasium as gym
import numpy as np
import torch
from .agent import TQCAgent
from .envs import get_env_dims, make_env
from .logger import create_agent
from .utils import get_device, seed_everything


def compute_overestimation_bias(
    agent: TQCAgent,
    env: gym.Env,
    n_samples: int = 50,
    gamma: float = 0.99,
    max_steps: int = 1000,
) -> Dict[str, Any]:
    """Empirically evaluate Q-estimation bias against true discounted Monte Carlo returns.

    For each evaluation trajectory, states are visited and evaluated:
    - Critic Q-prediction: Mean across all predicted quantiles E[Z(s, pi(s))].
    - True Monte Carlo return: G_t = sum_{k=0}^{H-1} gamma^k r_{t+k}.
    - Estimation bias: Delta = Q_pred - G_t.

    Args:
        agent: Trained actor-critic agent.
        env: Gymnasium environment instance.
        n_samples: Number of state-action evaluations to collect.
        gamma: Discount factor (default 0.99).
        max_steps: Maximum trajectory rollout horizon.

    Returns:
        Dictionary with statistical metrics and raw distributions.
    """
    q_predictions: List[float] = []
    mc_returns: List[float] = []
    biases: List[float] = []

    collected = 0

    while collected < n_samples:
        state, _ = env.reset()
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []

        done = False
        step = 0

        # Rollout full trajectory
        while not done and step < max_steps:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)

            trajectory_states.append(state)
            trajectory_actions.append(action)
            trajectory_rewards.append(float(reward))

            done = terminated or truncated
            state = next_state
            step += 1

        traj_len = len(trajectory_rewards)
        if traj_len == 0:
            continue

        # Compute discounted returns G_t backwards
        discounted_returns = np.zeros(traj_len, dtype=np.float32)
        running_g = 0.0
        for t in reversed(range(traj_len)):
            running_g = trajectory_rewards[t] + gamma * running_g
            discounted_returns[t] = running_g

        # Predict Q-values from critic ensemble for states in trajectory
        states_tensor = torch.as_tensor(
            np.array(trajectory_states), dtype=torch.float32, device=agent.device
        )
        actions_tensor = torch.as_tensor(
            np.array(trajectory_actions), dtype=torch.float32, device=agent.device
        )

        with torch.no_grad():
            # Critic predicts (B, M, N)
            quantiles = agent.critic(states_tensor, actions_tensor)
            # Expected Q-value is the mean across all quantiles and critics
            q_preds = quantiles.mean(dim=(1, 2)).cpu().numpy()

        # Record samples up to n_samples
        for t in range(traj_len):
            if collected >= n_samples:
                break
            q_est = float(q_preds[t])
            g_true = float(discounted_returns[t])
            bias = q_est - g_true

            q_predictions.append(q_est)
            mc_returns.append(g_true)
            biases.append(bias)
            collected += 1

    return {
        "n_samples": collected,
        "mean_predicted_q": float(np.mean(q_predictions)),
        "std_predicted_q": float(np.std(q_predictions)),
        "mean_true_return": float(np.mean(mc_returns)),
        "std_true_return": float(np.std(mc_returns)),
        "mean_bias": float(np.mean(biases)),
        "std_bias": float(np.std(biases)),
        "q_predictions": q_predictions,
        "mc_returns": mc_returns,
        "biases": biases,
    }


def save_bias_report(bias_dict: Dict[str, Any], output_path: str) -> None:
    """Save overestimation bias report dictionary to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bias_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TQC Overestimation Bias Evaluator")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained agent .pt")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4", help="Gymnasium Env")
    parser.add_argument("--algo", type=str, default="tqc", choices=["tqc", "sac"], help="Algorithm")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of state evaluations")
    parser.add_argument("--output", type=str, default="results/bias_report.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    args = parser.parse_args()
    seed_everything(args.seed)
    env = make_env(args.env_id, seed=args.seed)
    s_dim, a_dim = get_env_dims(env)

    agent = create_agent(args.algo, s_dim, a_dim, device=args.device, env_id=args.env_id)
    agent.load(args.model_path)

    report = compute_overestimation_bias(agent, env, n_samples=args.n_samples)
    save_bias_report(report, args.output)

    print("\n--- Overestimation Bias Analysis ---")
    print(f"Evaluated Samples: {report['n_samples']}")
    print(f"Mean Predicted Q:  {report['mean_predicted_q']:.2f} ± {report['std_predicted_q']:.2f}")
    print(f"Mean MC Return:    {report['mean_true_return']:.2f} ± {report['std_true_return']:.2f}")
    print(f"Mean Bias (Q - G): {report['mean_bias']:.2f} ± {report['std_bias']:.2f}")
    print(f"Report saved to:   {args.output}")


if __name__ == "__main__":
    main()
