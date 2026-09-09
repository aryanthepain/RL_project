import argparse
import os
import time
from typing import Optional, Union
import gymnasium as gym
import numpy as np
import torch
from .agent import TQCAgent
from .envs import get_env_dims, get_env_metadata, make_env
from .evaluate import evaluate_policy
from .logger import MetricsLogger, create_agent
from .replay_buffer import ReplayBuffer
from .utils import get_device, seed_everything


def train_tqc(
    env_id: str = "HalfCheetah-v4",
    seed: int = 0,
    total_timesteps: int = 1_000_000,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    warmup_steps: int = 10_000,
    batch_size: int = 256,
    buffer_capacity: int = 1_000_000,
    drop_top: Optional[int] = None,
    algo: str = "tqc",
    device: Union[str, torch.device] = "cpu",
    log_dir: str = "runs",
    exp_name: Optional[str] = None,
    checkpoint_freq: int = 50_000,
) -> TQCAgent:
    """Execute complete TQC training loop following Kuznetsov et al. (ICML 2020).

    Args:
        env_id: Gymnasium MuJoCo benchmark environment ID.
        seed: Random seed for reproducibility.
        total_timesteps: Total environment interaction steps.
        eval_freq: Timestep interval between deterministic policy evaluations.
        eval_episodes: Number of test episodes per evaluation.
        warmup_steps: Initial random exploration steps before gradient updates.
        batch_size: Transitions per mini-batch for agent updates (default 256).
        buffer_capacity: Maximum transitions in replay buffer (default 1,000,000).
        drop_top: Number of top quantiles to discard in target distribution.
        algo: Algorithm identifier ('tqc' or 'sac').
        device: Torch device ('cpu' or 'cuda').
        log_dir: Base directory for metrics and checkpoints.
        exp_name: Optional explicit experiment name.
        checkpoint_freq: Step interval for saving checkpoints.

    Returns:
        Trained TQCAgent instance.
    """
    dev = get_device(device)
    seed_everything(seed)

    # Initialize environments
    train_env = make_env(env_id, seed=seed)
    eval_env = make_env(env_id, seed=seed + 1000)

    state_dim, action_dim = get_env_dims(train_env)

    if exp_name is None:
        exp_name = f"{algo}_{env_id}_seed{seed}_{int(time.time())}"

    logger = MetricsLogger(log_dir=log_dir, exp_name=exp_name)

    # Instantiate Agent & Buffer
    agent = create_agent(
        algo=algo,
        state_dim=state_dim,
        action_dim=action_dim,
        device=dev,
        drop_top=drop_top,
        env_id=env_id,
    )
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=buffer_capacity,
        device=dev,
    )

    state, _ = train_env.reset(seed=seed)
    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0
    best_eval_return = -float("inf")
    latest_train_metrics = {}

    print(
        f"Starting {algo.upper()} on {env_id} (Seed {seed}) | "
        f"State: {state_dim}, Action: {action_dim}, Drop Top: {agent.drop_top}, Device: {dev}"
    )

    try:
        for step in range(1, total_timesteps + 1):
            # Action selection
            if step <= warmup_steps:
                action = train_env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)

            # Environment step
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            episode_reward += float(reward)
            episode_timesteps += 1

            # Store transition: only terminated is true MDP termination; truncation is artificial time limit
            done_bool = float(terminated)
            replay_buffer.add(state, action, reward, next_state, done_bool)

            state = next_state

            # Gradient update step once warmup is finished
            if step >= warmup_steps:
                latest_train_metrics = agent.update(replay_buffer, batch_size=batch_size)

            # Handle episode termination
            if terminated or truncated:
                episode_num += 1
                state, _ = train_env.reset()
                episode_reward = 0.0
                episode_timesteps = 0

            # Periodic deterministic evaluation
            if step % eval_freq == 0 or step == total_timesteps:
                eval_mean, eval_std = evaluate_policy(
                    agent, eval_env, n_episodes=eval_episodes, deterministic=True
                )

                metrics_to_log = {
                    "eval_return_mean": eval_mean,
                    "eval_return_std": eval_std,
                    "episodes": episode_num,
                }
                metrics_to_log.update(latest_train_metrics)
                logger.log_metrics(step=step, metrics=metrics_to_log, print_to_console=True)

                if eval_mean > best_eval_return:
                    best_eval_return = eval_mean
                    best_model_path = os.path.join(logger.exp_dir, "best_model.pt")
                    agent.save(best_model_path)

            # Periodic checkpoint
            if checkpoint_freq > 0 and step % checkpoint_freq == 0:
                ckpt_path = os.path.join(logger.exp_dir, f"checkpoint_{step}.pt")
                agent.save(ckpt_path)

        # Save final model
        final_model_path = os.path.join(logger.exp_dir, "final_model.pt")
        agent.save(final_model_path)
    finally:
        logger.close()
        train_env.close()
        eval_env.close()

    print(f"Training completed. Best Eval Return: {best_eval_return:.2f}. Logs in: {logger.exp_dir}")
    return agent


def main():
    parser = argparse.ArgumentParser(description="TQC / SAC Benchmark Training Pipeline")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4", help="Gymnasium MuJoCo Env")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--warmup-steps", type=int, default=10_000, help="Initial random exploration steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--drop-top", type=int, default=None, help="Top quantiles to discard")
    parser.add_argument("--algo", type=str, default="tqc", choices=["tqc", "sac"], help="Algorithm")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument("--log-dir", type=str, default="runs", help="Logging root directory")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Checkpoint frequency")

    args = parser.parse_args()
    train_tqc(**vars(args))


if __name__ == "__main__":
    main()
