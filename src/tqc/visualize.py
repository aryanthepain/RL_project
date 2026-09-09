import argparse
import os
import time
from typing import List, Optional
import gymnasium as gym
import imageio
import numpy as np
import torch
from .agent import TQCAgent
from .envs import get_env_dims, make_env
from .logger import create_agent
from .utils import get_device, seed_everything


def record_agent_video(
    agent: TQCAgent,
    env_id: str = "HalfCheetah-v4",
    output_path: str = "videos/rollout.gif",
    max_steps: int = 500,
    deterministic: bool = True,
    fps: int = 30,
    seed: int = 42,
) -> str:
    """Record an actor policy rollout and save as an animated GIF or MP4 video.

    Args:
        agent: Trained or initialized TQCAgent.
        env_id: Gymnasium MuJoCo environment name.
        output_path: Destination path (.gif or .mp4).
        max_steps: Maximum frames to capture in trajectory.
        deterministic: If True, uses mean policy action without exploration noise.
        fps: Playback frames per second (default 30).
        seed: Random seed for environment initial state.

    Returns:
        Path to the saved video file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    env = gym.make(env_id, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)

    frames: List[np.ndarray] = []
    total_reward = 0.0

    for step in range(max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action = agent.select_action(state, deterministic=deterministic)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        state = next_state

        if terminated or truncated:
            break

    env.close()

    if not frames:
        raise RuntimeError(f"No frames captured from environment {env_id}.")

    # Save animation (GIF or MP4)
    duration_ms = int(1000 / fps)
    if output_path.lower().endswith(".gif"):
        imageio.mimsave(output_path, frames, duration=duration_ms, loop=0)
    else:
        imageio.mimsave(output_path, frames, fps=fps)
    print(
        f"Saved rollout video ({len(frames)} frames, total reward: {total_reward:.2f}) -> {output_path}"
    )
    return output_path


def play_agent_interactive(
    agent: TQCAgent,
    env_id: str = "HalfCheetah-v4",
    n_episodes: int = 3,
    max_steps: int = 1000,
    deterministic: bool = True,
    fps: int = 60,
    seed: int = 42,
) -> None:
    """Launch native live desktop window rendering the agent's behavior."""
    env = gym.make(env_id, render_mode="human")

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        print(f"Starting Episode {ep}/{n_episodes}...")

        for step in range(max_steps):
            action = agent.select_action(state, deterministic=deterministic)
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            state = next_state

            # Regulate frame rate
            time.sleep(1.0 / fps)

            if terminated or truncated:
                break

        print(f"Episode {ep} finished. Total Return: {ep_reward:.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="TQC / MuJoCo Agent Visualizer")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4", help="Gymnasium Env ID")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained agent .pt")
    parser.add_argument("--algo", type=str, default="tqc", choices=["tqc", "sac"], help="Algorithm")
    parser.add_argument("--output", type=str, default="videos/halfcheetah_rollout.gif", help="Output video path")
    parser.add_argument("--max-steps", type=int, default=300, help="Max trajectory steps")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--interactive", action="store_true", help="Launch live interactive desktop window")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    seed_everything(args.seed)

    temp_env = make_env(args.env_id)
    state_dim, action_dim = get_env_dims(temp_env)
    temp_env.close()

    agent = create_agent(args.algo, state_dim, action_dim, device=args.device, env_id=args.env_id)
    if args.model_path and os.path.isfile(args.model_path):
        agent.load(args.model_path)
        print(f"Loaded weights from {args.model_path}")
    else:
        print("Running with initialized policy weights.")

    if args.interactive:
        play_agent_interactive(agent, env_id=args.env_id, max_steps=args.max_steps, seed=args.seed)
    else:
        record_agent_video(
            agent,
            env_id=args.env_id,
            output_path=args.output,
            max_steps=args.max_steps,
            fps=args.fps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
