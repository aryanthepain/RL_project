"""Render an evaluation rollout video with telemetry HUD from a TQC checkpoint."""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.tqc.envs import get_env_dims, make_env
from src.tqc.logger import create_agent
from src.tqc.visualize import rollout_checkpoint_with_telemetry, save_video


def render_video_from_checkpoint(
    checkpoint_path: str,
    output_path: str,
    env_id: str = "HalfCheetah-v4",
    max_steps: int = 500,
    fps: int = 30,
    seed: int = 42,
    step_num: int = 50000,
    total_steps: int = 50000,
) -> str:
    """Load a checkpoint, execute a deterministic rollout with HUD, and save as MP4."""
    ckpt_file = Path(checkpoint_path).resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" Rendering TQC Rollout Video: {env_id}")
    print(f" Checkpoint:  {ckpt_file}")
    print(f" Output Video: {out_file}")
    print(f" Steps / FPS:  {max_steps} frames @ {fps} fps")
    print("=" * 70)

    # Initialize environment to obtain dimensions
    temp_env = make_env(env_id)
    state_dim, action_dim = get_env_dims(temp_env)
    temp_env.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = create_agent("tqc", state_dim, action_dim, device=device, env_id=env_id)

    print(f"\nLoading weights from: {ckpt_file.name}...")
    agent.load(str(ckpt_file))

    print("\nExecuting evaluation rollout with telemetry HUD...")
    telemetry = rollout_checkpoint_with_telemetry(
        agent=agent,
        env_id=env_id,
        max_steps=max_steps,
        seed=seed,
        ckpt_idx=1,
        total_ckpts=1,
        step_num=step_num,
        total_steps=total_steps,
        overlay=True,
        fps=fps,
        render=True,
    )

    frames = telemetry["frames"]
    save_video(frames, str(out_file), fps=fps)

    ep_return = sum(telemetry["rewards"])
    mean_vx = np.mean(telemetry["velocities"]) if telemetry["velocities"] else 0.0
    print(f"\nVideo rendered successfully: {out_file}")
    print(f"Total Return: {ep_return:.2f} | Mean Forward Speed: {mean_vx:.2f} m/s | Total Frames: {len(frames)}")
    return str(out_file)


def main():
    parser = argparse.ArgumentParser(description="Render video from a TQC checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output", type=str, default="videos/halfcheetah_50k.mp4", help="Output MP4 file path")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4", help="Gymnasium environment ID")
    parser.add_argument("--frames", type=int, default=500, help="Frames to render (default: 500)")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--step", type=int, default=50000, help="Step number for HUD display")
    parser.add_argument("--total-steps", type=int, default=50000, help="Total training steps for HUD display")

    args = parser.parse_args()
    render_video_from_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        env_id=args.env,
        max_steps=args.frames,
        fps=args.fps,
        seed=args.seed,
        step_num=args.step,
        total_steps=args.total_steps,
    )


if __name__ == "__main__":
    main()
