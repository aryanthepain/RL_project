"""Generate 6 individual checkpoint videos and a synchronized 2x3 grid comparison from Kaggle run."""

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.tqc.envs import get_env_dims, make_env
from src.tqc.logger import create_agent
from src.tqc.visualize import (
    rollout_checkpoint_with_telemetry,
    save_video,
    tile_grid_frames,
)


def generate_kaggle_videos(
    run_dir: str = "runs/kaggle_halfcheetahv4_s42_20260910_002935",
    output_dir: str = "videos/kaggle_smoke",
    grid_output: str = "videos/kaggle_halfcheetah_6way_grid.mp4",
    frames_per_video: int = 300,
    fps: int = 30,
) -> dict:
    """Generate 6 checkpoint videos and a synchronized 2x3 grid comparison video.

    Args:
        run_dir: Directory containing downloaded Kaggle checkpoints.
        output_dir: Target folder for individual video files.
        grid_output: Target path for the 2x3 synchronized grid comparison MP4.
        frames_per_video: Number of frames per episode rollout (default 300 = 10s at 30fps).
        fps: Frames per second.

    Returns:
        Dictionary containing paths to all generated videos.
    """
    run_path = Path(run_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.dirname(Path(grid_output).resolve()), exist_ok=True)

    print("=" * 70)
    print(" Generating 6 Videos & Synchronized 2x3 Grid from Kaggle Run")
    print(f" Source Directory: {run_path}")
    print(f" Video Output:    {out_path}")
    print(f" Grid Video:      {grid_output}")
    print("=" * 70)

    # Initialize environment and agent
    env_id = "HalfCheetah-v4"
    temp_env = make_env(env_id)
    state_dim, action_dim = get_env_dims(temp_env)
    temp_env.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = create_agent("tqc", state_dim, action_dim, device=device, env_id=env_id)

    # Define the 6 evaluation slots
    slots = [
        {
            "name": "slot1_step0",
            "file": run_path / "checkpoint_0.pt",
            "label": "Step 0 (Random Baseline)",
            "step": 0,
            "seed": 42,
        },
        {
            "name": "slot2_step50",
            "file": run_path / "checkpoint_50.pt",
            "label": "Step 50 Checkpoint",
            "step": 50,
            "seed": 42,
        },
        {
            "name": "slot3_step100",
            "file": run_path / "checkpoint_100.pt",
            "label": "Step 100 Checkpoint",
            "step": 100,
            "seed": 42,
        },
        {
            "name": "slot4_best_model_s42",
            "file": run_path / "best_model.pt",
            "label": "Best Model (Eval Seed 42)",
            "step": 50,
            "seed": 42,
        },
        {
            "name": "slot5_best_model_s142",
            "file": run_path / "best_model.pt",
            "label": "Best Model (Eval Seed 142)",
            "step": 50,
            "seed": 142,
        },
        {
            "name": "slot6_final_model",
            "file": run_path / "final_model.pt",
            "label": "Final Model (Step 100)",
            "step": 100,
            "seed": 42,
        },
    ]

    all_video_frames: List[List[np.ndarray]] = []
    generated_videos: List[str] = []

    for i, slot in enumerate(slots, 1):
        ckpt_file = slot["file"]
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Missing required checkpoint file: {ckpt_file}")

        print(f"\n[{i}/6] Rolling out {slot['label']} ({slot['name']})...")
        agent.load(str(ckpt_file))

        telemetry = rollout_checkpoint_with_telemetry(
            agent=agent,
            env_id=env_id,
            max_steps=frames_per_video,
            seed=slot["seed"],
            ckpt_idx=i,
            total_ckpts=6,
            step_num=slot["step"],
            total_steps=100,
            overlay=True,
            fps=fps,
            render=True,
        )

        frames = telemetry["frames"]
        all_video_frames.append(frames)

        video_file = out_path / f"{slot['name']}.mp4"
        save_video(frames, str(video_file), fps=fps)
        generated_videos.append(str(video_file))

        ep_return = sum(telemetry["rewards"])
        mean_vx = np.mean(telemetry["velocities"]) if telemetry["velocities"] else 0.0
        print(f" -> Saved: {video_file.name} | Return: {ep_return:.2f} | Speed: {mean_vx:.2f} m/s")

    # Assemble the synchronized 2x3 comparison grid
    print(f"\n[Grid Synthesis] Assembling 6 video streams into synchronized 2x3 grid...")
    grid_frames = tile_grid_frames(all_video_frames, rows=2, cols=3, border_px=3)
    save_video(grid_frames, grid_output, fps=fps)
    print(f" -> Saved 2x3 Grid Comparison Video: {grid_output}")

    print("\n" + "=" * 70)
    print(" Generation Completed Successfully!")
    print(f" - 6 Individual Videos: {out_path}")
    print(f" - Synchronized 2x3 Grid: {grid_output}")
    print("=" * 70)

    return {
        "videos": generated_videos,
        "grid_video": str(Path(grid_output).resolve()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate 6 individual videos and a synchronized 2x3 grid comparison from Kaggle run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/kaggle_halfcheetahv4_s42_20260910_002935",
        help="Path to downloaded Kaggle run directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos/kaggle_smoke",
        help="Directory to save individual videos",
    )
    parser.add_argument(
        "--grid-output",
        type=str,
        default="videos/kaggle_halfcheetah_6way_grid.mp4",
        help="Path for final synchronized 2x3 grid MP4",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Frames per video (default: 300 = 10s at 30 fps)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback FPS (default: 30)",
    )

    args = parser.parse_args()

    generate_kaggle_videos(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        grid_output=args.grid_output,
        frames_per_video=args.frames,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
