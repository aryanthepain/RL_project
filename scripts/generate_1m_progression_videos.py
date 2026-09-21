"""Generate individual milestone videos and a synchronized 2x3 grid comparison from the 1M pilot run."""

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


def generate_1m_progression_videos(
    run_dir: str = "runs/kaggle_halfcheetah_pilot_1m_s42",
    output_dir: str = "videos/halfcheetah_1m_progression",
    grid_output: str = "videos/halfcheetah_1m_6way_grid.mp4",
    frames_per_video: int = 500,
    fps: int = 30,
    env_id: str = "HalfCheetah-v4",
    eval_seed: int = 42,
) -> dict:
    """Generate 6 milestone videos and a synchronized 2x3 grid comparison from the 1M run.

    Args:
        run_dir: Directory containing downloaded 1M run checkpoints.
        output_dir: Target folder for individual video files.
        grid_output: Target path for the 2x3 synchronized grid comparison MP4.
        frames_per_video: Number of frames per episode rollout (default: 500 = ~16.6s at 30fps).
        fps: Frames per second.
        env_id: Gymnasium environment ID.
        eval_seed: Seed for evaluation rollouts (identical across slots for fair comparison).

    Returns:
        Dictionary containing paths to all generated videos.
    """
    run_path = Path(run_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.dirname(Path(grid_output).resolve()), exist_ok=True)

    print("=" * 70)
    print(" Generating 6 Milestone Videos & Synchronized 2x3 Grid from 1M Run")
    print(f" Source Directory: {run_path}")
    print(f" Video Output:    {out_path}")
    print(f" Grid Video:      {grid_output}")
    print(f" Frames per Video: {frames_per_video} @ {fps} fps")
    print("=" * 70)

    # Initialize environment and agent
    temp_env = make_env(env_id)
    state_dim, action_dim = get_env_dims(temp_env)
    temp_env.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = create_agent("tqc", state_dim, action_dim, device=device, env_id=env_id)

    # Define the 6 evaluation slots across the 1,000,000 steps
    slots = [
        {
            "name": "milestone_1_step0",
            "file": run_path / "checkpoint_0.pt",
            "label": "Step 0 (Random Baseline)",
            "step": 0,
        },
        {
            "name": "milestone_2_step50k",
            "file": run_path / "checkpoint_50000.pt",
            "label": "Step 50k (Initial Gait)",
            "step": 50000,
        },
        {
            "name": "milestone_3_step250k",
            "file": run_path / "checkpoint_250000.pt",
            "label": "Step 250k (Fast Striding)",
            "step": 250000,
        },
        {
            "name": "milestone_4_step500k",
            "file": run_path / "checkpoint_500000.pt",
            "label": "Step 500k (High-Speed Gallop)",
            "step": 500000,
        },
        {
            "name": "milestone_5_step750k",
            "file": run_path / "checkpoint_750000.pt",
            "label": "Step 750k (Asymptotic Stabilization)",
            "step": 750000,
        },
        {
            "name": "milestone_6_step1m",
            "file": run_path / "best_model.pt" if (run_path / "best_model.pt").is_file() else run_path / "checkpoint_1000000.pt",
            "label": "Step 1M (Peak Asymptotic Sprint)",
            "step": 1000000,
        },
    ]

    all_video_frames: List[List[np.ndarray]] = []
    generated_videos: List[str] = []

    for i, slot in enumerate(slots, 1):
        ckpt_file = slot["file"]
        if not ckpt_file.is_file():
            # Check nested dir if exists
            nested = run_path / "kaggle_halfcheetah_pilot_s42" / ckpt_file.name
            if nested.is_file():
                ckpt_file = nested
            else:
                raise FileNotFoundError(f"Missing required checkpoint file: {ckpt_file}")

        print(f"\n[{i}/6] Rolling out {slot['label']} ({slot['name']})...")
        agent.load(str(ckpt_file))

        telemetry = rollout_checkpoint_with_telemetry(
            agent=agent,
            env_id=env_id,
            max_steps=frames_per_video,
            seed=eval_seed,
            ckpt_idx=i,
            total_ckpts=6,
            step_num=slot["step"],
            total_steps=1000000,
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
    print(f"\n[Grid Synthesis] Assembling 6 video streams into synchronized 2x3 comparison grid...")
    grid_frames = tile_grid_frames(all_video_frames, rows=2, cols=3, border_px=3)
    save_video(grid_frames, grid_output, fps=fps)
    print(f" -> Saved 2x3 Grid Comparison Video: {grid_output}")

    print("\n" + "=" * 70)
    print(" 1M Progression Video Generation Completed Successfully!")
    print(f" - 6 Milestone Videos: {out_path}")
    print(f" - Synchronized 2x3 Grid: {grid_output}")
    print("=" * 70)

    return {
        "videos": generated_videos,
        "grid_video": str(Path(grid_output).resolve()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate 6 milestone videos and a synchronized 2x3 grid comparison from the 1M run."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/kaggle_halfcheetah_pilot_1m_s42",
        help="Path to 1M run directory containing checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos/halfcheetah_1m_progression",
        help="Directory to save individual milestone videos",
    )
    parser.add_argument(
        "--grid-output",
        type=str,
        default="videos/halfcheetah_1m_6way_grid.mp4",
        help="Path for final synchronized 2x3 grid MP4",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=500,
        help="Frames per video (default: 500 = ~16.6s at 30 fps)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback FPS (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation seed (default: 42)",
    )

    args = parser.parse_args()

    generate_1m_progression_videos(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        grid_output=args.grid_output,
        frames_per_video=args.frames,
        fps=args.fps,
        eval_seed=args.seed,
    )


if __name__ == "__main__":
    main()
