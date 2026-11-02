"""Generate synchronized 2x3 progression grid comparison videos for Walker2d and Humanoid benchmarks."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

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


def generate_environment_grid(
    env_id: str,
    run_dir: str,
    output_grid_path: str,
    slots: List[Dict],
    total_steps: int = 1000000,
    frames_per_video: int = 300,
    fps: int = 30,
    eval_seed: int = 42,
) -> str:
    """Generate 6 milestone rollouts and tile them into a synchronized 2x3 comparison grid."""
    run_path = Path(run_dir).resolve()
    grid_out = Path(output_grid_path).resolve()
    grid_out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" Generating Synchronized 2x3 Progression Grid for {env_id}")
    print(f" Run Directory: {run_path}")
    print(f" Output Video:  {grid_out}")
    print(f" Frames:        {frames_per_video} frames @ {fps} fps")
    print("=" * 70)

    # Initialize environment and agent
    temp_env = make_env(env_id)
    state_dim, action_dim = get_env_dims(temp_env)
    temp_env.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = create_agent("tqc", state_dim, action_dim, device=device, env_id=env_id)

    all_video_frames: List[List[np.ndarray]] = []

    for i, slot in enumerate(slots, 1):
        ckpt_file = run_path / slot["file"]
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Missing required checkpoint: {ckpt_file}")

        print(f"\n[{i}/6] Rolling out {slot['label']} ({slot['file']})...")
        agent.load(str(ckpt_file))

        telemetry = rollout_checkpoint_with_telemetry(
            agent=agent,
            env_id=env_id,
            max_steps=frames_per_video,
            seed=eval_seed,
            ckpt_idx=i,
            total_ckpts=6,
            step_num=slot["step"],
            total_steps=total_steps,
            overlay=True,
            fps=fps,
            render=True,
            camera_name="track",
            truncate_inactive=False,
        )

        frames = telemetry["frames"]
        all_video_frames.append(frames)
        ep_return = sum(telemetry["rewards"])
        mean_vx = np.mean(telemetry["velocities"]) if telemetry["velocities"] else 0.0
        print(f" -> Slot {i} Rollout complete | Return: {ep_return:.2f} | Speed: {mean_vx:.2f} m/s | Frames: {len(frames)}")

    print(f"\n[Grid Synthesis] Tiling 6 video streams into 2x3 comparison grid...")
    grid_frames = tile_grid_frames(all_video_frames, rows=2, cols=3, border_px=3)
    save_video(grid_frames, str(grid_out), fps=fps)
    print(f"\nSaved synchronized 2x3 grid video: {grid_out}")
    return str(grid_out)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark progression grid videos.")
    parser.add_argument("--env", type=str, choices=["walker2d", "humanoid", "all"], default="all")
    parser.add_argument("--frames", type=int, default=300, help="Frames per video (default: 300)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")

    args = parser.parse_args()

    # Walker2d-v4 setup
    if args.env in ("walker2d", "all"):
        walker_run = "runs/kaggle_sharvanipallempati_tqc-walker2dv4-s42-10160158_20261016_083554"
        walker_slots = [
            {"file": "checkpoint_0.pt", "label": "Step 0 (Random Baseline)", "step": 0},
            {"file": "checkpoint_50000.pt", "label": "Step 50k (Early Balance)", "step": 50000},
            {"file": "checkpoint_250000.pt", "label": "Step 250k (Developing Gait)", "step": 250000},
            {"file": "checkpoint_500000.pt", "label": "Step 500k (Fast Striding)", "step": 500000},
            {"file": "checkpoint_750000.pt", "label": "Step 750k (Stable Locomotion)", "step": 750000},
            {"file": "best_model.pt", "label": "Step 1M (Peak Performance)", "step": 1000000},
        ]
        generate_environment_grid(
            env_id="Walker2d-v4",
            run_dir=walker_run,
            output_grid_path="videos/walker2d_6way_grid.mp4",
            slots=walker_slots,
            total_steps=1000000,
            frames_per_video=args.frames,
            fps=args.fps,
            eval_seed=args.seed,
        )

    # Humanoid-v4 setup
    if args.env in ("humanoid", "all"):
        humanoid_run = "runs/kaggle_sharvanipallempati_tqc-humanoidv4-s42"
        humanoid_slots = [
            {"file": "checkpoint_0.pt", "label": "Step 0 (Random Baseline)", "step": 0},
            {"file": "checkpoint_100000.pt", "label": "Step 100k (Upright Balance)", "step": 100000},
            {"file": "checkpoint_500000.pt", "label": "Step 500k (Forward Stepping)", "step": 500000},
            {"file": "checkpoint_1000000.pt", "label": "Step 1M (Bipedal Walk)", "step": 1000000},
            {"file": "checkpoint_1500000.pt", "label": "Step 1.5M (Dynamic Running)", "step": 1500000},
            {"file": "best_model.pt", "label": "Step 1.9M (Peak Locomotive Sprint)", "step": 1905000},
        ]
        generate_environment_grid(
            env_id="Humanoid-v4",
            run_dir=humanoid_run,
            output_grid_path="videos/humanoid_6way_grid.mp4",
            slots=humanoid_slots,
            total_steps=3000000,
            frames_per_video=args.frames,
            fps=args.fps,
            eval_seed=args.seed,
        )


if __name__ == "__main__":
    main()
