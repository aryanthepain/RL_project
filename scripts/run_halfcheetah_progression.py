import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import glob
import re
import time
from typing import List, Tuple
import numpy as np
import torch
from src.tqc.envs import get_env_dims, make_env
from src.tqc.logger import create_agent
from src.tqc.train import train_tqc
from src.tqc.visualize import (
    plot_progression_diagnostics,
    rollout_checkpoint_with_telemetry,
    save_video,
    stitch_video_files,
    tile_grid_frames,
)


def get_checkpoint_step(filename: str) -> int:
    """Extract step integer from checkpoint filename (e.g. checkpoint_100.pt -> 100)."""
    match = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(filename))
    return int(match.group(1)) if match else -1


def run_progression_pipeline(
    total_timesteps: int = 10_000,
    checkpoint_freq: int = 100,
    eval_frames: int = 500,
    fps: int = 30,
    seed: int = 42,
    output_dir: str = "videos",
    checkpoint_dir: str = "runs/progression_run",
    skip_train: bool = False,
    smoke_test: bool = False,
    device: str = "cpu",
    timelapse_factor: int = 1,
) -> dict:
    """Execute complete HalfCheetah progression training, rollouts, grid compositing, and montage."""
    if smoke_test:
        total_timesteps = 200
        checkpoint_freq = 100
        eval_frames = 10
        print("[SMOKE-TEST] Running fast regression test mode (200 steps, 10 frames per rollout)...")

    os.makedirs(output_dir, exist_ok=True)
    ckpt_video_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_video_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Step 1: Train or discover checkpoints
    if not skip_train:
        print(
            f"\n=== Step 1: Training HalfCheetah ({total_timesteps:,} steps, checkpoint every {checkpoint_freq} steps) ==="
        )
        # Ensure warmup_steps >= batch_size
        batch_sz = 32 if total_timesteps < 1000 or smoke_test else 256
        warmup = max(batch_sz, min(1000, total_timesteps // 4))

        train_tqc(
            env_id="HalfCheetah-v4",
            seed=seed,
            total_timesteps=total_timesteps,
            checkpoint_freq=checkpoint_freq,
            eval_freq=max(1000, checkpoint_freq * 5),
            warmup_steps=warmup,
            batch_size=batch_sz,
            device=device,
            log_dir=os.path.dirname(os.path.abspath(checkpoint_dir)),
            exp_name=os.path.basename(os.path.abspath(checkpoint_dir)),
        )

    # Locate checkpoints
    pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    ckpt_files = glob.glob(pattern)
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files matching {pattern} found.")

    # Filter and sort numerically
    valid_ckpts: List[Tuple[int, str]] = []
    for f in ckpt_files:
        step = get_checkpoint_step(f)
        if step >= 0:
            valid_ckpts.append((step, f))
    valid_ckpts.sort(key=lambda x: x[0])

    # If checkpoint_0 exists and we want steps > 0, keep or include according to total_ckpts
    # Filter out step 0 if user wants strictly 100 checkpoints from 100..10000, or include it
    ckpts_to_evaluate = [f for s, f in valid_ckpts if s > 0]
    if not ckpts_to_evaluate:
        ckpts_to_evaluate = [f for _, f in valid_ckpts]

    total_ckpts = len(ckpts_to_evaluate)
    print(f"\n=== Step 2: Found {total_ckpts} checkpoints to rollout and record ===")

    # Initialize agent for evaluation
    temp_env = make_env("HalfCheetah-v4")
    state_dim, action_dim = get_env_dims(temp_env)
    temp_env.close()

    agent = create_agent("tqc", state_dim, action_dim, device=device, env_id="HalfCheetah-v4")

    # Determine 6 milestone checkpoint indices for 2x3 grid
    if total_ckpts >= 6:
        # Checkpoints #1, #20, #40, #60, #80, #100 (or evenly spaced)
        grid_target_indices = [
            0,
            int(round(total_ckpts * 0.2)) - 1,
            int(round(total_ckpts * 0.4)) - 1,
            int(round(total_ckpts * 0.6)) - 1,
            int(round(total_ckpts * 0.8)) - 1,
            total_ckpts - 1,
        ]
    else:
        # Fewer than 6 checkpoints (e.g. smoke test) - duplicate last to reach 6 slots
        grid_target_indices = list(range(total_ckpts))
        while len(grid_target_indices) < 6:
            grid_target_indices.append(grid_target_indices[-1])

    grid_target_indices = [max(0, min(total_ckpts - 1, i)) for i in grid_target_indices]
    grid_frame_streams: List[List[np.ndarray]] = [[] for _ in range(6)]

    saved_video_paths: List[str] = []
    telemetry_records: List[dict] = []

    print(f"Rolling out {total_ckpts} checkpoints ({eval_frames} frames each)...")
    for idx_0, ckpt_path in enumerate(ckpts_to_evaluate):
        ckpt_idx = idx_0 + 1
        step_num = get_checkpoint_step(ckpt_path)
        agent.load(ckpt_path)

        telemetry = rollout_checkpoint_with_telemetry(
            agent=agent,
            env_id="HalfCheetah-v4",
            max_steps=eval_frames,
            seed=seed,
            ckpt_idx=ckpt_idx,
            total_ckpts=total_ckpts,
            step_num=step_num,
            total_steps=total_timesteps,
            overlay=True,
            fps=fps,
        )
        telemetry_records.append(telemetry)

        # Save individual checkpoint video
        video_out = os.path.join(ckpt_video_dir, f"ckpt_{step_num:05d}.mp4")
        save_video(telemetry["frames"], video_out, fps=fps)
        saved_video_paths.append(video_out)

        # Record for 6-way grid if in milestone list
        for slot_idx, target_idx in enumerate(grid_target_indices):
            if idx_0 == target_idx and not grid_frame_streams[slot_idx]:
                grid_frame_streams[slot_idx] = telemetry["frames"]

        if ckpt_idx % 10 == 0 or ckpt_idx == total_ckpts or smoke_test:
            print(
                f"[{ckpt_idx}/{total_ckpts}] Saved {video_out} | Return: {telemetry['total_return']:.1f} | Mean Speed: {np.mean(telemetry['velocities']):+.2f} m/s"
            )

    # Step 3: Create Synchronized 6-way 2x3 Grid Video
    print("\n=== Step 3: Synthesizing Synchronized 6-Way 2x3 Grid Comparison Video ===")
    grid_out_path = os.path.join(output_dir, "halfcheetah_6way_grid.mp4")
    # Verify all 6 slots are populated
    for i in range(6):
        if not grid_frame_streams[i]:
            grid_frame_streams[i] = grid_frame_streams[max(0, i - 1)]

    tiled_frames = tile_grid_frames(grid_frame_streams, rows=2, cols=3, border_px=3)
    save_video(tiled_frames, grid_out_path, fps=fps)
    print(f"Saved 6-Way Grid Video -> {grid_out_path}")

    # Step 4: Stitch All Checkpoints into Master Chronological Montage
    print("\n=== Step 4: Stitching Chronological Progression Master Montage ===")
    montage_out_path = os.path.join(output_dir, "halfcheetah_progression_montage.mp4")
    stitch_video_files(
        input_paths=saved_video_paths,
        output_path=montage_out_path,
        fps=fps,
        timelapse_factor=timelapse_factor,
    )
    print(f"Saved Master Montage -> {montage_out_path}")

    # Step 5: Behavioral Diagnostics Plot
    print("\n=== Step 5: Generating Progression Diagnostics Plot ===")
    diag_out_path = "results/halfcheetah_progression_diagnostics.png"
    # Select up to 10 spaced telemetries for legible velocity/return plots
    stride = max(1, len(telemetry_records) // 8)
    sample_indices = list(range(0, len(telemetry_records), stride))
    if (len(telemetry_records) - 1) not in sample_indices:
        sample_indices.append(len(telemetry_records) - 1)
    sampled_telemetries = [telemetry_records[i] for i in sample_indices]

    plot_progression_diagnostics(sampled_telemetries, diag_out_path)
    print(f"Saved Diagnostics Plot -> {diag_out_path}")

    summary = {
        "checkpoints_evaluated": total_ckpts,
        "individual_videos": len(saved_video_paths),
        "grid_video": grid_out_path,
        "montage_video": montage_out_path,
        "diagnostics_plot": diag_out_path,
    }
    print("\n=== Progression Pipeline Completed Successfully! ===")
    for k, v in summary.items():
        print(f" - {k}: {v}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="HalfCheetah Policy Progression Pipeline Runner")
    parser.add_argument("--total-timesteps", type=int, default=10_000, help="Total training steps")
    parser.add_argument("--checkpoint-freq", type=int, default=100, help="Step interval for saving checkpoints")
    parser.add_argument("--eval-frames", type=int, default=500, help="Frames per checkpoint rollout (~16.6s at 30 fps)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed")
    parser.add_argument("--output-dir", type=str, default="videos", help="Output videos directory")
    parser.add_argument("--checkpoint-dir", type=str, default="runs/progression_run", help="Checkpoint directory")
    parser.add_argument("--skip-train", action="store_true", help="Skip training; evaluate existing checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Fast smoke test mode")
    parser.add_argument("--device", type=str, default="cpu", help="Torch compute device")
    parser.add_argument("--timelapse-factor", type=int, default=1, help="Timelapse sampling factor for montage")
    parser.add_argument("--detach", action="store_true", help="Launch in detached background process")

    args = parser.parse_args()
    if args.detach:
        import subprocess
        os.makedirs("logs", exist_ok=True)
        log_f = open("logs/progression_generation.log", "w", encoding="utf-8")
        child_args = [sys.executable, "-u", os.path.abspath(__file__)] + [
            a for a in sys.argv[1:] if a != "--detach"
        ]
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        proc = subprocess.Popen(
            child_args,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        print(f"Spawned progression pipeline in background process (PID: {proc.pid})")
        print("Log destination: logs/progression_generation.log")
        return

    kwargs = {k: v for k, v in vars(args).items() if k != "detach"}
    run_progression_pipeline(**kwargs)


if __name__ == "__main__":
    main()
