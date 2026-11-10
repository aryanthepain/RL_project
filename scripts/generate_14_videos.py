#!/usr/bin/env python3
"""Batch Video Generation Pipeline: 14 Videos (6 Checkpoints + 1 Main Best Model per Phase)

Renders high-definition evaluation videos with real-time HUD telemetry and
dynamic 3D camera tracking that follows the robot's torso continuously.

Phases:
  1. Hopper-v4 (Phase 9 - Seed 43, Top Seed)
     - 0k, 50k, 100k, 250k, 500k, 1M, and best_model
  2. Ant-v4 (Phase 11 - Seed 42, Top Seed)
     - 0k, 50k, 100k, 250k, 500k, 1M, and best_model
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.tqc.envs import get_env_dims, make_env
from src.tqc.logger import create_agent
from src.tqc.visualize import rollout_checkpoint_with_telemetry, save_video

TARGET_VIDEOS = [
    # Phase 9: Hopper-v4 (Seed 43)
    {
        "phase": 9,
        "env_id": "Hopper-v4",
        "ckpt_path": "d:/RL_project-main/runs/Hopper-v4/tqc_Hopper-v4_seed43/checkpoint_0.pt",
        "out_path": "d:/RL_project-main/results/Hopper-v4/hopper_eval_0k.mp4",
        "step_num": 0,
        "ckpt_label": "0k (Untrained)",
        "badge": "CKPT 0",
    },
    {
        "phase": 9,
        "env_id": "Hopper-v4",
        "ckpt_path": "d:/RL_project-main/runs/Hopper-v4/tqc_Hopper-v4_seed43/checkpoint_50000.pt",
        "out_path": "d:/RL_project-main/results/Hopper-v4/hopper_eval_50k.mp4",
        "step_num": 50000,
        "ckpt_label": "50k",
        "badge": "CKPT 50k",
    },
    {
        "phase": 9,
        "env_id": "Hopper-v4",
        "ckpt_path": "d:/RL_project-main/runs/Hopper-v4/tqc_Hopper-v4_seed43/checkpoint_100000.pt",
        "out_path": "d:/RL_project-main/results/Hopper-v4/hopper_eval_100k.mp4",
        "step_num": 100000,
        "ckpt_label": "100k",
        "badge": "CKPT 100k",
    },
    {
        "phase": 9,
        "env_id": "Hopper-v4",
        "ckpt_path": "d:/RL_project-main/runs/Hopper-v4/tqc_Hopper-v4_seed43/checkpoint_250000.pt",
        "out_path": "d:/RL_project-main/results/Hopper-v4/hopper_eval_250k.mp4",
        "step_num": 250000,
        "ckpt_label": "250k",
        "badge": "CKPT 250k",
    },
    {
        "phase": 9,
        "env_id": "Hopper-v4",
        "ckpt_path": "d:/RL_project-main/runs/Hopper-v4/tqc_Hopper-v4_seed43/checkpoint_500000.pt",
        "out_path": "d:/RL_project-main/results/Hopper-v4/hopper_eval_500k.mp4",
        "step_num": 500000,
        "ckpt_label": "500k",
        "badge": "CKPT 500k",
    },
    {
        "phase": 9,
        "env_id": "Hopper-v4",
        "ckpt_path": "d:/RL_project-main/runs/Hopper-v4/tqc_Hopper-v4_seed43/checkpoint_1000000.pt",
        "out_path": "d:/RL_project-main/results/Hopper-v4/hopper_eval_1M.mp4",
        "step_num": 1000000,
        "ckpt_label": "1M (Final)",
        "badge": "CKPT 1M",
    },
    {
        "phase": 9,
        "env_id": "Hopper-v4",
        "ckpt_path": "d:/RL_project-main/runs/Hopper-v4/tqc_Hopper-v4_seed43/best_model.pt",
        "out_path": "d:/RL_project-main/results/Hopper-v4/hopper_eval_best.mp4",
        "step_num": 920000,
        "ckpt_label": "Peak Best Model",
        "badge": "BEST MODEL",
    },
    # Phase 11: Ant-v4 (Seed 42)
    {
        "phase": 11,
        "env_id": "Ant-v4",
        "ckpt_path": "d:/RL_project-main/runs/Ant-v4/tqc_Ant-v4_seed42/checkpoint_0.pt",
        "out_path": "d:/RL_project-main/results/Ant-v4/ant_eval_0k.mp4",
        "step_num": 0,
        "ckpt_label": "0k (Untrained)",
        "badge": "CKPT 0",
    },
    {
        "phase": 11,
        "env_id": "Ant-v4",
        "ckpt_path": "d:/RL_project-main/runs/Ant-v4/tqc_Ant-v4_seed42/checkpoint_50000.pt",
        "out_path": "d:/RL_project-main/results/Ant-v4/ant_eval_50k.mp4",
        "step_num": 50000,
        "ckpt_label": "50k",
        "badge": "CKPT 50k",
    },
    {
        "phase": 11,
        "env_id": "Ant-v4",
        "ckpt_path": "d:/RL_project-main/runs/Ant-v4/tqc_Ant-v4_seed42/checkpoint_100000.pt",
        "out_path": "d:/RL_project-main/results/Ant-v4/ant_eval_100k.mp4",
        "step_num": 100000,
        "ckpt_label": "100k",
        "badge": "CKPT 100k",
    },
    {
        "phase": 11,
        "env_id": "Ant-v4",
        "ckpt_path": "d:/RL_project-main/runs/Ant-v4/tqc_Ant-v4_seed42/checkpoint_250000.pt",
        "out_path": "d:/RL_project-main/results/Ant-v4/ant_eval_250k.mp4",
        "step_num": 250000,
        "ckpt_label": "250k",
        "badge": "CKPT 250k",
    },
    {
        "phase": 11,
        "env_id": "Ant-v4",
        "ckpt_path": "d:/RL_project-main/runs/Ant-v4/tqc_Ant-v4_seed42/checkpoint_500000.pt",
        "out_path": "d:/RL_project-main/results/Ant-v4/ant_eval_500k.mp4",
        "step_num": 500000,
        "ckpt_label": "500k",
        "badge": "CKPT 500k",
    },
    {
        "phase": 11,
        "env_id": "Ant-v4",
        "ckpt_path": "d:/RL_project-main/runs/Ant-v4/tqc_Ant-v4_seed42/checkpoint_1000000.pt",
        "out_path": "d:/RL_project-main/results/Ant-v4/ant_eval_1M.mp4",
        "step_num": 1000000,
        "ckpt_label": "1M (Final)",
        "badge": "CKPT 1M",
    },
    {
        "phase": 11,
        "env_id": "Ant-v4",
        "ckpt_path": "d:/RL_project-main/runs/Ant-v4/tqc_Ant-v4_seed42/best_model.pt",
        "out_path": "d:/RL_project-main/results/Ant-v4/ant_eval_best.mp4",
        "step_num": 980000,
        "ckpt_label": "Peak Best Model",
        "badge": "BEST MODEL",
    },
]


def render_all_videos(max_steps: int = 500, fps: int = 30) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"  LAUNCHING BATCH 14-VIDEO GENERATION PIPELINE")
    print(f"  Dynamic 3D Camera Tracking: ENABLED (Robot Torso Centered)")
    print(f"  HUD Telemetry Overlay:      ENABLED")
    print(f"  Max Steps per Video:        {max_steps} (~{max_steps / fps:.1f}s @ {fps} fps)")
    print(f"  Compute Device:             {device}")
    print("=" * 80)

    # Cache loaded agents per env_id
    agents = {}
    for item in TARGET_VIDEOS:
        env_id = item["env_id"]
        if env_id not in agents:
            temp_env = make_env(env_id)
            s_dim, a_dim = get_env_dims(temp_env)
            temp_env.close()
            agents[env_id] = create_agent("tqc", s_dim, a_dim, device=device, env_id=env_id)

    total = len(TARGET_VIDEOS)
    start_time = time.time()

    for idx, item in enumerate(TARGET_VIDEOS, 1):
        env_id = item["env_id"]
        ckpt = item["ckpt_path"]
        out = item["out_path"]
        step = item["step_num"]
        label = item["ckpt_label"]
        badge = item["badge"]

        print(f"\n[{idx}/{total}] Phase {item['phase']} ({env_id}) -> {label}")
        print(f"      Source: {Path(ckpt).name}")
        print(f"      Target: {Path(out).name}")

        agent = agents[env_id]
        agent.load(ckpt)

        os.makedirs(os.path.dirname(out), exist_ok=True)
        telemetry = rollout_checkpoint_with_telemetry(
            agent=agent,
            env_id=env_id,
            max_steps=max_steps,
            seed=42,
            ckpt_idx=idx if idx <= 7 else idx - 7,
            total_ckpts=7,
            step_num=step,
            total_steps=1_000_000,
            overlay=True,
            fps=fps,
            render=True,
            truncate_inactive=True,
        )

        frames = telemetry["frames"]
        save_video(frames, out, fps=fps)

        ep_ret = sum(telemetry["rewards"])
        mean_v = np.mean(telemetry["velocities"]) if telemetry["velocities"] else 0.0
        print(f"      [OK] Saved {len(frames)} frames | Return: {ep_ret:.1f} | Speed: {mean_v:.2f} m/s")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  All 14 Videos Successfully Generated in {elapsed:.1f}s!")
    print(f"  Hopper-v4 Videos (7): d:\\RL_project-main\\results\\Hopper-v4\\")
    print(f"  Ant-v4 Videos (7):    d:\\RL_project-main\\results\\Ant-v4\\")
    print("=" * 80)


if __name__ == "__main__":
    render_all_videos()
