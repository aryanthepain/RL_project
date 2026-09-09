import argparse
import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .agent import TQCAgent
from .envs import get_env_dims, make_env
from .logger import create_agent
from .utils import get_device, seed_everything


def draw_telemetry_hud(
    frame_rgb: np.ndarray,
    ckpt_idx: int = 1,
    total_ckpts: int = 100,
    step: int = 100,
    total_steps: int = 10000,
    t_sec: float = 0.0,
    v_x: float = 0.0,
    ep_return: float = 0.0,
    q_mean: float = 0.0,
    banner_height: int = 54,
) -> np.ndarray:
    """Render a semi-transparent HUD banner with telemetry and iteration metrics onto an RGB frame."""
    img = Image.fromarray(frame_rgb)
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Semi-transparent dark slate top bar
    draw.rectangle([(0, 0), (img.width, banner_height)], fill=(15, 23, 42, 210))

    # Font handling with fallback
    font = ImageFont.load_default()

    # Telemetry strings
    pct = (step / max(1, total_steps)) * 100.0
    text_l1 = (
        f"Iteration: Checkpoint #{ckpt_idx}/{total_ckpts} | "
        f"Training Step: {step:,}/{total_steps:,} ({pct:.1f}%)"
    )
    text_l2 = (
        f"Time: {t_sec:.1f}s | "
        f"Speed: {v_x:+.2f} m/s | "
        f"Return: {ep_return:.1f} | "
        f"Q-mean: {q_mean:.1f}"
    )

    draw.text((12, 6), text_l1, fill=(255, 255, 255, 255), font=font)
    draw.text((12, 28), text_l2, fill=(56, 189, 248, 255), font=font)

    composite = Image.alpha_composite(img.convert("RGBA"), overlay)
    return np.array(composite.convert("RGB"))


def rollout_checkpoint_with_telemetry(
    agent: TQCAgent,
    env_id: str = "HalfCheetah-v4",
    max_steps: int = 500,
    seed: int = 42,
    ckpt_idx: int = 1,
    total_ckpts: int = 100,
    step_num: int = 100,
    total_steps: int = 10000,
    overlay: bool = True,
    fps: int = 30,
    render: bool = True,
) -> Dict[str, Any]:
    """Execute evaluation rollout of an agent checkpoint while capturing per-frame telemetry.

    Args:
        agent: Trained or initialized TQCAgent.
        env_id: Gymnasium environment name.
        max_steps: Rollout frames / steps (default 500, ~16.6s at 30 fps).
        seed: Fixed evaluation seed for fair across-checkpoint comparisons.
        ckpt_idx: Checkpoint index (1-based).
        total_ckpts: Total checkpoints.
        step_num: Training step number for this checkpoint.
        total_steps: Total training steps.
        overlay: Whether to draw HUD overlay on frames.
        fps: Video playback frames per second.
        render: Whether to render RGB frames (set to False for fast metric-only rollout).

    Returns:
        Dictionary containing frames, velocities, rewards, returns, actions, q_means.
    """
    render_mode = "rgb_array" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    state, _ = env.reset(seed=seed)

    frames: List[np.ndarray] = []
    velocities: List[float] = []
    rewards: List[float] = []
    actions: List[np.ndarray] = []
    q_means: List[float] = []
    total_reward = 0.0

    for step in range(max_steps):
        raw_frame = env.render() if render else None
        action = agent.select_action(state, deterministic=True)

        # Compute critic Q-mean estimate for current (state, action)
        q_val = 0.0
        if hasattr(agent, "critic"):
            try:
                with torch.no_grad():
                    s_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    a_t = torch.as_tensor(action, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    q_quantiles = agent.critic(s_t, a_t)
                    q_val = float(q_quantiles.mean().item())
            except Exception:
                q_val = 0.0

        next_state, reward, terminated, truncated, info = env.step(action)
        v_x = float(info.get("x_velocity", 0.0))
        total_reward += float(reward)

        t_sec = step / float(fps)
        if raw_frame is not None:
            if overlay:
                frame_to_store = draw_telemetry_hud(
                    frame_rgb=raw_frame,
                    ckpt_idx=ckpt_idx,
                    total_ckpts=total_ckpts,
                    step=step_num,
                    total_steps=total_steps,
                    t_sec=t_sec,
                    v_x=v_x,
                    ep_return=total_reward,
                    q_mean=q_val,
                )
            else:
                frame_to_store = raw_frame
            frames.append(frame_to_store)

        velocities.append(v_x)
        rewards.append(float(reward))
        actions.append(action)
        q_means.append(q_val)
        state = next_state

        if terminated or truncated:
            break

    env.close()

    return {
        "frames": frames,
        "velocities": velocities,
        "rewards": rewards,
        "cumulative_returns": np.cumsum(rewards).tolist() if rewards else [],
        "actions": actions,
        "q_means": q_means,
        "total_return": total_reward,
        "ckpt_idx": ckpt_idx,
        "step_num": step_num,
    }


def tile_grid_frames(
    frames_per_video: List[List[np.ndarray]],
    rows: int = 2,
    cols: int = 3,
    border_px: int = 2,
    border_color: Tuple[int, int, int] = (30, 41, 59),
) -> List[np.ndarray]:
    """Tile multiple frame sequences into a synchronized grid video.

    Args:
        frames_per_video: List of length K, each a list of frames.
        rows: Grid row count.
        cols: Grid column count.
        border_px: Pixel width of border line between cells.
        border_color: RGB tuple for borders.

    Returns:
        List of synchronized composite frames.
    """
    total_slots = rows * cols
    if len(frames_per_video) < total_slots:
        raise ValueError(
            f"Expected at least {total_slots} video streams for {rows}x{cols} grid, got {len(frames_per_video)}"
        )

    # Synchronized temporal horizon T
    T = min(len(f) for f in frames_per_video[:total_slots])
    if T == 0:
        raise ValueError("Cannot tile empty frame lists.")

    sample_frame = frames_per_video[0][0]
    H, W, C = sample_frame.shape

    grid_frames: List[np.ndarray] = []

    # Pre-create horizontal and vertical borders
    v_border = np.full((H, border_px, C), border_color, dtype=np.uint8)
    grid_w = cols * W + (cols - 1) * border_px
    h_border = np.full((border_px, grid_w, C), border_color, dtype=np.uint8)

    for t in range(T):
        row_strips = []
        for r in range(rows):
            cells = []
            for c in range(cols):
                idx = r * cols + c
                cell_frame = frames_per_video[idx][t]
                if cell_frame.shape != (H, W, C):
                    # Resize if slight mismatch
                    cell_img = Image.fromarray(cell_frame).resize((W, H))
                    cell_frame = np.array(cell_img)
                cells.append(cell_frame)
                if c < cols - 1 and border_px > 0:
                    cells.append(v_border)
            row_strip = np.concatenate(cells, axis=1)
            row_strips.append(row_strip)
            if r < rows - 1 and border_px > 0:
                row_strips.append(h_border)

        grid_t = np.concatenate(row_strips, axis=0)
        grid_frames.append(grid_t)

    return grid_frames


def save_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> str:
    """Save a list of RGB frames as an MP4 or GIF video."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not frames:
        raise ValueError("No frames to save.")

    if output_path.lower().endswith(".gif"):
        duration_ms = int(1000 / fps)
        iio.imwrite(output_path, frames, duration=duration_ms, loop=0)
    else:
        iio.imwrite(output_path, frames, fps=fps)
    return output_path


def stitch_video_files(
    input_paths: List[str],
    output_path: str,
    fps: int = 30,
    timelapse_factor: int = 1,
) -> str:
    """Concatenate multiple video files sequentially into a single master video.

    Streams frames one video at a time to keep memory usage minimal.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not input_paths:
        raise ValueError("No input video paths provided to stitch.")

    all_frames: List[np.ndarray] = []
    for path in input_paths:
        if not os.path.isfile(path):
            continue
        try:
            frames = iio.imread(path)
            # Sample frames if timelapse is requested
            if timelapse_factor > 1:
                frames = frames[::timelapse_factor]
            all_frames.extend(frames)
        except Exception as e:
            print(f"Warning: Failed to read video {path}: {e}")

    if not all_frames:
        raise RuntimeError("No frames loaded from any input videos.")

    return save_video(all_frames, output_path, fps=fps)


def plot_progression_diagnostics(
    telemetries: List[Dict[str, Any]],
    output_path: str = "results/halfcheetah_progression_diagnostics.png",
) -> str:
    """Plot 4-panel diagnostic curves across checkpoint rollouts."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0f172a")

    for ax in axs.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")

    # Color palette
    cmap = plt.cm.viridis(np.linspace(0.2, 0.95, len(telemetries)))

    # Panel 1: Forward Velocities over time
    for idx, t in enumerate(telemetries):
        label = f"Step {t['step_num']:,}"
        axs[0, 0].plot(t["velocities"], label=label, color=cmap[idx], alpha=0.85, linewidth=1.5)
    axs[0, 0].set_title("Forward Velocity Profiles (vx)")
    axs[0, 0].set_xlabel("Rollout Step")
    axs[0, 0].set_ylabel("Velocity (m/s)")
    if len(telemetries) <= 10:
        axs[0, 0].legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)

    # Panel 2: Cumulative Return Curves
    for idx, t in enumerate(telemetries):
        axs[0, 1].plot(t["cumulative_returns"], color=cmap[idx], alpha=0.85, linewidth=1.5)
    axs[0, 1].set_title("Cumulative Return Progression")
    axs[0, 1].set_xlabel("Rollout Step")
    axs[0, 1].set_ylabel("Accumulated Reward")

    # Panel 3: Action Torque Magnitudes per Joint
    if telemetries:
        steps_list = [t["step_num"] for t in telemetries]
        mean_abs_torques = [
            float(np.mean(np.abs(np.array(t["actions"])))) if t["actions"] else 0.0
            for t in telemetries
        ]
        axs[1, 0].plot(steps_list, mean_abs_torques, marker="o", color="#38bdf8", linewidth=2)
        axs[1, 0].set_title("Mean Actuation Magnitude across Progression")
        axs[1, 0].set_xlabel("Training Step")
        axs[1, 0].set_ylabel("Mean |Action|")

    # Panel 4: Q-mean Progression
    if telemetries:
        steps_list = [t["step_num"] for t in telemetries]
        mean_qs = [
            float(np.mean(t["q_means"])) if t["q_means"] else 0.0
            for t in telemetries
        ]
        axs[1, 1].plot(steps_list, mean_qs, marker="s", color="#34d399", linewidth=2)
        axs[1, 1].set_title("Critic Q-mean Progression")
        axs[1, 1].set_xlabel("Training Step")
        axs[1, 1].set_ylabel("Estimated Q-value")

    plt.suptitle("HalfCheetah Policy Progression Diagnostics", color="#f8fafc", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return output_path


def record_agent_video(
    agent: TQCAgent,
    env_id: str = "HalfCheetah-v4",
    output_path: str = "videos/rollout.gif",
    max_steps: int = 500,
    deterministic: bool = True,
    fps: int = 30,
    seed: int = 42,
) -> str:
    """Record an actor policy rollout and save as an animated GIF or MP4 video (legacy wrapper)."""
    telemetry = rollout_checkpoint_with_telemetry(
        agent=agent,
        env_id=env_id,
        max_steps=max_steps,
        seed=seed,
        overlay=False,
        fps=fps,
    )
    return save_video(telemetry["frames"], output_path, fps=fps)


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
            time.sleep(1.0 / fps)
            if terminated or truncated:
                break
        print(f"Episode {ep} finished. Total Return: {ep_reward:.2f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="TQC / MuJoCo Progression Visualizer")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4", help="Gymnasium Env ID")
    parser.add_argument("--model-path", type=str, default=None, help="Path to single model .pt")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="List of checkpoint .pt paths")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory containing checkpoint_*.pt")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "grid", "montage", "all"], help="Visualization mode")
    parser.add_argument("--output-dir", type=str, default="videos", help="Output directory")
    parser.add_argument("--output", type=str, default=None, help="Explicit output video file")
    parser.add_argument("--max-steps", type=int, default=500, help="Max trajectory steps (500 frames ~ 16.6s)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--overlay", action="store_true", default=True, help="Draw HUD telemetry")
    parser.add_argument("--interactive", action="store_true", help="Launch live interactive desktop window")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    seed_everything(args.seed)

    temp_env = make_env(args.env_id)
    state_dim, action_dim = get_env_dims(temp_env)
    temp_env.close()

    agent = create_agent("tqc", state_dim, action_dim, device=args.device, env_id=args.env_id)

    if args.interactive:
        if args.model_path and os.path.isfile(args.model_path):
            agent.load(args.model_path)
        play_agent_interactive(agent, env_id=args.env_id, max_steps=args.max_steps, seed=args.seed)
        return

    # Single mode
    if args.mode == "single":
        if args.model_path and os.path.isfile(args.model_path):
            agent.load(args.model_path)
        out_path = args.output or os.path.join(args.output_dir, "halfcheetah_rollout.mp4")
        telemetry = rollout_checkpoint_with_telemetry(
            agent=agent,
            env_id=args.env_id,
            max_steps=args.max_steps,
            seed=args.seed,
            overlay=args.overlay,
            fps=args.fps,
        )
        saved = save_video(telemetry["frames"], out_path, fps=args.fps)
        print(f"Saved single video -> {saved}")


if __name__ == "__main__":
    main()
