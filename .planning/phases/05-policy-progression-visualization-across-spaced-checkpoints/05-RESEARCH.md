# Phase 05: Policy Progression Visualization Across Spaced Checkpoints - Research

**Researched:** 2026-09-09
**Domain:** Deep RL Policy Visualization, Frame Telemetry HUD, Video Compositing, Headless Multi-Checkpoint Rollouts
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md & User Instructions)

### Locked Decisions
- **D-01:** Each of the 100 checkpoint evaluation rollouts records 15–20 seconds (~500 frames at 30 fps) to capture gait locomotion, forward velocity, and stability dynamics.
- **D-02:** Checkpoints are captured every 100 training steps over a 10,000-step training regime, yielding exactly 100 spaced checkpoint model files (`checkpoint_100.pt`, ..., `checkpoint_10000.pt`).
- **D-03:** All 100 individual full 15–20s videos are saved in `videos/checkpoints/` (as `.mp4` / `.gif`).
- **D-04:** A master chronological progression reel is stitched end-to-end combining all 100 videos in sequence, with an optional accelerated timelapse flag for compact viewing.
- **D-05:** Full diagnostic on-screen HUD banner rendered across the top of each video frame:
  - Checkpoint Iteration: `Checkpoint #{idx}/100`
  - Training Timestep: `Step {step}/10,000 ({pct}%)`
  - Rollout Time: `t = {t_sec:.1f}s`
  - Instantaneous Forward Velocity: `v_x = {v_x:+.2f} m/s`
  - Cumulative Episode Return: `Return = {reward:.1f}`
  - Critic Q-Value Estimate: `Q-mean = {q_mean:.1f}`
- **D-06:** 6-Way 2x3 Synchronized Grid comparison video running under identical evaluation seeds, featuring Checkpoints #1 (100 steps), #20 (2,000 steps), #40 (4,000 steps), #60 (6,000 steps), #80 (8,000 steps), and #100 (10,000 steps).
- **CRITICAL OPERATIONAL CONSTRAINT:** The progression generation script must be launched in an independent background process / separate terminal (with decoupled logging to `logs/progression_generation.log` and background process isolation) so that even when Phase 6 begins in the agent workspace, the video rendering and stitching pipeline runs to completion uninterrupted.

### The Agent's Discretion
- HUD styling: semi-transparent dark banner (`rgba(0, 0, 0, 180)`), high-contrast white text, legible font size via PIL `ImageDraw`.
- Compositing: auto-layout grid logic for 2x3 arrangement with border separators and title headers.
- CLI flags: `--checkpoints`, `--checkpoint-dir`, `--mode`, `--overlay`, `--grid-cols`, `--timelapse-fps`.

### Deferred Ideas (OUT OF SCOPE)
- Multi-tier cloud execution offloading (Kaggle/Colab) -> Reserved for Phase 6 (GitHub Issue #4).
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

Single-tier Python RL framework with local filesystem artifact outputs.

| Capability | Primary Module | Secondary Module | Rationale |
|------------|---------------|------------------|-----------|
| Spaced Checkpoint Saving | `src/tqc/train.py` | `src/tqc/agent.py` | Saves model weights every N steps during training loop |
| Telemetry Rollout & HUD | `src/tqc/visualize.py` | `PIL.ImageDraw` | Runs policy, computes Q-quantiles, renders HUD onto RGB frames |
| Grid & Montage Compositing | `src/tqc/visualize.py` | `imageio.v3` / `numpy` | Tiles frames into 2x3 grid and stitches 100 videos chronologically |
| Decoupled Headless Runner | `scripts/run_halfcheetah_progression.py` | `pwsh / subprocess` | Manages background execution, logging, and status tracking |
</architectural_responsibility_map>

<research_summary>
## Summary

This phase delivers a policy progression visualization suite for HalfCheetah. The goal is to visually document how the TQC agent evolves from random thrashing (step 100) through early posture learning (steps 2k–4k) to smooth forward gallop (step 10k).

The system executes 10,000 training steps, taking snapshots every 100 steps. For each snapshot, a 500-frame (16.6s at 30 fps) evaluation rollout is performed under an identical fixed evaluation seed (`seed=42`). A dual-row semi-transparent HUD banner is superimposed over every frame showing iteration index, global step, elapsed time, forward speed, return, and Q-mean.

All 100 videos are saved individually in `videos/checkpoints/` and then stitched into a continuous chronological master reel. In parallel, a synchronized 6-way 2x3 grid comparison video is synthesized from checkpoints #1, #20, #40, #60, #80, #100 running simultaneously side-by-side. To ensure the user can immediately advance to Phase 6, the execution script is designed for background/detached daemon execution writing to `logs/progression_generation.log`.

**Primary recommendation:** Build lightweight, vectorized frame compositing in `visualize.py` using NumPy array slicing and PIL font rendering, orchestrated by `scripts/run_halfcheetah_progression.py` which supports both sync and detached background execution.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `gymnasium[mujoco]` | 1.1.1 | MuJoCo simulation & RGB array rendering | Standard RL environment framework |
| `imageio` | 2.37.0 | Video (`.mp4`) and animation (`.gif`) encoding | Robust, fast, zero-system-dependency encoder |
| `Pillow (PIL)` | 11.1.0 | HUD telemetry overlay rendering | Sub-millisecond text drawing with alpha compositing |
| `numpy` | 2.2.3 | Frame matrix slicing, padding, and grid tiling | High-performance memory-contiguous array operations |
| `torch` | 2.6.0 | TQC policy inference & critic quantile estimation | Core algorithm framework |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `matplotlib` | 3.10.0 | Progression telemetry diagnostics plot | Generating 4-panel diagnostic curves |
| `subprocess` | stdlib | Detached background runner | Decoupled background execution on Windows |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Workflow Diagram

```
[Training Loop (10k steps)]
        │
        ├── Every 100 steps ──> Save checkpoint_{step}.pt (100 total)
        │
[Progression Evaluator]
        │
        ├── For each of 100 ckpts:
        │       ├── Reset env (seed=42)
        │       ├── Rollout 500 steps (16.6s)
        │       ├── Compute (v_x, return, Q-mean) per step
        │       ├── Draw HUD Telemetry Banner (PIL ImageDraw)
        │       └── Save to videos/checkpoints/ckpt_{step}.mp4
        │
[Compositing Engine]
        ├── 1. Master Montage: Concatenate all 100 ckpts in sequence -> videos/halfcheetah_progression_montage.mp4
        └── 2. 2x3 Grid Video: Synchronize frames from (#1, #20, #40, #60, #80, #100) -> videos/halfcheetah_6way_grid.mp4
```

### Pattern 1: Fast HUD Banner Alpha Compositing
```python
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_telemetry_hud(frame_rgb: np.ndarray, ckpt_idx: int, total_ckpts: int, 
                       step: int, total_steps: int, t_sec: float, 
                       v_x: float, ep_return: float, q_mean: float) -> np.ndarray:
    img = Image.fromarray(frame_rgb)
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Semi-transparent top bar (height: 52px)
    draw.rectangle([(0, 0), (img.width, 52)], fill=(15, 23, 42, 200))
    
    # Line 1: Checkpoint & Global Step
    text_l1 = f"Checkpoint #{ckpt_idx}/{total_ckpts} | Step: {step:,}/{total_steps:,} ({step/total_steps*100:.1f}%)"
    # Line 2: Telemetry
    text_l2 = f"Time: {t_sec:.1f}s | Speed: {v_x:+.2f} m/s | Return: {ep_return:.1f} | Q-mean: {q_mean:.1f}"
    
    draw.text((10, 6), text_l1, fill=(255, 255, 255, 255))
    draw.text((10, 28), text_l2, fill=(56, 189, 248, 255))
    
    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    return np.array(img.convert("RGB"))
```

### Pattern 2: 2x3 Synchronous Grid Tiling
```python
def tile_2x3_grid(frames_by_checkpoint: list[list[np.ndarray]], labels: list[str]) -> list[np.ndarray]:
    # 6 lists of frames, synchronized length T
    T = min(len(f) for f in frames_by_checkpoint)
    H, W, C = frames_by_checkpoint[0][0].shape
    grid_frames = []
    
    for t in range(T):
        row1 = np.concatenate([frames_by_checkpoint[0][t], frames_by_checkpoint[1][t], frames_by_checkpoint[2][t]], axis=1)
        row2 = np.concatenate([frames_by_checkpoint[3][t], frames_by_checkpoint[4][t], frames_by_checkpoint[5][t]], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
        grid_frames.append(grid)
    return grid_frames
```

### Pattern 3: Detached Background Execution on Windows
To satisfy the user requirement of executing the long-running generation in a separate process that survives session transitions:
```powershell
Start-Process -FilePath "python" -ArgumentList "-u scripts/run_halfcheetah_progression.py --total-steps 10000 --eval-frames 500" -RedirectStandardOutput "logs/progression.log" -RedirectStandardError "logs/progression_err.log"
```
Or via Python:
```python
subprocess.Popen(
    [sys.executable, "-u", "scripts/run_halfcheetah_progression.py", ...],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    stdout=open("logs/progression.log", "w"),
    stderr=subprocess.STDOUT
)
```
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Video Encoding | Custom pipe to ffmpeg binary | `imageio.mimsave(..., fps=30)` | Handles codecs, container headers, and fallback cleanly |
| Frame Overlay | Low-level bitwise mask math | `PIL.ImageDraw` alpha overlay | Anti-aliased font rendering and transparent bounding rectangles |
| Multi-panel Tiling | Manual loop copying per pixel | `np.block` or `np.concatenate` | C-level SIMD array block copies |
| Subprocess Detach | Complex Windows service | `subprocess.DETACHED_PROCESS` or `Start-Process` | Native OS process lifecycle decoupling |
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Memory Overflow with 100 x 500 RGB Frames
**What goes wrong:** Holding 50,000 RGB frames (500x500x3) in RAM simultaneously consumes ~37.5 GB of RAM, causing MemoryError or thrashing.
**How to avoid:** Stream-encode each checkpoint video to disk immediately in `videos/checkpoints/ckpt_{step}.mp4`. For the master montage, stream frames or use an imageio video reader generator that streams one video at a time into the master file.
**Warning signs:** Rapid RAM consumption spike above 4 GB.

### Pitfall 2: HalfCheetah Early Termination
**What goes wrong:** HalfCheetah in Gymnasium continuous control has no termination (it only truncates at 1,000 steps). However, if an agent flips or falls, the step count still increments.
**How to avoid:** Ensure the rollout runs for the full requested frame horizon (`max_steps=500`) without early break unless environment explicitly signals truncated.

### Pitfall 3: Subprocess Deadlock with Unbuffered Output
**What goes wrong:** When running detached in the background, buffered stdout fills standard OS buffers and deadlocks.
**How to avoid:** Always pass `-u` flag to Python (`python -u ...`) to ensure immediate flushing to log files.
</common_pitfalls>

<open_questions>
## Open Questions
None. All design decisions were resolved and locked in 05-CONTEXT.md.
</open_questions>

<sources>
### Primary (HIGH confidence)
- Gymnasium MuJoCo HalfCheetah-v4 API specifications
- `bayesgroup/tqc_pytorch` checkpoint conventions
- Python ImageDraw and imageio.v3 official documentation
</sources>

<metadata>
**Research date:** 2026-09-09
**Valid until:** 2026-10-09
**Ready for planning:** YES
</metadata>
