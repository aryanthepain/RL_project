# HalfCheetah Policy Progression Visualization

This guide documents the HalfCheetah policy progression analysis and visualization suite implemented for Phase 5 (GitHub Issue #6).

---

## 1. Overview & Objectives

In reinforcement learning benchmark research, quantitative scalar scores (e.g. episodic return curves) often obscure how the underlying locomotion policy evolves. The progression visualization pipeline allows researchers to inspect qualitative gait evolution across training checkpoints alongside synchronized quantitative telemetry:

1. **Spaced Checkpoint Tracking**: Saves fine-grained policy snapshots (e.g. every 100 steps over 10,000 steps).
2. **On-Screen HUD Diagnostics**: Superimposes telemetry directly onto rendered frames:
   - Checkpoint / Iteration index (`Checkpoint #K / 100`)
   - Training timestep and progress percentage (`Step N / 10,000 (P%)`)
   - Trajectory time (`t = T.T s`)
   - Instantaneous forward velocity ($v_x\text{ in m/s}$)
   - Cumulative episodic return
   - Critic predicted Q-mean ($\mathbb{E}[Z(s, a)]$)
3. **Synchronized 6-Way 2x3 Comparison Grid**: Tiling 6 milestone checkpoints side-by-side running under identical evaluation seeds.
4. **Master Chronological Montage**: Sequential end-to-end concatenation of all 100 checkpoint videos showing policy evolution from random exploration to running locomotion.
5. **Decoupled Background Execution**: Full support for detached background execution so lengthy video rendering jobs run independently without blocking ongoing CLI sessions.

---

## 2. Directory & Artifact Structure

```
videos/
├── checkpoints/
│   ├── ckpt_00100.mp4       # 15–20s individual checkpoint rollout
│   ├── ckpt_00200.mp4
│   └── ... (100 videos)
├── halfcheetah_6way_grid.mp4 # Synchronized 2x3 comparison of milestones
└── halfcheetah_progression_montage.mp4 # Stitched chronological master reel

results/
└── halfcheetah_progression_diagnostics.png # 4-panel behavioral curves

logs/
├── progression_generation.log     # Background pipeline stdout stream
└── progression_generation_err.log # Background pipeline stderr stream
```

---

## 3. Quick Start & CLI Usage

### A. Detached Background Execution (Recommended for Long Runs)

To start the full 10,000-step training and 100-checkpoint video generation in the background so you can proceed to other tasks immediately:

```powershell
pwsh -File .\scripts\launch_progression_background.ps1 -TotalTimesteps 10000 -CheckpointFreq 100 -EvalFrames 500
```

**Monitoring output live:**
```powershell
Get-Content -Wait logs/progression_generation.log
```

### B. Direct CLI Execution

You can also run the pipeline synchronously in the current terminal:

```bash
# Full 10k-step run with 100 checkpoints (500 frames / ~16.6s per rollout)
python -u scripts/run_halfcheetah_progression.py \
    --total-timesteps 10000 \
    --checkpoint-freq 100 \
    --eval-frames 500 \
    --fps 30 \
    --output-dir videos

# Fast smoke-test mode (200 steps, 2 checkpoints, 10 frames each)
python -u scripts/run_halfcheetah_progression.py --smoke-test
```

### C. Evaluating Pre-Existing Checkpoints

If checkpoints have already been trained (or transferred from remote runs):

```bash
python -u scripts/run_halfcheetah_progression.py \
    --skip-train \
    --checkpoint-dir runs/progression_run \
    --eval-frames 500
```

### D. Single Checkpoint Video Rollout with HUD (GitHub Issue #5 / #6)

To render a standalone video with diagnostic HUD overlay from any single `.pt` model checkpoint:

```bash
python scripts/generate_checkpoint_video.py \
    --checkpoint runs/kaggle_halfcheetah_pilot_1m_s42/checkpoint_1000000.pt \
    --output videos/halfcheetah_1m.mp4 \
    --env HalfCheetah-v4 \
    --frames 500 \
    --fps 30
```

Or using the core module directly:

```bash
# Headless MP4/GIF export
python -m src.tqc.visualize \
    --model-path runs/demo_experiment/tqc_halfcheetah_demo/best_model.pt \
    --env-id HalfCheetah-v4 \
    --output videos/halfcheetah_rollout.mp4 \
    --max-steps 500

# Native live interactive desktop GLFW viewer
python -m src.tqc.visualize \
    --model-path runs/demo_experiment/tqc_halfcheetah_demo/best_model.pt \
    --env-id HalfCheetah-v4 \
    --interactive
```

### E. 1M Pilot Milestone Rollouts & Synchronized 2x3 Grid Comparison

To render all 6 milestone checkpoints (0, 50k, 200k, 500k, 800k, 1M) and composite them into a synchronized 2x3 comparison video:

```bash
python scripts/generate_1m_progression_videos.py \
    --run-dir runs/kaggle_halfcheetah_pilot_1m_s42 \
    --output-dir videos/halfcheetah_1m_progression \
    --grid-output videos/halfcheetah_1m_6way_grid.mp4 \
    --frames-per-video 500 \
    --fps 30
```

---

## 4. Video Compositing Details

### Full Diagnostic HUD Banner
Rendered on each frame using alpha compositing via PIL:
```
┌────────────────────────────────────────────────────────────────────────┐
│ Checkpoint #40/100 | Training Step: 4,000/10,000 (40.0%)               │
│ Time: 5.2s | Speed: +2.84 m/s | Return: 154.2 | Q-mean: 28.5          │
└────────────────────────────────────────────────────────────────────────┘
```

### 6-Way Synchronous Comparison Grid
Selects 6 milestone stages across training:
- **Slot 1**: Checkpoint #1 (Step 100) — Initial random / stumbling policy
- **Slot 2**: Checkpoint #20 (Step 2,000) — Initial torso stabilization
- **Slot 3**: Checkpoint #40 (Step 4,000) — Coordinated leg swing development
- **Slot 4**: Checkpoint #60 (Step 6,000) — Continuous forward momentum
- **Slot 5**: Checkpoint #80 (Step 8,000) — Refined gait frequency
- **Slot 6**: Checkpoint #100 (Step 10,000) — High-velocity forward running

The frames are tiled into a $2 \times 3$ grid using C-contiguous NumPy slicing with separating borders.

---

## 5. Diagnostic Plot Interpretation

The output figure `results/halfcheetah_progression_diagnostics.png` contains 4 diagnostic subplots:

1. **Forward Velocity Profiles ($v_x(t)$)**:
   Traces the cheetah's horizontal velocity across rollout steps. Untrained agents oscillate around zero; mature agents reach steady positive velocities ($> 3\text{ m/s}$).
2. **Cumulative Return Progression**:
   Visualizes the trajectory return slope. A steepening slope indicates higher reward rate.
3. **Mean Actuation Magnitude across Progression**:
   Monitors average absolute joint torques $\frac{1}{d_a} \sum |a_i|$. Highlights whether the policy uses full torque saturation or energy-efficient movements.
4. **Critic Q-mean Progression**:
   Tracks estimated return values from the TQC critic ensemble, confirming value function stabilization and absence of explosive divergence.

---

## 6. Remote Compute Integration (Phase 6 / Issue #4 Ready)

The progression pipeline is fully decoupled from the training compute source. In Phase 6 (Multi-Tier Compute Hierarchy: Kaggle MCP > Google Colab > Local GPU > Local CPU), remote training jobs on Kaggle or Colab simply export the saved `checkpoint_*.pt` directory. Running:

```bash
python scripts/run_halfcheetah_progression.py --skip-train --checkpoint-dir runs/kaggle_experiment_checkpoints/
```
will automatically generate the 100-video library, 6-way grid, and master progression reel locally.
