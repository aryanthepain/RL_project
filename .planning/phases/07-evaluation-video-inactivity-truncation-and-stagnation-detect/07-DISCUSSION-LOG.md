# Phase 07: Evaluation Video Inactivity Truncation and Stagnation Detection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-21
**Phase:** 07-evaluation-video-inactivity-truncation-and-stagnation-detect
**Areas discussed:** Detection Metrics & Stagnation Triggers, Video Buffer Truncation vs Simulation Stepping, HUD Visual Indicators & Trailing Padding, Multi-Video Grid & Master Reel Synchronization, CLI Arguments, Defaults & Opt-In Behavior

---

## Detection Metrics & Stagnation Triggers

| Option | Description | Selected |
|--------|-------------|----------|
| Dual Metric | Absolute forward speed (|v_x| < threshold) combined with rolling window position delta variance | |
| Velocity Threshold Only | Rely strictly on forward velocity magnitude (|v_x| < speed_threshold) | |
| You decide | Automatically select the best signal based on environment observation space and info dict availability | ✓ |

**User's choice:** "3. but also search online from people those who have conducted the same experiments to have the best metric possible. also the simulation itself will run for the whole duration or whatever the paper says. its just the video generation that we are targeting."
**Notes:** Decided on an adaptive multi-tiered resolver: `info['x_velocity']` -> `env.unwrapped.data.qvel[0]` -> state displacement norm. Inactivity triggers on $|v_x| < \text{threshold}$ (0.05 m/s) with patience 30 steps. Stagnation triggers on rolling displacement $|x_t - x_{t-W}| < 0.10\text{ m}$ combined with velocity variance.

---

## Video Buffer Truncation vs Simulation Stepping

| Option | Description | Selected |
|--------|-------------|----------|
| Skip env.render() after truncation | Simulation steps continue to max_steps at 50x speed without wasted OpenGL render calls | |
| Continue calling env.render() | Continue rendering without storing frames to ensure no state side-effects | |
| You decide | Benchmark or pick the cleanest render-skipping implementation | ✓ |

**User's choice:** "3. create a plan specifically to decide which of the above is the best solution"
**Notes:** Add a dedicated plan/spike task during Phase 7 planning to benchmark render-skipping speedup vs render-discarding and verify zero state/physics divergence in MuJoCo. Return dict includes explicit metadata: `truncated_at_step`, `total_sim_steps`, `is_truncated`, `completion_reason`. Configurable warmup grace period `--warmup-steps` defaults to 60 steps (~2.0s).

---

## HUD Visual Indicators & Trailing Padding

| Option | Description | Selected |
|--------|-------------|----------|
| 15 frames padding | Clean trailing buffer showing robot coming to rest without excessive dead footage | ✓ |
| 30 frames padding | Generous buffer clearly establishing motionless state | |
| 5 frames padding | Minimal padding | |

**User's choice:** 15 frames padding (~0.5s at 30 fps).
**Notes:** Amber status pill `[STATUS: INACTIVE TRUNCATED]` displayed in HUD top bar next to step counter during trailing frames and on the final frame. The final frame and padding frames display the final evaluation return achieved across the full simulation (`FINAL EP RET: XXX.X`).

---

## Multi-Video Grid & Master Reel Synchronization

| Option | Description | Selected |
|--------|-------------|----------|
| Freeze-frame extension | Hold the final frame with status badge until longest panel finishes | |
| Black / dimmed frame | Blank or dim the panel once truncation ends | |
| User composite | Hold frame and dim by 50%, no blacking out | ✓ |

**User's choice:** "1 and 2. dim the frame by 50% no blacking"
**Notes:** In 6-way grid, hold the final frame and dim the 3D environment rendering by 50% while keeping the HUD top bar at 100% full brightness and legible with the amber status badge until the longest panel finishes. In progression reel, instant transition (cut immediately after 15 padding frames) to the next checkpoint.

---

## CLI Arguments, Defaults & Opt-In Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Enabled by default | Truncation is ON by default (with '--no-truncate' flag to disable) | ✓ |
| Opt-in flag | Truncation is OFF by default | |

**User's choice:** Enabled by default.
**Notes:** Expose 6 CLI flags: `--no-truncate`, `--speed-threshold` (0.05), `--patience` (30), `--padding-frames` (15), `--warmup-steps` (60), `--stagnation-delta` (0.01). Fast deterministic mock tests + real MuJoCo integration tests on HalfCheetah-v4. Deferred todo to implement per-environment integration test suites once those environments are implemented.

---

## The Agent's Discretion

- Implementation details of the OpenCV alpha-blending mask for the 50% dimming effect.
- Internal fallback chain for extracting environment velocity across non-standard MuJoCo variants.

---

## Deferred Ideas

- Implement environment-specific integration test suites for each remaining MuJoCo benchmark environment (Ant-v4, Hopper-v4, Walker2d-v4, Humanoid-v4) once those environment training pipelines are implemented.
