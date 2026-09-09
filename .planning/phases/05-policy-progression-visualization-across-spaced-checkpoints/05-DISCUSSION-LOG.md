# Phase 05: Policy Progression Visualization Across Spaced Checkpoints - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-09
**Phase:** 05-policy-progression-visualization-across-spaced-checkpoints
**Areas discussed:** Rollout Horizon & Length, Montage Stitching & Video Composition, HUD Telemetry & Iteration Display, Synchronized Grid Comparison

---

## Checkpoint Rollout Horizon

| Option | Description | Selected |
|--------|-------------|----------|
| 100 frames | ~3.3 seconds at 30 fps per checkpoint | |
| 200 frames | ~6.6 seconds at 30 fps per checkpoint | |
| 50 frames | ~1.6 seconds at 30 fps per checkpoint | |
| 15–20 seconds | ~450–600 frames per checkpoint | ✓ |

**User's choice:** "let us 15-20 seconds each per checkpoint to see what is happeningh properly"
**Notes:** User specifically requested deep observation window (15–20s / ~500 frames) for each checkpoint.

---

## Montage Stitching & Video Composition

| Option | Description | Selected |
|--------|-------------|----------|
| Individual + Chronological Master | Save all 100 full 15-20s videos individually in videos/checkpoints/, and stitch them back-to-back into the complete chronological master video | ✓ |
| Stitched master only | Single complete master video only | |
| Condensed progression reel | First 2-3s of each checkpoint stitched | |

**User's choice:** Save all 100 full 15-20s videos individually in videos/checkpoints/, and stitch them back-to-back into the complete chronological master video (with optional fast-timelapse flag).
**Notes:** Retains full fidelity archive of each checkpoint while generating the master stitched chronicle.

---

## HUD Telemetry & Iteration Display

| Option | Description | Selected |
|--------|-------------|----------|
| Full Diagnostic Banner | Top bar displaying Checkpoint #{idx}/100, Step ({step}/10k), Rollout Time, Forward Speed (v_x), Cumulative Return, and Critic Q-mean | ✓ |
| Standard Telemetry Banner | Checkpoint #{idx}/100, Step ({step}/10k), Forward Velocity, Cumulative Return | |
| Minimal Corner Overlay | Compact text in upper left corner | |

**User's choice:** Full Diagnostic Banner
**Notes:** Specifically ensures the iteration / checkpoint number is clearly and properly displayed.

---

## Synchronized Grid Comparison

| Option | Description | Selected |
|--------|-------------|----------|
| 4-Way 2x2 Grid | Default checkpoints #1, #25, #50, #100 | |
| 6-Way 2x3 Grid | Checkpoints #1, #20, #40, #60, #80, #100 | ✓ |
| 3-Way 1x3 Strip | Checkpoints #1, #50, #100 | |

**User's choice:** 6-Way 2x3 Synchronized Grid (Checkpoints #1, #20, #40, #60, #80, #100) to see finer increments side-by-side.
**Notes:** Gives a 6-panel simultaneous side-by-side view across the 10,000-step progression under identical seeds.

---

## The Agent's Discretion

- Font rendering and HUD styling via PIL (`ImageDraw`).
- Border styling and grid cell labeling.
- Configurable CLI flags for custom checkpoint intervals or lists.

## Deferred Ideas

- Scaling to 1M–3M steps across full benchmark suite (Phase 6 / GitHub Issue #4).
