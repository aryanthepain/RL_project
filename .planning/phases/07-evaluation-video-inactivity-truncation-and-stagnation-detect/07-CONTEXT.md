# Phase 07: Evaluation Video Inactivity Truncation and Stagnation Detection - Context

**Gathered:** 2026-09-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement inactivity and stagnation detection mechanisms in `rollout_checkpoint_with_telemetry` and CLI tools in `src/tqc/visualize.py` to truncate redundant evaluation video frames when agents halt, fall, or oscillate in place. Maintain full simulation rollout step counts and cumulative return metric integrity, while adding trailing padding frames, clear HUD status pills, 50% dimmed frozen panels in multi-video grids, and configurable CLI flags.

</domain>

<decisions>
## Implementation Decisions

### Detection Metrics & Stagnation Triggers
- **D-01:** Adaptive multi-tiered detection signal across MuJoCo locomotion environments:
  - Primary metric: Forward velocity magnitude $|v_x| < \text{speed\_threshold}$ (default: 0.05 m/s).
  - Secondary metric: Stagnation variance — net coordinate displacement $|x_t - x_{t-W}| < 0.10\text{ m}$ combined with velocity variance $\text{std}(v_x) < \text{stagnation\_delta}$ over a rolling window of $W = 30$ steps (catches in-place spastic thrashing or wall-stuck behavior).
  - Multi-tiered velocity resolver: Check `info.get('x_velocity')` / `info.get('forward_velocity')` $\to$ `env.unwrapped.data.qvel[0]` (direct MuJoCo C physics accessor) $\to$ state displacement norm $\|\mathbf{s}_t - \mathbf{s}_{t-1}\|_2$.
- **D-02:** Thresholds & patience:
  - `speed_threshold`: 0.05 m/s.
  - `patience`: 30 consecutive inactive steps (~1.0s at 30 fps).
  - `stagnation_delta`: 0.01.

### Video Buffer Truncation vs Simulation Stepping
- **D-03:** Strict simulation integrity: The environment step loop executes for the full episode duration (`max_steps`, e.g. 500 or 1000 steps per paper/benchmark specification) so total cumulative return, environment step counts, and critic Q-mean metrics remain 100% faithful and uncorrupted. Only the video recording frame buffer is truncated upon inactivity. — **Reversibility:** costly — Fundamental evaluation contract relied on by all downstream benchmark comparison charts.
- **D-04:** Render skipping optimization: Create a dedicated plan/spike task during phase planning to benchmark whether to skip `env.render()` calls after video truncation triggers vs continuing render calls without storing frames, verifying speed gain (up to 50x CPU acceleration) and confirming zero physics/state divergence in MuJoCo.
- **D-05:** Return dictionary structure: Preserve existing keys and add explicit truncation metadata:
  - `truncated_at_step`: int (e.g. 75, step where video recording ended).
  - `total_sim_steps`: int (e.g. 500, full simulation steps executed).
  - `is_truncated`: bool (`True` if truncated early, `False` otherwise).
  - `completion_reason`: enum string (`'max_steps'`, `'env_terminated'`, `'inactivity_truncated'`, `'stagnation_truncated'`).
  - `frames`: List[np.ndarray] (length = truncated video frames + padding).
  - `velocities`, `rewards`, `actions`, `q_means`: List[float] (length = total simulation steps).
- **D-06:** Configurable warmup grace period: `--warmup-steps` with default = 60 steps (~2.0 seconds at 30 fps) to allow initial spawn drop, physics settle, and movement attempts before inactivity detection activates.

### HUD Visual Indicators & Trailing Padding
- **D-07:** Trailing padding frames: Preserve 15 trailing frames (~0.5s at 30 fps) after truncation triggers so deceleration to rest is visually natural without abrupt cuts.
- **D-08:** Visual status badge: Display a high-visibility amber status pill `[STATUS: INACTIVE TRUNCATED]` or `[TRUNCATED]` integrated directly into the HUD top banner next to the step counter during trailing padding frames and on the final frame.
- **D-09:** End-of-episode telemetry display: In trailing padding frames and on the final frame, display the final evaluation return achieved across the full simulation (`FINAL EP RET: XXX.X`) alongside the status pill so viewers see true policy evaluation performance.

### Multi-Video Grid & Master Reel Synchronization
- **D-10:** 6-way synchronized grid panel behavior: On truncated rollouts, hold the final frame and dim the 3D environment rendering by 50% (no blacking out), keeping the HUD top bar at 100% full brightness and legibility with the amber `[TRUNCATED]` pill and final simulation return, until the longest panel finishes.
- **D-11:** Master progression reel transition timing: Cut immediately (instant transition) to the next checkpoint right after the 15 trailing padding frames, creating a punchy, concise master reel without dead air.
- **D-12:** Progression reel format: Rely purely on the per-frame HUD bar (already displaying CKPT index, step number, return, and truncation status); no extra intro/outro title slides needed.

### CLI Arguments, Defaults & Opt-In Behavior
- **D-13:** Enabled by default: Video inactivity truncation is ON by default (`--truncate-inactive=True`), with a `--no-truncate` flag to disable if full 500/1000 frames are required.
- **D-14:** CLI argument suite in `src/tqc/visualize.py`:
  - `--no-truncate` (disables video truncation)
  - `--speed-threshold` (float, default: 0.05)
  - `--patience` (int, default: 30)
  - `--padding-frames` (int, default: 15)
  - `--warmup-steps` (int, default: 60)
  - `--stagnation-delta` (float, default: 0.01)
- **D-15:** Hybrid test suite: Fast deterministic mock tests (<1s) covering stationary, stagnant, and active rollouts + real MuJoCo integration test on HalfCheetah-v4 (random vs trained policy).
- **D-16:** Strict backward compatibility: All new parameters in `rollout_checkpoint_with_telemetry` are optional kwargs with agreed defaults; existing Phase 5 callers function without breaking.

### the agent's Discretion
- Exact OpenCV alpha-blending implementation for the 50% dimming mask in `create_progression_grid`.
- Adaptive fallback order for environment velocity extraction across custom vs standard MuJoCo environments.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Visualization & Algorithm
- `src/tqc/visualize.py` — Existing `rollout_checkpoint_with_telemetry`, `draw_telemetry_hud`, `create_progression_grid`, and `create_progression_reel`.
- `.planning/phases/05-policy-progression-visualization-across-spaced-checkpoints/05-CONTEXT.md` — Phase 5 visualization context and grid conventions.
- `.planning/ROADMAP.md` §Phase 7 — Scope anchor, success criteria, and dependency definitions.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `draw_telemetry_hud` in `src/tqc/visualize.py`: Draws the top HUD overlay with semi-transparent background and telemetry text; can be extended to support status pills and final simulation return display.
- `create_progression_grid` in `src/tqc/visualize.py`: Synchronizes multiple checkpoint videos into a tiled grid; can incorporate 50% dimming on panels whose video frames have ended.
- `create_progression_reel` in `src/tqc/visualize.py`: Concatenates checkpoint rollout frames into a master video file.

### Established Patterns
- Rollout loop iterates up to `max_steps` with deterministic action selection and optional critic Q-mean evaluation.
- Video generation uses `imageio` / OpenCV writer at 30 fps.

### Integration Points
- `rollout_checkpoint_with_telemetry()`: Add inactivity/stagnation detection buffer tracking, trailing padding, and truncation metadata.
- CLI argument parsing at the bottom of `src/tqc/visualize.py`: Expose `--no-truncate`, `--speed-threshold`, `--patience`, `--padding-frames`, `--warmup-steps`, and `--stagnation-delta`.

</code_context>

<specifics>
## Specific Ideas

- Visual styling for grid dimming: Dim only the 3D environment rendering by 50% while preserving the HUD bar at 100% full brightness and contrast with the amber status badge.
- When truncation occurs, the final frame and trailing padding frames display the final episode return from the end of the full simulation (`FINAL EP RET: XXX.X`).

</specifics>

<deferred>
## Deferred Ideas

### Future Environment-Specific Test Suites
- Create dedicated integration tests for each remaining MuJoCo continuous control benchmark environment (Ant-v4, Hopper-v4, Walker2d-v4, Humanoid-v4) once those environment training pipelines are implemented in subsequent benchmark phases.

</deferred>

---

*Phase: 07-evaluation-video-inactivity-truncation-and-stagnation-detect*
*Context gathered: 2026-09-21*
