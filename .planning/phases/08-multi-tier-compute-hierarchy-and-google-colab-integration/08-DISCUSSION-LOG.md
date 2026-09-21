# Phase 08: Multi-Tier Compute Hierarchy and Google Colab Integration - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-21
**Phase:** 08-multi-tier-compute-hierarchy-and-google-colab-integration
**Areas discussed:** Google Colab Notebook Architecture & Artifact Syncing, Multi-Tier Auto-Detection & Fallback Protocol, Benchmark Matrix Scaling & Presets across Environments, Colab Headless Rendering & Drive Storage Resilience

---

## Google Colab Notebook Architecture & Artifact Syncing

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid Git Clone + Pip Editable Install | Notebook auto-detects Colab, clones active repo branch, and installs dependencies via editable install with sanity test | ✓ |
| Monolithic / Standalone Inlined Notebook | Inline all TQC modules directly inside the notebook cells | |

| Option | Description | Selected |
|--------|-------------|----------|
| Dual-Target Persistence (Google Drive + Local Buffer) | Mount Google Drive for periodic streaming of checkpoints/metrics; fall back to local buffer with zip download if Drive unmounted | ✓ |
| Drive-Only Forced Auth | Require Google Drive login before training starts | |
| Local Ephemeral Only | Buffer all artifacts in Colab container only and trigger zip download at the end | |

**User's choice:** Hybrid Git Clone and Dual-Target Persistence.
**Notes:** User emphasized: "I want to be easily be able to delete the drive Folder itself to save space So please be robust to being Deleted the folder And other stuff like that". The saving logic must be defensively programmed against missing/deleted directories, auto-recreating directories or falling back smoothly.

---

## Multi-Tier Auto-Detection & Fallback Protocol

| Option | Description | Selected |
|--------|-------------|----------|
| Unified CLI Dispatcher (`scripts/run_experiments.py --tier auto`) | Centralized modular dispatcher with `tier_detector.py` and `dispatcher.py` logging detection decisions and fallback reasons | ✓ |
| Separate Dedicated Scripts per Tier | Distinct scripts (`run_colab.py`, `run_local.py`, etc.) with standalone check tool | |

| Option | Description | Selected |
|--------|-------------|----------|
| Strict Silent Auto-Detection | Automatic fallback strictly following hierarchy without user prompt | |
| Interactive Probe & Summary Table | Probe hardware/creds across all 4 tiers, print formatted summary table, and prompt user to confirm or override tier | ✓ |
| Config-First Preset Resolution | Prefer tier strictly set in YAML config | |

**User's choice:** Unified CLI dispatcher + Interactive Probe with strict priority hierarchy: Tier 1: Kaggle > Tier 2: Colab > Tier 3: Local GPU > Tier 4: Local CPU.
**Notes:** User specifically highlighted: "2. but also kaggle will be higher than colab". Probing displays all tiers with specs, and fallback events produce clear explanatory warning banners.

---

## Benchmark Matrix Scaling & Presets across Environments

| Option | Description | Selected |
|--------|-------------|----------|
| Full 5-Environment Monolithic Cluster Sweep | Run all 5 environments (HalfCheetah, Hopper, Walker2d, Ant, Humanoid) across 5 seeds in one massive execution | |
| Targeted Single-Environment Runs with Tri-Presets | Explicit `--env` flag (or Colab dropdown parameter) with `smoke`, `quick`, and `full` presets | ✓ |

| Option | Description | Selected |
|--------|-------------|----------|
| Automatic Checkpoint Resumption | Auto-detect latest `checkpoint_*.pt` in Drive / run dir and resume without resetting to step 0 | ✓ |
| Explicit Manual Resume Only | Require explicit `--resume-from` CLI argument | |

**User's choice:** Targeted single-environment runs with `smoke`, `quick`, and `full` presets + Automatic checkpoint resumption.
**Notes:** User stated: "I will not run The full thing All environments all at once So nothing as such for a full benchmark is required Just divide them into smoke, quick, and full And another flag For the environment that it will be for".

---

## Colab Headless Rendering, Live Telemetry & Drive Pruning

| Option | Description | Selected |
|--------|-------------|----------|
| In-Notebook HTML5 Player + EGL Headless Rendering + Live Curves | Configure `MUJOCO_GL=egl`, render rollouts using Phase 7 inactivity truncation and dimming, and display live matplotlib curves | ✓ |
| Post-Training CLI Rendering Only | Skip video rendering during training entirely | |

| Option | Description | Selected |
|--------|-------------|----------|
| Smart Retention with Pruning Utility | Save `best_model.pt`, `latest.pt`, and periodic checkpoints, with built-in notebook pruning cell and deletion resilience | ✓ |
| Minimal Retention | Save only `best_model.pt` and `latest.pt` | |

**User's choice:** In-notebook HTML5 player with headless EGL & Phase 7 inactivity truncation + Smart retention with pruning utility.

---

## Agent's Discretion

- Visual styling of Colab notebook forms, widgets, and header badges.
- Unit and integration tests for mocking Colab/Kaggle environments and validating notebook syntax.

## Deferred Ideas

- Full simultaneous 5-environment parallel cluster sweep (deferred in favor of targeted single-environment runs).
- Presentation slides and course evaluation defense materials (tracked in Issue #3).
