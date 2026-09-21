# Phase 08: Multi-Tier Compute Hierarchy and Google Colab Integration - Context

**Gathered:** 2026-09-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the complete multi-tier compute execution hierarchy (Tier 1: Kaggle API/headless GPU > Tier 2: Google Colab GPU > Tier 3: Local CUDA GPU > Tier 4: Local CPU fallback) along with a fully automated, runnable, self-contained Google Colab notebook (`notebooks/colab_tqc_benchmark.ipynb`) supporting MuJoCo continuous control benchmark training (HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4, Humanoid-v4), robust Google Drive artifact synchronization, automated checkpoint resumption, and an auto-detection dispatcher CLI/API.

</domain>

<decisions>
## Implementation Decisions

### Google Colab Notebook Architecture & Storage Resilience
- **D-01:** Hybrid Git clone with editable package install: The notebook auto-detects Google Colab environment (`google.colab`), clones `aryanthepain/RL_project` (or pulls active branch `feat/phase-8-multi-tier-compute-colab`), installs dependencies in editable mode (`pip install -e .`), and runs a fast 1-second MuJoCo sanity check. — **Reversibility:** reversible
- **D-02:** Dual-target persistence with Google Drive auto-mounting: Auto-detects if Google Drive is mounted (`/content/drive/MyDrive/tqc_runs/`). If mounted, continuously writes checkpoints (`checkpoint_*.pt`, `best_model.pt`, `latest.pt`), `metrics.csv`, and evaluation videos directly to Drive for crash resilience. If unmounted, buffers locally in `/content/runs/` with a one-click zip download button/cell. — **Reversibility:** costly — Defines notebook I/O paths and artifact synchronization contracts.
- **D-03:** Drive folder deletion resilience: File I/O operations defensively catch `FileNotFoundError` / missing directory exceptions and auto-recreate paths (`os.makedirs(drive_dir, exist_ok=True)`) or fall back to local buffer so that user manual deletion of the Drive folder to save space does not crash or interrupt active training. — **Reversibility:** reversible
- **D-04:** Dedicated Drive pruning & quota cleanup utility: Built-in notebook cell and CLI utility (`scripts/clean_runs.py` / notebook prune helper) enabling the user to inspect Drive storage usage, prune intermediate checkpoints, retain only `best_model.pt` + `latest.pt`, or delete entire old run folders with confirmation. — **Reversibility:** reversible

### Multi-Tier Auto-Detection & Fallback Protocol
- **D-05:** Compute Priority Order: Tier 1: Kaggle MCP / Headless GPU > Tier 2: Google Colab GPU > Tier 3: Local CUDA GPU > Tier 4: Local CPU fallback. — **Reversibility:** one-way — Core architectural hierarchy defining runner scheduling across the repository.
- **D-06:** Modular dispatcher architecture: Core logic encapsulated in `src/tqc/compute/tier_detector.py` and `src/tqc/compute/dispatcher.py`, exposed through a unified CLI entrypoint `scripts/run_experiments.py --tier auto|kaggle|colab|gpu|cpu`. — **Reversibility:** costly — Central orchestration interface for all subsequent benchmark and progression runs.
- **D-07:** Tier probe and interactive inspection: Probes environment state (Kaggle credentials + queue availability, Colab runtime, CUDA GPU name/VRAM, CPU thread pool), outputs a clean ASCII summary table, and presents an interactive confirmation/override prompt (bypassable via `--yes` or `--non-interactive` for scripts). — **Reversibility:** reversible
- **D-08:** Transparent fallback telemetry: Whenever auto-detection falls back from a higher tier to a lower tier, outputs an explicit warning banner detailing the reason (e.g., missing Kaggle API token, active Kaggle queue saturation, no CUDA device detected). — **Reversibility:** reversible

### Benchmark Matrix Scaling & Presets
- **D-09:** Environment-targeted execution: Rather than launching a monolithic 5-environment sweep simultaneously, runs are targeted by environment flag `--env` (or Colab dropdown: `HalfCheetah-v4`, `Hopper-v4`, `Walker2d-v4`, `Ant-v4`, `Humanoid-v4`). — **Reversibility:** reversible
- **D-10:** Tri-preset step sizing:
  - `smoke`: 100 – 1,000 steps (for sanity checks, rapid pre-flight validation)
  - `quick`: 10,000 steps (for quick evaluations / policy progression reels)
  - `full`: 1,000,000 steps for HalfCheetah, Hopper, Walker2d, Ant; 3,000,000 steps for Humanoid (ICML 2020 paper standard) — **Reversibility:** reversible
- **D-11:** Seamless checkpoint resumption: The training loop automatically detects existing `checkpoint_*.pt` in the target directory (Drive or local run folder), restores model weights, optimizer states, and step count, and resumes training from step $K$ without resetting to 0. — **Reversibility:** costly — Alters `src/tqc/train.py` checkpoint loading interface.

### Colab Headless Rendering, Live Telemetry & Phase 7 Integration
- **D-12:** Headless EGL rendering in Colab: Notebook setup cell configures `os.environ["MUJOCO_GL"] = "egl"` and installs OSMesa/EGL bindings to prevent GLFW display initialization crashes in cloud containers. — **Reversibility:** reversible
- **D-13:** In-notebook HTML5 evaluation video playback: Embedded video player (`IPython.display.Video`) rendering evaluation rollouts with HUD telemetry, utilizing Phase 7 inactivity truncation (`--truncate-inactive`) and selective scene dimming to ensure fast rendering and small video sizes. — **Reversibility:** reversible
- **D-14:** In-notebook live evaluation plotting: Interactive or refreshed matplotlib visualization cell displaying `eval_return_mean`, actor/critic losses, alpha temperature, and Q-value statistics during training. — **Reversibility:** reversible

### The Agent's Discretion
- Notebook layout, cell markdown styling, badge banners, and widget parameter defaults.
- Internal exception handling wrappers for Drive I/O sync retries.
- Specific unit and integration test fixtures for tier auto-detection mocking.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GitHub Issues
- `github:aryanthepain/RL_project#4` — Implement Compute Hierarchy for Experiments (Kaggle MCP > Google Colab > Local GPU > Local CPU)
- `github:aryanthepain/RL_project#7` — Pilot Remote Compute Execution on Kaggle with HalfCheetah-v4

### Documentation
- `docs/remote_execution.md` — Remote Compute Execution & Experiment Orchestration Guide (Phase 6)
- `docs/progression_visualization.md` — Visualization Engine, Telemetry HUD & Inactivity Truncation (Phase 5/7)
- `docs/paper_explanation.md` §5 — Empirical Evaluation & MuJoCo Continuous Control Benchmarks

### Codebase Implementations
- `src/tqc/remote/kaggle_auth.py` — Kaggle credential resolution and interactive prompting
- `src/tqc/remote/kaggle_packager.py` — Kernel packaging and remote entrypoint generator
- `src/tqc/remote/queue_manager.py` — Dual-slot Kaggle GPU queue manager
- `src/tqc/train.py` — Core training loop, Polyak updates, and checkpointing
- `src/tqc/benchmark.py` — Multi-seed benchmark runner and environment factory
- `src/tqc/visualize.py` — Inactivity truncation, OpenGL offscreen rendering, and HUD overlays
- `configs/benchmark_matrix.yaml` — Declarative benchmark presets

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/tqc/remote/kaggle_auth.py`: Existing Kaggle credential resolver with dual-source fallback and interactive prompts; directly reused in Tier 1 probe.
- `src/tqc/remote/queue_manager.py`: `DualSlotQueueManager` to inspect active Kaggle GPU slots.
- `train_tqc(...)` in `src/tqc/train.py`: Primary training loop with device selection and checkpoint saving; ready for checkpoint resumption parameter.
- `rollout_checkpoint_with_telemetry(...)` in `src/tqc/visualize.py`: Evaluates checkpoints with Phase 7 inactivity truncation and telemetry HUD.
- `configs/benchmark_matrix.yaml`: Preset configuration matrix, can be extended with `smoke`, `quick`, and `full` presets per environment.

### Established Patterns
- Checkpoints saved as `checkpoint_<step>.pt`, `best_model.pt`, and `final_model.pt`.
- Metrics stored as `metrics.csv` containing standard evaluation columns (`step`, `eval_return_mean`, `eval_return_std`, `critic_loss`, `actor_loss`, `alpha`, `q_mean`).
- Dual credential lookup pattern: checks configuration file first, falls back to environment variables, then interactive prompt.

### Integration Points
- `notebooks/colab_tqc_benchmark.ipynb`: New interactive notebook file serving as Tier 2 runtime.
- `src/tqc/compute/`: Directory housing tier detection and dispatching logic (`tier_detector.py`, `dispatcher.py`).
- `scripts/run_experiments.py`: Unified multi-tier experiment runner CLI.
- `scripts/clean_runs.py`: Disk quota inspection and checkpoint pruning utility.

</code_context>

<specifics>
## Specific Ideas

- "I want to be easily be able to delete the drive Folder itself to save space So please be robust to being Deleted the folder And other stuff like that"
- "Kaggle will be higher than Colab" -> Strict hierarchy: Kaggle (Tier 1) > Colab (Tier 2) > Local GPU (Tier 3) > Local CPU (Tier 4).
- "I will not run The full thing All environments all at once So nothing as such for a full benchmark is required Just divide them into smoke, quick, and full And another flag For the environment that it will be for"
- "Automatic checkpoint resumption: check Google Drive / local run directory for existing checkpoint_*.pt, restore replay buffer and optimizer states, and resume training seamlessly without restarting from step 0"

</specifics>

<deferred>
## Deferred Ideas

- Full simultaneous 5-environment parallel cluster sweep (deferred in favor of single-environment targeting per user direction).
- Presentation slides and course evaluation defense materials (tracked in Issue #3).

</deferred>

---

*Phase: 08-multi-tier-compute-hierarchy-and-google-colab-integration*
*Context gathered: 2026-09-21*
