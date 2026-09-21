# Phase 08: Multi-Tier Compute Hierarchy and Google Colab Integration - Research

**Date:** 2026-09-21
**Phase:** 08-multi-tier-compute-hierarchy-and-google-colab-integration
**Status:** Complete

---

## 1. Executive Summary & Objective

Phase 8 fulfills GitHub Issue #4 by delivering a robust, prioritized multi-tier compute hierarchy across four execution environments:
1. **Tier 1: Kaggle MCP / Headless GPU Kernels** (Highest Priority: dual T4 / P100 GPUs, 30h/week free quota, background queue orchestration established in Phase 6).
2. **Tier 2: Google Colab GPU** (Priority 2: interactive notebook execution with automated Git checkout, headless MuJoCo EGL rendering, dual Google Drive / local artifact streaming, and Drive deletion crash resilience).
3. **Tier 3: Local CUDA GPU** (Priority 3: direct consumer/workstation NVIDIA GPU execution with `cudnn.benchmark`).
4. **Tier 4: Local CPU Fallback** (Priority 4: multi-threaded CPU fallback for rapid smoke testing and environments without GPU access).

In addition to the tier architecture, this phase delivers:
- `src/tqc/compute/tier_detector.py`: Automated environment probing and tier resolution.
- `src/tqc/compute/dispatcher.py`: Job routing and parameter adaptation per tier.
- `scripts/run_experiments.py`: Unified CLI entrypoint with `--tier auto|kaggle|colab|gpu|cpu`, `--env <id>`, `--preset smoke|quick|full`, and interactive probe confirmations.
- `notebooks/colab_tqc_benchmark.ipynb`: Self-contained, runnable Google Colab notebook with parameterized form fields, HTML5 rollout playback with Phase 7 inactivity truncation, live training metrics curves, and Google Drive storage management.
- `scripts/clean_runs.py`: Disk quota inspection and checkpoint pruning utility to keep Drive storage within quota.
- Seamless checkpoint resumption in `src/tqc/train.py` with Drive folder deletion crash resilience.

---

## 2. Codebase Investigation & Integration Points

### 2.1 Existing Remote Infrastructure (Phase 6)
- **`src/tqc/remote/kaggle_auth.py`**:
  - Functions: `resolve_kaggle_credentials()`, `validate_kaggle_credentials()`, `ensure_kaggle_cli_installed()`.
  - Probes credentials from `~/.kaggle/kaggle.json`, environment variables (`KAGGLE_USERNAME`, `KAGGLE_KEY`), or interactive prompt.
  - Can be directly utilized by Tier 1 probe without duplicating credential logic.
- **`src/tqc/remote/queue_manager.py`**:
  - `DualSlotQueueManager`: manages max 2 concurrent jobs on Kaggle.
  - Used by Tier 1 dispatcher to verify slot availability before dispatching.
- **`src/tqc/remote/kaggle_packager.py` & `scripts/run_remote_experiment.py`**:
  - Packages source code into `tqc_source.tar.gz`, builds `remote_entrypoint.py`, and pushes kernel via Kaggle API.

### 2.2 Core Training & Checkpointing (`src/tqc/train.py` and `src/tqc/agent.py`)
- `agent.save(filepath)` and `agent.load(filepath)` in `src/tqc/agent.py` already serialize:
  `actor`, `critic`, `critic_target`, `log_alpha`, `actor_optimizer`, `critic_optimizer`, `alpha_optimizer`, `target_entropy`, `drop_top`, `gamma`, `tau`.
- In `src/tqc/train.py`:
  - Currently runs fixed `for step in range(1, total_timesteps + 1)`.
  - To support checkpoint resumption:
    - Add `resume: bool = False` or `resume_checkpoint: Optional[str] = None`.
    - Check for existing checkpoints (`checkpoint_*.pt` or `latest.pt`) in the run directory.
    - Extract `step` from checkpoint metadata or filename.
    - Resume the loop from `start_step = loaded_step + 1` up to `total_timesteps`.
  - Drive Folder Deletion Resilience:
    - User specifically requested: "I want to be easily be able to delete the drive Folder itself to save space So please be robust to being Deleted the folder".
    - In `train.py` and `logger.py`, checkpoint saving must defensively check `os.makedirs(save_dir, exist_ok=True)` immediately before `torch.save(...)` and catch `(FileNotFoundError, PermissionError)` to recreate the directory or fallback gracefully to a local emergency buffer so training never terminates abruptly.

### 2.3 Visualization & Inactivity Truncation (Phase 7)
- `src/tqc/visualize.py` provides:
  - `rollout_checkpoint_with_telemetry(...)` with inactivity/stagnation truncation (`--truncate-inactive`), 50% scene dimming, and amber HUD badges.
  - Headless rendering: MuJoCo offscreen rendering on Linux requires `export MUJOCO_GL=egl` or `osmesa`. In Colab containers with GPU runtimes, EGL provides fast hardware-accelerated offscreen rendering.

---

## 3. Architecture & Module Design

### 3.1 Tier Detection Engine (`src/tqc/compute/tier_detector.py`)
- Enum `ComputeTier`: `KAGGLE`, `COLAB`, `LOCAL_GPU`, `LOCAL_CPU`.
- Dataclass `TierCapability`:
  - `tier: ComputeTier`
  - `available: bool`
  - `name: str`
  - `details: Dict[str, Any]` (e.g. GPU device name, VRAM GB, CPU cores, Kaggle username, queue slots)
  - `reason: Optional[str]` (explanation if unavailable)
- Detection Logic:
  1. **Colab Probe**: checks `'google.colab' in sys.modules` or can import `google.colab`. If running inside Colab, Colab is active.
  2. **Kaggle Probe**: checks whether Kaggle credentials resolve and validate without user prompt, and checks CLI/API connectivity.
  3. **Local GPU Probe**: checks `torch.cuda.is_available()`. Probes device count, device name (`torch.cuda.get_device_name(0)`), and VRAM total.
  4. **Local CPU Probe**: checks `os.cpu_count()`, `torch.get_num_threads()`. Always available as baseline.
- Priority Resolution Order (D-05):
  - When running locally: **Kaggle (1) > Colab export (2) > Local GPU (3) > Local CPU (4)**.
  - When running inside Colab: detects Colab environment and defaults to Colab GPU execution.
  - Formatted ASCII Table output: renders a clear summary of all 4 tiers with status badges (`[READY]`, `[SKIPPED]`, `[UNAVAILABLE]`) and reasons.

### 3.2 Compute Dispatcher (`src/tqc/compute/dispatcher.py`)
- Resolves requested tier (`auto` or explicit tier string).
- In `auto` mode: probes tiers, logs summary table, prompts user interactively if in interactive TTY and `--yes` not passed, or selects highest tier automatically.
- Logs transparent fallback telemetry: if Tier 1 Kaggle is skipped because credentials are missing, logs a clear warning banner.
- Adapts parameters:
  - Sets appropriate `buffer_capacity` (1M on Kaggle/Colab/Local GPU, 100k-250k on Local CPU).
  - Configures `device` (`cuda` vs `cpu`).
  - Sets `torch.backends.cudnn.benchmark = True` when CUDA is active.

### 3.3 Unified Runner CLI (`scripts/run_experiments.py`)
- CLI Arguments:
  - `--tier [auto|kaggle|colab|gpu|cpu]` (default: `auto`)
  - `--env [HalfCheetah-v4|Hopper-v4|Walker2d-v4|Ant-v4|Humanoid-v4]` (default: `HalfCheetah-v4`)
  - `--preset [smoke|quick|full]` (default: `smoke`)
  - `--seed [int]` (default: `42`)
  - `--resume` / `--no-resume` (default: `True`)
  - `--yes` / `--non-interactive` (bypass interactive confirmation)
  - `--dry-run` (probe tiers and print configuration without starting training)
- Steps according to preset:
  - `smoke`: 100 steps (or 1,000 steps)
  - `quick`: 10,000 steps
  - `full`: 1,000,000 steps (3,000,000 steps for Humanoid-v4)

### 3.4 Google Colab Notebook (`notebooks/colab_tqc_benchmark.ipynb`)
- Structure:
  1. **Header & Badges**: Open in Colab badge, project title, ICML 2020 paper reference.
  2. **Environment & GPU Setup**: Probes GPU (`nvidia-smi`), sets `export MUJOCO_GL=egl`, installs apt packages (`libglew-dev`, `patchelf`, `ffmpeg`).
  3. **Repository Setup**: Clones `aryanthepain/RL_project`, checks out active branch, executes `pip install -e .`, runs a 1-second MuJoCo initialization test.
  4. **Drive Mount & Resilience**:
     - Optional Google Drive mount: `drive.mount('/content/drive')`.
     - Detects Drive directory `/content/drive/MyDrive/tqc_runs/` with defensive creation and fallback.
  5. **Experiment Configuration (Colab Form Widgets)**:
     - Environment dropdown `@param ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4"]`
     - Preset dropdown `@param ["smoke", "quick", "full"]`
     - Seed field `@param {type:"integer"}`
     - Checkpoint frequency field
     - Resume checkbox `@param {type:"boolean"}`
  6. **Training Execution Cell**: Launches training loop with live progress reporting, automatic checkpointing to Drive/local, and periodic evaluation logging.
  7. **Live Evaluation Metrics Plotting**: Generates and refreshes matplotlib curves for evaluation returns, losses, alpha, and Q-mean.
  8. **In-Notebook Video Player**: Runs evaluation rollout with Phase 7 inactivity truncation and HUD telemetry, rendering and embedding HTML5 video with `IPython.display.Video`.
  9. **Drive Pruning & Quota Cleanup Cell**: Built-in cell allowing the user to inspect Drive usage and delete old intermediate checkpoints while keeping `best_model.pt` and `latest.pt`.
  10. **One-Click Artifact Archive Download**: For users not mounting Drive, compresses `/content/runs/<exp>/` into `.tar.gz` and triggers Colab browser file download (`google.colab.files.download`).

### 3.5 Run Pruning Utility (`scripts/clean_runs.py`)
- Scans `runs/` or Google Drive `tqc_runs/`.
- Identifies runs, calculates total size in MB/GB.
- Modes:
  - `--inspect`: reports size per run, total disk footprint, and number of checkpoints.
  - `--prune`: retains only `best_model.pt`, `latest.pt`, and `metrics.csv`, deleting intermediate `checkpoint_*.pt` files.
  - `--delete --run <id>`: deletes specific run directory with confirmation.
  - `--dry-run`: previews deletions without modifying files.

---

## 4. Validation Architecture

1. **Unit Tests (`tests/test_tier_detector.py`)**:
   - Test `TierDetector` when Kaggle credentials present vs absent.
   - Test `TierDetector` when CUDA available vs mock CPU-only.
   - Test priority resolution hierarchy: Kaggle > Colab > Local GPU > Local CPU.
   - Test summary table formatting and fallback reasoning output.
2. **Integration Tests (`tests/test_compute_dispatcher.py`)**:
   - Test `ComputeDispatcher` parameter adaptation across presets (`smoke`, `quick`, `full`).
   - Test checkpoint resumption logic in mock training run.
   - Test Drive folder deletion resilience: mock saving when target directory is deleted mid-run and verify directory auto-recreation without exception.
3. **Notebook Lint & Structure Verification (`tests/test_colab_notebook.py`)**:
   - Validate `notebooks/colab_tqc_benchmark.ipynb` JSON format, cell syntax, and presence of all required sections (Drive mount, EGL setup, widget parameters, training cell, video player, pruning utility).
4. **Pruning Utility Tests (`tests/test_clean_runs.py`)**:
   - Test `--inspect`, `--prune`, and `--delete` modes on temporary mock run directories.

---

## 5. Risk Mitigation & Edge Cases

| Risk | Mitigation Strategy |
|------|---------------------|
| Google Drive folder deleted by user mid-training | Defensive `os.makedirs(dir, exist_ok=True)` before every write, fallback to local buffer `/content/runs/` with warning |
| Google Drive storage quota exceeded (15GB free limit) | Smart retention: keep only periodic + best + latest; built-in pruning utility `clean_runs.py` |
| MuJoCo OpenGL context crashes in headless Colab | Explicit `export MUJOCO_GL=egl` set in environment before importing Gymnasium/MuJoCo |
| Colab session disconnection / timeout | Auto-resumption detects existing checkpoints in Drive and resumes without restarting from step 0 |
| Missing Kaggle credentials on local machine | Probing warns cleanly and falls back to Local GPU or CPU with formatted diagnostic table |
