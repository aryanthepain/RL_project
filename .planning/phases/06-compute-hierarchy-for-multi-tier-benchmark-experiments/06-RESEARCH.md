# Phase 06: Compute Hierarchy for Multi-Tier Benchmark Experiments - Research

**Researched:** 2026-09-09  
**Domain:** Distributed RL Cloud Orchestration, Kaggle Headless Kernels, Dual-Slot Queue Management, Automated Artifact Sync & Verification  
**Confidence:** HIGH  

<user_constraints>
## User Constraints (from CONTEXT.md & User Instructions)

### Locked Decisions
- **D-01:** Compute Tier Prioritization: Tier-1 Kaggle Headless Kernels (dual T4/P100 GPUs, 30h/wk free quota) > Tier-2 Google Colab notebook export > Tier-3 Local CUDA GPU > Tier-4 Local CPU fallback.
- **D-02:** Project code deployed to Kaggle via Git clone or zipped codebase payload + kernel metadata JSON pushed automatically via Kaggle CLI / API (`kaggle kernels push`).
- **D-03:** Automatic CUDA auto-detection across all modules with explicit `--device` override and CPU fallback.
- **D-04:** Asynchronous polling via Kaggle CLI (`kaggle kernels status` / `output`) with progress log streaming and detached background execution support.
- **D-05:** Automated CLI sync script (`scripts/sync_remote_artifacts.py`) utilizing Kaggle CLI `kaggle kernels output` to pull checkpoints, metrics, and logs directly into `runs/<exp_name>/`.
- **D-06:** Tiered retention policy: save periodic checkpoints (every 50k/100k steps) plus best-eval and final models, bundled into a compressed `.tar.gz` archive before remote sync to conserve bandwidth.
- **D-07:** Timestamped run directories with platform prefixes (e.g., `runs/kaggle_halfcheetah_s42_YYYYMMDD_HHMM/`) and SHA256 integrity validation.
- **D-08:** Automatic post-sync verification: validate `metrics.csv` integrity and provide immediate CLI triggers for plotting curves and generating progression videos.
- **D-09:** Dual configuration support: Support both declarative YAML configuration (`configs/benchmark_matrix.yaml`) with named presets and pure CLI parameterized flags (`--envs`, `--seeds`, `--steps`).
- **D-10:** Scope focus: Pilot Kaggle execution specifically on `HalfCheetah-v4` first (tracked in GitHub Issue #7) before expanding to the rest of the benchmark suite (Hopper, Walker2d, Ant, Humanoid in Issue #4).
- **D-11:** Dual-slot queue manager: dispatches up to 2 concurrent GPU kernels on Kaggle, automatically queueing and launching remaining jobs as slots free up.
- **D-12:** TQC-only experimental matrix, evaluating empirical reproduction curves directly against published ICML 2020 baseline numbers.
- **D-13:** Automated pre-flight validation: runs a 100-step local smoke test verifying environment creation, GPU/CPU tensors, and checkpointing before pushing to remote cloud.
- **D-14:** Adaptive replay buffer capacity: 1M transitions (paper spec) on Kaggle/Colab remote tiers; configurable 100k–250k on local CPU/smoke runs to prevent system memory pressure.
- **D-15:** Keep exact paper batch size 256 with PyTorch CUDA optimizations enabled (`torch.backends.cudnn.benchmark = True`, non-blocking device transfers).
- **D-16:** Support both standard `~/.kaggle/kaggle.json` and `.env` fallback (`KAGGLE_USERNAME`, `KAGGLE_KEY`), with automated auth validation pre-check script AND interactive prompt asking for Kaggle credentials if missing.
- **D-17:** HalfCheetah pilot executes a full 1,000,000 steps with checkpoints every 50k steps and evaluation every 5k steps, preceded by an automated 100-step dry-run / smoke test to verify remote execution before running the full 1M steps.

### The Agent's Discretion
- Packaging script structure and helper modularization in `src/tqc/remote/`.
- Kaggle kernel script metadata templating (`kernel-metadata.json`).
- Polling sleep intervals, backoff, and CLI formatting for status display.
- Exact schema for `configs/benchmark_matrix.yaml`.

### Deferred Ideas (OUT OF SCOPE)
- Full multi-environment suite (Hopper, Walker2d, Ant, Humanoid) scaling deferred until HalfCheetah pilot passes (GitHub Issue #4).
- Google Colab automated notebook execution pipeline (Priority Tier 2).
- Presentation slides and evaluation defense materials (GitHub Issue #3).
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

Multi-tier compute orchestration framework interfacing between local workspace and remote execution platforms.

| Capability | Primary Module | Secondary Module | Rationale |
|------------|---------------|------------------|-----------|
| Kaggle Credential & Auth | `src/tqc/remote/kaggle_auth.py` | `dotenv` / `kaggle.api` | Validates `~/.kaggle/kaggle.json` or `.env` credentials with interactive fallback |
| Kernel Packaging & Metadata | `src/tqc/remote/kaggle_packager.py` | `src/tqc/remote/templates/` | Generates `kernel-metadata.json`, wraps remote training entrypoint, and bundles source |
| Dual-Slot Queue Manager | `src/tqc/remote/queue_manager.py` | `subprocess` / `kaggle` | Enforces max 2 concurrent GPU jobs, polls status, and dispatches pending jobs |
| Experiment Configuration | `configs/benchmark_matrix.yaml` | `src/tqc/remote/config.py` | Defines declarative matrix presets (pilot, smoke, full suite) |
| Pre-flight Smoke Runner | `scripts/run_remote_experiment.py` | `src/tqc/train.py` | Executes local 100-step dry run before launching remote workloads |
| Artifact Sync & Integrity | `scripts/sync_remote_artifacts.py` | `tarfile` / `hashlib` | Pulls remote outputs, extracts `.tar.gz`, verifies SHA256 and `metrics.csv` |
| Downstream Hook Triggers | `scripts/sync_remote_artifacts.py` | `src/tqc/plotter.py` / `scripts/run_halfcheetah_progression.py` | Automatically triggers return curve plots and progression video generation |
</architectural_responsibility_map>

<research_summary>
## Summary

This phase establishes a production-grade remote compute hierarchy and orchestration pipeline for large-scale TQC benchmark experiments. Deep reinforcement learning continuous control benchmarks (such as Kuznetsov et al. 2020) require 1M–3M environment interaction steps per seed across multiple seeds, which is prohibitive on local CPU or single consumer GPU setups. 

The primary compute tier is Kaggle Headless Kernels, leveraging Kaggle's free 30h/week GPU quota (NVIDIA T4 / P100) and headless CLI API. Kaggle enforces a strict limit of 2 concurrent GPU sessions. To maximize throughput without rate-limit errors or queue starvation, this phase introduces a dual-slot queue manager that monitors running kernels and automatically dispatches queued runs.

To ensure rapid debugging and prevent wasted remote GPU quota on broken scripts, an automated 100-step pre-flight smoke test runs locally before any remote dispatch. Once training completes on Kaggle, `scripts/sync_remote_artifacts.py` retrieves the compressed checkpoint archive (`.tar.gz`) and `metrics.csv`, verifies SHA256 checksum integrity, unpacks into `runs/kaggle_<env>_<seed>_<timestamp>/`, and seamlessly links with existing plotting (`src/tqc/plotter.py`) and progression visualization suites (`scripts/run_halfcheetah_progression.py`).

The pilot validation run targets **HalfCheetah-v4 (1M steps, seed 42)** as tracked in GitHub Issue #7, setting the baseline for the entire benchmark reproduction.
</research_summary>

<standard_stack>
## Standard Stack & Platform Interfaces

### Kaggle API & CLI Specifications
- **Kaggle CLI**: Python package `kaggle` provides `kaggle kernels push`, `kaggle kernels status`, `kaggle kernels output`.
- **Credential Storage**:
  1. Default file: `~/.kaggle/kaggle.json` containing `{"username": "...", "key": "..."}`.
  2. Environment variable fallback: `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- **Kernel Metadata JSON (`kernel-metadata.json`)**:
  ```json
  {
    "id": "{username}/{kernel_slug}",
    "title": "{kernel_title}",
    "code_file": "remote_kernel_entrypoint.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
  }
  ```
- **Execution States**:
  - `queued` -> In Kaggle job queue.
  - `running` -> GPU allocated, script executing.
  - `complete` -> Successfully terminated, outputs available via `kaggle kernels output`.
  - `error` -> Failed or timed out (Kaggle max execution window: 9 hours for GPU kernels).

### Remote Execution Environment Details
- Remote operating system: Ubuntu Linux inside Kaggle Docker container (`kaggle/python`).
- Dependencies to auto-install on entrypoint start:
  - `pip install gymnasium[mujoco] mujoco`
  - Ensure torch detects GPU (`torch.cuda.is_available()`).
- Replay Buffer Memory Guard:
  - Kaggle kernels have ~16GB-30GB RAM. A 1M step replay buffer of float32 transitions requires ~400MB RAM, which easily fits within Kaggle memory limits.

### Artifact Compression Protocol
- Remote script writes checkpoints every 50k steps: `checkpoint_50000.pt`, `checkpoint_100000.pt`, etc., plus `best_model.pt` and `metrics.csv`.
- At script completion, creates `artifacts.tar.gz` containing `checkpoints/`, `metrics.csv`, and `run_summary.json` with SHA256 checksums.
- Local `scripts/sync_remote_artifacts.py` downloads output files into a staging directory, computes SHA256 checksums to verify integrity, extracts to `runs/<exp_name>/`, and validates `metrics.csv` headers and row count.
</standard_stack>

<validation_architecture>
## Validation Architecture

### Automated Verification Gates
1. **Credentials & Auth Validation Test** (`tests/test_remote_auth.py`):
   - Tests credential resolution order (`~/.kaggle/kaggle.json` vs `.env` vs missing).
   - Mock tests interactive credential collection and file generation.
2. **Packager & Metadata Test** (`tests/test_remote_packager.py`):
   - Validates generated `kernel-metadata.json` conforms to Kaggle schema requirements (`enable_gpu=True`, `enable_internet=True`).
   - Verifies entrypoint script template generation with parameter injection.
3. **Queue Manager Test** (`tests/test_remote_queue.py`):
   - Tests dual-slot concurrency enforcement (max 2 active jobs).
   - Tests state transitions (`queued` -> `running` -> `complete`) and queue dequeuing.
4. **Artifact Sync & Integrity Test** (`tests/test_remote_sync.py`):
   - Tests downloading and extraction of mock `artifacts.tar.gz`.
   - Validates SHA256 checksum verification and failure on corrupted archives.
   - Validates `metrics.csv` structure verification.
5. **Pre-flight Smoke Execution Test**:
   - Automated local execution of 100 steps on HalfCheetah-v4 verifying that checkpointing, device allocation, and logging operate error-free before remote dispatch.
</validation_architecture>
