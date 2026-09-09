# Phase 06: Compute Hierarchy for Multi-Tier Benchmark Experiments - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-09
**Phase:** 06-compute-hierarchy-for-multi-tier-benchmark-experiments
**Areas discussed:** Compute Tier Prioritization & Remote Runner, Artifact & Checkpoint Sync Protocol, Experiment Matrix & Dispatch Schema, Hardware Adaptation & HalfCheetah Pilot

---

## Compute Tier Prioritization & Remote Runner

| Option | Description | Selected |
|--------|-------------|----------|
| Kaggle API / Headless Kernels as Tier-1 (dual T4/P100 GPUs, 30h/wk free), Colab Tier-2, Local CUDA Tier-3, Local CPU Tier-4 | Multi-tier hierarchy prioritizing free cloud GPU quotas | ✓ |
| Google Colab primary with standalone interactive/headless Jupyter notebooks (.ipynb) and Google Drive sync | Focus on Colab ecosystem | |
| Local-first with CUDA GPU priority | Local execution priority | |

**User's choice:** Kaggle API / Headless Kernels as Tier-1, with Colab Tier-2, Local CUDA Tier-3, and Local CPU Tier-4.
**Code deployment:** Git clone + Kaggle kernel metadata JSON pushed automatically via Kaggle CLI / API (`kaggle kernels push`).
**Device handling:** Automatic CUDA auto-detection with explicit `--device` override and CPU fallback.
**Job monitoring:** Asynchronous polling via Kaggle CLI (`kaggle kernels status` / `output`) with progress log streaming and detached background support.

---

## Artifact & Checkpoint Sync Protocol

| Option | Description | Selected |
|--------|-------------|----------|
| Automated CLI sync script (`scripts/sync_remote_artifacts.py`) using `kaggle kernels output` | Direct CLI pull into runs/<exp_name>/ | ✓ |
| Cloud storage intermediary | Hugging Face Hub / Google Drive | |
| Manual zip bundle download | Manual web UI download | |

**User's choice:** Automated CLI sync script (`scripts/sync_remote_artifacts.py`) utilizing Kaggle CLI `kaggle kernels output`.
**Retention policy:** Tiered retention: save periodic checkpoints (every 50k/100k steps) plus best-eval and final models, bundled into a compressed `.tar.gz` before download.
**Naming & collision:** Timestamped run directories with platform prefixes (e.g. `runs/kaggle_halfcheetah_s42_YYYYMMDD_HHMM/`) and SHA256 integrity validation.
**Local integration:** Automatic post-sync verification: validate `metrics.csv` integrity and provide one-click CLI triggers for plotting curves and generating progression videos.

---

## Experiment Matrix & Dispatch Schema

| Option | Description | Selected |
|--------|-------------|----------|
| Declarative YAML configuration (`configs/benchmark_matrix.yaml`) | Structured matrix configuration | ✓ |
| Pure CLI parameterized flags | Flexible command-line dispatching | ✓ |
| Python configuration dataclasses | Embedded Python configs | |

**User's choice:** Both declarative YAML and pure CLI parameterized flags.
**Scope boundary:** "first test the kaggle for only cheetah. we will move to others later. create an issue on gh regarding the same." (Created GitHub Issue #7).
**Concurrency & scheduling:** Dual-slot queue manager: dispatches up to 2 concurrent GPU kernels on Kaggle, automatically queueing and launching remaining jobs as slots free up.
**Baselines:** TQC-only experimental matrix, comparing directly against published ICML 2020 baseline numbers.
**Pre-flight validation:** Automated pre-flight validation: runs a 100-step local smoke test verifying environment creation, GPU/CPU tensors, and checkpointing before pushing to remote cloud.

---

## Hardware Adaptation & HalfCheetah Pilot

| Option | Description | Selected |
|--------|-------------|----------|
| Adaptive replay buffer capacity (1M cloud GPU, 100k-250k local CPU) | Memory management across tiers | ✓ |
| Fixed 1M transitions everywhere | Strict paper spec | |

**User's choice:** Adaptive capacity (1M cloud GPU, configurable 100k-250k on local CPU/smoke runs).
**Batch size & CUDA:** Keep paper exact batch size 256 with PyTorch CUDA optimizations enabled (`cudnn.benchmark = True`, non-blocking device transfers).
**Kaggle credentials:** Support standard `~/.kaggle/kaggle.json` and `.env` fallback (`KAGGLE_USERNAME`, `KAGGLE_KEY`), with automated auth validation pre-check script AND interactive prompt asking for Kaggle credentials if missing.
**Training horizon:** Full 1,000,000 steps with checkpoints every 50k steps and evaluation every 5k steps, preceded by an automated 100-step dry-run / smoke test to verify remote execution before running the full 1M steps.

---

## The Agent's Discretion

- Packaging script structure and helper modularization in `src/tqc/remote/`.
- Kaggle kernel script metadata templating (`kernel-metadata.json`).
- Polling sleep intervals, backoff, and CLI formatting for status display.

## Deferred Ideas

- Full multi-environment suite (Hopper, Walker2d, Ant, Humanoid) scaling deferred until HalfCheetah pilot passes (GitHub Issue #4).
- Google Colab automated notebook execution pipeline (Priority Tier 2).
- Presentation slides and evaluation defense materials (GitHub Issue #3).
