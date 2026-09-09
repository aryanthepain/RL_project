# Phase 06: Compute Hierarchy for Multi-Tier Benchmark Experiments - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a unified, multi-tiered compute and experiment orchestration framework allowing TQC benchmarks to scale from local execution to heavy multi-seed benchmark training (1M–3M steps) across Kaggle, Google Colab, Local GPU, and Local CPU fallback. Delivers automated Kaggle kernel packaging, dual-slot queue management, automated pre-flight local smoke validation, a dedicated HalfCheetah-v4 pilot run (1M steps, seed 42) tracked in GitHub Issue #7, automated artifact/checkpoint synchronization back to local `runs/`, and downstream integration with plotting and progression visualization suites.

</domain>

<decisions>
## Implementation Decisions

### Compute Tier Prioritization & Remote Runner
- **D-01:** Kaggle API / Headless Kernels as Tier-1 (dual T4/P100 GPUs, 30h/wk free quota), with Colab notebook export as Tier-2, Local CUDA as Tier-3, and Local CPU fallback as Tier-4. — **Reversibility:** costly — Defines runner execution abstractions and packaging format.
- **D-02:** Project code deployed to Kaggle via Git clone + kernel metadata JSON pushed automatically via Kaggle CLI / API (`kaggle kernels push`).
- **D-03:** Automatic CUDA auto-detection across all modules with explicit `--device` override and CPU fallback.
- **D-04:** Asynchronous polling via Kaggle CLI (`kaggle kernels status` / `output`) with progress log streaming and detached background execution support.

### Artifact & Checkpoint Sync Protocol
- **D-05:** Automated CLI sync script (`scripts/sync_remote_artifacts.py`) utilizing Kaggle CLI `kaggle kernels output` to pull checkpoints, metrics, and logs directly into `runs/<exp_name>/`.
- **D-06:** Tiered retention policy: save periodic checkpoints (every 50k/100k steps) plus best-eval and final models, bundled into a compressed `.tar.gz` archive before remote sync to conserve bandwidth.
- **D-07:** Timestamped run directories with platform prefixes (e.g. `runs/kaggle_halfcheetah_s42_YYYYMMDD_HHMM/`) and SHA256 integrity validation.
- **D-08:** Automatic post-sync verification: validate `metrics.csv` integrity and provide immediate CLI triggers for plotting curves and generating progression videos.

### Experiment Matrix & Dispatch Schema
- **D-09:** Dual configuration support: Support both declarative YAML configuration (`configs/benchmark_matrix.yaml`) with named presets and pure CLI parameterized flags (`--envs`, `--seeds`, `--steps`).
- **D-10:** Scope focus: Pilot Kaggle execution specifically on `HalfCheetah-v4` first (tracked in GitHub Issue #7) before expanding to the rest of the benchmark suite (Hopper, Walker2d, Ant, Humanoid in Issue #4).
- **D-11:** Dual-slot queue manager: dispatches up to 2 concurrent GPU kernels on Kaggle, automatically queueing and launching remaining jobs as slots free up.
- **D-12:** TQC-only experimental matrix, evaluating empirical reproduction curves directly against published ICML 2020 baseline numbers.
- **D-13:** Automated pre-flight validation: runs a 100-step local smoke test verifying environment creation, GPU/CPU tensors, and checkpointing before pushing to remote cloud.

### Hardware Adaptation & HalfCheetah Pilot
- **D-14:** Adaptive replay buffer capacity: 1M transitions (paper spec) on Kaggle/Colab remote tiers; configurable 100k–250k on local CPU/smoke runs to prevent system memory pressure.
- **D-15:** Keep exact paper batch size 256 with PyTorch CUDA optimizations enabled (`torch.backends.cudnn.benchmark = True`, non-blocking device transfers).
- **D-16:** Support both standard `~/.kaggle/kaggle.json` and `.env` fallback (`KAGGLE_USERNAME`, `KAGGLE_KEY`), with automated auth validation pre-check script AND interactive prompt asking for Kaggle credentials if missing.
- **D-17:** HalfCheetah pilot executes a full 1,000,000 steps with checkpoints every 50k steps and evaluation every 5k steps, preceded by an automated 100-step dry-run / smoke test to verify remote execution before running the full 1M steps.

### The Agent's Discretion
- Packaging script structure and helper modularization in `src/tqc/remote/`.
- Kaggle kernel script metadata templating (`kernel-metadata.json`).
- Polling sleep intervals, backoff, and CLI formatting for status display.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GitHub Issues
- `github:aryanthepain/RL_project#4` — Implement Compute Hierarchy for Experiments (Kaggle MCP > Google Colab > Local GPU > Local CPU)
- `github:aryanthepain/RL_project#7` — Pilot Remote Compute Execution on Kaggle with HalfCheetah-v4

### Documentation
- `docs/progression_visualization.md` §6 — Remote Compute Integration (Phase 6 / Issue #4 Ready)
- `docs/paper_explanation.md` §5 — Empirical Evaluation & MuJoCo Continuous Control Benchmarks

### Codebase Implementations
- `src/tqc/train.py` — Core training loop, checkpointing, and device assignment
- `src/tqc/benchmark.py` — Local multi-seed benchmark runner
- `src/tqc/logger.py` — MetricsLogger CSV/JSONL output schema
- `scripts/run_halfcheetah_progression.py` — Progression visualization suite (ingests checkpoints)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `train_tqc(...)` in `src/tqc/train.py`: Primary training routine with device support, checkpoint saving, and evaluation metrics logging.
- `src/tqc/benchmark.py`: Contains existing benchmark execution logic, easily wrapped for remote job dispatching.
- `scripts/run_halfcheetah_progression.py`: Accepts `--skip-train --checkpoint-dir <path>` to visualize policies from synced checkpoints.
- `src/tqc/plotter.py`: Ingests `runs/` directories to generate comparative return curves and confidence bands.

### Established Patterns
- Checkpoints saved as `checkpoint_*.pt` via `torch.save(agent.state_dict(), ...)`.
- Metrics stored as `metrics.csv` with standard columns: `step`, `eval_return_mean`, `eval_return_std`, `critic_loss`, `actor_loss`, `alpha`, `q_mean`.
- Detached background execution pattern established in Phase 5 via `scripts/launch_progression_background.ps1`.

### Integration Points
- `scripts/remote/`: Directory for Kaggle kernel templates, packaging, and dispatch scripts.
- `scripts/sync_remote_artifacts.py`: Sync script pulling remote outputs into `runs/<exp_name>/`.
- `configs/benchmark_matrix.yaml`: Declarative experiment configuration.

</code_context>

<specifics>
## Specific Ideas

- "both 1 and 2. also let us first test the kaggle for only cheetah. we will move to others later. create an issue on gh regarding the same"
- "start wth a dry run of 100 steps to see if everything is working"
- "Support standard ~/.kaggle/kaggle.json and .env fallback, with automated auth validation pre-check script AND interactive prompt asking for Kaggle credentials if missing"

</specifics>

<deferred>
## Deferred Ideas

- Full multi-environment suite (Hopper, Walker2d, Ant, Humanoid) scaling deferred until HalfCheetah pilot passes (GitHub Issue #4).
- Google Colab automated notebook execution pipeline (Priority Tier 2).
- Presentation slides and evaluation defense materials (GitHub Issue #3).

</deferred>

---

*Phase: 06-compute-hierarchy-for-multi-tier-benchmark-experiments*
*Context gathered: 2026-09-09*
