# Roadmap: TQC Paper Reproduction & Analysis (ICML 2020)

## Overview

A systematic 4-phase horizontal roadmap taking the project from foundational theoretical documentation to core algorithm implementation, environment infrastructure, and final multi-seed benchmark experimental reproduction on MuJoCo continuous control tasks.

## Phases

- [x] **Phase 1: Paper Theoretical & Algorithmic Documentation** - Comprehensive explanation document detailing problem motivation, mathematical formulation, truncation operator, and architectures (Issue #1).
- [x] **Phase 2: Core Algorithm & Quantile Networks** - Modular PyTorch implementation of actor, critic ensemble ($M=5, N=25$), Huber quantile loss, truncation operator, and automatic entropy tuning.
- [x] **Phase 3: Environment Harness, Replay Buffer & Training Pipeline** - 1M replay buffer, Gymnasium MuJoCo environment integration, reproducible seeding, and training loop with evaluation logging (Issue #2 foundation).
- [x] **Phase 4: Multi-Seed Benchmark Experiments, Bias Analysis & Replication Curves** - Benchmarking runs on MuJoCo suite, overestimation bias analysis, and reproduction plotting matching ICML 2020 paper figures (Issue #2 completion).
- [x] **Phase 5: Policy Progression Visualization Across Spaced Checkpoints** - Focused visualization pipeline for HalfCheetah actor policy across 100 checkpoints with 6-way grid, master montage, and HUD telemetry (Issues #5, #6).
- [x] **Phase 6: Compute Hierarchy for Multi-Tier Benchmark Experiments** - Priority compute framework with Kaggle GPU remote execution, pre-flight checks, dual-slot queue, and artifact synchronization (Issue #7).
- [x] **Phase 7: Evaluation Video Inactivity Truncation and Stagnation Detection** - Early video recording truncation on stationary velocity or stagnation delta thresholds with trailing frame padding (Issue #8). (completed 2026-09-21)
- [ ] **Phase 8: Multi-Tier Compute Hierarchy and Google Colab Integration** - Complete multi-tier compute hierarchy (Kaggle > Colab > Local GPU > Local CPU) with runnable Colab notebook and tier auto-detection fallback (Issue #4).

## Phase Details

### Phase 1: Paper Theoretical & Algorithmic Documentation

**Goal**: Produce an authoritative, comprehensive mathematical and conceptual explanation document (`docs/paper_explanation.md`) analyzing the ICML 2020 paper "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics".
**Depends on**: Nothing (first phase)
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):

  1. `docs/paper_explanation.md` clearly explains deep Q-learning overestimation bias, why SAC/TD3/QR-DQN can under- or overestimate, and how distributional quantile critics mitigate this.
  2. Mathematical formulations of the distributional Bellman equation, Huber quantile loss, and target quantile mixture distribution are thoroughly detailed with complete notation.
  3. Truncation operator mechanics (sorting $M \times N$ quantiles, discarding top $d$ atoms, effect of $d$) are explained with diagrams and pseudocode.
  4. Exact hyperparameters and network architecture catalog matching `bayesgroup/tqc_pytorch` are documented.

**Plans**: 1 plan

Plans:

- [x] 01-01: Author comprehensive theoretical paper explanation markdown document

### Phase 2: Core Algorithm & Quantile Networks

**Goal**: Implement the modular PyTorch TQC agent including squashed Gaussian actor, ensemble of $M=5$ quantile critics ($N=25$ each), Huber quantile regression loss, top-$d$ truncation operator, and automatic entropy temperature tuning.
**Depends on**: Phase 1
**Requirements**: ALGO-01, ALGO-02, ALGO-03, ALGO-04, ALGO-05
**Success Criteria** (what must be TRUE):

  1. Actor network forward pass produces valid action samples and log-probabilities with reparameterization and tanh squashing.
  2. Critic ensemble predicts $M \times N = 125$ quantiles given state-action pairs.
  3. Truncation operator correctly pools, sorts, and discards top $d$ quantiles to construct target values.
  4. Huber quantile loss and soft Polyak target updates ($\tau=0.005$) execute deterministically with unit tests passing.

**Plans**: 2 plans

Plans:

- [x] 02-01: Implement actor, quantile critic ensemble, and truncation loss modules
- [x] 02-02: Implement TQC agent update step, Polyak target updates, and auto-alpha tuning with unit tests

### Phase 3: Environment Harness, Replay Buffer & Training Pipeline

**Goal**: Build the 1M uniform replay buffer, Gymnasium MuJoCo benchmark environment wrappers (HalfCheetah, Hopper, Walker2d, Ant, Humanoid), seed control, and training/evaluation pipeline with metrics logging and SAC baseline support.
**Depends on**: Phase 2
**Requirements**: ENV-01, ENV-02, ENV-03, EXP-01, EXP-02, EXP-03
**Success Criteria** (what must be TRUE):

  1. Replay buffer handles $10^6$ transitions and samples mini-batches of 256 transitions efficiently.
  2. Environment harness seamlessly instantiates Gymnasium MuJoCo environments with standardized observation/action bounds and reproducible seeding.
  3. Training loop runs full episodes with periodic deterministic evaluation and logs rewards, losses, alpha, and Q-statistics.

**Plans**: 2 plans

Plans:

- [x] 03-01: Implement replay buffer and Gymnasium MuJoCo environment harness with seed control
- [x] 03-02: Implement end-to-end training loop, evaluation runner, checkpointing, and metrics logger

### Phase 4: Multi-Seed Benchmark Experiments, Bias Analysis & Replication Curves

**Goal**: Execute experimental training runs across MuJoCo benchmarks, run empirical overestimation bias diagnostics against Monte Carlo rollouts, and generate publication-quality reproduction plots and comparative tables.
**Depends on**: Phase 3
**Requirements**: VIZ-01, VIZ-02, VIZ-03
**Success Criteria** (what must be TRUE):

  1. Training and evaluation curves with confidence intervals generated for benchmark environments.
  2. Overestimation bias evaluation script computes and plots estimated Q-values vs true discounted returns.
  3. Results comparison table generated against Kuznetsov et al. (ICML 2020) reported numbers.

**Plans**: 2 plans

Plans:

- [x] 04-01: Execute benchmark training runs and overestimation bias Monte Carlo evaluations
- [x] 04-02: Generate replication return curves, bias comparison plots, and comparative results summary

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|---|---|---|---|
| 1. Paper Theoretical & Algorithmic Documentation | 1/1 | Complete | 2026-09-09 |
| 2. Core Algorithm & Quantile Networks | 2/2 | Complete | 2026-09-09 |
| 3. Environment Harness, Replay Buffer & Training Pipeline | 2/2 | Complete | 2026-09-09 |
| 4. Multi-Seed Benchmark Experiments, Bias Analysis & Replication Curves | 2/2 | Complete | 2026-09-09 |
| 5. Policy Progression Visualization Across Spaced Checkpoints | 2/2 | Complete    | 2026-09-09 |
| 6. Compute Hierarchy for Multi-Tier Benchmark Experiments | 2/2 | Complete | 2026-09-10 |
| 7. Evaluation Video Inactivity Truncation and Stagnation Detection | 2/2 | Complete   | 2026-09-21 |
| 8. Multi-Tier Compute Hierarchy and Google Colab Integration | 0/0 | Not planned | - |

### Phase 5: Policy Progression Visualization Across Spaced Checkpoints

**Goal**: Build a focused visualization pipeline for the HalfCheetah actor policy across 100 spaced checkpoints (10,000 steps), recording 15–20s rollouts, synchronized 6-way comparison grid, master chronological progression montage, and HUD telemetry banners with decoupled background execution.
**Depends on**: Phase 4
**Requirements**: VIZ-04
**Success Criteria** (what must be TRUE):

  1. Training loop saves step 0 baseline and periodic checkpoints every 100 steps over 10,000 steps.
  2. Frame HUD banner displays checkpoint number, step count, elapsed time, forward speed, return, and Q-mean.
  3. 6-way 2x3 synchronized grid video generated comparing milestones (#1, #20, #40, #60, #80, #100).
  4. Chronological master montage stitches all 100 videos back-to-back.
  5. Background process launcher allows long video generation to execute independently in a separate terminal without blocking future phases.

**Plans**: 2 plans

Plans:

- [x] 05-01: Core Visualization Engine, Frame Telemetry HUD & Compositing Primitives
- [x] 05-02: End-to-End Progression Runner with Detached Background Terminal Support

### Phase 6: Compute Hierarchy for Multi-Tier Benchmark Experiments

**Goal**: Establish a prioritized compute and experiment orchestration framework (Tier-1 Kaggle GPU > Tier-2 Colab > Tier-3 Local GPU > Tier-4 Local CPU) and pilot remote cloud training on HalfCheetah-v4 (1M steps) with automated artifact synchronization back to local runs.
**Depends on**: Phase 5
**Requirements**: Issue #4, Issue #7
**Success Criteria** (what must be TRUE):

  1. Dual credential detection supports `~/.kaggle/kaggle.json` and `.env` with automated validation and interactive prompts.
  2. Automated Kaggle runner packages kernel metadata, performs a 100-step pre-flight smoke test, and dispatches remote GPU training.
  3. Dual-slot queue manager tracks and schedules concurrent remote runs respecting platform limits.
  4. Remote artifact sync script pulls compressed checkpoints (`.tar.gz`) and `metrics.csv` to local `runs/` with SHA256 integrity validation.
  5. Initial HalfCheetah-v4 pilot run executes on Kaggle GPU, demonstrating remote training and local downstream visualization integration.

**Plans**: 2 plans

Plans:

- [x] 06-01: Remote Compute Foundation, Kaggle Authentication & Kernel Packager
- [x] 06-02: Dual-Slot Queue Manager, Artifact Sync, Experiment Runner & Documentation

### Phase 7: Evaluation Video Inactivity Truncation and Stagnation Detection

**Goal**: Implement inactivity and stagnation detection mechanisms in `rollout_checkpoint_with_telemetry` and CLI tools in `src/tqc/visualize.py` to truncate redundant evaluation video frames when agents halt or oscillate in place, preserving trailing padding and evaluation return metrics integrity.
**Depends on**: Phase 5, Phase 6
**Requirements**: Issue #8
**Success Criteria** (what must be TRUE):

  1. `rollout_checkpoint_with_telemetry()` detects when forward speed $|v_x| < \text{speed\_threshold}$ for consecutive steps or when speed delta variance within rolling window is below threshold.
  2. Video frame recording truncates cleanly while maintaining configurable trailing padding frames to avoid abrupt visual cuts.
  3. Environment step count and return metrics remain uncorrupted (only video frame recording buffer is truncated).
  4. CLI arguments (`--truncate-inactive`, thresholds, patience, padding) are exposed in `src/tqc/visualize.py` with sensible defaults.
  5. Unit tests verify truncation logic on stationary/stagnant and active agent mock rollouts.

**Plans**: 2/2 plans complete

Plans:

- [x] 07-01-PLAN.md
- [x] 07-02-PLAN.md

- [x] 07-01: Inactivity Truncation and Render Skipping Engine
- [x] 07-02: Presentation Layer, Selective Grid Dimming & CLI

### Phase 8: Multi-Tier Compute Hierarchy and Google Colab Integration

**Goal**: Establish the complete prioritized execution pipeline (Tier 1: Kaggle MCP > Tier 2: Google Colab > Tier 3: Local GPU > Tier 4: Local CPU) with an automated runnable Google Colab notebook for MuJoCo benchmark reproduction and an auto-detection fallback helper.
**Depends on**: Phase 6, Phase 7
**Requirements**: Issue #4
**Success Criteria** (what must be TRUE):

  1. Runnable, self-contained Google Colab notebook (`notebooks/colab_tqc_benchmark.ipynb`) provided for MuJoCo benchmark training with Google Drive / artifact sync.
  2. Multi-tier compute dispatcher auto-detects highest available environment (Kaggle credentials → Colab runtime → CUDA GPU → CPU fallback).
  3. Seamless benchmark execution dispatch across tiers for MuJoCo environments (HalfCheetah, Hopper, Walker2d, Ant, Humanoid).
  4. Unit and integration tests verify fallback resolution and notebook validity.

**Plans**: 0 plans

Plans:

- [ ] TBD (run /gsd-plan-phase 8 to break down)
