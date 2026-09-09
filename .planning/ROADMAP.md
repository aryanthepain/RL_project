# Roadmap: TQC Paper Reproduction & Analysis (ICML 2020)

## Overview

A systematic 4-phase horizontal roadmap taking the project from foundational theoretical documentation to core algorithm implementation, environment infrastructure, and final multi-seed benchmark experimental reproduction on MuJoCo continuous control tasks.

## Phases

- [x] **Phase 1: Paper Theoretical & Algorithmic Documentation** - Comprehensive explanation document detailing problem motivation, mathematical formulation, truncation operator, and architectures (Issue #1).
- [x] **Phase 2: Core Algorithm & Quantile Networks** - Modular PyTorch implementation of actor, critic ensemble ($M=5, N=25$), Huber quantile loss, truncation operator, and automatic entropy tuning.
- [ ] **Phase 3: Environment Harness, Replay Buffer & Training Pipeline** - 1M replay buffer, Gymnasium MuJoCo environment integration, reproducible seeding, and training loop with evaluation logging (Issue #2 foundation).
- [ ] **Phase 4: Multi-Seed Benchmark Experiments, Bias Analysis & Replication Curves** - Benchmarking runs on MuJoCo suite, overestimation bias analysis, and reproduction plotting matching ICML 2020 paper figures (Issue #2 completion).

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
- [ ] 03-01: Implement replay buffer and Gymnasium MuJoCo environment harness with seed control
- [ ] 03-02: Implement end-to-end training loop, evaluation runner, checkpointing, and metrics logger

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
- [ ] 04-01: Execute benchmark training runs and overestimation bias Monte Carlo evaluations
- [ ] 04-02: Generate replication return curves, bias comparison plots, and comparative results summary

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Paper Theoretical & Algorithmic Documentation | 1/1 | Complete | 2026-09-09 |
| 2. Core Algorithm & Quantile Networks | 2/2 | Complete | 2026-09-09 |
| 3. Environment Harness, Replay Buffer & Training Pipeline | 0/2 | Not started | - |
| 4. Multi-Seed Benchmark Experiments, Bias Analysis & Replication Curves | 0/2 | Not started | - |

