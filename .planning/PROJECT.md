# TQC Paper Reproduction & Analysis (ICML 2020)

## What This Is

An academic and empirical reproduction of the ICML 2020 paper "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics" (TQC) by Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, and Dmitry Vetrov. The project delivers a comprehensive mathematical and conceptual explanation document, an exact PyTorch-based reproduction of the TQC algorithm following the paper's specification, and experimental evaluation on standard continuous control benchmarks across MuJoCo environments.

## Core Value

Faithful, reproducible implementation and rigorous empirical evaluation of Truncated Quantile Critics against standard baselines (like Soft Actor-Critic) on MuJoCo continuous control benchmarks, backed by a comprehensive theoretical explanation document.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Detailed explanatory Markdown document (`docs/paper_explanation.md`) covering problem motivation (overestimation bias in actor-critic methods, limitations of SAC/TD3/QR-DQN), theoretical formulation, distributional Bellman equation, quantile regression, truncation operator, network architectures, and hyperparameter dynamics.
- [ ] Modular PyTorch implementation of the TQC algorithm following the paper's exact architecture: actor network (squashed Gaussian policy), ensemble of M=5 quantile critics each outputting N=25 quantiles, top-d quantile truncation operator, target distribution calculation, Huber quantile loss, and entropy temperature (alpha) auto-tuning.
- [ ] Replay buffer, environment wrappers, and training pipeline supporting Gymnasium/Gym MuJoCo continuous control tasks.
- [ ] Experiment runner with multi-seed training, checkpointing, and metrics logging (rewards, Q-value estimations, critic losses, alpha values).
- [ ] Evaluation on full continuous control benchmark suite: HalfCheetah, Hopper, Walker2d, Ant, and Humanoid.
- [ ] Plotting and visualization pipeline reproducing training return curves and overestimation bias analysis comparing TQC with baselines (SAC).
- [ ] Unit and integration tests verifying network forward passes, quantile loss computation, truncation correctness, and training step validity.

### Out of Scope

- [ ] Presentation slides and evaluation defense materials — deferred to Issue #3 until experimental reproduction and analysis are completed.
- [ ] Rigorous theoretical proofs of lemmas and theorems from the paper — explicitly permitted to skip per course instructions.
- [ ] Mobile or web UI deployments — command-line and script-based execution only.

## Context

- **Source Paper**: "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics", ICML 2020 (PMLR 119:5556-5566).
- **Official Reference**: `bayesgroup/tqc_pytorch`.
- **Target Environments**: MuJoCo physics engine via Gymnasium (v4/v5 or v2/v3 compatible).
- **Course Context**: RL course project requiring paper understanding, conceptual documentation, and empirical reproduction before the endsem presentation.
- **GitHub Issues**:
  - Issue #1: Explanation document for what the paper is about.
  - Issue #2: Reproduce the experimental results in the paper.
  - Issue #3: Create presentation materials for project evaluation.

## Constraints

- **Tech Stack**: Python 3.10+, PyTorch, Gymnasium/Gym, MuJoCo, NumPy, Matplotlib/Seaborn.
- **Algorithm Fidelity**: Must adhere to the exact architecture, hyperparameters, and operators described in the Kuznetsov et al. paper.
- **Timeline**: Must complete reproduction and documentation before the course endsem presentation deadline.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Implement pure PyTorch matching author's architecture | Ensures exact fidelity to the paper's specifications and hyperparameters | — Pending |
| Cover full benchmark suite (HalfCheetah, Hopper, Walker2d, Ant, Humanoid) | Provides complete reproduction matching all figures and tables in the paper | — Pending |
| Defer presentation creation to Issue #3 | Prioritizes experimental rigor and conceptual depth first | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-09 after initialization*
