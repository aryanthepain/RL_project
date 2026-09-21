# Requirements: TQC Paper Reproduction & Analysis (ICML 2020)

**Defined:** 2026-09-09
**Core Value:** Faithful, reproducible implementation and rigorous empirical evaluation of Truncated Quantile Critics against standard baselines (like Soft Actor-Critic) on MuJoCo continuous control benchmarks, backed by a comprehensive theoretical explanation document.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Theoretical Documentation (DOC)

- [x] **DOC-01**: Comprehensive paper overview and motivation explaining overestimation bias in deep Q-learning, why clipped double Q-learning (SAC, TD3) can still under- or overestimate, and how distributional quantile critics resolve this.
- [x] **DOC-02**: Formal mathematical explanation of the distributional Bellman equation, quantile regression loss (Huber quantile loss), and the quantile mixture distribution across critic ensembles.
- [x] **DOC-03**: Detailed analysis of the Truncation operator: sorting pooled quantiles, discarding top $d$ atoms, controlling return estimation bias, and the empirical impact of the truncation parameter $d$.
- [x] **DOC-04**: Architectural breakdown, pseudo-code, and hyperparameter catalog matching the paper's exact specification (`bayesgroup/tqc_pytorch`).

### Core Algorithm & Quantile Networks (ALGO)

- [x] **ALGO-01**: Actor network implementing squashed Gaussian policy with reparameterization trick, tanh squashing, and log-probability calculation.
- [x] **ALGO-02**: Critic ensemble network implementing $M=5$ independent multi-layer perceptron quantile critics, each predicting $N=25$ quantiles.
- [x] **ALGO-03**: Quantile regression loss computation using Huber quantile loss with asymmetric penalty weights.
- [x] **ALGO-04**: Truncation operator pooling $M \times N = 125$ quantiles from the target critic ensemble, sorting in ascending order, and dropping the top $d$ quantiles to construct the target distribution.
- [x] **ALGO-05**: Soft target network updates with Polyak averaging parameter $\tau = 0.005$ and automatic entropy temperature ($\alpha$) tuning via dual gradient descent.

### Environment & Buffer Infrastructure (ENV)

- [x] **ENV-01**: High-performance uniform replay buffer storing $(s, a, r, s', \text{done})$ transitions with capacity $10^6$ and fast batch sampling.
- [x] **ENV-02**: Standardized Gymnasium MuJoCo environment harness supporting HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4, and Humanoid-v4 with observation and action space validation.
- [x] **ENV-03**: Seed management system ensuring reproducibility across PyTorch, NumPy, Python random, and Gymnasium environment state.

### Training & Benchmarking Pipeline (EXP)

- [x] **EXP-01**: Multi-seed training pipeline running 1M/3M steps per environment with periodic deterministic evaluation episodes.
- [x] **EXP-02**: Structured metric logger capturing evaluation returns, critic loss, actor loss, alpha value, and Q-estimation statistics.
- [x] **EXP-03**: Baseline comparison support (logging or running SAC baseline under identical evaluation protocols).

### Results Analysis & Publication Curves (VIZ)

- [x] **VIZ-01**: Plotting scripts generating smoothed return curves with standard deviation/interquartile confidence intervals across seeds for all MuJoCo benchmarks.
- [x] **VIZ-02**: Empirical overestimation bias analysis script comparing estimated Q-values against actual Monte Carlo rollouts.
- [x] **VIZ-03**: Summary results table comparing reproduced scores against Table 1 & Table 2 in Kuznetsov et al. (ICML 2020).

### Evaluation Video Inactivity Truncation & Stagnation Detection (TRUNC / Issue #8)

- [x] **TRUNC-01**: Adaptive dual-metric inactivity and stagnation detection in rollout loop with configurable warmup and trailing padding frames.
- [x] **TRUNC-02**: OpenGL render-skipping optimization on post-truncation steps with strict simulation return and step count preservation.
- [x] **TRUNC-03**: Amber status badge HUD overlays, 50% selective 3D scene dimming for freeze-frame multi-video grids, progression reels, and CLI flags.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Extensions & Presentation (EXT)

- **EXT-01**: Methodological extensions or hyperparameter ablation (e.g., dynamic truncation $d$, alternative ensemble sizes $M$) to test potential outperformance as invited by course problem statement.
- **PRES-01**: Slide deck and evaluation defense presentation materials (tracked in Issue #3).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Formal proofs of lemmas/theorems | Explicitly permitted to skip per course instructions; focus is on concepts and empirical reproduction |
| Web UI / Dashboard application | CLI and script-driven workflow is standard for deep RL benchmarks |
| Non-continuous control environments (Atari/Discrete) | TQC is explicitly designed for continuous action spaces |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DOC-01 | Phase 1 | Complete |
| DOC-02 | Phase 1 | Complete |
| DOC-03 | Phase 1 | Complete |
| DOC-04 | Phase 1 | Complete |
| ALGO-01 | Phase 2 | Complete |
| ALGO-02 | Phase 2 | Complete |
| ALGO-03 | Phase 2 | Complete |
| ALGO-04 | Phase 2 | Complete |
| ALGO-05 | Phase 2 | Complete |
| ENV-01 | Phase 3 | Complete |
| ENV-02 | Phase 3 | Complete |
| ENV-03 | Phase 3 | Complete |
| EXP-01 | Phase 3 | Complete |
| EXP-02 | Phase 3 | Complete |
| EXP-03 | Phase 3 | Complete |
| VIZ-01 | Phase 4 | Complete |
| VIZ-02 | Phase 4 | Complete |
| VIZ-03 | Phase 4 | Complete |
| Issue #8 | Phase 7 | Complete |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-09*
*Last updated: 2026-09-09 after initial definition*
