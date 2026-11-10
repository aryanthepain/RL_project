# Phase 9: Benchmark Training: Hopper-v4 Context

## Overview
Phase 9 executes the distributed multi-seed benchmark training for **Hopper-v4** across 5 random seeds (`42, 43, 44, 45, 46`) for 1,000,000 steps each, reproducing Table 1 & Table 2 of Kuznetsov et al. (ICML 2020) "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics".

## Mathematical & Hyperparameter Invariants
- **Target Environment**: `Hopper-v4` (Observation Space: 11-dimensional continuous Box, Action Space: 3-dimensional continuous Box $[-1, 1]$).
- **Critic Ensemble**: $M=5$ critic networks, each predicting $N=25$ quantiles at cumulative probabilities $\tau_i = \frac{2i - 1}{2N}$ for $i \in \{1, \dots, N\}$.
- **Pooled Quantile Mixture**: $M \times N = 125$ atoms.
- **Truncation Parameter**: $d=5$ topmost quantiles dropped ($k = 125 - 5 = 120$ quantiles retained) per ICML 2020 paper specification (Table 2 & Section 4.2).
- **Target Performance**: $3,695 \pm 157$ (Table 1) / $3.71 \pm 0.16 \times 10^3$ (Table 2), Max score: $4,288$.
- **Baseline Comparison**: SAC ($3,456 \pm 88$), TD3 ($3,310 \pm 550$).

## Deliverables
1. `scripts/run_hopper_benchmark.py`: Unified orchestrator supporting multi-seed training, deterministic evaluation, overestimation bias analysis, learning curve plotting, and results table generation.
2. `configs/benchmark_matrix.yaml`: Dedicated presets `hopper_phase9` and `hopper_pilot`.
3. `notebooks/kaggle_tqc_hopper_phase09.ipynb`: Autonomous Kaggle GPU notebook with headless EGL rendering, artifact archiving, and in-notebook video playback.
4. `tests/test_hopper_benchmark.py`: Automated test suite verifying environment specs, hyperparameter metadata, matrix presets, fast smoke training, and plot rendering.
