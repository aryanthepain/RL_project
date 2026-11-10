# Phase 11: Benchmark Training: Ant-v4 Context

## Overview
Phase 11 executes the distributed multi-seed benchmark training for **Ant-v4** across 5 random seeds (`42, 43, 44, 45, 46`) for 1,000,000 steps each, reproducing Table 1 & Table 2 of Kuznetsov et al. (ICML 2020) "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics".

## Mathematical & Hyperparameter Invariants
- **Target Environment**: `Ant-v4` (Observation Space: 27-dimensional continuous Box, Action Space: 8-dimensional continuous Box $[-1, 1]$ across quadruped joints).
- **Critic Ensemble**: $M=5$ critic networks, each predicting $N=25$ quantiles at cumulative probabilities $\tau_i = \frac{2i - 1}{2N}$ for $i \in \{1, \dots, N\}$.
- **Pooled Quantile Mixture**: $M \times N = 125$ atoms.
- **Truncation Parameter**: $d=5$ topmost quantiles dropped ($k = 125 - 5 = 120$ quantiles retained) ensuring learning stability in the high-dimensional 8-DoF action space.
- **Target Performance**: $6,293 \pm 276$ (Table 1) / $8.01 \pm 0.87 \times 10^3$ (Table 2), Max score: $9,011$.
- **Baseline Comparison**: SAC ($5,850 \pm 198$ / $6.16 \pm 0.93 \times 10^3$), TD3 ($5.68 \pm 1.04 \times 10^3$), TrulyPPO ($-0.01 \pm 0.00$).

## Deliverables
1. `scripts/run_ant_benchmark.py`: Unified orchestrator supporting multi-seed training, 8-DoF continuous control evaluation, overestimation bias analysis, learning curve plotting, and results table generation.
2. `configs/benchmark_matrix.yaml`: Dedicated presets `ant_phase11` and `ant_pilot`.
3. `notebooks/kaggle_tqc_ant_phase11.ipynb`: Autonomous Kaggle GPU notebook with headless EGL rendering, artifact archiving, and in-notebook video playback.
4. `tests/test_ant_benchmark.py`: Automated test suite verifying environment specs, 8-DoF action space integrity, hyperparameter metadata, matrix presets, fast smoke training, and plot rendering.
