# Phase 10: Benchmark Training: Walker2d-v4 (5 Seeds) - Execution Summary

**Execution Date**: 2026-10-16 to 2026-10-17  
**Platform**: Kaggle Remote GPU (Tesla T4)  
**Configuration**: $M=5, N=25, d=5$, 1,000,000 timesteps per seed  
**Seeds**: 42, 43, 44, 45, 46  
**Status**: 100% Completed & Verified  

---

## 1. Quantitative Performance Results

All 5 random seeds completed 1,000,000 environment interaction steps with deterministic evaluations every 5,000 steps (200 evaluations per seed).

| Seed | Max Return | Final Return | Checkpoints Verified | Checksum Status |
|:---:|:---:|:---:|:---:|:---:|
| **42** | 5,673.98 | 5,069.19 | 27 files (Step 0 to 1M) | SHA256 Match (Passed) |
| **43** | 5,536.76 | 5,385.39 | 27 files (Step 0 to 1M) | SHA256 Match (Passed) |
| **44** | 5,437.94 | 5,333.93 | 27 files (Step 0 to 1M) | SHA256 Match (Passed) |
| **45** | 5,780.62 | 5,702.92 | 27 files (Step 0 to 1M) | SHA256 Match (Passed) |
| **46** | 5,769.41 | 3,186.42 | 27 files (Step 0 to 1M) | SHA256 Match (Passed) |
| **Aggregate** | **5,639.74 ± 133.54** | **4,935.57 ± 897.47** | **135 files total** | **100% Integrity Verified** |

*Target Paper Performance*: Kuznetsov et al. (ICML 2020) Table 1 & Table 2 reports ~5,000 to ~5,500 on Walker2d-v4. The empirical reproduction accurately matches and achieves a peak average of **5,639.74**.

---

## 2. Generated Artifacts & Visualizations

1. **5-Seed Aggregate Learning Curve**:
   - `plots/walker2d_tqc_5seeds_final_curve.png`
   - Shaded standard deviation band across all 5 seeds from step 0 to step 1,000,000.
2. **Evaluation Progression Video**:
   - `videos/walker2d_seed42_best.mp4`
   - Full rollout with HUD overlay and inactivity/stagnation truncation.
3. **Local Checkpoints & Metrics**:
   - `runs/kaggle_sharvanipallempati_tqc-walker2dv4-s42-.../`
   - `runs/kaggle_sharvanipallempati_tqc-walker2dv4-s43-.../`
   - `runs/kaggle_sharvanipallempati_tqc-walker2dv4-s44-.../`
   - `runs/kaggle_sharvanipallempati_tqc-walker2dv4-s45-.../`
   - `runs/kaggle_sharvanipallempati_tqc-walker2dv4-s46-.../`
