---
phase: 11-benchmark-training-ant-v4
plan: 01
status: completed
date: 2026-11-06
tasks_completed: 4
files_created:
  - scripts/run_ant_benchmark.py
  - notebooks/kaggle_tqc_ant_phase11.ipynb
  - tests/test_ant_benchmark.py
files_modified:
  - configs/benchmark_matrix.yaml
  - scripts/run_remote_experiment.py
  - src/tqc/remote/kaggle_auth.py
  - src/tqc/remote/kaggle_packager.py
tests_passed:
  - tests/test_ant_benchmark.py (4 tests passing)
---

# Phase 11 Plan 01: Ant-v4 Benchmark Training Summary

## Summary of Accomplishments
1. **Benchmark Configuration**:
   - Added `ant_pilot` (single seed 42) and `ant_phase11` (seeds 42..46, 1M steps, drop_top 5) presets to `configs/benchmark_matrix.yaml`.
2. **Dedicated Benchmark Runner (`scripts/run_ant_benchmark.py`)**:
   - Implemented distributed multi-seed training across seeds `[42, 43, 44, 45, 46]` for the 8-DoF continuous action space.
   - Built overestimation bias analysis comparing predicted Q-quantiles against true discounted Monte Carlo rollouts.
   - Integrated automated reproduction plotting (`ant_reproduction_learning_curves.png`, `ant_bias_diagnostics.png`, `ant_results_table.md`).
3. **Kaggle GPU Notebook (`notebooks/kaggle_tqc_ant_phase11.ipynb`)**:
   - Generated self-contained, 9-cell Kaggle GPU benchmark notebook with headless EGL rendering and artifact packaging.
4. **Deterministic Verification**:
   - Verified 4 automated tests passing in `tests/test_ant_benchmark.py`.
   - Executed live smoke benchmark producing verified artifacts in `runs/Ant-v4/` and `results/Ant-v4/`.
