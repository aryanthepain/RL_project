---
phase: 09-benchmark-training-hopper-v4
plan: 01
status: completed
date: 2026-11-06
tasks_completed: 4
files_created:
  - scripts/run_hopper_benchmark.py
  - notebooks/kaggle_tqc_hopper_phase09.ipynb
  - tests/test_hopper_benchmark.py
files_modified:
  - configs/benchmark_matrix.yaml
  - scripts/run_remote_experiment.py
  - src/tqc/remote/kaggle_auth.py
  - src/tqc/remote/kaggle_packager.py
tests_passed:
  - tests/test_hopper_benchmark.py (4 tests passing)
---

# Phase 09 Plan 01: Hopper-v4 Benchmark Training Summary

## Summary of Accomplishments
1. **Benchmark Configuration**:
   - Added `hopper_pilot` (single seed 42) and `hopper_phase9` (seeds 42..46, 1M steps, drop_top 5) presets to `configs/benchmark_matrix.yaml`.
2. **Dedicated Benchmark Runner (`scripts/run_hopper_benchmark.py`)**:
   - Implemented distributed multi-seed training across seeds `[42, 43, 44, 45, 46]`.
   - Built overestimation bias analysis comparing predicted Q-quantiles against true discounted Monte Carlo rollouts.
   - Integrated automated reproduction plotting (`hopper_reproduction_learning_curves.png`, `hopper_bias_diagnostics.png`, `hopper_results_table.md`).
3. **Kaggle GPU Notebook (`notebooks/kaggle_tqc_hopper_phase09.ipynb`)**:
   - Generated self-contained, 9-cell Kaggle GPU benchmark notebook with headless EGL rendering and artifact packaging.
4. **Deterministic Verification**:
   - Verified 4 automated tests passing in `tests/test_hopper_benchmark.py`.
   - Executed live smoke benchmark producing verified artifacts in `runs/Hopper-v4/` and `results/Hopper-v4/`.
