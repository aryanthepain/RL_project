---
phase: 04-multi-seed-benchmark-experiments-bias-analysis-replication-curves
plan: 02
subsystem: viz
tags: [visualization, matplotlib, learning-curves, overestimation-bias, latex-table, unit-tests]

requires:
  - plan: 04-01
    provides: Benchmark orchestrator and Overestimation bias engine
provides:
  - Publication-grade plotting in src/tqc/plotter.py
  - Comparative benchmark table generator in src/tqc/eval_table.py
  - Unit and integration tests in tests/test_plotter.py, tests/test_eval_table.py
affects:
  - Issue #2 completion (reproduction and analysis)

actuals:
  tokens: 20000
  tasks: 2
  commits: 1

tech-stack:
  added: [matplotlib, seaborn]
  patterns:
    - Headless Matplotlib rendering (backend 'Agg') for server/CLI environments
    - Multi-seed learning curve interpolation with shaded +/- 1 std error bands
    - Side-by-side diagnostic plots comparing predicted Q against true Monte Carlo returns
    - Dual Markdown and publication-ready LaTeX table generation

key-files:
  created:
    - src/tqc/plotter.py
    - src/tqc/eval_table.py
    - tests/test_plotter.py
    - tests/test_eval_table.py
  modified:
    - src/tqc/__init__.py

key-decisions:
  - "Configured matplotlib to use Agg backend ensuring clean headless image generation on Windows/Linux"
  - "Interpolated variable-step seed logs onto uniform grid for smooth multi-seed aggregation"
  - "Embedded official ICML 2020 Table 1 reported numbers for direct baseline comparison"

patterns-established:
  - "Automated unit tests with synthetic data verifying image dimensions and table parsing without requiring full training compute"

requirements-completed:
  - VIZ-01
  - VIZ-03

coverage:
  - id: D1
    description: "Publication-grade learning curve plotting with shaded confidence intervals across seeds"
    requirement: VIZ-01
    verification:
      - kind: unit
        ref: "tests/test_plotter.py::test_plot_learning_curves"
        status: pass
    human_judgment: false
  - id: D2
    description: "Comparative results table generator comparing empirical results against ICML 2020 Table 1"
    requirement: VIZ-03
    verification:
      - kind: unit
        ref: "tests/test_eval_table.py::test_generate_comparison_table_with_mock_data"
        status: pass
    human_judgment: false

duration: 4min
completed: 2026-09-09
status: complete
---

# Phase 4: Plan 02 Summary

**Implemented Publication Plotting Engine (Learning Curves & Bias Comparison) and Comparative Results Table Generator with 100% passing tests.**

## Performance
- **Duration:** 4 min
- **Started:** 2026-09-09T22:29:00Z
- **Completed:** 2026-09-09T22:31:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Implemented `plot_learning_curves` (`src/tqc/plotter.py`) generating smoothed multi-seed return curves with shaded confidence intervals (VIZ-01).
- Implemented `plot_bias_comparison` (`src/tqc/plotter.py`) producing side-by-side estimation error bar charts and Q vs Monte Carlo scatter diagrams (VIZ-02).
- Implemented `generate_comparison_table` (`src/tqc/eval_table.py`) outputting Markdown and LaTeX tables comparing empirical returns against published ICML 2020 numbers (VIZ-03).
- Added comprehensive unit tests in `tests/test_plotter.py` and `tests/test_eval_table.py`.

## Milestone Readiness
- All 4 roadmap phases are complete.
- Core algorithm, environment harness, replay buffer, training pipeline, multi-seed benchmarking, bias analysis, plotting, and results synthesis are fully implemented and verified.
