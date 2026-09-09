# Phase 4: Multi-Seed Benchmark Experiments, Bias Analysis & Replication Curves - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver empirical replication, overestimation bias evaluation, and visualization tools completing Issue #2:
1. Multi-seed benchmark runner (`src/tqc/benchmark.py`) orchestrating training runs across continuous locomotion benchmarks (HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4, Humanoid-v4) for both TQC and SAC baseline.
2. Monte Carlo overestimation bias analysis script (`src/tqc/bias_analysis.py`) comparing critic quantile predictions $\hat{Q}(s, a)$ against true discounted Monte Carlo returns $G_t = \sum_{k=0}^T \gamma^k r_{t+k}$ (VIZ-02).
3. Publication-grade plotting module (`src/tqc/plotter.py`) rendering learning return curves with shaded confidence intervals (mean ± std / IQR) matching the visual quality of Figures 1 & 2 in Kuznetsov et al. (ICML 2020) (VIZ-01).
4. Results synthesis and comparative table generator (`src/tqc/eval_table.py`) benchmarking empirical returns against published ICML 2020 numbers (Table 1 & Table 2) (VIZ-03).
5. Comprehensive test suite (`tests/test_bias.py`, `tests/test_plotter.py`, `tests/test_eval_table.py`) verifying mathematical correctness of bias calculations, plot rendering, and table formatting.
</domain>

<decisions>
## Implementation Decisions

### Multi-Seed Benchmark Runner
- **D-01:** Implement `run_benchmark(...)` in `src/tqc/benchmark.py` supporting sequential or batch execution across environments, algorithms (`tqc`, `sac`), and seeds.
- **D-02:** Provide dry-run / fast test execution modes for rapid verification without burning compute.

### Empirical Overestimation Bias Analysis (VIZ-02)
- **D-03:** Implement `compute_overestimation_bias(agent, env, n_samples=50, gamma=0.99, max_steps=1000)` in `src/tqc/bias_analysis.py`.
- **D-04:** For each sampled state $s_t$:
  - Calculate estimated Q-value from the critic ensemble: $\hat{Q}(s_t, a_t)$ (mean of all predicted quantiles or mean of truncated quantiles).
  - Execute full trajectory rollout under deterministic policy $\pi(s)$ to compute true empirical discounted return $G_t = \sum_{k=0}^{T-1} \gamma^k r_{t+k}$.
  - Compute bias $\delta_t = \hat{Q}(s_t, a_t) - G_t$.
- **D-05:** Output bias distributions, summary statistics (mean bias, standard error), and generate visual diagnostic plots comparing $\hat{Q}$ vs $G_t$.

### Replication Curves & Visualization (VIZ-01)
- **D-06:** Implement `plot_learning_curves(log_dirs, output_path)` in `src/tqc/plotter.py` reading structured `metrics.csv` / `metrics.jsonl` files.
- **D-07:** Aggregate across seeds by aligning timesteps, interpolate onto uniform grid, and plot mean curve with shaded $\pm 1 \text{std}$ (or 25th-75th percentile) band.
- **D-08:** Support comparative multi-algorithm overlay (TQC in vibrant blue, SAC in amber/orange) with publication-quality typography and legends.

### Comparative Table Generation (VIZ-03)
- **D-09:** Implement `generate_comparison_table(results_dir, output_path)` in `src/tqc/eval_table.py`.
- **D-10:** Embed reference baseline scores from Kuznetsov et al. (ICML 2020 Table 1):
  - HalfCheetah: TQC 14,028 ± 232 vs SAC 11,489 ± 255
  - Hopper: TQC 3,695 ± 157 vs SAC 3,456 ± 88
  - Walker2d: TQC 5,640 ± 192 vs SAC 4,892 ± 204
  - Ant: TQC 6,293 ± 276 vs SAC 5,850 ± 198
  - Humanoid: TQC 6,564 ± 350 vs SAC 5,420 ± 410
- **D-11:** Produce both clean Markdown and LaTeX table outputs.

</decisions>

<canonical_refs>
## Canonical References

### Project Documentation
- `docs/paper_explanation.md` — Section 1 (Motivation & Overestimation Bias), Section 4 (Truncation Operator), Section 5 (Empirical Evaluation).
- `.planning/REQUIREMENTS.md` — VIZ-01, VIZ-02, VIZ-03.
- `.planning/ROADMAP.md` — Phase 4 success criteria.

### External Literature
- Kuznetsov et al. (ICML 2020) — "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics".
- Fujimoto et al. (2018) — "Addressing Function Approximation Error in Actor-Critic Methods" (TD3 bias measurement methodology).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/tqc/agent.py`: `TQCAgent` with `critic(states, actions)` and `select_action`.
- `src/tqc/train.py`: `train_tqc(...)` entry point producing `metrics.csv` and `metrics.jsonl`.
- `src/tqc/logger.py`: `MetricsLogger` data layout.
- `src/tqc/envs.py`: `make_env`, `get_env_dims`, `get_env_metadata`.

</code_context>

<specifics>
## Specific Ideas

- Unit test bias analysis using a mock agent with known Q predictions and deterministic mock transitions to assert exact bias calculation.
- Unit test plotter using synthetic CSV metric files to verify PNG generation and matplotlib figure styling without needing actual training logs.
- Unit test table generator to ensure markdown and LaTeX output formatting with percentage differences.

</specifics>

<deferred>
## Deferred Ideas

- Course presentation slide deck deferred to GitHub Issue #3.

</deferred>

---

*Phase: 04-multi-seed-benchmark-experiments-bias-analysis-replication-curves*
*Context gathered: 2026-09-09*
