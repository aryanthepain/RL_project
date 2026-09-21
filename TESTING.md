# Verification & Testing Guide

This document details the testing architecture, test suites, execution commands, and verification protocols for the **TQC Paper Reproduction** codebase.

---

## 1. Testing Philosophy & Invariants

1. **Deterministic Verification**: Every mathematical calculation, neural network forward pass, truncation sorting step, and compute hierarchy transition is guarded by deterministic unit tests.
2. **Zero Speculative Code**: No PR or feature branch is merged into `main` without 100% passing tests.
3. **Execution Command**: Always invoke pytest via the Python module syntax:
   ```bash
   python -m pytest
   ```
   *(Running bare `pytest` may fail to discover the repository root on some operating systems).*

---

## 2. Test Execution Quick Reference

```bash
# 1. Run the entire test suite (105 tests across 25 suites)
python -m pytest

# 2. Run with verbose output and test timing
python -m pytest -v --durations=10

# 3. Run a specific test suite
python -m pytest tests/test_agent.py
python -m pytest tests/test_truncation.py
python -m pytest tests/test_tier_detector.py

# 4. Fast CPU smoke test for training loop
python -m pytest tests/test_train.py -k test_train_tqc_smoke_cpu

# 5. Filter tests by keyword expression
python -m pytest -k "truncation or compute"
```

---

## 3. Test Suite Taxonomy & Coverage

The repository includes **25 dedicated test modules** covering every layer of the system:

### A. Algorithmic Math & Networks
- [`tests/test_truncation.py`](file:///d:/projects/RL_project/tests/test_truncation.py): Validates top-$d$ quantile sorting, pool truncation ($M \times N - d$), and asymmetric Huber quantile regression loss computation against hand-calculated values.
- [`tests/test_critic.py`](file:///d:/projects/RL_project/tests/test_critic.py): Verifies ensemble architecture ($M=5$ critics with $N=25$ quantiles each) and output tensor shape integrity.
- [`tests/test_actor.py`](file:///d:/projects/RL_project/tests/test_actor.py): Tests squashed Gaussian reparameterized policy, log-probability calculation with tanh correction, and bounded action bounds.
- [`tests/test_bias.py`](file:///d:/projects/RL_project/tests/test_bias.py): Validates overestimation bias measurement formulas against Monte Carlo return rollouts.
- [`tests/test_buffer.py`](file:///d:/projects/RL_project/tests/test_buffer.py): Verifies circular FIFO replay buffer storage, uniform batch sampling, and memory boundary conditions.

### B. Agent & Training Harness
- [`tests/test_agent.py`](file:///d:/projects/RL_project/tests/test_agent.py): Tests the complete `TQCAgent` update step, target quantile mixture construction, critic loss, actor policy loss, Polyak target updates ($\tau=0.005$), and automatic entropy temperature ($\alpha$) tuning.
- [`tests/test_train.py`](file:///d:/projects/RL_project/tests/test_train.py): Verifies end-to-end training loop, evaluation rollouts, periodic checkpointing, and resume-from-checkpoint capability.
- [`tests/test_envs.py`](file:///d:/projects/RL_project/tests/test_envs.py): Validates Gymnasium MuJoCo environment instantiation, observation/action space dimensions, and deterministic seeding.
- [`tests/test_logger.py`](file:///d:/projects/RL_project/tests/test_logger.py): Tests CSV metric logging, column schema consistency, and disk writing.
- [`tests/test_utils.py`](file:///d:/projects/RL_project/tests/test_utils.py): Validates random seed propagation across PyTorch, NumPy, and Python standard libraries.

### C. Video Telemetry & Inactivity Truncation
- [`tests/test_visualize.py`](file:///d:/projects/RL_project/tests/test_visualize.py): Verifies evaluation rollout recording and HUD telemetry banner generation.
- [`tests/test_visualize_truncation.py`](file:///d:/projects/RL_project/tests/test_visualize_truncation.py): Validates forward-velocity stagnation detection, early frame truncation, trailing frame padding, and metric preservation.
- [`tests/test_visualize_grid_truncation.py`](file:///d:/projects/RL_project/tests/test_visualize_grid_truncation.py): Tests multi-checkpoint 2x3 comparison grid rendering with truncated streams and dimming effects.
- [`tests/test_progression_runner.py`](file:///d:/projects/RL_project/tests/test_progression_runner.py): Verifies the progression checkpoint rendering orchestrator.

### D. Multi-Tier Compute Hierarchy & Dispatch
- [`tests/test_tier_detector.py`](file:///d:/projects/RL_project/tests/test_tier_detector.py): Tests automatic hardware probing (Kaggle credentials, Colab environment, NVIDIA CUDA GPU, CPU fallback) and hierarchy resolution.
- [`tests/test_compute_dispatcher.py`](file:///d:/projects/RL_project/tests/test_compute_dispatcher.py): Tests unified CLI job routing, preset parameter expansion, dry-run mode, and fallback resolution.
- [`tests/test_colab_notebook.py`](file:///d:/projects/RL_project/tests/test_colab_notebook.py): Validates programmatic generation and syntax integrity of `notebooks/colab_tqc_benchmark.ipynb`.
- [`tests/test_clean_runs.py`](file:///d:/projects/RL_project/tests/test_clean_runs.py): Tests disk usage auditing, run inspection, dry-run pruning, and safe deletion.

### E. Remote Cloud Execution (Kaggle)
- [`tests/test_remote_auth.py`](file:///d:/projects/RL_project/tests/test_remote_auth.py): Tests Kaggle API credential resolution from `~/.kaggle/kaggle.json`, environment variables, and interactive prompts.
- [`tests/test_remote_packager.py`](file:///d:/projects/RL_project/tests/test_remote_packager.py): Verifies kernel metadata generation, standalone source tarball creation, and remote entrypoint bundling.
- [`tests/test_remote_queue.py`](file:///d:/projects/RL_project/tests/test_remote_queue.py): Tests asynchronous dual-slot queue management enforcing Kaggle's 2-GPU concurrency limits.
- [`tests/test_remote_sync.py`](file:///d:/projects/RL_project/tests/test_remote_sync.py): Tests remote artifact pulling, SHA256 checksum verification, and `metrics.csv` schema validation.

---

## 4. Mocking & Test Isolation Guidelines

All external services and platform-specific hardware dependencies are strictly mocked in unit tests:
- **Kaggle API**: Mocked via `unittest.mock.patch` to verify queue logic without requiring live network calls or consuming weekly GPU quotas.
- **CUDA Device States**: Tested using mocked `torch.cuda.is_available()` to verify CPU fallbacks on non-GPU developer machines.
- **Gymnasium Render Frames**: Headless rendering uses synthetic frame buffers during testing to execute rapidly on local and CI runners.

---

## 5. Pre-Merge Verification Checklist

Before creating a Pull Request, confirm:
1. `python -m pytest` executes and outputs:
   ```text
   ================= 105 passed in ~65s =================
   ```
2. No temporary output files or uncommitted run logs are left behind.
3. Test coverage is maintained for any newly added functions or modules.
