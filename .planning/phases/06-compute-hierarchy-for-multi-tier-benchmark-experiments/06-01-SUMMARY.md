---
phase: 06-compute-hierarchy-for-multi-tier-benchmark-experiments
plan: 01
subsystem: infra
tags: [kaggle, cloud, packaging, authentication, smoke-test, pytorch]

requires:
  - phase: 04-full-suite-mujoco-benchmark-orchestration
    provides: Training loops, metrics logging, checkpointing, and evaluation protocols
provides:
  - Kaggle authentication resolver with dual-source support and interactive prompt fallback
  - Kaggle kernel packager and metadata builder (kernel-metadata.json, remote_entrypoint.py, tqc_source.tar.gz)
  - Pre-flight local smoke validator executing 100-step test before cloud deployment
  - Comprehensive unit test suites for auth and packaging
affects:
  - 06-02-PLAN.md
  - scripts/run_remote_experiment.py
  - scripts/sync_remote_artifacts.py

actuals:
  tokens: 18500
  tasks: 2
  commits: 0

tech-stack:
  added: [kaggle]
  patterns: [dual-credential-resolution, isolated-archive-packaging, preflight-smoke-validation]

key-files:
  created:
    - src/tqc/remote/__init__.py
    - src/tqc/remote/kaggle_auth.py
    - src/tqc/remote/kaggle_packager.py
    - tests/test_remote_auth.py
    - tests/test_remote_packager.py
  modified: []

key-decisions:
  - "D-01 & D-16: Dual-credential resolution checks ~/.kaggle/kaggle.json first, falls back to KAGGLE_USERNAME/KAGGLE_KEY environment variables and .env, with interactive prompting fallback if credentials are absent."
  - "D-02, D-03, D-14, D-15: Kernel packaging compiles kernel-metadata.json, packages src/ into tqc_source.tar.gz, and generates remote_entrypoint.py configuring GPU auto-detection, cuDNN benchmark mode, 1M replay buffer, batch size 256, and SHA256 checksum generation."
  - "D-13: Preflight local smoke validation automatically scales warmup steps and batch size dynamically to guarantee error-free execution on small step dry runs."

patterns-established:
  - "Kaggle credential priority hierarchy with secure prompt fallback"
  - "Remote entrypoint self-contained execution with archive unpack and SHA256 generation"
  - "Pre-flight local dry-run validation gate before remote workload dispatch"

requirements-completed:
  - Issue #4
  - Issue #7

coverage:
  - id: D1
    description: "Kaggle credential detection from ~/.kaggle/kaggle.json, environment variables, and interactive prompt with API validation"
    requirement: "Issue #4"
    verification:
      - kind: unit
        ref: "tests/test_remote_auth.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "Kernel packager generating kernel-metadata.json, remote_entrypoint.py, and tqc_source.tar.gz"
    requirement: "Issue #4"
    verification:
      - kind: unit
        ref: "tests/test_remote_packager.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "Pre-flight local smoke validator verifying environment creation, agent training, and checkpoint creation"
    requirement: "Issue #7"
    verification:
      - kind: unit
        ref: "tests/test_remote_packager.py#test_run_local_preflight_smoke_success"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-10
status: complete
---

# Phase 06: Compute Hierarchy - Plan 01 Summary

**Kaggle dual-credential authentication, kernel packaging with CUDA optimizations, and local pre-flight smoke validator for remote TQC benchmark execution.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-09T18:37:00Z
- **Completed:** 2026-09-09T18:49:00Z
- **Tasks:** 2 completed
- **Files modified:** 5 created

## Accomplishments

- Implemented `src/tqc/remote/kaggle_auth.py` with multi-source credential resolution (`~/.kaggle/kaggle.json`, environment variables, `.env`), programmatic API validation, and interactive user prompts.
- Implemented `src/tqc/remote/kaggle_packager.py` generating `kernel-metadata.json`, self-contained `remote_entrypoint.py` with automated dependency installation (`gymnasium[mujoco]`, `mujoco`), CUDA detection, PyTorch cuDNN benchmark optimization, and `artifacts.tar.gz` packaging with SHA256 checksums.
- Created `run_local_preflight_smoke` for running a 100-step dry run validating environment creation, tensor arithmetic, metrics logging, and checkpoint generation prior to remote push.
- Delivered 15 unit tests across `tests/test_remote_auth.py` and `tests/test_remote_packager.py` with 100% pass rate.

## Files Created/Modified

- `src/tqc/remote/__init__.py` - Remote module exports
- `src/tqc/remote/kaggle_auth.py` - Kaggle credential resolver and validator
- `src/tqc/remote/kaggle_packager.py` - Kernel packager, entrypoint builder, and smoke runner
- `tests/test_remote_auth.py` - Unit tests for authentication
- `tests/test_remote_packager.py` - Unit tests for packager and smoke runner

## Decisions Made

- Added dynamic scaling for `warmup_steps` and `batch_size` in `run_local_preflight_smoke` so smoke runs with arbitrary step limits (such as 30 or 100 steps) never exceed available replay buffer transitions.
- Checkpoint discovery in preflight smoke validator uses recursive `.rglob("*.pt")` to locate checkpoints regardless of flat or nested folder structures.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug Fix] Adapted warmup_steps and batch_size dynamically for smoke runs**
- **Found during:** Task 2 (`test_run_local_preflight_smoke_success`)
- **Issue:** Static warmup steps (20) with batch size (32) raised ValueError when buffer had fewer transitions than batch size.
- **Fix:** Scaled `warmup_steps = min(20, max(5, steps // 3))` and `batch_size = min(16, warmup_steps)`.
- **Files modified:** `src/tqc/remote/kaggle_packager.py`
- **Verification:** Unit tests pass with 0 errors.

**2. [Rule 1 - Bug Fix] Checkpoint search in smoke runner changed to rglob**
- **Found during:** Task 2 (`test_run_local_preflight_smoke_success`)
- **Issue:** `MetricsLogger` saves checkpoints at the root of `exp_dir` (`checkpoint_*.pt`, `best_model.pt`) rather than an empty `checkpoints/` subfolder.
- **Fix:** Switched search from `(run_dir / "checkpoints").glob("*.pt")` to `run_dir.rglob("*.pt")`.
- **Files modified:** `src/tqc/remote/kaggle_packager.py`
- **Verification:** Test passed in 8.35s.

## User Setup Required

None - Kaggle credentials can be placed in `~/.kaggle/kaggle.json` or `.env`, or provided via interactive prompt during execution.

## Next Phase Readiness

Plan 06-01 completed. Ready for Plan 06-02 (Dual-Slot Queue Manager, Artifact Sync Script, Unified Experiment Runner, Declarative Config, and Documentation).
