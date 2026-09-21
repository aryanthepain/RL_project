# Engineering & Team Collaboration Conventions

This document defines the strict engineering guidelines, Git workflow protocols, coding standards, and verification invariants required for all contributors on the **TQC Paper Reproduction** project.

---

## 1. Branching & Pull Request Lifecycle

### The Golden Invariant: Never Push Directly to `main`
The `main` branch is protected and contains only verified, reproducible milestone releases. All work—whether an entire GSD phase, an experimental feature, or a small bugfix—must originate on a dedicated branch.

### Branch Naming Standards
- **GSD Phases**: `gsd/phase-<N>-<slug>` (e.g., `gsd/phase-09-hopper-runs`)
- **Features & Enhancements**: `feat/<short-slug>` (e.g., `feat/team-onboarding-docs`)
- **Bug Fixes**: `fix/<short-slug>` (e.g., `fix/inactivity-threshold-edge-case`)
- **Documentation**: `docs/<short-slug>` (e.g., `docs/colab-guide-update`)

### 1 Task → 1 Session → 1 PR Contract
- Keep every unit of work strictly bounded, modular, and verifiable.
- Never bundle unrelated changes (e.g., refactoring network architectures while modifying plot scripts).

---

## 2. GitHub Issue Linkage Standard (Zero Orphan Issues)

Every piece of work corresponds to an open GitHub issue.

1. **Before Starting**: Check existing open issues using GitHub CLI:
   ```bash
   gh issue list --state open
   ```
2. **Pull Request Linking**: Every PR description and commit message MUST include the closing keyword:
   - `Closes #<issue-id>` (or `Fixes #<issue-id>`)
3. **Automatic Closure Verification**:
   - Verify all corresponding issues are closed upon merge.
   - If an issue does not close automatically, immediately comment on the issue with a link to the merged PR and commit SHA, and close it.

---

## 3. Python Code Style & Quality Standards

1. **PEP 8 Compliance**: Code must be clean, readable, and strictly adhere to PEP 8 standards (4-space indentation, snake_case functions/variables, PascalCase classes).
2. **Explicit Type Annotations**: All function signatures, return types, and class attributes must use Python standard `typing`:
   ```python
   def evaluate_agent(
       env_id: str,
       agent: TQCAgent,
       episodes: int = 10,
       seed: Optional[int] = None,
   ) -> Tuple[float, float]:
       ...
   ```
3. **Docstrings**: Public classes and methods must include clear Google-style docstrings detailing arguments, returns, and mathematical context:
   ```python
   """Truncates quantile predictions by discarding the top-d atoms per critic.

   Args:
       quantiles: Tensor of shape (batch_size, num_critics * num_quantiles)
       drop_top: Total number of highest quantiles to truncate across ensemble.

   Returns:
       Truncated quantiles tensor of shape (batch_size, num_critics * num_quantiles - drop_top).
   """
   ```
4. **Deterministic Seeding**: Any script involving stochastic operations must invoke `seed_everything(seed)` from `src/tqc/utils.py` to guarantee 100% reproducibility across PyTorch, NumPy, Python random, and Gymnasium action/observation spaces.

---

## 4. Deterministic Verification Gate

Before staging, committing, or opening a PR, run the local test suite:

```bash
# Full test suite execution (must show 105 passed, 0 failed)
python -m pytest

# Fast smoke run on agent logic
python -m pytest tests/test_train.py -k test_train_tqc_smoke_cpu
```

> [!IMPORTANT]
> A pull request with failing tests or unhandled exceptions will not be reviewed or merged.

---

## 5. Artifact Hygiene & Secrets Management

1. **Never Commit Secrets**:
   - Never commit `kaggle.json`, `.env`, or API tokens.
   - Use `.env.example` as a template and copy it to a local `.env` that is strictly ignored by Git.
2. **Never Commit Large Binary Run Files**:
   - Checkpoint files (`.pt`, `.pth`), compressed archives (`.tar.gz`, `.zip`), raw replay buffers, and generated `.mp4` evaluation videos must NEVER be committed to Git.
   - All run artifacts belong in `runs/` or `videos/`, which are enforced in `.gitignore`.
3. **Disk Quota Governance**:
   - Use `python scripts/clean_runs.py --inspect` to audit disk usage.
   - Use `python scripts/clean_runs.py --prune` to remove non-milestone checkpoints before sharing branches.

---

## 6. Multi-Developer Compute Etiquette

- **Separate Kaggle Accounts**: Each team member must configure their own Kaggle API credentials.
- **Resource Respect**: Kaggle provides 30 GPU hours per week per user and allows at most 2 concurrent GPU jobs. Do not launch overlapping jobs from the same account without using the `DualSlotQueueManager`.
- **Syncing Runs**: When syncing remote runs from Kaggle, run `python scripts/sync_remote_artifacts.py --kernel-slug <slug>` to verify cryptographic SHA256 hashes and validate `metrics.csv`.
