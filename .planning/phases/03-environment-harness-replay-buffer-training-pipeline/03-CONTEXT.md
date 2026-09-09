# Phase 3: Environment Harness, Replay Buffer & Training Pipeline - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the environment, data collection, and training orchestration layer around the core TQC algorithm developed in Phase 2:
1. High-capacity uniform Replay Buffer (`src/tqc/replay_buffer.py`) with pre-allocated NumPy storage up to $10^6$ transitions and efficient PyTorch tensor batch sampling (ENV-01).
2. Standardized Gymnasium MuJoCo Environment Harness (`src/tqc/envs.py`) with support for HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4, and Humanoid-v4, verifying observation/action space dimensions and bounds (ENV-02).
3. Universal seed management (`src/tqc/utils.py`) guaranteeing reproducible executions across PyTorch, NumPy, Python random, and Gymnasium environments (ENV-03).
4. Full Training Pipeline and Loop (`src/tqc/train.py`) executing warm-up exploration, 1-step-per-environment gradient updates, periodic evaluation, and model checkpointing (EXP-01).
5. Structured Metrics Logger (`src/tqc/logger.py`) recording training diagnostics, evaluation returns, losses, temperature ($\alpha$), and quantile distribution statistics (EXP-02).
6. Soft Actor-Critic (SAC) baseline support or compatibility mode for fair empirical benchmarking (EXP-03).
</domain>

<decisions>
## Implementation Decisions

### Replay Buffer Architecture (ENV-01)
- **D-01:** Implement `ReplayBuffer(state_dim, action_dim, capacity=1_000_000, device="cpu")` using pre-allocated contiguous NumPy arrays (`np.empty` or `np.zeros`) for `states`, `actions`, `rewards`, `next_states`, and `dones`.
- **D-02:** Track insertion index with circular pointer and maintain a `size` counter.
- **D-03:** `sample(batch_size)` draws random indices via `np.random.randint` and returns a tuple of PyTorch tensors `(states, actions, rewards, next_states, dones)` formatted with dtype `torch.float32` ready for consumption by `TQCAgent.update()`.

### Gymnasium MuJoCo Environment Harness (ENV-02)
- **D-04:** Implement `make_env(env_id, seed=None)` wrapping Gymnasium MuJoCo environments (`HalfCheetah-v4`, `Hopper-v4`, `Walker2d-v4`, `Ant-v4`, `Humanoid-v4`, or `v5` fallback).
- **D-05:** Handle Gymnasium reset tuple `(obs, info)` and step tuple `(next_obs, reward, terminated, truncated, info)`. In RL training, define `done = terminated` (or `terminated and not truncated` for standard infinite-horizon time limit handling).
- **D-06:** Provide environment metadata helper returning `state_dim`, `action_dim`, and default recommended truncation parameter $d$ ($d=2$ for `Humanoid`, $d=5$ for standard continuous locomotion).

### Reproducibility & Seeding (ENV-03)
- **D-07:** Implement `seed_everything(seed)` in `src/tqc/utils.py` fixing Python `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and setting `torch.backends.cudnn.deterministic = True`.
- **D-08:** Seed Gymnasium environments and action spaces explicitly during environment instantiation and resets.

### Metrics Logger & SAC Baseline Support (EXP-02, EXP-03)
- **D-09:** Implement `MetricsLogger` outputting to both structured CSV and JSONL files with real-time console summaries. Capture: `step`, `episodes`, `eval_return_mean`, `eval_return_std`, `critic_loss`, `actor_loss`, `alpha`, `target_entropy`, and critic quantile prediction summary statistics.
- **D-10:** Provide standard SAC baseline support by configuring TQC with $M=2, N=1, d=0$ (or standalone SAC comparison agent) using the same harness and replay buffer.

### Training Loop Execution (EXP-01)
- **D-11:** Implement `train_tqc(...)` in `src/tqc/train.py` with CLI arguments parsing (`argparse`). Warm up replay buffer with 10,000 initial random exploration steps before gradient updates begin.
- **D-12:** Run deterministic evaluation episodes (10 episodes without exploration noise) every 5,000 / 10,000 steps and save the best agent weights.
</decisions>

<canonical_refs>
## Canonical References

### Project Documentation
- `docs/paper_explanation.md` — Section 4 (Mathematical Formulation) & Section 6 (Hyperparameter Catalog).
- `.planning/REQUIREMENTS.md` — ENV-01 through ENV-03, EXP-01 through EXP-03.
- `.planning/ROADMAP.md` — Phase 3 success criteria and plan definitions.

### Reference Implementations
- `bayesgroup/tqc_pytorch` — Reference training loop, buffer sampling, and evaluation protocol.
- Haarnoja et al. (2018) — Soft Actor-Critic algorithms and applications.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/tqc/actor.py`: Squashed Gaussian policy with stochastic and deterministic evaluation.
- `src/tqc/critic.py`: Quantile critic ensemble ($M=5, N=25$).
- `src/tqc/truncation.py`: Huber quantile loss and top-$d$ truncation operator.
- `src/tqc/agent.py`: `TQCAgent` coordinating gradient descent, target updates, auto-alpha tuning, and checkpointing.

### Established Patterns
- Clean modular design in `src/tqc/`.
- Unit and integration tests under `tests/` with deterministic assertions.
- Support for CPU and CUDA device toggling.

</code_context>

<specifics>
## Specific Ideas

- Test replay buffer sampling speed and tensor conversion efficiency.
- Test Gymnasium environment harness with both deterministic evaluation rollouts and training step transitions.
- Validate that all 5 benchmark environments (`HalfCheetah-v4`, `Hopper-v4`, `Walker2d-v4`, `Ant-v4`, `Humanoid-v4`) instantiate and step without errors.

</specifics>

<deferred>
## Deferred Ideas

- Full multi-million step multi-seed training runs and comparative curve plotting deferred to Phase 4.
- Course presentation slide generation deferred to Issue #3.

</deferred>

---

*Phase: 03-environment-harness-replay-buffer-training-pipeline*
*Context gathered: 2026-09-09*
