# Phase 2: Core Algorithm & Quantile Networks - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Implementation of the complete, modular PyTorch TQC algorithm following the ICML 2020 paper and `docs/paper_explanation.md`:
1. Squashed Gaussian Actor network (`src/tqc/actor.py`) with reparameterization trick and analytical tanh squashing log-prob adjustment (ALGO-01).
2. Quantile Critic Ensemble (`src/tqc/critic.py`) with $M=5$ independent 3-layer MLPs predicting $N=25$ quantiles each (ALGO-02).
3. Quantile Huber loss and top-$d$ Truncation Operator (`src/tqc/truncation.py`) (ALGO-03, ALGO-04).
4. Full TQC Agent (`src/tqc/agent.py`) coordinating policy updates, critic updates, Polyak target smoothing ($\tau=0.005$), and automatic entropy temperature tuning ($\alpha$) (ALGO-05).
5. Comprehensive unit test suite (`tests/`) verifying shapes, gradients, loss computation, truncation correctness, and numerical stability.
</domain>

<decisions>
## Implementation Decisions

### Package Architecture
- **D-01:** Place core algorithm code in `src/tqc/` with clean modular separation: `actor.py`, `critic.py`, `truncation.py`, and `agent.py`.
- **D-02:** Strict adherence to paper hyperparameters: 3-layer 512-512-512 MLPs with ReLU activations for both actor and critic networks.

### Actor Design (ALGO-01)
- **D-03:** The Actor takes observation tensor $s$, passes through hidden layers, and outputs $\mu(s)$ and $\log \sigma(s)$ clamped to $[\text{LOG\_STD\_MIN}=-20, \text{LOG\_STD\_MAX}=2]$.
- **D-04:** Reparameterization sampling: $\tilde{a} = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, squashed via $a = \tanh(\tilde{a})$.
- **D-05:** Compute log-probability correcting for the Jacobian: $\log \pi(a | s) = \log \mu(\tilde{a}|s) - \sum \log(1 - \tanh^2(\tilde{a}) + 10^{-6})$. Provide both stochastic sampling and deterministic evaluation mode ($a = \tanh(\mu)$).

### Critic Ensemble & Quantile Representation (ALGO-02)
- **D-06:** Construct an ensemble of $M=5$ independent critic networks (`CriticEnsemble`), each outputting $N=25$ quantiles for input $(s, a)$.
- **D-07:** Enable parallel forward execution across ensemble members using PyTorch batched operations or list of modules for numerical clarity and performance.

### Quantile Huber Loss & Truncation (ALGO-03, ALGO-04)
- **D-08:** Implement `quantile_huber_loss(predictions, targets, tau, kappa=1.0)` following the exact asymmetric weighting formula.
- **D-09:** Implement `truncate_quantiles(target_quantiles, drop_top=d)`: pool all $M \times N = 125$ quantiles, sort in ascending order, and drop the top $d$ atoms, returning $k = 125 - d$ retained atoms.

### TQC Agent & Auto-Alpha (ALGO-05)
- **D-10:** Maintain target critic ensemble $\bar{\psi}$ updated via Polyak averaging: $\bar{\psi} \leftarrow \tau \psi + (1 - \tau) \bar{\psi}$ with $\tau = 0.005$.
- **D-11:** Automatically tune temperature $\alpha$ by optimizing $\log \alpha$ with Adam optimizer against target entropy $\bar{\mathcal{H}} = -\dim(\mathcal{A})$.
- **D-12:** Expose clean API: `select_action(state, deterministic=False)`, `update(replay_buffer, batch_size=256)`, and state dict serialization (`save/load`).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Documentation
- `docs/paper_explanation.md` — Section 4 (Mathematical Formulation) & Section 6 (Architecture Catalog).
- `.planning/REQUIREMENTS.md` — ALGO-01 through ALGO-05 specifications.
- `.planning/ROADMAP.md` — Phase 2 success criteria.

### External Specifications
- `https://github.com/bayesgroup/tqc_pytorch` — Official reference implementation of TQC.
- Kuznetsov et al. (ICML 2020) — PMLR 119:5556-5566.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None yet (greenfield).

### Established Patterns
- Modular, well-typed Python with docstrings and type annotations.
- Deterministic PyTorch unit tests with fixed random seeds verifying tensor shapes, gradients, and loss computations.

</code_context>

<specifics>
## Specific Ideas

- Ensure modularity so Phase 3 environment harness and training loop can import `from tqc import TQCAgent` directly.
- Include unit tests verifying gradient flow: critic gradients flow to critics only, actor gradients flow through the critic mean to actor only.

</specifics>

<deferred>
## Deferred Ideas

- Replay buffer and Gym environment integration deferred to Phase 3.
- Benchmark training runs deferred to Phase 4.

</deferred>

---

*Phase: 02-core-algorithm-quantile-networks*
*Context gathered: 2026-09-09*
