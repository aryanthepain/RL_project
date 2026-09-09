# Phase 1: Paper Theoretical & Algorithmic Documentation - Context

**Gathered:** 2026-09-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Production of an authoritative, pedagogical, and conceptually deep markdown document (`docs/paper_explanation.md`) analyzing the ICML 2020 paper "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics" (TQC) by Kuznetsov et al. The document serves as the primary theoretical knowledge artifact for team understanding, study, and presentation prep, covering DOC-01 through DOC-04.
</domain>

<decisions>
## Implementation Decisions

### Tone & Pedagogical Style
- **D-01:** Focus on conceptual clarity, intuition, and relatable analogies for human understanding rather than dense proofs. Skip rigorous lemmas/theorems per course allowance.
- **D-02:** Explain the mechanics of why standard Q-learning overestimates, why SAC/TD3 clipped double Q-learning under- or overestimates depending on environment, and how quantile representations provide the full distribution of returns.

### Visual Diagrams & Formats
- **D-03:** Include Mermaid and ASCII flowcharts/diagrams to visualize:
  1. The Actor-Critic interaction in TQC.
  2. The quantile prediction array ($M=5$ critics $\times N=25$ quantiles).
  3. The Truncation operator (pooling 125 quantiles, sorting in ascending order, dropping top $d$ atoms, and computing the target).
- **D-04:** Do NOT include source code or pseudocode blocks in the explanation document itself. Instead, provide clean step-by-step algorithmic workflows and mathematical logic in prose and equations.

### Section Structure (7 Core Sections)
- **D-05:** Structure `docs/paper_explanation.md` into 7 distinct sections:
  1. Executive Summary & Problem Motivation (Overestimation bias in continuous RL).
  2. Theoretical Background (SAC, TD3, QR-DQN, and distributional RL foundations).
  3. TQC Core Innovation (Distributional critic ensemble, quantile representations, truncation mechanism).
  4. Mathematical Formulation (Distributional Bellman operator, Huber quantile regression loss, top-$d$ truncation target calculation, entropy temperature $\alpha$ dual optimization).
  5. Algorithmic Workflow & Logic (Step-by-step operational loop without code).
  6. Network Architectures & Hyperparameters (Explicit catalog matching `bayesgroup/tqc_pytorch`: MLP layer dimensions, Adam learning rates, replay buffer size, batch size, target update $\tau$, truncation parameter $d$ per environment).
  7. Empirical Performance & MuJoCo Benchmark Analysis (Findings on HalfCheetah, Hopper, Walker2d, Ant, Humanoid).

### Agent's Discretion
- Choice of specific analogies (e.g. explaining quantile truncation as discarding outlier optimistic lottery tickets).
- Formatting of mathematical equations using standard LaTeX syntax.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Paper & Research
- `http://proceedings.mlr.press/v119/kuznetsov20a/kuznetsov20a.pdf` — "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics", ICML 2020.
- `https://github.com/bayesgroup/tqc_pytorch` — Official PyTorch implementation and hyperparameter specification.
- `https://research.samsung.com/blog/Fixing-overestimation-bias-in-continuous-reinforcement-learning` — Samsung Research blog on fixing overestimation bias.

### Project Context
- `.planning/PROJECT.md` — Core value, project boundaries, and constraints.
- `.planning/REQUIREMENTS.md` — DOC-01 through DOC-04 specifications.
- `.planning/ROADMAP.md` — Phase 1 success criteria.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None yet (greenfield repository).

### Established Patterns
- Clean Markdown with GitHub-flavored LaTeX math blocks (`$$ ... $$` and `$ ... $`) and Mermaid diagrams (` ```mermaid ... ``` `).

### Integration Points
- `docs/paper_explanation.md` will be directly referenced in Phase 2 for network architectural design and loss implementation.

</code_context>

<specifics>
## Specific Ideas

- Emphasize the intuitive difference between point-estimate value clipping (e.g. $\min(Q_1, Q_2)$ in SAC/TD3, which can be overly pessimistic or still overestimate) vs distributional quantile truncation (which provides fine-grained, smooth control over estimation bias by varying $d$).
- No code or pseudocode in the document.

</specifics>

<deferred>
## Deferred Ideas

- Slide deck and presentation materials deferred to Issue #3 (to be generated after empirical reproduction in Phase 4).

</deferred>

---

*Phase: 01-paper-theoretical-algorithmic-documentation*
*Context gathered: 2026-09-09*
