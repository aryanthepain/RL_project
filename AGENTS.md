<!-- GSD:project-start source:PROJECT.md -->

## Project

**TQC Paper Reproduction & Analysis (ICML 2020)**

An academic and empirical reproduction of the ICML 2020 paper "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics" (TQC) by Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, and Dmitry Vetrov. The project delivers a comprehensive mathematical and conceptual explanation document, an exact PyTorch-based reproduction of the TQC algorithm following the paper's specification, and experimental evaluation on standard continuous control benchmarks across MuJoCo environments.

**Core Value:** Faithful, reproducible implementation and rigorous empirical evaluation of Truncated Quantile Critics against standard baselines (like Soft Actor-Critic) on MuJoCo continuous control benchmarks, backed by a comprehensive theoretical explanation document.

### Constraints

- **Tech Stack**: Python 3.10+, PyTorch, Gymnasium/Gym, MuJoCo, NumPy, Matplotlib/Seaborn.
- **Algorithm Fidelity**: Must adhere to the exact architecture, hyperparameters, and operators described in the Kuznetsov et al. paper.
- **Timeline**: Must complete reproduction and documentation before the course endsem presentation deadline.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->

## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

- **Issue & PR Linkage**: Every PR must query open issues via `gh issue list --state open` and link relevant issues using `Closes #<issue>` / `Fixes #<issue>` in the title and description.
- **Issue Closure**: Verify all corresponding issues are closed upon merge. If an issue was not closed automatically, immediately comment and close it with a comprehensive completion summary referencing the merged PR and commit SHA.
- **Phase & Feature Branching**: Never commit phase or feature work directly to the default branch (`main`). At the start of every new phase, create a dedicated feature/phase branch (following `gsd/phase-<N>-<slug>` or prompt the user) so that all work is cleanly isolated until PR merge.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.agents/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
