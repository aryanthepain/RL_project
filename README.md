# TQC: Truncated Quantile Critics (ICML 2020) Reproduction

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo%20Continuous%20Control-green.svg)](https://gymnasium.farama.org/)
[![Tests](https://img.shields.io/badge/Tests-105%20Passing-brightgreen.svg)](TESTING.md)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An academic and empirical reproduction of the ICML 2020 paper:  
**"Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics"**  
*Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, Dmitry Vetrov (Higher School of Economics & Yandex).*

This repository provides an exact PyTorch reproduction of the TQC algorithm, automated multi-tier compute dispatching (Kaggle Cloud GPU > Google Colab > Local GPU > Local CPU), a progression visualization engine with inactivity truncation, and distributed multi-seed benchmark training infrastructure across standard continuous control MuJoCo tasks.

---

## Table of Contents

1. [Quickstart (60 Seconds)](#1-quickstart-60-seconds)
2. [Complete Developer Setup](#2-complete-developer-setup)
   - [Python Environment](#python-environment)
   - [PyTorch & CUDA Acceleration](#pytorch--cuda-acceleration)
   - [MuJoCo Simulation Harness](#mujoco-simulation-harness)
3. [Multi-Developer Team Collaboration](#3-multi-developer-team-collaboration)
   - [Team Setup & Separate Kaggle Accounts](#team-setup--separate-kaggle-accounts)
   - [Git Branching & PR Invariants](#git-branching--pr-invariants)
4. [Multi-Tier Compute Hierarchy](#4-multi-tier-compute-hierarchy)
5. [Creating & Managing Kaggle Cloud Runs](#5-creating--managing-kaggle-cloud-runs)
6. [Google Colab Setup & Generation](#6-google-colab-setup--generation)
7. [Get Stuff Done (GSD) Workflow Guide](#7-get-stuff-done-gsd-workflow-guide)
   - [GSD Setup & Prerequisites](#gsd-setup--prerequisites)
   - [Core GSD Commands & Lifecycle](#core-gsd-commands--lifecycle)
   - [Step-by-Step Phase Workflow](#step-by-step-phase-workflow)
   - [The GSD Golden Invariants](#the-gsd-golden-invariants)
8. [CLI Reference & Help Sections](#8-cli-reference--help-sections)
9. [Repository Architecture & File Map](#9-repository-architecture--file-map)
10. [Benchmark Roadmap & Phase Tracking](#10-benchmark-roadmap--phase-tracking)

---

## 1. Quickstart (60 Seconds)

Clone the repository and run the pre-flight CPU smoke test in three simple commands:

```bash
# 1. Clone the repository
git clone https://github.com/aryanthepain/RL_project.git
cd RL_project

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies & run deterministic test gate
pip install -r requirements.txt
python -m pytest
```

When you see `105 passed`, your environment is fully operational!

---

## 2. Complete Developer Setup

### Python Environment
We recommend **Python 3.10, 3.11, or 3.12**.

```bash
# Check your Python version
python --version
```

### PyTorch & CUDA Acceleration

#### For NVIDIA GPU Users (Windows / Linux)
To enable CUDA 12.x GPU acceleration, install the official CUDA-enabled PyTorch wheel:

```bash
# CUDA 12.1 wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU detection
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

#### For CPU / Mac / WSL2 Users
The standard `pip install -r requirements.txt` installs standard PyTorch. The entire training pipeline automatically falls back to CPU if no CUDA device is detected.

### MuJoCo Simulation Harness
Gymnasium 0.29+ uses the official `mujoco>=3.0.0` Python bindings. No external C++ binaries or license keys are required.

To verify your MuJoCo physics engine installation:
```bash
python -c "import gymnasium as gym; env = gym.make('HalfCheetah-v4'); print('MuJoCo initialized successfully:', env.reset()[0].shape)"
```

---

## 3. Multi-Developer Team Collaboration

Our development team consists of 3 developers collaborating on distributed training and evaluation. To prevent collisions, secrets leaks, and merge conflicts, follow these conventions:

### Team Setup & Separate Kaggle Accounts
Every developer on the team should maintain their own Kaggle account and API token.

> [!TIP]
> **Resource Scaling**: Kaggle provides 30 GPU hours/week and up to 2 concurrent GPU jobs per user account. With 3 developers using separate accounts, the team has access to **6 concurrent GPU slots** and **90 hours/week of free cloud compute**, allowing rapid completion of multi-seed benchmark training!

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings).
2. Scroll to **API** and click **Create New Token** to download `kaggle.json`.
3. Configure your credentials locally using either:
   - **Method A (Recommended)**: Copy `.env.example` to `.env` in the project root:
     ```bash
     cp .env.example .env
     ```
     Fill in your credentials:
     ```ini
     KAGGLE_USERNAME=your_kaggle_username
     KAGGLE_KEY=your_kaggle_api_key
     ```
   - **Method B**: Place `kaggle.json` in `~/.kaggle/kaggle.json` (`C:\Users\<username>\.kaggle\kaggle.json` on Windows).
4. Verify your credentials using the built-in diagnostic probe:
   ```bash
   python -c "from src.tqc.remote.kaggle_auth import resolve_kaggle_credentials; print(resolve_kaggle_credentials())"
   ```

### Git Branching & PR Invariants

- **Default Branch (`main`) is Protected**: Never commit directly to `main`.
- **Feature & Phase Branches**:
  - For benchmark trainings: `gsd/phase-09-hopper-runs`
  - For new features: `feat/<name>`
  - For bug fixes: `fix/<name>`
- **Link Every PR to a GitHub Issue**:
  Include `Closes #<issue_number>` in the title and description so GitHub automatically establishes the closure link.
- **Never Commit Large Artifacts**:
  Model checkpoints (`.pt`), video files (`.mp4`), and compressed archives (`.tar.gz`) are strictly ignored via `.gitignore`. Always sync large runs using `scripts/sync_remote_artifacts.py`.

---

## 4. Multi-Tier Compute Hierarchy

The system incorporates an automated 4-tier compute resolver that dynamically chooses the best hardware tier available:

```
Tier 1: Kaggle Remote GPU  ──► Tesla P100 / Dual T4 (Cloud, High Throughput)
Tier 2: Google Colab GPU   ──► Tesla T4 (Cloud Notebook, Interactive)
Tier 3: Local NVIDIA GPU   ──► GeForce RTX / Workstation GPU (CUDA)
Tier 4: Local CPU          ──► Multi-threaded Host Fallback (Smoke Testing)
```

To probe and inspect available compute tiers on your machine:
```bash
python -m src.tqc.compute.dispatcher
```

Output preview:
```text
================================================================================
TQC Compute Hierarchy Probe Summary
================================================================================
Tier 1: KAGGLE     | AVAILABLE | Kaggle API Authenticated (2 GPU slots)
Tier 2: COLAB      | UNAVAILABLE | Run from Google Colab environment
Tier 3: LOCAL_GPU  | UNAVAILABLE | PyTorch built without CUDA / no device
Tier 4: LOCAL_CPU  | AVAILABLE | 16 Logical Cores (Intel/AMD)
================================================================================
Selected Default Tier: KAGGLE
```

---

## 5. Creating & Managing Kaggle Cloud Runs

The Kaggle remote execution pipeline automatically packages code, performs pre-flight smoke tests, submits detached GPU kernels, enforces 2-slot concurrency, and pulls back checkpoints.

### 1. Launching a Pilot Run (HalfCheetah-v4, 1M Steps)
```bash
python scripts/run_remote_experiment.py --preset pilot
```

### 2. Launching Runs for Specific Environments & Seeds
```bash
# Run Hopper-v4 on seed 42 for 1,000,000 steps
python scripts/run_remote_experiment.py --env Hopper-v4 --seed 42 --steps 1000000

# Run Ant-v4 detached in the background
python scripts/run_remote_experiment.py --env Ant-v4 --seed 43 --steps 1000000 --detach
```

### 3. Pre-Flight Dry Runs
Verify packaging and kernel metadata without submitting to the cloud:
```bash
python scripts/run_remote_experiment.py --preset pilot --dry-run
```

### 4. Syncing Remote Results & Validating Integrity
Once a Kaggle kernel completes, pull the artifacts back to your local `runs/` directory:
```bash
python scripts/sync_remote_artifacts.py --kernel-slug <your-username>/kaggle-halfcheetah-pilot-s42
```
The synchronizer performs cryptographic SHA256 validation on all weights, validates the `metrics.csv` column schema, and creates downstream video rendering triggers.

### 5. Managing Disk Quota
Auditing and pruning checkpoint directories:
```bash
# Inspect all runs and disk consumption
python scripts/clean_runs.py --inspect

# Prune intermediate checkpoints (keeps step 0, final, and milestone checkpoints)
python scripts/clean_runs.py --prune
```

---

## 6. Google Colab Setup & Generation

We provide an automated generator to produce a standalone, reproducible Google Colab notebook configured with headless MuJoCo, Google Drive checkpoint syncing, and in-notebook video playback.

### Generating the Colab Notebook
```bash
python scripts/build_colab_notebook.py
```
This generates `notebooks/colab_tqc_benchmark.ipynb`.

### Running in Colab
1. Upload `notebooks/colab_tqc_benchmark.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Set runtime type to **GPU** (`Runtime > Change runtime type > T4 GPU`).
3. Run all cells:
   - Cell 1: Clones this repository and installs MuJoCo dependencies.
   - Cell 2: Runs automated pre-flight tier detection.
   - Cell 3: Executes benchmark training (`HalfCheetah-v4`, `Hopper-v4`, etc.).
   - Cell 4: Renders evaluation videos and renders them inline using HTML5 video tags.
   - Cell 5: Syncs checkpoints and metrics directly to your Google Drive.

---

## 7. Get Stuff Done (GSD) Workflow Guide

This project is structured and executed using the **GSD (Get Stuff Done)** autonomous execution framework. GSD provides persistent context across sessions in `.planning/`, enforces rigorous planning gates, and guarantees deterministic verification before shipping code.

### GSD Setup & Prerequisites

To set up and run GSD workflows with your AI agent, configure the following prerequisites:

#### 1. AI Agent Environment (Antigravity IDE or Claude Code)
GSD operates through specialized agent skills and rules:
- **Global Customizations**: Skills are automatically discovered and loaded from `~/.gemini/config/skills/` (or `.agents/skills/` in the project root).
- **Global Rules & Invariants**: Loaded from `~/.gemini/config/rules/global_rules.md`, enforcing lean orchestrator context, Ponytail anti-bloat ladder, and Karpathy surgical disciplines.

#### 2. GitHub CLI (`gh`) Authentication
GSD automates open issue discovery, PR title/body formatting, and issue-to-PR closure linkage.
```bash
# Authenticate GitHub CLI
gh auth login

# Verify credentials and view open project issues
gh issue list --state open
```

#### 3. Node.js Runtime (v18+)
GSD utilizes background worker subagents (`node ./scripts/subagent.js` or global equivalent) to offload multi-file codebase research, isolated code synthesis, and independent adversarial code reviews without polluting the primary orchestrator's context window:
```bash
# Verify Node.js is installed
node --version
```

#### 4. PowerShell (`pwsh`)
PowerShell is used by GSD to execute deterministic local test gates and trigger spoken completion alerts via `agent-alarm.ps1`:
```bash
# Verify PowerShell 7+
pwsh -Command "Write-Host 'PowerShell operational'"
```

#### 5. Project State Synchronization
Because all GSD planning artifacts (`.planning/PROJECT.md`, `ROADMAP.md`, `REQUIREMENTS.md`, `STATE.md`, and `config.json`) are committed to Git, no external database or server is required:
- When you clone this repository, GSD automatically discovers the existing phases, requirements, and progress state.
- **Audit GSD Health**: Run `/gsd-health` in your agent prompt to diagnose `.planning/` file integrity.
- **Inspect Current Position**: Run `/gsd-progress` to display the active phase, completed plans, and next planned action.

---

### Core GSD Commands & Lifecycle

When collaborating with an Antigravity AI pair programmer, interact using the following slash commands:

| Command | Purpose | When to Use |
|---|---|---|
| **`/gsd-progress`** | Situational situational report | Check current milestone, active phase, completed plans, and next actions. |
| **`/gsd-discuss-phase <N>`** | Phase kickoff & Socratic grilling | Gather requirements, surface trade-offs, and align before planning. |
| **`/gsd-plan-phase <N>`** | Create atomic phase plan (`PLAN.md`) | Break down phase into atomic, testable execution plans with dependency waves. |
| **`/gsd-execute-phase <N>`** | Autonomous plan execution | Execute plans with automated test gates and atomic git commits. |
| **`/gsd-verify-work`** | Interactive UAT | Verify deliverables meet paper fidelity and acceptance criteria. |
| **`/gsd-ship`** | PR preparation & automated closure | Run test gates, discover open GitHub issues, and open PR with `Closes #...`. |
| **`/gsd-quick "<task>"`** | Fast ad-hoc task delivery | Execute small bugfixes, documentation updates, or minor tweaks. |
| **`/gsd-debug "<problem>"`** | Systematic root-cause debugging | Persistent debugging workflow across context resets. |

---

### Step-by-Step Phase Workflow

Here is the standard lifecycle for executing any planned phase (e.g., Phase 09 Hopper-v4):

```bash
# Step 1: Check project status and confirm active phase
/gsd-progress

# Step 2: Discuss phase requirements and resolve ambiguities
/gsd-discuss-phase 9

# Step 3: Author detailed execution plans (09-01-PLAN.md, etc.)
/gsd-plan-phase 9

# Step 4: Execute all plans on dedicated branch (gsd/phase-09-hopper-runs)
/gsd-execute-phase 9

# Step 5: Validate deliverables against success criteria
/gsd-verify-work

# Step 6: Verify 105 tests pass and ship PR linking Issue #13
/gsd-ship
```

---

### The GSD Golden Invariants

> [!IMPORTANT]
> 1. **No Unplanned Commits**: Never edit codebase files outside of a planned GSD phase or `/gsd-quick` task. This ensures the `.planning/` roadmap and Git history remain 100% in sync across all team members.
> 2. **Protected `main` Branch**: Never push directly to `main`. Always create dedicated feature/phase branches (`gsd/phase-<N>-<slug>` or `feat/<name>`).
> 3. **Zero Orphan Issues**: Every PR must query open issues via `gh issue list --state open` and link the corresponding issue with `Closes #<issue>` in the title, body, and commit.
> 4. **Deterministic Verification**: Every change must pass `python -m pytest tests/` before staging or merging.

---

## 8. CLI Reference & Help Sections

All tools include detailed help documentation via the `--help` flag:

### Main Training Engine
```bash
python -m src.tqc.train --help
```
Key arguments:
- `--env`: Gymnasium environment ID (`HalfCheetah-v4`, `Ant-v4`, etc.).
- `--seed`: Random seed for reproducibility (default: `0`).
- `--total-timesteps`: Total environment steps (default: `1000000`).
- `--eval-freq`: Evaluation interval in steps (default: `5000`).
- `--checkpoint-freq`: Checkpoint frequency in steps (default: `50000`).
- `--device`: Target device (`cuda`, `cpu`).
- `--resume`: Resume from existing checkpoint if available.

### Remote Kaggle Orchestrator
```bash
python scripts/run_remote_experiment.py --help
```
Key arguments:
- `--preset`: Named preset from `configs/benchmark_matrix.yaml` (`pilot`, `smoke`, `benchmark_suite`).
- `--env`: Environment ID override.
- `--seed`: Seed override.
- `--steps`: Timesteps override.
- `--dry-run`: Validate packaging without submitting.
- `--detach`: Return immediately after kernel push.

### Progression Video & Inactivity Truncation
```bash
python -m src.tqc.visualize --help
```
Key arguments:
- `--checkpoint-dir`: Path to saved checkpoint directory.
- `--truncate-inactive`: Automatically truncate frames when the agent falls or stagnates.
- `--speed-threshold`: Minimum forward speed threshold (default: `0.05`).
- `--patience-steps`: Consecutive inactive steps before truncating (default: `50`).
- `--padding-frames`: Trailing frames retained after truncation (default: `15`).

---

## 9. Repository Architecture & File Map

```text
RL_project/
├── .github/                       # GitHub collaboration & PR templates
│   └── pull_request_template.md
├── configs/                       # Declarative experiment matrix
│   └── benchmark_matrix.yaml
├── docs/                          # In-depth technical papers and guides
│   ├── paper_explanation.md       # Comprehensive mathematical theory
│   ├── remote_execution.md        # Kaggle execution deep dive
│   └── progression_visualization.md
├── notebooks/                     # Colab notebooks
│   └── colab_tqc_benchmark.ipynb
├── scripts/                       # Orchestration & utility tools
│   ├── build_colab_notebook.py    # Generates runnable Colab notebook
│   ├── clean_runs.py              # Audits and prunes disk quotas
│   ├── run_remote_experiment.py   # Packages and pushes Kaggle jobs
│   └── sync_remote_artifacts.py   # Pulls and validates remote weights
├── src/tqc/                       # Core TQC algorithm library
│   ├── actor.py                   # Squashed Gaussian policy network
│   ├── critic.py                  # Quantile critic ensemble (M=5, N=25)
│   ├── truncation.py              # Top-d truncation operator & Huber loss
│   ├── buffer.py                  # Circular 1M transition replay buffer
│   ├── agent.py                   # TQCAgent training & updates
│   ├── train.py                   # Main training loop
│   ├── visualize.py               # Video rendering & stagnation detection
│   ├── compute/                   # Multi-tier hardware dispatcher
│   └── remote/                    # Kaggle authentication & packaging
├── tests/                         # 25 test suites (105 tests passing)
├── ARCHITECTURE.md                # System design & mathematical mapping
├── CONVENTIONS.md                 # Git branching, PRs, & coding standards
├── STACK.md                       # Comprehensive tech stack catalog
├── TESTING.md                     # Verification handbook & test taxonomy
└── requirements.txt               # Locked reproducible dependencies
```

---

## 10. Benchmark Roadmap & Phase Tracking

The reproduction is structured across distinct roadmap phases tracked in `.planning/ROADMAP.md`:

| Phase | Description | Status | GitHub Issue |
|:---:|:---|:---:|:---:|
| **01** | Theoretical & Algorithmic Documentation | Completed | [#1](https://github.com/aryanthepain/RL_project/issues/1) |
| **02** | Core Algorithm & Quantile Networks | Completed | [#2](https://github.com/aryanthepain/RL_project/issues/2) |
| **03** | Environment Harness & Replay Buffer | Completed | [#2](https://github.com/aryanthepain/RL_project/issues/2) |
| **04** | Benchmark Experiments & Replication Curves | Completed | [#2](https://github.com/aryanthepain/RL_project/issues/2) |
| **05** | Policy Progression Visualization | Completed | [#5](https://github.com/aryanthepain/RL_project/issues/5), [#6](https://github.com/aryanthepain/RL_project/issues/6) |
| **06** | Compute Hierarchy & Kaggle Remote Execution | Completed | [#7](https://github.com/aryanthepain/RL_project/issues/7) |
| **07** | Evaluation Video Inactivity Truncation | Completed | [#8](https://github.com/aryanthepain/RL_project/issues/8) |
| **08** | Multi-Tier Compute Hierarchy & Colab | Completed | [#4](https://github.com/aryanthepain/RL_project/issues/4) |
| **Foundation** | Team Onboarding, Documentation Suite & Collaboration | Completed | [#12](https://github.com/aryanthepain/RL_project/issues/12) |
| **09** | Benchmark Training: Hopper-v4 (5 seeds) | Upcoming | [#13](https://github.com/aryanthepain/RL_project/issues/13) |
| **10** | Benchmark Training: Walker2d-v4 (5 seeds) | Upcoming | [#14](https://github.com/aryanthepain/RL_project/issues/14) |
| **11** | Benchmark Training: Ant-v4 (5 seeds) | Upcoming | [#17](https://github.com/aryanthepain/RL_project/issues/17) |
| **12** | Benchmark Training: Humanoid-v4 (5 seeds) | Upcoming | [#15](https://github.com/aryanthepain/RL_project/issues/15) |
| **13** | Cross-Benchmark Suite Aggregation & Replication Curves | Upcoming | [#18](https://github.com/aryanthepain/RL_project/issues/18) |

---

## References

- Arsenii Kuznetsov, Pavel Shvechikov, Alexander Grishin, Dmitry Vetrov. *"Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics"*. Proceedings of the 37th International Conference on Machine Learning (ICML 2020). [arXiv:2005.04269](https://arxiv.org/abs/2005.04269).
- Official TQC Reference Implementation: [bayesgroup/tqc_pytorch](https://github.com/bayesgroup/tqc_pytorch).
