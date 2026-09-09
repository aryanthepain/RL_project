# Remote Compute Execution & Experiment Orchestration Guide

**Project:** TQC Paper Reproduction & Analysis (ICML 2020)  
**Author:** AI Pair Programmer & Research Team  
**Scope:** GitHub Issue #4 (Compute Hierarchy) & GitHub Issue #7 (HalfCheetah-v4 Pilot)

---

## 1. Compute Hierarchy Architecture (D-01, D-03)

Training continuous control reinforcement learning agents (such as Truncated Quantile Critics) over 1,000,000 to 3,000,000 timesteps across multiple random seeds requires substantial compute. To balance availability, throughput, and hardware constraints, this project implements a 4-tier compute hierarchy:

```
+-------------------------------------------------------------------------+
|                  Tier-1: Kaggle Headless Kernels (Default)               |
|  - Free 30 hours/week GPU quota (NVIDIA Tesla T4 / P100)                |
|  - Headless API orchestration via Kaggle CLI & kernel-metadata.json      |
|  - Dual-slot queue manager (max 2 concurrent GPU jobs)                  |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  Tier-2: Google Colab GPU Export                         |
|  - Jupyter Notebook runner for interactive GPU exploration               |
|  - Google Drive artifact persistence and manual checkpoint exports      |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  Tier-3: Local NVIDIA CUDA GPU                           |
|  - Consumer GPU local training with torch.backends.cudnn.benchmark      |
|  - Immediate feedback loop without cloud upload latency                 |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  Tier-4: Local CPU Fallback                              |
|  - Lightweight verification runs, CI/CD regression tests                |
|  - Automated 100-step pre-flight smoke validation (D-13)                |
+-------------------------------------------------------------------------+
```

---

## 2. Authentication & Credential Resolution (D-16)

The framework features a dual-source credential resolver with automated interactive prompting fallback:

### Method A: Kaggle API Token File (Standard)
1. Go to your [Kaggle Account Settings](https://www.kaggle.com/settings).
2. Click **Create New Token**. A file named `kaggle.json` will be downloaded.
3. Place `kaggle.json` in:
   - **Linux / macOS:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<Username>\.kaggle\kaggle.json`
4. Set permissions to owner-only:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Method B: Environment Variables / `.env`
Alternatively, set the credentials directly in your environment or in a root `.env` file:
```bash
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key_token
```

### Method C: Automated Interactive Prompt
If credentials cannot be located during execution, `scripts/run_remote_experiment.py` interactively prompts you to enter your username and API key, automatically creating and securing `~/.kaggle/kaggle.json`.

---

## 3. Pre-Flight Smoke Validation Gate (D-13, D-17)

Before pushing any heavy 1M-step workload to Kaggle GPU instances, an automated **100-step local dry run** executes on CPU:
- Verifies MuJoCo physics engine loading and environment initialization.
- Validates actor-critic tensor forward and backward passes.
- Confirms replay buffer sample extraction and Bellman quantile target contraction.
- Validates checkpoint serialization (`checkpoint_*.pt`, `best_model.pt`) and `metrics.csv` generation.

If this pre-flight test fails, the cloud dispatch is aborted immediately to avoid consuming weekly GPU quota on broken jobs.

---

## 4. HalfCheetah-v4 Pilot Workflow (GitHub Issue #7)

To validate the end-to-end cloud pipeline before scaling to the full 5-environment MuJoCo benchmark suite, we execute a pilot run on `HalfCheetah-v4` (1,000,000 steps, seed 42).

### 4.1 Running the Pilot (Blocking Mode)
Runs local pre-flight smoke test, packages code into `tqc_source.tar.gz`, pushes kernel to Kaggle, monitors execution progress, and automatically downloads and verifies artifacts upon completion:
```powershell
python scripts/run_remote_experiment.py --preset pilot
```

### 4.2 Running the Pilot in Detached Mode (Background)
For long-running 1M-step training, use `--detach` to submit the kernel and immediately release your local terminal:
```powershell
python scripts/run_remote_experiment.py --preset pilot --detach
```
The runner will print the deployed Kernel ID (e.g., `username/tqc-halfcheetahv4-s42-09101200`).

### 4.3 Monitoring Kernel Status
Check live execution status at any time:
```powershell
kaggle kernels status <username>/<kernel-slug>
```
Possible statuses: `queued`, `running`, `complete`, `error`.

### 4.4 Synchronizing Artifacts Manually
Once the kernel status reaches `complete`:
```powershell
python scripts/sync_remote_artifacts.py --kernel-id <username>/<kernel-slug>
```
This command:
1. Downloads remote outputs via `kaggle kernels output`.
2. Unpacks `artifacts.tar.gz`.
3. Validates SHA256 hashes against `checksums.sha256` (D-07).
4. Validates schema and record count in `metrics.csv` (D-08).
5. Outputs instant visualization commands for plotting and video generation.

---

## 5. Dual-Slot Queue Manager (D-11)

Kaggle limits users to a maximum of **2 concurrent GPU kernels**. To run multi-seed or multi-environment experiment batches:

```python
from src.tqc.remote.queue_manager import DualSlotQueueManager

queue_mgr = DualSlotQueueManager(max_concurrent_slots=2)

# Enqueue multiple jobs
queue_mgr.add_job("hc_s42", "build_s42", "HalfCheetah-v4", 42)
queue_mgr.add_job("hc_s43", "build_s43", "HalfCheetah-v4", 43)
queue_mgr.add_job("hc_s44", "build_s44", "HalfCheetah-v4", 44)

# Automatically dispatches up to 2 jobs, queuing the 3rd until a slot opens:
queue_mgr.monitor_until_completion(poll_interval=30)
```

---

## 6. Declarative Configuration: `configs/benchmark_matrix.yaml` (D-09)

The benchmark matrix configuration defines reproducible presets and hyperparameter profiles:

```yaml
default_platform: kaggle
max_concurrent_gpu_slots: 2

presets:
  pilot:
    env_id: "HalfCheetah-v4"
    seed: 42
    total_timesteps: 1000000
    checkpoint_freq: 50000
    eval_freq: 5000
    batch_size: 256
    buffer_capacity: 1000000
    device: "cuda"

  smoke:
    env_id: "HalfCheetah-v4"
    seed: 42
    total_timesteps: 100
    checkpoint_freq: 50
    eval_freq: 50
    device: "cpu"

  benchmark_suite:
    environments:
      - name: "HalfCheetah-v4"
        steps: 1000000
      - name: "Hopper-v4"
        steps: 1000000
      - name: "Walker2d-v4"
        steps: 1000000
      - name: "Ant-v4"
        steps: 1000000
      - name: "Humanoid-v4"
        steps: 3000000
    seeds: [42, 43, 44, 45, 46]
```

To run with CLI overrides:
```powershell
python scripts/run_remote_experiment.py --env Hopper-v4 --seed 101 --steps 1000000 --device cuda
```

---

## 7. Downstream Visualization Integration (D-08)

Once artifacts are synchronized into `runs/`:

### 1. Comparative Learning Curves
```powershell
python -m src.tqc.plotter --log-dirs "runs/kaggle_halfcheetah_s42_20260910"
```

### 2. Progression Videos & Synchronized Grid Montages
```powershell
python scripts/run_halfcheetah_progression.py --skip-train --checkpoint-dir "runs/kaggle_halfcheetah_s42_20260910"
```

Generates:
- `videos/checkpoints/ckpt_*.mp4`: Individual milestone rollouts.
- `videos/halfcheetah_6way_grid.mp4`: Synchronized 2x3 policy progression grid.
- `videos/halfcheetah_progression_montage.mp4`: Chronological gait evolution montage.
- `results/halfcheetah_progression_diagnostics.png`: Multi-panel progression analytics.
