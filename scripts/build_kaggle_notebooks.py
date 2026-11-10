"""Generator script for Kaggle reproduction notebooks:
- notebooks/kaggle_tqc_hopper_phase09.ipynb (Phase 9: Hopper-v4)
- notebooks/kaggle_tqc_ant_phase11.ipynb (Phase 11: Ant-v4)
"""

import base64
import io
import json
import os
import tarfile
from pathlib import Path


def _get_b64_source_payload() -> str:
    """Bundle src/, scripts/, and configs/ into a base64-encoded tar.gz string."""
    repo_root = Path(__file__).resolve().parent.parent
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            tar.add(src_dir, arcname="src")
        scripts_dir = repo_root / "scripts"
        if scripts_dir.is_dir():
            tar.add(scripts_dir, arcname="scripts")
        configs_dir = repo_root / "configs"
        if configs_dir.is_dir():
            tar.add(configs_dir, arcname="configs")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_kaggle_benchmark_notebook(
    env_id: str,
    phase_num: str,
    target_score: str,
    output_path: str,
) -> None:
    """Generate a self-contained Kaggle notebook tailored for MuJoCo TQC benchmarks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_name = env_id.split("-")[0]
    runner_script = f"scripts/run_{clean_name.lower()}_benchmark.py"
    b64_source = _get_b64_source_payload()

    cells = [
        # Cell 1: Markdown Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# TQC Reproduction: {env_id} Benchmark Training (ICML 2020) - Phase {phase_num}\n",
                "\n",
                f"Autonomous Kaggle GPU reproduction notebook for **Truncated Quantile Critics (TQC)** on **{env_id}**.\n",
                f"- **Paper Reference**: Arsenii Kuznetsov et al. *\"Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics\"* (ICML 2020).\n",
                f"- **Published Target Score**: **{target_score}** (Table 1 / Table 2)\n",
                f"- **Critic Configuration**: $M=5$ critics, $N=25$ quantiles ($MN=125$ atoms), $d=5$ top quantiles discarded\n",
                f"- **Default Multi-Seed Suite**: Seeds `[42, 43, 44, 45, 46]` (1,000,000 steps each)\n",
            ],
        },
        # Cell 2: GPU Detection & System Setup
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Verify GPU Accelerator and Configure Headless EGL Rendering\n",
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "\n",
                "os.environ[\"MUJOCO_GL\"] = \"egl\"\n",
                "\n",
                "import torch\n",
                "print('=== Hardware Environment ===')\n",
                "print(f'PyTorch Version : {torch.__version__}')\n",
                "print(f'CUDA Available  : {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f'GPU Device      : {gpu_name} ({vram:.2f} GB VRAM)')\n",
                "    torch.backends.cudnn.benchmark = True\n",
                "else:\n",
                "    print('WARNING: Running without GPU accelerator. In Kaggle, enable GPU under Notebook Options -> Accelerator -> GPU T4 x 2 or P100.')\n",
            ],
        },
        # Cell 3: Dependency Installation
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Install Gymnasium MuJoCo Simulation Harness and Rendering Libraries\n",
                "!pip install -q \"gymnasium[mujoco]>=0.29.0\" \"mujoco>=3.0.0\" imageio imageio-ffmpeg pyyaml\n",
                "\n",
                "import gymnasium as gym\n",
                f"test_env = gym.make('{env_id}')\n",
                "obs, _ = test_env.reset()\n",
                f"print(f'Environment {env_id} operational. State shape: {{obs.shape}}, Action shape: {{test_env.action_space.shape}}')\n",
                "test_env.close()\n",
            ],
        },
        # Cell 4: Repository Setup
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Setup Project Repository in Kaggle Workspace\n",
                "import os\n",
                "import sys\n",
                "import base64\n",
                "import io\n",
                "import tarfile\n",
                "\n",
                "REPO_DIR = '/kaggle/working/RL_project'\n",
                "os.makedirs(REPO_DIR, exist_ok=True)\n",
                "os.chdir(REPO_DIR)\n",
                "if REPO_DIR not in sys.path:\n",
                "    sys.path.insert(0, REPO_DIR)\n",
                "\n",
                "# Unpack embedded complete project codebase\n",
                f'b64_payload = """{b64_source}"""\n',
                "if b64_payload:\n",
                "    tar_bytes = base64.b64decode(b64_payload.strip())\n",
                "    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:\n",
                "        tar.extractall(path='.')\n",
                "    print('Successfully unpacked embedded TQC project codebase into workspace.')\n",
                "else:\n",
                "    !git clone https://github.com/aryanthepain/RL_project.git {REPO_DIR}\n",
                "\n",
                "print(f'Working directory set to: {os.getcwd()}')\n",
            ],
        },
        # Cell 5: Benchmark Execution Configuration
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Configure Benchmark Parameters\n",
                f"ENV_ID = '{env_id}'\n",
                "SEEDS = [42, 43, 44, 45, 46]  # Multi-seed evaluation suite\n",
                "TOTAL_STEPS = 1_000_000        # Standard 1M benchmark steps\n",
                "EVAL_FREQ = 5_000              # Evaluation every 5k steps\n",
                "EVAL_EPISODES = 10             # 10 deterministic rollouts per evaluation\n",
                "CHECKPOINT_FREQ = 50_000       # Periodic checkpoint saving\n",
                "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                f"LOG_DIR = f'runs/{env_id}'\n",
                f"RESULTS_DIR = f'results/{env_id}'\n",
                "\n",
                "print(f'Configuration ready for {ENV_ID}: {len(SEEDS)} seeds x {TOTAL_STEPS:,} steps on {DEVICE}')\n",
            ],
        },
        # Cell 6: Execute Benchmark Training
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 5. Execute Phase {phase_num} Benchmark Training & Bias Diagnostics\n",
                "# Set --seeds to 42 for a single-seed pilot run, or --seeds 42 43 44 45 46 for the full suite\n",
                f"!python {runner_script} \\\n",
                "    --seeds 42 \\\n",
                "    --steps 1000000 \\\n",
                "    --eval-freq 5000 \\\n",
                "    --eval-episodes 10 \\\n",
                "    --checkpoint-freq 50000 \\\n",
                f"    --log-dir runs/{env_id} \\\n",
                f"    --results-dir results/{env_id} \\\n",
                "    --mode full\n",
            ],
        },
        # Cell 7: Live Replication Curves and Comparative Table
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Display Replication Learning Curves and Diagnostics\n",
                "from IPython.display import Image, Markdown, display\n",
                "import os\n",
                "\n",
                f"curves_path = f'results/{env_id}/{clean_name.lower()}_reproduction_learning_curves.png'\n",
                f"bias_plot_path = f'results/{env_id}/{clean_name.lower()}_bias_diagnostics.png'\n",
                f"table_path = f'results/{env_id}/{clean_name.lower()}_results_table.md'\n",
                "\n",
                "if os.path.isfile(curves_path):\n",
                "    print('Replication Learning Curves:')\n",
                "    display(Image(curves_path))\n",
                "\n",
                "if os.path.isfile(bias_plot_path):\n",
                "    print('Overestimation Bias Diagnostics:')\n",
                "    display(Image(bias_plot_path))\n",
                "\n",
                "if os.path.isfile(table_path):\n",
                "    with open(table_path, 'r', encoding='utf-8') as f:\n",
                "        display(Markdown(f.read()))\n",
            ],
        },
        # Cell 8: Inactivity-Truncated Progression Video
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Render Evaluation Video with Adaptive Inactivity Truncation\n",
                "import glob\n",
                "from IPython.display import Video, display\n",
                "\n",
                f"ckpts = sorted(glob.glob(f'runs/{env_id}/**/best_model.pt', recursive=True))\n",
                "if ckpts:\n",
                "    best_ckpt = ckpts[0]\n",
                f"    video_out = f'/kaggle/working/{clean_name.lower()}_eval.mp4'\n",
                "    print(f'Rendering evaluation video from: {best_ckpt}')\n",
                f"    !python src/tqc/visualize.py --checkpoint {{best_ckpt}} --env {env_id} --output {{video_out}} --truncate-inactive --episodes 1\n",
                "    if os.path.isfile(video_out):\n",
                "        display(Video(video_out, embed=True, width=640))\n",
                "else:\n",
                "    print('No checkpoints found to render video.')\n",
            ],
        },
        # Cell 9: Artifact Archiving & SHA256 Checksums for Download
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. Package Artifacts into .tar.gz with SHA256 Verification\n",
                "import hashlib\n",
                "import tarfile\n",
                "\n",
                f"archive_name = '/kaggle/working/tqc_{clean_name.lower()}_phase{phase_num}_artifacts.tar.gz'\n",
                f"target_run_dir = 'runs/{env_id}'\n",
                "\n",
                "if os.path.isdir(target_run_dir):\n",
                "    print(f'Creating compressed artifact archive: {archive_name}')\n",
                "    with tarfile.open(archive_name, 'w:gz') as tar:\n",
                f"        tar.add(target_run_dir, arcname='{env_id}')\n",
                f"        if os.path.isdir('results/{env_id}'):\n",
                f"            tar.add('results/{env_id}', arcname='results_{clean_name.lower()}')\n",
                "\n",
                "    h = hashlib.sha256()\n",
                "    with open(archive_name, 'rb') as f:\n",
                "        while chunk := f.read(8192):\n",
                "            h.update(chunk)\n",
                "    sha256 = h.hexdigest()\n",
                "    print(f'Artifact package ready ({os.path.getsize(archive_name) / (1024*1024):.2f} MB)')\n",
                "    print(f'SHA256: {sha256}')\n",
                f"    with open('/kaggle/working/checksums_{clean_name.lower()}.sha256', 'w') as f:\n",
                "        f.write(f'{sha256}  {os.path.basename(archive_name)}\\n')\n",
                "    print('Done! Artifacts are available in Kaggle Output for one-click download or API sync.')\n",
                "else:\n",
                f"    print('Run directory runs/{env_id} not found.')\n",
            ],
        },
    ]

    notebook_data = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "dockerImageVersionId": 30880,
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"Generated {output_path} ({len(cells)} cells).")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    # Part 9: Hopper-v4
    hopper_path = repo_root / "notebooks" / "kaggle_tqc_hopper_phase09.ipynb"
    create_kaggle_benchmark_notebook(
        env_id="Hopper-v4",
        phase_num="09",
        target_score="3,695 ± 157 (ICML 2020 Table 1)",
        output_path=str(hopper_path),
    )

    # Part 11: Ant-v4
    ant_path = repo_root / "notebooks" / "kaggle_tqc_ant_phase11.ipynb"
    create_kaggle_benchmark_notebook(
        env_id="Ant-v4",
        phase_num="11",
        target_score="6,293 ± 276 (ICML 2020 Table 1) / 8.01 ± 0.87k (Table 2)",
        output_path=str(ant_path),
    )


if __name__ == "__main__":
    main()
