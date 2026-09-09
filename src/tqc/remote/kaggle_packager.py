"""Kaggle kernel packager, remote entrypoint generator, and local pre-flight smoke validator."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.tqc.train import train_tqc


class KagglePackagingError(Exception):
    """Raised when kernel packaging or pre-flight verification fails."""
    pass


def build_kernel_metadata(
    username: str,
    kernel_slug: str,
    title: str,
    code_file: str = "remote_entrypoint.py",
    enable_gpu: bool = True,
    enable_internet: bool = True,
    is_private: bool = True,
) -> Dict[str, Any]:
    """Build standard Kaggle kernel-metadata.json configuration.

    Args:
        username: Kaggle username.
        kernel_slug: URL slug for the kernel.
        title: Human-readable kernel title.
        code_file: Entrypoint Python filename.
        enable_gpu: Whether to request GPU accelerator (default True).
        enable_internet: Whether internet access is enabled (default True).
        is_private: Whether the kernel is private (default True).

    Returns:
        Dictionary adhering to Kaggle CLI kernel-metadata schema.
    """
    if not username or not kernel_slug or not title:
        raise KagglePackagingError("username, kernel_slug, and title are required for kernel metadata.")

    return {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true" if is_private else "false",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def create_remote_entrypoint(
    env_id: str = "HalfCheetah-v4",
    seed: int = 42,
    total_timesteps: int = 1_000_000,
    checkpoint_freq: int = 50_000,
    eval_freq: int = 5_000,
    exp_name: str = "kaggle_halfcheetah_pilot",
    device: str = "cuda",
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    warmup_steps: int = 10_000,
    eval_episodes: int = 10,
) -> str:
    """Generate the remote Python script that runs on the Kaggle GPU worker.

    The generated script:
    1. Installs gymnasium MuJoCo bindings if missing.
    2. Unpacks tqc_source.tar.gz into execution directory.
    3. Auto-detects CUDA and enables cudnn benchmark optimizations.
    4. Executes train_tqc with exact paper parameters.
    5. Bundles output checkpoints and metrics into artifacts.tar.gz.
    6. Generates SHA256 checksums file.

    Returns:
        String containing complete Python script.
    """
    return f'''# Auto-generated Kaggle Remote Entrypoint for TQC Benchmark
import os
import sys
import subprocess
import tarfile
import hashlib
from pathlib import Path

print("=== Starting Remote TQC Worker ===")
print("Python version:", sys.version)

# 1. Install MuJoCo and Gymnasium dependencies
print("\\n[1/5] Ensuring required packages are installed...")
packages = ["gymnasium[mujoco]", "mujoco"]
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
except Exception as e:
    print(f"Warning: pip install encountered issue: {{e}}")

# 2. Unpack bundled source code
print("\\n[2/5] Setting up TQC source environment...")
archive_path = Path("tqc_source.tar.gz")
if archive_path.is_file():
    print("Extracting tqc_source.tar.gz...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=".")

# Add current working directory to sys.path
cwd = str(Path(".").resolve())
if cwd not in sys.path:
    sys.path.insert(0, cwd)

import torch
from src.tqc.train import train_tqc

# 3. Hardware verification & CUDA optimizations
print("\\n[3/5] Inspecting hardware accelerators...")
has_cuda = torch.cuda.is_available()
print(f"CUDA Available: {{has_cuda}}")
if has_cuda:
    print(f"Device Name: {{torch.cuda.get_device_name(0)}}")
    print(f"Device Count: {{torch.cuda.device_count()}}")
    torch.backends.cudnn.benchmark = True

chosen_device = "cuda" if (has_cuda and "{device}" != "cpu") else "cpu"
print(f"Using device: {{chosen_device}}")

# 4. Launch TQC Training
print("\\n[4/5] Launching TQC Training Loop...")
exp_dir = Path("runs") / "{exp_name}"
exp_dir.mkdir(parents=True, exist_ok=True)

agent = train_tqc(
    env_id="{env_id}",
    seed={seed},
    total_timesteps={total_timesteps},
    eval_freq={eval_freq},
    eval_episodes={eval_episodes},
    warmup_steps={warmup_steps},
    batch_size={batch_size},
    buffer_capacity={buffer_size},
    device=chosen_device,
    log_dir="runs",
    exp_name="{exp_name}",
    checkpoint_freq={checkpoint_freq},
)
print("Training execution finished.")

# 5. Bundle artifacts and compute SHA256 hashes
print("\\n[5/5] Packaging run artifacts and generating checksums...")
artifacts_archive = Path("artifacts.tar.gz")
checksum_file = Path("checksums.sha256")

with tarfile.open(artifacts_archive, "w:gz") as tar:
    if exp_dir.exists():
        tar.add(exp_dir, arcname="{exp_name}")

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

checksum_entries = []
# Hash archive
if artifacts_archive.exists():
    checksum_entries.append(f"{{sha256_file(artifacts_archive)}}  artifacts.tar.gz")

# Hash individual output files
for p in exp_dir.rglob("*"):
    if p.is_file():
        rel = p.relative_to(exp_dir.parent)
        checksum_entries.append(f"{{sha256_file(p)}}  {{rel}}")

with open(checksum_file, "w", encoding="utf-8") as f:
    f.write("\\n".join(checksum_entries) + "\\n")

print(f"Successfully created {{artifacts_archive.name}} ({{artifacts_archive.stat().st_size}} bytes)")
print("=== Remote Execution Completed Successfully ===")
'''


def package_kernel_directory(
    target_dir: Union[str, Path],
    metadata: Dict[str, Any],
    entrypoint_code: str,
    source_root: Optional[Union[str, Path]] = None,
) -> str:
    """Bundle project source code, metadata, and entrypoint script for Kaggle dispatch.

    Args:
        target_dir: Destination directory for kernel payload.
        metadata: Kaggle kernel-metadata dictionary.
        entrypoint_code: Generated Python script for remote execution.
        source_root: Root directory of project containing src/ (defaults to current dir).

    Returns:
        Absolute string path to prepared kernel directory.
    """
    dest_path = Path(target_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    # 1. Write kernel-metadata.json
    metadata_file = dest_path / "kernel-metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 2. Write entrypoint script
    code_filename = metadata.get("code_file", "remote_entrypoint.py")
    entrypoint_file = dest_path / code_filename
    with open(entrypoint_file, "w", encoding="utf-8") as f:
        f.write(entrypoint_code)

    # 3. Compress source directory into tqc_source.tar.gz
    root = Path(source_root).resolve() if source_root else Path.cwd().resolve()
    src_dir = root / "src"
    if not src_dir.is_dir():
        raise KagglePackagingError(f"Cannot find source directory at {src_dir}")

    source_archive = dest_path / "tqc_source.tar.gz"
    with tarfile.open(source_archive, "w:gz") as tar:
        tar.add(src_dir, arcname="src")

    return str(dest_path)


def run_local_preflight_smoke(
    env_id: str = "HalfCheetah-v4",
    steps: int = 100,
    device: str = "cpu",
    seed: int = 42,
) -> bool:
    """Execute a 100-step local dry run to verify environment, tensors, and checkpointing.

    Args:
        env_id: Target Gymnasium benchmark environment.
        steps: Number of environment steps (default 100).
        device: Torch device (default 'cpu').
        seed: Random seed.

    Returns:
        True if all components (env, agent, checkpointing, metrics) succeed.

    Raises:
        KagglePackagingError: If smoke run fails or expected files are missing.
    """
    temp_dir = tempfile.mkdtemp(prefix="tqc_smoke_")
    warmup_steps = min(20, max(5, steps // 3))
    batch_size = min(16, warmup_steps)
    eval_freq = max(10, steps // 2)
    checkpoint_freq = max(10, steps // 2)

    try:
        agent = train_tqc(
            env_id=env_id,
            seed=seed,
            total_timesteps=steps,
            eval_freq=eval_freq,
            eval_episodes=1,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            buffer_capacity=1_000,
            device=device,
            log_dir=temp_dir,
            exp_name="smoke_test",
            checkpoint_freq=checkpoint_freq,
        )

        run_dir = Path(temp_dir) / "smoke_test"
        if not run_dir.exists():
            raise KagglePackagingError(f"Smoke run directory was not created: {run_dir}")

        metrics_csv = run_dir / "metrics.csv"
        if not metrics_csv.is_file():
            raise KagglePackagingError(f"metrics.csv was not generated in smoke run: {metrics_csv}")

        checkpoints = list(run_dir.rglob("*.pt"))
        if not checkpoints:
            raise KagglePackagingError("No checkpoints were generated during smoke run.")

        return True
    except Exception as e:
        raise KagglePackagingError(f"Pre-flight smoke run failed: {e}") from e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
