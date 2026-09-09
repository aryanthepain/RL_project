"""Sync, unpack, and verify remote Kaggle experiment artifacts with SHA256 integrity checks."""

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, List, Optional, Union


class ArtifactIntegrityError(Exception):
    """Raised when artifact download, SHA256 verification, or metrics schema validation fails."""
    pass


def sha256_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def download_kernel_output(kernel_id: str, download_dir: Union[str, Path]) -> bool:
    """Download output artifacts from a completed Kaggle kernel via the Kaggle CLI.

    Args:
        kernel_id: Kaggle kernel identifier (e.g. username/slug).
        download_dir: Destination directory for downloaded files.

    Returns:
        True if download succeeded, False otherwise.
    """
    dest = Path(download_dir)
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "kernels", "output", kernel_id, "-p", str(dest)]
    print(f"[ArtifactSync] Executing: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ArtifactSync Error] Failed to fetch output: {e.stderr or e.stdout}")
        return False
    except Exception as e:
        print(f"[ArtifactSync Error] Unexpected error: {e}")
        return False


def verify_and_extract_artifacts(
    download_dir: Union[str, Path],
    target_run_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Unpack artifacts.tar.gz, verify SHA256 checksums, and validate metrics.csv schema.

    Args:
        download_dir: Directory containing raw Kaggle kernel output.
        target_run_dir: Final destination run directory in local runs/.

    Returns:
        Dictionary containing verified file list and metrics row count.

    Raises:
        ArtifactIntegrityError: If checksums mismatch or metrics schema is invalid.
    """
    src_path = Path(download_dir).resolve()
    dest_path = Path(target_run_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    # 1. Unpack artifacts.tar.gz if present
    archive_file = src_path / "artifacts.tar.gz"
    if not archive_file.is_file():
        # Check if archive exists inside nested folders
        nested = list(src_path.glob("**/artifacts.tar.gz"))
        if nested:
            archive_file = nested[0]

    if archive_file.is_file():
        print(f"[ArtifactSync] Unpacking {archive_file.name} to {dest_path}...")
        with tarfile.open(archive_file, "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(path=dest_path, filter="data")
            else:
                tar.extractall(path=dest_path)
    else:
        # Copy raw files if no tar.gz
        print(f"[ArtifactSync] No artifacts.tar.gz found; copying files directly to {dest_path}...")
        for item in src_path.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_path / item.name)
            elif item.is_dir() and item != dest_path:
                shutil.copytree(item, dest_path / item.name, dirs_exist_ok=True)

    # If unpacked files were placed inside an inner subfolder (e.g. dest_path / exp_name),
    # promote them if metrics.csv is inside the subfolder
    subdirs = [d for d in dest_path.iterdir() if d.is_dir() and (d / "metrics.csv").is_file()]
    if subdirs and not (dest_path / "metrics.csv").is_file():
        inner = subdirs[0]
        for item in inner.iterdir():
            target = dest_path / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    # 2. Checksum validation
    checksum_file = src_path / "checksums.sha256"
    if not checksum_file.is_file():
        nested_checksums = list(src_path.glob("**/checksums.sha256")) + list(dest_path.glob("**/checksums.sha256"))
        if nested_checksums:
            checksum_file = nested_checksums[0]

    verified_files = []
    if checksum_file.is_file():
        print(f"[ArtifactSync] Validating SHA256 integrity against {checksum_file.name}...")
        with open(checksum_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                expected_hash, rel_file = parts[0], parts[1].strip()

                # Find candidate matching file in dest_path or src_path
                candidates = [
                    dest_path / rel_file,
                    dest_path / Path(rel_file).name,
                    src_path / rel_file,
                    src_path / Path(rel_file).name,
                ]
                match = next((c for c in candidates if c.is_file()), None)

                if match:
                    actual_hash = sha256_file(match)
                    if actual_hash != expected_hash:
                        raise ArtifactIntegrityError(
                            f"SHA256 mismatch for {match.name}! Expected {expected_hash}, got {actual_hash}"
                        )
                    verified_files.append(str(match.name))
        print(f"[ArtifactSync] Successfully verified {len(verified_files)} files against checksum manifest.")
    else:
        print("[ArtifactSync] Warning: No checksums.sha256 found. Proceeding with schema validation.")

    # 3. Schema validation of metrics.csv
    metrics_csv = dest_path / "metrics.csv"
    if not metrics_csv.is_file():
        nested_metrics = list(dest_path.glob("**/metrics.csv"))
        if nested_metrics:
            metrics_csv = nested_metrics[0]
        else:
            raise ArtifactIntegrityError(f"Missing required metrics.csv in synced artifacts: {dest_path}")

    metrics_rows = 0
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"step", "eval_return_mean"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ArtifactIntegrityError(
                f"metrics.csv missing required columns: {required_cols - set(reader.fieldnames or [])}"
            )
        for row in reader:
            metrics_rows += 1

    if metrics_rows == 0:
        raise ArtifactIntegrityError("metrics.csv contains zero data rows.")

    print(f"[ArtifactSync] Validated metrics.csv schema: {metrics_rows} evaluation records found.")

    return {
        "status": "verified",
        "target_dir": str(dest_path),
        "metrics_csv": str(metrics_csv),
        "metrics_rows": metrics_rows,
        "verified_files": verified_files,
    }


def print_downstream_triggers(target_run_dir: Union[str, Path]) -> None:
    """Display recommended CLI commands for plotting and progression visualization."""
    path_str = str(Path(target_run_dir).resolve())
    print("\n" + "=" * 70)
    print(" Downstream Verification & Visualization Triggers")
    print("=" * 70)
    print("1. Plot Return Curve & Diagnostics:")
    print(f"   python -m src.tqc.plotter --log-dirs \"{path_str}\"")
    print("\n2. Generate Progression Videos from Checkpoints:")
    print(f"   python scripts/run_halfcheetah_progression.py --skip-train --checkpoint-dir \"{path_str}\"")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Synchronize and verify Kaggle kernel output artifacts.")
    parser.add_argument("--kernel-id", type=str, help="Kaggle kernel ID (e.g. username/slug)")
    parser.add_argument("--download-dir", type=str, default="kaggle_downloads", help="Temporary staging directory")
    parser.add_argument("--target-dir", type=str, help="Target local run directory (e.g. runs/kaggle_halfcheetah_...)")
    parser.add_argument("--verify-only", type=str, help="Skip download and verify an existing directory directly")

    args = parser.parse_args()

    if args.verify_only:
        target_dir = Path(args.verify_only)
        result = verify_and_extract_artifacts(target_dir, target_dir)
        print_downstream_triggers(target_dir)
        return

    if not args.kernel_id:
        parser.error("--kernel-id is required when not in --verify-only mode.")

    download_path = Path(args.download_dir) / args.kernel_id.replace("/", "_")
    success = download_kernel_output(args.kernel_id, download_path)
    if not success:
        sys.exit(1)

    target_dir = args.target_dir or f"runs/kaggle_{args.kernel_id.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    result = verify_and_extract_artifacts(download_path, target_dir)
    print_downstream_triggers(target_dir)


if __name__ == "__main__":
    main()
