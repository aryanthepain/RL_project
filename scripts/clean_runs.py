#!/usr/bin/env python3
"""Disk quota inspection, checkpoint pruning, and run cleanup utility.

Enables inspecting disk usage across runs/ or Google Drive tqc_runs/,
pruning intermediate checkpoints while preserving best_model.pt, latest.pt,
final_model.pt, metrics.csv, and evaluation videos.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PRESERVED_FILENAMES = {
    "best_model.pt",
    "latest.pt",
    "final_model.pt",
    "metrics.csv",
    "config.json",
    "config.yaml",
}

VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi"}


@dataclass
class RunInfo:
    """Metadata describing a single experiment run directory."""

    run_id: str
    path: str
    total_size_bytes: int = 0
    checkpoint_count: int = 0
    intermediate_checkpoints: List[str] = field(default_factory=list)
    preserved_files: List[str] = field(default_factory=list)

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def prunable_size_bytes(self) -> int:
        total = 0
        for f in self.intermediate_checkpoints:
            try:
                total += os.path.getsize(f)
            except OSError:
                pass
        return total

    @property
    def prunable_size_mb(self) -> float:
        return self.prunable_size_bytes / (1024 * 1024)


def scan_run_directories(root_dir: str) -> List[RunInfo]:
    """Scan root_dir for run subdirectories and return their inspection metadata."""
    if not os.path.isdir(root_dir):
        return []

    runs: List[RunInfo] = []
    for entry in sorted(os.listdir(root_dir)):
        run_path = os.path.join(root_dir, entry)
        if not os.path.isdir(run_path):
            continue

        info = RunInfo(run_id=entry, path=run_path)
        for root, _, files in os.walk(run_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    info.total_size_bytes += size
                except OSError:
                    continue

                ext = os.path.splitext(file)[1].lower()
                if file in PRESERVED_FILENAMES or ext in VIDEO_EXTENSIONS:
                    info.preserved_files.append(file_path)
                elif re.match(r"^checkpoint_\d+\.pt$", file):
                    info.checkpoint_count += 1
                    info.intermediate_checkpoints.append(file_path)

        runs.append(info)
    return runs


def format_inspection_table(runs: List[RunInfo]) -> str:
    """Format ASCII report table of run disk usages and prunable capacity."""
    if not runs:
        return "No experiment run directories found."

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("                  TQC EXPERIMENT RUN DISK USAGE REPORT                  ")
    lines.append("=" * 80)
    lines.append(
        f"{'Run ID':<35} | {'Total Size':<11} | {'Prunable':<11} | {'Checkpoints':<12}"
    )
    lines.append("-" * 80)

    grand_total_bytes = 0
    grand_prunable_bytes = 0
    grand_ckpts = 0

    for r in runs:
        grand_total_bytes += r.total_size_bytes
        grand_prunable_bytes += r.prunable_size_bytes
        grand_ckpts += r.checkpoint_count

        total_str = f"{r.total_size_mb:.2f} MB"
        prunable_str = f"{r.prunable_size_mb:.2f} MB"
        lines.append(
            f"{r.run_id[:35]:<35} | {total_str:<11} | {prunable_str:<11} | {r.checkpoint_count:<12}"
        )

    lines.append("-" * 80)
    total_all_mb = grand_total_bytes / (1024 * 1024)
    prunable_all_mb = grand_prunable_bytes / (1024 * 1024)
    lines.append(
        f"{'TOTALS (' + str(len(runs)) + ' runs)':<35} | {f'{total_all_mb:.2f} MB':<11} | "
        f"{f'{prunable_all_mb:.2f} MB':<11} | {grand_ckpts:<12}"
    )
    lines.append("=" * 80)
    return "\n".join(lines)


def prune_intermediate_checkpoints(run_dir: str, dry_run: bool = False) -> Tuple[int, int]:
    """Delete intermediate checkpoint_*.pt files, preserving best, latest, and metrics.

    Returns:
        Tuple of (deleted_files_count, bytes_freed).
    """
    if not os.path.isdir(run_dir):
        return 0, 0

    deleted_count = 0
    freed_bytes = 0

    for root, _, files in os.walk(run_dir):
        for file in files:
            if re.match(r"^checkpoint_\d+\.pt$", file):
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                except OSError:
                    size = 0

                freed_bytes += size
                deleted_count += 1

                if not dry_run:
                    try:
                        os.remove(file_path)
                    except OSError as exc:
                        print(f"  [ERROR] Could not delete {file_path}: {exc}")

    return deleted_count, freed_bytes


def delete_run_directory(run_dir: str, dry_run: bool = False) -> int:
    """Delete an entire run directory, returning bytes freed."""
    if not os.path.isdir(run_dir):
        return 0

    total_bytes = 0
    for root, _, files in os.walk(run_dir):
        for file in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, file))
            except OSError:
                pass

    if not dry_run:
        shutil.rmtree(run_dir, ignore_errors=True)

    return total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and clean up TQC runs and checkpoint storage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="runs",
        help="Root directory containing experiment runs (e.g. runs or /content/drive/MyDrive/tqc_runs)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect disk usage and print summary table (default)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete intermediate checkpoints while preserving best_model, latest, and metrics",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete an entire run directory (requires --run <id>)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run directory name to target for pruning or deletion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without deleting files",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Bypass interactive confirmation prompt",
    )

    args = parser.parse_args()

    # Default action is inspect
    if not args.prune and not args.delete:
        args.inspect = True

    runs = scan_run_directories(args.dir)

    if args.inspect:
        print(format_inspection_table(runs))

    if args.prune:
        targets = [r for r in runs if args.run is None or r.run_id == args.run]
        if not targets:
            print(f"No matching runs found to prune in '{args.dir}'.")
            return

        total_files = sum(len(r.intermediate_checkpoints) for r in targets)
        total_mb = sum(r.prunable_size_mb for r in targets)
        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"\n{prefix}Pruning intermediate checkpoints across {len(targets)} run(s):")
        print(f"  Target files : {total_files}")
        print(f"  Reclaimable  : {total_mb:.2f} MB")

        if not args.yes and not args.dry_run and sys.stdin.isatty():
            choice = input("Proceed with pruning? [y/N]: ").strip().lower()
            if choice not in ["y", "yes"]:
                print("Pruning cancelled.")
                return

        total_del, total_freed = 0, 0
        for r in targets:
            cnt, freed = prune_intermediate_checkpoints(r.path, dry_run=args.dry_run)
            total_del += cnt
            total_freed += freed
            print(f"  {r.run_id}: pruned {cnt} checkpoints ({freed / (1024 * 1024):.2f} MB freed)")

        print(f"\n{prefix}Finished. {total_del} checkpoints pruned, {total_freed / (1024 * 1024):.2f} MB freed.")

    if args.delete:
        if not args.run:
            print("Error: --delete requires --run <run_id> parameter.")
            sys.exit(1)

        target_path = os.path.join(args.dir, args.run)
        if not os.path.isdir(target_path):
            print(f"Error: Run directory '{target_path}' does not exist.")
            sys.exit(1)

        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"\n{prefix}Deleting entire run directory: {target_path}")

        if not args.yes and not args.dry_run and sys.stdin.isatty():
            choice = input(f"Are you sure you want to permanently delete {args.run}? [y/N]: ").strip().lower()
            if choice not in ["y", "yes"]:
                print("Deletion cancelled.")
                return

        freed = delete_run_directory(target_path, dry_run=args.dry_run)
        print(f"{prefix}Deleted {target_path} ({freed / (1024 * 1024):.2f} MB freed).")


if __name__ == "__main__":
    main()
