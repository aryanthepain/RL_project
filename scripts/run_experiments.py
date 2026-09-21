#!/usr/bin/env python3
"""Unified Multi-Tier Experiment Runner CLI for TQC Benchmarks.

Provides automated hardware probing, compute hierarchy resolution,
and workload dispatch across Kaggle, Google Colab, Local GPU, and Local CPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tqc.compute.dispatcher import (
    ComputeDispatcher,
    PRESET_CONFIGS,
    SUPPORTED_ENVS,
)
from src.tqc.compute.tier_detector import ComputeTier, TierDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified TQC Benchmark Experiment Runner across Compute Hierarchy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        "--env-id",
        dest="env_id",
        type=str,
        default="HalfCheetah-v4",
        choices=SUPPORTED_ENVS,
        help="Target MuJoCo benchmark environment",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="quick",
        choices=list(PRESET_CONFIGS.keys()),
        help="Experiment workload preset (smoke: 100 steps, quick: 10k steps, full: 1M/3M steps)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="auto",
        choices=["auto", "kaggle", "colab", "gpu", "cpu"],
        help="Compute tier target ('auto' resolves highest available tier)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Auto-resume from existing checkpoint if found",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not auto-resume; start fresh run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview workload dispatch plan without executing",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Bypass interactive confirmation prompt",
    )
    parser.add_argument(
        "--list-tiers",
        action="store_true",
        help="Probe hardware and print compute hierarchy status table, then exit",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Explicit experiment folder name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = TierDetector()

    if args.list_tiers:
        print(detector.format_summary_table())
        sys.exit(0)

    dispatcher = ComputeDispatcher(detector=detector)
    effective_tier, probes = dispatcher.resolve_tier(args.tier)

    # Display report table
    print(detector.format_summary_table(probed=probes, selected_tier=effective_tier))

    # Interactive confirmation prompt if in a terminal and not bypassed
    if not args.yes and not args.dry_run and sys.stdin.isatty():
        try:
            choice = input(f"\nDispatch {args.preset} on {args.env_id} to [{effective_tier.value.upper()}]? [Y/n]: ").strip().lower()
            if choice in ["n", "no"]:
                print("Dispatch aborted by user.")
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)

    dispatcher.dispatch(
        env_id=args.env_id,
        preset=args.preset,
        tier=effective_tier,
        seed=args.seed,
        resume=args.resume,
        dry_run=args.dry_run,
        log_dir=args.log_dir,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    main()
