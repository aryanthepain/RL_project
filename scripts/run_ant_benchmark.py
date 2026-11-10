#!/usr/bin/env python3
"""Phase 11: Ant-v4 Distributed Multi-Seed Benchmark Orchestrator.

Implements benchmark reproduction for Ant-v4 across 5 random seeds
(42, 43, 44, 45, 46) on the 8-dimensional continuous action space
following Kuznetsov et al. (ICML 2020) Table 1 & 2.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.tqc.agent import TQCAgent
from src.tqc.bias_analysis import compute_overestimation_bias, save_bias_report
from src.tqc.compute.dispatcher import ComputeDispatcher, PRESET_CONFIGS
from src.tqc.compute.tier_detector import ComputeTier, TierDetector
from src.tqc.envs import get_env_dims, get_env_metadata, make_env
from src.tqc.evaluate import evaluate_policy
from src.tqc.eval_table import generate_comparison_table
from src.tqc.plotter import plot_bias_comparison, plot_learning_curves
from src.tqc.train import train_tqc
from src.tqc.utils import get_device, seed_everything

# Phase 11 Target Environment Specification
ENV_ID = "Ant-v4"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_DROP_TOP = 5  # ICML 2020: d=5 (Table 2 / Section 4.2)
DEFAULT_TIMESTEPS = 1_000_000  # 1M steps standard benchmark
PAPER_REFERENCE_SCORE = "6,293 ± 276"  # Kuznetsov et al. (ICML 2020), Table 1 (or 8.01 ± 0.87k Table 2)


def run_ant_training(
    seeds: List[int],
    timesteps: int = DEFAULT_TIMESTEPS,
    drop_top: int = DEFAULT_DROP_TOP,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    warmup_steps: int = 10_000,
    batch_size: int = 256,
    checkpoint_freq: int = 50_000,
    device: str = "cpu",
    log_dir: str = "runs/Ant-v4",
    resume: bool = True,
    fast_test: bool = False,
) -> Dict[int, str]:
    """Execute TQC training runs across random seeds for Ant-v4 (8-DoF action space).

    Args:
        seeds: List of random seeds to execute.
        timesteps: Environment interaction steps per seed.
        drop_top: Top quantiles to truncate from pooled critic mixture.
        eval_freq: Evaluation interval in steps.
        eval_episodes: Deterministic test episodes per evaluation.
        warmup_steps: Initial random action exploration steps.
        batch_size: Mini-batch gradient update size.
        checkpoint_freq: Interval for saving model checkpoints.
        device: Compute device ('cuda' or 'cpu').
        log_dir: Base directory to store logs and checkpoints.
        resume: Auto-resume from existing checkpoint if found.
        fast_test: Fast smoke verification mode override.

    Returns:
        Dict mapping seed to the experiment directory path.
    """
    dev = get_device(device)
    os.makedirs(log_dir, exist_ok=True)
    completed_runs: Dict[int, str] = {}

    run_timesteps = 100 if fast_test else timesteps
    run_warmup = 20 if fast_test else warmup_steps
    run_eval_freq = 50 if fast_test else eval_freq
    run_eval_episodes = 2 if fast_test else eval_episodes
    run_batch_size = 32 if fast_test else batch_size
    run_ckpt_freq = 50 if fast_test else checkpoint_freq

    print("=" * 75)
    print(f"  PHASE 11: ANT-v4 BENCHMARK TRAINING (8-DoF Continuous Control)")
    print(f"  Seeds: {seeds} | Timesteps: {run_timesteps:,} | drop_top: {drop_top} | Device: {dev}")
    print("=" * 75)

    for idx, seed in enumerate(seeds, 1):
        exp_name = f"tqc_{ENV_ID}_seed{seed}"
        print(f"\n[{idx}/{len(seeds)}] Launching Seed {seed} -> {exp_name}...")

        agent = train_tqc(
            env_id=ENV_ID,
            seed=seed,
            total_timesteps=run_timesteps,
            eval_freq=run_eval_freq,
            eval_episodes=run_eval_episodes,
            warmup_steps=run_warmup,
            batch_size=run_batch_size,
            drop_top=drop_top,
            algo="tqc",
            device=dev,
            log_dir=log_dir,
            exp_name=exp_name,
            checkpoint_freq=run_ckpt_freq,
            resume=resume,
        )

        seed_run_dir = os.path.join(log_dir, exp_name)
        completed_runs[seed] = seed_run_dir
        print(f"  [DONE] Seed {seed} completed. Artifacts stored in: {seed_run_dir}")

    return completed_runs


def run_ant_bias_analysis(
    run_dirs: List[str],
    results_dir: str = "results/Ant-v4",
    n_samples: int = 30,
    gamma: float = 0.99,
    max_steps: int = 200,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Compute empirical overestimation bias against Monte Carlo discounted rollouts."""
    dev = get_device(device)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("  PHASE 11: EMPIRICAL OVERESTIMATION BIAS DIAGNOSTICS (Ant-v4)")
    print("=" * 75)

    bias_reports = []
    dummy_env = make_env(ENV_ID, seed=999)
    state_dim, action_dim = get_env_dims(dummy_env)
    dummy_env.close()

    for r_dir in run_dirs:
        best_model_path = os.path.join(r_dir, "best_model.pt")
        final_model_path = os.path.join(r_dir, "final_model.pt")
        target_model = best_model_path if os.path.isfile(best_model_path) else final_model_path

        if not os.path.isfile(target_model):
            print(f"  [SKIP] No checkpoint found in {r_dir} for bias evaluation.")
            continue

        agent = TQCAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            drop_top=DEFAULT_DROP_TOP,
            device=dev,
        )
        agent.load(target_model)

        eval_env = make_env(ENV_ID, seed=12345)
        print(f"  Evaluating overestimation bias for model: {target_model}...")
        report = compute_overestimation_bias(
            agent=agent,
            env=eval_env,
            n_samples=n_samples,
            gamma=gamma,
            max_steps=max_steps,
        )
        eval_env.close()

        bias_reports.append(report)
        print(
            f"    Predicted Q: {report['mean_predicted_q']:8.2f} | "
            f"MC Return: {report['mean_true_return']:8.2f} | "
            f"Bias: {report['mean_bias']:8.2f}"
        )

    if not bias_reports:
        print("  No bias reports generated.")
        return {}

    # Aggregate bias metrics
    agg_bias = {
        "env_id": ENV_ID,
        "n_runs": len(bias_reports),
        "mean_predicted_q": float(np.mean([r["mean_predicted_q"] for r in bias_reports])),
        "mean_true_return": float(np.mean([r["mean_true_return"] for r in bias_reports])),
        "mean_bias": float(np.mean([r["mean_bias"] for r in bias_reports])),
        "std_bias": float(np.std([r["mean_bias"] for r in bias_reports])),
        "raw_reports": bias_reports,
    }

    report_path = os.path.join(results_dir, "ant_bias_report.json")
    save_bias_report(agg_bias, report_path)
    print(f"\n  [SAVED] Aggregated Ant-v4 bias diagnostics -> {report_path}")
    return agg_bias


def generate_ant_reproduction_plots(
    log_dir: str = "runs/Ant-v4",
    results_dir: str = "results/Ant-v4",
    bias_report: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate replication return curves and overestimation bias plots for Ant-v4."""
    os.makedirs(results_dir, exist_ok=True)
    pattern = os.path.join(log_dir, f"tqc_{ENV_ID}_seed*")
    seed_dirs = sorted(glob.glob(pattern))

    if not seed_dirs:
        seed_dirs = [
            os.path.join(log_dir, d)
            for d in os.listdir(log_dir)
            if os.path.isdir(os.path.join(log_dir, d))
        ]

    print("\n" + "=" * 75)
    print(f"  PHASE 11: RENDERING REPLICATION PLOTS ({len(seed_dirs)} seeds)")
    print("=" * 75)

    if seed_dirs:
        curve_output = os.path.join(results_dir, "ant_reproduction_learning_curves.png")
        plot_learning_curves(
            run_dirs_by_algo={"TQC (Ours - Truncated M=5, N=25, d=5)": seed_dirs},
            output_path=curve_output,
            env_name=ENV_ID,
            title=f"TQC Reproduction Learning Curves: {ENV_ID} (5 Seeds)",
        )
        print(f"  [SAVED] Reproduction Curves -> {curve_output}")

    if bias_report and "raw_reports" in bias_report:
        bias_plot_output = os.path.join(results_dir, "ant_bias_diagnostics.png")
        plot_bias_comparison(
            bias_reports_by_algo={"TQC (d=5)": bias_report},
            output_path=bias_plot_output,
            env_name=ENV_ID,
            title=f"Empirical Overestimation Bias Diagnostic: {ENV_ID}",
        )
        print(f"  [SAVED] Bias Diagnostic Plot -> {bias_plot_output}")

    table_path = os.path.join(results_dir, "ant_results_table.md")
    md_table, _ = generate_comparison_table(results_dir=log_dir)
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"  [SAVED] Comparison Table -> {table_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 11: Ant-v4 Distributed Multi-Seed Benchmark Orchestrator (ICML 2020)"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds for benchmark evaluation (default: 42 43 44 45 46)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Total timesteps per seed (default: 1,000,000)",
    )
    parser.add_argument(
        "--drop-top",
        type=int,
        default=DEFAULT_DROP_TOP,
        help="Quantiles to truncate per critic mixture (default: 5)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="auto",
        choices=["auto", "kaggle", "colab", "gpu", "cpu"],
        help="Target compute tier in hierarchy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["train", "eval", "bias", "plot", "smoke", "full"],
        help="Execution workflow mode",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint saving interval in steps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluation interval in steps",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Evaluation episodes per interval",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for updates",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs/Ant-v4",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/Ant-v4",
        help="Directory to save plots and tables",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast 100-step smoke test for verification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview execution dispatch plan without running",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Do not resume from existing checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = TierDetector()
    dispatcher = ComputeDispatcher(detector=detector)
    effective_tier, probes = dispatcher.resolve_tier(args.tier)

    print(detector.format_summary_table(probed=probes, selected_tier=effective_tier))

    fast_mode = args.smoke or (args.mode == "smoke")
    chosen_device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dry_run:
        print("\n[DRY RUN] Ant-v4 Benchmark Dispatch Plan:")
        print(f"  Environment:     {ENV_ID}")
        print(f"  Seeds:           {args.seeds}")
        print(f"  Timesteps/seed:  {'100 (Smoke)' if fast_mode else f'{args.steps:,}'}")
        print(f"  Action Space:    8-DoF continuous action Box([-1, 1])")
        print(f"  Truncation d:    {args.drop_top} (k = 125 - {args.drop_top} = {125 - args.drop_top})")
        print(f"  Resolved Tier:   {effective_tier.value.upper()}")
        print(f"  Compute Device:  {chosen_device}")
        print(f"  Output Directory:{args.log_dir}")
        print(f"  Paper Target:    {PAPER_REFERENCE_SCORE}")
        return

    run_dirs: List[str] = []
    if args.mode in ["train", "full", "smoke"]:
        run_dict = run_ant_training(
            seeds=args.seeds,
            timesteps=args.steps,
            drop_top=args.drop_top,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            checkpoint_freq=args.checkpoint_freq,
            device=chosen_device,
            log_dir=args.log_dir,
            resume=args.resume,
            fast_test=fast_mode,
        )
        run_dirs = list(run_dict.values())
    else:
        pattern = os.path.join(args.log_dir, f"tqc_{ENV_ID}_seed*")
        run_dirs = sorted(glob.glob(pattern))

    bias_data = None
    if args.mode in ["bias", "full"]:
        bias_data = run_ant_bias_analysis(
            run_dirs=run_dirs,
            results_dir=args.results_dir,
            n_samples=10 if fast_mode else 30,
            device=chosen_device,
        )

    if args.mode in ["plot", "full", "smoke"]:
        if bias_data is None:
            report_path = os.path.join(args.results_dir, "ant_bias_report.json")
            if os.path.exists(report_path):
                import json
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        bias_data = json.load(f)
                except Exception as e:
                    print(f"  [WARN] Failed to load existing bias report: {e}")
        generate_ant_reproduction_plots(
            log_dir=args.log_dir,
            results_dir=args.results_dir,
            bias_report=bias_data,
        )

    print("\n" + "=" * 75)
    print(f"  Ant-v4 Benchmark Orchestration Finished Successfully!")
    print(f"  Logs & Checkpoints: {args.log_dir}")
    print(f"  Plots & Tables:     {args.results_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
