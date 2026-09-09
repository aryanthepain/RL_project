import csv
import json
import os
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np


ALGO_COLORS = {
    "TQC": "#1f77b4",       # Deep vibrant blue
    "tqc": "#1f77b4",
    "SAC": "#ff7f0e",       # Vivid amber / orange
    "sac": "#ff7f0e",
    "TD3": "#2ca02c",       # Green
    "td3": "#2ca02c",
}


def load_run_data(run_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """Load step and eval_return_mean from a run directory (CSV or JSONL)."""
    csv_path = os.path.join(run_dir, "metrics.csv")
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")

    steps = []
    returns = []

    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "eval_return_mean" in row and row["eval_return_mean"] != "":
                    try:
                        step = float(row["step"])
                        ret = float(row["eval_return_mean"])
                        steps.append(step)
                        returns.append(ret)
                    except (ValueError, TypeError):
                        continue
    elif os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "eval_return_mean" in data:
                        steps.append(float(data["step"]))
                        returns.append(float(data["eval_return_mean"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

    if len(steps) == 0:
        return None

    steps_arr = np.array(steps, dtype=np.float64)
    returns_arr = np.array(returns, dtype=np.float64)

    # Sort by step
    sort_idx = np.argsort(steps_arr)
    return {"steps": steps_arr[sort_idx], "returns": returns_arr[sort_idx]}


def plot_learning_curves(
    run_dirs_by_algo: Dict[str, List[str]],
    output_path: str,
    env_name: str = "HalfCheetah-v4",
    title: Optional[str] = None,
    grid_points: int = 100,
) -> None:
    """Plot smoothed evaluation return curves with shaded std error bands across seeds.

    Args:
        run_dirs_by_algo: Dict mapping algorithm name ('TQC', 'SAC') to list of run directories.
        output_path: Output image path (PNG or PDF).
        env_name: Environment name for plot labels.
        title: Optional plot title override.
        grid_points: Number of uniform interpolation grid points.
    """
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.grid(True, linestyle="--", alpha=0.5, color="#d3d3d3")

    for algo, run_dirs in run_dirs_by_algo.items():
        runs_data = []
        max_step = 0.0
        min_step = float("inf")

        for r_dir in run_dirs:
            data = load_run_data(r_dir)
            if data is not None:
                runs_data.append(data)
                min_step = min(min_step, data["steps"][0])
                max_step = max(max_step, data["steps"][-1])

        if not runs_data:
            continue

        if max_step <= min_step:
            max_step = min_step + 1.0

        common_steps = np.linspace(min_step, max_step, grid_points)
        interpolated_runs = []

        for data in runs_data:
            if len(data["steps"]) == 1:
                interp = np.full_like(common_steps, data["returns"][0])
            else:
                interp = np.interp(common_steps, data["steps"], data["returns"])
            interpolated_runs.append(interp)

        interpolated_arr = np.array(interpolated_runs)  # (N_runs, grid_points)
        mean_curve = np.mean(interpolated_arr, axis=0)
        std_curve = np.std(interpolated_arr, axis=0)

        color = ALGO_COLORS.get(algo, None)
        label = f"{algo} ({len(runs_data)} seeds)"

        line = plt.plot(common_steps, mean_curve, label=label, color=color, linewidth=2.2)[0]
        plt.fill_between(
            common_steps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=line.get_color(),
            alpha=0.2,
        )

    plt.xlabel("Environment Timesteps", fontsize=12, fontweight="medium")
    plt.ylabel("Average Return", fontsize=12, fontweight="medium")
    plt.title(title or f"Empirical Reproduction: {env_name}", fontsize=14, fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="lower right", fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_bias_comparison(
    bias_reports_by_algo: Dict[str, Dict[str, Any]],
    output_path: str,
    env_name: str = "HalfCheetah-v4",
    title: Optional[str] = None,
) -> None:
    """Generate comparative figures for overestimation bias (Delta = Q_pred - G_true).

    Args:
        bias_reports_by_algo: Dict mapping algorithm name to bias analysis report dict.
        output_path: Destination figure path.
        env_name: Environment identifier.
        title: Plot title override.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    algos = list(bias_reports_by_algo.keys())
    mean_biases = [bias_reports_by_algo[a]["mean_bias"] for a in algos]
    std_biases = [bias_reports_by_algo[a].get("std_bias", 0.0) for a in algos]
    colors = [ALGO_COLORS.get(a, "#333333") for a in algos]

    # Subplot 1: Average Estimation Bias
    bars = ax1.bar(algos, mean_biases, yerr=std_biases, color=colors, capsize=6, alpha=0.85, width=0.5)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax1.set_ylabel("Estimation Bias: E[Q(s, a) - G]", fontsize=11, fontweight="medium")
    ax1.set_title("Average Estimation Error", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Q-prediction vs True MC return
    for a in algos:
        rep = bias_reports_by_algo[a]
        q_preds = rep.get("q_predictions", [])
        mc_returns = rep.get("mc_returns", [])
        if q_preds and mc_returns:
            c = ALGO_COLORS.get(a, None)
            ax2.scatter(mc_returns, q_preds, alpha=0.6, label=a, color=c, edgecolors="none", s=35)

    # Diagonal reference line (perfect calibration)
    all_vals = []
    for a in algos:
        all_vals.extend(bias_reports_by_algo[a].get("mc_returns", []))
        all_vals.extend(bias_reports_by_algo[a].get("q_predictions", []))
    if all_vals:
        min_v, max_v = min(all_vals), max(all_vals)
        ax2.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.7, label="Ideal Calibration (Q = G)")

    ax2.set_xlabel("True Monte Carlo Return (G)", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Predicted Q-Value", fontsize=11, fontweight="medium")
    ax2.set_title(f"Predicted Q vs Empirical Return ({env_name})", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=10)

    plt.suptitle(title or f"Overestimation Bias Analysis: {env_name}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
