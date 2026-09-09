import os
import shutil
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tqc.agent import TQCAgent
from src.tqc.bias_analysis import compute_overestimation_bias, save_bias_report
from src.tqc.envs import get_env_dims, make_env
from src.tqc.eval_table import generate_comparison_table
from src.tqc.plotter import plot_bias_comparison, plot_learning_curves
from src.tqc.train import train_tqc


def main():
    print("=" * 70)
    print("  TQC vs. SAC Empirical Reproduction Demo Experiment (HalfCheetah-v4)")
    print("=" * 70)

    env_id = "HalfCheetah-v4"
    total_steps = 1500
    warmup_steps = 300
    eval_freq = 300
    eval_episodes = 5
    batch_size = 64
    seed = 42

    runs_dir = "runs/demo_experiment"
    results_dir = "results/demo_experiment"
    artifact_dir = r"C:\Users\Aryan Gupta\.gemini\antigravity-ide\brain\9e3b6c26-95ca-4342-a5ad-689cd1a54045"

    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Train TQC
    # -------------------------------------------------------------
    print(f"\n[1/4] Training TQC on {env_id} ({total_steps} steps, drop_top=5)...")
    tqc_exp_name = "tqc_halfcheetah_demo"
    tqc_agent = train_tqc(
        env_id=env_id,
        seed=seed,
        total_timesteps=total_steps,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        algo="tqc",
        device="cpu",
        log_dir=runs_dir,
        exp_name=tqc_exp_name,
        checkpoint_freq=0,
    )

    # -------------------------------------------------------------
    # 2. Train SAC
    # -------------------------------------------------------------
    print(f"\n[2/4] Training SAC Baseline on {env_id} ({total_steps} steps)...")
    sac_exp_name = "sac_halfcheetah_demo"
    sac_agent = train_tqc(
        env_id=env_id,
        seed=seed,
        total_timesteps=total_steps,
        eval_freq=eval_freq,
        eval_episodes=eval_episodes,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        algo="sac",
        device="cpu",
        log_dir=runs_dir,
        exp_name=sac_exp_name,
        checkpoint_freq=0,
    )

    # -------------------------------------------------------------
    # 3. Overestimation Bias Analysis
    # -------------------------------------------------------------
    print("\n[3/4] Computing Empirical Overestimation Bias against Monte Carlo Returns...")
    eval_env = make_env(env_id, seed=seed + 500)

    print("  -> Evaluating TQC bias...")
    tqc_bias = compute_overestimation_bias(
        tqc_agent, eval_env, n_samples=30, gamma=0.99, max_steps=200
    )
    tqc_bias_path = os.path.join(results_dir, "tqc_bias.json")
    save_bias_report(tqc_bias, tqc_bias_path)

    print("  -> Evaluating SAC bias...")
    sac_bias = compute_overestimation_bias(
        sac_agent, eval_env, n_samples=30, gamma=0.99, max_steps=200
    )
    sac_bias_path = os.path.join(results_dir, "sac_bias.json")
    save_bias_report(sac_bias, sac_bias_path)

    eval_env.close()

    print(f"  TQC: Mean Q = {tqc_bias['mean_predicted_q']:.2f}, Mean MC Return = {tqc_bias['mean_true_return']:.2f}, Bias = {tqc_bias['mean_bias']:.2f}")
    print(f"  SAC: Mean Q = {sac_bias['mean_predicted_q']:.2f}, Mean MC Return = {sac_bias['mean_true_return']:.2f}, Bias = {sac_bias['mean_bias']:.2f}")

    # -------------------------------------------------------------
    # 4. Generate Plots & Comparison Table
    # -------------------------------------------------------------
    print("\n[4/4] Rendering publication-style curves & comparative figures...")

    # Learning Curves
    tqc_run_dir = os.path.join(runs_dir, tqc_exp_name)
    sac_run_dir = os.path.join(runs_dir, sac_exp_name)
    curves_path = os.path.join(results_dir, "demo_learning_curves.png")
    plot_learning_curves(
        run_dirs_by_algo={"TQC (Ours)": [tqc_run_dir], "SAC Baseline": [sac_run_dir]},
        output_path=curves_path,
        env_name="HalfCheetah-v4",
        title="TQC vs SAC Learning Returns (HalfCheetah-v4 Demonstration)",
    )

    # Bias Plot
    bias_plot_path = os.path.join(results_dir, "demo_bias_comparison.png")
    plot_bias_comparison(
        bias_reports_by_algo={"TQC (Truncated)": tqc_bias, "SAC (Baseline)": sac_bias},
        output_path=bias_plot_path,
        env_name="HalfCheetah-v4",
        title="Overestimation Bias Diagnostic: TQC vs SAC",
    )

    # Copy plots to artifact directory for markdown display
    art_curves_path = os.path.join(artifact_dir, "demo_learning_curves.png")
    art_bias_path = os.path.join(artifact_dir, "demo_bias_comparison.png")
    shutil.copy2(curves_path, art_curves_path)
    shutil.copy2(bias_plot_path, art_bias_path)

    # Comparison Table
    table_path = os.path.join(results_dir, "demo_comparison_table.md")
    md_table, _ = generate_comparison_table(results_dir=runs_dir)
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print("\n" + "=" * 70)
    print("  Demo Complete! Artifacts Generated:")
    print(f"  - Learning Curves:   {curves_path}")
    print(f"  - Bias Diagnostics:  {bias_plot_path}")
    print(f"  - Comparison Table:  {table_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
