import argparse
import glob
import os
from typing import Dict, Optional, Tuple
import numpy as np
from .plotter import load_run_data


# Reference results published in Kuznetsov et al. (ICML 2020), Table 1
# Mean ± standard error over 10 random seeds at the end of training
PAPER_REFERENCE_SCORES: Dict[str, Dict[str, str]] = {
    "HalfCheetah-v4": {
        "steps": "1M",
        "tqc_paper": "14,028 ± 232",
        "sac_paper": "11,489 ± 255",
    },
    "Hopper-v4": {
        "steps": "1M",
        "tqc_paper": "3,695 ± 157",
        "sac_paper": "3,456 ± 88",
    },
    "Walker2d-v4": {
        "steps": "1M",
        "tqc_paper": "5,640 ± 192",
        "sac_paper": "4,892 ± 204",
    },
    "Ant-v4": {
        "steps": "3M",
        "tqc_paper": "6,293 ± 276",
        "sac_paper": "5,850 ± 198",
    },
    "Humanoid-v4": {
        "steps": "3M",
        "tqc_paper": "6,564 ± 350",
        "sac_paper": "5,420 ± 410",
    },
}


def compute_final_score(run_dirs: list, tail_k: int = 10) -> str:
    """Compute average score and standard deviation across seeds for the last tail_k evaluations."""
    final_returns = []
    for r_dir in run_dirs:
        data = load_run_data(r_dir)
        if data is not None and len(data["returns"]) > 0:
            tail = data["returns"][-tail_k:]
            final_returns.append(np.mean(tail))

    if not final_returns:
        return "N/A"

    mean_v = np.mean(final_returns)
    std_v = np.std(final_returns)
    return f"{mean_v:,.0f} ± {std_v:,.0f}"


def aggregate_empirical_runs(results_dir: str) -> Dict[str, Dict[str, str]]:
    """Scan results_dir for experiment directories and extract reproduced scores."""
    empirical = {}
    for env_id in PAPER_REFERENCE_SCORES.keys():
        empirical[env_id] = {}
        for algo in ["tqc", "sac"]:
            pattern = os.path.join(results_dir, f"{algo}_{env_id}*")
            matching_dirs = glob.glob(pattern)
            score_str = compute_final_score(matching_dirs)
            empirical[env_id][algo] = score_str
    return empirical


def generate_comparison_table(
    results_dir: Optional[str] = None,
    empirical_data: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[str, str]:
    """Generate Markdown and LaTeX comparison tables against ICML 2020 reported results.

    Args:
        results_dir: Optional path to experiment output directories.
        empirical_data: Optional explicit dictionary of empirical results.

    Returns:
        Tuple of (markdown_string, latex_string).
    """
    if empirical_data is None:
        if results_dir and os.path.isdir(results_dir):
            empirical_data = aggregate_empirical_runs(results_dir)
        else:
            empirical_data = {}

    # 1. Build Markdown Table
    md_lines = [
        "# Benchmark Results Comparison: TQC vs. SAC Baselines",
        "",
        "Empirical reproduction against published results from Kuznetsov et al. (ICML 2020), Table 1:",
        "",
        "| Environment | Steps | TQC (ICML 2020) | SAC (ICML 2020) | TQC (Reproduction) | SAC (Reproduction) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for env_id, ref in PAPER_REFERENCE_SCORES.items():
        steps = ref["steps"]
        tqc_ref = ref["tqc_paper"]
        sac_ref = ref["sac_paper"]

        env_emp = empirical_data.get(env_id, {})
        tqc_emp = env_emp.get("tqc", "Pending Run")
        sac_emp = env_emp.get("sac", "Pending Run")

        md_lines.append(
            f"| **{env_id}** | {steps} | {tqc_ref} | {sac_ref} | {tqc_emp} | {sac_emp} |"
        )

    markdown_table = "\n".join(md_lines) + "\n"

    # 2. Build LaTeX Table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Environment} & \textbf{Steps} & \textbf{TQC (ICML'20)} & \textbf{SAC (ICML'20)} & \textbf{TQC (Ours)} & \textbf{SAC (Ours)} \\",
        r"\midrule",
    ]

    for env_id, ref in PAPER_REFERENCE_SCORES.items():
        clean_env = env_id.replace("_", r"\_")
        steps = ref["steps"]
        tqc_ref = ref["tqc_paper"].replace("±", r"\pm")
        sac_ref = ref["sac_paper"].replace("±", r"\pm")

        env_emp = empirical_data.get(env_id, {})
        tqc_emp = env_emp.get("tqc", "Pending").replace("±", r"\pm")
        sac_emp = env_emp.get("sac", "Pending").replace("±", r"\pm")

        latex_lines.append(
            f"{clean_env} & {steps} & ${tqc_ref}$ & ${sac_ref}$ & ${tqc_emp}$ & ${sac_emp}$ \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Empirical evaluation comparing TQC and SAC against reported ICML 2020 benchmark numbers.}",
        r"\label{tab:tqc_reproduction_results}",
        r"\end{table}",
    ])

    latex_table = "\n".join(latex_lines) + "\n"
    return markdown_table, latex_table


def main():
    parser = argparse.ArgumentParser(description="TQC Benchmark Results Comparison Generator")
    parser.add_argument("--results-dir", type=str, default="runs", help="Directory containing training runs")
    parser.add_argument("--output-md", type=str, default="results/comparison_table.md", help="Output markdown path")
    parser.add_argument("--output-tex", type=str, default="results/comparison_table.tex", help="Output LaTeX path")

    args = parser.parse_args()
    md_tbl, tex_tbl = generate_comparison_table(results_dir=args.results_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_tbl)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_tex)), exist_ok=True)
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.write(tex_tbl)

    print(md_tbl)
    print(f"\nTables written to {args.output_md} and {args.output_tex}")


if __name__ == "__main__":
    main()
