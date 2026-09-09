"""
Truncated Quantile Critics (TQC) PyTorch implementation.
Based on Kuznetsov et al. (ICML 2020):
'Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics'
"""

from .actor import Actor
from .agent import TQCAgent
from .benchmark import run_benchmark
from .bias_analysis import compute_overestimation_bias, save_bias_report
from .critic import Critic, CriticEnsemble
from .envs import BENCHMARK_ENVS, get_env_dims, get_env_metadata, make_env
from .eval_table import PAPER_REFERENCE_SCORES, generate_comparison_table
from .evaluate import evaluate_policy
from .logger import MetricsLogger, create_agent
from .plotter import plot_bias_comparison, plot_learning_curves
from .replay_buffer import ReplayBuffer
from .train import train_tqc
from .truncation import quantile_huber_loss, truncate_quantiles
from .utils import get_device, seed_everything

__all__ = [
    "Actor",
    "BENCHMARK_ENVS",
    "Critic",
    "CriticEnsemble",
    "MetricsLogger",
    "PAPER_REFERENCE_SCORES",
    "ReplayBuffer",
    "TQCAgent",
    "compute_overestimation_bias",
    "create_agent",
    "evaluate_policy",
    "generate_comparison_table",
    "get_device",
    "get_env_dims",
    "get_env_metadata",
    "make_env",
    "plot_bias_comparison",
    "plot_learning_curves",
    "quantile_huber_loss",
    "run_benchmark",
    "save_bias_report",
    "seed_everything",
    "train_tqc",
    "truncate_quantiles",
]
