"""
Truncated Quantile Critics (TQC) PyTorch implementation.
Based on Kuznetsov et al. (ICML 2020):
'Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics'
"""

from .actor import Actor
from .agent import TQCAgent
from .critic import Critic, CriticEnsemble
from .envs import BENCHMARK_ENVS, get_env_dims, get_env_metadata, make_env
from .evaluate import evaluate_policy
from .logger import MetricsLogger, create_agent
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
    "ReplayBuffer",
    "TQCAgent",
    "create_agent",
    "evaluate_policy",
    "get_device",
    "get_env_dims",
    "get_env_metadata",
    "make_env",
    "quantile_huber_loss",
    "seed_everything",
    "train_tqc",
    "truncate_quantiles",
]
