import csv
import json
import os
import time
from typing import Any, Dict, Optional, Union
import torch
from .agent import TQCAgent
from .envs import get_env_metadata


class MetricsLogger:
    """Structured metric logger writing real-time metrics to CSV, JSONL, and console."""

    def __init__(self, log_dir: str = "runs", exp_name: Optional[str] = None) -> None:
        self.log_dir = log_dir
        if exp_name is None:
            exp_name = f"exp_{int(time.time())}"
        self.exp_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.csv_path = os.path.join(self.exp_dir, "metrics.csv")
        self.jsonl_path = os.path.join(self.exp_dir, "metrics.jsonl")

        self.csv_file = None
        self.csv_writer = None
        self.fieldnames = None
        self.jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

    def log_metrics(self, step: int, metrics: Dict[str, Any], print_to_console: bool = True) -> None:
        """Log a dictionary of metrics for the given training step."""
        row = {"step": step, "timestamp": time.time()}
        row.update(metrics)

        # Write to JSONL
        self.jsonl_file.write(json.dumps(row) + "\n")
        self.jsonl_file.flush()

        # Write to CSV
        if self.csv_writer is None:
            self.fieldnames = list(row.keys())
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()

        # If new keys appear, adjust writer
        if set(row.keys()) != set(self.fieldnames):
            # Write only existing fields or recreate
            row_to_write = {k: row.get(k, "") for k in self.fieldnames}
        else:
            row_to_write = row

        self.csv_writer.writerow(row_to_write)
        self.csv_file.flush()

        if print_to_console:
            eval_ret = metrics.get("eval_return_mean", None)
            c_loss = metrics.get("critic_loss", None)
            a_loss = metrics.get("actor_loss", None)
            alpha = metrics.get("alpha", None)

            msg = f"[Step {step:7d}]"
            if eval_ret is not None:
                eval_std = metrics.get("eval_return_std", 0.0)
                msg += f" Eval: {eval_ret:8.2f} ± {eval_std:6.2f} |"
            if c_loss is not None:
                msg += f" Critic Loss: {c_loss:6.3f} |"
            if a_loss is not None:
                msg += f" Actor Loss: {a_loss:6.3f} |"
            if alpha is not None:
                msg += f" Alpha: {alpha:6.4f}"
            print(msg)

    def close(self) -> None:
        """Flush and close all open log files."""
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.close()
        if self.jsonl_file is not None and not self.jsonl_file.closed:
            self.jsonl_file.close()


def create_agent(
    algo: str,
    state_dim: int,
    action_dim: int,
    device: Union[str, torch.device] = "cpu",
    drop_top: Optional[int] = None,
    env_id: Optional[str] = None,
) -> TQCAgent:
    """Create agent configured either for TQC (paper default) or SAC baseline.

    Args:
        algo: 'tqc' or 'sac'.
        state_dim: State dimension.
        action_dim: Continuous action dimension.
        device: Torch device.
        drop_top: Number of top quantiles to drop. If None, retrieved from env metadata.
        env_id: Optional environment ID to lookup default drop_top.
    """
    algo_lower = algo.lower()
    if algo_lower == "tqc":
        if drop_top is None:
            if env_id is not None:
                drop_top = get_env_metadata(env_id)["drop_top"]
            else:
                drop_top = 5
        return TQCAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_critics=5,
            n_quantiles=25,
            drop_top=drop_top,
            device=device,
        )
    elif algo_lower == "sac":
        # Standard SAC can be represented as TQC with M=2 critics, N=1 quantile each, drop_top=0
        # which performs clipped double Q-learning on scalar Q values.
        return TQCAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_critics=2,
            n_quantiles=1,
            drop_top=0,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm '{algo}'. Supported: 'tqc', 'sac'.")
