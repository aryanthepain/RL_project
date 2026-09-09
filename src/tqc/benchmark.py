import argparse
from typing import Dict, List, Optional, Union
import torch
from .envs import BENCHMARK_ENVS, get_env_metadata
from .train import train_tqc
from .utils import get_device


DEFAULT_BENCHMARK_ENVS = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]


def run_benchmark(
    env_ids: Optional[List[str]] = None,
    algos: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    total_timesteps: Optional[int] = None,
    eval_freq: int = 5_000,
    eval_episodes: int = 10,
    warmup_steps: int = 10_000,
    batch_size: int = 256,
    fast_test: bool = False,
    device: Union[str, torch.device] = "cpu",
    log_dir: str = "runs",
) -> List[Dict[str, Union[str, int, float]]]:
    """Orchestrate training runs across multiple environments, algorithms, and random seeds.

    Args:
        env_ids: List of environment IDs (default: full continuous locomotion suite).
        algos: List of algorithms, e.g. ['tqc', 'sac'].
        seeds: List of random seeds to evaluate.
        total_timesteps: Timesteps per run. If None, loaded from env metadata.
        eval_freq: Evaluation frequency.
        eval_episodes: Number of episodes per evaluation.
        warmup_steps: Warmup exploration steps before updates.
        batch_size: Mini-batch size.
        fast_test: If True, overrides steps to run rapid test iterations.
        device: Torch compute device.
        log_dir: Directory to store run outputs.

    Returns:
        List of completed run summaries.
    """
    if env_ids is None:
        env_ids = DEFAULT_BENCHMARK_ENVS
    if algos is None:
        algos = ["tqc", "sac"]
    if seeds is None:
        seeds = [0, 1, 2]

    dev = get_device(device)
    results = []

    for env_id in env_ids:
        meta = get_env_metadata(env_id)
        run_timesteps = (
            30 if fast_test else (total_timesteps or meta["total_timesteps"])
        )
        run_warmup = 10 if fast_test else warmup_steps
        run_eval_freq = 15 if fast_test else eval_freq
        run_eval_episodes = 1 if fast_test else eval_episodes
        run_batch_size = 4 if fast_test else batch_size

        for algo in algos:
            for seed in seeds:
                exp_name = f"{algo}_{env_id}_seed{seed}"
                print(
                    f"\n>>> Running Benchmark: {algo.upper()} on {env_id} | Seed: {seed} | Steps: {run_timesteps}"
                )

                agent = train_tqc(
                    env_id=env_id,
                    seed=seed,
                    total_timesteps=run_timesteps,
                    eval_freq=run_eval_freq,
                    eval_episodes=run_eval_episodes,
                    warmup_steps=run_warmup,
                    batch_size=run_batch_size,
                    algo=algo,
                    device=dev,
                    log_dir=log_dir,
                    exp_name=exp_name,
                    checkpoint_freq=0 if fast_test else 50_000,
                )

                results.append(
                    {
                        "env_id": env_id,
                        "algo": algo,
                        "seed": seed,
                        "exp_name": exp_name,
                        "timesteps": run_timesteps,
                    }
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="TQC Multi-Seed Benchmark Suite Orchestrator")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=DEFAULT_BENCHMARK_ENVS,
        help="List of environments to run",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["tqc", "sac"],
        choices=["tqc", "sac"],
        help="Algorithms to benchmark",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Override timesteps per run")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    parser.add_argument("--log-dir", type=str, default="runs", help="Output directory")
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Fast testing mode running 30 timesteps per run",
    )

    args = parser.parse_args()
    run_benchmark(
        env_ids=args.envs,
        algos=args.algos,
        seeds=args.seeds,
        total_timesteps=args.timesteps,
        fast_test=args.fast_test,
        device=args.device,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
