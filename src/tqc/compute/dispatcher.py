"""Compute dispatcher for executing TQC benchmark workloads across compute tiers."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

from src.tqc.compute.tier_detector import ComputeTier, TierCapability, TierDetector
from src.tqc.train import train_tqc

SUPPORTED_ENVS = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "timesteps": 100,
        "eval_freq": 50,
        "eval_episodes": 2,
        "warmup_steps": 20,
        "batch_size": 32,
        "checkpoint_freq": 50,
        "buffer_capacity": 10_000,
    },
    "quick": {
        "timesteps": 10_000,
        "eval_freq": 2_000,
        "eval_episodes": 5,
        "warmup_steps": 1_000,
        "batch_size": 256,
        "checkpoint_freq": 5_000,
        "buffer_capacity": 100_000,
    },
    "full": {
        "timesteps": 1_000_000,
        "eval_freq": 5_000,
        "eval_episodes": 10,
        "warmup_steps": 10_000,
        "batch_size": 256,
        "checkpoint_freq": 50_000,
        "buffer_capacity": 1_000_000,
    },
}


def get_preset_params(preset: str, env_id: str) -> Dict[str, Any]:
    """Resolve training parameters for a preset, adapting for env requirements."""
    if preset not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset '{preset}'. Supported presets: {list(PRESET_CONFIGS.keys())}"
        )

    params = dict(PRESET_CONFIGS[preset])
    if preset == "full" and env_id == "Humanoid-v4":
        # Humanoid requires 3M steps in Kuznetsov et al. (ICML 2020)
        params["timesteps"] = 3_000_000
        params["eval_freq"] = 10_000

    return params


class ComputeDispatcher:
    """Dispatches TQC experiment runs across the compute hierarchy."""

    def __init__(self, detector: Optional[TierDetector] = None) -> None:
        self.detector = detector or TierDetector()

    def resolve_tier(
        self, requested_tier: Optional[Union[ComputeTier, str]] = None
    ) -> Tuple[ComputeTier, Dict[ComputeTier, TierCapability]]:
        """Resolve the effective compute tier, returning tier and capability probes."""
        probes = self.detector.probe_all()

        if isinstance(requested_tier, ComputeTier):
            matched_tier = requested_tier
        elif requested_tier is None or str(requested_tier).lower() == "auto":
            return self.detector.resolve_highest_tier(probes), probes
        else:
            target_str = str(requested_tier).lower()
            matched_tier = None
            for tier in ComputeTier:
                if tier.value == target_str or tier.name.lower() == target_str:
                    matched_tier = tier
                    break

        if matched_tier is None:
            raise ValueError(
                f"Unknown compute tier '{requested_tier}'. Choices: auto, kaggle, colab, gpu, cpu"
            )

        cap = probes.get(matched_tier)
        if not cap or not cap.available:
            print(f"\n[WARNING] Requested tier '{matched_tier.value}' is not available:")
            print(f"  Reason: {cap.reason if cap else 'Unknown'}")
            highest = self.detector.resolve_highest_tier(probes)
            print(f"  Falling back to highest available tier: '{highest.value}'\n")
            return highest, probes

        return matched_tier, probes

    def dispatch(
        self,
        env_id: str = "HalfCheetah-v4",
        preset: str = "quick",
        tier: Optional[Union[ComputeTier, str]] = "auto",
        seed: int = 42,
        resume: bool = True,
        dry_run: bool = False,
        interactive: bool = False,
        log_dir: str = "runs",
        exp_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Dispatch experiment to the appropriate tier."""
        if env_id not in SUPPORTED_ENVS:
            raise ValueError(
                f"Unsupported environment '{env_id}'. Supported: {SUPPORTED_ENVS}"
            )

        effective_tier, probes = self.resolve_tier(tier)
        params = get_preset_params(preset, env_id)

        dispatch_meta = {
            "env_id": env_id,
            "preset": preset,
            "effective_tier": effective_tier.value,
            "seed": seed,
            "resume": resume,
            "params": params,
            "dry_run": dry_run,
        }

        if dry_run:
            print("\n[DRY RUN] Workload Dispatch Plan:")
            print(f"  Environment : {env_id}")
            print(f"  Preset      : {preset} ({params['timesteps']:,} steps)")
            print(f"  Tier        : {effective_tier.value.upper()}")
            print(f"  Seed        : {seed}")
            print(f"  Auto-Resume : {resume}")
            print(f"  Parameters  : {params}")
            return dispatch_meta

        # Dispatch execution by tier
        if effective_tier == ComputeTier.KAGGLE:
            print(f"\n[DISPATCH] Launching on Tier 1 (Kaggle Remote GPU)...")
            cmd = [
                sys.executable,
                "scripts/run_remote_experiment.py",
                "--preset",
                preset,
                "--seed",
                str(seed),
            ]
            print(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd)
            dispatch_meta["returncode"] = res.returncode
            return dispatch_meta

        elif effective_tier == ComputeTier.COLAB:
            print(f"\n[DISPATCH] Tier 2 (Google Colab):")
            cap = probes.get(ComputeTier.COLAB)
            if cap and cap.available:
                # Inside Colab execution
                device = "cuda" if torch.cuda.is_available() else "cpu"
                drive_dir = "/content/drive/MyDrive/tqc_runs"
                target_log_dir = drive_dir if os.path.exists(drive_dir) else log_dir
                print(f"Running inside Colab session on device={device}, log_dir={target_log_dir}")
                agent = train_tqc(
                    env_id=env_id,
                    seed=seed,
                    total_timesteps=params["timesteps"],
                    eval_freq=params["eval_freq"],
                    eval_episodes=params["eval_episodes"],
                    warmup_steps=params["warmup_steps"],
                    batch_size=params["batch_size"],
                    buffer_capacity=params["buffer_capacity"],
                    device=device,
                    log_dir=target_log_dir,
                    exp_name=exp_name,
                    checkpoint_freq=params["checkpoint_freq"],
                    resume=resume,
                )
                dispatch_meta["agent"] = agent
                return dispatch_meta
            else:
                msg = (
                    "To execute on Google Colab:\n"
                    "  1. Open notebooks/colab_tqc_benchmark.ipynb in Google Colab.\n"
                    "  2. Enable GPU hardware accelerator (T4 / A100).\n"
                    "  3. Mount Google Drive and run all cells."
                )
                print(msg)
                dispatch_meta["colab_instructions"] = msg
                return dispatch_meta

        elif effective_tier == ComputeTier.LOCAL_GPU:
            print(f"\n[DISPATCH] Launching on Tier 3 (Local CUDA GPU)...")
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
            agent = train_tqc(
                env_id=env_id,
                seed=seed,
                total_timesteps=params["timesteps"],
                eval_freq=params["eval_freq"],
                eval_episodes=params["eval_episodes"],
                warmup_steps=params["warmup_steps"],
                batch_size=params["batch_size"],
                buffer_capacity=params["buffer_capacity"],
                device="cuda",
                log_dir=log_dir,
                exp_name=exp_name,
                checkpoint_freq=params["checkpoint_freq"],
                resume=resume,
            )
            dispatch_meta["agent"] = agent
            return dispatch_meta

        else:  # ComputeTier.LOCAL_CPU
            print(f"\n[DISPATCH] Launching on Tier 4 (Local CPU Fallback)...")
            cpu_buffer = min(params["buffer_capacity"], 200_000)
            agent = train_tqc(
                env_id=env_id,
                seed=seed,
                total_timesteps=params["timesteps"],
                eval_freq=params["eval_freq"],
                eval_episodes=params["eval_episodes"],
                warmup_steps=params["warmup_steps"],
                batch_size=params["batch_size"],
                buffer_capacity=cpu_buffer,
                device="cpu",
                log_dir=log_dir,
                exp_name=exp_name,
                checkpoint_freq=params["checkpoint_freq"],
                resume=resume,
            )
            dispatch_meta["agent"] = agent
            return dispatch_meta
