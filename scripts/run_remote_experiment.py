"""Unified runner for remote TQC benchmark experiments across compute tiers."""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, Dict, Optional
import yaml

from src.tqc.remote.kaggle_auth import (
    KaggleAuthError,
    ensure_kaggle_cli_installed,
    resolve_kaggle_credentials,
    validate_kaggle_credentials,
)
from src.tqc.remote.kaggle_packager import (
    KagglePackagingError,
    build_kernel_metadata,
    create_remote_entrypoint,
    package_kernel_directory,
    run_local_preflight_smoke,
)
from src.tqc.remote.queue_manager import DualSlotQueueManager, RemoteJob
from scripts.sync_remote_artifacts import (
    download_kernel_output,
    print_downstream_triggers,
    verify_and_extract_artifacts,
)


def load_matrix_config(config_path: str = "configs/benchmark_matrix.yaml") -> Dict[str, Any]:
    """Load declarative experiment matrix YAML configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_experiment(
    preset_name: str = "pilot",
    config_path: str = "configs/benchmark_matrix.yaml",
    env_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    steps_override: Optional[int] = None,
    device_override: Optional[str] = None,
    dry_run: bool = False,
    skip_smoke: bool = False,
    detach: bool = False,
    poll_interval: int = 30,
    build_dir: str = ".build_kernel",
    sync_only: Optional[str] = None,
) -> None:
    """Execute complete remote experiment lifecycle.

    Workflow:
        1. (Optional) Sync-only mode for existing kernel.
        2. Credential resolution & validation.
        3. Automated pre-flight local smoke verification (100 steps).
        4. Parameter resolution & kernel script generation.
        5. Kernel bundling (kernel-metadata.json, remote_entrypoint.py, tqc_source.tar.gz).
        6. Queue submission & background monitoring or detached handoff.
        7. Artifact synchronization & SHA256 integrity verification upon completion.
    """
    if sync_only:
        print(f"\n[Sync Only Mode] Synchronizing artifacts for kernel: {sync_only}")
        staging_dir = Path("kaggle_downloads") / sync_only.replace("/", "_")
        if download_kernel_output(sync_only, staging_dir):
            target_dir = f"runs/kaggle_{sync_only.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            verify_and_extract_artifacts(staging_dir, target_dir)
            print_downstream_triggers(target_dir)
        return

    print("=" * 70)
    print(" TQC Remote Benchmark Experiment Runner")
    print(" Kuznetsov et al. (ICML 2020) Reproduction Pipeline")
    print("=" * 70)

    # 1. Resolve Kaggle Credentials
    print("\n[Step 1/6] Resolving Kaggle Credentials...")
    try:
        username, api_key = resolve_kaggle_credentials(prompt_if_missing=True)
        print(f"Credentials resolved for user: {username}")
    except KaggleAuthError as e:
        print(f"[Error] Failed to resolve credentials: {e}")
        sys.exit(1)

    if not ensure_kaggle_cli_installed():
        print("[Error] Kaggle CLI is not installed or available in PATH. Please run 'pip install kaggle'.")
        sys.exit(1)

    # 2. Resolve Experiment Parameters
    print("\n[Step 2/6] Loading Experiment Configuration...")
    matrix_cfg = load_matrix_config(config_path)
    presets = matrix_cfg.get("presets", {})
    if preset_name not in presets and preset_name != "custom":
        print(f"[Error] Preset '{preset_name}' not defined in {config_path}. Available: {list(presets.keys())}")
        sys.exit(1)

    preset_cfg = presets.get(preset_name, {})
    env_id = env_override or preset_cfg.get("env_id", "HalfCheetah-v4")
    seed = seed_override if seed_override is not None else preset_cfg.get("seed", 42)
    steps = steps_override or preset_cfg.get("total_timesteps", 1_000_000)
    device = device_override or preset_cfg.get("device", "cuda")
    ckpt_freq = preset_cfg.get("checkpoint_freq", 50_000)
    eval_freq = preset_cfg.get("eval_freq", 5_000)
    eval_episodes = preset_cfg.get("eval_episodes", 10)
    warmup_steps = preset_cfg.get("warmup_steps", 10_000)
    batch_size = preset_cfg.get("batch_size", 256)
    buffer_capacity = preset_cfg.get("buffer_capacity", 1_000_000)
    exp_name = preset_cfg.get("exp_name", f"kaggle_{env_id.lower()}_s{seed}")

    print(f"Target Preset:        {preset_name}")
    print(f"Environment:          {env_id}")
    print(f"Random Seed:          {seed}")
    print(f"Total Steps:          {steps:,}")
    print(f"Checkpoint Freq:      {ckpt_freq:,}")
    print(f"Eval Freq:            {eval_freq:,}")
    print(f"Batch Size:           {batch_size}")
    print(f"Replay Buffer:        {buffer_capacity:,}")
    print(f"Compute Device:       {device}")

    # 3. Pre-flight Local Smoke Test (100 steps)
    if not skip_smoke and not dry_run:
        print("\n[Step 3/6] Running 100-step Local Pre-Flight Smoke Test (D-13)...")
        try:
            run_local_preflight_smoke(env_id=env_id, steps=100, device="cpu", seed=seed)
            print("Local pre-flight smoke test PASSED successfully!")
        except KagglePackagingError as e:
            print(f"[Error] Pre-flight smoke validation failed: {e}")
            print("Aborting remote dispatch to prevent quota consumption.")
            sys.exit(1)
    else:
        print("\n[Step 3/6] Skipping local pre-flight smoke test.")

    # 4. Kernel Code & Metadata Generation
    print("\n[Step 4/6] Generating Remote Kernel Code & Metadata...")
    clean_env = env_id.lower().replace("-", "").replace("_", "")
    kernel_slug = f"tqc-{clean_env}-s{seed}-{datetime.now().strftime('%m%d%H%M')}"
    kernel_title = kernel_slug

    metadata = build_kernel_metadata(
        username=username,
        kernel_slug=kernel_slug,
        title=kernel_title,
        code_file="remote_entrypoint.py",
        enable_gpu=True,
        enable_internet=True,
        is_private=True,
    )

    entrypoint_code = create_remote_entrypoint(
        env_id=env_id,
        seed=seed,
        total_timesteps=steps,
        checkpoint_freq=ckpt_freq,
        eval_freq=eval_freq,
        exp_name=exp_name,
        device=device,
        buffer_size=buffer_capacity,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        eval_episodes=eval_episodes,
    )

    # 5. Kernel Packaging
    print("\n[Step 5/6] Packaging Kernel Payload...")
    target_build = Path(build_dir).resolve()
    packaged_path = package_kernel_directory(
        target_dir=target_build,
        metadata=metadata,
        entrypoint_code=entrypoint_code,
    )
    print(f"Kernel packaged successfully in: {packaged_path}")

    if dry_run:
        print("\n[Dry Run] Packaging complete. Files generated:")
        for p in target_build.iterdir():
            print(f" - {p.name} ({p.stat().st_size:,} bytes)")
        print("\nSkipping remote push (--dry-run).")
        return

    # 6. Queue Submission & Monitoring
    print("\n[Step 6/6] Submitting Job to Kaggle Queue...")
    queue_mgr = DualSlotQueueManager(
        max_concurrent_slots=matrix_cfg.get("max_concurrent_gpu_slots", 2),
    )
    kernel_id = f"{username}/{kernel_slug}"
    job = queue_mgr.add_job(
        job_id=f"{env_id}_s{seed}",
        kernel_dir=packaged_path,
        env_id=env_id,
        seed=seed,
        kernel_id=kernel_id,
    )

    if detach:
        print("\n[Detached Mode] Pushing kernel and detaching...")
        queue_mgr.step_queue()
        print("\n" + "=" * 70)
        print(" Kernel Dispatched in Detached Mode")
        print("=" * 70)
        print(f"Kernel ID:    {kernel_id}")
        print("\nTo inspect live kernel status:")
        print(f"  kaggle kernels status {kernel_id}")
        print("\nTo pull artifacts when completed:")
        print(f"  python scripts/sync_remote_artifacts.py --kernel-id {kernel_id}")
        print("=" * 70 + "\n")
        return

    print(f"Dispatching and monitoring kernel: {kernel_id}")

    def on_complete(completed_job: RemoteJob):
        print(f"\n[Job Complete] Syncing artifacts for: {completed_job.kernel_id}")
        staging = Path("kaggle_downloads") / completed_job.kernel_id.replace("/", "_")
        if download_kernel_output(completed_job.kernel_id, staging):
            target_dir = f"runs/kaggle_{clean_env}_s{completed_job.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            verify_and_extract_artifacts(staging, target_dir)
            print_downstream_triggers(target_dir)

    queue_mgr.step_queue()
    queue_mgr.monitor_until_completion(
        poll_interval=poll_interval,
        on_job_complete=on_complete,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run TQC benchmark experiments on remote Kaggle GPU compute tier."
    )
    parser.add_argument("--preset", type=str, default="pilot", help="Named preset from configs/benchmark_matrix.yaml (default: pilot)")
    parser.add_argument("--config", type=str, default="configs/benchmark_matrix.yaml", help="Path to matrix configuration YAML")
    parser.add_argument("--env", type=str, help="Override environment ID (e.g. HalfCheetah-v4)")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--steps", type=int, help="Override total interaction steps")
    parser.add_argument("--device", type=str, help="Override compute device (cuda or cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Package kernel directory without pushing to Kaggle")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip 100-step pre-flight local smoke test")
    parser.add_argument("--detach", action="store_true", help="Submit kernel to Kaggle and exit immediately without blocking")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds (default: 30)")
    parser.add_argument("--build-dir", type=str, default=".build_kernel", help="Kernel build output directory")
    parser.add_argument("--sync-only", type=str, help="Directly sync and verify an existing kernel ID without training")

    args = parser.parse_args()

    run_experiment(
        preset_name=args.preset,
        config_path=args.config,
        env_override=args.env,
        seed_override=args.seed,
        steps_override=args.steps,
        device_override=args.device,
        dry_run=args.dry_run,
        skip_smoke=args.skip_smoke,
        detach=args.detach,
        poll_interval=args.poll_interval,
        build_dir=args.build_dir,
        sync_only=args.sync_only,
    )


if __name__ == "__main__":
    main()
