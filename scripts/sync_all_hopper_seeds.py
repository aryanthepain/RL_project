"""Synchronize all remaining Hopper-v4 seeds (44, 45, 46) from Kaggle."""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.sync_remote_artifacts import download_kernel_output, verify_and_extract_artifacts

def sync_seed(seed: int):
    kernel_id = f"sandhyasubbiah/tqc-hopper-phase09-s{seed}"
    staging_dir = repo_root / "kaggle_downloads" / f"s{seed}"
    target_dir = repo_root / "runs" / "Hopper-v4" / f"tqc_Hopper-v4_seed{seed}"
    
    print(f"\n=======================================================")
    print(f" Syncing Seed {seed} ({kernel_id})")
    print(f"=======================================================")
    
    # 1. Download
    staging_dir.mkdir(parents=True, exist_ok=True)
    success = download_kernel_output(kernel_id, staging_dir)
    if not success:
        print(f"[ERROR] Failed to download {kernel_id}")
        return False
        
    # 2. Extract & Verify
    res = verify_and_extract_artifacts(staging_dir, target_dir)
    print(f"Successfully verified and extracted Seed {seed}: {res['metrics_rows']} evaluation records found.")
    
    # If unpacked into inner tqc_Hopper-v4_seed{seed} directory, flatten it
    inner_dir = target_dir / "Hopper-v4" / f"tqc_Hopper-v4_seed{seed}"
    if inner_dir.is_dir() and (inner_dir / "metrics.csv").is_file():
        for item in inner_dir.iterdir():
            dest_item = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest_item)
        shutil.rmtree(target_dir / "Hopper-v4", ignore_errors=True)
        
    return True

if __name__ == "__main__":
    for s in [44, 45, 46]:
        sync_seed(s)
