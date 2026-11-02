"""Robust Kaggle kernel output downloader with per-file retry and resumption."""

import os
import sys
import time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

def download_with_retries(kernel_id: str, target_dir: str, max_retries: int = 5):
    api = KaggleApi()
    api.authenticate()
    owner_slug, kernel_slug, _ = api.parse_kernel_string(kernel_id)
    
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Connecting to Kaggle API to list files for {kernel_id}...")
    with api.build_kaggle_client() as kaggle:
        from kaggle.models import ApiListKernelSessionOutputRequest
        token = None
        all_items = []
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner_slug
            req.kernel_slug = kernel_slug
            api._set_paging(req, 100, token)
            res = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            if res.files:
                all_items.extend(res.files)
            token = res.next_page_token
            if not token:
                break
                
    print(f"Found {len(all_items)} files in kernel output.")
    
    # Sort files: prioritize metrics.csv, metrics.jsonl, and best_model.pt
    def priority(item):
        name = item.file_name.lower()
        if "metrics.csv" in name:
            return 0
        if "metrics.jsonl" in name:
            return 1
        if "best_model.pt" in name:
            return 2
        return 10
        
    all_items.sort(key=priority)
    
    for idx, item in enumerate(all_items, 1):
        outfile = os.path.join(target_dir, item.file_name)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        
        # Check if already fully downloaded
        if os.path.isfile(outfile) and os.path.getsize(outfile) > 0:
            print(f"[{idx}/{len(all_items)}] Already downloaded: {item.file_name} ({os.path.getsize(outfile):,} bytes)")
            continue
            
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[{idx}/{len(all_items)}] Downloading {item.file_name} (Attempt {attempt}/{max_retries})...")
                resp = requests.get(item.url, stream=True, timeout=120)
                resp.raise_for_status()
                with open(outfile, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                if os.path.getsize(outfile) > 0:
                    print(f" -> Completed: {item.file_name} ({os.path.getsize(outfile):,} bytes)")
                    success = True
                    break
            except Exception as e:
                print(f" -> Attempt {attempt} failed: {e}")
                time.sleep(2 * attempt)
                if os.path.isfile(outfile):
                    try:
                        os.remove(outfile)
                    except OSError:
                        pass
                        
        if not success:
            print(f"WARNING: Could not download {item.file_name} after {max_retries} attempts.")

    print("\nDownload process completed.")

if __name__ == "__main__":
    kernel = sys.argv[1] if len(sys.argv) > 1 else "sharvanipallempati/tqc-humanoidv4-s46-10261531"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"kaggle_downloads\sharvanipallempati_tqc-humanoidv4-s46-10261531"
    download_with_retries(kernel, out_dir)
