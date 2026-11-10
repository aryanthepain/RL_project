"""Helper to build standalone Kaggle notebook bundles for Phase 11 (Ant-v4)."""

import json
import os
import shutil
from pathlib import Path
from scripts.build_kaggle_notebooks import create_kaggle_benchmark_notebook

def build_seed_bundle(seed: int, env_id: str = "Ant-v4", phase_num: str = "11"):
    repo_root = Path(__file__).resolve().parent.parent
    build_dir = repo_root / f".build_nb_ant_s{seed}"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    nb_path = build_dir / f"kaggle_tqc_ant_phase11.ipynb"
    create_kaggle_benchmark_notebook(
        env_id=env_id,
        phase_num=phase_num,
        target_score="6,293 ± 276 (ICML 2020 Table 1) / 8.01 ± 0.87k (Table 2)",
        output_path=str(nb_path),
    )
    
    # Adjust Cell 6 to execute the specific seed
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "run_ant_benchmark.py" in src:
            new_src = []
            for line in cell["source"]:
                if line.strip().startswith("--seeds 42"):
                    new_src.append(f"    --seeds {seed} \\\n")
                else:
                    new_src.append(line)
            cell["source"] = new_src
            break
            
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    meta = {
        "id": f"sandhyasubbiah/tqc-ant-phase11-s{seed}",
        "title": f"tqc-ant-phase11-s{seed}",
        "code_file": "kaggle_tqc_ant_phase11.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(build_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Successfully configured bundle for Ant-v4 Seed {seed} in {build_dir}")

if __name__ == "__main__":
    for s in [42, 43, 44, 45, 46]:
        build_seed_bundle(s)
