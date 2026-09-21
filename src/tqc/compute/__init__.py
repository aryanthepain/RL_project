"""Multi-Tier Compute Subsystem for TQC Benchmark Experiments.

Hierarchy:
  Tier 1: Kaggle (2x T4 / P100 remote execution via Kaggle API)
  Tier 2: Colab (Interactive remote GPU execution with Drive persistence)
  Tier 3: Local GPU (Local CUDA acceleration)
  Tier 4: Local CPU (Local CPU thread pool fallback)
"""

from src.tqc.compute.dispatcher import ComputeDispatcher, PRESET_CONFIGS, SUPPORTED_ENVS
from src.tqc.compute.tier_detector import (
    ComputeTier,
    TIER_PRIORITY,
    TierCapability,
    TierDetector,
)

__all__ = [
    "ComputeTier",
    "TIER_PRIORITY",
    "TierCapability",
    "TierDetector",
    "ComputeDispatcher",
    "PRESET_CONFIGS",
    "SUPPORTED_ENVS",
]
