"""Tier detection engine and compute hierarchy resolution for TQC experiments.

Hierarchy:
  Tier 1: Kaggle Remote GPU (2x T4 / P100 via Kaggle API)
  Tier 2: Google Colab GPU (Interactive cloud execution with Drive persistence)
  Tier 3: Local CUDA GPU (Local hardware acceleration)
  Tier 4: Local CPU Fallback (Multi-threaded CPU execution)
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class ComputeTier(str, Enum):
    """Compute tier identifiers in descending order of execution priority."""

    KAGGLE = "kaggle"
    COLAB = "colab"
    LOCAL_GPU = "gpu"
    LOCAL_CPU = "cpu"


TIER_PRIORITY: List[ComputeTier] = [
    ComputeTier.KAGGLE,
    ComputeTier.COLAB,
    ComputeTier.LOCAL_GPU,
    ComputeTier.LOCAL_CPU,
]


@dataclass
class TierCapability:
    """Hardware capability and availability status of a compute tier."""

    tier: ComputeTier
    available: bool
    name: str
    details: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None


class TierDetector:
    """Probes runtime environment and hardware to detect available compute tiers."""

    def __init__(self) -> None:
        pass

    def probe_kaggle(self) -> TierCapability:
        """Probe for configured Kaggle API credentials."""
        try:
            from src.tqc.remote.kaggle_auth import KaggleAuthError, resolve_kaggle_credentials

            username, key = resolve_kaggle_credentials(prompt_if_missing=False)
            if username and key:
                masked_key = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
                return TierCapability(
                    tier=ComputeTier.KAGGLE,
                    available=True,
                    name="Kaggle Remote GPU",
                    details={
                        "username": username,
                        "key": masked_key,
                        "hardware": "2x T4 / P100 (30h/week quota)",
                    },
                )
            return TierCapability(
                tier=ComputeTier.KAGGLE,
                available=False,
                name="Kaggle Remote GPU",
                details={},
                reason="Credentials empty or incomplete",
            )
        except Exception as exc:
            return TierCapability(
                tier=ComputeTier.KAGGLE,
                available=False,
                name="Kaggle Remote GPU",
                details={},
                reason=f"Kaggle credentials not configured ({exc})",
            )

    def probe_colab(self) -> TierCapability:
        """Probe if running inside a Google Colab environment."""
        try:
            in_colab = (
                "google.colab" in sys.modules
                or os.path.exists("/content")
                or "COLAB_GPU" in os.environ
                or "COLAB_RELEASE_TAG" in os.environ
            )
            if in_colab:
                gpu_spec = os.environ.get("COLAB_GPU", "None detected")
                return TierCapability(
                    tier=ComputeTier.COLAB,
                    available=True,
                    name="Google Colab GPU",
                    details={
                        "runtime": "Google Colab",
                        "colab_gpu": gpu_spec,
                        "drive_accessible": os.path.exists("/content/drive"),
                    },
                )
            return TierCapability(
                tier=ComputeTier.COLAB,
                available=False,
                name="Google Colab GPU",
                details={},
                reason="Not running inside Google Colab environment (notebook export available)",
            )
        except Exception as exc:
            return TierCapability(
                tier=ComputeTier.COLAB,
                available=False,
                name="Google Colab GPU",
                details={},
                reason=f"Colab probe error: {exc}",
            )

    def probe_local_gpu(self) -> TierCapability:
        """Probe local CUDA GPU capability via PyTorch."""
        try:
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                dev_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024**3)
                cuda_ver = torch.version.cuda or "unknown"
                return TierCapability(
                    tier=ComputeTier.LOCAL_GPU,
                    available=True,
                    name="Local CUDA GPU",
                    details={
                        "device_count": count,
                        "device_name": dev_name,
                        "vram_gb": round(vram_gb, 2),
                        "cuda_version": cuda_ver,
                    },
                )
            return TierCapability(
                tier=ComputeTier.LOCAL_GPU,
                available=False,
                name="Local CUDA GPU",
                details={},
                reason="No CUDA-capable GPU detected by PyTorch",
            )
        except Exception as exc:
            return TierCapability(
                tier=ComputeTier.LOCAL_GPU,
                available=False,
                name="Local CUDA GPU",
                details={},
                reason=f"CUDA query error: {exc}",
            )

    def probe_local_cpu(self) -> TierCapability:
        """Probe local CPU resources (always available as the fallback tier)."""
        try:
            cores = os.cpu_count() or 1
            threads = torch.get_num_threads()
            os_name = platform.system()
            arch = platform.machine()
            return TierCapability(
                tier=ComputeTier.LOCAL_CPU,
                available=True,
                name="Local CPU Fallback",
                details={
                    "cores": cores,
                    "threads": threads,
                    "os": os_name,
                    "architecture": arch,
                },
            )
        except Exception as exc:
            return TierCapability(
                tier=ComputeTier.LOCAL_CPU,
                available=True,
                name="Local CPU Fallback",
                details={"error": str(exc)},
            )

    def probe_all(self) -> Dict[ComputeTier, TierCapability]:
        """Probe all compute tiers and return their capabilities dictionary."""
        return {
            ComputeTier.KAGGLE: self.probe_kaggle(),
            ComputeTier.COLAB: self.probe_colab(),
            ComputeTier.LOCAL_GPU: self.probe_local_gpu(),
            ComputeTier.LOCAL_CPU: self.probe_local_cpu(),
        }

    def resolve_highest_tier(
        self, probed: Optional[Dict[ComputeTier, TierCapability]] = None
    ) -> ComputeTier:
        """Resolve highest available compute tier following strict hierarchy:

        Tier 1 (Kaggle) > Tier 2 (Colab) > Tier 3 (Local GPU) > Tier 4 (Local CPU).
        """
        if probed is None:
            probed = self.probe_all()

        for tier in TIER_PRIORITY:
            cap = probed.get(tier)
            if cap and cap.available:
                return tier

        return ComputeTier.LOCAL_CPU

    def format_summary_table(
        self,
        probed: Optional[Dict[ComputeTier, TierCapability]] = None,
        selected_tier: Optional[ComputeTier] = None,
    ) -> str:
        """Format an informative ASCII summary table with badges and fallback telemetry."""
        if probed is None:
            probed = self.probe_all()
        if selected_tier is None:
            selected_tier = self.resolve_highest_tier(probed)

        lines: List[str] = []
        lines.append("=" * 78)
        lines.append("                TQC COMPUTE HIERARCHY DETECTION REPORT                ")
        lines.append("=" * 78)
        lines.append(
            f"{'Tier':<8} | {'Name':<20} | {'Status':<13} | {'Details / Fallback Reason':<30}"
        )
        lines.append("-" * 78)

        # Track any skipped higher tiers for fallback warnings
        skipped_warnings: List[str] = []
        higher_tier = True

        for tier in TIER_PRIORITY:
            cap = probed.get(tier)
            if not cap:
                continue

            if tier == selected_tier:
                badge = "[READY]"
                higher_tier = False
            elif cap.available and higher_tier:
                # Available but bypassed? (e.g. if selected_tier was forced lower)
                badge = "[AVAILABLE]"
            elif cap.available and not higher_tier:
                badge = "[SKIPPED]"
            else:
                badge = "[UNAVAILABLE]"
                if higher_tier and cap.reason:
                    skipped_warnings.append(f"  * Tier {tier.value.upper()}: {cap.reason}")

            # Format detail string
            if cap.available:
                if tier == ComputeTier.KAGGLE:
                    info = f"{cap.details.get('username')} ({cap.details.get('hardware', '')})"
                elif tier == ComputeTier.COLAB:
                    info = f"{cap.details.get('runtime', '')} (GPU: {cap.details.get('colab_gpu', '')})"
                elif tier == ComputeTier.LOCAL_GPU:
                    info = f"{cap.details.get('device_name')} ({cap.details.get('vram_gb')} GB VRAM)"
                elif tier == ComputeTier.LOCAL_CPU:
                    info = f"{cap.details.get('cores')} Cores / {cap.details.get('threads')} Threads ({cap.details.get('os')})"
                else:
                    info = str(cap.details)
            else:
                info = cap.reason or "Not available"

            # Truncate if excessively long
            if len(info) > 30:
                info = info[:27] + "..."

            tier_idx = f"T{TIER_PRIORITY.index(tier) + 1} ({tier.value})"
            lines.append(f"{tier_idx:<8} | {cap.name:<20} | {badge:<13} | {info:<30}")

        lines.append("-" * 78)
        lines.append(f"SELECTED COMPUTE TIER: {selected_tier.value.upper()} (Priority Resolution)")

        if skipped_warnings:
            lines.append("\nFallback Telemetry (Higher Tiers Bypassed):")
            for warn in skipped_warnings:
                lines.append(f"  {warn}")

        lines.append("=" * 78)
        return "\n".join(lines)
