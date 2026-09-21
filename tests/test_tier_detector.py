"""Unit tests for TierDetector and compute hierarchy resolution."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from src.tqc.compute.tier_detector import (
    ComputeTier,
    TierCapability,
    TierDetector,
    TIER_PRIORITY,
)


class TestTierDetector(unittest.TestCase):
    """Test suite for TierDetector probe methods, resolution, and formatting."""

    def setUp(self) -> None:
        self.detector = TierDetector()

    def test_tier_priority_order(self) -> None:
        """Verify strict priority ordering (Kaggle > Colab > GPU > CPU)."""
        expected = [
            ComputeTier.KAGGLE,
            ComputeTier.COLAB,
            ComputeTier.LOCAL_GPU,
            ComputeTier.LOCAL_CPU,
        ]
        self.assertEqual(TIER_PRIORITY, expected)

    @patch("src.tqc.remote.kaggle_auth.resolve_kaggle_credentials")
    def test_probe_kaggle_available(self, mock_creds: MagicMock) -> None:
        """Test probe_kaggle when valid credentials exist."""
        mock_creds.return_value = ("testuser", "testkey12345")
        cap = self.detector.probe_kaggle()
        self.assertEqual(cap.tier, ComputeTier.KAGGLE)
        self.assertTrue(cap.available)
        self.assertEqual(cap.details["username"], "testuser")
        self.assertIn("key", cap.details)

    @patch("src.tqc.remote.kaggle_auth.resolve_kaggle_credentials")
    def test_probe_kaggle_unavailable(self, mock_creds: MagicMock) -> None:
        """Test probe_kaggle when credentials are missing or raise error."""
        from src.tqc.remote.kaggle_auth import KaggleAuthError

        mock_creds.side_effect = KaggleAuthError("Missing kaggle.json")
        cap = self.detector.probe_kaggle()
        self.assertEqual(cap.tier, ComputeTier.KAGGLE)
        self.assertFalse(cap.available)
        self.assertIn("not configured", cap.reason.lower())

    def test_probe_colab(self) -> None:
        """Test probe_colab runtime detection."""
        cap = self.detector.probe_colab()
        self.assertEqual(cap.tier, ComputeTier.COLAB)
        # Outside colab by default in local test
        self.assertIsInstance(cap.available, bool)

    @patch("torch.cuda.is_available")
    def test_probe_local_gpu_available(self, mock_cuda: MagicMock) -> None:
        """Test probe_local_gpu when CUDA is available."""
        mock_cuda.return_value = True
        with patch("torch.cuda.device_count", return_value=1), \
             patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"), \
             patch("torch.cuda.get_device_properties") as mock_props:
            mock_prop_obj = MagicMock()
            mock_prop_obj.total_memory = 24 * (1024**3)
            mock_props.return_value = mock_prop_obj

            cap = self.detector.probe_local_gpu()
            self.assertEqual(cap.tier, ComputeTier.LOCAL_GPU)
            self.assertTrue(cap.available)
            self.assertEqual(cap.details["device_name"], "NVIDIA RTX 4090")
            self.assertEqual(cap.details["vram_gb"], 24.0)

    @patch("torch.cuda.is_available")
    def test_probe_local_gpu_unavailable(self, mock_cuda: MagicMock) -> None:
        """Test probe_local_gpu when CUDA is unavailable."""
        mock_cuda.return_value = False
        cap = self.detector.probe_local_gpu()
        self.assertEqual(cap.tier, ComputeTier.LOCAL_GPU)
        self.assertFalse(cap.available)
        self.assertIn("No CUDA-capable GPU", cap.reason)

    def test_probe_local_cpu(self) -> None:
        """Test probe_local_cpu is always available as fallback."""
        cap = self.detector.probe_local_cpu()
        self.assertEqual(cap.tier, ComputeTier.LOCAL_CPU)
        self.assertTrue(cap.available)
        self.assertIn("cores", cap.details)
        self.assertIn("threads", cap.details)

    def test_resolve_highest_tier_hierarchy(self) -> None:
        """Test hierarchy resolution across varied availability combinations."""
        # 1. All available -> Kaggle wins
        probes_all = {
            ComputeTier.KAGGLE: TierCapability(ComputeTier.KAGGLE, True, "Kaggle"),
            ComputeTier.COLAB: TierCapability(ComputeTier.COLAB, True, "Colab"),
            ComputeTier.LOCAL_GPU: TierCapability(ComputeTier.LOCAL_GPU, True, "GPU"),
            ComputeTier.LOCAL_CPU: TierCapability(ComputeTier.LOCAL_CPU, True, "CPU"),
        }
        self.assertEqual(self.detector.resolve_highest_tier(probes_all), ComputeTier.KAGGLE)

        # 2. Kaggle missing -> Colab wins
        probes_colab = {
            ComputeTier.KAGGLE: TierCapability(ComputeTier.KAGGLE, False, "Kaggle", reason="no creds"),
            ComputeTier.COLAB: TierCapability(ComputeTier.COLAB, True, "Colab"),
            ComputeTier.LOCAL_GPU: TierCapability(ComputeTier.LOCAL_GPU, True, "GPU"),
            ComputeTier.LOCAL_CPU: TierCapability(ComputeTier.LOCAL_CPU, True, "CPU"),
        }
        self.assertEqual(self.detector.resolve_highest_tier(probes_colab), ComputeTier.COLAB)

        # 3. Kaggle & Colab missing -> GPU wins
        probes_gpu = {
            ComputeTier.KAGGLE: TierCapability(ComputeTier.KAGGLE, False, "Kaggle", reason="no creds"),
            ComputeTier.COLAB: TierCapability(ComputeTier.COLAB, False, "Colab", reason="no colab"),
            ComputeTier.LOCAL_GPU: TierCapability(ComputeTier.LOCAL_GPU, True, "GPU"),
            ComputeTier.LOCAL_CPU: TierCapability(ComputeTier.LOCAL_CPU, True, "CPU"),
        }
        self.assertEqual(self.detector.resolve_highest_tier(probes_gpu), ComputeTier.LOCAL_GPU)

        # 4. Only CPU available -> CPU fallback
        probes_cpu = {
            ComputeTier.KAGGLE: TierCapability(ComputeTier.KAGGLE, False, "Kaggle", reason="no creds"),
            ComputeTier.COLAB: TierCapability(ComputeTier.COLAB, False, "Colab", reason="no colab"),
            ComputeTier.LOCAL_GPU: TierCapability(ComputeTier.LOCAL_GPU, False, "GPU", reason="no cuda"),
            ComputeTier.LOCAL_CPU: TierCapability(ComputeTier.LOCAL_CPU, True, "CPU"),
        }
        self.assertEqual(self.detector.resolve_highest_tier(probes_cpu), ComputeTier.LOCAL_CPU)

    def test_format_summary_table(self) -> None:
        """Test summary table rendering and status badges."""
        probes = {
            ComputeTier.KAGGLE: TierCapability(ComputeTier.KAGGLE, False, "Kaggle", reason="No key"),
            ComputeTier.COLAB: TierCapability(ComputeTier.COLAB, False, "Colab", reason="Not in colab"),
            ComputeTier.LOCAL_GPU: TierCapability(ComputeTier.LOCAL_GPU, True, "Local GPU", details={"device_name": "RTX 3080", "vram_gb": 10.0}),
            ComputeTier.LOCAL_CPU: TierCapability(ComputeTier.LOCAL_CPU, True, "Local CPU", details={"cores": 8, "threads": 8, "os": "Linux"}),
        }
        table = self.detector.format_summary_table(probed=probes, selected_tier=ComputeTier.LOCAL_GPU)
        self.assertIn("TQC COMPUTE HIERARCHY DETECTION REPORT", table)
        self.assertIn("[READY]", table)
        self.assertIn("[UNAVAILABLE]", table)
        self.assertIn("[SKIPPED]", table)
        self.assertIn("SELECTED COMPUTE TIER: GPU", table)
        self.assertIn("Fallback Telemetry", table)


if __name__ == "__main__":
    unittest.main()
