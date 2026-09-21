"""Integration and unit tests for ComputeDispatcher and checkpoint resumption."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from src.tqc.compute.dispatcher import (
    ComputeDispatcher,
    get_preset_params,
    PRESET_CONFIGS,
    SUPPORTED_ENVS,
)
from src.tqc.compute.tier_detector import ComputeTier, TierCapability, TierDetector
from src.tqc.train import find_latest_checkpoint, safe_save, train_tqc


class TestComputeDispatcher(unittest.TestCase):
    """Test suite for ComputeDispatcher parameter resolution and execution routing."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.detector = TierDetector()
        self.dispatcher = ComputeDispatcher(detector=self.detector)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preset_params(self) -> None:
        """Test parameter resolution for standard environments and presets."""
        smoke_params = get_preset_params("smoke", "HalfCheetah-v4")
        self.assertEqual(smoke_params["timesteps"], 100)
        self.assertEqual(smoke_params["eval_freq"], 50)

        quick_params = get_preset_params("quick", "Hopper-v4")
        self.assertEqual(quick_params["timesteps"], 10_000)

        full_cheetah = get_preset_params("full", "HalfCheetah-v4")
        self.assertEqual(full_cheetah["timesteps"], 1_000_000)

        # Humanoid-v4 specific 3M requirement (D-10 / ICML 2020)
        full_humanoid = get_preset_params("full", "Humanoid-v4")
        self.assertEqual(full_humanoid["timesteps"], 3_000_000)

    def test_dry_run_dispatch(self) -> None:
        """Test dry-run mode returns valid metadata dictionary without executing."""
        meta = self.dispatcher.dispatch(
            env_id="HalfCheetah-v4",
            preset="smoke",
            tier="cpu",
            dry_run=True,
        )
        self.assertTrue(meta["dry_run"])
        self.assertEqual(meta["effective_tier"], "cpu")
        self.assertEqual(meta["params"]["timesteps"], 100)

    def test_tier_fallback_warning_when_unavailable(self) -> None:
        """Test dispatcher falls back to highest available tier when requested tier is unavailable."""
        mock_probes = {
            ComputeTier.KAGGLE: TierCapability(ComputeTier.KAGGLE, False, "Kaggle", reason="no key"),
            ComputeTier.COLAB: TierCapability(ComputeTier.COLAB, False, "Colab", reason="not in colab"),
            ComputeTier.LOCAL_GPU: TierCapability(ComputeTier.LOCAL_GPU, False, "GPU", reason="no cuda"),
            ComputeTier.LOCAL_CPU: TierCapability(ComputeTier.LOCAL_CPU, True, "CPU"),
        }
        mock_detector = MagicMock()
        mock_detector.probe_all.return_value = mock_probes
        mock_detector.resolve_highest_tier.return_value = ComputeTier.LOCAL_CPU

        dispatcher = ComputeDispatcher(detector=mock_detector)
        resolved_tier, _ = dispatcher.resolve_tier("gpu")
        self.assertEqual(resolved_tier, ComputeTier.LOCAL_CPU)

    def test_find_latest_checkpoint_and_resumption_flow(self) -> None:
        """Test find_latest_checkpoint scans and selects highest checkpoint."""
        # Create mock checkpoint files
        exp_dir = os.path.join(self.temp_dir, "exp_test")
        os.makedirs(exp_dir, exist_ok=True)

        ckpt_100 = os.path.join(exp_dir, "checkpoint_100.pt")
        ckpt_500 = os.path.join(exp_dir, "checkpoint_500.pt")
        ckpt_300 = os.path.join(exp_dir, "checkpoint_300.pt")

        torch.save({"step": 100}, ckpt_100)
        torch.save({"step": 500}, ckpt_500)
        torch.save({"step": 300}, ckpt_300)

        latest = find_latest_checkpoint(exp_dir)
        self.assertIsNotNone(latest)
        self.assertEqual(latest[0], ckpt_500)
        self.assertEqual(latest[1], 500)

    def test_training_smoke_and_resumption_integration(self) -> None:
        """Integration test: train short smoke run, save checkpoint, then resume."""
        exp_name = "test_smoke_resume_run"

        # 1. Run 30 steps initial training with checkpoint every 10 steps
        agent1 = train_tqc(
            env_id="HalfCheetah-v4",
            seed=42,
            total_timesteps=30,
            eval_freq=15,
            eval_episodes=1,
            warmup_steps=10,
            batch_size=16,
            buffer_capacity=10_000,
            checkpoint_freq=10,
            log_dir=self.temp_dir,
            exp_name=exp_name,
            device="cpu",
        )

        run_path = os.path.join(self.temp_dir, exp_name)
        self.assertTrue(os.path.exists(os.path.join(run_path, "checkpoint_10.pt")))
        self.assertTrue(os.path.exists(os.path.join(run_path, "checkpoint_30.pt")))

        # 2. Resume training from step 30 up to 40 steps
        agent2 = train_tqc(
            env_id="HalfCheetah-v4",
            seed=42,
            total_timesteps=40,
            eval_freq=20,
            eval_episodes=1,
            warmup_steps=10,
            batch_size=16,
            buffer_capacity=10_000,
            checkpoint_freq=10,
            log_dir=self.temp_dir,
            exp_name=exp_name,
            device="cpu",
            resume=True,
        )

        self.assertTrue(os.path.exists(os.path.join(run_path, "checkpoint_40.pt")))
        self.assertTrue(os.path.exists(os.path.join(run_path, "final_model.pt")))


if __name__ == "__main__":
    unittest.main()
