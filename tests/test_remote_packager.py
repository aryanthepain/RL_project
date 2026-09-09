"""Unit tests for Kaggle packager, remote entrypoint builder, and pre-flight smoke validator."""

import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tqc.remote.kaggle_packager import (
    KagglePackagingError,
    build_kernel_metadata,
    create_remote_entrypoint,
    package_kernel_directory,
    run_local_preflight_smoke,
)


class TestRemotePackager(unittest.TestCase):
    """Test suite for Kaggle packaging, remote script generation, and preflight smoke tests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_kernel_metadata_valid(self):
        """Test building valid kernel-metadata.json payload."""
        metadata = build_kernel_metadata(
            username="test_researcher",
            kernel_slug="tqc-halfcheetah-pilot",
            title="TQC HalfCheetah Pilot Run",
            code_file="run_entry.py",
            enable_gpu=True,
            enable_internet=True,
            is_private=True,
        )

        self.assertEqual(metadata["id"], "test_researcher/tqc-halfcheetah-pilot")
        self.assertEqual(metadata["title"], "TQC HalfCheetah Pilot Run")
        self.assertEqual(metadata["code_file"], "run_entry.py")
        self.assertEqual(metadata["language"], "python")
        self.assertEqual(metadata["kernel_type"], "script")
        self.assertEqual(metadata["enable_gpu"], "true")
        self.assertEqual(metadata["enable_internet"], "true")
        self.assertEqual(metadata["is_private"], "true")

    def test_build_kernel_metadata_missing_args(self):
        """Test metadata builder raises error on empty required fields."""
        with self.assertRaises(KagglePackagingError):
            build_kernel_metadata("", "slug", "title")
        with self.assertRaises(KagglePackagingError):
            build_kernel_metadata("user", "", "title")
        with self.assertRaises(KagglePackagingError):
            build_kernel_metadata("user", "slug", "")

    def test_create_remote_entrypoint(self):
        """Test generated entrypoint contains dependencies, CUDA setup, and artifact bundling."""
        code = create_remote_entrypoint(
            env_id="HalfCheetah-v4",
            seed=42,
            total_timesteps=1_000_000,
            checkpoint_freq=50_000,
            eval_freq=5_000,
            exp_name="test_pilot",
            device="cuda",
            buffer_size=1_000_000,
            batch_size=256,
        )

        self.assertIn("gymnasium[mujoco]", code)
        self.assertIn("tqc_source.tar.gz", code)
        self.assertIn("torch.backends.cudnn.benchmark = True", code)
        self.assertIn("train_tqc(", code)
        self.assertIn('env_id="HalfCheetah-v4"', code)
        self.assertIn("seed=42", code)
        self.assertIn("total_timesteps=1000000", code)
        self.assertIn("artifacts.tar.gz", code)
        self.assertIn("checksums.sha256", code)

    def test_package_kernel_directory(self):
        """Test package_kernel_directory bundles metadata, entrypoint, and tqc_source.tar.gz."""
        # Create a mock source directory
        fake_root = Path(self.temp_dir) / "fake_root"
        fake_src = fake_root / "src" / "tqc"
        fake_src.mkdir(parents=True)
        (fake_src / "dummy.py").write_text("# dummy code", encoding="utf-8")

        target_dir = Path(self.temp_dir) / "kernel_bundle"
        metadata = build_kernel_metadata(
            username="test_user",
            kernel_slug="test-slug",
            title="Test Kernel",
        )
        entrypoint_code = "print('Hello remote!')"

        out_path = package_kernel_directory(
            target_dir=target_dir,
            metadata=metadata,
            entrypoint_code=entrypoint_code,
            source_root=fake_root,
        )

        self.assertEqual(out_path, str(target_dir.resolve()))
        self.assertTrue((target_dir / "kernel-metadata.json").is_file())
        self.assertTrue((target_dir / "remote_entrypoint.py").is_file())
        self.assertTrue((target_dir / "tqc_source.tar.gz").is_file())

        # Verify contents of tqc_source.tar.gz
        with tarfile.open(target_dir / "tqc_source.tar.gz", "r:gz") as tar:
            members = tar.getnames()
            self.assertTrue(any("dummy.py" in m for m in members))

    def test_run_local_preflight_smoke_success(self):
        """Test that local preflight smoke test completes cleanly and produces artifacts."""
        # Run a 30-step smoke test
        result = run_local_preflight_smoke(
            env_id="HalfCheetah-v4",
            steps=30,
            device="cpu",
            seed=42,
        )
        self.assertTrue(result)

    @patch("src.tqc.remote.kaggle_packager.train_tqc")
    def test_run_local_preflight_smoke_failure(self, mock_train):
        """Test that smoke test failure raises KagglePackagingError."""
        mock_train.side_effect = RuntimeError("Environment initialization error")
        with self.assertRaises(KagglePackagingError) as ctx:
            run_local_preflight_smoke(env_id="HalfCheetah-v4", steps=10)
        self.assertIn("Pre-flight smoke run failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
