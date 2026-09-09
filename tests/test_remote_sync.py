"""Unit tests for remote artifact download, extraction, SHA256 verification, and schema checks."""

import csv
import hashlib
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.sync_remote_artifacts import (
    ArtifactIntegrityError,
    sha256_file,
    verify_and_extract_artifacts,
)


class TestRemoteSync(unittest.TestCase):
    """Test suite for artifact synchronization and checksum verification."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.download_dir = Path(self.temp_dir) / "download"
        self.target_dir = Path(self.temp_dir) / "target"
        self.download_dir.mkdir(parents=True)
        self.target_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_run(self, corrupt_file: bool = False, missing_cols: bool = False, empty_rows: bool = False):
        """Create a mock remote run directory with archive and checksums."""
        staging = Path(self.temp_dir) / "staging"
        staging.mkdir(parents=True, exist_ok=True)

        # 1. Create metrics.csv
        metrics_path = staging / "metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if missing_cols:
                writer.writerow(["step", "loss"])
            else:
                writer.writerow(["step", "eval_return_mean", "eval_return_std", "critic_loss"])

            if not empty_rows:
                writer.writerow([100, 15.4, 1.2, 0.45])
                writer.writerow([200, 32.1, 0.8, 0.31])

        # 2. Create a mock checkpoint
        ckpt_path = staging / "checkpoint_100.pt"
        ckpt_path.write_bytes(b"dummy_weights_tensor_bytes")

        # 3. Create archive
        archive_path = self.download_dir / "artifacts.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(metrics_path, arcname="metrics.csv")
            tar.add(ckpt_path, arcname="checkpoint_100.pt")

        # 4. Generate checksums
        checksum_entries = [
            f"{sha256_file(archive_path)}  artifacts.tar.gz",
            f"{sha256_file(metrics_path)}  metrics.csv",
        ]
        if corrupt_file:
            checksum_entries.append("bad_sha256_hash_1234567890abcdef  checkpoint_100.pt")
        else:
            checksum_entries.append(f"{sha256_file(ckpt_path)}  checkpoint_100.pt")

        checksum_path = self.download_dir / "checksums.sha256"
        checksum_path.write_text("\n".join(checksum_entries) + "\n", encoding="utf-8")

    def test_verify_and_extract_success(self):
        """Test clean archive extraction and SHA256 verification."""
        self._create_mock_run()

        result = verify_and_extract_artifacts(self.download_dir, self.target_dir)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["metrics_rows"], 2)
        self.assertTrue((self.target_dir / "metrics.csv").is_file())
        self.assertTrue((self.target_dir / "checkpoint_100.pt").is_file())

    def test_checksum_mismatch_raises_error(self):
        """Test that a corrupt file hash raises ArtifactIntegrityError."""
        self._create_mock_run(corrupt_file=True)

        with self.assertRaises(ArtifactIntegrityError) as ctx:
            verify_and_extract_artifacts(self.download_dir, self.target_dir)
        self.assertIn("SHA256 mismatch", str(ctx.exception))

    def test_missing_metrics_columns_raises_error(self):
        """Test that invalid metrics.csv schema raises ArtifactIntegrityError."""
        self._create_mock_run(missing_cols=True)

        with self.assertRaises(ArtifactIntegrityError) as ctx:
            verify_and_extract_artifacts(self.download_dir, self.target_dir)
        self.assertIn("metrics.csv missing required columns", str(ctx.exception))

    def test_empty_metrics_rows_raises_error(self):
        """Test that empty metrics.csv raises ArtifactIntegrityError."""
        self._create_mock_run(empty_rows=True)

        with self.assertRaises(ArtifactIntegrityError) as ctx:
            verify_and_extract_artifacts(self.download_dir, self.target_dir)
        self.assertIn("zero data rows", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
