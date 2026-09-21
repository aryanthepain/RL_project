"""Unit tests for clean_runs.py inspection and checkpoint pruning utility."""

import os
import shutil
import tempfile
import unittest

from scripts.clean_runs import (
    delete_run_directory,
    format_inspection_table,
    prune_intermediate_checkpoints,
    scan_run_directories,
)


class TestCleanRuns(unittest.TestCase):
    """Test suite for disk quota inspection, pruning, and run cleanup."""

    def setUp(self) -> None:
        self.temp_root = tempfile.mkdtemp()

        # Create mock run directory: run_01
        self.run1 = os.path.join(self.temp_root, "tqc_HalfCheetah-v4_seed0")
        os.makedirs(self.run1, exist_ok=True)

        self._write_file(os.path.join(self.run1, "checkpoint_1000.pt"), b"dummy_ckpt_1000" * 100)
        self._write_file(os.path.join(self.run1, "checkpoint_2000.pt"), b"dummy_ckpt_2000" * 100)
        self._write_file(os.path.join(self.run1, "best_model.pt"), b"dummy_best_model" * 100)
        self._write_file(os.path.join(self.run1, "latest.pt"), b"dummy_latest" * 100)
        self._write_file(os.path.join(self.run1, "final_model.pt"), b"dummy_final" * 100)
        self._write_file(os.path.join(self.run1, "metrics.csv"), b"step,reward\n1000,10.0\n")
        self._write_file(os.path.join(self.run1, "eval_video.mp4"), b"dummy_video_bytes")

        # Create mock run directory: run_02
        self.run2 = os.path.join(self.temp_root, "tqc_Hopper-v4_seed1")
        os.makedirs(self.run2, exist_ok=True)
        self._write_file(os.path.join(self.run2, "checkpoint_500.pt"), b"dummy_ckpt_500" * 50)
        self._write_file(os.path.join(self.run2, "latest.pt"), b"dummy_latest_500" * 50)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _write_file(self, path: str, content: bytes) -> None:
        with open(path, "wb") as f:
            f.write(content)

    def test_scan_run_directories(self) -> None:
        """Test scanning finds runs, counts checkpoints, and records sizes."""
        runs = scan_run_directories(self.temp_root)
        self.assertEqual(len(runs), 2)

        run1_info = next(r for r in runs if "HalfCheetah" in r.run_id)
        self.assertEqual(run1_info.checkpoint_count, 2)
        self.assertEqual(len(run1_info.intermediate_checkpoints), 2)
        self.assertTrue(run1_info.total_size_bytes > 0)
        self.assertTrue(run1_info.prunable_size_bytes > 0)

        # Check preserved files
        preserved_names = [os.path.basename(p) for p in run1_info.preserved_files]
        self.assertIn("best_model.pt", preserved_names)
        self.assertIn("latest.pt", preserved_names)
        self.assertIn("final_model.pt", preserved_names)
        self.assertIn("metrics.csv", preserved_names)
        self.assertIn("eval_video.mp4", preserved_names)

    def test_format_inspection_table(self) -> None:
        """Test inspection table formats summary report correctly."""
        runs = scan_run_directories(self.temp_root)
        table = format_inspection_table(runs)
        self.assertIn("TQC EXPERIMENT RUN DISK USAGE REPORT", table)
        self.assertIn("HalfCheetah", table)
        self.assertIn("Hopper", table)
        self.assertIn("TOTALS (2 runs)", table)

    def test_prune_dry_run(self) -> None:
        """Test prune with dry_run=True does not remove any files."""
        deleted_count, freed_bytes = prune_intermediate_checkpoints(self.run1, dry_run=True)
        self.assertEqual(deleted_count, 2)
        self.assertTrue(freed_bytes > 0)

        # Verify files still exist
        self.assertTrue(os.path.exists(os.path.join(self.run1, "checkpoint_1000.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.run1, "checkpoint_2000.pt")))

    def test_prune_live(self) -> None:
        """Test prune deletes intermediate checkpoints but strictly keeps best, latest, metrics, videos."""
        deleted_count, freed_bytes = prune_intermediate_checkpoints(self.run1, dry_run=False)
        self.assertEqual(deleted_count, 2)
        self.assertTrue(freed_bytes > 0)

        # Intermediate checkpoints must be gone
        self.assertFalse(os.path.exists(os.path.join(self.run1, "checkpoint_1000.pt")))
        self.assertFalse(os.path.exists(os.path.join(self.run1, "checkpoint_2000.pt")))

        # Critical assets MUST be preserved
        self.assertTrue(os.path.exists(os.path.join(self.run1, "best_model.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.run1, "latest.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.run1, "final_model.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.run1, "metrics.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.run1, "eval_video.mp4")))

    def test_delete_run_directory(self) -> None:
        """Test deleting an entire run directory."""
        freed = delete_run_directory(self.run2, dry_run=False)
        self.assertTrue(freed > 0)
        self.assertFalse(os.path.exists(self.run2))


if __name__ == "__main__":
    unittest.main()
