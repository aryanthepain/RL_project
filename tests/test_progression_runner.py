import os
import shutil
import tempfile
import unittest
from scripts.run_halfcheetah_progression import get_checkpoint_step, run_progression_pipeline


class TestProgressionRunner(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ckpt_dir = os.path.join(self.test_dir, "runs", "test_prog")
        self.video_dir = os.path.join(self.test_dir, "videos")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_get_checkpoint_step(self):
        self.assertEqual(get_checkpoint_step("checkpoint_0.pt"), 0)
        self.assertEqual(get_checkpoint_step("/path/to/checkpoint_100.pt"), 100)
        self.assertEqual(get_checkpoint_step("checkpoint_10000.pt"), 10000)
        self.assertEqual(get_checkpoint_step("best_model.pt"), -1)

    def test_run_progression_pipeline_smoke(self):
        # Run ultra-fast smoke test: 40 steps, 20-step checkpoint, 3 frames
        summary = run_progression_pipeline(
            total_timesteps=40,
            checkpoint_freq=20,
            eval_frames=3,
            fps=10,
            seed=42,
            output_dir=self.video_dir,
            checkpoint_dir=self.ckpt_dir,
            skip_train=False,
            smoke_test=False,  # explicit fast parameters
            device="cpu",
        )

        self.assertGreaterEqual(summary["checkpoints_evaluated"], 2)
        self.assertTrue(os.path.isfile(summary["grid_video"]))
        self.assertTrue(os.path.isfile(summary["montage_video"]))
        self.assertTrue(os.path.isfile(summary["diagnostics_plot"]))
        self.assertGreater(os.path.getsize(summary["grid_video"]), 500)
        self.assertGreater(os.path.getsize(summary["montage_video"]), 500)


if __name__ == "__main__":
    unittest.main()
