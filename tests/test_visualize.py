import os
import shutil
import tempfile
import unittest
import numpy as np
from src.tqc.agent import TQCAgent
from src.tqc.envs import get_env_dims, make_env
from src.tqc.visualize import (
    draw_telemetry_hud,
    plot_progression_diagnostics,
    record_agent_video,
    rollout_checkpoint_with_telemetry,
    save_video,
    stitch_video_files,
    tile_grid_frames,
)


class TestVisualize(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        env = make_env("HalfCheetah-v4", seed=42)
        state_dim, action_dim = get_env_dims(env)
        env.close()
        self.agent = TQCAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_draw_telemetry_hud(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        hud_frame = draw_telemetry_hud(
            frame_rgb=frame,
            ckpt_idx=1,
            total_ckpts=100,
            step=100,
            total_steps=10000,
            t_sec=1.5,
            v_x=2.45,
            ep_return=120.5,
            q_mean=45.2,
        )
        self.assertEqual(hud_frame.shape, (240, 320, 3))
        self.assertEqual(hud_frame.dtype, np.uint8)
        # Top banner should have non-zero pixels from text and background
        self.assertGreater(hud_frame[:50, :].sum(), 0)

    def test_rollout_checkpoint_with_telemetry(self):
        telemetry = rollout_checkpoint_with_telemetry(
            agent=self.agent,
            env_id="HalfCheetah-v4",
            max_steps=5,  # fast test: 5 steps
            seed=42,
            ckpt_idx=1,
            total_ckpts=100,
            step_num=100,
            total_steps=10000,
            overlay=True,
            fps=30,
        )
        self.assertIn("frames", telemetry)
        self.assertIn("velocities", telemetry)
        self.assertIn("rewards", telemetry)
        self.assertIn("cumulative_returns", telemetry)
        self.assertIn("actions", telemetry)
        self.assertIn("q_means", telemetry)
        self.assertEqual(len(telemetry["frames"]), 5)
        self.assertEqual(len(telemetry["velocities"]), 5)

    def test_tile_grid_frames_2x3(self):
        # 6 mock video streams, each with 3 frames of size 100x120x3
        frames_per_video = [
            [np.full((100, 120, 3), fill_value=i * 40, dtype=np.uint8) for _ in range(3)]
            for i in range(6)
        ]
        tiled = tile_grid_frames(frames_per_video, rows=2, cols=3, border_px=2)
        self.assertEqual(len(tiled), 3)
        expected_h = 2 * 100 + 1 * 2  # 202
        expected_w = 3 * 120 + 2 * 2  # 364
        self.assertEqual(tiled[0].shape, (expected_h, expected_w, 3))

    def test_save_and_stitch_video_files(self):
        # Create 2 short test videos
        frames_a = [np.full((64, 64, 3), 50, dtype=np.uint8) for _ in range(5)]
        frames_b = [np.full((64, 64, 3), 150, dtype=np.uint8) for _ in range(5)]

        video_a = os.path.join(self.test_dir, "video_a.mp4")
        video_b = os.path.join(self.test_dir, "video_b.mp4")
        save_video(frames_a, video_a, fps=10)
        save_video(frames_b, video_b, fps=10)

        self.assertTrue(os.path.isfile(video_a))
        self.assertTrue(os.path.isfile(video_b))

        stitched_path = os.path.join(self.test_dir, "stitched.mp4")
        stitch_video_files([video_a, video_b], stitched_path, fps=10)
        self.assertTrue(os.path.isfile(stitched_path))
        self.assertGreater(os.path.getsize(stitched_path), 500)

    def test_plot_progression_diagnostics(self):
        telemetries = [
            {
                "step_num": 100,
                "velocities": [1.0, 1.2, 1.1],
                "cumulative_returns": [0.5, 1.2, 2.0],
                "actions": [np.array([0.1, -0.2]), np.array([0.2, -0.1]), np.array([0.15, -0.25])],
                "q_means": [5.0, 5.2, 5.1],
            },
            {
                "step_num": 500,
                "velocities": [2.0, 2.5, 3.0],
                "cumulative_returns": [1.5, 3.5, 6.0],
                "actions": [np.array([0.5, -0.4]), np.array([0.6, -0.5]), np.array([0.55, -0.45])],
                "q_means": [15.0, 16.2, 17.1],
            },
        ]
        out_png = os.path.join(self.test_dir, "diagnostics.png")
        plot_progression_diagnostics(telemetries, out_png)
        self.assertTrue(os.path.isfile(out_png))
        self.assertGreater(os.path.getsize(out_png), 1000)

    def test_record_agent_video_gif(self):
        out_gif = os.path.join(self.test_dir, "test_rollout.gif")
        saved_path = record_agent_video(
            agent=self.agent,
            env_id="HalfCheetah-v4",
            output_path=out_gif,
            max_steps=5,
            fps=10,
            seed=42,
        )
        self.assertEqual(saved_path, out_gif)
        self.assertTrue(os.path.isfile(out_gif))
        self.assertGreater(os.path.getsize(out_gif), 1000)


if __name__ == "__main__":
    unittest.main()
