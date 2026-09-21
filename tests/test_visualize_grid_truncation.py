import os
import tempfile
import unittest
import numpy as np
from src.tqc.visualize import (
    create_progression_grid,
    create_progression_reel,
    dim_environment_scene,
    draw_telemetry_hud,
    rollout_checkpoint_with_telemetry,
    tile_grid_frames,
)


class DummyZeroAgent:
    """Mock agent returning all-zero actions."""

    def __init__(self, action_dim: int = 6):
        self.action_dim = action_dim
        self.device = "cpu"

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


class TestVisualizeGridTruncation(unittest.TestCase):
    def test_dim_environment_scene_preserves_banner(self):
        """D-10: Dimming reduces 3D scene brightness by 50% while preserving HUD banner at 100%."""
        H, W = 480, 640
        banner_h = 54
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # Banner region: fill with 150
        frame[:banner_h, :, :] = 150
        # Scene region: fill with 200
        frame[banner_h:, :, :] = 200

        dimmed = dim_environment_scene(frame, banner_height=banner_h, dim_factor=0.5)

        # Banner unchanged
        np.testing.assert_array_equal(dimmed[:banner_h, :, :], 150)
        # Scene dimmed by 50%
        np.testing.assert_array_equal(dimmed[banner_h:, :, :], 100)

    def test_tile_grid_frames_extends_shorter_panels(self):
        """D-10: tile_grid_frames pads shorter streams to max temporal horizon with 50% dimmed frames."""
        H, W, C = 100, 100, 3
        banner_h = 20

        # Stream 0: short stream (20 frames)
        stream_short = [np.full((H, W, C), fill_value=100, dtype=np.uint8) for _ in range(20)]
        # Stream 1..5: long streams (50 frames)
        other_streams = [
            [np.full((H, W, C), fill_value=120, dtype=np.uint8) for _ in range(50)]
            for _ in range(5)
        ]
        all_streams = [stream_short] + other_streams

        tiled = tile_grid_frames(
            frames_per_video=all_streams,
            rows=2,
            cols=3,
            border_px=2,
            banner_height=banner_h,
        )

        # Output horizon should equal maximum stream length (50 frames)
        self.assertEqual(len(tiled), 50)

        # Check that beyond frame 20, cell (0, 0) is dimmed by 50% below banner
        final_tiled_frame = tiled[45]
        # Cell 0 is top-left (y: 0..100, x: 0..100)
        cell0 = final_tiled_frame[:H, :W]
        # Banner region should retain original 100
        np.testing.assert_array_equal(cell0[:banner_h, :, :], 100)
        # Scene region should be dimmed to 50
        np.testing.assert_array_equal(cell0[banner_h:, :, :], 50)

    def test_create_progression_grid_and_reel(self):
        """D-10 & D-11: Progression grid and reel handle dictionary inputs and write videos."""
        H, W, C = 64, 64, 3
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_video_path = os.path.join(tmpdir, "test_grid.mp4")
            reel_video_path = os.path.join(tmpdir, "test_reel.mp4")

            # 6 mock checkpoint results
            rollouts = []
            for idx in range(6):
                length = 15 if idx == 0 else 30
                frames = [np.full((H, W, C), idx * 30 + 10, dtype=np.uint8) for _ in range(length)]
                rollouts.append({"frames": frames, "ckpt_idx": idx + 1})

            # Create grid
            grid_frames = create_progression_grid(
                rollout_results=rollouts,
                output_path=grid_video_path,
                rows=2,
                cols=3,
                fps=30,
                banner_height=10,
            )
            self.assertEqual(len(grid_frames), 30)
            self.assertTrue(os.path.isfile(grid_video_path))
            self.assertGreater(os.path.getsize(grid_video_path), 0)

            # Create reel
            reel_frames = create_progression_reel(
                rollout_results=rollouts,
                output_path=reel_video_path,
                fps=30,
            )
            # 15 + 5 * 30 = 165 total frames
            self.assertEqual(len(reel_frames), 165)
            self.assertTrue(os.path.isfile(reel_video_path))
            self.assertGreater(os.path.getsize(reel_video_path), 0)

    def test_real_halfcheetah_stationary_truncation(self):
        """D-15: Real MuJoCo HalfCheetah-v4 test truncates frames under stationary policy while preserving full steps."""
        agent = DummyZeroAgent(action_dim=6)
        max_steps = 100

        telemetry = rollout_checkpoint_with_telemetry(
            agent=agent,
            env_id="HalfCheetah-v4",
            max_steps=max_steps,
            seed=42,
            render=True,
            overlay=True,
            truncate_inactive=True,
            speed_threshold=0.05,
            patience=10,
            padding_frames=5,
            warmup_steps=20,
        )

        # Under zero actions, HalfCheetah settles quickly and stops moving forward
        self.assertTrue(telemetry["is_truncated"])
        self.assertIn(telemetry["completion_reason"], ["inactivity_truncated", "stagnation_truncated"])
        self.assertIsNotNone(telemetry["truncated_at_step"])
        self.assertLess(len(telemetry["frames"]), max_steps)
        # Simulation loop executed full steps
        self.assertEqual(telemetry["total_sim_steps"], max_steps)
        self.assertEqual(len(telemetry["rewards"]), max_steps)
        self.assertEqual(len(telemetry["velocities"]), max_steps)


if __name__ == "__main__":
    unittest.main()
