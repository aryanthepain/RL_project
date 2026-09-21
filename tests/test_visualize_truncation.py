import unittest
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from src.tqc.visualize import (
    RolloutInactivityDetector,
    extract_forward_position,
    extract_forward_velocity,
    rollout_checkpoint_with_telemetry,
)


class MockAgent:
    """Deterministic dummy agent for testing rollouts."""

    def __init__(self, action_dim: int = 2):
        self.action_dim = action_dim
        self.device = "cpu"

    def select_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


class MockGymEnv(gym.Env):
    """Configurable mock gymnasium environment for testing render & truncation behaviors."""

    def __init__(
        self,
        velocity_generator=None,
        position_generator=None,
        reward_val: float = 1.0,
        render_shape: Tuple[int, int, int] = (64, 64, 3),
    ):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.velocity_generator = velocity_generator or (lambda step: 0.0)
        self.position_generator = position_generator
        self.reward_val = reward_val
        self.render_shape = render_shape
        self.current_step = 0
        self.render_calls = 0
        self.current_pos = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.render_calls = 0
        self.current_pos = 0.0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        v = self.velocity_generator(self.current_step)
        if self.position_generator is not None:
            p = self.position_generator(self.current_step)
        else:
            self.current_pos += v * 0.05
            p = self.current_pos
        self.current_step += 1
        obs = np.zeros(4, dtype=np.float32)
        reward = self.reward_val
        terminated = False
        truncated = False
        info = {"x_velocity": v, "x_position": p}
        return obs, reward, terminated, truncated, info

    def render(self):
        self.render_calls += 1
        return np.zeros(self.render_shape, dtype=np.uint8)


class TestVisualizeTruncation(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()

    def test_detector_warmup_and_inactivity_trigger(self):
        """Test RolloutInactivityDetector respects warmup and triggers after consecutive inactive steps."""
        detector = RolloutInactivityDetector(
            speed_threshold=0.05,
            patience=5,
            padding_frames=3,
            warmup_steps=10,
        )

        # Warmup period (steps 0..9) with 0 velocity
        for i in range(10):
            cont = detector.step(i, 0.0)
            self.assertTrue(cont)
            self.assertFalse(detector.is_truncated)

        # Steps 10..13 (4 steps < patience 5)
        for i in range(10, 14):
            cont = detector.step(i, 0.0)
            self.assertTrue(cont)
            self.assertFalse(detector.is_truncated)

        # Step 14 (5th step == patience) -> Triggers truncation!
        cont = detector.step(14, 0.0)
        self.assertTrue(cont)
        self.assertTrue(detector.is_truncated)
        self.assertEqual(detector.completion_reason, "inactivity_truncated")
        self.assertEqual(detector.trigger_step, 14)

        # Padding frames: 3 frames (steps 15, 16, 17)
        self.assertTrue(detector.step(15, 0.0))
        self.assertTrue(detector.step(16, 0.0))
        # Final padding frame returns False to stop subsequent recording
        self.assertFalse(detector.step(17, 0.0))

        # Further steps continue to return False
        self.assertFalse(detector.step(18, 0.0))

    def test_detector_stagnation_trigger(self):
        """Test RolloutInactivityDetector triggers on oscillating/thrashing behavior with low variance and displacement."""
        detector = RolloutInactivityDetector(
            speed_threshold=0.05,
            stagnation_delta=0.01,
            warmup_steps=5,
            window_size=10,
            padding_frames=2,
        )

        # Steps 0..4: warmup
        for i in range(5):
            detector.step(i, 0.06, position=0.0)
            self.assertFalse(detector.is_truncated)

        # Steps 5..8: filling window
        for i in range(5, 9):
            cont = detector.step(i, 0.06, position=0.01 * (i % 2))
            self.assertTrue(cont)
            self.assertFalse(detector.is_truncated)

        # Step 9: 10th item fills window -> stagnation triggers!
        cont = detector.step(9, 0.06, position=0.01 * (9 % 2))
        self.assertTrue(cont)
        self.assertTrue(detector.is_truncated)
        self.assertEqual(detector.completion_reason, "stagnation_truncated")
        self.assertEqual(detector.trigger_step, 9)

        # Padding frame 1 (step 10)
        self.assertTrue(detector.step(10, 0.06, position=0.0))
        # Padding frame 2 (step 11, final padding frame -> returns False)
        self.assertFalse(detector.step(11, 0.06, position=0.0))

    def test_rollout_stationary_agent_truncation(self):
        """D-01, D-03, D-04, D-05: Stationary agent truncates frames, skips render, and preserves full returns."""
        # 10 warmup + 5 patience + 3 padding = 18 recorded frames out of 50 steps
        env = MockGymEnv(velocity_generator=lambda s: 0.0, reward_val=2.5)
        max_steps = 50

        res = rollout_checkpoint_with_telemetry(
            agent=self.agent,
            env_id=env,
            max_steps=max_steps,
            render=True,
            overlay=True,
            truncate_inactive=True,
            speed_threshold=0.05,
            patience=5,
            padding_frames=3,
            warmup_steps=10,
        )

        self.assertTrue(res["is_truncated"])
        self.assertEqual(res["completion_reason"], "inactivity_truncated")
        self.assertEqual(res["truncated_at_step"], 14)
        self.assertEqual(res["total_sim_steps"], 50)
        self.assertEqual(len(res["frames"]), 18)
        self.assertEqual(len(res["velocities"]), 50)
        self.assertEqual(len(res["rewards"]), 50)
        self.assertEqual(res["total_return"], 50 * 2.5)

        # D-04: env.render() was only called 18 times, not 50 times!
        self.assertEqual(env.render_calls, 18)

    def test_rollout_active_agent_no_truncation(self):
        """Active agent moving fast should never be truncated."""
        env = MockGymEnv(velocity_generator=lambda s: 1.5, reward_val=1.0)
        max_steps = 40

        res = rollout_checkpoint_with_telemetry(
            agent=self.agent,
            env_id=env,
            max_steps=max_steps,
            render=True,
            overlay=True,
            truncate_inactive=True,
            speed_threshold=0.05,
            warmup_steps=10,
        )

        self.assertFalse(res["is_truncated"])
        self.assertEqual(res["completion_reason"], "max_steps")
        self.assertIsNone(res["truncated_at_step"])
        self.assertEqual(res["total_sim_steps"], 40)
        self.assertEqual(len(res["frames"]), 40)
        self.assertEqual(len(res["velocities"]), 40)
        self.assertEqual(env.render_calls, 40)

    def test_rollout_disabled_truncation(self):
        """When truncate_inactive=False, records all frames even if stationary."""
        env = MockGymEnv(velocity_generator=lambda s: 0.0, reward_val=1.0)
        max_steps = 30

        res = rollout_checkpoint_with_telemetry(
            agent=self.agent,
            env_id=env,
            max_steps=max_steps,
            render=True,
            truncate_inactive=False,
        )

        self.assertFalse(res["is_truncated"])
        self.assertEqual(len(res["frames"]), 30)
        self.assertEqual(len(res["velocities"]), 30)
        self.assertEqual(env.render_calls, 30)

    def test_velocity_and_position_extractors(self):
        """Test multi-tiered velocity and position extraction helpers."""
        # 1. Standard x_velocity
        self.assertAlmostEqual(extract_forward_velocity({"x_velocity": 1.25}, None), 1.25)

        # 2. 2D Euclidean norm for Ant-v4
        self.assertAlmostEqual(
            extract_forward_velocity({"x_velocity": 3.0, "y_velocity": 4.0}, None),
            5.0,
        )

        # 3. forward_velocity key
        self.assertAlmostEqual(extract_forward_velocity({"forward_velocity": 2.1}, None), 2.1)

        # 4. Fallback to 0.0
        self.assertEqual(extract_forward_velocity({}, None), 0.0)

        # 5. Position extractor
        self.assertAlmostEqual(extract_forward_position({"x_position": 10.5}, None), 10.5)
        self.assertIsNone(extract_forward_position({}, None))


if __name__ == "__main__":
    unittest.main()
