import unittest
import numpy as np
import gymnasium as gym
from src.tqc.envs import BENCHMARK_ENVS, get_env_dims, get_env_metadata, make_env


class TestEnvs(unittest.TestCase):
    def test_get_env_metadata(self):
        meta_hc = get_env_metadata("HalfCheetah-v4")
        self.assertEqual(meta_hc["drop_top"], 5)
        self.assertEqual(meta_hc["total_timesteps"], 1_000_000)

        meta_hum = get_env_metadata("Humanoid-v4")
        self.assertEqual(meta_hum["drop_top"], 2)
        self.assertEqual(meta_hum["total_timesteps"], 3_000_000)

        meta_custom = get_env_metadata("SomeCustomHumanoid-v1")
        self.assertEqual(meta_custom["drop_top"], 2)

    def test_make_env_and_dims_halfcheetah(self):
        env = make_env("HalfCheetah-v4", seed=42)
        state_dim, action_dim = get_env_dims(env)

        self.assertEqual(state_dim, 17)
        self.assertEqual(action_dim, 6)

        obs, info = env.reset()
        self.assertEqual(obs.shape, (17,))

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(next_obs.shape, (17,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

        env.close()

    def test_make_env_and_dims_hopper(self):
        env = make_env("Hopper-v4", seed=42)
        state_dim, action_dim = get_env_dims(env)

        self.assertEqual(state_dim, 11)
        self.assertEqual(action_dim, 3)

        obs, _ = env.reset()
        self.assertEqual(obs.shape, (11,))
        env.close()

    def test_seeding_reproducibility(self):
        env1 = make_env("HalfCheetah-v4", seed=100)
        obs1, _ = env1.reset(seed=100)
        a1 = env1.action_space.sample()
        next_obs1, r1, _, _, _ = env1.step(a1)
        env1.close()

        env2 = make_env("HalfCheetah-v4", seed=100)
        obs2, _ = env2.reset(seed=100)
        a2 = env2.action_space.sample()
        next_obs2, r2, _, _, _ = env2.step(a2)
        env2.close()

        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(next_obs1, next_obs2)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
