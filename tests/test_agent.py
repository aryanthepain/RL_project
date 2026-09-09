import os
import tempfile
import unittest
import numpy as np
import torch
from src.tqc.agent import TQCAgent


class TestTQCAgent(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.state_dim = 10
        self.action_dim = 3
        self.hidden_dim = 128  # Faster for tests
        self.n_critics = 5
        self.n_quantiles = 25
        self.drop_top = 5
        self.agent = TQCAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            n_critics=self.n_critics,
            n_quantiles=self.n_quantiles,
            drop_top=self.drop_top,
            gamma=0.99,
            tau=0.005,
            lr=3e-4,
        )

    def test_action_selection(self):
        # Single state numpy array
        state = np.random.randn(self.state_dim).astype(np.float32)
        action_det = self.agent.select_action(state, deterministic=True)
        action_det2 = self.agent.select_action(state, deterministic=True)

        self.assertEqual(action_det.shape, (self.action_dim,))
        self.assertTrue((action_det >= -1.0).all() and (action_det <= 1.0).all())
        # Deterministic actions must be identical
        np.testing.assert_allclose(action_det, action_det2, atol=1e-5)

        # Stochastic action
        action_stoch = self.agent.select_action(state, deterministic=False)
        self.assertEqual(action_stoch.shape, (self.action_dim,))
        self.assertTrue((action_stoch >= -1.0).all() and (action_stoch <= 1.0).all())

    def test_update_step(self):
        batch_size = 32
        states = np.random.randn(batch_size, self.state_dim).astype(np.float32)
        actions = np.random.uniform(-1, 1, size=(batch_size, self.action_dim)).astype(np.float32)
        rewards = np.random.randn(batch_size, 1).astype(np.float32)
        next_states = np.random.randn(batch_size, self.state_dim).astype(np.float32)
        dones = (np.random.rand(batch_size, 1) > 0.9).astype(np.float32)

        batch = (states, actions, rewards, next_states, dones)
        metrics = self.agent.update(batch)

        # Check metrics
        self.assertIn("critic_loss", metrics)
        self.assertIn("actor_loss", metrics)
        self.assertIn("alpha_loss", metrics)
        self.assertIn("alpha", metrics)

        self.assertFalse(np.isnan(metrics["critic_loss"]))
        self.assertFalse(np.isnan(metrics["actor_loss"]))
        self.assertFalse(np.isnan(metrics["alpha_loss"]))
        self.assertFalse(np.isnan(metrics["alpha"]))
        self.assertTrue(metrics["alpha"] > 0)

    def test_polyak_target_update(self):
        batch_size = 16
        batch = (
            np.random.randn(batch_size, self.state_dim).astype(np.float32),
            np.random.uniform(-1, 1, size=(batch_size, self.action_dim)).astype(np.float32),
            np.random.randn(batch_size, 1).astype(np.float32),
            np.random.randn(batch_size, self.state_dim).astype(np.float32),
            np.zeros((batch_size, 1), dtype=np.float32),
        )

        initial_target_p = list(self.agent.critic_target.parameters())[0].clone()
        initial_online_p = list(self.agent.critic.parameters())[0].clone()

        self.agent.update(batch)

        updated_target_p = list(self.agent.critic_target.parameters())[0]
        # Target parameter should have moved towards online parameter
        self.assertFalse(torch.allclose(initial_target_p, updated_target_p))

    def test_save_and_load_checkpoint(self):
        state = np.random.randn(self.state_dim).astype(np.float32)
        act_before = self.agent.select_action(state, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "tqc_checkpoint.pt")
            self.agent.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            # New agent with fresh initialization
            new_agent = TQCAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                n_critics=self.n_critics,
                n_quantiles=self.n_quantiles,
                drop_top=self.drop_top,
            )

            # Check actions differ before load
            act_new_before = new_agent.select_action(state, deterministic=True)
            self.assertFalse(np.allclose(act_before, act_new_before, atol=1e-4))

            # Load weights
            new_agent.load(save_path)
            act_after = new_agent.select_action(state, deterministic=True)

            # Actions must now match exactly
            np.testing.assert_allclose(act_before, act_after, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
