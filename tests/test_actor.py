import unittest
import torch
from src.tqc.actor import Actor, LOG_STD_MIN, LOG_STD_MAX


class TestActor(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.state_dim = 17
        self.action_dim = 6
        self.hidden_dim = 512
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim)

    def test_forward_shapes_and_bounds(self):
        batch_size = 32
        state = torch.randn(batch_size, self.state_dim)
        action, log_prob, mean_action = self.actor(state)

        # Check shapes
        self.assertEqual(action.shape, (batch_size, self.action_dim))
        self.assertEqual(log_prob.shape, (batch_size, 1))
        self.assertEqual(mean_action.shape, (batch_size, self.action_dim))

        # Check bounds: tanh outputs must be strictly in [-1, 1]
        self.assertTrue((action >= -1.0).all() and (action <= 1.0).all())
        self.assertTrue((mean_action >= -1.0).all() and (mean_action <= 1.0).all())

        # Check that log_prob is finite
        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())

    def test_reparameterization_gradient_flow(self):
        batch_size = 8
        state = torch.randn(batch_size, self.state_dim)
        action, log_prob, _ = self.actor(state)

        loss = action.sum() + log_prob.sum()
        loss.backward()

        # Check gradients exist in all actor layers
        for name, param in self.actor.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} has NaN")

    def test_deterministic_act(self):
        state = torch.randn(self.state_dim)
        act1 = self.actor.act(state, deterministic=True)
        act2 = self.actor.act(state, deterministic=True)
        self.assertTrue(torch.allclose(act1, act2))
        self.assertEqual(act1.shape, (self.action_dim,))


if __name__ == "__main__":
    unittest.main()
