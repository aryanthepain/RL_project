import unittest
import torch
from src.tqc.critic import Critic, CriticEnsemble


class TestCritic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.state_dim = 17
        self.action_dim = 6
        self.hidden_dim = 512
        self.n_critics = 5
        self.n_quantiles = 25

    def test_single_critic(self):
        batch_size = 16
        critic = Critic(self.state_dim, self.action_dim, self.hidden_dim, self.n_quantiles)
        state = torch.randn(batch_size, self.state_dim)
        action = torch.randn(batch_size, self.action_dim)

        out = critic(state, action)
        self.assertEqual(out.shape, (batch_size, self.n_quantiles))
        self.assertFalse(torch.isnan(out).any())

    def test_critic_ensemble(self):
        batch_size = 32
        ensemble = CriticEnsemble(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.n_critics,
            self.n_quantiles,
        )
        state = torch.randn(batch_size, self.state_dim)
        action = torch.randn(batch_size, self.action_dim)

        out = ensemble(state, action)
        # Expected shape: (batch_size, n_critics, n_quantiles) -> (32, 5, 25)
        self.assertEqual(out.shape, (batch_size, self.n_critics, self.n_quantiles))
        self.assertFalse(torch.isnan(out).any())

    def test_ensemble_independence(self):
        # Verify that critic weights are initialized independently
        ensemble = CriticEnsemble(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.n_critics,
            self.n_quantiles,
        )
        p1 = list(ensemble.critics[0].parameters())[0]
        p2 = list(ensemble.critics[1].parameters())[0]
        self.assertFalse(torch.allclose(p1, p2))

    def test_gradient_flow(self):
        batch_size = 8
        ensemble = CriticEnsemble(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.n_critics,
            self.n_quantiles,
        )
        state = torch.randn(batch_size, self.state_dim)
        action = torch.randn(batch_size, self.action_dim)

        out = ensemble(state, action)
        loss = out.sum()
        loss.backward()

        for name, param in ensemble.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")


if __name__ == "__main__":
    unittest.main()
