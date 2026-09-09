import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


class Actor(nn.Module):
    """
    Squashed Gaussian Actor policy network for continuous action spaces.
    Outputs mean and log standard deviation, samples using the reparameterization trick,
    and applies tanh squashing with analytical log-probability adjustment.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """
        Forward pass through the actor network.

        Args:
            state: Tensor of shape (batch_size, state_dim)

        Returns:
            action: Squashed action tensor of shape (batch_size, action_dim) in [-1, 1]
            log_prob: Log-probability of sampled action, shape (batch_size, 1)
            mean_action: Deterministic squashed mean action, shape (batch_size, action_dim)
        """
        h = self.net(state)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        raw_action = dist.rsample()  # Reparameterization trick
        action = torch.tanh(raw_action)

        # Log-probability adjusted for tanh squashing (Jacobian determinant correction)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mu)

        return action, log_prob, mean_action

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Inference action generation.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
            deterministic: If True, return tanh(mu) without stochastic noise

        Returns:
            action: Tensor in [-1, 1]
        """
        single_env = state.dim() == 1
        if single_env:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action, _, mean_action = self.forward(state)
            selected_action = mean_action if deterministic else action

        if single_env:
            selected_action = selected_action.squeeze(0)

        return selected_action
