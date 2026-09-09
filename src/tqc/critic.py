import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    Single Distributional Quantile Critic network.
    3-layer MLP predicting N quantiles for a state-action pair.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_quantiles: int = 25,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_quantiles),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, action_dim)
        Returns:
            quantiles: (batch_size, n_quantiles)
        """
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)


class CriticEnsemble(nn.Module):
    """
    Ensemble of M independent Quantile Critics.
    Evaluates state-action pairs across all critics simultaneously.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_critics: int = 5,
        n_quantiles: int = 25,
    ):
        super().__init__()
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.critics = nn.ModuleList(
            [
                Critic(state_dim, action_dim, hidden_dim, n_quantiles)
                for _ in range(n_critics)
            ]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass across all critics in the ensemble.

        Args:
            state: Tensor of shape (batch_size, state_dim)
            action: Tensor of shape (batch_size, action_dim)

        Returns:
            quantiles: Tensor of shape (batch_size, n_critics, n_quantiles)
        """
        sa = torch.cat([state, action], dim=-1)
        # Predict quantiles for each critic
        quantiles = torch.stack([critic.net(sa) for critic in self.critics], dim=1)
        return quantiles
