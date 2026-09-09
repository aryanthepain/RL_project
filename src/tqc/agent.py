import copy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actor import Actor
from .critic import CriticEnsemble
from .truncation import quantile_huber_loss, truncate_quantiles


class TQCAgent:
    """
    Truncated Quantile Critics (TQC) Agent.
    Implements continuous control with distributional critics, parametric truncation,
    and automatic entropy temperature tuning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_critics: int = 5,
        n_quantiles: int = 25,
        drop_top: int = 5,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.drop_top = drop_top
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # Actor Network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic Ensemble
        self.critic = CriticEnsemble(
            state_dim, action_dim, hidden_dim, n_critics, n_quantiles
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic Entropy Tuning
        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        """Current policy entropy temperature alpha."""
        return self.log_alpha.exp()

    def select_action(
        self, state: Union[np.ndarray, torch.Tensor], deterministic: bool = False
    ) -> np.ndarray:
        """
        Select an action given an observation state.

        Args:
            state: Observation vector (state_dim,) or (batch_size, state_dim)
            deterministic: Whether to return deterministic mean action without noise

        Returns:
            action: Action as numpy array in [-1, 1]
        """
        if isinstance(state, np.ndarray):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(device=self.device, dtype=torch.float32)

        action_tensor = self.actor.act(state_tensor, deterministic=deterministic)
        return action_tensor.cpu().numpy()

    def update(
        self,
        batch: Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            Any,
        ],
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Perform a single gradient update step across critics, actor, and temperature.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones) OR a ReplayBuffer
            batch_size: Mini-batch size when sampling from a ReplayBuffer (default 256)

        Returns:
            metrics: Dictionary containing losses, alpha, and Q-statistics
        """
        if hasattr(batch, "sample"):
            sampled_batch = batch.sample(batch_size)
        else:
            sampled_batch = batch

        states, actions, rewards, next_states, dones = [
            torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in sampled_batch
        ]

        # Ensure rewards and dones are shape (B, 1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)

        # ----------------------------
        # 1. Critic Target Calculation
        # ----------------------------
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor(next_states)
            target_quantiles = self.critic_target(next_states, next_actions)  # (B, M, N)
            retained_quantiles = truncate_quantiles(
                target_quantiles, drop_top=self.drop_top
            )  # (B, MN - d)
            # Soft target shift: z - alpha * log_pi
            next_values = retained_quantiles - self.alpha * next_log_probs  # (B, MN - d)
            targets = rewards + self.gamma * (1.0 - dones) * next_values

        # ----------------------------
        # 2. Critic Ensemble Update
        # ----------------------------
        current_quantiles = self.critic(states, actions)  # (B, M, N)
        critic_loss = quantile_huber_loss(current_quantiles, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------
        # 3. Actor Policy Update
        # ----------------------------
        new_actions, log_probs, _ = self.actor(states)
        q_quantiles = self.critic(states, new_actions)  # (B, M, N)
        q_mean = q_quantiles.mean(dim=(1, 2), keepdim=True)  # (B, 1)

        actor_loss = (self.alpha.detach() * log_probs - q_mean).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------------
        # 4. Entropy Temperature (alpha) Update
        # ----------------------------
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ----------------------------
        # 5. Polyak Target Smoothing
        # ----------------------------
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "target_q_mean": targets.mean().item(),
            "current_q_mean": current_quantiles.mean().item(),
        }

    def save(self, filepath: str):
        """Save model weights and optimizer states."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "target_entropy": self.target_entropy,
            "drop_top": self.drop_top,
            "gamma": self.gamma,
            "tau": self.tau,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """Load model weights and optimizer states."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        if "target_entropy" in checkpoint:
            self.target_entropy = checkpoint["target_entropy"]
        if "drop_top" in checkpoint:
            self.drop_top = checkpoint["drop_top"]
        if "gamma" in checkpoint:
            self.gamma = checkpoint["gamma"]
        if "tau" in checkpoint:
            self.tau = checkpoint["tau"]
