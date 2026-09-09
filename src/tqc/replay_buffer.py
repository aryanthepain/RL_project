from typing import Tuple, Union
import numpy as np
import torch


class ReplayBuffer:
    """Uniform replay buffer storing transitions using contiguous NumPy arrays.

    Pre-allocates memory for states, actions, rewards, next_states, and done flags
    up to a specified capacity (default 1,000,000) and provides fast vectorized
    batch sampling converted directly into PyTorch tensors on the desired device.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = torch.device(device)

        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.ptr: int = 0
        self.size: int = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        next_state: np.ndarray,
        done: Union[bool, float, np.ndarray],
    ) -> None:
        """Store a new transition in the buffer."""
        self.states[self.ptr] = np.asarray(state, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        self.rewards[self.ptr] = np.asarray(reward, dtype=np.float32)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a uniform random mini-batch of transitions as PyTorch tensors."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")
        if batch_size > self.size:
            raise ValueError(
                f"Requested batch_size ({batch_size}) exceeds current buffer size ({self.size})."
            )

        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            self.next_states[indices], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self.size
