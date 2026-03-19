"""
PolicyNetwork: MLP policy for grid control.

Outputs generator setpoints given the grid state observation.
Uses sigmoid output to ensure actions are in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    Small MLP policy: obs → action.

    Architecture: obs_dim → 64 → 64 → act_dim (sigmoid)

    Args:
        obs_dim: Size of observation vector
        act_dim: Size of action vector (number of generators)
        hidden_dim: Width of hidden layers
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Mean head: deterministic action output
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        # Log std: learnable but action-independent (diagonal Gaussian)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Tensor of shape (..., obs_dim)
        Returns:
            (mean, std): Tensors of shape (..., act_dim)
        """
        h = self.net(obs)
        mean = torch.sigmoid(self.mean_head(h))  # actions in (0, 1)
        std = torch.exp(self.log_std.clamp(-4, 0))  # std in (0.018, 1.0)
        return mean, std

    def get_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and return (action, log_prob).

        Args:
            obs: Tensor of shape (obs_dim,) or (batch, obs_dim)
        Returns:
            action: Tensor in [0, 1]^act_dim
            log_prob: Scalar log probability of the action
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        # Sample in raw space, then clamp
        raw = dist.rsample()
        action = torch.sigmoid(raw)  # squash to (0, 1)
        # Log prob with change of variables: log p(a) = log p(raw) - log |da/draw|
        log_prob = dist.log_prob(raw) - torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic (mean) action for evaluation."""
        mean, _ = self.forward(obs)
        return mean

    def flat_weights(self) -> torch.Tensor:
        """Return all parameters as a single flat vector."""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def load_flat_weights(self, flat: torch.Tensor):
        """Load parameters from a flat vector (in-place)."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
            offset += numel
