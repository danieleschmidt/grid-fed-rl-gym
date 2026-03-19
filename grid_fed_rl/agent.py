"""
LocalRLAgent: Trains a PolicyNetwork on a local GridEnv using REINFORCE.

Each grid operator (utility company) has their own GridEnv and trains
a local policy without sharing raw grid data with others.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional

from .env import GridEnv
from .policy import PolicyNetwork


class LocalRLAgent:
    """
    REINFORCE agent for local grid control.

    Args:
        env: GridEnv instance for this operator
        operator_id: String identifier (e.g. "operator_0")
        lr: Learning rate
        gamma: Discount factor
        device: Torch device
    """

    def __init__(
        self,
        env: GridEnv,
        operator_id: str = "operator_0",
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.env = env
        self.operator_id = operator_id
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy = PolicyNetwork(
            obs_dim=env.obs_dim,
            act_dim=env.act_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.episode_rewards: List[float] = []

    def collect_episode(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """
        Run one episode and collect (log_probs, actions, rewards).

        Returns:
            log_probs: List of log-probability tensors
            actions: List of action tensors
            rewards: List of scalar rewards
        """
        obs = self.env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action, log_prob = self.policy.get_action(obs_tensor)

            obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
            log_probs.append(log_prob)
            rewards.append(reward)

        return log_probs, [], rewards

    def train_episode(self) -> float:
        """
        Train for one episode using REINFORCE with baseline.

        Returns:
            total_reward: Sum of rewards for this episode
        """
        log_probs, _, rewards = self.collect_episode()
        total_reward = sum(rewards)
        self.episode_rewards.append(total_reward)

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # Normalize returns for variance reduction
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        # REINFORCE loss: -sum(log_prob * return)
        loss = 0.0
        for log_prob, G in zip(log_probs, returns_tensor):
            loss = loss + (-log_prob * G)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_reward

    def train(self, n_episodes: int, verbose: bool = False) -> List[float]:
        """
        Train for n_episodes.

        Args:
            n_episodes: Number of training episodes
            verbose: Print progress every 20 episodes
        Returns:
            List of episode rewards
        """
        rewards = []
        for ep in range(n_episodes):
            r = self.train_episode()
            rewards.append(r)
            if verbose and (ep + 1) % 20 == 0:
                avg = np.mean(rewards[-20:])
                print(f"  [{self.operator_id}] ep {ep+1}/{n_episodes} | avg reward: {avg:.2f}")
        return rewards

    def evaluate(self, n_episodes: int = 5) -> Tuple[float, float]:
        """
        Evaluate policy deterministically.

        Returns:
            (mean_reward, mean_stability_score)
        """
        rewards = []
        stability_scores = []

        with torch.no_grad():
            for _ in range(n_episodes):
                obs = self.env.reset()
                ep_reward = 0.0
                done = False
                while not done:
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                    action = self.policy.deterministic_action(obs_tensor)
                    obs, reward, done, _ = self.env.step(action.cpu().numpy())
                    ep_reward += reward
                rewards.append(ep_reward)
                stability_scores.append(self.env.grid_stability_score())

        return float(np.mean(rewards)), float(np.mean(stability_scores))

    def get_policy_weights(self) -> torch.Tensor:
        """Return flat policy weights for federation."""
        return self.policy.flat_weights().clone()

    def set_policy_weights(self, flat_weights: torch.Tensor):
        """Load federated weights into local policy."""
        self.policy.load_flat_weights(flat_weights)
        # Reset optimizer state after weight update
        for group in self.optimizer.param_groups:
            group['params'] = list(self.policy.parameters())
        self.optimizer.state.clear()
