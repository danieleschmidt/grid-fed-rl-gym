"""Offline reinforcement learning algorithms for grid control."""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

from .base import OfflineRLAlgorithm, ActorCriticBase, TrainingMetrics, GridDataset


class CQL(OfflineRLAlgorithm, ActorCriticBase):
    """Conservative Q-Learning (CQL) for offline RL."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        conservative_weight: float = 5.0,
        with_lagrange: bool = False,
        lagrange_threshold: float = 10.0,
        device: str = "auto",
        **kwargs
    ) -> None:
        super().__init__(state_dim, action_dim, conservative_weight, device, **kwargs)
        ActorCriticBase.__init__(self, state_dim, action_dim, hidden_dims, device=device, **kwargs)
        
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.with_lagrange = with_lagrange
        self.lagrange_threshold = lagrange_threshold
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()) + list(self.critic_target.parameters()), lr=lr
        )
        
        # Lagrange multiplier for CQL
        if self.with_lagrange:
            self.log_alpha_prime = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_prime_optimizer = optim.Adam([self.log_alpha_prime], lr=lr)
            
    def _build_networks(self):
        """Build actor and critic networks."""
        # Actor network (policy)
        self.actor = self._build_actor().to(self.device)
        
        # Twin critic networks
        self.critic = self._build_twin_critic().to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Freeze target network
        for param in self.critic_target.parameters():
            param.requires_grad = False
            
    def _build_actor(self) -> nn.Module:
        """Build actor network with tanh output."""
        return self._build_mlp(
            self.state_dim,
            self.action_dim * 2,  # Mean and log_std
            self.hidden_dims,
            self.activation
        )
        
    def _build_twin_critic(self) -> nn.Module:
        """Build twin Q-networks."""
        class TwinCritic(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims, activation):
                super().__init__()
                input_dim = state_dim + action_dim
                
                self.q1 = self._build_mlp(input_dim, 1, hidden_dims, activation)
                self.q2 = self._build_mlp(input_dim, 1, hidden_dims, activation)
                
            def _build_mlp(self, input_dim, output_dim, hidden_dims, activation):
                layers = []
                dims = [input_dim] + hidden_dims + [output_dim]
                
                for i in range(len(dims) - 1):
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                    if i < len(dims) - 2:
                        layers.append(activation)
                        
                return nn.Sequential(*layers)
                
            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                return self.q1(x), self.q2(x)
                
        return TwinCritic(self.state_dim, self.action_dim, self.hidden_dims, self.activation)
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action using current policy."""
        state = self.to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self._get_action_and_log_prob(state, deterministic=eval_mode)
            
        return self.to_numpy(action.squeeze(0))
        
    def _get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False):
        """Get action and log probability from policy."""
        actor_output = self.actor(state)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
        return action, log_prob
        
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Update CQL with batch of data."""
        self.training_step += 1
        
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        terminals = batch["terminals"]
        
        # Update critic
        critic_loss, q_loss, cql_loss = self._update_critic(
            states, actions, rewards, next_states, terminals
        )
        
        # Update actor
        actor_loss = self._update_actor(states)
        
        # Update target networks
        self._update_target_networks()
        
        # Update lagrange multiplier
        alpha_prime_loss = 0.0
        if self.with_lagrange:
            alpha_prime_loss = self._update_lagrange_multiplier(cql_loss)
            
        return TrainingMetrics(
            loss=critic_loss + actor_loss,
            q_loss=q_loss,
            policy_loss=actor_loss,
            alpha_loss=alpha_prime_loss
        )
        
    def _update_critic(self, states, actions, rewards, next_states, terminals):
        """Update critic networks."""
        with torch.no_grad():
            # Target Q-values
            next_actions, next_log_probs = self._get_action_and_log_prob(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - terminals) * target_q
            
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Bellman loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss
        
        # CQL regularization
        cql_loss = self._compute_cql_loss(states, actions, current_q1, current_q2)
        
        # Total critic loss
        if self.with_lagrange:
            alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime), min=0.0, max=1e6)
            critic_loss = q_loss + alpha_prime * (cql_loss - self.lagrange_threshold)
        else:
            critic_loss = q_loss + self.conservative_weight * cql_loss
            
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item(), q_loss.item(), cql_loss.item()
        
    def _compute_cql_loss(self, states, actions, q1_values, q2_values):
        """Compute CQL regularization loss."""
        batch_size = states.shape[0]
        
        # Sample random actions
        random_actions = torch.FloatTensor(
            batch_size, self.action_dim
        ).uniform_(-1, 1).to(self.device)
        
        # Sample actions from current policy
        policy_actions, _ = self._get_action_and_log_prob(states)
        
        # Q-values for random and policy actions
        random_q1, random_q2 = self.critic(states, random_actions)
        policy_q1, policy_q2 = self.critic(states, policy_actions)
        
        # CQL loss: log-sum-exp of Q-values minus dataset Q-values
        cat_q1 = torch.cat([random_q1, policy_q1, q1_values], dim=0)
        cat_q2 = torch.cat([random_q2, policy_q2, q2_values], dim=0)
        
        cql_q1_loss = torch.logsumexp(cat_q1, dim=0) - q1_values.mean()
        cql_q2_loss = torch.logsumexp(cat_q2, dim=0) - q2_values.mean()
        
        return cql_q1_loss + cql_q2_loss
        
    def _update_actor(self, states):
        """Update actor network."""
        actions, log_probs = self._get_action_and_log_prob(states)
        q1, q2 = self.critic(states, actions)
        q_values = torch.min(q1, q2)
        
        # Actor loss: maximize Q-values, minimize entropy
        actor_loss = (self.alpha * log_probs - q_values).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
        
    def _update_target_networks(self):
        """Soft update target networks."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def _update_lagrange_multiplier(self, cql_loss):
        """Update Lagrange multiplier for CQL."""
        alpha_prime = torch.clamp(torch.exp(self.log_alpha_prime), min=0.0, max=1e6)
        alpha_prime_loss = alpha_prime * (cql_loss - self.lagrange_threshold).detach()
        
        self.alpha_prime_optimizer.zero_grad()
        alpha_prime_loss.backward()
        self.alpha_prime_optimizer.step()
        
        return alpha_prime_loss.item()
        
    def train_offline(
        self,
        dataset: GridDataset,
        num_epochs: int,
        batch_size: int = 256,
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train CQL on offline dataset."""
        metrics_history = []
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            # Number of batches per epoch
            num_batches = max(1, dataset.size // batch_size)
            
            for _ in range(num_batches):
                batch = dataset.sample_batch(batch_size)
                metrics = self.update(batch)
                epoch_metrics.append(metrics)
                
            # Average metrics for epoch
            avg_metrics = TrainingMetrics(
                loss=np.mean([m.loss for m in epoch_metrics]),
                q_loss=np.mean([m.q_loss for m in epoch_metrics]),
                policy_loss=np.mean([m.policy_loss for m in epoch_metrics]),
                alpha_loss=np.mean([m.alpha_loss for m in epoch_metrics])
            )
            
            metrics_history.append(avg_metrics)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={avg_metrics.loss:.4f}, "
                      f"Q-Loss={avg_metrics.q_loss:.4f}, "
                      f"Policy-Loss={avg_metrics.policy_loss:.4f}")
                      
        return metrics_history
        
    def save(self, path: str) -> None:
        """Save model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_step": self.training_step
        }, path)
        
    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint["training_step"]


class IQL(OfflineRLAlgorithm, ActorCriticBase):
    """Implicit Q-Learning (IQL) for offline RL."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.7,
        temperature: float = 3.0,
        device: str = "auto",
        **kwargs
    ) -> None:
        super().__init__(state_dim, action_dim, device=device, **kwargs)
        ActorCriticBase.__init__(self, state_dim, action_dim, hidden_dims, device=device, **kwargs)
        
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=lr)
        
    def _build_networks(self):
        """Build actor, critic, and value networks."""
        # Actor
        self.actor = self._build_actor().to(self.device)
        
        # Critic (twin Q-networks)
        self.critic = self._build_twin_critic().to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Value function
        self.value_function = self._build_mlp(
            self.state_dim, 1, self.hidden_dims, self.activation
        ).to(self.device)
        
        # Freeze target network
        for param in self.critic_target.parameters():
            param.requires_grad = False
            
    def _build_actor(self) -> nn.Module:
        """Build actor network."""
        return self._build_mlp(
            self.state_dim,
            self.action_dim * 2,  # Mean and log_std
            self.hidden_dims,
            self.activation
        )
        
    def _build_twin_critic(self) -> nn.Module:
        """Build twin Q-networks."""
        class TwinCritic(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims, activation):
                super().__init__()
                input_dim = state_dim + action_dim
                
                # Build MLPs
                self.q1_layers = []
                self.q2_layers = []
                dims = [input_dim] + hidden_dims + [1]
                
                for i in range(len(dims) - 1):
                    self.q1_layers.append(nn.Linear(dims[i], dims[i + 1]))
                    self.q2_layers.append(nn.Linear(dims[i], dims[i + 1]))
                    
                self.q1 = nn.ModuleList(self.q1_layers)
                self.q2 = nn.ModuleList(self.q2_layers)
                self.activation = activation
                
            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                
                # Q1
                q1_out = x
                for i, layer in enumerate(self.q1):
                    q1_out = layer(q1_out)
                    if i < len(self.q1) - 1:
                        q1_out = self.activation(q1_out)
                        
                # Q2
                q2_out = x
                for i, layer in enumerate(self.q2):
                    q2_out = layer(q2_out)
                    if i < len(self.q2) - 1:
                        q2_out = self.activation(q2_out)
                        
                return q1_out, q2_out
                
        return TwinCritic(self.state_dim, self.action_dim, self.hidden_dims, self.activation)
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action using current policy."""
        state = self.to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self._get_action_and_log_prob(state, deterministic=eval_mode)
            
        return self.to_numpy(action.squeeze(0))
        
    def _get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False):
        """Get action and log probability from policy."""
        actor_output = self.actor(state)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
        return action, log_prob
        
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Update IQL with batch of data."""
        self.training_step += 1
        
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        terminals = batch["terminals"]
        
        # Update value function
        value_loss = self._update_value_function(states, actions)
        
        # Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, terminals)
        
        # Update actor
        actor_loss = self._update_actor(states, actions)
        
        # Update target networks
        self._update_target_networks()
        
        return TrainingMetrics(
            loss=value_loss + critic_loss + actor_loss,
            q_loss=critic_loss,
            policy_loss=actor_loss
        )
        
    def _update_value_function(self, states, actions):
        """Update value function using expectile regression."""
        with torch.no_grad():
            q1, q2 = self.critic_target(states, actions)
            q_values = torch.min(q1, q2)
            
        v_values = self.value_function(states)
        
        # Expectile loss
        diff = q_values - v_values
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        value_loss = (weight * diff.pow(2)).mean()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return value_loss.item()
        
    def _update_critic(self, states, actions, rewards, next_states, terminals):
        """Update critic networks."""
        with torch.no_grad():
            target_v = self.value_function(next_states)
            target_q = rewards + self.gamma * (1 - terminals) * target_v
            
        current_q1, current_q2 = self.critic(states, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
        
    def _update_actor(self, states, actions):
        """Update actor using advantage weighted regression."""
        with torch.no_grad():
            v_values = self.value_function(states)
            q1, q2 = self.critic(states, actions)
            q_values = torch.min(q1, q2)
            
            advantages = q_values - v_values
            weights = torch.exp(advantages / self.temperature)
            weights = torch.clamp(weights, max=100.0)  # Prevent overflow
            
        # Get action log probabilities
        _, log_probs = self._get_action_and_log_prob(states)
        
        # Weighted behavior cloning loss
        actor_loss = -(weights * log_probs).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
        
    def _update_target_networks(self):
        """Soft update target networks."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def train_offline(
        self,
        dataset: GridDataset,
        num_epochs: int,
        batch_size: int = 256,
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train IQL on offline dataset."""
        metrics_history = []
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            num_batches = max(1, dataset.size // batch_size)
            
            for _ in range(num_batches):
                batch = dataset.sample_batch(batch_size)
                metrics = self.update(batch)
                epoch_metrics.append(metrics)
                
            avg_metrics = TrainingMetrics(
                loss=np.mean([m.loss for m in epoch_metrics]),
                q_loss=np.mean([m.q_loss for m in epoch_metrics]),
                policy_loss=np.mean([m.policy_loss for m in epoch_metrics])
            )
            
            metrics_history.append(avg_metrics)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={avg_metrics.loss:.4f}")
                
        return metrics_history
        
    def save(self, path: str) -> None:
        """Save model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "value_function": self.value_function.state_dict(),
            "training_step": self.training_step
        }, path)
        
    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.value_function.load_state_dict(checkpoint["value_function"])
        self.training_step = checkpoint["training_step"]


class AWR(OfflineRLAlgorithm):
    """Advantage Weighted Regression (AWR) for offline RL."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        temperature: float = 1.0,
        device: str = "auto",
        **kwargs
    ) -> None:
        super().__init__(state_dim, action_dim, device=device, **kwargs)
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.temperature = temperature
        self.hidden_dims = hidden_dims
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def _build_networks(self):
        """Build actor and critic networks."""
        # Actor (policy)
        actor_layers = []
        dims = [self.state_dim] + self.hidden_dims + [self.action_dim * 2]
        
        for i in range(len(dims) - 1):
            actor_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                actor_layers.append(nn.ReLU())
                
        self.actor = nn.Sequential(*actor_layers).to(self.device)
        
        # Critic (value function)
        critic_layers = []
        dims = [self.state_dim] + self.hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            critic_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                critic_layers.append(nn.ReLU())
                
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action using current policy."""
        state = self.to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self._get_action_and_log_prob(state, deterministic=eval_mode)
            
        return self.to_numpy(action.squeeze(0))
        
    def _get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False):
        """Get action and log probability."""
        actor_output = self.actor(state)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
        return action, log_prob
        
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Update AWR with batch of data."""
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        
        # Compute advantages
        with torch.no_grad():
            values = self.critic(states)
            # Simplified advantage computation (could use GAE)
            advantages = rewards - values.squeeze()
            
        # Update critic
        critic_loss = F.mse_loss(values.squeeze(), rewards)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor with advantage weighting
        _, log_probs = self._get_action_and_log_prob(states)
        
        # AWR loss with exponential weighting
        weights = torch.exp(advantages / self.temperature)
        weights = torch.clamp(weights, max=20.0)  # Prevent overflow
        
        actor_loss = -(weights.detach() * log_probs.squeeze()).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return TrainingMetrics(
            loss=critic_loss.item() + actor_loss.item(),
            q_loss=critic_loss.item(),
            policy_loss=actor_loss.item()
        )
        
    def train_offline(
        self,
        dataset: GridDataset,
        num_epochs: int,
        batch_size: int = 256,
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train AWR on offline dataset."""
        metrics_history = []
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            num_batches = max(1, dataset.size // batch_size)
            
            for _ in range(num_batches):
                batch = dataset.sample_batch(batch_size)
                metrics = self.update(batch)
                epoch_metrics.append(metrics)
                
            avg_metrics = TrainingMetrics(
                loss=np.mean([m.loss for m in epoch_metrics]),
                q_loss=np.mean([m.q_loss for m in epoch_metrics]),
                policy_loss=np.mean([m.policy_loss for m in epoch_metrics])
            )
            
            metrics_history.append(avg_metrics)
            
        return metrics_history
        
    def save(self, path: str) -> None:
        """Save model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "training_step": self.training_step
        }, path)
        
    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.training_step = checkpoint["training_step"]