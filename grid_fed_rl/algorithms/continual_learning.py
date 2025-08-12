"""Continual Federated Learning for Adaptive Grid Evolution.

This module implements continual learning algorithms that enable federated RL systems
to adapt to evolving grid infrastructure, changing load patterns, and new renewable
installations without catastrophic forgetting.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import copy
import pickle

from .base import OfflineRLAlgorithm, ActorCriticBase, TrainingMetrics
from ..federated.core import FederatedClient, ClientUpdate
from ..utils.validation import validate_constraints


@dataclass
class Task:
    """Definition of a continual learning task."""
    task_id: str
    description: str
    start_time: float
    data_distribution: Dict[str, Any]
    grid_configuration: Dict[str, Any]
    performance_threshold: float = 0.8


@dataclass
class Memory:
    """Memory buffer for continual learning."""
    states: List[torch.Tensor]
    actions: List[torch.Tensor]
    rewards: List[torch.Tensor]
    task_ids: List[str]
    importance_weights: List[float]
    timestamps: List[float]


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""
    
    def __init__(self, model: nn.Module, dataset: DataLoader, importance_threshold: float = 1e-3):
        self.model = model
        self.importance_threshold = importance_threshold
        self.fisher_information = {}
        self.optimal_parameters = {}
        
        self._compute_fisher_information(dataset)
        self._store_optimal_parameters()
    
    def _compute_fisher_information(self, dataset: DataLoader):
        """Compute Fisher Information Matrix diagonal approximation."""
        self.fisher_information = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param)
        
        # Compute Fisher information from dataset
        self.model.train()
        num_samples = 0
        
        for batch in dataset:
            self.model.zero_grad()
            
            # Forward pass to get log probabilities
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                outputs = self.model(inputs)
            else:
                outputs = self.model(batch)
            
            # Compute log likelihood (simplified)
            if outputs.dim() > 1:
                log_likelihood = F.log_softmax(outputs, dim=-1)
                # Sample from categorical distribution
                sampled_outputs = torch.multinomial(F.softmax(outputs, dim=-1), 1).squeeze()
                loss = F.nll_loss(log_likelihood, sampled_outputs)
            else:
                # For regression-like outputs
                loss = torch.sum(outputs**2) / outputs.numel()
            
            # Backward pass
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
                    
            num_samples += inputs.size(0) if hasattr(inputs, 'size') else 1
        
        # Normalize by number of samples
        if num_samples > 0:
            for name in self.fisher_information:
                self.fisher_information[name] /= num_samples
                
                # Clamp to prevent numerical issues
                self.fisher_information[name] = torch.clamp(
                    self.fisher_information[name], min=self.importance_threshold
                )
    
    def _store_optimal_parameters(self):
        """Store current parameters as optimal for the previous task."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.optimal_parameters[name] = param.data.clone()
    
    def compute_ewc_loss(self, ewc_lambda: float = 1000.0) -> torch.Tensor:
        """Compute EWC regularization loss."""
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.optimal_parameters:
                fisher = self.fisher_information.get(name, torch.zeros_like(param))
                optimal = self.optimal_parameters[name]
                ewc_loss += torch.sum(fisher * (param - optimal) ** 2) / 2
        
        return ewc_lambda * ewc_loss


class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Network for continual learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        max_columns: int = 5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.max_columns = max_columns
        self.num_columns = 1
        
        # Progressive columns
        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        
        # Initialize first column
        self._add_column()
    
    def _add_column(self):
        """Add a new column to the progressive network."""
        if self.num_columns >= self.max_columns:
            return False
        
        # Create new column
        layers = []
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Add input from previous columns
            if self.num_columns > 1:
                lateral_input_dim = hidden_dim * (self.num_columns - 1)
                total_input_dim = prev_dim + lateral_input_dim
            else:
                total_input_dim = prev_dim
            
            layers.append(nn.Linear(total_input_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        column = nn.Sequential(*layers)
        self.columns.append(column)
        
        # Add lateral connections if not first column
        if self.num_columns > 1:
            lateral_layers = nn.ModuleList()
            for i in range(len(self.hidden_dims)):
                lateral_layer = nn.Linear(
                    self.hidden_dims[i], 
                    self.hidden_dims[i] * (self.num_columns - 1)
                )
                lateral_layers.append(lateral_layer)
            self.lateral_connections.append(lateral_layers)
        
        self.num_columns += 1
        return True
    
    def forward(self, x: torch.Tensor, column_idx: Optional[int] = None) -> torch.Tensor:
        """Forward pass through progressive network."""
        if column_idx is None:
            column_idx = self.num_columns - 1  # Use latest column
        
        if column_idx >= self.num_columns:
            column_idx = self.num_columns - 1
        
        # Store activations from previous columns
        prev_activations = []
        
        # Forward through all columns up to target column
        for col_idx in range(column_idx + 1):
            if col_idx == 0:
                # First column uses only input
                output = self.columns[col_idx](x)
            else:
                # Later columns use input + lateral connections
                lateral_inputs = []
                
                # Get lateral connections from previous columns
                for prev_col_idx in range(col_idx):
                    if prev_col_idx < len(prev_activations):
                        lateral_input = self.lateral_connections[col_idx - 1][0](
                            prev_activations[prev_col_idx]
                        )
                        lateral_inputs.append(lateral_input)
                
                if lateral_inputs:
                    combined_lateral = torch.cat(lateral_inputs, dim=-1)
                    combined_input = torch.cat([x, combined_lateral], dim=-1)
                else:
                    combined_input = x
                
                output = self.columns[col_idx](combined_input)
            
            # Store intermediate activations for lateral connections
            if col_idx < column_idx:
                # Extract hidden layer activations (excluding output layer)
                hidden_activations = []
                current_input = x
                
                for layer_idx in range(0, len(self.columns[col_idx]) - 1, 2):
                    current_input = self.columns[col_idx][layer_idx:layer_idx+2](current_input)
                    if layer_idx // 2 < len(self.hidden_dims):
                        hidden_activations.append(current_input)
                
                if hidden_activations:
                    prev_activations.append(hidden_activations[-1])  # Use last hidden layer
        
        return output
    
    def freeze_columns(self, num_columns: int):
        """Freeze parameters in the first num_columns."""
        for col_idx in range(min(num_columns, self.num_columns)):
            for param in self.columns[col_idx].parameters():
                param.requires_grad = False


class ContinualFederatedRL(OfflineRLAlgorithm, ActorCriticBase):
    """Continual Federated Reinforcement Learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        lr: float = 3e-4,
        gamma: float = 0.99,
        continual_method: str = "ewc",  # "ewc", "progressive", "replay", "multitask"
        memory_size: int = 10000,
        ewc_lambda: float = 1000.0,
        replay_ratio: float = 0.2,
        device: str = "auto",
        **kwargs
    ):
        super().__init__(state_dim, action_dim, device=device, **kwargs)
        ActorCriticBase.__init__(self, state_dim, action_dim, hidden_dims, device=device, **kwargs)
        
        self.lr = lr
        self.gamma = gamma
        self.continual_method = continual_method
        self.memory_size = memory_size
        self.ewc_lambda = ewc_lambda
        self.replay_ratio = replay_ratio
        
        # Task management
        self.current_task: Optional[Task] = None
        self.task_history: List[Task] = []
        self.task_performance: Dict[str, List[float]] = {}
        
        # Continual learning components
        self.memory_buffer = Memory([], [], [], [], [], [])
        self.ewc_regularizers: Dict[str, ElasticWeightConsolidation] = {}
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)
        
        # Continual learning metrics
        self.forgetting_measures = []
        self.task_transfer_scores = []
        self.memory_utilization = []
        
        self.logger = logging.getLogger(__name__)
    
    def _build_networks(self):
        """Build continual learning networks."""
        if self.continual_method == "progressive":
            # Progressive neural networks
            self.actor = ProgressiveNeuralNetwork(
                self.state_dim, self.hidden_dims, self.action_dim * 2
            ).to(self.device)
            
            self.critic = ProgressiveNeuralNetwork(
                self.state_dim + self.action_dim, self.hidden_dims, 1
            ).to(self.device)
            
        else:
            # Standard networks with regularization
            self.actor = self._build_actor_network().to(self.device)
            self.critic = self._build_critic_network().to(self.device)
    
    def _build_actor_network(self) -> nn.Module:
        """Build standard actor network."""
        layers = []
        prev_dim = self.state_dim
        
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, self.action_dim * 2))
        return nn.Sequential(*layers)
    
    def _build_critic_network(self) -> nn.Module:
        """Build standard critic network."""
        layers = []
        prev_dim = self.state_dim + self.action_dim
        
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def start_new_task(
        self,
        task: Task,
        previous_task_data: Optional[DataLoader] = None
    ) -> bool:
        """Start learning a new task."""
        self.logger.info(f"Starting new task: {task.task_id} - {task.description}")
        
        # Store EWC regularizer for previous task
        if self.current_task is not None and previous_task_data is not None:
            if self.continual_method == "ewc":
                self._store_ewc_regularizer(previous_task_data)
            elif self.continual_method == "progressive":
                self._handle_progressive_task_switch()
        
        # Update task information
        if self.current_task is not None:
            self.task_history.append(self.current_task)
        
        self.current_task = task
        self.task_performance[task.task_id] = []
        
        # Initialize task-specific components
        if self.continual_method == "progressive":
            success = self._add_progressive_column()
            if not success:
                self.logger.warning("Could not add new column, using existing capacity")
        
        return True
    
    def _store_ewc_regularizer(self, dataset: DataLoader):
        """Store EWC regularizer for current task."""
        if self.current_task is None:
            return
        
        # Create EWC regularizers for actor and critic
        self.ewc_regularizers[f"{self.current_task.task_id}_actor"] = ElasticWeightConsolidation(
            self.actor, dataset
        )
        self.ewc_regularizers[f"{self.current_task.task_id}_critic"] = ElasticWeightConsolidation(
            self.critic, dataset
        )
    
    def _handle_progressive_task_switch(self):
        """Handle task switching for progressive networks."""
        if hasattr(self.actor, 'freeze_columns'):
            # Freeze previous columns
            self.actor.freeze_columns(self.actor.num_columns - 1)
            self.critic.freeze_columns(self.critic.num_columns - 1)
    
    def _add_progressive_column(self) -> bool:
        """Add new column for progressive networks."""
        actor_success = self.actor._add_column()
        critic_success = self.critic._add_column()
        
        if actor_success and critic_success:
            # Update optimizers to include new parameters
            self.actor_optimizer.add_param_group({
                'params': list(self.actor.columns[-1].parameters())
            })
            self.critic_optimizer.add_param_group({
                'params': list(self.critic.columns[-1].parameters())
            })
        
        return actor_success and critic_success
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action with continual learning considerations."""
        state_tensor = self.to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if self.continual_method == "progressive":
                actor_output = self.actor(state_tensor)
            else:
                actor_output = self.actor(state_tensor)
            
            mean, log_std = torch.chunk(actor_output, 2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            
            if eval_mode:
                action = torch.tanh(mean)
            else:
                normal = torch.distributions.Normal(mean, std)
                action = torch.tanh(normal.sample())
        
        return self.to_numpy(action.squeeze(0))
    
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Continual learning update step."""
        self.training_step += 1
        
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        terminals = batch["terminals"]
        
        # Store data in memory buffer
        self._update_memory_buffer(states, actions, rewards)
        
        # Prepare training data with replay
        train_states, train_actions, train_rewards, train_next_states, train_terminals = \
            self._prepare_continual_training_data(states, actions, rewards, next_states, terminals)
        
        # Standard RL updates
        critic_loss = self._update_critic_continual(
            train_states, train_actions, train_rewards, train_next_states, train_terminals
        )
        
        actor_loss = self._update_actor_continual(train_states)
        
        # Add continual learning regularization
        continual_loss = self._compute_continual_regularization()
        
        # Track performance for current task
        if self.current_task is not None:
            task_performance = torch.mean(train_rewards).item()
            self.task_performance[self.current_task.task_id].append(task_performance)
        
        # Compute forgetting measure
        self._update_forgetting_measures()
        
        total_loss = critic_loss + actor_loss + continual_loss
        
        return TrainingMetrics(
            loss=total_loss,
            q_loss=critic_loss,
            policy_loss=actor_loss + continual_loss,
            mean_q_value=torch.mean(train_rewards).item()
        )
    
    def _update_memory_buffer(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ):
        """Update memory buffer for experience replay."""
        if len(self.memory_buffer.states) >= self.memory_size:
            # Remove oldest experiences
            num_to_remove = len(self.memory_buffer.states) - self.memory_size + states.shape[0]
            for _ in range(num_to_remove):
                if self.memory_buffer.states:
                    self.memory_buffer.states.pop(0)
                    self.memory_buffer.actions.pop(0)
                    self.memory_buffer.rewards.pop(0)
                    self.memory_buffer.task_ids.pop(0)
                    self.memory_buffer.importance_weights.pop(0)
                    self.memory_buffer.timestamps.pop(0)
        
        # Add new experiences
        current_time = self.training_step
        current_task_id = self.current_task.task_id if self.current_task else "unknown"
        
        for i in range(states.shape[0]):
            self.memory_buffer.states.append(states[i].clone())
            self.memory_buffer.actions.append(actions[i].clone())
            self.memory_buffer.rewards.append(rewards[i].clone())
            self.memory_buffer.task_ids.append(current_task_id)
            self.memory_buffer.importance_weights.append(1.0)  # Can be adjusted based on importance
            self.memory_buffer.timestamps.append(current_time)
    
    def _prepare_continual_training_data(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminals: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Prepare training data with experience replay."""
        
        if self.continual_method == "replay" and len(self.memory_buffer.states) > 0:
            # Sample from memory buffer
            num_replay = int(states.shape[0] * self.replay_ratio)
            if num_replay > 0 and num_replay <= len(self.memory_buffer.states):
                
                # Sample indices
                replay_indices = np.random.choice(
                    len(self.memory_buffer.states), num_replay, replace=False
                )
                
                # Get replay data
                replay_states = torch.stack([self.memory_buffer.states[i] for i in replay_indices])
                replay_actions = torch.stack([self.memory_buffer.actions[i] for i in replay_indices])
                replay_rewards = torch.stack([self.memory_buffer.rewards[i] for i in replay_indices])
                
                # For simplicity, use current next_states and terminals for replay
                replay_next_states = next_states[:num_replay]
                replay_terminals = terminals[:num_replay]
                
                # Combine current and replay data
                combined_states = torch.cat([states, replay_states], dim=0)
                combined_actions = torch.cat([actions, replay_actions], dim=0)
                combined_rewards = torch.cat([rewards, replay_rewards], dim=0)
                combined_next_states = torch.cat([next_states, replay_next_states], dim=0)
                combined_terminals = torch.cat([terminals, replay_terminals], dim=0)
                
                return combined_states, combined_actions, combined_rewards, combined_next_states, combined_terminals
        
        # Return original data if no replay
        return states, actions, rewards, next_states, terminals
    
    def _update_critic_continual(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminals: torch.Tensor
    ) -> float:
        """Update critic with continual learning considerations."""
        
        with torch.no_grad():
            # Get next actions
            if self.continual_method == "progressive":
                next_actor_output = self.actor(next_states)
            else:
                next_actor_output = self.actor(next_states)
            
            next_mean, next_log_std = torch.chunk(next_actor_output, 2, dim=-1)
            next_std = torch.exp(torch.clamp(next_log_std, -20, 2))
            next_actions = torch.tanh(torch.distributions.Normal(next_mean, next_std).sample())
            
            # Target Q-values
            if self.continual_method == "progressive":
                next_state_actions = torch.cat([next_states, next_actions], dim=-1)
                target_q = self.critic(next_state_actions)
            else:
                next_state_actions = torch.cat([next_states, next_actions], dim=-1)
                target_q = self.critic(next_state_actions)
            
            target_values = rewards.unsqueeze(-1) + self.gamma * (1 - terminals.unsqueeze(-1)) * target_q
        
        # Current Q-values
        if self.continual_method == "progressive":
            state_actions = torch.cat([states, actions], dim=-1)
            current_q = self.critic(state_actions)
        else:
            state_actions = torch.cat([states, actions], dim=-1)
            current_q = self.critic(state_actions)
        
        # Bellman loss
        critic_loss = F.mse_loss(current_q, target_values)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor_continual(self, states: torch.Tensor) -> float:
        """Update actor with continual learning considerations."""
        
        if self.continual_method == "progressive":
            actor_output = self.actor(states)
        else:
            actor_output = self.actor(states)
        
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        std = torch.exp(torch.clamp(log_std, -20, 2))
        actions = torch.tanh(torch.distributions.Normal(mean, std).rsample())
        
        # Q-values for policy improvement
        if self.continual_method == "progressive":
            state_actions = torch.cat([states, actions], dim=-1)
            q_values = self.critic(state_actions)
        else:
            state_actions = torch.cat([states, actions], dim=-1)
            q_values = self.critic(state_actions)
        
        actor_loss = -q_values.mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _compute_continual_regularization(self) -> float:
        """Compute continual learning regularization terms."""
        continual_loss = 0.0
        
        if self.continual_method == "ewc":
            # EWC regularization
            for regularizer_name, ewc_reg in self.ewc_regularizers.items():
                if "actor" in regularizer_name:
                    ewc_loss = ewc_reg.compute_ewc_loss(self.ewc_lambda)
                    continual_loss += ewc_loss.item()
                elif "critic" in regularizer_name:
                    ewc_loss = ewc_reg.compute_ewc_loss(self.ewc_lambda)
                    continual_loss += ewc_loss.item()
        
        return continual_loss
    
    def _update_forgetting_measures(self):
        """Update forgetting measures across tasks."""
        if len(self.task_history) < 2:
            return
        
        # Simple forgetting measure: performance drop on previous tasks
        total_forgetting = 0.0
        num_previous_tasks = 0
        
        for prev_task in self.task_history[:-1]:  # Exclude current task
            task_id = prev_task.task_id
            if task_id in self.task_performance and len(self.task_performance[task_id]) >= 2:
                initial_performance = np.mean(self.task_performance[task_id][:10])
                recent_performance = np.mean(self.task_performance[task_id][-10:])
                
                forgetting = max(0.0, initial_performance - recent_performance)
                total_forgetting += forgetting
                num_previous_tasks += 1
        
        if num_previous_tasks > 0:
            avg_forgetting = total_forgetting / num_previous_tasks
            self.forgetting_measures.append(avg_forgetting)
    
    def evaluate_task_transfer(self, evaluation_data: Dict[str, DataLoader]) -> Dict[str, float]:
        """Evaluate transfer learning performance across tasks."""
        transfer_scores = {}
        
        with torch.no_grad():
            for task_id, dataloader in evaluation_data.items():
                total_reward = 0.0
                num_samples = 0
                
                for batch in dataloader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                        states, actions, rewards = batch[:3]
                    else:
                        continue
                    
                    # Evaluate current policy on task data
                    if self.continual_method == "progressive":
                        # Find appropriate column for this task
                        if task_id == self.current_task.task_id:
                            column_idx = None  # Use latest column
                        else:
                            # Use column associated with this task (simplified)
                            column_idx = 0  # Default to first column
                        
                        actor_output = self.actor(states, column_idx)
                    else:
                        actor_output = self.actor(states)
                    
                    mean, log_std = torch.chunk(actor_output, 2, dim=-1)
                    predicted_actions = torch.tanh(mean)
                    
                    # Simple reward proxy (could be more sophisticated)
                    action_similarity = F.cosine_similarity(predicted_actions, actions, dim=-1)
                    task_reward = torch.mean(action_similarity).item()
                    
                    total_reward += task_reward * states.shape[0]
                    num_samples += states.shape[0]
                
                if num_samples > 0:
                    transfer_scores[task_id] = total_reward / num_samples
                else:
                    transfer_scores[task_id] = 0.0
        
        return transfer_scores
    
    def get_continual_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive continual learning metrics."""
        metrics = {
            "current_task": self.current_task.task_id if self.current_task else None,
            "num_completed_tasks": len(self.task_history),
            "total_tasks": len(self.task_history) + (1 if self.current_task else 0),
            "continual_method": self.continual_method,
            "memory_utilization": len(self.memory_buffer.states) / self.memory_size if self.memory_size > 0 else 0.0,
            "average_forgetting": np.mean(self.forgetting_measures) if self.forgetting_measures else 0.0,
            "forgetting_trend": np.polyfit(range(len(self.forgetting_measures)), self.forgetting_measures, 1)[0] if len(self.forgetting_measures) > 1 else 0.0
        }
        
        # Task-specific performance
        if self.current_task and self.current_task.task_id in self.task_performance:
            recent_performance = self.task_performance[self.current_task.task_id][-10:]
            metrics["current_task_performance"] = {
                "mean": np.mean(recent_performance) if recent_performance else 0.0,
                "std": np.std(recent_performance) if recent_performance else 0.0,
                "trend": np.polyfit(range(len(recent_performance)), recent_performance, 1)[0] if len(recent_performance) > 1 else 0.0
            }
        
        # Memory diversity
        if len(self.memory_buffer.task_ids) > 0:
            task_counts = {}
            for task_id in self.memory_buffer.task_ids:
                task_counts[task_id] = task_counts.get(task_id, 0) + 1
            
            metrics["memory_diversity"] = {
                "num_tasks_in_memory": len(task_counts),
                "task_distribution": task_counts
            }
        
        # Progressive network specific metrics
        if self.continual_method == "progressive" and hasattr(self.actor, 'num_columns'):
            metrics["progressive_network"] = {
                "num_columns": self.actor.num_columns,
                "column_utilization": self.actor.num_columns / self.actor.max_columns
            }
        
        return metrics
    
    def train_offline(
        self,
        dataset,
        num_epochs: int,
        batch_size: int = 256,
        task_info: Optional[Task] = None,
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train with continual learning considerations."""
        if task_info and task_info != self.current_task:
            # Create dummy dataloader for EWC if switching tasks
            if self.current_task is not None:
                dummy_data = TensorDataset(
                    torch.randn(100, self.state_dim),
                    torch.randn(100, self.action_dim)
                )
                dummy_loader = DataLoader(dummy_data, batch_size=32)
                self.start_new_task(task_info, dummy_loader)
            else:
                self.start_new_task(task_info)
        
        metrics_history = []
        
        self.logger.info(
            f"Starting continual learning training: {num_epochs} epochs, "
            f"method={self.continual_method}, current_task={self.current_task.task_id if self.current_task else 'None'}"
        )
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            num_batches = max(1, dataset.size // batch_size)
            
            for _ in range(num_batches):
                batch = dataset.sample_batch(batch_size)
                metrics = self.update(batch)
                epoch_metrics.append(metrics)
            
            # Average epoch metrics
            avg_metrics = TrainingMetrics(
                loss=np.mean([m.loss for m in epoch_metrics]),
                q_loss=np.mean([m.q_loss for m in epoch_metrics]),
                policy_loss=np.mean([m.policy_loss for m in epoch_metrics])
            )
            
            metrics_history.append(avg_metrics)
            
            if epoch % 50 == 0:
                cl_metrics = self.get_continual_learning_metrics()
                forgetting = cl_metrics.get("average_forgetting", 0.0)
                memory_util = cl_metrics.get("memory_utilization", 0.0)
                
                self.logger.info(
                    f"Epoch {epoch}: Loss={avg_metrics.loss:.4f}, "
                    f"Forgetting={forgetting:.4f}, "
                    f"MemUtil={memory_util:.3f}"
                )
        
        return metrics_history
    
    def save(self, path: str) -> None:
        """Save continual learning model."""
        save_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "current_task": self.current_task,
            "task_history": self.task_history,
            "task_performance": self.task_performance,
            "forgetting_measures": self.forgetting_measures,
            "continual_method": self.continual_method,
            "training_step": self.training_step
        }
        
        # Save memory buffer
        if len(self.memory_buffer.states) > 0:
            save_dict["memory_buffer"] = {
                "states": self.memory_buffer.states,
                "actions": self.memory_buffer.actions,
                "rewards": self.memory_buffer.rewards,
                "task_ids": self.memory_buffer.task_ids,
                "importance_weights": self.memory_buffer.importance_weights,
                "timestamps": self.memory_buffer.timestamps
            }
        
        # Save EWC regularizers
        if self.ewc_regularizers:
            save_dict["ewc_regularizers"] = {}
            for name, ewc_reg in self.ewc_regularizers.items():
                save_dict["ewc_regularizers"][name] = {
                    "fisher_information": ewc_reg.fisher_information,
                    "optimal_parameters": ewc_reg.optimal_parameters
                }
        
        torch.save(save_dict, path)
    
    def load(self, path: str) -> None:
        """Load continual learning model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.current_task = checkpoint.get("current_task")
        self.task_history = checkpoint.get("task_history", [])
        self.task_performance = checkpoint.get("task_performance", {})
        self.forgetting_measures = checkpoint.get("forgetting_measures", [])
        self.training_step = checkpoint.get("training_step", 0)
        
        # Load memory buffer
        if "memory_buffer" in checkpoint:
            mb = checkpoint["memory_buffer"]
            self.memory_buffer.states = mb["states"]
            self.memory_buffer.actions = mb["actions"]
            self.memory_buffer.rewards = mb["rewards"]
            self.memory_buffer.task_ids = mb["task_ids"]
            self.memory_buffer.importance_weights = mb["importance_weights"]
            self.memory_buffer.timestamps = mb["timestamps"]
        
        # Load EWC regularizers (would need to reconstruct properly)
        if "ewc_regularizers" in checkpoint:
            # Note: This is a simplified loading - in practice, you'd need to
            # reconstruct the EWC objects properly with the saved fisher information
            self.logger.info("EWC regularizers found in checkpoint (simplified loading)")


class ContinualFederatedClient(FederatedClient):
    """Federated client with continual learning capabilities."""
    
    def __init__(
        self,
        client_id: str,
        algorithm: ContinualFederatedRL,
        grid_data: List[Dict[str, Any]],
        task_sequence: List[Task],
        adaptation_threshold: float = 0.1
    ):
        super().__init__(client_id, algorithm)
        self.local_data = grid_data
        self.task_sequence = task_sequence
        self.adaptation_threshold = adaptation_threshold
        self.current_task_index = 0
        self.adaptation_scores = []
    
    def local_update(
        self,
        global_parameters: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int
    ) -> ClientUpdate:
        """Continual learning local update."""
        # Set global parameters
        self.algorithm.set_parameters(global_parameters)
        
        # Check if we need to switch tasks
        self._check_task_progression()
        
        initial_metrics = self.algorithm.get_continual_learning_metrics()
        
        # Local training with current task
        for epoch in range(epochs):
            batch_data = self._sample_batch(batch_size)
            batch_tensor = self._convert_to_tensor_batch(batch_data)
            
            metrics = self.algorithm.update(batch_tensor)
            
            # Track adaptation
            if hasattr(self.algorithm, 'task_performance'):
                current_task_id = self.algorithm.current_task.task_id if self.algorithm.current_task else "unknown"
                if current_task_id in self.algorithm.task_performance:
                    recent_performance = self.algorithm.task_performance[current_task_id][-5:]
                    if len(recent_performance) >= 2:
                        adaptation_score = np.mean(recent_performance[-2:]) - np.mean(recent_performance[:2])
                        self.adaptation_scores.append(adaptation_score)
        
        final_metrics = self.algorithm.get_continual_learning_metrics()
        updated_params = self.algorithm.get_parameters()
        
        # Calculate continual learning improvements
        forgetting_reduction = (
            initial_metrics.get("average_forgetting", 0.0) - 
            final_metrics.get("average_forgetting", 0.0)
        )
        
        return ClientUpdate(
            client_id=self.client_id,
            parameters=updated_params,
            num_samples=len(self.local_data),
            loss=final_metrics.get("current_task_performance", {}).get("mean", 0.0),
            metrics={
                "forgetting_reduction": forgetting_reduction,
                "current_task": final_metrics.get("current_task"),
                "num_completed_tasks": final_metrics.get("num_completed_tasks", 0),
                "memory_utilization": final_metrics.get("memory_utilization", 0.0),
                "adaptation_score": np.mean(self.adaptation_scores[-10:]) if self.adaptation_scores else 0.0,
                "continual_method": self.algorithm.continual_method
            }
        )
    
    def _check_task_progression(self):
        """Check if it's time to progress to the next task."""
        if self.current_task_index < len(self.task_sequence):
            current_task = self.task_sequence[self.current_task_index]
            
            # Simple progression logic: switch after a certain number of updates
            if self.algorithm.training_step > 0 and self.algorithm.training_step % 1000 == 0:
                if self.current_task_index < len(self.task_sequence) - 1:
                    self.current_task_index += 1
                    next_task = self.task_sequence[self.current_task_index]
                    self.algorithm.start_new_task(next_task)
    
    def _convert_to_tensor_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Convert batch data to tensor format."""
        observations = torch.FloatTensor([d["state"] for d in batch_data])
        actions = torch.FloatTensor([d["action"] for d in batch_data])
        rewards = torch.FloatTensor([d["reward"] for d in batch_data])
        next_observations = torch.FloatTensor([d["next_state"] for d in batch_data])
        terminals = torch.FloatTensor([d.get("done", False) for d in batch_data])
        
        return {
            "observations": observations.to(self.algorithm.device),
            "actions": actions.to(self.algorithm.device),
            "rewards": rewards.to(self.algorithm.device),
            "next_observations": next_observations.to(self.algorithm.device),
            "terminals": terminals.to(self.algorithm.device)
        }