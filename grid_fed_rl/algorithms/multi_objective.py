"""Multi-Objective Federated Reinforcement Learning for Power Systems.

This module implements novel multi-objective federated RL algorithms that simultaneously
optimize economic efficiency, grid stability, environmental impact, and system resilience.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cdist

from .base import OfflineRLAlgorithm, ActorCriticBase, TrainingMetrics
from ..federated.core import FederatedClient, ClientUpdate
from ..utils.validation import validate_constraints


@dataclass
class Objective:
    """Definition of an optimization objective."""
    name: str
    objective_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    weight: float = 1.0
    maximize: bool = False  # True for maximization, False for minimization
    normalization_factor: float = 1.0
    target_value: Optional[float] = None


@dataclass 
class ParetoSolution:
    """A solution on the Pareto front."""
    parameters: Dict[str, np.ndarray]
    objective_values: Dict[str, float]
    dominates: int = 0  # Number of solutions this dominates
    dominated_by: int = 0  # Number of solutions that dominate this
    crowding_distance: float = 0.0


class MultiObjectiveCritic(nn.Module):
    """Multi-objective critic network with separate heads for each objective."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        objectives: List[Objective],
        hidden_dims: List[int] = [256, 256, 128],
        shared_layers: int = 2
    ):
        super().__init__()
        self.objectives = objectives
        self.num_objectives = len(objectives)
        
        # Shared feature extractor
        shared_dims = hidden_dims[:shared_layers]
        self.shared_layers = nn.ModuleList()
        prev_dim = state_dim + action_dim
        
        for dim in shared_dims:
            self.shared_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        # Objective-specific heads
        self.objective_heads = nn.ModuleDict()
        remaining_dims = hidden_dims[shared_layers:]
        
        for obj in objectives:
            layers = []
            current_dim = prev_dim
            
            for dim in remaining_dims:
                layers.extend([
                    nn.Linear(current_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(0.05)
                ])
                current_dim = dim
                
            layers.append(nn.Linear(current_dim, 1))
            self.objective_heads[obj.name] = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning Q-values for each objective."""
        x = torch.cat([state, action], dim=-1)
        
        # Shared feature extraction
        for layer in self.shared_layers:
            x = layer(x)
        
        # Objective-specific Q-values
        q_values = {}
        for obj in self.objectives:
            q_values[obj.name] = self.objective_heads[obj.name](x)
            
        return q_values


class ParetoActor(nn.Module):
    """Actor network with Pareto-optimal policy learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        objectives: List[Objective],
        hidden_dims: List[int] = [256, 256, 128],
        preference_dim: int = 32
    ):
        super().__init__()
        self.objectives = objectives
        self.num_objectives = len(objectives)
        self.preference_dim = preference_dim
        
        # Preference encoder
        self.preference_encoder = nn.Sequential(
            nn.Linear(self.num_objectives, preference_dim),
            nn.ReLU(),
            nn.Linear(preference_dim, preference_dim),
            nn.ReLU()
        )
        
        # Policy network with preference conditioning
        self.policy_net = self._build_policy_network(
            state_dim + preference_dim, action_dim * 2, hidden_dims
        )
        
        # Objective prediction network (for preference learning)
        self.objective_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, self.num_objectives)
        )
        
    def _build_policy_network(self, input_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build policy network."""
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(
        self,
        state: torch.Tensor,
        preferences: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with preference-conditioned policy."""
        batch_size = state.shape[0]
        
        # Encode preferences
        pref_encoded = self.preference_encoder(preferences)
        
        # Condition policy on preferences
        conditioned_state = torch.cat([state, pref_encoded], dim=-1)
        policy_output = self.policy_net(conditioned_state)
        
        mean, log_std = torch.chunk(policy_output, 2, dim=-1)
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
        
        # Predict objective values
        state_action = torch.cat([state, action], dim=-1)
        predicted_objectives = self.objective_predictor(state_action)
        
        return action, log_prob, predicted_objectives


class NSGA2Selector:
    """Non-Dominated Sorting Genetic Algorithm II for Pareto front selection."""
    
    def __init__(self, objectives: List[Objective]):
        self.objectives = objectives
        self.num_objectives = len(objectives)
        
    def non_dominated_sort(self, solutions: List[ParetoSolution]) -> List[List[ParetoSolution]]:
        """Sort solutions into Pareto fronts."""
        fronts = [[]]
        
        for i, sol_i in enumerate(solutions):
            sol_i.dominates = 0
            sol_i.dominated_by = 0
            
            for j, sol_j in enumerate(solutions):
                if i != j:
                    if self._dominates(sol_i, sol_j):
                        sol_i.dominates += 1
                    elif self._dominates(sol_j, sol_i):
                        sol_i.dominated_by += 1
            
            if sol_i.dominated_by == 0:
                fronts[0].append(sol_i)
        
        # Create subsequent fronts
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            
            for sol_i in fronts[front_idx]:
                for sol_j in solutions:
                    if sol_i != sol_j and self._dominates(sol_i, sol_j):
                        sol_j.dominated_by -= 1
                        if sol_j.dominated_by == 0:
                            next_front.append(sol_j)
            
            if next_front:
                fronts.append(next_front)
            front_idx += 1
        
        return fronts[:-1]  # Remove empty last front
    
    def _dominates(self, sol_a: ParetoSolution, sol_b: ParetoSolution) -> bool:
        """Check if solution A dominates solution B."""
        better_in_at_least_one = False
        
        for obj in self.objectives:
            val_a = sol_a.objective_values[obj.name]
            val_b = sol_b.objective_values[obj.name]
            
            if obj.maximize:
                if val_a < val_b:
                    return False
                elif val_a > val_b:
                    better_in_at_least_one = True
            else:
                if val_a > val_b:
                    return False
                elif val_a < val_b:
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def calculate_crowding_distance(self, front: List[ParetoSolution]) -> None:
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for sol in front:
            sol.crowding_distance = 0.0
        
        for obj in self.objectives:
            # Sort by objective value
            front.sort(key=lambda x: x.objective_values[obj.name])
            
            # Boundary solutions have infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate crowding distance for intermediate solutions
            obj_min = front[0].objective_values[obj.name]
            obj_max = front[-1].objective_values[obj.name]
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (front[i + 1].objective_values[obj.name] - 
                              front[i - 1].objective_values[obj.name]) / obj_range
                    front[i].crowding_distance += distance
    
    def select_solutions(
        self,
        solutions: List[ParetoSolution],
        num_select: int
    ) -> List[ParetoSolution]:
        """Select best solutions using NSGA-II selection."""
        # Non-dominated sorting
        fronts = self.non_dominated_sort(solutions)
        
        # Calculate crowding distance for each front
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Select solutions
        selected = []
        front_idx = 0
        
        while len(selected) < num_select and front_idx < len(fronts):
            front = fronts[front_idx]
            
            if len(selected) + len(front) <= num_select:
                # Add entire front
                selected.extend(front)
            else:
                # Sort by crowding distance and add best
                remaining = num_select - len(selected)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                selected.extend(front[:remaining])
            
            front_idx += 1
        
        return selected


class MOFRL(OfflineRLAlgorithm, ActorCriticBase):
    """Multi-Objective Federated Reinforcement Learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        objectives: List[Objective],
        hidden_dims: List[int] = [256, 256, 128],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        population_size: int = 50,
        preference_learning: bool = True,
        device: str = "auto",
        **kwargs
    ):
        super().__init__(state_dim, action_dim, device=device, **kwargs)
        ActorCriticBase.__init__(self, state_dim, action_dim, hidden_dims, device=device, **kwargs)
        
        self.objectives = objectives
        self.population_size = population_size
        self.preference_learning = preference_learning
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)
        
        # Pareto front management
        self.pareto_selector = NSGA2Selector(objectives)
        self.pareto_solutions: List[ParetoSolution] = []
        self.current_preferences = torch.ones(len(objectives)) / len(objectives)  # Equal preferences initially
        
        # Training history
        self.objective_history = {obj.name: [] for obj in objectives}
        self.pareto_front_size_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """Build multi-objective networks."""
        # Multi-objective critic
        self.critic = MultiObjectiveCritic(
            self.state_dim,
            self.action_dim, 
            self.objectives,
            self.hidden_dims
        ).to(self.device)
        
        self.critic_target = MultiObjectiveCritic(
            self.state_dim,
            self.action_dim,
            self.objectives, 
            self.hidden_dims
        ).to(self.device)
        
        # Copy parameters to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Preference-conditioned actor
        self.actor = ParetoActor(
            self.state_dim,
            self.action_dim,
            self.objectives,
            self.hidden_dims
        ).to(self.device)
    
    def select_action(
        self,
        state: np.ndarray,
        preferences: Optional[np.ndarray] = None,
        eval_mode: bool = False
    ) -> np.ndarray:
        """Select action with preference conditioning."""
        state_tensor = self.to_tensor(state).unsqueeze(0)
        
        if preferences is None:
            preferences = self.current_preferences.clone()
        else:
            preferences = torch.FloatTensor(preferences).to(self.device)
        
        preferences = preferences.unsqueeze(0)
        
        with torch.no_grad():
            action, _, _ = self.actor(state_tensor, preferences, deterministic=eval_mode)
        
        return self.to_numpy(action.squeeze(0))
    
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Multi-objective update step."""
        self.training_step += 1
        
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]  # Should be multi-objective rewards
        next_states = batch["next_observations"]
        terminals = batch["terminals"]
        
        # Convert single rewards to multi-objective if needed
        if len(rewards.shape) == 1:
            # Use default objective decomposition
            mo_rewards = self._decompose_rewards(states, actions, rewards)
        else:
            mo_rewards = rewards
        
        # Update critics for each objective
        critic_losses = self._update_critics(states, actions, mo_rewards, next_states, terminals)
        
        # Update preference-conditioned actor
        actor_loss = self._update_actor(states)
        
        # Update Pareto front
        self._update_pareto_front(states, actions, mo_rewards)
        
        # Update preferences if preference learning is enabled
        if self.preference_learning:
            self._update_preferences(states, actions, mo_rewards)
        
        # Update target networks
        self._soft_update_targets()
        
        # Track objective values
        for i, obj in enumerate(self.objectives):
            obj_values = mo_rewards[:, i] if len(mo_rewards.shape) > 1 else mo_rewards
            self.objective_history[obj.name].append(float(torch.mean(obj_values)))
        
        total_loss = sum(critic_losses.values()) + actor_loss
        
        return TrainingMetrics(
            loss=total_loss,
            q_loss=sum(critic_losses.values()),
            policy_loss=actor_loss,
            mean_q_value=torch.mean(torch.stack(list(critic_losses.values()))).item()
        )
    
    def _decompose_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """Decompose scalar rewards into multi-objective rewards."""
        batch_size = states.shape[0]
        mo_rewards = torch.zeros(batch_size, len(self.objectives), device=self.device)
        
        for i, obj in enumerate(self.objectives):
            # Apply objective function
            obj_reward = obj.objective_fn(states, actions, rewards)
            
            # Normalize
            obj_reward = obj_reward / obj.normalization_factor
            
            # Handle maximization vs minimization
            if not obj.maximize:
                obj_reward = -obj_reward
                
            mo_rewards[:, i] = obj_reward.squeeze()
        
        return mo_rewards
    
    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor, 
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminals: torch.Tensor
    ) -> Dict[str, float]:
        """Update multi-objective critics."""
        batch_size = states.shape[0]
        losses = {}
        
        with torch.no_grad():
            # Sample preferences for next actions
            pref_batch = torch.rand(batch_size, len(self.objectives), device=self.device)
            pref_batch = F.softmax(pref_batch, dim=-1)  # Normalize preferences
            
            # Get next actions
            next_actions, _, _ = self.actor(next_states, pref_batch)
            
            # Get target Q-values for each objective
            target_q_values = self.critic_target(next_states, next_actions)
        
        # Current Q-values
        current_q_values = self.critic(states, actions)
        
        # Update each objective's critic
        for i, obj in enumerate(self.objectives):
            obj_rewards = rewards[:, i] if len(rewards.shape) > 1 else rewards
            
            # Target values
            target_q = target_q_values[obj.name]
            target_values = obj_rewards.unsqueeze(-1) + self.gamma * (1 - terminals.unsqueeze(-1)) * target_q
            
            # Current values
            current_q = current_q_values[obj.name]
            
            # Critic loss
            critic_loss = F.mse_loss(current_q, target_values.detach())
            losses[obj.name] = critic_loss.item()
            
        # Combined critic loss
        total_critic_loss = sum(torch.tensor(loss) for loss in losses.values())
        
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return losses
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """Update preference-conditioned actor."""
        batch_size = states.shape[0]
        
        # Sample diverse preferences
        preferences = torch.rand(batch_size, len(self.objectives), device=self.device)
        preferences = F.softmax(preferences, dim=-1)
        
        # Get actions and objective predictions
        actions, log_probs, predicted_objectives = self.actor(states, preferences)
        
        # Get Q-values for each objective
        q_values = self.critic(states, actions)
        
        # Scalarize using preferences
        scalarized_q = torch.zeros(batch_size, 1, device=self.device)
        for i, obj in enumerate(self.objectives):
            weight = preferences[:, i].unsqueeze(-1)
            scalarized_q += weight * q_values[obj.name]
        
        # Actor loss: maximize scalarized Q-value
        actor_loss = -scalarized_q.mean()
        
        # Add objective prediction loss if available
        if predicted_objectives is not None and len(self.objective_history[self.objectives[0].name]) > 0:
            # Use recent objective values as targets
            target_objectives = torch.zeros_like(predicted_objectives)
            for i, obj in enumerate(self.objectives):
                recent_values = self.objective_history[obj.name][-100:]  # Last 100 values
                target_objectives[:, i] = np.mean(recent_values) if recent_values else 0.0
                
            prediction_loss = F.mse_loss(predicted_objectives, target_objectives)
            actor_loss += 0.1 * prediction_loss
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_pareto_front(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ):
        """Update Pareto front with new solutions."""
        # Get current model parameters
        current_params = {
            name: param.detach().cpu().numpy() 
            for name, param in self.actor.named_parameters()
        }
        
        # Calculate objective values
        objective_values = {}
        for i, obj in enumerate(self.objectives):
            if len(rewards.shape) > 1:
                obj_reward = torch.mean(rewards[:, i]).item()
            else:
                obj_reward = torch.mean(rewards).item()
            objective_values[obj.name] = obj_reward
        
        # Create new solution
        new_solution = ParetoSolution(
            parameters=current_params,
            objective_values=objective_values
        )
        
        # Add to solutions list
        self.pareto_solutions.append(new_solution)
        
        # Keep only best solutions
        if len(self.pareto_solutions) > self.population_size:
            self.pareto_solutions = self.pareto_selector.select_solutions(
                self.pareto_solutions, 
                self.population_size
            )
        
        self.pareto_front_size_history.append(len(self.pareto_solutions))
    
    def _update_preferences(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ):
        """Update preference vector based on performance."""
        # Simple preference adaptation: favor objectives with lower recent performance
        if len(self.pareto_solutions) > 1:
            # Calculate normalized objective ranges on current Pareto front
            obj_ranges = {}
            for obj in self.objectives:
                values = [sol.objective_values[obj.name] for sol in self.pareto_solutions]
                obj_ranges[obj.name] = max(values) - min(values) if len(values) > 1 else 1.0
            
            # Update preferences inversely proportional to achievement
            total_range = sum(obj_ranges.values())
            for i, obj in enumerate(self.objectives):
                if total_range > 0:
                    # Higher weight for objectives with larger improvement potential
                    self.current_preferences[i] = obj_ranges[obj.name] / total_range
        
        # Add small random noise to encourage exploration
        noise = torch.randn_like(self.current_preferences) * 0.01
        self.current_preferences = F.softmax(self.current_preferences + noise, dim=0)
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get current Pareto front solutions."""
        if not self.pareto_solutions:
            return []
        
        # Sort into fronts
        fronts = self.pareto_selector.non_dominated_sort(self.pareto_solutions)
        
        # Return first front (Pareto optimal solutions)
        pareto_front = []
        if fronts:
            for sol in fronts[0]:
                pareto_front.append({
                    "objective_values": sol.objective_values.copy(),
                    "crowding_distance": sol.crowding_distance,
                    "dominates": sol.dominates,
                    "dominated_by": sol.dominated_by
                })
        
        return pareto_front
    
    def get_multi_objective_metrics(self) -> Dict[str, Any]:
        """Get comprehensive multi-objective training metrics."""
        pareto_front = self.get_pareto_front()
        
        metrics = {
            "pareto_front_size": len(pareto_front),
            "total_solutions": len(self.pareto_solutions),
            "current_preferences": self.current_preferences.cpu().numpy().tolist(),
            "objective_history": {
                name: history[-100:] for name, history in self.objective_history.items()
            },
            "pareto_front_evolution": self.pareto_front_size_history[-100:],
        }
        
        if pareto_front:
            # Calculate hypervolume approximation
            obj_values = np.array([[sol["objective_values"][obj.name] for obj in self.objectives] 
                                  for sol in pareto_front])
            
            # Simple hypervolume calculation (reference point at origin)
            reference_point = np.zeros(len(self.objectives))
            hypervolume = self._calculate_hypervolume(obj_values, reference_point)
            metrics["hypervolume"] = hypervolume
            
            # Coverage metrics
            for i, obj in enumerate(self.objectives):
                values = obj_values[:, i]
                metrics[f"{obj.name}_range"] = float(np.max(values) - np.min(values))
                metrics[f"{obj.name}_mean"] = float(np.mean(values))
                metrics[f"{obj.name}_std"] = float(np.std(values))
        
        return metrics
    
    def _calculate_hypervolume(self, points: np.ndarray, reference: np.ndarray) -> float:
        """Calculate hypervolume (simplified 2D/3D implementation)."""
        if len(points) == 0:
            return 0.0
        
        if points.shape[1] == 2:
            # 2D hypervolume
            sorted_points = points[np.argsort(points[:, 0])]
            volume = 0.0
            for i, point in enumerate(sorted_points):
                if i == 0:
                    width = point[0] - reference[0]
                else:
                    width = point[0] - sorted_points[i-1][0]
                height = point[1] - reference[1]
                volume += width * height
            return max(0.0, volume)
        else:
            # For higher dimensions, use simple approximation
            dominated_volume = 1.0
            for dim in range(points.shape[1]):
                dim_range = np.max(points[:, dim]) - reference[dim]
                dominated_volume *= max(0.0, dim_range)
            return dominated_volume / len(points)  # Normalize by number of points
    
    def train_offline(
        self,
        dataset,
        num_epochs: int,
        batch_size: int = 256,
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train multi-objective federated RL."""
        metrics_history = []
        
        self.logger.info(f"Starting MOFRL training: {num_epochs} epochs, {len(self.objectives)} objectives")
        
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
                mo_metrics = self.get_multi_objective_metrics()
                pareto_size = mo_metrics.get("pareto_front_size", 0)
                hypervolume = mo_metrics.get("hypervolume", 0.0)
                
                self.logger.info(
                    f"Epoch {epoch}: Loss={avg_metrics.loss:.4f}, "
                    f"ParetoSize={pareto_size}, "
                    f"Hypervolume={hypervolume:.4f}"
                )
        
        return metrics_history
    
    def save(self, path: str) -> None:
        """Save multi-objective model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "pareto_solutions": self.pareto_solutions,
            "current_preferences": self.current_preferences,
            "objective_history": self.objective_history,
            "training_step": self.training_step
        }, path)
    
    def load(self, path: str) -> None:
        """Load multi-objective model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.pareto_solutions = checkpoint.get("pareto_solutions", [])
        self.current_preferences = checkpoint.get("current_preferences", 
                                                  torch.ones(len(self.objectives)) / len(self.objectives))
        self.objective_history = checkpoint.get("objective_history", 
                                               {obj.name: [] for obj in self.objectives})
        self.training_step = checkpoint.get("training_step", 0)


# Utility functions for creating power system objectives

def economic_efficiency_objective(cost_weight: float = 1.0) -> Objective:
    """Create economic efficiency objective (minimize operational costs)."""
    def economic_cost_fn(states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        # Assume actions represent generation levels
        generation_cost = torch.sum(actions ** 2, dim=-1)  # Quadratic cost function
        return -generation_cost  # Negative because we minimize cost
    
    return Objective(
        name="economic_efficiency",
        objective_fn=economic_cost_fn,
        weight=cost_weight,
        maximize=False,  # Minimize cost
        normalization_factor=1000.0
    )


def grid_stability_objective(stability_weight: float = 1.0) -> Objective:
    """Create grid stability objective (maximize voltage and frequency stability)."""
    def stability_fn(states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        # Assume state includes voltage deviations
        if states.shape[-1] >= 10:
            voltage_deviations = torch.abs(states[..., :10] - 1.0)  # Deviation from 1.0 pu
            stability_measure = -torch.sum(voltage_deviations, dim=-1)  # Better when deviations are small
        else:
            stability_measure = rewards  # Fallback to reward signal
        
        return stability_measure
    
    return Objective(
        name="grid_stability",
        objective_fn=stability_fn,
        weight=stability_weight,
        maximize=True,  # Maximize stability
        normalization_factor=10.0
    )


def environmental_impact_objective(emission_weight: float = 1.0) -> Objective:
    """Create environmental impact objective (minimize emissions, maximize renewables)."""
    def environmental_fn(states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        # Assume actions include renewable/conventional generation split
        if actions.shape[-1] >= 10:
            renewable_gen = torch.sum(actions[..., :5], dim=-1)  # First 5 as renewables
            conventional_gen = torch.sum(actions[..., 5:10], dim=-1)  # Next 5 as conventional
            
            # Reward renewable generation, penalize conventional
            environmental_score = renewable_gen - 2.0 * conventional_gen
        else:
            environmental_score = rewards  # Fallback
        
        return environmental_score
    
    return Objective(
        name="environmental_impact",
        objective_fn=environmental_fn,
        weight=emission_weight,
        maximize=True,  # Maximize environmental benefit
        normalization_factor=100.0
    )


def system_resilience_objective(resilience_weight: float = 1.0) -> Objective:
    """Create system resilience objective (maximize ability to handle contingencies)."""
    def resilience_fn(states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        # Measure based on reserve margins and load balancing
        if states.shape[-1] >= 20 and actions.shape[-1] >= 10:
            total_generation = torch.sum(actions[..., :10], dim=-1)
            total_load = torch.sum(states[..., 10:20], dim=-1)
            
            # Reserve margin (generation capacity above load)
            reserve_margin = total_generation - total_load
            
            # Penalize both excess and insufficient reserves
            optimal_reserve = 0.2 * total_load  # 20% reserve
            resilience_score = -torch.abs(reserve_margin - optimal_reserve)
        else:
            resilience_score = rewards  # Fallback
        
        return resilience_score
    
    return Objective(
        name="system_resilience",
        objective_fn=resilience_fn,
        weight=resilience_weight,
        maximize=True,  # Maximize resilience
        normalization_factor=50.0
    )