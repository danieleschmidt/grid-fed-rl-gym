"""Physics-Informed Federated Reinforcement Learning for Power Systems.

This module implements novel physics-informed federated RL algorithms that embed
power flow equations and grid stability constraints directly into the learning process.
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

from .base import OfflineRLAlgorithm, ActorCriticBase, TrainingMetrics
from ..federated.core import FederatedClient, ClientUpdate
from ..utils.validation import validate_constraints
from ..utils.exceptions import PhysicsViolationError, ConvergenceError


@dataclass
class PhysicsConstraint:
    """Physics-based constraint for power systems."""
    name: str
    constraint_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    weight: float = 1.0
    hard_constraint: bool = True
    tolerance: float = 1e-3


class PowerFlowNetwork(nn.Module):
    """Neural network that embeds power flow equations."""
    
    def __init__(
        self,
        num_buses: int,
        num_lines: int,
        hidden_dims: List[int] = [128, 64, 32]
    ):
        super().__init__()
        self.num_buses = num_buses
        self.num_lines = num_lines
        
        # Admittance matrix embedding
        self.admittance_encoder = nn.Linear(num_buses * num_buses, hidden_dims[0])
        
        # Power flow approximator  
        self.pf_layers = nn.ModuleList()
        prev_dim = hidden_dims[0] + num_buses * 2  # + P,Q injections
        
        for hidden_dim in hidden_dims:
            self.pf_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Output: voltage magnitudes and angles
        self.voltage_head = nn.Linear(prev_dim, num_buses)  # |V|
        self.angle_head = nn.Linear(prev_dim, num_buses)    # θ
        
        # Physics-informed regularization
        self.physics_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, admittance_matrix: torch.Tensor, power_injections: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through power flow network.
        
        Args:
            admittance_matrix: Bus admittance matrix [batch, num_buses, num_buses]
            power_injections: P,Q injections [batch, num_buses, 2]
            
        Returns:
            voltage_mags: Voltage magnitudes [batch, num_buses]
            voltage_angles: Voltage angles [batch, num_buses]
        """
        batch_size = admittance_matrix.shape[0]
        
        # Encode admittance matrix
        Y_flat = admittance_matrix.view(batch_size, -1)
        Y_encoded = torch.relu(self.admittance_encoder(Y_flat))
        
        # Flatten power injections
        PQ_flat = power_injections.view(batch_size, -1)
        
        # Concatenate features
        x = torch.cat([Y_encoded, PQ_flat], dim=1)
        
        # Pass through power flow approximator
        for layer in self.pf_layers:
            x = layer(x)
            
        # Predict voltages
        voltage_mags = torch.sigmoid(self.voltage_head(x)) + 0.5  # [0.5, 1.5] pu
        voltage_angles = torch.tanh(self.angle_head(x)) * np.pi    # [-π, π] radians
        
        return voltage_mags, voltage_angles
    
    def compute_physics_loss(
        self,
        voltage_mags: torch.Tensor,
        voltage_angles: torch.Tensor,
        admittance_matrix: torch.Tensor,
        target_injections: torch.Tensor
    ) -> torch.Tensor:
        """Compute physics-informed loss based on power flow equations."""
        batch_size = voltage_mags.shape[0]
        
        # Convert to complex voltages
        V_complex = voltage_mags * torch.exp(1j * voltage_angles)
        
        # Compute power injections from voltages: S = V * conj(Y * V)
        Y_complex = admittance_matrix.real + 1j * admittance_matrix.imag
        I = torch.matmul(Y_complex, V_complex.unsqueeze(-1)).squeeze(-1)
        S_computed = V_complex * torch.conj(I)
        
        # Split into P and Q
        P_computed = S_computed.real
        Q_computed = S_computed.imag
        computed_injections = torch.stack([P_computed, Q_computed], dim=-1)
        
        # Physics loss: power flow equation residual
        power_residual = F.mse_loss(computed_injections, target_injections)
        
        return power_residual


class PhysicsInformedActor(nn.Module):
    """Actor network with embedded physics constraints."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_buses: int,
        constraints: List[PhysicsConstraint],
        hidden_dims: List[int] = [256, 256, 128]
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_buses = num_buses
        self.constraints = constraints
        
        # Standard policy network
        self.policy_net = self._build_policy_network(state_dim, action_dim, hidden_dims)
        
        # Physics constraint checker
        self.constraint_checker = self._build_constraint_network(state_dim + action_dim, len(constraints))
        
        # Constraint violation predictor
        self.violation_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def _build_policy_network(self, input_dim: int, output_dim: int, hidden_dims: List[int]) -> nn.Module:
        """Build policy network with normalization layers."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Output layer for mean and log_std
        layers.append(nn.Linear(prev_dim, output_dim * 2))
        
        return nn.Sequential(*layers)
    
    def _build_constraint_network(self, input_dim: int, num_constraints: int) -> nn.Module:
        """Build constraint checking network."""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_constraints)
        )
        
    def forward(self, state: torch.Tensor, return_constraints: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through physics-informed actor."""
        batch_size = state.shape[0]
        
        # Policy network output
        policy_output = self.policy_net(state)
        mean, log_std = torch.chunk(policy_output, 2, dim=-1)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        # Sample action
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        constraint_values = None
        if return_constraints:
            # Check physics constraints
            state_action = torch.cat([state, action], dim=-1)
            constraint_values = self.constraint_checker(state_action)
            
            # Apply physics-based action correction
            violation_risk = self.violation_predictor(state_action)
            if torch.any(violation_risk > 0.5):
                action = self._correct_physics_violations(state, action, constraint_values)
                
        return action, log_prob, constraint_values
    
    def _correct_physics_violations(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        constraint_values: torch.Tensor
    ) -> torch.Tensor:
        """Correct actions that violate physics constraints."""
        corrected_action = action.clone()
        
        # Simple gradient-based correction
        with torch.enable_grad():
            action_var = action.clone().requires_grad_(True)
            state_action = torch.cat([state, action_var], dim=-1)
            constraints = self.constraint_checker(state_action)
            
            # Minimize constraint violations
            violation_loss = torch.sum(torch.relu(-constraints), dim=-1)  # Penalize negative values
            
            if violation_loss.sum() > 0:
                grad = torch.autograd.grad(violation_loss.mean(), action_var)[0]
                corrected_action = action - 0.1 * grad.detach()  # Small correction step
                corrected_action = torch.tanh(corrected_action)  # Keep in valid range
                
        return corrected_action


class PIFRL(OfflineRLAlgorithm, ActorCriticBase):
    """Physics-Informed Federated Reinforcement Learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_buses: int,
        physics_constraints: List[PhysicsConstraint],
        hidden_dims: List[int] = [256, 256, 128],
        lr: float = 3e-4,
        gamma: float = 0.99,
        physics_weight: float = 10.0,
        constraint_tolerance: float = 1e-3,
        device: str = "auto",
        **kwargs
    ):
        super().__init__(state_dim, action_dim, device=device, **kwargs)
        ActorCriticBase.__init__(self, state_dim, action_dim, hidden_dims, device=device, **kwargs)
        
        self.num_buses = num_buses
        self.physics_constraints = physics_constraints
        self.physics_weight = physics_weight
        self.constraint_tolerance = constraint_tolerance
        self.lr = lr
        self.gamma = gamma
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)
        self.physics_optimizer = optim.Adam(self.power_flow_net.parameters(), lr=lr/2)
        
        # Physics loss tracking
        self.physics_loss_history = []
        self.constraint_violations = []
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """Build physics-informed networks."""
        # Physics-informed actor
        self.actor = PhysicsInformedActor(
            self.state_dim, 
            self.action_dim,
            self.num_buses,
            self.physics_constraints,
            self.hidden_dims
        ).to(self.device)
        
        # Standard twin critic
        self.critic = self._build_twin_critic().to(self.device)
        
        # Power flow network for physics embedding
        self.power_flow_net = PowerFlowNetwork(self.num_buses, self.num_buses * 2).to(self.device)
        
    def _build_twin_critic(self) -> nn.Module:
        """Build twin Q-networks with physics awareness."""
        class PhysicsInformedCritic(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims, physics_dim=32):
                super().__init__()
                
                # Physics feature extractor
                self.physics_extractor = nn.Sequential(
                    nn.Linear(state_dim + action_dim, physics_dim),
                    nn.ReLU(),
                    nn.Linear(physics_dim, physics_dim)
                )
                
                # Twin Q-networks with physics features
                input_dim = state_dim + action_dim + physics_dim
                
                self.q1 = self._build_q_network(input_dim, hidden_dims)
                self.q2 = self._build_q_network(input_dim, hidden_dims)
                
            def _build_q_network(self, input_dim, hidden_dims):
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    prev_dim = hidden_dim
                    
                layers.append(nn.Linear(prev_dim, 1))
                return nn.Sequential(*layers)
                
            def forward(self, state, action):
                x = torch.cat([state, action], dim=-1)
                physics_features = self.physics_extractor(x)
                augmented_input = torch.cat([x, physics_features], dim=-1)
                
                return self.q1(augmented_input), self.q2(augmented_input)
                
        return PhysicsInformedCritic(self.state_dim, self.action_dim, self.hidden_dims)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select physics-informed action."""
        state_tensor = self.to_tensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if eval_mode:
                # Deterministic action with constraint checking
                action, _, constraints = self.actor(state_tensor, return_constraints=True)
                
                # Log constraint status
                if constraints is not None:
                    violations = torch.sum(constraints < -self.constraint_tolerance)
                    if violations > 0:
                        self.logger.debug(f"Action has {violations} constraint violations")
            else:
                action, _, _ = self.actor(state_tensor, return_constraints=False)
        
        return self.to_numpy(action.squeeze(0))
    
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Physics-informed update step."""
        self.training_step += 1
        
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        terminals = batch["terminals"]
        
        # Extract physics information from states
        admittance_matrices, power_injections = self._extract_physics_data(states)
        
        # Update power flow network
        physics_loss = self._update_power_flow_network(
            admittance_matrices, power_injections, states
        )
        
        # Update critic with physics-informed features
        critic_loss = self._update_critic(states, actions, rewards, next_states, terminals)
        
        # Update physics-informed actor
        actor_loss = self._update_actor(states)
        
        # Track physics violations
        with torch.no_grad():
            _, _, constraints = self.actor(states, return_constraints=True)
            if constraints is not None:
                violations = torch.sum(constraints < -self.constraint_tolerance, dim=-1)
                self.constraint_violations.extend(violations.cpu().numpy())
        
        self.physics_loss_history.append(physics_loss)
        
        total_loss = critic_loss + actor_loss + self.physics_weight * physics_loss
        
        return TrainingMetrics(
            loss=total_loss,
            q_loss=critic_loss,
            policy_loss=actor_loss,
            alpha_loss=physics_loss
        )
    
    def _extract_physics_data(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract admittance matrices and power injections from states."""
        batch_size = states.shape[0]
        
        # For this implementation, assume specific state structure
        # In practice, this would be based on actual grid feeder structure
        
        # Create simplified admittance matrices (symmetric for this example)
        admittance_matrices = torch.zeros(batch_size, self.num_buses, self.num_buses, device=self.device)
        
        for i in range(batch_size):
            # Simplified: create admittance matrix from state features
            # In practice, this would use actual grid topology
            diag_values = torch.abs(states[i, :self.num_buses]) + 1e-3
            admittance_matrices[i] = torch.diag(diag_values)
            
            # Add off-diagonal coupling (simplified)
            for j in range(min(self.num_buses-1, 5)):  # Connect first few buses
                coupling = 0.1 * torch.abs(states[i, j])
                admittance_matrices[i, j, j+1] = -coupling
                admittance_matrices[i, j+1, j] = -coupling
                admittance_matrices[i, j, j] += coupling
                admittance_matrices[i, j+1, j+1] += coupling
        
        # Extract power injections (P, Q for each bus)
        if states.shape[1] >= self.num_buses * 2:
            power_injections = states[:, self.num_buses:self.num_buses*3].view(batch_size, self.num_buses, 2)
        else:
            # Default power injections
            power_injections = torch.zeros(batch_size, self.num_buses, 2, device=self.device)
            
        return admittance_matrices, power_injections
    
    def _update_power_flow_network(
        self,
        admittance_matrices: torch.Tensor,
        power_injections: torch.Tensor,
        states: torch.Tensor
    ) -> float:
        """Update power flow approximator network."""
        # Get voltage predictions
        voltage_mags, voltage_angles = self.power_flow_net(admittance_matrices, power_injections)
        
        # Extract target voltages from states (if available)
        batch_size = states.shape[0]
        if states.shape[1] >= self.num_buses * 4:  # Has voltage targets
            target_mags = states[:, self.num_buses*3:self.num_buses*4]
            target_angles = states[:, self.num_buses*4:self.num_buses*5] if states.shape[1] >= self.num_buses*5 else torch.zeros_like(target_mags)
        else:
            # Use power flow solution as target (simplified)
            target_mags = torch.ones_like(voltage_mags)  # 1.0 pu nominal
            target_angles = torch.zeros_like(voltage_angles)
            
        # Physics-informed loss
        physics_loss = self.power_flow_net.compute_physics_loss(
            voltage_mags, voltage_angles, admittance_matrices.real, power_injections
        )
        
        # Supervised loss (if targets available)
        supervised_loss = F.mse_loss(voltage_mags, target_mags) + F.mse_loss(voltage_angles, target_angles)
        
        total_loss = physics_loss + 0.1 * supervised_loss
        
        self.physics_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.power_flow_net.parameters(), 1.0)
        self.physics_optimizer.step()
        
        return total_loss.item()
    
    def _update_critic(self, states, actions, rewards, next_states, terminals):
        """Update physics-informed critic networks."""
        with torch.no_grad():
            # Get next actions with physics constraints
            next_actions, next_log_probs, _ = self.actor(next_states, return_constraints=False)
            
            # Target Q-values
            target_q1, target_q2 = self.critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - terminals) * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, states):
        """Update physics-informed actor."""
        # Get actions with constraint checking
        actions, log_probs, constraint_values = self.actor(states, return_constraints=True)
        
        # Q-values for policy improvement
        q1, q2 = self.critic(states, actions)
        q_values = torch.min(q1, q2)
        
        # Standard actor loss
        actor_loss = -q_values.mean()
        
        # Physics constraint penalty
        if constraint_values is not None:
            constraint_violations = torch.relu(-constraint_values)  # Penalize negative values
            constraint_penalty = torch.mean(torch.sum(constraint_violations, dim=-1))
            actor_loss += self.physics_weight * constraint_penalty
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def train_offline(
        self,
        dataset,
        num_epochs: int,
        batch_size: int = 256,
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train PIFRL on offline dataset with physics constraints."""
        metrics_history = []
        
        self.logger.info(f"Starting PIFRL training: {num_epochs} epochs, physics_weight={self.physics_weight}")
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            num_batches = max(1, dataset.size // batch_size)
            
            for batch_idx in range(num_batches):
                batch = dataset.sample_batch(batch_size)
                metrics = self.update(batch)
                epoch_metrics.append(metrics)
            
            # Average epoch metrics
            avg_metrics = TrainingMetrics(
                loss=np.mean([m.loss for m in epoch_metrics]),
                q_loss=np.mean([m.q_loss for m in epoch_metrics]),
                policy_loss=np.mean([m.policy_loss for m in epoch_metrics]),
                alpha_loss=np.mean([m.alpha_loss for m in epoch_metrics])
            )
            
            metrics_history.append(avg_metrics)
            
            if epoch % 50 == 0:
                violation_rate = np.mean(self.constraint_violations[-1000:]) if self.constraint_violations else 0
                physics_loss = np.mean(self.physics_loss_history[-100:]) if self.physics_loss_history else 0
                
                self.logger.info(
                    f"Epoch {epoch}: Loss={avg_metrics.loss:.4f}, "
                    f"Physics={physics_loss:.4f}, "
                    f"Violations={violation_rate:.3f}"
                )
        
        return metrics_history
    
    def get_physics_metrics(self) -> Dict[str, Any]:
        """Get physics-specific training metrics."""
        if not self.physics_loss_history:
            return {}
            
        recent_violations = self.constraint_violations[-1000:] if self.constraint_violations else []
        
        return {
            "physics_loss": {
                "current": self.physics_loss_history[-1],
                "mean": np.mean(self.physics_loss_history),
                "std": np.std(self.physics_loss_history),
                "trend": np.polyfit(range(len(self.physics_loss_history)), self.physics_loss_history, 1)[0]
            },
            "constraint_violations": {
                "total_count": len(self.constraint_violations),
                "recent_rate": np.mean(recent_violations) if recent_violations else 0.0,
                "violation_distribution": np.histogram(recent_violations, bins=10)[0].tolist() if recent_violations else []
            },
            "physics_weight": self.physics_weight,
            "constraint_tolerance": self.constraint_tolerance
        }
    
    def save(self, path: str) -> None:
        """Save physics-informed model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "power_flow_net": self.power_flow_net.state_dict(),
            "physics_loss_history": self.physics_loss_history,
            "constraint_violations": self.constraint_violations,
            "training_step": self.training_step
        }, path)
    
    def load(self, path: str) -> None:
        """Load physics-informed model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.power_flow_net.load_state_dict(checkpoint["power_flow_net"])
        self.physics_loss_history = checkpoint.get("physics_loss_history", [])
        self.constraint_violations = checkpoint.get("constraint_violations", [])
        self.training_step = checkpoint.get("training_step", 0)


class PIFRLClient(FederatedClient):
    """Federated client with physics-informed RL."""
    
    def __init__(
        self,
        client_id: str,
        algorithm: PIFRL,
        grid_data: List[Dict[str, Any]],
        physics_constraints: List[PhysicsConstraint],
        local_physics_weight: float = 1.0
    ):
        super().__init__(client_id, algorithm)
        self.local_data = grid_data
        self.physics_constraints = physics_constraints
        self.local_physics_weight = local_physics_weight
        self.constraint_history = []
        
    def local_update(
        self,
        global_parameters: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int
    ) -> ClientUpdate:
        """Physics-aware local training."""
        # Set global parameters
        self.algorithm.set_parameters(global_parameters)
        
        initial_metrics = self.algorithm.get_physics_metrics()
        
        # Local training with physics constraints
        for epoch in range(epochs):
            batch_data = self._sample_batch(batch_size)
            
            # Convert to tensor format expected by algorithm
            batch_tensor = self._convert_to_tensor_batch(batch_data)
            metrics = self.algorithm.update(batch_tensor)
            
            # Track constraint violations
            if hasattr(self.algorithm, 'constraint_violations'):
                recent_violations = self.algorithm.constraint_violations[-batch_size:]
                self.constraint_history.extend(recent_violations)
        
        final_metrics = self.algorithm.get_physics_metrics()
        updated_params = self.algorithm.get_parameters()
        
        # Calculate physics-informed metrics
        physics_improvement = (
            initial_metrics.get("physics_loss", {}).get("current", 1.0) - 
            final_metrics.get("physics_loss", {}).get("current", 1.0)
        )
        
        violation_rate = np.mean(self.constraint_history[-1000:]) if self.constraint_history else 0.0
        
        return ClientUpdate(
            client_id=self.client_id,
            parameters=updated_params,
            num_samples=len(self.local_data),
            loss=final_metrics.get("physics_loss", {}).get("current", 0.0),
            metrics={
                "physics_improvement": physics_improvement,
                "violation_rate": violation_rate,
                "constraint_satisfaction": 1.0 - violation_rate,
                "local_physics_weight": self.local_physics_weight,
                "num_constraints": len(self.physics_constraints)
            }
        )
    
    def _convert_to_tensor_batch(self, batch_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Convert batch data to tensor format."""
        # Extract fields and convert to tensors
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


# Utility functions for creating physics constraints

def voltage_stability_constraint(min_voltage: float = 0.95, max_voltage: float = 1.05) -> PhysicsConstraint:
    """Create voltage stability constraint."""
    def voltage_constraint_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Assume voltages are in specific positions in state
        if state.shape[-1] >= 10:
            voltages = state[..., :10]  # First 10 values as bus voltages
            min_margin = voltages - min_voltage
            max_margin = max_voltage - voltages
            return torch.min(torch.min(min_margin, dim=-1)[0], torch.min(max_margin, dim=-1)[0])
        return torch.ones(state.shape[0], device=state.device)  # Default safe
    
    return PhysicsConstraint(
        name="voltage_stability",
        constraint_fn=voltage_constraint_fn,
        weight=10.0,
        hard_constraint=True,
        tolerance=1e-3
    )


def power_balance_constraint(tolerance: float = 1e-2) -> PhysicsConstraint:
    """Create power balance constraint."""
    def power_balance_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Extract generation and load from state/action
        batch_size = state.shape[0]
        
        if state.shape[-1] >= 20 and action.shape[-1] >= 10:
            generation = action[..., :10]  # First 10 actions as generation
            load = state[..., 10:20]       # State positions 10-19 as load
            
            power_imbalance = torch.abs(torch.sum(generation, dim=-1) - torch.sum(load, dim=-1))
            return tolerance - power_imbalance  # Positive when satisfied
        
        return torch.ones(batch_size, device=state.device)  # Default safe
    
    return PhysicsConstraint(
        name="power_balance",
        constraint_fn=power_balance_fn,
        weight=20.0,
        hard_constraint=True,
        tolerance=tolerance
    )


def thermal_limit_constraint(max_loading: float = 0.9) -> PhysicsConstraint:
    """Create thermal loading constraint for lines."""
    def thermal_constraint_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Assume line flows can be computed from state
        if state.shape[-1] >= 30:
            line_flows = torch.abs(state[..., 20:30])  # Positions 20-29 as line flows
            loadings = line_flows / max_loading  # Normalize by thermal limit
            max_loading_violation = max_loading - torch.max(loadings, dim=-1)[0]
            return max_loading_violation
        
        return torch.ones(state.shape[0], device=state.device)
    
    return PhysicsConstraint(
        name="thermal_limits", 
        constraint_fn=thermal_constraint_fn,
        weight=15.0,
        hard_constraint=False,
        tolerance=1e-2
    )