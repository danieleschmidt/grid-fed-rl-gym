"""Safety-constrained reinforcement learning algorithms."""

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Safe RL algorithms will use numpy fallback.")

from .base import BaseAlgorithm
from ..utils.validation import validate_constraints, sanitize_config
from ..utils.exceptions import SafetyViolationError, InvalidConstraintError
from ..utils.safety import ConstraintChecker, SafetyShield


@dataclass 
class SafetyConstraint:
    """Definition of a safety constraint."""
    name: str
    constraint_function: Callable[[np.ndarray], float]  # Returns constraint value (>= 0 is safe)
    violation_penalty: float = 100.0
    hard_constraint: bool = False  # If True, prevents action execution
    tolerance: float = 0.0


class ConstraintBuffer:
    """Buffer for storing constraint violation data."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0
        
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        constraint_values: Dict[str, float],
        violations: Dict[str, bool]
    ) -> None:
        """Add constraint data to buffer."""
        data = {
            "state": state.copy(),
            "action": action.copy(),
            "constraint_values": constraint_values.copy(),
            "violations": violations.copy(),
            "timestamp": np.datetime64('now')
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch from constraint buffer."""
        if len(self.buffer) == 0:
            return []
            
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
        
    def get_violation_rate(self, constraint_name: str) -> float:
        """Get violation rate for specific constraint."""
        if not self.buffer:
            return 0.0
            
        violations = [data["violations"].get(constraint_name, False) for data in self.buffer]
        return np.mean(violations)


class SafetyLayer:
    """Neural network layer that enforces safety constraints."""
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        constraints: List[SafetyConstraint],
        hidden_dims: List[int] = [64, 32]
    ):
        if TORCH_AVAILABLE:
            # Import here to avoid issues when torch not available
            import torch.nn as nn
            
            class TorchSafetyLayer(nn.Module):
                def __init__(self, input_dim, action_dim, num_constraints, hidden_dims):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim + action_dim
                    
                    for hidden_dim in hidden_dims:
                        layers.extend([
                            nn.Linear(prev_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.1)
                        ])
                        prev_dim = hidden_dim
                        
                    layers.append(nn.Linear(prev_dim, num_constraints))
                    self.safety_critic = nn.Sequential(*layers)
                    
                def forward(self, x):
                    return self.safety_critic(x)
            
            self.torch_layer = TorchSafetyLayer(input_dim, action_dim, len(constraints), hidden_dims)
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.constraints = constraints
        
        # Numpy fallback weights
        self.weights = []
        prev_dim = input_dim + action_dim
        
        for hidden_dim in hidden_dims:
            self.weights.append(np.random.normal(0, 0.1, (prev_dim, hidden_dim)))
            prev_dim = hidden_dim
            
        self.weights.append(np.random.normal(0, 0.1, (prev_dim, len(constraints))))
            
    def forward(self, state: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """Predict constraint violations for state-action pair."""
        if TORCH_AVAILABLE and hasattr(self, 'torch_layer'):
            import torch
            state_tensor = torch.FloatTensor(state)
            action_tensor = torch.FloatTensor(action)
            input_tensor = torch.cat([state_tensor, action_tensor], dim=-1)
            
            constraint_values = self.torch_layer(input_tensor).detach().numpy()
        else:
            # Numpy fallback
            x = np.concatenate([state, action])
            
            for i, weight in enumerate(self.weights[:-1]):
                x = np.maximum(0, np.dot(x, weight))  # ReLU activation
                
            constraint_values = np.dot(x, self.weights[-1])
            
        # Map to constraint dictionary
        result = {}
        for i, constraint in enumerate(self.constraints):
            if i < len(constraint_values):
                result[constraint.name] = constraint_values[i]
                
        return result
        
    def update(self, batch_data: List[Dict[str, Any]], learning_rate: float = 1e-3) -> float:
        """Update safety layer from constraint violation data."""
        if not batch_data:
            return 0.0
            
        if TORCH_AVAILABLE and hasattr(self, 'safety_critic'):
            return self._torch_update(batch_data, learning_rate)
        else:
            return self._numpy_update(batch_data, learning_rate)
            
    def _torch_update(self, batch_data: List[Dict[str, Any]], learning_rate: float) -> float:
        """PyTorch-based update."""
        optimizer = optim.Adam(self.safety_critic.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        states = torch.FloatTensor([data["state"] for data in batch_data])
        actions = torch.FloatTensor([data["action"] for data in batch_data])
        
        # Target values: actual constraint evaluations
        targets = []
        for data in batch_data:
            target = [data["constraint_values"].get(c.name, 0.0) for c in self.constraints]
            targets.append(target)
        targets = torch.FloatTensor(targets)
        
        # Forward pass
        inputs = torch.cat([states, actions], dim=-1)
        predictions = self.safety_critic(inputs)
        
        # Compute loss and update
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
    def _numpy_update(self, batch_data: List[Dict[str, Any]], learning_rate: float) -> float:
        """Numpy-based gradient descent update."""
        if not batch_data:
            return 0.0
            
        # Prepare training data
        X = []
        y = []
        
        for data in batch_data:
            x = np.concatenate([data["state"], data["action"]])
            target = [data["constraint_values"].get(c.name, 0.0) for c in self.constraints]
            X.append(x)
            y.append(target)
            
        X = np.array(X)
        y = np.array(y)
        
        # Forward pass
        predictions = []
        activations = [X]  # Store activations for backprop
        
        for i, weight in enumerate(self.weights[:-1]):
            z = np.dot(activations[-1], weight)
            a = np.maximum(0, z)  # ReLU
            activations.append(a)
            
        # Output layer
        predictions = np.dot(activations[-1], self.weights[-1])
        
        # Compute loss (MSE)
        loss = np.mean((predictions - y) ** 2)
        
        # Backward pass (simplified)
        d_output = 2 * (predictions - y) / len(batch_data)
        
        # Update output weights
        self.weights[-1] -= learning_rate * np.dot(activations[-1].T, d_output)
        
        return loss


class SafeRL(BaseAlgorithm):
    """Safety-constrained reinforcement learning base algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        constraints: List[SafetyConstraint],
        safety_weight: float = 10.0,
        constraint_buffer_size: int = 10000,
        safety_layer_lr: float = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraints = constraints
        self.safety_weight = safety_weight
        self.safety_layer_lr = safety_layer_lr
        
        # Initialize safety components
        self.constraint_buffer = ConstraintBuffer(constraint_buffer_size)
        self.safety_layer = SafetyLayer(state_dim, action_dim, constraints)
        self.constraint_checker = ConstraintChecker(constraints)
        
        # Track safety metrics
        self.total_violations = 0
        self.total_actions = 0
        self.constraint_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def get_safe_action(
        self,
        state: np.ndarray,
        unsafe_action: np.ndarray,
        max_corrections: int = 10
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get safety-corrected action."""
        self.total_actions += 1
        
        # Check if original action is safe
        constraint_values = self._evaluate_constraints(state, unsafe_action)
        violations = {name: val < 0 for name, val in constraint_values.items()}
        
        if not any(violations.values()):
            # Original action is safe
            return unsafe_action, {
                "corrections_made": 0,
                "constraint_values": constraint_values,
                "violations": violations,
                "safe": True
            }
            
        # Action violates constraints - attempt correction
        self.total_violations += 1
        corrected_action = unsafe_action.copy()
        corrections_made = 0
        
        for correction in range(max_corrections):
            # Use gradient-based correction
            corrected_action = self._correct_action(state, corrected_action)
            corrections_made += 1
            
            # Check if corrected action is safe
            new_constraint_values = self._evaluate_constraints(state, corrected_action)
            new_violations = {name: val < 0 for name, val in new_constraint_values.items()}
            
            if not any(new_violations.values()):
                # Found safe action
                break
                
        # Store constraint data for learning
        self.constraint_buffer.add(state, corrected_action, new_constraint_values, new_violations)
        
        # Update constraint history
        self.constraint_history.append({
            "violations": new_violations,
            "corrections": corrections_made,
            "constraint_values": new_constraint_values
        })
        
        return corrected_action, {
            "corrections_made": corrections_made,
            "constraint_values": new_constraint_values,
            "violations": new_violations,
            "safe": not any(new_violations.values()),
            "original_violations": violations
        }
        
    def _evaluate_constraints(self, state: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """Evaluate all constraints for state-action pair."""
        constraint_values = {}
        
        # Use safety layer predictions if available
        try:
            predicted_values = self.safety_layer.forward(state, action)
            constraint_values.update(predicted_values)
        except Exception as e:
            self.logger.debug(f"Safety layer evaluation failed: {e}")
            
        # Also evaluate actual constraint functions
        for constraint in self.constraints:
            try:
                # Constraint functions may need both state and action
                value = constraint.constraint_function(np.concatenate([state, action]))
                constraint_values[constraint.name] = value
            except Exception as e:
                self.logger.warning(f"Constraint {constraint.name} evaluation failed: {e}")
                constraint_values[constraint.name] = -1.0  # Assume violation on error
                
        return constraint_values
        
    def _correct_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Correct action to reduce constraint violations."""
        corrected = action.copy()
        
        # Simple gradient-based correction
        epsilon = 1e-4
        
        for i in range(len(action)):
            # Finite difference gradient
            action_plus = action.copy()
            action_plus[i] += epsilon
            
            action_minus = action.copy() 
            action_minus[i] -= epsilon
            
            # Evaluate constraint violations
            violations_plus = sum(
                max(0, -val) for val in self._evaluate_constraints(state, action_plus).values()
            )
            violations_minus = sum(
                max(0, -val) for val in self._evaluate_constraints(state, action_minus).values()
            )
            
            # Move in direction that reduces violations
            if violations_plus < violations_minus:
                corrected[i] += epsilon * 10  # Scale up the correction
            else:
                corrected[i] -= epsilon * 10
                
        return corrected
        
    def update_safety_layer(self, batch_size: int = 64) -> float:
        """Update safety layer from constraint buffer."""
        batch_data = self.constraint_buffer.sample(batch_size)
        if not batch_data:
            return 0.0
            
        return self.safety_layer.update(batch_data, self.safety_layer_lr)
        
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get comprehensive safety metrics."""
        if self.total_actions == 0:
            return {
                "violation_rate": 0.0,
                "total_violations": 0,
                "total_actions": 0
            }
            
        # Overall violation rate
        violation_rate = self.total_violations / self.total_actions
        
        # Per-constraint violation rates
        constraint_rates = {}
        for constraint in self.constraints:
            constraint_rates[constraint.name] = self.constraint_buffer.get_violation_rate(constraint.name)
            
        # Recent safety performance
        recent_history = self.constraint_history[-100:] if self.constraint_history else []
        recent_violations = sum(
            any(entry["violations"].values()) for entry in recent_history
        ) if recent_history else 0
        recent_rate = recent_violations / max(len(recent_history), 1)
        
        return {
            "violation_rate": violation_rate,
            "total_violations": self.total_violations,
            "total_actions": self.total_actions,
            "constraint_violation_rates": constraint_rates,
            "recent_violation_rate": recent_rate,
            "buffer_size": len(self.constraint_buffer.buffer),
            "avg_corrections_per_violation": np.mean([
                entry["corrections"] for entry in recent_history if any(entry["violations"].values())
            ]) if recent_history else 0.0
        }
        

class ConstrainedPolicyOptimization(SafeRL):
    """Constrained Policy Optimization (CPO) algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        constraints: List[SafetyConstraint],
        constraint_tolerance: float = 0.1,
        penalty_lr: float = 1e-2,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, constraints, **kwargs)
        
        self.constraint_tolerance = constraint_tolerance
        self.penalty_lr = penalty_lr
        
        # Lagrange multipliers for constraints
        self.constraint_penalties = {c.name: 1.0 for c in constraints}
        
    def update_penalties(self, constraint_violations: Dict[str, float]) -> None:
        """Update Lagrange multipliers based on constraint violations."""
        for name, violation in constraint_violations.items():
            if name in self.constraint_penalties:
                # Increase penalty if constraint is violated
                if violation > self.constraint_tolerance:
                    self.constraint_penalties[name] *= (1 + self.penalty_lr)
                # Decrease penalty if constraint is well-satisfied
                elif violation < -self.constraint_tolerance:
                    self.constraint_penalties[name] *= max(0.1, 1 - self.penalty_lr)
                    
    def compute_constrained_loss(
        self,
        base_loss: float,
        constraint_values: Dict[str, float]
    ) -> float:
        """Compute loss with constraint penalties."""
        total_loss = base_loss
        
        for name, value in constraint_values.items():
            if name in self.constraint_penalties:
                # Add penalty for constraint violations (value < 0)
                penalty = self.constraint_penalties[name]
                violation = max(0, -value)  # Only penalize violations
                total_loss += penalty * violation
                
        return total_loss
        
    def train_step(self, batch_data: List[Dict[str, Any]]) -> float:
        """Training step with constraint optimization."""
        # Base algorithm training (to be implemented by subclasses)
        base_loss = self._base_train_step(batch_data)
        
        # Update constraint penalties
        if batch_data:
            avg_violations = {}
            for constraint in self.constraints:
                violations = [
                    data.get("constraint_values", {}).get(constraint.name, 0.0)
                    for data in batch_data
                ]
                avg_violations[constraint.name] = np.mean(violations)
                
            self.update_penalties(avg_violations)
            
        # Update safety layer
        safety_loss = self.update_safety_layer(len(batch_data))
        
        return base_loss + safety_loss
        
    def _base_train_step(self, batch_data: List[Dict[str, Any]]) -> float:
        """Base algorithm training step (to be implemented by subclasses)."""
        # Placeholder implementation
        if not batch_data:
            return 0.0
            
        # Simple loss calculation for demonstration
        losses = [data.get("loss", 1.0) for data in batch_data]
        return np.mean(losses)


# Utility functions for creating common safety constraints

def voltage_constraint(min_voltage: float = 0.95, max_voltage: float = 1.05) -> SafetyConstraint:
    """Create voltage constraint for power systems."""
    def voltage_check(state_action: np.ndarray) -> float:
        # Assume first N values in state are bus voltages
        # This is a simplified example - real implementation would need proper state parsing
        if len(state_action) < 3:
            return 1.0  # Assume safe if insufficient data
            
        voltages = state_action[:3]  # First 3 values as voltages
        min_margin = np.min(voltages - min_voltage)
        max_margin = np.min(max_voltage - voltages)
        return min(min_margin, max_margin)
        
    return SafetyConstraint(
        name="voltage_limits",
        constraint_function=voltage_check,
        violation_penalty=50.0,
        hard_constraint=True
    )


def frequency_constraint(min_freq: float = 59.5, max_freq: float = 60.5) -> SafetyConstraint:
    """Create frequency constraint for power systems."""
    def frequency_check(state_action: np.ndarray) -> float:
        # Assume frequency is at a specific position in state
        if len(state_action) < 10:
            return 1.0
            
        frequency = state_action[9]  # Example position
        min_margin = frequency - min_freq
        max_margin = max_freq - frequency
        return min(min_margin, max_margin)
        
    return SafetyConstraint(
        name="frequency_limits", 
        constraint_function=frequency_check,
        violation_penalty=100.0,
        hard_constraint=True
    )


def thermal_constraint(max_loading: float = 0.8) -> SafetyConstraint:
    """Create thermal loading constraint for power lines."""
    def thermal_check(state_action: np.ndarray) -> float:
        # Assume line loadings are at specific positions
        if len(state_action) < 8:
            return 1.0
            
        loadings = state_action[4:8]  # Example positions
        max_margin = max_loading - np.max(np.abs(loadings))
        return max_margin
        
    return SafetyConstraint(
        name="thermal_limits",
        constraint_function=thermal_check,
        violation_penalty=75.0,
        hard_constraint=False
    )