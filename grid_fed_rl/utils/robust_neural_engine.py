"""Robust neural network engine with comprehensive error handling."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import time
from contextlib import contextmanager
import traceback
from dataclasses import dataclass

from .exceptions import GridEnvironmentError
from .validation import validate_tensor_input
from .monitoring import SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class NeuralNetworkHealth:
    """Health metrics for neural network operations."""
    gradient_norm: float
    weight_norm: float
    activation_stats: Dict[str, float]
    loss_value: float
    convergence_rate: float
    numerical_stability: bool
    memory_usage_mb: float
    computation_time_ms: float


class RobustNeuralEngine:
    """Production-ready neural network engine with robust error handling."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        numerical_stability_check: bool = True,
        gradient_clipping: float = 1.0,
        memory_monitoring: bool = True
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.numerical_stability_check = numerical_stability_check
        self.gradient_clipping = gradient_clipping
        self.memory_monitoring = memory_monitoring
        
        # Performance tracking
        self.execution_times = []
        self.memory_usage = []
        self.error_count = 0
        self.total_operations = 0
        
        # Numerical stability tracking
        self.nan_count = 0
        self.inf_count = 0
        self.overflow_count = 0
        
        logger.info(f"Initialized RobustNeuralEngine on device: {self.device}")
    
    @contextmanager
    def safe_computation(self, operation_name: str = "neural_computation"):
        """Context manager for safe neural network computations."""
        start_time = time.time()
        initial_memory = self._get_memory_usage() if self.memory_monitoring else 0
        
        try:
            self.total_operations += 1
            yield
            
            # Check for numerical issues
            if self.numerical_stability_check:
                self._check_numerical_stability()
                
        except RuntimeError as e:
            self.error_count += 1
            if "out of memory" in str(e).lower():
                logger.error(f"GPU memory exhausted during {operation_name}")
                torch.cuda.empty_cache()
                raise GridEnvironmentError(f"GPU memory exhausted: {e}")
            elif "cuda" in str(e).lower():
                logger.error(f"CUDA error during {operation_name}: {e}")
                raise GridEnvironmentError(f"CUDA error: {e}")
            else:
                logger.error(f"Runtime error during {operation_name}: {e}")
                raise
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Unexpected error during {operation_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise GridEnvironmentError(f"Neural computation failed: {e}")
            
        finally:
            computation_time = (time.time() - start_time) * 1000
            final_memory = self._get_memory_usage() if self.memory_monitoring else 0
            
            self.execution_times.append(computation_time)
            if self.memory_monitoring:
                self.memory_usage.append(final_memory - initial_memory)
            
            # Log performance if slow
            if computation_time > 100:  # More than 100ms
                logger.warning(f"Slow {operation_name}: {computation_time:.2f}ms")
    
    def safe_forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        training: bool = False
    ) -> Tuple[torch.Tensor, NeuralNetworkHealth]:
        """Safely execute forward pass with comprehensive monitoring."""
        
        with self.safe_computation("forward_pass"):
            # Input validation
            if not validate_tensor_input(inputs):
                raise GridEnvironmentError("Invalid input tensor")
            
            # Set model mode
            model.train(training)
            
            # Execute forward pass
            outputs = model(inputs.to(self.device))
            
            # Health check
            health = self._assess_network_health(model, inputs, outputs)
            
            return outputs, health
    
    def safe_backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        retain_graph: bool = False
    ) -> NeuralNetworkHealth:
        """Safely execute backward pass with gradient monitoring."""
        
        with self.safe_computation("backward_pass"):
            # Backward pass
            loss.backward(retain_graph=retain_graph)
            
            # Gradient clipping
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Health assessment
            health = self._assess_gradient_health(model, loss)
            
            return health
    
    def safe_inference(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Safe inference with automatic batching for large inputs."""
        
        with torch.no_grad():
            model.eval()
            
            if batch_size is None or inputs.size(0) <= batch_size:
                with self.safe_computation("inference"):
                    outputs, _ = self.safe_forward(model, inputs, training=False)
                    return outputs
            else:
                # Batch processing for large inputs
                outputs = []
                for i in range(0, inputs.size(0), batch_size):
                    batch = inputs[i:i+batch_size]
                    with self.safe_computation(f"inference_batch_{i}"):
                        batch_output, _ = self.safe_forward(model, batch, training=False)
                        outputs.append(batch_output.cpu())
                
                return torch.cat(outputs, dim=0).to(self.device)
    
    def _assess_network_health(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor
    ) -> NeuralNetworkHealth:
        """Assess overall health of neural network."""
        
        # Gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        gradient_norm = total_norm ** (1. / 2)
        
        # Weight norm
        weight_norm = sum(p.data.norm(2).item() for p in model.parameters())
        
        # Activation statistics
        activation_stats = {
            'mean': outputs.mean().item(),
            'std': outputs.std().item(),
            'min': outputs.min().item(),
            'max': outputs.max().item()
        }
        
        # Numerical stability
        numerical_stability = not (
            torch.isnan(outputs).any() or 
            torch.isinf(outputs).any() or
            torch.abs(outputs).max() > 1e6
        )
        
        return NeuralNetworkHealth(
            gradient_norm=gradient_norm,
            weight_norm=weight_norm,
            activation_stats=activation_stats,
            loss_value=0.0,  # Will be set separately
            convergence_rate=0.0,  # Will be computed from history
            numerical_stability=numerical_stability,
            memory_usage_mb=self._get_memory_usage(),
            computation_time_ms=self.execution_times[-1] if self.execution_times else 0.0
        )
    
    def _assess_gradient_health(
        self,
        model: nn.Module,
        loss: torch.Tensor
    ) -> NeuralNetworkHealth:
        """Assess gradient health after backward pass."""
        
        # Gradient analysis
        gradients = []
        for p in model.parameters():
            if p.grad is not None:
                gradients.append(p.grad.data.flatten())
        
        if gradients:
            all_gradients = torch.cat(gradients)
            gradient_norm = all_gradients.norm(2).item()
            
            # Check for gradient problems
            nan_gradients = torch.isnan(all_gradients).sum().item()
            inf_gradients = torch.isinf(all_gradients).sum().item()
            
            if nan_gradients > 0:
                self.nan_count += 1
                logger.warning(f"Found {nan_gradients} NaN gradients")
            
            if inf_gradients > 0:
                self.inf_count += 1
                logger.warning(f"Found {inf_gradients} infinite gradients")
        else:
            gradient_norm = 0.0
        
        return NeuralNetworkHealth(
            gradient_norm=gradient_norm,
            weight_norm=0.0,  # Not computed here
            activation_stats={},
            loss_value=loss.item(),
            convergence_rate=0.0,
            numerical_stability=not (torch.isnan(loss) or torch.isinf(loss)),
            memory_usage_mb=self._get_memory_usage(),
            computation_time_ms=self.execution_times[-1] if self.execution_times else 0.0
        )
    
    def _check_numerical_stability(self):
        """Check for numerical stability issues."""
        if torch.cuda.is_available():
            # Check CUDA errors
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                logger.error(f"CUDA synchronization failed: {e}")
                raise GridEnvironmentError(f"CUDA numerical instability: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.execution_times:
            return {'status': 'No operations performed yet'}
        
        return {
            'total_operations': self.total_operations,
            'error_count': self.error_count,
            'error_rate': self.error_count / self.total_operations,
            'avg_computation_time_ms': np.mean(self.execution_times),
            'max_computation_time_ms': np.max(self.execution_times),
            'min_computation_time_ms': np.min(self.execution_times),
            'avg_memory_usage_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'numerical_issues': {
                'nan_count': self.nan_count,
                'inf_count': self.inf_count,
                'overflow_count': self.overflow_count
            },
            'device': str(self.device)
        }
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.execution_times.clear()
        self.memory_usage.clear()
        self.error_count = 0
        self.total_operations = 0
        self.nan_count = 0
        self.inf_count = 0
        self.overflow_count = 0
        logger.info("Performance statistics reset")


class RobustPolicyNetwork(nn.Module):
    """Robust policy network with built-in safety features."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bounded actions
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass with input validation."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if state.size(-1) != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {state.size(-1)}")
        
        return self.network(state)
    
    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from state with error handling."""
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action = self.forward(state_tensor)
            
            return action.cpu().numpy().squeeze()
            
        except Exception as e:
            logger.error(f"Error in get_action: {e}")
            # Return safe zero action
            return np.zeros(self.action_dim)