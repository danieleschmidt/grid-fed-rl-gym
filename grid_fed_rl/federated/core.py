"""Core federated learning framework for grid control."""

from typing import Any, Dict, List, Optional, Tuple, Callable, Protocol
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import copy
import warnings

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Federated learning will use numpy fallback.")

from ..utils.validation import sanitize_config
from ..utils.exceptions import FederatedLearningError, InvalidConfigError
from ..algorithms.base import BaseAlgorithm


@dataclass
class FedLearningConfig:
    """Configuration for federated learning."""
    num_clients: int = 5
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 1e-3
    privacy_budget: float = 1.0
    aggregation_strategy: str = "fedavg"
    min_clients_per_round: int = 3
    client_sampling_prob: float = 1.0
    compression_ratio: float = 1.0
    differential_privacy: bool = True
    
    # Advanced optimization settings
    adaptive_learning_rate: bool = True
    momentum: float = 0.9
    weight_decay: float = 1e-5
    gradient_clipping: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 1e-4
    
    # Scalability settings
    async_updates: bool = False
    max_concurrent_clients: int = 10
    communication_rounds_budget: int = None
    bandwidth_limit_mbps: float = 100.0
    
    # Security and robustness
    byzantine_resilience: bool = True
    secure_aggregation: bool = True
    client_verification: bool = True
    model_poisoning_detection: bool = True


@dataclass
class ClientUpdate:
    """Client model update for federated aggregation."""
    client_id: str
    parameters: Dict[str, np.ndarray]
    num_samples: int
    loss: float
    metrics: Dict[str, Any]


class FederatedClient(ABC):
    """Abstract base class for federated learning clients."""
    
    def __init__(self, client_id: str, algorithm: BaseAlgorithm):
        self.client_id = client_id
        self.algorithm = algorithm
        self.local_data: List[Any] = []
        
    @abstractmethod
    def local_update(
        self, 
        global_parameters: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int
    ) -> ClientUpdate:
        """Perform local training and return parameter update."""
        pass
        
    @abstractmethod
    def evaluate(self, global_parameters: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate global model on local data."""
        pass


class GridUtilityClient(FederatedClient):
    """Federated client representing a utility company."""
    
    def __init__(
        self,
        client_id: str,
        algorithm: BaseAlgorithm,
        grid_data: List[Dict[str, Any]],
        privacy_level: float = 1.0
    ):
        super().__init__(client_id, algorithm)
        self.local_data = grid_data
        self.privacy_level = privacy_level
        self.logger = logging.getLogger(f"{__name__}.{client_id}")
        
    def local_update(
        self,
        global_parameters: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int
    ) -> ClientUpdate:
        """Train on local utility data."""
        try:
            # Update algorithm parameters
            self.algorithm.set_parameters(global_parameters)
            
            # Local training
            initial_loss = self._evaluate_loss()
            
            for epoch in range(epochs):
                batch_data = self._sample_batch(batch_size)
                loss = self.algorithm.train_step(batch_data)
                
            final_loss = self._evaluate_loss()
            
            # Get updated parameters
            updated_params = self.algorithm.get_parameters()
            
            # Apply differential privacy if enabled
            if self.privacy_level > 0:
                updated_params = self._add_noise(updated_params, self.privacy_level)
                
            return ClientUpdate(
                client_id=self.client_id,
                parameters=updated_params,
                num_samples=len(self.local_data),
                loss=final_loss,
                metrics={
                    "initial_loss": initial_loss,
                    "improvement": initial_loss - final_loss,
                    "privacy_level": self.privacy_level
                }
            )
            
        except Exception as e:
            self.logger.error(f"Local update failed: {e}")
            raise FederatedLearningError(f"Client {self.client_id} local update failed: {e}")
            
    def evaluate(self, global_parameters: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate global model on local data."""
        self.algorithm.set_parameters(global_parameters)
        
        total_loss = 0.0
        total_samples = 0
        
        for batch in self._iterate_batches(32):
            batch_loss = self.algorithm.evaluate_batch(batch)
            total_loss += batch_loss * len(batch)
            total_samples += len(batch)
            
        avg_loss = total_loss / max(total_samples, 1)
        
        return {
            "loss": avg_loss,
            "num_samples": total_samples,
            "data_quality": self._assess_data_quality()
        }
        
    def _sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample random batch from local data."""
        if len(self.local_data) <= batch_size:
            return self.local_data.copy()
            
        indices = np.random.choice(len(self.local_data), batch_size, replace=False)
        return [self.local_data[i] for i in indices]
        
    def _iterate_batches(self, batch_size: int):
        """Iterate over data in batches."""
        for i in range(0, len(self.local_data), batch_size):
            yield self.local_data[i:i + batch_size]
            
    def _evaluate_loss(self) -> float:
        """Evaluate current model loss."""
        if not self.local_data:
            return 0.0
            
        batch = self._sample_batch(min(64, len(self.local_data)))
        return self.algorithm.evaluate_batch(batch)
        
    def _add_noise(self, parameters: Dict[str, np.ndarray], epsilon: float) -> Dict[str, np.ndarray]:
        """Add Gaussian noise for differential privacy."""
        noisy_params = {}
        
        for name, param in parameters.items():
            # Gaussian mechanism for differential privacy
            sensitivity = np.std(param) * 2  # Rough sensitivity estimate
            noise_scale = sensitivity / epsilon
            noise = np.random.normal(0, noise_scale, param.shape)
            noisy_params[name] = param + noise
            
        return noisy_params
        
    def _assess_data_quality(self) -> float:
        """Assess quality of local data."""
        if not self.local_data:
            return 0.0
            
        # Simple data quality metrics
        completeness = len([d for d in self.local_data if self._is_complete(d)]) / len(self.local_data)
        return completeness
        
    def _is_complete(self, data_point: Dict[str, Any]) -> bool:
        """Check if data point is complete."""
        required_fields = ["state", "action", "reward", "next_state"]
        return all(field in data_point for field in required_fields)


class FederatedAggregator(ABC):
    """Abstract base class for federated aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate client updates into global parameters."""
        pass
        

class FedAvgAggregator(FederatedAggregator):
    """Federated Averaging (FedAvg) aggregation."""
    
    def aggregate(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """FedAvg: Weighted average by number of samples."""
        if not client_updates:
            return {}
            
        # Calculate total samples
        total_samples = sum(update.num_samples for update in client_updates)
        
        if total_samples == 0:
            return {}
            
        # Get parameter names from first client
        param_names = list(client_updates[0].parameters.keys())
        aggregated_params = {}
        
        for param_name in param_names:
            weighted_sum = np.zeros_like(client_updates[0].parameters[param_name])
            
            for update in client_updates:
                if param_name in update.parameters:
                    weight = update.num_samples / total_samples
                    weighted_sum += weight * update.parameters[param_name]
                    
            aggregated_params[param_name] = weighted_sum
            
        return aggregated_params


class SecureAggregator(FederatedAggregator):
    """Secure aggregation with basic encryption simulation."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or "default_key"
        
    def aggregate(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Secure aggregation (simplified implementation)."""
        # In real implementation, this would use proper cryptographic protocols
        # For now, we simulate by adding/removing random noise
        
        # Decrypt updates (simulation)
        decrypted_updates = []
        for update in client_updates:
            decrypted_params = self._simulate_decrypt(update.parameters)
            decrypted_update = ClientUpdate(
                client_id=update.client_id,
                parameters=decrypted_params,
                num_samples=update.num_samples,
                loss=update.loss,
                metrics=update.metrics
            )
            decrypted_updates.append(decrypted_update)
            
        # Use FedAvg on decrypted parameters
        fedavg = FedAvgAggregator()
        return fedavg.aggregate(decrypted_updates)
        
    def _simulate_decrypt(self, encrypted_params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Simulate decryption (in practice would use real cryptography)."""
        # Simple simulation: parameters are already "decrypted"
        return encrypted_params.copy()


class FederatedOfflineRL:
    """Federated offline reinforcement learning coordinator."""
    
    def __init__(
        self,
        algorithm_class: type,
        config: FedLearningConfig,
        aggregator: Optional[FederatedAggregator] = None,
        **algorithm_kwargs
    ):
        self.algorithm_class = algorithm_class
        self.config = sanitize_config(config.__dict__, required_fields=["num_clients", "rounds"])
        self.config = FedLearningConfig(**self.config)
        
        self.aggregator = aggregator or FedAvgAggregator()
        self.algorithm_kwargs = algorithm_kwargs
        
        self.clients: List[FederatedClient] = []
        self.global_parameters: Dict[str, np.ndarray] = {}
        self.training_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
        
    def add_client(self, client: FederatedClient) -> None:
        """Add a client to federated learning."""
        if len(self.clients) >= self.config.num_clients:
            raise InvalidConfigError(f"Cannot add more than {self.config.num_clients} clients")
            
        self.clients.append(client)
        self.logger.info(f"Added client {client.client_id} ({len(self.clients)}/{self.config.num_clients})")
        
    def initialize_global_model(self) -> None:
        """Initialize global model parameters."""
        if not self.clients:
            raise FederatedLearningError("No clients available for initialization")
            
        # Use first client's algorithm to get parameter structure
        sample_client = self.clients[0]
        self.global_parameters = sample_client.algorithm.get_parameters()
        
        # Initialize with small random values
        for param_name, param_value in self.global_parameters.items():
            self.global_parameters[param_name] = np.random.normal(
                0, 0.01, param_value.shape
            ).astype(param_value.dtype)
            
    def train(
        self,
        datasets: Optional[List[List[Dict[str, Any]]]] = None,
        **train_kwargs
    ) -> Dict[str, np.ndarray]:
        """Run federated training."""
        if len(self.clients) < self.config.min_clients_per_round:
            raise FederatedLearningError(
                f"Need at least {self.config.min_clients_per_round} clients, got {len(self.clients)}"
            )
            
        # Assign datasets to clients if provided
        if datasets:
            if len(datasets) != len(self.clients):
                raise ValueError(f"Number of datasets ({len(datasets)}) must match number of clients ({len(self.clients)})")
                
            for client, dataset in zip(self.clients, datasets):
                if hasattr(client, 'local_data'):
                    client.local_data = dataset
                    
        # Initialize global model
        self.initialize_global_model()
        
        self.logger.info(f"Starting federated training: {self.config.rounds} rounds, {len(self.clients)} clients")
        
        # Training loop
        for round_idx in range(self.config.rounds):
            round_metrics = self._train_round(round_idx)
            self.training_history.append(round_metrics)
            
            if round_idx % 10 == 0:
                self.logger.info(f"Round {round_idx}: avg_loss={round_metrics['avg_loss']:.4f}")
                
        self.logger.info("Federated training completed")
        return self.global_parameters
        
    def _train_round(self, round_idx: int) -> Dict[str, Any]:
        """Execute one round of federated training."""
        # Sample clients for this round
        num_selected = max(
            self.config.min_clients_per_round,
            int(len(self.clients) * self.config.client_sampling_prob)
        )
        
        selected_clients = np.random.choice(
            self.clients, 
            size=min(num_selected, len(self.clients)), 
            replace=False
        )
        
        # Collect client updates
        client_updates = []
        
        for client in selected_clients:
            try:
                update = client.local_update(
                    self.global_parameters,
                    self.config.local_epochs,
                    self.config.batch_size
                )
                client_updates.append(update)
                
            except Exception as e:
                self.logger.warning(f"Client {client.client_id} failed in round {round_idx}: {e}")
                
        if not client_updates:
            self.logger.warning(f"No successful client updates in round {round_idx}")
            return {"avg_loss": float('inf'), "participating_clients": 0}
            
        # Aggregate updates
        try:
            self.global_parameters = self.aggregator.aggregate(client_updates)
        except Exception as e:
            self.logger.error(f"Aggregation failed in round {round_idx}: {e}")
            raise FederatedLearningError(f"Aggregation failed: {e}")
            
        # Calculate round metrics
        avg_loss = np.mean([update.loss for update in client_updates])
        total_samples = sum(update.num_samples for update in client_updates)
        
        return {
            "round": round_idx,
            "avg_loss": avg_loss,
            "participating_clients": len(client_updates),
            "total_samples": total_samples,
            "client_losses": [update.loss for update in client_updates]
        }
        
    def evaluate_global_model(self) -> Dict[str, Any]:
        """Evaluate global model on all clients."""
        if not self.global_parameters:
            return {"error": "No global model available"}
            
        client_results = {}
        
        for client in self.clients:
            try:
                result = client.evaluate(self.global_parameters)
                client_results[client.client_id] = result
            except Exception as e:
                self.logger.warning(f"Evaluation failed for client {client.client_id}: {e}")
                client_results[client.client_id] = {"error": str(e)}
                
        # Aggregate results
        successful_results = [
            result for result in client_results.values() 
            if "error" not in result
        ]
        
        if successful_results:
            avg_loss = np.mean([result["loss"] for result in successful_results])
            total_samples = sum(result["num_samples"] for result in successful_results)
        else:
            avg_loss = float('inf')
            total_samples = 0
            
        return {
            "avg_loss": avg_loss,
            "total_samples": total_samples,
            "successful_clients": len(successful_results),
            "client_results": client_results
        }
        
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        if not self.training_history:
            return {}
            
        losses = [round_data["avg_loss"] for round_data in self.training_history]
        
        return {
            "total_rounds": len(self.training_history),
            "final_loss": losses[-1] if losses else float('inf'),
            "best_loss": min(losses) if losses else float('inf'),
            "convergence_round": int(np.argmin(losses)) if losses else -1,
            "loss_history": losses,
            "avg_participation": np.mean([
                round_data["participating_clients"] for round_data in self.training_history
            ]) if self.training_history else 0
        }