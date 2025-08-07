"""Privacy mechanisms for federated learning."""

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
import logging

from ..utils.validation import validate_privacy_parameters
from ..utils.exceptions import PrivacyError


@dataclass
class PrivacyBudget:
    """Privacy budget management for differential privacy."""
    epsilon: float = 1.0
    delta: float = 1e-5
    spent_epsilon: float = 0.0
    max_queries: int = 1000
    query_count: int = 0
    
    def can_spend(self, epsilon_cost: float) -> bool:
        """Check if we can afford to spend epsilon."""
        return (self.spent_epsilon + epsilon_cost <= self.epsilon and 
                self.query_count < self.max_queries)
    
    def spend(self, epsilon_cost: float) -> None:
        """Spend privacy budget."""
        if not self.can_spend(epsilon_cost):
            raise PrivacyError(f"Insufficient privacy budget. Need {epsilon_cost}, have {self.epsilon - self.spent_epsilon}")
        
        self.spent_epsilon += epsilon_cost
        self.query_count += 1
    
    def remaining(self) -> float:
        """Get remaining privacy budget."""
        return self.epsilon - self.spent_epsilon


class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mechanism: str = "gaussian",
        sensitivity: float = 1.0
    ):
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        self.mechanism = mechanism
        self.sensitivity = sensitivity
        self.logger = logging.getLogger(__name__)
        
        if mechanism not in ["gaussian", "laplace"]:
            raise ValueError(f"Unsupported mechanism: {mechanism}")
    
    def add_noise(
        self,
        data: np.ndarray,
        epsilon: Optional[float] = None,
        sensitivity: Optional[float] = None
    ) -> np.ndarray:
        """Add differentially private noise to data."""
        eps = epsilon or self.budget.epsilon / 10  # Use 1/10 of budget by default
        sens = sensitivity or self.sensitivity
        
        if not self.budget.can_spend(eps):
            self.logger.warning("Insufficient privacy budget, using minimal noise")
            eps = min(eps, self.budget.remaining())
        
        if self.mechanism == "gaussian":
            noise = self._gaussian_noise(data.shape, eps, sens)
        else:  # laplace
            noise = self._laplace_noise(data.shape, eps, sens)
        
        self.budget.spend(eps)
        noisy_data = data + noise
        
        self.logger.debug(f"Added {self.mechanism} noise with epsilon={eps:.4f}")
        return noisy_data
    
    def _gaussian_noise(self, shape: Tuple[int, ...], epsilon: float, sensitivity: float) -> np.ndarray:
        """Generate Gaussian noise for (epsilon, delta)-DP."""
        # Gaussian mechanism: sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        sigma = np.sqrt(2 * np.log(1.25 / self.budget.delta)) * sensitivity / epsilon
        return np.random.normal(0, sigma, shape)
    
    def _laplace_noise(self, shape: Tuple[int, ...], epsilon: float, sensitivity: float) -> np.ndarray:
        """Generate Laplace noise for epsilon-DP."""
        # Laplace mechanism: b = sensitivity / epsilon
        b = sensitivity / epsilon
        return np.random.laplace(0, b, shape)
    
    def clip_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        clip_norm: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Clip gradients for bounded sensitivity."""
        clipped = {}
        
        for name, grad in gradients.items():
            grad_norm = np.linalg.norm(grad)
            if grad_norm > clip_norm:
                clipped[name] = grad * (clip_norm / grad_norm)
            else:
                clipped[name] = grad.copy()
        
        return clipped
    
    def private_sum(
        self,
        values: List[np.ndarray],
        epsilon: float,
        sensitivity: Optional[float] = None
    ) -> np.ndarray:
        """Compute differentially private sum."""
        if not values:
            return np.array([])
        
        # Compute sum
        total = sum(values)
        
        # Add noise
        sens = sensitivity or len(values)  # Each contribution adds to sensitivity
        noisy_sum = self.add_noise(total, epsilon, sens)
        
        return noisy_sum
    
    def private_mean(
        self,
        values: List[np.ndarray],
        epsilon: float,
        sensitivity: Optional[float] = None
    ) -> np.ndarray:
        """Compute differentially private mean."""
        if not values:
            return np.array([])
        
        # Compute noisy sum then divide by count
        noisy_sum = self.private_sum(values, epsilon, sensitivity)
        return noisy_sum / len(values)
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current privacy budget status."""
        return {
            "total_epsilon": self.budget.epsilon,
            "spent_epsilon": self.budget.spent_epsilon,
            "remaining_epsilon": self.budget.remaining(),
            "delta": self.budget.delta,
            "query_count": self.budget.query_count,
            "max_queries": self.budget.max_queries,
            "budget_exhausted": self.budget.remaining() <= 0
        }


class SecureAggregation:
    """Secure aggregation protocol for federated learning."""
    
    def __init__(self, num_clients: int, threshold: int = None):
        self.num_clients = num_clients
        self.threshold = threshold or max(2, num_clients // 2)
        self.client_keys = {}
        self.logger = logging.getLogger(__name__)
        
        if self.threshold > num_clients:
            raise ValueError(f"Threshold ({threshold}) cannot exceed number of clients ({num_clients})")
    
    def generate_client_keys(self) -> Dict[str, bytes]:
        """Generate encryption keys for clients (simulated)."""
        keys = {}
        for i in range(self.num_clients):
            client_id = f"client_{i}"
            # In real implementation, this would use proper key generation
            keys[client_id] = np.random.bytes(32)
        
        self.client_keys = keys
        return keys
    
    def encrypt_update(
        self,
        client_id: str,
        parameters: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Encrypt client update (simulated)."""
        if client_id not in self.client_keys:
            raise PrivacyError(f"No key found for client {client_id}")
        
        # Simplified encryption: add deterministic "encryption" mask
        encrypted = {}
        key_seed = hash(self.client_keys[client_id]) % 2**32
        rng = np.random.RandomState(key_seed)
        
        for name, param in parameters.items():
            # Simple additive mask (in real implementation would use proper encryption)
            mask = rng.normal(0, 0.1, param.shape)
            encrypted[name] = param + mask
        
        return encrypted
    
    def decrypt_and_aggregate(
        self,
        encrypted_updates: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Decrypt and aggregate client updates (simulated)."""
        if len(encrypted_updates) < self.threshold:
            raise PrivacyError(f"Insufficient clients for secure aggregation. Need {self.threshold}, got {len(encrypted_updates)}")
        
        # In real implementation, this would:
        # 1. Use secret sharing to reconstruct individual updates
        # 2. Aggregate without revealing individual contributions
        # 3. Return only the aggregated result
        
        # Simplified: decrypt then aggregate
        decrypted_updates = []
        
        for update_data in encrypted_updates:
            client_id = update_data["client_id"]
            encrypted_params = update_data["parameters"]
            
            # Decrypt (reverse of encryption)
            if client_id in self.client_keys:
                key_seed = hash(self.client_keys[client_id]) % 2**32
                rng = np.random.RandomState(key_seed)
                
                decrypted = {}
                for name, param in encrypted_params.items():
                    mask = rng.normal(0, 0.1, param.shape)
                    decrypted[name] = param - mask  # Remove mask
                
                decrypted_updates.append({
                    "parameters": decrypted,
                    "num_samples": update_data.get("num_samples", 1)
                })
        
        # Aggregate decrypted parameters
        return self._aggregate_parameters(decrypted_updates)
    
    def _aggregate_parameters(
        self,
        updates: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """Aggregate decrypted parameters using weighted average."""
        if not updates:
            return {}
        
        total_samples = sum(update["num_samples"] for update in updates)
        param_names = list(updates[0]["parameters"].keys())
        aggregated = {}
        
        for name in param_names:
            weighted_sum = np.zeros_like(updates[0]["parameters"][name])
            
            for update in updates:
                weight = update["num_samples"] / total_samples
                weighted_sum += weight * update["parameters"][name]
            
            aggregated[name] = weighted_sum
        
        return aggregated


class PrivacyAccountant:
    """Track and manage privacy spending across federated learning."""
    
    def __init__(self, total_epsilon: float = 1.0, delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.epsilon_spent = 0.0
        self.privacy_ledger: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def spend_privacy_budget(
        self,
        epsilon: float,
        mechanism: str,
        purpose: str = "federated_round"
    ) -> bool:
        """Spend privacy budget and record the transaction."""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        if self.epsilon_spent + epsilon > self.total_epsilon:
            self.logger.warning(f"Privacy budget exhausted. Cannot spend {epsilon}, remaining: {self.remaining_epsilon()}")
            return False
        
        self.epsilon_spent += epsilon
        
        self.privacy_ledger.append({
            "epsilon": epsilon,
            "mechanism": mechanism,
            "purpose": purpose,
            "timestamp": np.datetime64('now'),
            "cumulative_spent": self.epsilon_spent
        })
        
        self.logger.info(f"Privacy budget spent: {epsilon:.4f} for {purpose} ({self.remaining_epsilon():.4f} remaining)")
        return True
    
    def remaining_epsilon(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.total_epsilon - self.epsilon_spent)
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.remaining_epsilon() <= 0
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report."""
        return {
            "total_epsilon": self.total_epsilon,
            "epsilon_spent": self.epsilon_spent,
            "remaining_epsilon": self.remaining_epsilon(),
            "delta": self.delta,
            "num_transactions": len(self.privacy_ledger),
            "budget_exhausted": self.is_budget_exhausted(),
            "ledger": self.privacy_ledger.copy()
        }
    
    def recommend_epsilon_per_round(self, num_rounds: int, safety_margin: float = 0.1) -> float:
        """Recommend epsilon per round for given number of rounds."""
        available_budget = self.remaining_epsilon() * (1 - safety_margin)
        if num_rounds <= 0:
            return 0.0
        return available_budget / num_rounds


def create_private_federated_setup(
    num_clients: int,
    total_epsilon: float = 1.0,
    delta: float = 1e-5,
    secure_aggregation: bool = True
) -> Dict[str, Any]:
    """Create a privacy-preserving federated learning setup."""
    
    # Privacy accountant
    accountant = PrivacyAccountant(total_epsilon, delta)
    
    # Differential privacy mechanism
    dp_mechanism = DifferentialPrivacy(
        epsilon=total_epsilon,
        delta=delta,
        mechanism="gaussian"
    )
    
    # Secure aggregation (if enabled)
    secure_agg = None
    if secure_aggregation:
        secure_agg = SecureAggregation(
            num_clients=num_clients,
            threshold=max(2, num_clients // 2)
        )
        client_keys = secure_agg.generate_client_keys()
    else:
        client_keys = {}
    
    return {
        "privacy_accountant": accountant,
        "dp_mechanism": dp_mechanism,
        "secure_aggregation": secure_agg,
        "client_keys": client_keys,
        "privacy_config": {
            "total_epsilon": total_epsilon,
            "delta": delta,
            "num_clients": num_clients,
            "secure_aggregation_enabled": secure_aggregation
        }
    }