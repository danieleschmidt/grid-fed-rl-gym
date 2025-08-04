"""Base classes for reinforcement learning algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import torch
    import torch.nn as nn
except ImportError:
    from . import torch_stub as torch
    nn = torch.nn
import numpy as np
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    loss: float
    q_loss: float = 0.0
    policy_loss: float = 0.0
    alpha_loss: float = 0.0
    mean_q_value: float = 0.0
    episode_return: float = 0.0


class BaseAlgorithm(ABC):
    """Abstract base class for all RL algorithms."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "auto",
        **kwargs
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.training_step = 0
        
    @abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action given state."""
        pass
        
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Update algorithm with batch of data."""
        pass
        
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to path."""
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from path."""
        pass
        
    def to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor."""
        return torch.FloatTensor(array).to(self.device)
        
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        return tensor.detach().cpu().numpy()


class OfflineRLAlgorithm(BaseAlgorithm):
    """Base class for offline RL algorithms."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        conservative_weight: float = 1.0,
        device: str = "auto",
        **kwargs
    ) -> None:
        super().__init__(state_dim, action_dim, device, **kwargs)
        self.conservative_weight = conservative_weight
        
    @abstractmethod
    def train_offline(
        self,
        dataset: Dict[str, np.ndarray],
        num_epochs: int,
        batch_size: int = 256,
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train algorithm on offline dataset."""
        pass
        
    def evaluate_dataset(self, dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate algorithm on dataset."""
        states = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        
        returns = []
        episode_returns = []
        current_return = 0
        
        for i in range(len(rewards)):
            current_return += rewards[i]
            
            # Check if episode ended (terminal or timeout)
            if i == len(rewards) - 1 or dataset.get("terminals", [False] * len(rewards))[i]:
                episode_returns.append(current_return)
                current_return = 0
                
        return {
            "mean_episode_return": np.mean(episode_returns) if episode_returns else 0.0,
            "std_episode_return": np.std(episode_returns) if episode_returns else 0.0,
            "num_episodes": len(episode_returns),
            "total_steps": len(states)
        }


class ActorCriticBase(BaseAlgorithm):
    """Base class for actor-critic algorithms."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        device: str = "auto",
        **kwargs
    ) -> None:
        super().__init__(state_dim, action_dim, device, **kwargs)
        self.hidden_dims = hidden_dims
        self.activation = self._get_activation(activation)
        
        # Will be initialized by subclasses
        self.actor: Optional[nn.Module] = None
        self.critic: Optional[nn.Module] = None
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "elu":
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: nn.Module,
        output_activation: Optional[nn.Module] = None
    ) -> nn.Module:
        """Build MLP network."""
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if i < len(dims) - 2:  # Not last layer
                layers.append(activation)
            elif output_activation is not None:
                layers.append(output_activation)
                
        return nn.Sequential(*layers)


class GridDataset:
    """Dataset class for grid control data."""
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        normalize: bool = True,
        device: str = "auto"
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.next_observations = next_observations
        self.terminals = terminals
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        if normalize:
            self._normalize_data()
            
        self.size = len(observations)
        
    def _normalize_data(self):
        """Normalize observations and actions."""
        # Normalize observations
        self.obs_mean = np.mean(self.observations, axis=0)
        self.obs_std = np.std(self.observations, axis=0) + 1e-6
        self.observations = (self.observations - self.obs_mean) / self.obs_std
        self.next_observations = (self.next_observations - self.obs_mean) / self.obs_std
        
        # Normalize actions
        self.action_mean = np.mean(self.actions, axis=0)
        self.action_std = np.std(self.actions, axis=0) + 1e-6
        self.actions = (self.actions - self.action_mean) / self.action_std
        
        # Normalize rewards
        self.reward_mean = np.mean(self.rewards)
        self.reward_std = np.std(self.rewards) + 1e-6
        self.rewards = (self.rewards - self.reward_mean) / self.reward_std
        
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from dataset."""
        indices = np.random.choice(self.size, batch_size, replace=True)
        
        batch = {
            "observations": torch.FloatTensor(self.observations[indices]).to(self.device),
            "actions": torch.FloatTensor(self.actions[indices]).to(self.device),
            "rewards": torch.FloatTensor(self.rewards[indices]).to(self.device),
            "next_observations": torch.FloatTensor(self.next_observations[indices]).to(self.device),
            "terminals": torch.FloatTensor(self.terminals[indices]).to(self.device)
        }
        
        return batch
        
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors."""
        return {
            "observations": torch.FloatTensor(self.observations).to(self.device),
            "actions": torch.FloatTensor(self.actions).to(self.device),
            "rewards": torch.FloatTensor(self.rewards).to(self.device),
            "next_observations": torch.FloatTensor(self.next_observations).to(self.device),
            "terminals": torch.FloatTensor(self.terminals).to(self.device)
        }
        
    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action."""
        if hasattr(self, 'action_mean'):
            action_mean = torch.FloatTensor(self.action_mean).to(action.device)
            action_std = torch.FloatTensor(self.action_std).to(action.device)
            return action * action_std + action_mean
        return action
        
    def denormalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Denormalize observation."""
        if hasattr(self, 'obs_mean'):
            obs_mean = torch.FloatTensor(self.obs_mean).to(obs.device)
            obs_std = torch.FloatTensor(self.obs_std).to(obs.device)
            return obs * obs_std + obs_mean
        return obs


def collect_random_data(env, num_steps: int) -> Dict[str, np.ndarray]:
    """Collect random data from environment."""
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_observations.append(next_obs)
        terminals.append(terminated or truncated)
        
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "next_observations": np.array(next_observations),
        "terminals": np.array(terminals)
    }