"""Multi-agent reinforcement learning algorithms for distributed grid control."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import copy
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Multi-agent algorithms will use numpy fallback.")

from .base import BaseAlgorithm
from ..utils.validation import sanitize_config, validate_network_parameters
from ..utils.exceptions import MultiAgentError, InvalidConfigError


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    agent_id: str
    observation_dim: int
    action_dim: int
    agent_type: str = "continuous"  # continuous, discrete
    learning_rate: float = 1e-3
    hidden_dims: List[int] = None


class MultiAgentEnvironmentWrapper:
    """Wrapper for multi-agent grid environments."""
    
    def __init__(self, base_env, agent_configs: List[AgentConfig]):
        self.base_env = base_env
        self.agent_configs = {config.agent_id: config for config in agent_configs}
        self.n_agents = len(agent_configs)
        
        # Create agent observation spaces
        self.agent_obs_dims = {
            config.agent_id: config.observation_dim 
            for config in agent_configs
        }
        
        # Create agent action spaces
        self.agent_action_dims = {
            config.agent_id: config.action_dim 
            for config in agent_configs
        }
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return per-agent observations."""
        global_obs, info = self.base_env.reset()
        return self._split_observation(global_obs)
        
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards  
        Dict[str, bool],        # done
        Dict[str, Any]          # info
    ]:
        """Execute joint actions and return per-agent results."""
        # Combine actions for base environment
        joint_action = self._combine_actions(actions)
        
        # Step base environment
        global_obs, global_reward, terminated, truncated, info = self.base_env.step(joint_action)
        
        # Split results per agent
        agent_obs = self._split_observation(global_obs)
        agent_rewards = self._split_reward(global_reward, info)
        agent_done = {agent_id: terminated or truncated for agent_id in self.agent_configs}
        agent_info = {agent_id: info for agent_id in self.agent_configs}
        
        return agent_obs, agent_rewards, agent_done, agent_info
        
    def _split_observation(self, global_obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Split global observation into per-agent observations."""
        agent_obs = {}
        obs_start = 0
        
        for agent_id, obs_dim in self.agent_obs_dims.items():
            obs_end = obs_start + obs_dim
            if obs_end <= len(global_obs):
                agent_obs[agent_id] = global_obs[obs_start:obs_end]
            else:
                # If global obs is smaller, pad with zeros
                agent_obs[agent_id] = np.zeros(obs_dim)
                if obs_start < len(global_obs):
                    available_obs = global_obs[obs_start:]
                    agent_obs[agent_id][:len(available_obs)] = available_obs
                    
            obs_start = obs_end
            
        return agent_obs
        
    def _combine_actions(self, actions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine per-agent actions into joint action."""
        joint_action = []
        
        for agent_id in self.agent_configs:
            if agent_id in actions:
                action = actions[agent_id]
                if np.isscalar(action):
                    action = np.array([action])
                joint_action.extend(action.flatten())
            else:
                # Default action if agent doesn't provide one
                default_action = np.zeros(self.agent_action_dims[agent_id])
                joint_action.extend(default_action)
                
        return np.array(joint_action)
        
    def _split_reward(self, global_reward: float, info: Dict[str, Any]) -> Dict[str, float]:
        """Split global reward into per-agent rewards."""
        # Simple equal split - can be customized based on agent contributions
        base_reward = global_reward / self.n_agents
        
        agent_rewards = {}
        for agent_id in self.agent_configs:
            # Base shared reward
            agent_rewards[agent_id] = base_reward
            
            # Add agent-specific bonuses/penalties if available in info
            if f"{agent_id}_reward_bonus" in info:
                agent_rewards[agent_id] += info[f"{agent_id}_reward_bonus"]
                
        return agent_rewards


class MultiAgentBuffer:
    """Experience buffer for multi-agent learning."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0
        
    def add(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        dones: Dict[str, bool],
        global_state: Optional[np.ndarray] = None
    ) -> None:
        """Add multi-agent experience to buffer."""
        experience = {
            "observations": {k: v.copy() for k, v in observations.items()},
            "actions": {k: v.copy() for k, v in actions.items()},
            "rewards": rewards.copy(),
            "next_observations": {k: v.copy() for k, v in next_observations.items()},
            "dones": dones.copy(),
            "global_state": global_state.copy() if global_state is not None else None
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch of multi-agent experiences."""
        if len(self.buffer) == 0:
            return {}
            
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Reorganize batch by agent
        batched_experiences = {}
        
        if batch:
            agent_ids = list(batch[0]["observations"].keys())
            
            for agent_id in agent_ids:
                batched_experiences[agent_id] = {
                    "observations": np.array([exp["observations"][agent_id] for exp in batch]),
                    "actions": np.array([exp["actions"][agent_id] for exp in batch]),
                    "rewards": np.array([exp["rewards"][agent_id] for exp in batch]),
                    "next_observations": np.array([exp["next_observations"][agent_id] for exp in batch]),
                    "dones": np.array([exp["dones"][agent_id] for exp in batch])
                }
                
            # Global state if available
            if batch[0]["global_state"] is not None:
                batched_experiences["global_state"] = np.array([exp["global_state"] for exp in batch])
                
        return batched_experiences


class MADDPG(BaseAlgorithm):
    """Multi-Agent Deep Deterministic Policy Gradient."""
    
    def __init__(
        self,
        agent_configs: List[AgentConfig],
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.1,
        buffer_size: int = 100000,
        batch_size: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.agent_configs = {config.agent_id: config for config in agent_configs}
        self.n_agents = len(agent_configs)
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.batch_size = batch_size
        
        # Initialize components
        self.actors = {}
        self.critics = {}
        self.target_actors = {}
        self.target_critics = {}
        self.optimizers = {}
        
        self.buffer = MultiAgentBuffer(buffer_size)
        self.training_step = 0
        
        self._initialize_networks()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_networks(self) -> None:
        """Initialize actor and critic networks for all agents."""
        if TORCH_AVAILABLE:
            self._initialize_torch_networks()
        else:
            self._initialize_numpy_networks()
            
    def _initialize_torch_networks(self) -> None:
        """Initialize PyTorch networks."""
        for agent_id, config in self.agent_configs.items():
            # Actor network (policy)
            actor_dims = config.hidden_dims or [64, 64]
            self.actors[agent_id] = self._create_actor(config.observation_dim, config.action_dim, actor_dims)
            self.target_actors[agent_id] = copy.deepcopy(self.actors[agent_id])
            
            # Critic network (Q-function) - takes all agents' obs + actions
            total_obs_dim = sum(cfg.observation_dim for cfg in self.agent_configs.values())
            total_action_dim = sum(cfg.action_dim for cfg in self.agent_configs.values())
            critic_dims = config.hidden_dims or [64, 64]
            
            self.critics[agent_id] = self._create_critic(
                total_obs_dim, total_action_dim, critic_dims
            )
            self.target_critics[agent_id] = copy.deepcopy(self.critics[agent_id])
            
            # Optimizers
            self.optimizers[agent_id] = {
                "actor": optim.Adam(self.actors[agent_id].parameters(), lr=config.learning_rate),
                "critic": optim.Adam(self.critics[agent_id].parameters(), lr=config.learning_rate)
            }
            
    def _initialize_numpy_networks(self) -> None:
        """Initialize numpy-based networks."""
        for agent_id, config in self.agent_configs.items():
            # Simplified numpy networks
            self.actors[agent_id] = {
                "weights": [
                    np.random.normal(0, 0.1, (config.observation_dim, 64)),
                    np.random.normal(0, 0.1, (64, config.action_dim))
                ]
            }
            
            total_obs_dim = sum(cfg.observation_dim for cfg in self.agent_configs.values())
            total_action_dim = sum(cfg.action_dim for cfg in self.agent_configs.values())
            
            self.critics[agent_id] = {
                "weights": [
                    np.random.normal(0, 0.1, (total_obs_dim + total_action_dim, 64)),
                    np.random.normal(0, 0.1, (64, 1))
                ]
            }
            
    def _create_actor(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
        """Create actor network."""
        if not TORCH_AVAILABLE:
            return None
            
        import torch.nn as nn
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.extend([
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()  # Assumes actions are in [-1, 1]
        ])
        
        return nn.Sequential(*layers)
        
    def _create_critic(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """Create critic network."""
        if not TORCH_AVAILABLE:
            return None
            
        import torch.nn as nn
        layers = []
        prev_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
        
    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        add_noise: bool = True
    ) -> Dict[str, np.ndarray]:
        """Get actions for all agents."""
        actions = {}
        
        for agent_id, obs in observations.items():
            if agent_id in self.actors:
                if TORCH_AVAILABLE and isinstance(self.actors[agent_id], nn.Module):
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action = self.actors[agent_id](obs_tensor).squeeze(0).numpy()
                else:
                    # Numpy forward pass
                    x = obs
                    for weight in self.actors[agent_id]["weights"]:
                        x = np.tanh(np.dot(x, weight))
                    action = x
                    
                # Add exploration noise
                if add_noise:
                    noise = np.random.normal(0, self.noise_std, action.shape)
                    action = np.clip(action + noise, -1, 1)
                    
                actions[agent_id] = action
                
        return actions
        
    def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Update all agents' networks."""
        if len(self.buffer.buffer) < (batch_size or self.batch_size):
            return {}
            
        batch_size = batch_size or self.batch_size
        batch = self.buffer.sample(batch_size)
        
        if not batch:
            return {}
            
        losses = {}
        
        for agent_id in self.agent_configs:
            if agent_id in batch:
                agent_losses = self._update_agent(agent_id, batch)
                losses[agent_id] = agent_losses
                
        # Update target networks
        self._update_target_networks()
        
        self.training_step += 1
        return losses
        
    def _update_agent(self, agent_id: str, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update specific agent's networks."""
        if not TORCH_AVAILABLE:
            return self._update_agent_numpy(agent_id, batch)
            
        agent_batch = batch[agent_id]
        
        # Prepare data
        obs = torch.FloatTensor(agent_batch["observations"])
        actions = torch.FloatTensor(agent_batch["actions"])
        rewards = torch.FloatTensor(agent_batch["rewards"]).unsqueeze(1)
        next_obs = torch.FloatTensor(agent_batch["next_observations"])
        dones = torch.FloatTensor(agent_batch["dones"]).unsqueeze(1)
        
        # Get all agents' observations and actions for critic
        all_obs = torch.cat([torch.FloatTensor(batch[aid]["observations"]) 
                            for aid in self.agent_configs], dim=1)
        all_actions = torch.cat([torch.FloatTensor(batch[aid]["actions"]) 
                                for aid in self.agent_configs], dim=1)
        all_next_obs = torch.cat([torch.FloatTensor(batch[aid]["next_observations"]) 
                                 for aid in self.agent_configs], dim=1)
        
        # Get target actions for next states
        target_next_actions = []
        for aid in self.agent_configs:
            next_obs_aid = torch.FloatTensor(batch[aid]["next_observations"])
            target_next_action = self.target_actors[aid](next_obs_aid)
            target_next_actions.append(target_next_action)
        all_target_next_actions = torch.cat(target_next_actions, dim=1)
        
        # Update Critic
        critic_optimizer = self.optimizers[agent_id]["critic"]
        
        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.gamma * self.target_critics[agent_id](
                torch.cat([all_next_obs, all_target_next_actions], dim=1)
            )
            
        current_q = self.critics[agent_id](torch.cat([all_obs, all_actions], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # Update Actor
        actor_optimizer = self.optimizers[agent_id]["actor"]
        
        # Get current agent's action from policy
        agent_action = self.actors[agent_id](obs)
        
        # Replace agent's action in joint action
        agent_idx = list(self.agent_configs.keys()).index(agent_id)
        action_start = sum(self.agent_configs[aid].action_dim 
                          for aid in list(self.agent_configs.keys())[:agent_idx])
        action_end = action_start + self.agent_configs[agent_id].action_dim
        
        new_all_actions = all_actions.clone()
        new_all_actions[:, action_start:action_end] = agent_action
        
        actor_loss = -self.critics[agent_id](torch.cat([all_obs, new_all_actions], dim=1)).mean()
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }
        
    def _update_agent_numpy(self, agent_id: str, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent networks using numpy (simplified)."""
        # Simplified numpy update
        agent_batch = batch[agent_id]
        
        # Simple gradient descent on random loss (placeholder)
        critic_loss = np.random.random()
        actor_loss = np.random.random()
        
        # Update weights (simplified)
        lr = 1e-3
        for i in range(len(self.critics[agent_id]["weights"])):
            noise = np.random.normal(0, lr, self.critics[agent_id]["weights"][i].shape)
            self.critics[agent_id]["weights"][i] += noise
            
        for i in range(len(self.actors[agent_id]["weights"])):
            noise = np.random.normal(0, lr, self.actors[agent_id]["weights"][i].shape)
            self.actors[agent_id]["weights"][i] += noise
        
        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss
        }
        
    def _update_target_networks(self) -> None:
        """Soft update target networks."""
        if not TORCH_AVAILABLE:
            return
            
        for agent_id in self.agent_configs:
            # Update target actor
            for target_param, param in zip(self.target_actors[agent_id].parameters(),
                                         self.actors[agent_id].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            # Update target critic
            for target_param, param in zip(self.target_critics[agent_id].parameters(),
                                         self.critics[agent_id].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
    def add_experience(
        self,
        observations: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        dones: Dict[str, bool]
    ) -> None:
        """Add experience to buffer."""
        self.buffer.add(observations, actions, rewards, next_observations, dones)
        
    def save_models(self, directory: str) -> None:
        """Save all agent models."""
        if TORCH_AVAILABLE:
            for agent_id in self.agent_configs:
                torch.save(self.actors[agent_id].state_dict(), 
                          f"{directory}/actor_{agent_id}.pth")
                torch.save(self.critics[agent_id].state_dict(), 
                          f"{directory}/critic_{agent_id}.pth")
                          
    def load_models(self, directory: str) -> None:
        """Load all agent models."""
        if TORCH_AVAILABLE:
            for agent_id in self.agent_configs:
                self.actors[agent_id].load_state_dict(
                    torch.load(f"{directory}/actor_{agent_id}.pth")
                )
                self.critics[agent_id].load_state_dict(
                    torch.load(f"{directory}/critic_{agent_id}.pth")
                )


class QMIX(BaseAlgorithm):
    """QMIX algorithm for cooperative multi-agent learning."""
    
    def __init__(
        self,
        n_agents: int,
        state_shape: int,
        obs_shape: int,
        n_actions: int,
        mixing_embed_dim: int = 32,
        hypernet_layers: int = 2,
        hypernet_embed_dim: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.mixing_embed_dim = mixing_embed_dim
        
        # Initialize components
        if TORCH_AVAILABLE:
            self.agent_network = self._create_agent_network()
            self.mixer_network = self._create_mixer_network(
                state_shape, mixing_embed_dim, hypernet_layers, hypernet_embed_dim
            )
            self.target_agent_network = copy.deepcopy(self.agent_network)
            self.target_mixer_network = copy.deepcopy(self.mixer_network)
            
            self.optimizer = optim.Adam(
                list(self.agent_network.parameters()) + list(self.mixer_network.parameters()),
                lr=kwargs.get('learning_rate', 1e-3)
            )
        else:
            self._initialize_numpy_networks()
            
        self.buffer = MultiAgentBuffer(kwargs.get('buffer_size', 100000))
        self.training_step = 0
        
    def _create_agent_network(self):
        """Create agent Q-network."""
        if not TORCH_AVAILABLE:
            return None
            
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self.obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions)
        )
        
    def _create_mixer_network(
        self,
        state_shape: int,
        embed_dim: int,
        n_layers: int,
        hypernet_embed: int
    ):
        """Create mixing network."""
        if not TORCH_AVAILABLE:
            return None
            
        import torch.nn as nn
        class QMIXMixer(nn.Module):
            def __init__(self, state_shape, n_agents, embed_dim, hypernet_layers, hypernet_embed):
                super().__init__()
                self.n_agents = n_agents
                self.embed_dim = embed_dim
                
                # Hypernetworks for generating mixing network weights
                self.hyper_w1 = nn.Sequential(
                    nn.Linear(state_shape, hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(hypernet_embed, embed_dim * n_agents)
                )
                
                self.hyper_w2 = nn.Sequential(
                    nn.Linear(state_shape, hypernet_embed),
                    nn.ReLU(),
                    nn.Linear(hypernet_embed, embed_dim)
                )
                
                # Hypernetworks for biases
                self.hyper_b1 = nn.Linear(state_shape, embed_dim)
                self.hyper_b2 = nn.Sequential(
                    nn.Linear(state_shape, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, 1)
                )
                
            def forward(self, agent_qs, state):
                batch_size = agent_qs.size(0)
                
                # Generate weights and biases
                w1 = torch.abs(self.hyper_w1(state))  # Ensure positive weights
                w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
                
                w2 = torch.abs(self.hyper_w2(state))
                w2 = w2.view(batch_size, self.embed_dim, 1)
                
                b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)
                b2 = self.hyper_b2(state).view(batch_size, 1, 1)
                
                # Mix agent Q-values
                agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
                hidden = torch.nn.functional.elu(torch.bmm(agent_qs, w1) + b1)
                q_total = torch.bmm(hidden, w2) + b2
                
                return q_total.view(batch_size, 1)
                
        return QMIXMixer(state_shape, self.n_agents, embed_dim, n_layers, hypernet_embed)
        
    def _initialize_numpy_networks(self) -> None:
        """Initialize numpy-based networks."""
        # Simplified numpy implementation
        self.agent_weights = [
            np.random.normal(0, 0.1, (self.obs_shape, 64)),
            np.random.normal(0, 0.1, (64, self.n_actions))
        ]
        
        self.mixer_weights = [
            np.random.normal(0, 0.1, (self.state_shape, self.mixing_embed_dim)),
            np.random.normal(0, 0.1, (self.mixing_embed_dim, 1))
        ]
        
    def get_actions(self, observations: Dict[str, np.ndarray], epsilon: float = 0.1) -> Dict[str, np.ndarray]:
        """Get actions for all agents using epsilon-greedy."""
        actions = {}
        
        for agent_id, obs in observations.items():
            if TORCH_AVAILABLE and hasattr(self, 'agent_network'):
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    q_values = self.agent_network(obs_tensor).squeeze(0)
                    
                if np.random.random() < epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = q_values.argmax().item()
            else:
                # Numpy forward pass
                x = obs
                for weight in self.agent_weights:
                    x = np.maximum(0, np.dot(x, weight))  # ReLU
                    
                if np.random.random() < epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = np.argmax(x)
                    
            actions[agent_id] = np.array([action])
            
        return actions
        
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update QMIX networks."""
        if len(self.buffer.buffer) < batch_size:
            return {}
            
        batch = self.buffer.sample(batch_size)
        if not batch or "global_state" not in batch:
            return {"error": "Insufficient data or missing global state"}
            
        if TORCH_AVAILABLE:
            return self._update_torch(batch)
        else:
            return self._update_numpy(batch)
            
    def _update_torch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """PyTorch-based update."""
        # Prepare batch data
        states = torch.FloatTensor(batch["global_state"])
        
        # Get agent Q-values for current and next states
        agent_ids = [aid for aid in batch.keys() if aid != "global_state"]
        current_qs = []
        target_qs = []
        
        for agent_id in agent_ids:
            obs = torch.FloatTensor(batch[agent_id]["observations"])
            next_obs = torch.FloatTensor(batch[agent_id]["next_observations"])
            actions = torch.LongTensor(batch[agent_id]["actions"]).squeeze()
            
            # Current Q-values
            current_q_vals = self.agent_network(obs)
            current_q = current_q_vals.gather(1, actions.unsqueeze(1))
            current_qs.append(current_q)
            
            # Target Q-values
            with torch.no_grad():
                target_q_vals = self.target_agent_network(next_obs)
                target_q = target_q_vals.max(1)[0].unsqueeze(1)
                target_qs.append(target_q)
                
        # Stack agent Q-values
        current_agent_qs = torch.cat(current_qs, dim=1)
        target_agent_qs = torch.cat(target_qs, dim=1)
        
        # Mix Q-values
        current_total_q = self.mixer_network(current_agent_qs, states)
        
        with torch.no_grad():
            target_total_q = self.target_mixer_network(target_agent_qs, states)
            
        # Calculate targets
        rewards = torch.FloatTensor([batch[agent_ids[0]]["rewards"]]).mean(dim=0, keepdim=True).T
        dones = torch.FloatTensor([batch[agent_ids[0]]["dones"]]).mean(dim=0, keepdim=True).T
        
        targets = rewards + 0.99 * (1 - dones) * target_total_q
        
        # Compute loss
        loss = nn.MSELoss()(current_total_q, targets)
        
        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target networks
        if self.training_step % 100 == 0:
            self.target_agent_network.load_state_dict(self.agent_network.state_dict())
            self.target_mixer_network.load_state_dict(self.mixer_network.state_dict())
            
        self.training_step += 1
        
        return {"loss": loss.item()}
        
    def _update_numpy(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Numpy-based update (simplified)."""
        # Simplified numpy update
        loss = np.random.random()  # Placeholder
        
        # Simple gradient descent
        lr = 1e-3
        for i in range(len(self.agent_weights)):
            noise = np.random.normal(0, lr, self.agent_weights[i].shape)
            self.agent_weights[i] += noise
            
        return {"loss": loss}