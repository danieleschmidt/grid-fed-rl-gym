"""Graph Neural Network-based Federated Reinforcement Learning for Power Systems.

This module implements GNN-based federated RL algorithms that leverage the inherent
graph structure of power systems for scalable learning and improved generalization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch_geometric.transforms as T
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from .base import OfflineRLAlgorithm, ActorCriticBase, TrainingMetrics
from ..federated.core import FederatedClient, ClientUpdate
from ..utils.validation import validate_constraints


@dataclass
class GraphTopology:
    """Graph topology representation for power systems."""
    num_nodes: int
    edge_index: torch.Tensor  # [2, num_edges]
    edge_attr: Optional[torch.Tensor] = None  # Edge features (impedances, capacities)
    node_types: Optional[torch.Tensor] = None  # Bus types (slack, PV, PQ)
    adjacency_matrix: Optional[torch.Tensor] = None


class PowerSystemGraph:
    """Utility class for creating power system graphs."""
    
    @staticmethod
    def create_ieee_bus_graph(num_buses: int = 13) -> GraphTopology:
        """Create graph for IEEE test systems."""
        if num_buses == 13:
            # IEEE 13-bus system topology
            edges = [
                (0, 1), (1, 2), (1, 4), (1, 6),  # From bus 650
                (4, 5), (6, 7), (6, 8), (2, 3),  # Connections
                (8, 9), (8, 10), (6, 11), (11, 12)  # Final connections
            ]
        elif num_buses == 34:
            # IEEE 34-bus system (simplified)
            edges = [(i, i+1) for i in range(num_buses-1)]  # Linear topology
            # Add some radial branches
            radial_branches = [(0, 10), (5, 15), (10, 20), (15, 25)]
            edges.extend(radial_branches)
        elif num_buses == 123:
            # IEEE 123-bus system (highly simplified)
            edges = [(i, i+1) for i in range(num_buses-1)]
            # Add multiple radial branches
            for i in range(0, num_buses, 10):
                for j in range(1, min(6, num_buses-i)):
                    if i+j < num_buses:
                        edges.append((i, i+j))
        else:
            # Generic radial system
            edges = [(i, i+1) for i in range(num_buses-1)]
        
        # Convert to tensor format
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Add reverse edges for undirected graph
        reverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, reverse_edges], dim=1)
        
        # Node types: 0=slack, 1=PV, 2=PQ
        node_types = torch.zeros(num_buses, dtype=torch.long)
        node_types[0] = 0  # First bus is slack
        node_types[1:min(3, num_buses)] = 1  # Next few are PV
        node_types[min(3, num_buses):] = 2  # Rest are PQ
        
        return GraphTopology(
            num_nodes=num_buses,
            edge_index=edge_index,
            node_types=node_types
        )
    
    @staticmethod
    def create_transmission_graph(num_nodes: int = 50) -> GraphTopology:
        """Create larger transmission system graph."""
        edges = []
        
        # Create a small-world network structure
        # Ring topology
        for i in range(num_nodes):
            edges.append((i, (i + 1) % num_nodes))
        
        # Add random long-distance connections
        np.random.seed(42)  # For reproducibility
        num_long_connections = num_nodes // 5
        
        for _ in range(num_long_connections):
            i, j = np.random.choice(num_nodes, 2, replace=False)
            if (i, j) not in edges and (j, i) not in edges:
                edges.append((i, j))
        
        # Add radial connections for robustness
        for i in range(0, num_nodes, 10):
            if i + 5 < num_nodes:
                edges.append((i, i + 5))
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Make undirected
        reverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, reverse_edges], dim=1)
        
        # Node types for transmission (more PV buses)
        node_types = torch.zeros(num_nodes, dtype=torch.long)
        node_types[0] = 0  # Slack bus
        node_types[1:num_nodes//3] = 1  # PV buses (generators)
        node_types[num_nodes//3:] = 2  # PQ buses (loads)
        
        return GraphTopology(
            num_nodes=num_nodes,
            edge_index=edge_index,
            node_types=node_types
        )


class GraphConvolutionalLayer(nn.Module):
    """Enhanced graph convolutional layer for power systems."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        normalize: bool = True,
        activation: str = "relu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        # Graph convolution
        if edge_dim is not None:
            # Use edge features
            self.conv = GraphConv(in_channels, out_channels, bias=bias)
            self.edge_encoder = nn.Linear(edge_dim, out_channels)
        else:
            self.conv = GCNConv(in_channels, out_channels, bias=bias, normalize=normalize)
            self.edge_encoder = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Residual connection preparation
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through graph convolutional layer."""
        residual = x
        
        # Graph convolution
        if self.edge_encoder is not None and edge_attr is not None:
            edge_features = self.edge_encoder(edge_attr)
            x = self.conv(x, edge_index, edge_features)
        else:
            x = self.conv(x, edge_index)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Activation
        x = self.activation(x)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        x = x + residual
        
        # Dropout
        x = self.dropout(x)
        
        return x


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for power system modeling."""
    
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: Optional[int],
        hidden_dims: List[int],
        output_dim: int,
        num_node_types: int = 3,
        pooling: str = "mean",
        residual_connections: bool = True
    ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_dim = output_dim
        self.pooling = pooling
        
        # Node type embedding
        self.node_type_embedding = nn.Embedding(num_node_types, node_input_dim // 4)
        
        # Input projection
        embedding_dim = node_input_dim + node_input_dim // 4
        if hidden_dims:
            self.input_projection = nn.Linear(embedding_dim, hidden_dims[0])
            current_dim = hidden_dims[0]
        else:
            self.input_projection = nn.Linear(embedding_dim, output_dim)
            current_dim = output_dim
            hidden_dims = []
        
        # Graph convolutional layers
        self.gnn_layers = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            layer = GraphConvolutionalLayer(
                in_channels=current_dim,
                out_channels=hidden_dim,
                edge_dim=edge_input_dim,
                normalize=True
            )
            self.gnn_layers.append(layer)
            current_dim = hidden_dim
        
        # Output projection
        if hidden_dims:
            self.output_projection = nn.Linear(current_dim, output_dim)
        else:
            self.output_projection = None
        
        # Global pooling for graph-level representations
        if pooling == "mean":
            self.global_pool = global_mean_pool
        elif pooling == "max":
            self.global_pool = global_max_pool
        elif pooling == "add":
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_types: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_node_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through GNN."""
        
        # Add node type embeddings
        if node_types is not None:
            type_embeddings = self.node_type_embedding(node_types)
            x = torch.cat([x, type_embeddings], dim=-1)
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # Graph convolutional layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr, batch)
        
        # Output projection
        if self.output_projection is not None:
            node_features = self.output_projection(x)
        else:
            node_features = x
        
        # Global pooling for graph-level representation
        if batch is not None:
            graph_features = self.global_pool(node_features, batch)
        else:
            graph_features = torch.mean(node_features, dim=0, keepdim=True)
        
        if return_node_features:
            return graph_features, node_features
        else:
            return graph_features


class GraphActor(nn.Module):
    """Graph-based actor network for power system control."""
    
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: Optional[int],
        action_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        graph_topology: Optional[GraphTopology] = None
    ):
        super().__init__()
        self.action_dim = action_dim
        self.graph_topology = graph_topology
        
        # Graph neural network for feature extraction
        self.gnn = GraphNeuralNetwork(
            node_input_dim=node_feature_dim,
            edge_input_dim=edge_feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else 32,
            pooling="mean"
        )
        
        # Policy head for actions
        final_dim = hidden_dims[-1] if hidden_dims else 32
        self.policy_mean = nn.Sequential(
            nn.Linear(final_dim, action_dim),
            nn.Tanh()  # Bounded actions
        )
        
        self.policy_logstd = nn.Sequential(
            nn.Linear(final_dim, action_dim),
        )
        
        # Node-level action prediction (for distributed control)
        self.node_action_head = nn.Linear(hidden_dims[-1] if hidden_dims else 32, 1)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_types: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through graph actor."""
        
        # Extract features using GNN
        graph_features, node_embeddings = self.gnn(
            node_features, edge_index, edge_attr, node_types, batch, return_node_features=True
        )
        
        # Global policy
        policy_mean = self.policy_mean(graph_features)
        policy_logstd = self.policy_logstd(graph_features)
        policy_logstd = torch.clamp(policy_logstd, -20, 2)
        
        # Node-level actions (for distributed control signals)
        node_actions = self.node_action_head(node_embeddings)
        
        return policy_mean, policy_logstd, node_actions


class GraphCritic(nn.Module):
    """Graph-based critic network for value estimation."""
    
    def __init__(
        self,
        node_feature_dim: int,
        action_dim: int,
        edge_feature_dim: Optional[int] = None,
        hidden_dims: List[int] = [128, 64, 32]
    ):
        super().__init__()
        
        # Separate GNNs for state and state-action encoding
        self.state_gnn = GraphNeuralNetwork(
            node_input_dim=node_feature_dim,
            edge_input_dim=edge_feature_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            pooling="mean"
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        actions: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        node_types: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through graph critic."""
        
        # Encode state using GNN
        state_features = self.state_gnn(
            node_features, edge_index, edge_attr, node_types, batch
        )
        
        # Encode actions
        action_features = self.action_encoder(actions)
        
        # Combine state and action features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        # Predict value
        value = self.value_head(combined_features)
        
        return value


class GNFRL(OfflineRLAlgorithm, ActorCriticBase):
    """Graph Neural Network-based Federated Reinforcement Learning."""
    
    def __init__(
        self,
        graph_topology: GraphTopology,
        node_feature_dim: int,
        action_dim: int,
        edge_feature_dim: Optional[int] = None,
        hidden_dims: List[int] = [128, 64, 32],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "auto",
        **kwargs
    ):
        # Initialize base classes with derived dimensions
        state_dim = graph_topology.num_nodes * node_feature_dim  # Flattened for base class
        super().__init__(state_dim, action_dim, device=device, **kwargs)
        ActorCriticBase.__init__(self, state_dim, action_dim, hidden_dims, device=device, **kwargs)
        
        self.graph_topology = graph_topology
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        
        # Move graph topology to device
        self.graph_topology.edge_index = self.graph_topology.edge_index.to(self.device)
        if self.graph_topology.node_types is not None:
            self.graph_topology.node_types = self.graph_topology.node_types.to(self.device)
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)
        
        # Graph-specific metrics
        self.node_importance_history = []
        self.graph_connectivity_metrics = []
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """Build graph-based networks."""
        
        # Graph-based actor
        self.actor = GraphActor(
            node_feature_dim=self.node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            graph_topology=self.graph_topology
        ).to(self.device)
        
        # Twin graph critics
        self.critic1 = GraphCritic(
            node_feature_dim=self.node_feature_dim,
            action_dim=self.action_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        self.critic2 = GraphCritic(
            node_feature_dim=self.node_feature_dim,
            action_dim=self.action_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Target critics
        self.critic1_target = GraphCritic(
            node_feature_dim=self.node_feature_dim,
            action_dim=self.action_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        self.critic2_target = GraphCritic(
            node_feature_dim=self.node_feature_dim,
            action_dim=self.action_dim,
            edge_feature_dim=self.edge_feature_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)
        
        # Copy parameters to targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
    
    def _state_to_graph(self, state: torch.Tensor) -> torch.Tensor:
        """Convert flattened state to node features."""
        batch_size = state.shape[0]
        # Reshape state to [batch_size * num_nodes, node_feature_dim]
        node_features = state.view(batch_size * self.graph_topology.num_nodes, self.node_feature_dim)
        return node_features
    
    def _create_batch_graph(self, batch_size: int) -> torch.Tensor:
        """Create batch tensor for multiple graphs."""
        batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.graph_topology.num_nodes)
        return batch
    
    def _expand_edge_index(self, batch_size: int) -> torch.Tensor:
        """Expand edge index for batch processing."""
        edge_index = self.graph_topology.edge_index
        
        if batch_size == 1:
            return edge_index
        
        # Create batch edge indices
        batch_edge_indices = []
        num_nodes = self.graph_topology.num_nodes
        
        for i in range(batch_size):
            offset = i * num_nodes
            batch_edge_index = edge_index + offset
            batch_edge_indices.append(batch_edge_index)
        
        return torch.cat(batch_edge_indices, dim=1)
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action using graph neural networks."""
        state_tensor = self.to_tensor(state).unsqueeze(0)
        batch_size = 1
        
        # Convert to graph format
        node_features = self._state_to_graph(state_tensor)
        edge_index = self._expand_edge_index(batch_size)
        batch = self._create_batch_graph(batch_size)
        
        with torch.no_grad():
            policy_mean, policy_logstd, node_actions = self.actor(
                node_features,
                edge_index,
                node_types=self.graph_topology.node_types.repeat(batch_size) if self.graph_topology.node_types is not None else None,
                batch=batch
            )
            
            if eval_mode:
                action = policy_mean
            else:
                std = torch.exp(policy_logstd)
                normal = torch.distributions.Normal(policy_mean, std)
                action = normal.sample()
                action = torch.tanh(action)  # Bounded actions
        
        return self.to_numpy(action.squeeze(0))
    
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Graph-aware update step."""
        self.training_step += 1
        
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        terminals = batch["terminals"]
        
        batch_size = states.shape[0]
        
        # Convert to graph format
        node_features = self._state_to_graph(states)
        next_node_features = self._state_to_graph(next_states)
        edge_index = self._expand_edge_index(batch_size)
        graph_batch = self._create_batch_graph(batch_size)
        
        node_types = None
        if self.graph_topology.node_types is not None:
            node_types = self.graph_topology.node_types.repeat(batch_size)
        
        # Update critics
        critic_loss = self._update_graph_critics(
            node_features, next_node_features, actions, rewards, terminals,
            edge_index, graph_batch, node_types
        )
        
        # Update actor
        actor_loss = self._update_graph_actor(
            node_features, edge_index, graph_batch, node_types
        )
        
        # Update target networks
        self._soft_update_targets()
        
        # Track graph-specific metrics
        self._track_graph_metrics(node_features, edge_index, graph_batch)
        
        total_loss = critic_loss + actor_loss
        
        return TrainingMetrics(
            loss=total_loss,
            q_loss=critic_loss,
            policy_loss=actor_loss,
            mean_q_value=torch.mean(rewards).item()
        )
    
    def _update_graph_critics(
        self,
        node_features: torch.Tensor,
        next_node_features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_types: Optional[torch.Tensor]
    ) -> float:
        """Update graph-based critics."""
        
        with torch.no_grad():
            # Get next actions
            next_policy_mean, next_policy_logstd, _ = self.actor(
                next_node_features, edge_index, node_types=node_types, batch=batch
            )
            
            # Sample next actions
            next_std = torch.exp(next_policy_logstd)
            next_normal = torch.distributions.Normal(next_policy_mean, next_std)
            next_actions = torch.tanh(next_normal.sample())
            
            # Target Q-values
            target_q1 = self.critic1_target(
                next_node_features, edge_index, next_actions, node_types=node_types, batch=batch
            )
            target_q2 = self.critic2_target(
                next_node_features, edge_index, next_actions, node_types=node_types, batch=batch
            )
            
            target_q = torch.min(target_q1, target_q2)
            target_values = rewards.unsqueeze(-1) + self.gamma * (1 - terminals.unsqueeze(-1)) * target_q
        
        # Current Q-values
        current_q1 = self.critic1(node_features, edge_index, actions, node_types=node_types, batch=batch)
        current_q2 = self.critic2(node_features, edge_index, actions, node_types=node_types, batch=batch)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_values)
        critic2_loss = F.mse_loss(current_q2, target_values)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_graph_actor(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_types: Optional[torch.Tensor]
    ) -> float:
        """Update graph-based actor."""
        
        # Get actions from current policy
        policy_mean, policy_logstd, node_actions = self.actor(
            node_features, edge_index, node_types=node_types, batch=batch
        )
        
        # Sample actions
        std = torch.exp(policy_logstd)
        normal = torch.distributions.Normal(policy_mean, std)
        sampled_actions = torch.tanh(normal.rsample())
        
        # Q-values for policy improvement
        q1 = self.critic1(node_features, edge_index, sampled_actions, node_types=node_types, batch=batch)
        q2 = self.critic2(node_features, edge_index, sampled_actions, node_types=node_types, batch=batch)
        q_values = torch.min(q1, q2)
        
        # Actor loss
        actor_loss = -q_values.mean()
        
        # Add node-level regularization (encourage distributed coordination)
        if node_actions is not None:
            node_regularization = 0.01 * torch.mean(node_actions ** 2)
            actor_loss += node_regularization
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _track_graph_metrics(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ):
        """Track graph-specific metrics."""
        with torch.no_grad():
            # Node importance (based on feature magnitudes)
            node_importance = torch.norm(node_features, dim=1)
            self.node_importance_history.append(node_importance.mean().item())
            
            # Graph connectivity metrics
            num_edges = edge_index.shape[1]
            num_nodes = self.graph_topology.num_nodes
            connectivity = num_edges / (num_nodes * (num_nodes - 1))  # Density
            self.graph_connectivity_metrics.append(connectivity)
    
    def get_graph_metrics(self) -> Dict[str, Any]:
        """Get graph-specific training metrics."""
        metrics = {
            "node_importance": {
                "mean": np.mean(self.node_importance_history[-100:]) if self.node_importance_history else 0.0,
                "std": np.std(self.node_importance_history[-100:]) if self.node_importance_history else 0.0,
                "trend": np.polyfit(range(len(self.node_importance_history)), 
                                  self.node_importance_history, 1)[0] if len(self.node_importance_history) > 1 else 0.0
            },
            "graph_connectivity": np.mean(self.graph_connectivity_metrics) if self.graph_connectivity_metrics else 0.0,
            "num_nodes": self.graph_topology.num_nodes,
            "num_edges": self.graph_topology.edge_index.shape[1] // 2,  # Undirected edges
            "graph_density": self.graph_connectivity_metrics[-1] if self.graph_connectivity_metrics else 0.0
        }
        
        return metrics
    
    def train_offline(
        self,
        dataset,
        num_epochs: int,
        batch_size: int = 64,  # Smaller batch size for graph processing
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train graph-based federated RL."""
        metrics_history = []
        
        self.logger.info(f"Starting GNFRL training: {num_epochs} epochs, {self.graph_topology.num_nodes} nodes")
        
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
                graph_metrics = self.get_graph_metrics()
                node_importance = graph_metrics["node_importance"]["mean"]
                
                self.logger.info(
                    f"Epoch {epoch}: Loss={avg_metrics.loss:.4f}, "
                    f"NodeImportance={node_importance:.4f}, "
                    f"GraphDensity={graph_metrics['graph_density']:.4f}"
                )
        
        return metrics_history
    
    def save(self, path: str) -> None:
        """Save graph-based model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "graph_topology": self.graph_topology,
            "node_importance_history": self.node_importance_history,
            "training_step": self.training_step
        }, path)
    
    def load(self, path: str) -> None:
        """Load graph-based model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.node_importance_history = checkpoint.get("node_importance_history", [])
        self.training_step = checkpoint.get("training_step", 0)


class GNFRLClient(FederatedClient):
    """Federated client with graph neural networks."""
    
    def __init__(
        self,
        client_id: str,
        algorithm: GNFRL,
        grid_data: List[Dict[str, Any]],
        local_graph_topology: Optional[GraphTopology] = None
    ):
        super().__init__(client_id, algorithm)
        self.local_data = grid_data
        self.local_graph_topology = local_graph_topology or algorithm.graph_topology
        self.graph_adaptation_history = []
        
    def local_update(
        self,
        global_parameters: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int
    ) -> ClientUpdate:
        """Graph-aware local training."""
        # Set global parameters
        self.algorithm.set_parameters(global_parameters)
        
        initial_graph_metrics = self.algorithm.get_graph_metrics()
        
        # Local training
        for epoch in range(epochs):
            batch_data = self._sample_batch(batch_size)
            batch_tensor = self._convert_to_tensor_batch(batch_data)
            
            metrics = self.algorithm.update(batch_tensor)
            
            # Track graph adaptation
            current_metrics = self.algorithm.get_graph_metrics()
            adaptation_score = abs(
                current_metrics["node_importance"]["mean"] - 
                initial_graph_metrics["node_importance"]["mean"]
            )
            self.graph_adaptation_history.append(adaptation_score)
        
        final_graph_metrics = self.algorithm.get_graph_metrics()
        updated_params = self.algorithm.get_parameters()
        
        # Calculate graph-specific improvements
        node_importance_change = (
            final_graph_metrics["node_importance"]["mean"] - 
            initial_graph_metrics["node_importance"]["mean"]
        )
        
        return ClientUpdate(
            client_id=self.client_id,
            parameters=updated_params,
            num_samples=len(self.local_data),
            loss=final_graph_metrics["node_importance"]["mean"],
            metrics={
                "node_importance_change": node_importance_change,
                "graph_adaptation_score": np.mean(self.graph_adaptation_history[-epochs:]) if self.graph_adaptation_history else 0.0,
                "num_nodes": self.local_graph_topology.num_nodes,
                "num_edges": self.local_graph_topology.edge_index.shape[1] // 2,
                "local_graph_density": final_graph_metrics["graph_density"],
                "node_importance_trend": final_graph_metrics["node_importance"]["trend"]
            }
        )
    
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


# Utility functions for creating graph topologies

def create_hierarchical_grid_graph(
    transmission_nodes: int = 20,
    distribution_nodes_per_feeder: int = 10,
    num_feeders: int = 3
) -> GraphTopology:
    """Create hierarchical transmission-distribution graph."""
    total_nodes = transmission_nodes + num_feeders * distribution_nodes_per_feeder
    edges = []
    
    # Transmission level connections (ring topology)
    for i in range(transmission_nodes):
        edges.append((i, (i + 1) % transmission_nodes))
        
        # Add some cross-connections for robustness
        if i % 5 == 0 and i + transmission_nodes // 2 < transmission_nodes:
            edges.append((i, i + transmission_nodes // 2))
    
    # Distribution feeders
    feeder_start = transmission_nodes
    for feeder in range(num_feeders):
        # Connect feeder root to transmission
        transmission_node = feeder * (transmission_nodes // num_feeders)
        feeder_root = feeder_start + feeder * distribution_nodes_per_feeder
        edges.append((transmission_node, feeder_root))
        
        # Create radial distribution feeder
        for i in range(distribution_nodes_per_feeder - 1):
            node_a = feeder_root + i
            node_b = feeder_root + i + 1
            edges.append((node_a, node_b))
    
    # Convert to undirected edges
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    reverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index = torch.cat([edge_index, reverse_edges], dim=1)
    
    # Node types: transmission (PV/slack), distribution (PQ)
    node_types = torch.zeros(total_nodes, dtype=torch.long)
    node_types[0] = 0  # Slack bus
    node_types[1:transmission_nodes] = 1  # Transmission PV buses
    node_types[transmission_nodes:] = 2  # Distribution PQ buses
    
    return GraphTopology(
        num_nodes=total_nodes,
        edge_index=edge_index,
        node_types=node_types
    )