"""Uncertainty-Aware Federated Reinforcement Learning for Power Systems.

This module implements Bayesian federated RL algorithms with uncertainty quantification
for robust renewable energy integration and risk-aware grid control.
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent, MultivariateNormal
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import math

from .base import OfflineRLAlgorithm, ActorCriticBase, TrainingMetrics
from ..federated.core import FederatedClient, ClientUpdate
from ..utils.validation import validate_constraints


@dataclass
class UncertaintyMetrics:
    """Metrics for tracking uncertainty in predictions."""
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    total_uncertainty: float     # Combined uncertainty
    confidence_interval: Tuple[float, float]  # 95% confidence interval
    prediction_variance: float
    uncertainty_reduction: float = 0.0  # Reduction from previous step


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_variance: float = 1.0,
        variational_inference: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.variational_inference = variational_inference
        
        # Weight mean and variance parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))  # log(sigma)
        
        # Bias mean and variance parameters  
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))
        
        # Prior parameters
        self.prior_variance = prior_variance
        self.register_buffer('prior_weight_mu', torch.zeros(out_features, in_features))
        self.register_buffer('prior_weight_sigma', torch.full((out_features, in_features), math.sqrt(prior_variance)))
        self.register_buffer('prior_bias_mu', torch.zeros(out_features))
        self.register_buffer('prior_bias_sigma', torch.full((out_features,), math.sqrt(prior_variance)))
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.weight_mu, 0, 0.1)
        nn.init.normal_(self.bias_mu, 0, 0.1)
        
    @property
    def weight_sigma(self):
        """Get weight standard deviation."""
        return torch.log1p(torch.exp(self.weight_rho))
    
    @property
    def bias_sigma(self):
        """Get bias standard deviation."""  
        return torch.log1p(torch.exp(self.bias_rho))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling."""
        if self.training and self.variational_inference:
            # Sample weights from posterior
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + self.weight_sigma * weight_eps
            
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + self.bias_sigma * bias_eps
        else:
            # Use mean weights for deterministic inference
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(input, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        # KL for weights
        weight_kl = self._kl_gaussian(
            self.weight_mu, self.weight_sigma,
            self.prior_weight_mu, self.prior_weight_sigma
        )
        
        # KL for bias
        bias_kl = self._kl_gaussian(
            self.bias_mu, self.bias_sigma,
            self.prior_bias_mu, self.prior_bias_sigma
        )
        
        return weight_kl + bias_kl
    
    def _kl_gaussian(self, mu_q, sigma_q, mu_p, sigma_p):
        """KL divergence between two Gaussian distributions."""
        kl = torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
        return kl.sum()


class BayesianMLP(nn.Module):
    """Multi-layer perceptron with Bayesian layers."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        prior_variance: float = 1.0,
        dropout_rate: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_variance = prior_variance
        
        # Build layers
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(BayesianLinear(prev_dim, hidden_dim, prior_variance))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Output layer
        self.output_layer = BayesianLinear(prev_dim, output_dim, prior_variance)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Bayesian MLP."""
        for layer, dropout in zip(self.layers, self.dropout_layers):
            x = layer(x)
            x = self.activation(x)
            x = dropout(x)
            
        return self.output_layer(x)
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence of all Bayesian layers."""
        kl_div = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_div += layer.kl_divergence()
                
        if isinstance(self.output_layer, BayesianLinear):
            kl_div += self.output_layer.kl_divergence()
            
        return kl_div
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with uncertainty quantification."""
        self.train()  # Enable dropout and weight sampling
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_samples, batch_size, output_dim]
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)
        epistemic_uncertainty = torch.sqrt(variance)
        
        return mean_pred, epistemic_uncertainty, predictions


class UncertaintyAwareCritic(nn.Module):
    """Critic network with uncertainty estimation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        prior_variance: float = 1.0,
        num_ensemble: int = 5
    ):
        super().__init__()
        self.num_ensemble = num_ensemble
        
        # Ensemble of Bayesian critics
        self.critics = nn.ModuleList()
        for _ in range(num_ensemble):
            critic = BayesianMLP(
                input_dim=state_dim + action_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
                prior_variance=prior_variance
            )
            self.critics.append(critic)
            
        # Aleatoric uncertainty head (for data noise)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Softplus()  # Ensure positive variance
        )
        
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, UncertaintyMetrics]]:
        """Forward pass with optional uncertainty estimation."""
        x = torch.cat([state, action], dim=-1)
        
        # Get predictions from ensemble
        q_values = []
        for critic in self.critics:
            q_val = critic(x)
            q_values.append(q_val)
            
        q_values = torch.stack(q_values)  # [num_ensemble, batch_size, 1]
        mean_q = torch.mean(q_values, dim=0)
        
        if not return_uncertainty:
            return mean_q
        
        # Calculate uncertainties
        epistemic_var = torch.var(q_values, dim=0)
        epistemic_uncertainty = torch.sqrt(epistemic_var).mean().item()
        
        aleatoric_var = self.aleatoric_head(x)
        aleatoric_uncertainty = torch.sqrt(aleatoric_var).mean().item()
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Confidence interval (95%)
        q_std = torch.sqrt(epistemic_var + aleatoric_var)
        confidence_lower = mean_q - 1.96 * q_std
        confidence_upper = mean_q + 1.96 * q_std
        
        uncertainty_metrics = UncertaintyMetrics(
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_interval=(confidence_lower.mean().item(), confidence_upper.mean().item()),
            prediction_variance=epistemic_var.mean().item() + aleatoric_var.mean().item()
        )
        
        return mean_q, uncertainty_metrics
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence from all ensemble members."""
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for critic in self.critics:
            total_kl += critic.kl_divergence()
            
        return total_kl / self.num_ensemble  # Average over ensemble


class RenewableUncertaintyModel(nn.Module):
    """Model for predicting renewable generation uncertainty."""
    
    def __init__(
        self,
        state_dim: int,
        weather_dim: int = 4,  # solar, wind, temp, cloud_cover
        hidden_dims: List[int] = [128, 64, 32]
    ):
        super().__init__()
        self.weather_dim = weather_dim
        
        # Weather feature extractor
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Renewable generation predictor
        self.renewable_predictor = BayesianMLP(
            input_dim=state_dim + 32,  # state + weather features
            output_dim=10,  # Predict multiple renewable sources
            hidden_dims=hidden_dims,
            prior_variance=0.5
        )
        
        # Uncertainty predictor for renewable volatility
        self.volatility_predictor = nn.Sequential(
            nn.Linear(state_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),  # Volatility for each source
            nn.Softplus()  # Ensure positive
        )
        
    def forward(
        self,
        state: torch.Tensor,
        weather: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict renewable generation with uncertainty."""
        # Encode weather features
        weather_features = self.weather_encoder(weather)
        
        # Combined input
        combined_input = torch.cat([state, weather_features], dim=-1)
        
        # Predict renewable generation
        renewable_pred = self.renewable_predictor(combined_input)
        
        # Predict volatility
        volatility = self.volatility_predictor(combined_input)
        
        # Sample multiple predictions for uncertainty estimation
        predictions = []
        for _ in range(20):  # Monte Carlo sampling
            pred = self.renewable_predictor(combined_input)
            predictions.append(pred)
            
        predictions = torch.stack(predictions)
        uncertainty = torch.std(predictions, dim=0)
        
        return renewable_pred, uncertainty, volatility


class UAFRL(OfflineRLAlgorithm, ActorCriticBase):
    """Uncertainty-Aware Federated Reinforcement Learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 128],
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        uncertainty_weight: float = 1.0,
        risk_aversion: float = 0.5,
        num_mc_samples: int = 50,
        device: str = "auto",
        **kwargs
    ):
        super().__init__(state_dim, action_dim, device=device, **kwargs)
        ActorCriticBase.__init__(self, state_dim, action_dim, hidden_dims, device=device, **kwargs)
        
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.uncertainty_weight = uncertainty_weight
        self.risk_aversion = risk_aversion  # Higher = more risk averse
        self.num_mc_samples = num_mc_samples
        
        # Build networks
        self._build_networks()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)
        self.renewable_optimizer = optim.Adam(self.renewable_model.parameters(), lr=lr/2)
        
        # Uncertainty tracking
        self.uncertainty_history = []
        self.risk_adjusted_returns = []
        self.renewable_prediction_errors = []
        
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """Build uncertainty-aware networks."""
        # Standard actor with dropout for MC sampling
        self.actor = self._build_actor_network().to(self.device)
        
        # Uncertainty-aware critic ensemble
        self.critic = UncertaintyAwareCritic(
            self.state_dim,
            self.action_dim,
            self.hidden_dims,
            prior_variance=1.0,
            num_ensemble=5
        ).to(self.device)
        
        # Target critic
        self.critic_target = UncertaintyAwareCritic(
            self.state_dim,
            self.action_dim,
            self.hidden_dims,
            prior_variance=1.0,
            num_ensemble=5
        ).to(self.device)
        
        # Copy parameters
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Renewable uncertainty model
        self.renewable_model = RenewableUncertaintyModel(
            self.state_dim
        ).to(self.device)
        
    def _build_actor_network(self) -> nn.Module:
        """Build actor network with MC dropout."""
        layers = []
        prev_dim = self.state_dim
        
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Keep dropout active for uncertainty
            ])
            prev_dim = dim
            
        # Output mean and log_std
        layers.append(nn.Linear(prev_dim, self.action_dim * 2))
        
        return nn.Sequential(*layers)
        
    def select_action(
        self,
        state: np.ndarray,
        weather: Optional[np.ndarray] = None,
        eval_mode: bool = False,
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, UncertaintyMetrics]]:
        """Select action with uncertainty quantification."""
        state_tensor = self.to_tensor(state).unsqueeze(0)
        
        if weather is None:
            weather = np.array([0.5, 5.0, 25.0, 0.3])  # Default weather
        weather_tensor = self.to_tensor(weather).unsqueeze(0)
        
        if eval_mode and not return_uncertainty:
            # Deterministic action
            with torch.no_grad():
                action, _ = self._get_action_and_log_prob(state_tensor, deterministic=True)
            return self.to_numpy(action.squeeze(0))
        
        # Monte Carlo sampling for uncertainty
        actions = []
        renewable_preds = []
        
        self.actor.train()  # Enable dropout
        
        for _ in range(self.num_mc_samples):
            with torch.no_grad():
                action, _ = self._get_action_and_log_prob(state_tensor, deterministic=False)
                actions.append(action)
                
                # Predict renewable uncertainty
                renewable_pred, renewable_unc, _ = self.renewable_model(state_tensor, weather_tensor)
                renewable_preds.append(renewable_pred)
        
        actions = torch.stack(actions)  # [num_samples, 1, action_dim]
        renewable_preds = torch.stack(renewable_preds)
        
        # Calculate uncertainties
        action_mean = torch.mean(actions, dim=0)
        action_var = torch.var(actions, dim=0)
        action_uncertainty = torch.sqrt(action_var)
        
        renewable_mean = torch.mean(renewable_preds, dim=0)
        renewable_var = torch.var(renewable_preds, dim=0)
        
        if not return_uncertainty:
            return self.to_numpy(action_mean.squeeze(0))
        
        # Compute comprehensive uncertainty metrics
        epistemic_unc = torch.mean(action_uncertainty).item()
        renewable_unc_val = torch.mean(torch.sqrt(renewable_var)).item()
        total_unc = epistemic_unc + renewable_unc_val
        
        uncertainty_metrics = UncertaintyMetrics(
            epistemic_uncertainty=epistemic_unc,
            aleatoric_uncertainty=renewable_unc_val,
            total_uncertainty=total_unc,
            confidence_interval=(
                (action_mean - 1.96 * action_uncertainty).mean().item(),
                (action_mean + 1.96 * action_uncertainty).mean().item()
            ),
            prediction_variance=torch.mean(action_var).item()
        )
        
        return self.to_numpy(action_mean.squeeze(0)), uncertainty_metrics
        
    def _get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False):
        """Get action and log probability."""
        actor_output = self.actor(state)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def update(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Uncertainty-aware update step."""
        self.training_step += 1
        
        states = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_observations"]
        terminals = batch["terminals"]
        
        # Extract weather data if available
        if "weather" in batch:
            weather = batch["weather"]
        else:
            weather = torch.zeros(states.shape[0], 4, device=self.device)  # Default weather
        
        # Update renewable uncertainty model
        renewable_loss = self._update_renewable_model(states, weather, rewards)
        
        # Update uncertainty-aware critic
        critic_loss, uncertainty_metrics = self._update_critic_with_uncertainty(
            states, actions, rewards, next_states, terminals, weather
        )
        
        # Update risk-aware actor
        actor_loss = self._update_risk_aware_actor(states, weather)
        
        # Update target networks
        self._soft_update_targets()
        
        # Track uncertainty metrics
        self.uncertainty_history.append(uncertainty_metrics)
        
        # Calculate risk-adjusted returns
        risk_adjusted_return = torch.mean(rewards) - self.risk_aversion * uncertainty_metrics.total_uncertainty
        self.risk_adjusted_returns.append(risk_adjusted_return)
        
        total_loss = critic_loss + actor_loss + 0.1 * renewable_loss
        
        return TrainingMetrics(
            loss=total_loss,
            q_loss=critic_loss,
            policy_loss=actor_loss,
            alpha_loss=renewable_loss,
            mean_q_value=torch.mean(rewards).item()
        )
    
    def _update_renewable_model(
        self,
        states: torch.Tensor,
        weather: torch.Tensor,
        rewards: torch.Tensor
    ) -> float:
        """Update renewable generation uncertainty model."""
        # Predict renewable generation
        renewable_pred, uncertainty, volatility = self.renewable_model(states, weather)
        
        # Assume rewards contain renewable generation information
        if states.shape[-1] >= 20:
            actual_renewable = states[:, 10:20]  # Extract actual renewable generation
            
            # Prediction loss
            prediction_loss = F.mse_loss(renewable_pred, actual_renewable)
            
            # Uncertainty calibration loss
            prediction_errors = torch.abs(renewable_pred - actual_renewable)
            uncertainty_loss = F.mse_loss(uncertainty, prediction_errors)
            
            # Volatility loss (encourage higher uncertainty for volatile periods)
            volatility_target = torch.std(actual_renewable, dim=0, keepdim=True).expand_as(volatility)
            volatility_loss = F.mse_loss(volatility, volatility_target)
            
            total_loss = prediction_loss + 0.5 * uncertainty_loss + 0.1 * volatility_loss
        else:
            # Fallback: use reward signal
            total_loss = F.mse_loss(renewable_pred, rewards.unsqueeze(-1).expand_as(renewable_pred))
        
        # Add KL divergence regularization
        kl_div = self.renewable_model.renewable_predictor.kl_divergence()
        total_loss += 1e-4 * kl_div
        
        self.renewable_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.renewable_model.parameters(), 1.0)
        self.renewable_optimizer.step()
        
        return total_loss.item()
    
    def _update_critic_with_uncertainty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminals: torch.Tensor,
        weather: torch.Tensor
    ) -> Tuple[float, UncertaintyMetrics]:
        """Update critic with uncertainty quantification."""
        batch_size = states.shape[0]
        
        with torch.no_grad():
            # Get next actions with uncertainty
            next_actions, _ = self._get_action_and_log_prob(next_states)
            
            # Target Q-values with uncertainty
            target_q, target_uncertainty = self.critic_target(
                next_states, next_actions, return_uncertainty=True
            )
            
            # Risk-adjusted target values
            risk_penalty = self.risk_aversion * target_uncertainty.total_uncertainty
            target_values = rewards.unsqueeze(-1) + self.gamma * (1 - terminals.unsqueeze(-1)) * (target_q - risk_penalty)
        
        # Current Q-values with uncertainty
        current_q, current_uncertainty = self.critic(states, actions, return_uncertainty=True)
        
        # Bellman loss with uncertainty weighting
        bellman_loss = F.mse_loss(current_q, target_values)
        
        # KL divergence regularization
        kl_div = self.critic.kl_divergence()
        
        # Uncertainty calibration loss
        # Encourage higher uncertainty for out-of-distribution states
        renewable_pred, renewable_unc, _ = self.renewable_model(states, weather)
        ood_penalty = torch.mean(renewable_unc) * self.uncertainty_weight
        
        total_critic_loss = bellman_loss + 1e-4 * kl_div + 0.1 * ood_penalty
        
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return total_critic_loss.item(), current_uncertainty
    
    def _update_risk_aware_actor(self, states: torch.Tensor, weather: torch.Tensor) -> float:
        """Update actor with risk-aware objective."""
        # Get actions with uncertainty
        actions, log_probs = self._get_action_and_log_prob(states)
        
        # Q-values with uncertainty
        q_values, q_uncertainty = self.critic(states, actions, return_uncertainty=True)
        
        # Risk-adjusted actor loss
        risk_penalty = self.risk_aversion * q_uncertainty.total_uncertainty
        risk_adjusted_q = q_values - risk_penalty
        
        actor_loss = -risk_adjusted_q.mean()
        
        # Add renewable uncertainty penalty
        renewable_pred, renewable_unc, _ = self.renewable_model(states, weather)
        renewable_risk_penalty = self.risk_aversion * torch.mean(renewable_unc)
        actor_loss += 0.1 * renewable_risk_penalty
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks."""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_uncertainty_metrics(self) -> Dict[str, Any]:
        """Get comprehensive uncertainty metrics."""
        if not self.uncertainty_history:
            return {}
        
        recent_uncertainties = self.uncertainty_history[-100:]
        
        metrics = {
            "epistemic_uncertainty": {
                "mean": np.mean([u.epistemic_uncertainty for u in recent_uncertainties]),
                "std": np.std([u.epistemic_uncertainty for u in recent_uncertainties]),
                "trend": np.polyfit(range(len(recent_uncertainties)), 
                                  [u.epistemic_uncertainty for u in recent_uncertainties], 1)[0]
            },
            "aleatoric_uncertainty": {
                "mean": np.mean([u.aleatoric_uncertainty for u in recent_uncertainties]),
                "std": np.std([u.aleatoric_uncertainty for u in recent_uncertainties])
            },
            "total_uncertainty": {
                "mean": np.mean([u.total_uncertainty for u in recent_uncertainties]),
                "std": np.std([u.total_uncertainty for u in recent_uncertainties]),
                "reduction": np.mean([u.uncertainty_reduction for u in recent_uncertainties if u.uncertainty_reduction > 0])
            },
            "risk_adjusted_returns": {
                "mean": np.mean(self.risk_adjusted_returns[-100:]) if self.risk_adjusted_returns else 0.0,
                "std": np.std(self.risk_adjusted_returns[-100:]) if self.risk_adjusted_returns else 0.0
            },
            "uncertainty_calibration": self._calculate_uncertainty_calibration(),
            "risk_aversion": self.risk_aversion,
            "uncertainty_weight": self.uncertainty_weight
        }
        
        return metrics
    
    def _calculate_uncertainty_calibration(self) -> Dict[str, float]:
        """Calculate how well uncertainty estimates are calibrated."""
        if len(self.renewable_prediction_errors) < 50:
            return {"calibration_error": 0.0, "coverage": 0.0}
        
        recent_errors = self.renewable_prediction_errors[-100:]
        
        # Simple calibration metrics
        error_percentiles = np.percentile(recent_errors, [5, 25, 75, 95])
        
        return {
            "calibration_error": np.std(recent_errors),  # Lower is better
            "coverage": len([e for e in recent_errors if abs(e) < 1.0]) / len(recent_errors),  # Higher is better
            "error_range": error_percentiles[3] - error_percentiles[0]
        }
    
    def train_offline(
        self,
        dataset,
        num_epochs: int,
        batch_size: int = 256,
        **kwargs
    ) -> List[TrainingMetrics]:
        """Train uncertainty-aware federated RL."""
        metrics_history = []
        
        self.logger.info(f"Starting UAFRL training: {num_epochs} epochs, risk_aversion={self.risk_aversion}")
        
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
                policy_loss=np.mean([m.policy_loss for m in epoch_metrics]),
                alpha_loss=np.mean([m.alpha_loss for m in epoch_metrics])
            )
            
            metrics_history.append(avg_metrics)
            
            if epoch % 50 == 0:
                uncertainty_metrics = self.get_uncertainty_metrics()
                total_unc = uncertainty_metrics.get("total_uncertainty", {}).get("mean", 0.0)
                risk_adj_return = uncertainty_metrics.get("risk_adjusted_returns", {}).get("mean", 0.0)
                
                self.logger.info(
                    f"Epoch {epoch}: Loss={avg_metrics.loss:.4f}, "
                    f"Uncertainty={total_unc:.4f}, "
                    f"RiskAdjReturn={risk_adj_return:.4f}"
                )
        
        return metrics_history
    
    def save(self, path: str) -> None:
        """Save uncertainty-aware model."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "renewable_model": self.renewable_model.state_dict(),
            "uncertainty_history": self.uncertainty_history,
            "risk_adjusted_returns": self.risk_adjusted_returns,
            "training_step": self.training_step
        }, path)
    
    def load(self, path: str) -> None:
        """Load uncertainty-aware model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.renewable_model.load_state_dict(checkpoint["renewable_model"])
        self.uncertainty_history = checkpoint.get("uncertainty_history", [])
        self.risk_adjusted_returns = checkpoint.get("risk_adjusted_returns", [])
        self.training_step = checkpoint.get("training_step", 0)


class UAFRLClient(FederatedClient):
    """Federated client with uncertainty-aware RL."""
    
    def __init__(
        self,
        client_id: str,
        algorithm: UAFRL,
        grid_data: List[Dict[str, Any]],
        local_weather_data: Optional[List[np.ndarray]] = None,
        uncertainty_threshold: float = 0.5
    ):
        super().__init__(client_id, algorithm)
        self.local_data = grid_data
        self.local_weather_data = local_weather_data or []
        self.uncertainty_threshold = uncertainty_threshold
        self.high_uncertainty_count = 0
        
    def local_update(
        self,
        global_parameters: Dict[str, np.ndarray],
        epochs: int,
        batch_size: int
    ) -> ClientUpdate:
        """Uncertainty-aware local training."""
        # Set global parameters
        self.algorithm.set_parameters(global_parameters)
        
        initial_uncertainty = self.algorithm.get_uncertainty_metrics()
        
        # Local training
        for epoch in range(epochs):
            batch_data = self._sample_batch(batch_size)
            
            # Add weather data if available
            if self.local_weather_data:
                weather_batch = np.random.choice(self.local_weather_data, size=batch_size)
                batch_tensor = self._convert_to_tensor_batch_with_weather(batch_data, weather_batch)
            else:
                batch_tensor = self._convert_to_tensor_batch(batch_data)
                
            metrics = self.algorithm.update(batch_tensor)
            
            # Track high uncertainty episodes
            if hasattr(self.algorithm, 'uncertainty_history') and self.algorithm.uncertainty_history:
                recent_unc = self.algorithm.uncertainty_history[-1]
                if recent_unc.total_uncertainty > self.uncertainty_threshold:
                    self.high_uncertainty_count += 1
        
        final_uncertainty = self.algorithm.get_uncertainty_metrics()
        updated_params = self.algorithm.get_parameters()
        
        # Calculate uncertainty reduction
        initial_total_unc = initial_uncertainty.get("total_uncertainty", {}).get("mean", 0.0)
        final_total_unc = final_uncertainty.get("total_uncertainty", {}).get("mean", 0.0)
        uncertainty_reduction = max(0.0, initial_total_unc - final_total_unc)
        
        return ClientUpdate(
            client_id=self.client_id,
            parameters=updated_params,
            num_samples=len(self.local_data),
            loss=final_uncertainty.get("risk_adjusted_returns", {}).get("mean", 0.0),
            metrics={
                "uncertainty_reduction": uncertainty_reduction,
                "final_total_uncertainty": final_total_unc,
                "high_uncertainty_episodes": self.high_uncertainty_count,
                "epistemic_uncertainty": final_uncertainty.get("epistemic_uncertainty", {}).get("mean", 0.0),
                "aleatoric_uncertainty": final_uncertainty.get("aleatoric_uncertainty", {}).get("mean", 0.0),
                "has_weather_data": len(self.local_weather_data) > 0
            }
        )
    
    def _convert_to_tensor_batch_with_weather(
        self,
        batch_data: List[Dict[str, Any]],
        weather_batch: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Convert batch data with weather information to tensors."""
        batch_tensor = self._convert_to_tensor_batch(batch_data)
        
        # Add weather data
        weather_tensor = torch.FloatTensor(weather_batch).to(self.algorithm.device)
        batch_tensor["weather"] = weather_tensor
        
        return batch_tensor
    
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