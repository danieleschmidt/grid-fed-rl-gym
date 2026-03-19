"""
FederatedGridCoordinator: FedAvg aggregation for grid control policies.

Implements Federated Averaging (McMahan et al., 2017) over policy weights.
No raw grid data is shared — only model weights are exchanged.

Each round:
1. Broadcast global model to all operators
2. Each operator trains locally for K episodes
3. Collect updated weights from all operators
4. Average weights → new global model
5. Distribute global model back
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .agent import LocalRLAgent


@dataclass
class FederationRoundResult:
    """Results from a single federation round."""
    round_num: int
    mean_reward: float
    std_reward: float
    mean_stability: float
    operator_rewards: List[float]
    operator_stability: List[float]
    weight_divergence: float  # std of operator weights before aggregation


class FederatedGridCoordinator:
    """
    Coordinates federated learning across multiple grid operators.

    Args:
        agents: List of LocalRLAgent instances (one per operator)
        local_episodes_per_round: Training episodes each operator runs per round
    """

    def __init__(
        self,
        agents: List[LocalRLAgent],
        local_episodes_per_round: int = 10,
    ):
        self.agents = agents
        self.n_operators = len(agents)
        self.local_episodes_per_round = local_episodes_per_round
        self.round_history: List[FederationRoundResult] = []

        # Initialize global model as average of initial weights
        self.global_weights = self._fedavg(
            [a.get_policy_weights() for a in agents]
        )
        # Broadcast initial global model to all operators
        self._broadcast(self.global_weights)

    def _fedavg(
        self,
        weight_list: List[torch.Tensor],
        sample_counts: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Weighted average of model weights (FedAvg).

        Args:
            weight_list: List of flat weight tensors from each operator
            sample_counts: Optional per-operator sample counts for weighted avg.
                           If None, uses uniform weighting.
        Returns:
            Aggregated flat weight tensor
        """
        if sample_counts is None:
            weights = torch.stack(weight_list)
            return weights.mean(dim=0)

        total = sum(sample_counts)
        agg = torch.zeros_like(weight_list[0])
        for w, n in zip(weight_list, sample_counts):
            agg += w * (n / total)
        return agg

    def _broadcast(self, global_weights: torch.Tensor):
        """Send global model to all operators."""
        for agent in self.agents:
            agent.set_policy_weights(global_weights.clone())

    def run_round(self, round_num: int, verbose: bool = False) -> FederationRoundResult:
        """
        Execute one federation round:
        1. Each operator trains locally
        2. Collect weights
        3. FedAvg
        4. Distribute

        Returns:
            FederationRoundResult with metrics
        """
        # Step 1: Local training
        local_weights = []
        rewards = []
        stability_scores = []

        for agent in self.agents:
            ep_rewards = agent.train(
                n_episodes=self.local_episodes_per_round,
                verbose=False,
            )
            local_weights.append(agent.get_policy_weights())
            rewards.append(np.mean(ep_rewards))

        # Measure weight divergence before aggregation
        weight_stack = torch.stack(local_weights)
        weight_divergence = float(weight_stack.std(dim=0).mean().item())

        # Step 2: FedAvg aggregation
        self.global_weights = self._fedavg(local_weights)

        # Step 3: Broadcast global model
        self._broadcast(self.global_weights)

        # Step 4: Evaluate all operators with new global model
        for agent in self.agents:
            _, stab = agent.evaluate(n_episodes=3)
            stability_scores.append(stab)

        result = FederationRoundResult(
            round_num=round_num,
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            mean_stability=float(np.mean(stability_scores)),
            operator_rewards=rewards,
            operator_stability=stability_scores,
            weight_divergence=weight_divergence,
        )
        self.round_history.append(result)

        if verbose:
            print(
                f"  Round {round_num:2d} | "
                f"avg_reward={result.mean_reward:7.2f} | "
                f"stability={result.mean_stability:.3f} | "
                f"weight_div={result.weight_divergence:.4f}"
            )

        return result

    def run_federation(self, n_rounds: int, verbose: bool = True) -> List[FederationRoundResult]:
        """
        Run a full federation for n_rounds.

        Returns:
            List of FederationRoundResult per round
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Federated Training: {self.n_operators} operators, {n_rounds} rounds")
            print(f"  local_episodes_per_round: {self.local_episodes_per_round}")
            print(f"{'='*60}")

        results = []
        for r in range(1, n_rounds + 1):
            result = self.run_round(r, verbose=verbose)
            results.append(result)

        return results

    def get_global_policy_weights(self) -> torch.Tensor:
        """Return the current global model weights."""
        return self.global_weights.clone()
