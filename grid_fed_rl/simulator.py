"""
GridSimulator: Orchestrates multi-operator federation and measures grid stability.

Compares:
  - Local-only training: each operator trains independently, no federation
  - Federated training: operators share policy weights via FedAvg each round

The stability metric measures how well all operators maintain voltage
within acceptable limits across their respective grid segments.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .env import GridEnv
from .agent import LocalRLAgent
from .federated import FederatedGridCoordinator


@dataclass
class SimulationResult:
    """Results from a full simulation comparison."""
    n_operators: int
    n_rounds: int
    local_stability_scores: List[float]    # per-operator, after local training
    federated_stability_scores: List[float]  # per-operator, after federation
    local_rewards: List[float]
    federated_rewards: List[float]
    per_round_stability: List[float]        # mean stability per federation round
    per_round_reward: List[float]           # mean reward per federation round

    @property
    def mean_local_stability(self) -> float:
        return float(np.mean(self.local_stability_scores))

    @property
    def mean_federated_stability(self) -> float:
        return float(np.mean(self.federated_stability_scores))

    @property
    def stability_improvement(self) -> float:
        """Relative improvement in stability from federation."""
        base = self.mean_local_stability
        if base == 0:
            return 0.0
        return (self.mean_federated_stability - base) / (abs(base) + 1e-8)

    @property
    def mean_local_reward(self) -> float:
        return float(np.mean(self.local_rewards))

    @property
    def mean_federated_reward(self) -> float:
        return float(np.mean(self.federated_rewards))


class GridSimulator:
    """
    Runs full federated RL simulation and compares with local-only baseline.

    Args:
        n_operators: Number of independent grid operators
        n_buses_per_operator: Grid size for each operator
        n_local_pretrain_episodes: Episodes each operator trains before federation
        seed: Base random seed (each operator gets seed + i)
    """

    def __init__(
        self,
        n_operators: int = 3,
        n_buses_per_operator: int = 5,
        n_local_pretrain_episodes: int = 30,
        seed: int = 42,
    ):
        self.n_operators = n_operators
        self.n_buses_per_operator = n_buses_per_operator
        self.n_local_pretrain_episodes = n_local_pretrain_episodes
        self.seed = seed

    def _create_agents(self) -> List[LocalRLAgent]:
        """Create independent GridEnv + LocalRLAgent for each operator."""
        agents = []
        for i in range(self.n_operators):
            env = GridEnv(
                n_buses=self.n_buses_per_operator,
                seed=self.seed + i * 100,
            )
            agent = LocalRLAgent(
                env=env,
                operator_id=f"operator_{i}",
                lr=3e-4,
                gamma=0.99,
            )
            agents.append(agent)
        return agents

    def run(
        self,
        n_federation_rounds: int = 5,
        local_episodes_per_round: int = 10,
        verbose: bool = True,
    ) -> SimulationResult:
        """
        Full simulation: local baseline + federated training.

        Steps:
          1. Create agents for each operator
          2. Local-only training baseline (n_local_pretrain_episodes each)
          3. Evaluate local performance
          4. Reset policies and run federation for n_federation_rounds
          5. Evaluate federated performance
          6. Compare and report

        Args:
            n_federation_rounds: Number of federation rounds
            local_episodes_per_round: Local training steps per round per operator
            verbose: Print progress

        Returns:
            SimulationResult with full comparison metrics
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"GridSimulator: {self.n_operators} operators, "
                  f"{self.n_buses_per_operator} buses each")
            print(f"{'='*60}")

        # ── Phase 1: Local-only baseline ──────────────────────────────
        if verbose:
            print(f"\n[Phase 1] Local-only training "
                  f"({self.n_local_pretrain_episodes} episodes each)")

        local_agents = self._create_agents()
        for agent in local_agents:
            agent.train(
                n_episodes=self.n_local_pretrain_episodes,
                verbose=verbose,
            )

        local_rewards = []
        local_stability = []
        for agent in local_agents:
            r, s = agent.evaluate(n_episodes=5)
            local_rewards.append(r)
            local_stability.append(s)
            if verbose:
                print(f"  {agent.operator_id}: reward={r:.2f}, stability={s:.3f}")

        if verbose:
            print(f"  → Mean local stability: {np.mean(local_stability):.3f}")

        # ── Phase 2: Federated training ───────────────────────────────
        if verbose:
            print(f"\n[Phase 2] Federated training "
                  f"({n_federation_rounds} rounds × {local_episodes_per_round} eps)")

        fed_agents = self._create_agents()
        coordinator = FederatedGridCoordinator(
            agents=fed_agents,
            local_episodes_per_round=local_episodes_per_round,
        )
        round_results = coordinator.run_federation(
            n_rounds=n_federation_rounds,
            verbose=verbose,
        )

        per_round_stability = [r.mean_stability for r in round_results]
        per_round_reward = [r.mean_reward for r in round_results]

        fed_rewards = []
        fed_stability = []
        for agent in fed_agents:
            r, s = agent.evaluate(n_episodes=5)
            fed_rewards.append(r)
            fed_stability.append(s)
            if verbose:
                print(f"  {agent.operator_id}: reward={r:.2f}, stability={s:.3f}")

        if verbose:
            print(f"  → Mean federated stability: {np.mean(fed_stability):.3f}")

        result = SimulationResult(
            n_operators=self.n_operators,
            n_rounds=n_federation_rounds,
            local_stability_scores=local_stability,
            federated_stability_scores=fed_stability,
            local_rewards=local_rewards,
            federated_rewards=fed_rewards,
            per_round_stability=per_round_stability,
            per_round_reward=per_round_reward,
        )

        if verbose:
            self._print_summary(result)

        return result

    def _print_summary(self, result: SimulationResult):
        """Print final comparison summary."""
        print(f"\n{'='*60}")
        print("SUMMARY: Federated vs Local-Only Performance")
        print(f"{'='*60}")
        print(f"  Operators:           {result.n_operators}")
        print(f"  Federation rounds:   {result.n_rounds}")
        print(f"")
        print(f"  LOCAL-ONLY:")
        print(f"    Mean reward:       {result.mean_local_reward:8.2f}")
        print(f"    Mean stability:    {result.mean_local_stability:.4f}")
        print(f"")
        print(f"  FEDERATED:")
        print(f"    Mean reward:       {result.mean_federated_reward:8.2f}")
        print(f"    Mean stability:    {result.mean_federated_stability:.4f}")
        print(f"")
        improvement_pct = result.stability_improvement * 100
        reward_delta = result.mean_federated_reward - result.mean_local_reward
        print(f"  IMPROVEMENT:")
        print(f"    Stability:         {improvement_pct:+.1f}%")
        print(f"    Reward delta:      {reward_delta:+.2f}")
        print(f"")
        print(f"  Stability per round: {[f'{s:.3f}' for s in result.per_round_stability]}")
        print(f"{'='*60}")
