"""
GridEnv: Power grid control environment.

Models a simple power grid as:
  - Nodes (buses): have load demand, generation capacity, and voltage
  - Edges (transmission lines): carry power between buses
  - State: [load_i, generation_i, voltage_i] for each bus i
  - Action: generator setpoints (normalized [0, 1]) for each bus
  - Reward: -(grid_loss + voltage_violation_penalty)

The grid follows simplified DC power flow physics:
  - Power balance: sum(generation) ≈ sum(load) + losses
  - Voltage deviation from nominal (1.0 pu) incurs penalty
  - Transmission losses proportional to flow^2 * line resistance
"""

import numpy as np
from typing import Tuple, Optional


class GridEnv:
    """
    Simple power grid environment for RL training.

    Args:
        n_buses: Number of buses (nodes) in the grid
        seed: Random seed for reproducibility
        load_variation: Std dev of load fluctuation per step
        voltage_limit: Max allowed voltage deviation from 1.0 pu
    """

    def __init__(
        self,
        n_buses: int = 5,
        seed: Optional[int] = None,
        load_variation: float = 0.05,
        voltage_limit: float = 0.1,
    ):
        self.n_buses = n_buses
        self.load_variation = load_variation
        self.voltage_limit = voltage_limit
        self.rng = np.random.default_rng(seed)

        # Grid topology: random sparse connectivity
        self._build_topology()

        # State/action dimensions
        self.obs_dim = n_buses * 3  # [load, gen, voltage] per bus
        self.act_dim = n_buses      # setpoint per bus

        # Nominal operating parameters
        self.nominal_load = self.rng.uniform(0.3, 0.8, size=n_buses)
        self.max_gen = np.ones(n_buses)  # max generation capacity per bus

        # Episode state
        self.load = None
        self.generation = None
        self.voltage = None
        self.step_count = 0
        self.max_steps = 200

    def _build_topology(self):
        """Build adjacency matrix with line resistances."""
        n = self.n_buses
        # Ensure connectivity: ring topology + some random edges
        adj = np.zeros((n, n))
        for i in range(n):
            j = (i + 1) % n
            r = self.rng.uniform(0.01, 0.05)
            adj[i, j] = r
            adj[j, i] = r
        # Add 1-2 extra random cross-connections
        extra = max(1, n // 3)
        for _ in range(extra):
            i, j = self.rng.choice(n, size=2, replace=False)
            if adj[i, j] == 0:
                r = self.rng.uniform(0.02, 0.08)
                adj[i, j] = r
                adj[j, i] = r
        self.line_resistance = adj  # resistance[i,j] > 0 means line exists

    def reset(self) -> np.ndarray:
        """Reset to initial state, return observation."""
        self.load = self.nominal_load.copy() + self.rng.normal(0, 0.02, self.n_buses)
        self.load = np.clip(self.load, 0.1, 1.0)
        self.generation = self.load.copy()  # start balanced
        self.voltage = np.ones(self.n_buses)
        self.step_count = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply generator setpoints and advance one timestep.

        Args:
            action: Array of shape (n_buses,) with values in [0, 1]
                    representing generator setpoints as fraction of max_gen

        Returns:
            (observation, reward, done, info)
        """
        # Clip action to valid range
        action = np.clip(action, 0.0, 1.0)
        self.generation = action * self.max_gen

        # Update load with stochastic variation
        self.load = self.load + self.rng.normal(0, self.load_variation, self.n_buses)
        self.load = np.clip(self.load, 0.1, 1.0)

        # Simplified DC power flow: compute voltage deviations
        net_injection = self.generation - self.load  # positive = surplus
        self._update_voltages(net_injection)

        # Compute transmission losses
        grid_loss = self._compute_losses()

        # Voltage violation penalty
        voltage_dev = np.abs(self.voltage - 1.0)
        violation_mask = voltage_dev > self.voltage_limit
        voltage_penalty = np.sum(voltage_dev[violation_mask] ** 2) * 10.0

        # Power imbalance penalty (generation should match load + losses)
        total_gen = np.sum(self.generation)
        total_load = np.sum(self.load)
        imbalance_penalty = abs(total_gen - total_load - grid_loss) * 5.0

        reward = -(grid_loss + voltage_penalty + imbalance_penalty)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "grid_loss": grid_loss,
            "voltage_penalty": voltage_penalty,
            "imbalance_penalty": imbalance_penalty,
            "voltage_violations": int(np.sum(violation_mask)),
            "max_voltage_dev": float(np.max(voltage_dev)),
        }

        return self._get_obs(), float(reward), done, info

    def _update_voltages(self, net_injection: np.ndarray):
        """
        Simplified voltage update: voltage proportional to local net injection
        plus neighbor influence through susceptance.
        """
        # Laplacian-based voltage sensitivity (linearized)
        influence = np.zeros(self.n_buses)
        for i in range(self.n_buses):
            for j in range(self.n_buses):
                r = self.line_resistance[i, j]
                if r > 0:
                    # Voltage propagation from neighbors
                    influence[i] += (net_injection[j] - net_injection[i]) * r * 0.5

        self.voltage = 1.0 + 0.1 * net_injection + 0.05 * influence
        # Clamp to realistic range
        self.voltage = np.clip(self.voltage, 0.8, 1.2)

    def _compute_losses(self) -> float:
        """Compute I²R transmission losses."""
        total_loss = 0.0
        net_injection = self.generation - self.load
        for i in range(self.n_buses):
            for j in range(i + 1, self.n_buses):
                r = self.line_resistance[i, j]
                if r > 0:
                    # Approximate current flow from voltage/resistance
                    flow = (net_injection[i] - net_injection[j]) / (r + 1e-6) * 0.01
                    total_loss += flow ** 2 * r
        return float(total_loss)

    def _get_obs(self) -> np.ndarray:
        """Return flattened [load, generation, voltage] observation."""
        return np.concatenate([self.load, self.generation, self.voltage]).astype(np.float32)

    def grid_stability_score(self) -> float:
        """
        Returns a [0, 1] stability score: 1.0 = perfect, 0.0 = severe instability.
        Based on voltage deviations from nominal.
        """
        if self.voltage is None:
            return 0.0
        voltage_dev = np.abs(self.voltage - 1.0)
        # Score = fraction of buses within voltage limit
        within_limit = np.mean(voltage_dev <= self.voltage_limit)
        avg_dev = np.mean(voltage_dev)
        return float(0.7 * within_limit + 0.3 * max(0, 1.0 - avg_dev / self.voltage_limit))
