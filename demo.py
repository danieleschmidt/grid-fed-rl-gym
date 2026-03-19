#!/usr/bin/env python3
"""
Demo: Federated RL for Power Grid Control

3 grid operators, 5 federation rounds.
Compares collective federated performance vs local-only training.
"""

from grid_fed_rl import GridSimulator

if __name__ == "__main__":
    sim = GridSimulator(
        n_operators=3,
        n_buses_per_operator=5,
        n_local_pretrain_episodes=30,
        seed=42,
    )

    result = sim.run(
        n_federation_rounds=5,
        local_episodes_per_round=15,
        verbose=True,
    )

    # Final verdict
    if result.stability_improvement > 0:
        print(f"\n✓ Federation improved grid stability by "
              f"{result.stability_improvement*100:.1f}% over local-only training.")
    else:
        print(f"\n~ Federation did not improve over local-only "
              f"(diff={result.stability_improvement*100:.1f}%). "
              f"More rounds or larger grids may help.")
