# grid-fed-rl-gym

**Federated Reinforcement Learning for Power Grid Control**

Multiple grid operators train RL agents locally on their own grid segments, then federate their policies via FedAvg — improving collective grid stability without sharing raw operational data.

## Motivation

Power grid control is safety-critical: generators must be dispatched to balance load, minimize transmission losses, and keep bus voltages within ±10% of nominal. Large grid operators already have RL approaches, but sharing raw SCADA data across utility boundaries is a non-starter for both privacy and regulatory reasons.

Federated learning solves this: operators share only **model weights**, not data. Each round, policies improve from the experience of the whole fleet.

## Architecture

```
GridEnv (per operator)
  ├── state:   [load_i, generation_i, voltage_i]  for each bus i
  ├── action:  generator setpoints (normalized [0,1])
  └── reward:  -(grid_loss + voltage_violation_penalty + imbalance_penalty)

PolicyNetwork (MLP: obs → action)
  └── obs_dim → 64 → 64 → act_dim (sigmoid output)

LocalRLAgent
  └── REINFORCE with return normalization + gradient clipping

FederatedGridCoordinator
  └── FedAvg over flat policy weights, N-operator broadcast

GridSimulator
  └── Orchestrates local baseline + federated comparison
```

## Quick Start

```bash
# Install
pip install -e .

# Run demo: 3 operators, 5 federation rounds
python demo.py

# Run tests
pytest tests/ -v
```

## Demo Output

```
GridSimulator: 3 operators, 5 buses each

[Phase 1] Local-only training (30 episodes each)
  operator_0: reward=-459.08, stability=0.930
  operator_1: reward=-559.21, stability=0.919
  operator_2: reward=-488.12, stability=0.927
  → Mean local stability: 0.925

[Phase 2] Federated training (5 rounds × 15 eps)
  Round  1 | avg_reward=-637.77 | stability=0.914 | weight_div=0.0007
  Round  2 | avg_reward=-597.71 | stability=0.934 | weight_div=0.0007
  Round  3 | avg_reward=-653.71 | stability=0.934 | weight_div=0.0007
  ...

  FEDERATED mean reward: -486.87 (+15.27 vs local-only)
```

## Components

### `GridEnv`

Simple DC power flow environment. A grid is a set of buses (nodes) connected by transmission lines (edges), each with resistance. The agent controls generator setpoints; the environment stochastically evolves load demand each step.

```python
from grid_fed_rl import GridEnv

env = GridEnv(n_buses=5, seed=42)
obs = env.reset()  # shape: (15,) = [load×5, gen×5, voltage×5]

action = env.rng.uniform(0, 1, size=5)  # generator setpoints
obs, reward, done, info = env.step(action)
print(info)  # grid_loss, voltage_violations, max_voltage_dev, ...
```

### `PolicyNetwork`

Small MLP with sigmoid output — actions are always in [0, 1]. Supports both stochastic sampling (REINFORCE) and deterministic evaluation.

```python
from grid_fed_rl import PolicyNetwork
import torch

policy = PolicyNetwork(obs_dim=15, act_dim=5)
obs = torch.randn(15)
action, log_prob = policy.get_action(obs)
```

### `LocalRLAgent`

REINFORCE agent with return normalization and gradient clipping. Exposes `get_policy_weights()` / `set_policy_weights()` for federation.

```python
from grid_fed_rl import GridEnv, LocalRLAgent

env = GridEnv(n_buses=5, seed=0)
agent = LocalRLAgent(env, operator_id="operator_0", lr=3e-4)
rewards = agent.train(n_episodes=50)
mean_reward, stability = agent.evaluate()
```

### `FederatedGridCoordinator`

FedAvg over policy weights. Each round: broadcast → local train → collect → aggregate → broadcast.

```python
from grid_fed_rl import GridEnv, LocalRLAgent, FederatedGridCoordinator

agents = [LocalRLAgent(GridEnv(n_buses=5, seed=i)) for i in range(3)]
coord = FederatedGridCoordinator(agents, local_episodes_per_round=10)
results = coord.run_federation(n_rounds=5, verbose=True)
```

### `GridSimulator`

Full comparison: runs local-only baseline and federated training with matching compute budget, then reports metrics side by side.

```python
from grid_fed_rl import GridSimulator

sim = GridSimulator(n_operators=3, n_buses_per_operator=5, seed=42)
result = sim.run(n_federation_rounds=5, local_episodes_per_round=15)
print(f"Reward delta: {result.mean_federated_reward - result.mean_local_reward:+.2f}")
```

## Grid Physics

The environment uses a simplified linearized DC power flow model:

- **Voltage update**: proportional to net injection (generation − load) plus neighbor influence through line resistance
- **Transmission losses**: I²R, where flow ∝ injection difference / resistance  
- **Voltage violation penalty**: quadratic for deviations beyond ±10% nominal (1.0 pu)
- **Imbalance penalty**: penalizes total generation ≠ total load + losses

This is not a full AC solver — it's intentionally lightweight for RL training throughput.

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy ≥ 1.24

## License

MIT
