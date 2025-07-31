# Grid-Fed-RL-Gym

Digital twin framework for power distribution networks with federated offline reinforcement learning capabilities. Based on Tsinghua's Fed-RL grid reliability study (January 2025), this toolkit enables distributed training of grid control policies while preserving utility data privacy.

## Overview

Grid-Fed-RL-Gym provides a comprehensive environment for training and deploying reinforcement learning agents on power distribution networks. The framework supports federated learning across multiple utility companies, offline RL for safety-critical applications, and high-fidelity grid simulations compliant with industry standards.

## Key Features

- **Digital Twin Simulation**: High-fidelity power flow modeling with real-time dynamics
- **Federated Learning**: Privacy-preserving distributed training across utilities
- **Offline RL**: Learn from historical data without online exploration
- **Safety Constraints**: Hard constraints on voltage, frequency, and equipment limits
- **Multi-Agent Support**: Coordinate multiple grid controllers and DERs
- **Industry Standards**: IEEE test feeders and CIM-compliant data models

## Installation

```bash
# Basic installation
pip install grid-fed-rl-gym

# With all power system solvers
pip install grid-fed-rl-gym[solvers]

# With federated learning support
pip install grid-fed-rl-gym[federated]

# Full installation
pip install grid-fed-rl-gym[full]

# From source
git clone https://github.com/yourusername/grid-fed-rl-gym
cd grid-fed-rl-gym
pip install -e ".[dev]"
```

## Quick Start

### Basic Grid Environment

```python
from grid_fed_rl import GridEnvironment
from grid_fed_rl.feeders import IEEE13Bus

# Create grid environment
env = GridEnvironment(
    feeder=IEEE13Bus(),
    timestep=1.0,  # seconds
    episode_length=86400,  # 24 hours
    stochastic_loads=True,
    renewable_sources=["solar", "wind"]
)

# Run random policy
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    if done:
        print(f"Episode reward: {info['episode_reward']}")
        print(f"Violations: {info['constraint_violations']}")
        obs = env.reset()
```

### Federated Offline RL Training

```python
from grid_fed_rl.federated import FederatedOfflineRL
from grid_fed_rl.algorithms import ConservativeQL

# Initialize federated learner
fed_learner = FederatedOfflineRL(
    algorithm=ConservativeQL,
    num_clients=5,  # 5 utility companies
    rounds=100,
    privacy_budget=10.0,  # Differential privacy
    aggregation="fedavg"
)

# Each utility loads their historical data
client_datasets = [
    utility.load_historical_data()
    for utility in participating_utilities
]

# Federated training
global_policy = fed_learner.train(
    datasets=client_datasets,
    local_epochs=10,
    batch_size=256,
    safety_penalty=100.0
)

# Deploy learned policy
controller = GridController(policy=global_policy)
env.set_controller(controller)
```

## Architecture

```
grid-fed-rl-gym/
├── grid_fed_rl/
│   ├── environments/
│   │   ├── grid_env/       # Core grid environment
│   │   ├── dynamics/       # Power system dynamics
│   │   └── stochastic/     # Load/generation models
│   ├── feeders/
│   │   ├── ieee/           # IEEE test feeders
│   │   ├── synthetic/      # Synthetic networks
│   │   └── custom/         # User-defined feeders
│   ├── algorithms/
│   │   ├── offline/        # Offline RL algorithms
│   │   ├── safe/           # Safety-constrained RL
│   │   └── multi_agent/    # Multi-agent algorithms
│   ├── federated/
│   │   ├── aggregation/    # Federated aggregation
│   │   ├── privacy/        # Differential privacy
│   │   └── communication/  # Network protocols
│   ├── controllers/
│   │   ├── voltage/        # Voltage regulators
│   │   ├── frequency/      # Frequency control
│   │   └── der/            # DER controllers
│   └── evaluation/
│       ├── metrics/        # Performance metrics
│       ├── reliability/    # Reliability indices
│       └── economics/      # Economic analysis
├── data/
│   ├── load_profiles/      # Historical load data
│   ├── renewable/          # Solar/wind profiles
│   └── outages/            # Outage statistics
└── visualization/          # Grid visualization tools
```

## Power System Modeling

### Grid Components

```python
from grid_fed_rl.components import (
    Bus, Line, Transformer, Load,
    SolarPV, WindTurbine, Battery
)

# Build custom feeder
feeder = CustomFeeder()

# Add buses
feeder.add_bus(Bus(id=1, voltage_level=12.47e3, type="slack"))
feeder.add_bus(Bus(id=2, voltage_level=12.47e3, type="pq"))

# Add line
feeder.add_line(Line(
    from_bus=1,
    to_bus=2,
    resistance=0.01,  # pu
    reactance=0.02,   # pu
    rating=5e6        # VA
))

# Add renewable generation
solar = SolarPV(
    bus=2,
    capacity=2e6,  # 2 MW
    efficiency=0.18,
    panel_area=10000  # m²
)
feeder.add_der(solar)

# Add battery storage
battery = Battery(
    bus=2,
    capacity=1e6,     # 1 MWh
    power_rating=0.5e6,  # 500 kW
    efficiency=0.95,
    initial_soc=0.5
)
feeder.add_der(battery)
```

### Power Flow Solvers

```python
from grid_fed_rl.solvers import (
    NewtonRaphson, FastDecoupled,
    DistributionPowerFlow, UnbalancedPowerFlow
)

# Configure solver
solver = UnbalancedPowerFlow(
    tolerance=1e-6,
    max_iterations=50,
    acceleration_factor=1.6
)

# Solve power flow
solution = solver.solve(feeder, loading_conditions)

print(f"Convergence: {solution.converged}")
print(f"Iterations: {solution.iterations}")
print(f"Bus voltages: {solution.bus_voltages}")
print(f"Line flows: {solution.line_flows}")
```

## Federated Learning

### Privacy-Preserving Training

```python
from grid_fed_rl.federated import SecureAggregator
from grid_fed_rl.privacy import DifferentialPrivacy

# Secure aggregation with encryption
aggregator = SecureAggregator(
    encryption="homomorphic",
    key_size=2048
)

# Differential privacy
privacy = DifferentialPrivacy(
    epsilon=1.0,
    delta=1e-5,
    mechanism="gaussian"
)

# Federated training with privacy
fed_trainer = FederatedTrainer(
    aggregator=aggregator,
    privacy=privacy,
    clip_norm=1.0
)

# Each client trains locally
for round in range(num_rounds):
    client_updates = []
    
    for client in clients:
        # Local training with privacy
        local_update = client.train_local(
            data=client.data,
            epochs=local_epochs,
            add_noise=privacy.add_noise
        )
        
        # Encrypt update
        encrypted_update = aggregator.encrypt(local_update)
        client_updates.append(encrypted_update)
    
    # Secure aggregation
    global_update = aggregator.aggregate(client_updates)
    
    # Update global model
    global_model.apply_update(global_update)
```

### Communication Efficiency

```python
from grid_fed_rl.federated import CompressionStrategy

# Configure compression
compression = CompressionStrategy(
    method="top_k_sparsification",
    compression_ratio=0.1,  # Send only 10% of parameters
    error_feedback=True
)

# Quantization for bandwidth reduction
quantization = QuantizationStrategy(
    bits=8,
    stochastic=True
)

# Efficient federated learning
efficient_fed = EfficientFedRL(
    compression=compression,
    quantization=quantization,
    local_sgd_steps=20  # More local computation
)
```

## Offline RL Algorithms

### Conservative Q-Learning (CQL)

```python
from grid_fed_rl.algorithms import CQL

# Initialize CQL for grid control
cql = CQL(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dims=[256, 256, 256],
    conservative_weight=5.0,
    tau=0.005
)

# Train on historical data
dataset = GridDataset(
    trajectories=historical_trajectories,
    normalize=True,
    augment_safety_violations=True
)

cql.train(
    dataset=dataset,
    batch_size=256,
    num_epochs=1000,
    eval_env=env
)
```

### Implicit Q-Learning (IQL)

```python
from grid_fed_rl.algorithms import IQL

# IQL for safe grid control
iql = IQL(
    env=env,
    expectile=0.7,  # Conservative policy
    temperature=3.0,
    discount=0.99
)

# Incorporate safety constraints
safety_critic = SafetyCritic(
    constraint_functions=[
        voltage_constraint,
        thermal_constraint,
        stability_constraint
    ]
)

iql.set_safety_critic(safety_critic)
iql.train(dataset, safety_weight=10.0)
```

## Multi-Agent Coordination

### Distributed Energy Resources

```python
from grid_fed_rl.multi_agent import MultiAgentEnv, QMIX

# Multi-agent environment with DERs
ma_env = MultiAgentEnv(
    feeder=feeder,
    agents={
        "battery_1": BatteryAgent(battery_1),
        "battery_2": BatteryAgent(battery_2),
        "solar_curtail": CurtailmentAgent(solar_farm),
        "load_control": LoadControlAgent(flexible_loads)
    }
)

# QMIX for coordination
qmix = QMIX(
    n_agents=4,
    state_shape=ma_env.get_state_size(),
    obs_shape=ma_env.get_obs_size(),
    n_actions=ma_env.get_total_actions()
)

# Centralized training, decentralized execution
qmix.train(
    env=ma_env,
    episodes=10000,
    decentralized_execution=True
)
```

## Safety and Constraints

### Constraint-Aware Training

```python
from grid_fed_rl.safety import ConstraintLayer, SafetyShield

# Define operational constraints
constraints = {
    "voltage": lambda v: 0.95 <= v <= 1.05,  # pu
    "line_flow": lambda s: s <= line_ratings,
    "transformer": lambda s: s <= transformer_ratings,
    "frequency": lambda f: 59.5 <= f <= 60.5  # Hz
}

# Add constraint layer to policy
policy = ConstraintAwarePolicy(
    base_policy=learned_policy,
    constraints=constraints,
    violation_penalty=1000
)

# Safety shield for deployment
shield = SafetyShield(
    policy=policy,
    backup_controller=rule_based_controller,
    intervention_threshold=0.95
)

# Safe execution
action = shield.get_action(observation)
if shield.intervened:
    logger.warning(f"Safety intervention at step {t}")
```

## Evaluation and Metrics

### Reliability Indices

```python
from grid_fed_rl.evaluation import ReliabilityMetrics

metrics = ReliabilityMetrics()

# Standard reliability indices
results = metrics.calculate(
    simulation_results,
    indices=["SAIDI", "SAIFI", "CAIDI", "MAIFI", "ENS"]
)

print(f"SAIDI: {results['SAIDI']:.2f} minutes/customer")
print(f"SAIFI: {results['SAIFI']:.2f} interruptions/customer")
print(f"Energy Not Served: {results['ENS']:.2f} MWh")

# Economic analysis
economics = EconomicAnalysis(
    energy_price=0.15,  # $/kWh
    demand_charge=20,   # $/kW
    outage_cost=150     # $/MWh
)

cost_savings = economics.evaluate(
    baseline=rule_based_results,
    rl_policy=rl_results
)
```

### Visualization

```python
from grid_fed_rl.visualization import GridVisualizer

viz = GridVisualizer(feeder)

# Real-time visualization
viz.animate(
    simulation_results,
    show_power_flow=True,
    show_voltages=True,
    show_der_output=True,
    speed=10  # 10x real-time
)

# Generate report
viz.generate_report(
    results=evaluation_results,
    include_plots=["voltage_profile", "load_duration", "der_dispatch"],
    format="pdf",
    output="grid_analysis_report.pdf"
)
```

## Real-World Integration

### SCADA Integration

```python
from grid_fed_rl.integration import SCADAInterface

# Connect to SCADA system
scada = SCADAInterface(
    protocol="DNP3",
    master_ip="192.168.1.100",
    port=20000
)

# Real-time control loop
controller = RLController(policy=trained_policy)

while True:
    # Read measurements
    measurements = scada.read_measurements()
    
    # Compute control actions
    state = env.process_measurements(measurements)
    action = controller.get_action(state)
    
    # Send control commands
    commands = env.action_to_commands(action)
    scada.send_commands(commands)
    
    time.sleep(control_period)
```

### Hardware-in-the-Loop

```python
from grid_fed_rl.hil import HILSimulator

# Configure HIL testing
hil = HILSimulator(
    rtds_interface="OPAL-RT",
    sampling_rate=10000,  # Hz
    analog_channels=32,
    digital_channels=64
)

# Test RL controller
hil.load_model(feeder_model)
hil.connect_controller(rl_controller)

# Run test scenarios
scenarios = ["fault", "renewable_ramp", "peak_load", "islanding"]

for scenario in scenarios:
    results = hil.run_scenario(
        scenario,
        duration=300,  # seconds
        record_data=True
    )
    
    # Validate performance
    assert results.max_voltage < 1.05
    assert results.min_frequency > 59.5
    assert results.no_equipment_violations
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{grid_fed_rl_gym,
  title={Grid-Fed-RL-Gym: Federated Reinforcement Learning for Power Grids},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/grid-fed-rl-gym}
}

@article{tsinghua_fed_rl_2025,
  title={Federated Offline RL for Grid Reliability},
  author={Tsinghua University},
  journal={IEEE Transactions on Smart Grid},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Tsinghua University for Fed-RL research
- IEEE PES for test feeder data
- Power systems community for domain expertise
