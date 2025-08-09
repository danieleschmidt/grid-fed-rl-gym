# Grid-Fed-RL-Gym API Reference

Complete API reference for Grid-Fed-RL-Gym framework components.

## Core Environment API

### GridEnvironment

The main reinforcement learning environment for power grid simulation.

```python
from grid_fed_rl import GridEnvironment
from grid_fed_rl.feeders import IEEE13Bus

env = GridEnvironment(
    feeder=IEEE13Bus(),
    timestep=1.0,
    episode_length=86400,
    stochastic_loads=True,
    renewable_sources=["solar", "wind"],
    safety_penalty=100.0
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feeder` | `Feeder` | **Required** | Power network topology |
| `timestep` | `float` | `1.0` | Simulation timestep (seconds) |
| `episode_length` | `int` | `86400` | Episode length (timesteps) |
| `stochastic_loads` | `bool` | `True` | Enable load uncertainty |
| `renewable_sources` | `List[str]` | `None` | Renewable types: `["solar", "wind"]` |
| `weather_variation` | `bool` | `True` | Enable weather dynamics |
| `voltage_limits` | `Tuple[float, float]` | `(0.95, 1.05)` | Voltage limits (p.u.) |
| `frequency_limits` | `Tuple[float, float]` | `(59.5, 60.5)` | Frequency limits (Hz) |
| `safety_penalty` | `float` | `100.0` | Constraint violation penalty |

#### Methods

##### `reset(seed=None, options=None) -> Tuple[np.ndarray, Dict]`
Reset environment to initial state.

**Parameters:**
- `seed` (int, optional): Random seed for reproducibility
- `options` (dict, optional): Additional reset options

**Returns:**
- `observation` (np.ndarray): Initial state observation
- `info` (dict): Additional information

**Example:**
```python
obs, info = env.reset(seed=42)
print(f"Initial observation shape: {obs.shape}")
print(f"Info keys: {list(info.keys())}")
```

##### `step(action) -> Tuple[np.ndarray, float, bool, bool, Dict]`
Execute one simulation step.

**Parameters:**
- `action` (np.ndarray): Control actions

**Returns:**
- `observation` (np.ndarray): Next state observation  
- `reward` (float): Step reward
- `terminated` (bool): Episode termination flag
- `truncated` (bool): Episode truncation flag
- `info` (dict): Step information

**Example:**
```python
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

##### `render(mode="human") -> Optional[np.ndarray]`
Render environment state.

**Parameters:**
- `mode` (str): Render mode ("human", "rgb_array")

**Returns:**
- RGB array for "rgb_array" mode, None otherwise

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `action_space` | `Box` | Action space specification |
| `observation_space` | `Box` | Observation space specification |
| `current_step` | `int` | Current simulation step |
| `episode_reward` | `float` | Cumulative episode reward |

## Feeder Networks API

### IEEE Test Feeders

Standard IEEE test feeder implementations.

```python
from grid_fed_rl.feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus

# IEEE 13-bus test feeder
feeder_13 = IEEE13Bus()

# IEEE 34-bus test feeder  
feeder_34 = IEEE34Bus()

# IEEE 123-bus test feeder
feeder_123 = IEEE123Bus()
```

### Custom Feeder

Create custom power distribution networks.

```python
from grid_fed_rl.feeders import CustomFeeder
from grid_fed_rl.components import Bus, Line, Load, SolarPV

feeder = CustomFeeder()

# Add buses
feeder.add_bus(Bus(id=1, voltage_level=12.47e3, bus_type="slack"))
feeder.add_bus(Bus(id=2, voltage_level=12.47e3, bus_type="pq"))

# Add line
feeder.add_line(Line(
    from_bus=1, to_bus=2,
    resistance=0.01,  # p.u.
    reactance=0.02,   # p.u.
    rating=5e6        # VA
))

# Add load
feeder.add_load(Load(
    bus=2,
    active_power=2e6,    # W
    reactive_power=0.5e6 # VAR
))

# Add renewable DER
feeder.add_der(SolarPV(
    bus=2,
    capacity=1e6,      # W
    efficiency=0.18,   # 18% efficiency
    panel_area=5556    # m²
))
```

## Algorithms API

### Offline Reinforcement Learning

#### Conservative Q-Learning (CQL)

```python
from grid_fed_rl.algorithms import CQL
from grid_fed_rl.data import GridDataset

# Initialize CQL
cql = CQL(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0], 
    hidden_dims=[256, 256],
    conservative_weight=5.0,
    tau=0.005
)

# Prepare dataset
dataset = GridDataset(
    trajectories=historical_data,
    normalize=True
)

# Train
cql.train(
    dataset=dataset,
    batch_size=256,
    num_epochs=1000,
    eval_env=env
)

# Get trained policy
policy = cql.get_policy()
```

#### Implicit Q-Learning (IQL)

```python
from grid_fed_rl.algorithms import IQL

iql = IQL(
    env=env,
    expectile=0.7,
    temperature=3.0,
    discount=0.99
)

iql.train(dataset, safety_weight=10.0)
```

### Multi-Agent Algorithms

#### QMIX

```python
from grid_fed_rl.algorithms import QMIX
from grid_fed_rl.environments import MultiAgentGridEnvironment

# Multi-agent environment
ma_env = MultiAgentGridEnvironment(
    feeder=feeder,
    agents={
        "battery_1": BatteryAgent(),
        "solar_curtail": CurtailmentAgent(),
        "load_control": LoadControlAgent()
    }
)

# QMIX algorithm
qmix = QMIX(
    n_agents=3,
    state_shape=ma_env.get_state_size(),
    obs_shape=ma_env.get_obs_size(),
    n_actions=ma_env.get_total_actions()
)

qmix.train(env=ma_env, episodes=10000)
```

## Federated Learning API

### FederatedOfflineRL

Privacy-preserving distributed training across utilities.

```python
from grid_fed_rl.federated import FederatedOfflineRL
from grid_fed_rl.algorithms import CQL

# Initialize federated learner
fed_learner = FederatedOfflineRL(
    algorithm=CQL,
    num_clients=5,
    rounds=100,
    privacy_budget=10.0,
    aggregation="fedavg"
)

# Client datasets
client_datasets = [
    utility.load_historical_data() 
    for utility in utilities
]

# Federated training
global_policy = fed_learner.train(
    datasets=client_datasets,
    local_epochs=10,
    batch_size=256,
    safety_penalty=100.0
)
```

### Privacy Mechanisms

#### Differential Privacy

```python
from grid_fed_rl.privacy import DifferentialPrivacy

privacy = DifferentialPrivacy(
    epsilon=1.0,        # Privacy budget
    delta=1e-5,         # Failure probability
    mechanism="gaussian" # Noise mechanism
)

# Add noise to gradients
noisy_gradients = privacy.add_noise(gradients)
```

#### Secure Aggregation

```python
from grid_fed_rl.federated import SecureAggregator

aggregator = SecureAggregator(
    encryption="homomorphic",
    key_size=2048
)

# Encrypt client updates
encrypted_updates = [
    aggregator.encrypt(update) 
    for update in client_updates
]

# Aggregate securely
global_update = aggregator.aggregate(encrypted_updates)
```

## Power System Components API

### Bus

```python
from grid_fed_rl.components import Bus

bus = Bus(
    id=1,                    # Bus identifier
    voltage_level=12.47e3,   # Nominal voltage (V)
    bus_type="slack",        # Bus type: slack, pv, pq
    base_voltage=1.0,        # Base voltage (p.u.)
    voltage_magnitude=1.0,   # Current voltage magnitude (p.u.)
    voltage_angle=0.0        # Current voltage angle (rad)
)
```

### Line

```python
from grid_fed_rl.components import Line

line = Line(
    id="line_1_2",      # Line identifier
    from_bus=1,         # From bus ID
    to_bus=2,           # To bus ID
    resistance=0.01,    # Resistance (p.u.)
    reactance=0.02,     # Reactance (p.u.)
    susceptance=0.0,    # Susceptance (p.u.)
    rating=5e6,         # Thermal rating (VA)
    length=1.0,         # Length (km)
    num_phases=3        # Number of phases
)
```

### Distributed Energy Resources

#### Solar PV

```python
from grid_fed_rl.components import SolarPV

solar = SolarPV(
    id="solar_1",
    bus=2,
    capacity=2e6,          # Rated capacity (W)
    efficiency=0.18,       # Panel efficiency
    panel_area=11111,      # Panel area (m²)
    tilt_angle=30,         # Tilt angle (degrees)
    azimuth_angle=180,     # Azimuth angle (degrees)
    temperature_coeff=-0.004  # Temperature coefficient
)
```

#### Wind Turbine

```python
from grid_fed_rl.components import WindTurbine

wind = WindTurbine(
    id="wind_1",
    bus=3,
    capacity=3e6,          # Rated capacity (W)
    cut_in_speed=3.0,      # Cut-in wind speed (m/s)
    rated_speed=12.0,      # Rated wind speed (m/s)
    cut_out_speed=25.0,    # Cut-out wind speed (m/s)
    hub_height=80.0,       # Hub height (m)
    rotor_diameter=100.0   # Rotor diameter (m)
)
```

#### Battery Storage

```python
from grid_fed_rl.components import Battery

battery = Battery(
    id="battery_1",
    bus=2,
    capacity=4e6,          # Energy capacity (Wh)
    power_rating=1e6,      # Power rating (W)
    efficiency=0.95,       # Round-trip efficiency
    initial_soc=0.5,       # Initial state of charge
    min_soc=0.1,          # Minimum SOC
    max_soc=0.9,          # Maximum SOC
    self_discharge=0.001   # Self-discharge rate (1/h)
)
```

## Power Flow Solvers API

### Newton-Raphson Solver

```python
from grid_fed_rl.solvers import NewtonRaphsonSolver

solver = NewtonRaphsonSolver(
    tolerance=1e-6,        # Convergence tolerance
    max_iterations=50,     # Maximum iterations
    acceleration_factor=1.0 # Acceleration factor
)

solution = solver.solve(feeder, load_conditions)

print(f"Converged: {solution.converged}")
print(f"Iterations: {solution.iterations}")
print(f"Bus voltages: {solution.bus_voltages}")
print(f"System losses: {solution.losses}")
```

### Robust Power Flow Solver

Solver with fallback mechanisms for difficult cases.

```python
from grid_fed_rl.solvers import RobustPowerFlowSolver

robust_solver = RobustPowerFlowSolver(
    primary_solver="newton_raphson",
    fallback_solvers=["fast_decoupled", "dc_power_flow"],
    tolerance=1e-4,
    max_iterations=20
)

solution = robust_solver.solve(feeder, load_conditions)
```

## Evaluation and Metrics API

### Reliability Metrics

```python
from grid_fed_rl.evaluation import ReliabilityMetrics

metrics = ReliabilityMetrics()

# Calculate standard reliability indices
results = metrics.calculate(
    simulation_results,
    indices=["SAIDI", "SAIFI", "CAIDI", "MAIFI", "ENS"]
)

print(f"SAIDI: {results['SAIDI']:.2f} minutes/customer")
print(f"SAIFI: {results['SAIFI']:.2f} interruptions/customer") 
print(f"Energy Not Served: {results['ENS']:.2f} MWh")
```

### Economic Analysis

```python
from grid_fed_rl.evaluation import EconomicAnalysis

economics = EconomicAnalysis(
    energy_price=0.15,    # $/kWh
    demand_charge=20,     # $/kW
    outage_cost=150,      # $/MWh
    carbon_price=50       # $/tCO2
)

cost_analysis = economics.evaluate(
    baseline=baseline_results,
    rl_policy=rl_results,
    time_horizon=8760  # hours (1 year)
)

print(f"Annual cost savings: ${cost_analysis['savings']:.2f}")
print(f"ROI: {cost_analysis['roi']:.1%}")
```

## Visualization API

### Grid Visualization

```python
from grid_fed_rl.visualization import GridVisualizer

viz = GridVisualizer(feeder)

# Real-time animation
viz.animate(
    simulation_results,
    show_power_flow=True,
    show_voltages=True,
    show_der_output=True,
    speed=10  # 10x real-time
)

# Static plots
fig = viz.plot_voltage_profile(simulation_results)
fig = viz.plot_power_flow(simulation_results, timestep=100)
fig = viz.plot_der_dispatch(simulation_results)

# Generate comprehensive report
viz.generate_report(
    results=results,
    output_file="grid_analysis_report.pdf",
    include_plots=["voltage_profile", "load_duration", "der_dispatch"]
)
```

## CLI API

### Command Line Interface

```bash
# Basic usage
grid-fed-rl --help

# Train an offline RL agent
grid-fed-rl train --algorithm CQL --data historical_data.csv --feeder IEEE13

# Evaluate a trained policy
grid-fed-rl evaluate --policy trained_policy.pkl --episodes 100

# Run federated learning
grid-fed-rl federated --clients 5 --rounds 100 --data-dir federated_data/

# Start web server
grid-fed-rl serve --host 0.0.0.0 --port 8080

# Generate synthetic data
grid-fed-rl generate-data --feeder IEEE13 --episodes 1000 --output synthetic_data.csv
```

### Programmatic CLI

```python
from grid_fed_rl.cli import CLI

cli = CLI()

# Train model
cli.train(
    algorithm="CQL",
    data_path="historical_data.csv", 
    feeder="IEEE13",
    output_dir="./models/"
)

# Evaluate model  
results = cli.evaluate(
    policy_path="./models/trained_policy.pkl",
    episodes=100,
    render=False
)
```

## Error Handling

### Exception Types

```python
from grid_fed_rl.exceptions import (
    PowerFlowError,
    InvalidActionError,
    SafetyLimitExceededError,
    ConvergenceError,
    DataValidationError
)

try:
    obs, reward, terminated, truncated, info = env.step(action)
except InvalidActionError as e:
    print(f"Invalid action: {e}")
except SafetyLimitExceededError as e:
    print(f"Safety violation: {e}")
except PowerFlowError as e:
    print(f"Power flow failed: {e}")
```

### Input Validation

```python
from grid_fed_rl.utils import validate_action, validate_network_parameters

# Validate actions
try:
    validated_action = validate_action(action, env.action_space)
except ValueError as e:
    print(f"Action validation failed: {e}")

# Validate network parameters
try:
    validate_network_parameters(feeder_config)
except DataValidationError as e:
    print(f"Network validation failed: {e}")
```

## Configuration Management

### Environment Configuration

```python
from grid_fed_rl.config import GridConfig

config = GridConfig(
    solver={
        "type": "newton_raphson",
        "tolerance": 1e-6,
        "max_iterations": 50
    },
    dynamics={
        "frequency_response": True,
        "voltage_dynamics": True,
        "load_modeling": "exponential"
    },
    safety={
        "voltage_limits": [0.95, 1.05],
        "frequency_limits": [59.5, 60.5],
        "enable_protection": True
    }
)

env = GridEnvironment.from_config(config)
```

### Global Settings

```python
from grid_fed_rl import set_global_config

set_global_config({
    "logging": {
        "level": "INFO",
        "file": "grid_rl.log"
    },
    "performance": {
        "cache_size": 1000,
        "parallel_workers": 4
    },
    "security": {
        "enable_input_validation": True,
        "sanitize_outputs": True
    }
})
```

This API reference provides comprehensive documentation for all major components and functions in Grid-Fed-RL-Gym. For additional examples and tutorials, see the [documentation](docs/) directory.