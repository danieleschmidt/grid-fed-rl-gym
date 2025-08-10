# Grid-Fed-RL-Gym API Reference

## Core Classes

### GridEnvironment

The main environment class implementing the OpenAI Gym interface for power grid simulation.

```python
class GridEnvironment(BaseGridEnvironment):
    """
    Grid reinforcement learning environment with power flow simulation.
    
    Args:
        feeder: Power system feeder configuration
        timestep: Simulation timestep in seconds (default: 1.0)
        episode_length: Maximum steps per episode (default: 1000)
        stochastic_loads: Enable stochastic load variations (default: True)
        renewable_sources: List of renewable source types (default: [])
        safety_margin: Safety constraint margin (default: 0.05)
        
    Returns:
        GridEnvironment: Configured environment instance
    """
```

#### Methods

- `reset() -> np.ndarray`: Reset environment and return initial observation
- `step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]`: Execute action and return transition
- `render(mode: str = 'human') -> None`: Visualize current state
- `close() -> None`: Clean up resources

#### Properties

- `observation_space: gym.Space`: Environment observation space
- `action_space: gym.Space`: Environment action space  
- `current_state: Dict`: Current system state
- `safety_violations: List`: Active safety constraint violations

### IEEE Test Feeders

Standard IEEE test system implementations.

#### IEEE13Bus

```python
class IEEE13Bus(BaseFeeder):
    """IEEE 13-bus distribution test feeder."""
    
    def __init__(self, base_voltage_kv: float = 4.16, 
                 base_power_mva: float = 10.0):
        """Initialize IEEE 13-bus system."""
```

#### IEEE34Bus

```python
class IEEE34Bus(BaseFeeder):
    """IEEE 34-bus distribution test feeder."""
```

#### IEEE123Bus

```python
class IEEE123Bus(BaseFeeder):
    """IEEE 123-bus distribution test feeder."""
```

## Federated Learning

### FederatedOfflineRL

Main coordinator for federated reinforcement learning.

```python
class FederatedOfflineRL:
    """
    Federated offline reinforcement learning coordinator.
    
    Args:
        algorithm: Base RL algorithm class (CQL, IQL, etc.)
        num_clients: Number of participating utilities
        rounds: Number of federated training rounds
        privacy_budget: Differential privacy budget (epsilon)
        aggregation: Aggregation strategy ('fedavg', 'weighted')
    """
    
    def train(self, datasets: List[Dataset], 
              local_epochs: int = 10,
              batch_size: int = 256) -> Policy:
        """Train federated policy."""
        
    def evaluate(self, test_env: GridEnvironment) -> Dict[str, float]:
        """Evaluate federated policy."""
```

## Offline RL Algorithms

### Conservative Q-Learning (CQL)

```python
class CQL(OfflineRLAlgorithm):
    """
    Conservative Q-Learning for safe offline RL.
    
    Args:
        state_dim: Dimension of observation space
        action_dim: Dimension of action space  
        hidden_dims: Hidden layer dimensions
        conservative_weight: CQL regularization weight (default: 5.0)
        learning_rate: Optimizer learning rate (default: 3e-4)
    """
```

### Implicit Q-Learning (IQL)

```python
class IQL(OfflineRLAlgorithm):
    """
    Implicit Q-Learning for safe offline RL.
    
    Args:
        env: Grid environment instance
        expectile: Expectile parameter for value estimation (default: 0.7)
        temperature: Policy extraction temperature (default: 3.0)
        discount: Reward discount factor (default: 0.99)
    """
```

## Power System Components

### Bus

```python
class Bus:
    """
    Electrical bus in power system.
    
    Args:
        id: Unique bus identifier
        voltage_level: Nominal voltage level (V)
        bus_type: 'slack', 'pv', or 'pq'
        voltage_magnitude: Per-unit voltage magnitude
        voltage_angle: Voltage angle (radians)
    """
```

### Line

```python
class Line:
    """
    Transmission/distribution line.
    
    Args:
        from_bus: Source bus ID
        to_bus: Destination bus ID  
        resistance: Resistance (pu)
        reactance: Reactance (pu)
        susceptance: Susceptance (pu)
        rating: Thermal rating (VA)
    """
```

### Load

```python
class Load:
    """
    Electrical load component.
    
    Args:
        bus: Connected bus ID
        active_power: Active power demand (W)
        reactive_power: Reactive power demand (VAR)
        load_model: 'constant_power', 'constant_impedance', 'composite'
    """
```

## Utility Classes

### InputValidator

Comprehensive input validation for security.

```python
class InputValidator:
    """Security-focused input validation utilities."""
    
    @staticmethod
    def validate_numeric_input(value: Any, 
                             min_val: float = None,
                             max_val: float = None) -> Tuple[bool, str]:
        """Validate numeric inputs with bounds checking."""
        
    @staticmethod  
    def validate_array_shape(array: np.ndarray,
                           expected_shape: tuple) -> Tuple[bool, str]:
        """Validate array dimensions and size."""
```

### PerformanceProfiler

Performance monitoring and optimization tools.

```python
class PerformanceProfiler:
    """Performance profiling and optimization utilities."""
    
    def profile_function(self, func: Callable) -> Dict[str, Any]:
        """Profile function execution metrics."""
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
```

### SecurityAuditor

Security scanning and vulnerability assessment.

```python
class SecurityAuditor:
    """Comprehensive security auditing system."""
    
    def audit_system(self, config: Dict) -> Dict[str, Any]:
        """Run complete security audit."""
        
    def scan_code(self, code: str) -> List[SecurityIssue]:
        """Scan code for security vulnerabilities."""
```

## Configuration Classes

### GridConfig

```python
@dataclass
class GridConfig:
    """Grid environment configuration."""
    timestep: float = 1.0
    episode_length: int = 1000
    base_voltage: float = 12.47e3
    base_power: float = 10e6
    safety_margin: float = 0.05
    enable_visualization: bool = False
```

### FederatedConfig

```python  
@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    num_clients: int = 5
    rounds: int = 100
    local_epochs: int = 10
    batch_size: int = 256
    privacy_budget: float = 10.0
    secure_aggregation: bool = True
```

## Exception Classes

### GridEnvironmentError

```python
class GridEnvironmentError(Exception):
    """Base exception for grid environment errors."""
    pass
```

### PowerFlowError

```python
class PowerFlowError(GridEnvironmentError):
    """Power flow convergence or calculation errors."""
    pass
```

### SafetyViolationError

```python
class SafetyViolationError(GridEnvironmentError):
    """Safety constraint violation errors."""
    pass
```

## Examples

### Basic Usage

```python
import grid_fed_rl as gfrl

# Create environment
env = gfrl.GridEnvironment(
    feeder=gfrl.IEEE13Bus(),
    timestep=1.0,
    episode_length=1000
)

# Run episode
obs = env.reset()
total_reward = 0

for step in range(1000):
    action = env.action_space.sample()  # Random policy
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    if done:
        break

print(f"Episode reward: {total_reward}")
```

### Federated Training

```python
# Initialize federated learner
fed_rl = gfrl.FederatedOfflineRL(
    algorithm=gfrl.CQL,
    num_clients=5,
    rounds=100,
    privacy_budget=10.0
)

# Load client datasets
datasets = [load_utility_data(i) for i in range(5)]

# Train federated policy  
policy = fed_rl.train(datasets, local_epochs=10)

# Evaluate policy
results = fed_rl.evaluate(test_env)
print(f"Policy performance: {results}")
```