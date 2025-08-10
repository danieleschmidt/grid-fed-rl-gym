# Grid-Fed-RL-Gym Tutorials

## Tutorial 1: Getting Started with Grid Environments

### Setup

First, install the package and dependencies:

```bash
pip install grid-fed-rl-gym[full]
```

### Creating Your First Grid Environment

```python
import numpy as np
import matplotlib.pyplot as plt
from grid_fed_rl import GridEnvironment
from grid_fed_rl.feeders import IEEE13Bus

# Create a standard IEEE 13-bus test feeder environment
env = GridEnvironment(
    feeder=IEEE13Bus(),
    timestep=1.0,           # 1-second timesteps
    episode_length=1000,    # 1000 steps per episode
    stochastic_loads=True,  # Enable load variations
    renewable_sources=["solar", "wind"]
)

print(f"Observation space: {env.observation_space.shape}")
print(f"Action space: {env.action_space.shape}")
```

### Understanding Observations and Actions

```python
# Reset environment and examine initial state
obs = env.reset()
print("Initial observation vector:")
print(f"- Bus voltages: {obs[:13]}")
print(f"- Line flows: {obs[13:25]}")  
print(f"- Load levels: {obs[25:]}")

# Sample random action
action = env.action_space.sample()
print(f"Action vector: {action}")
print("Actions control voltage regulators, capacitor banks, and DER output")
```

### Running a Basic Episode

```python
def run_episode(env, policy='random', max_steps=100):
    """Run single episode with given policy."""
    obs = env.reset()
    total_reward = 0
    violations = []
    
    for step in range(max_steps):
        if policy == 'random':
            action = env.action_space.sample()
        elif policy == 'conservative':
            # Conservative policy - minimal actions
            action = np.zeros_like(env.action_space.sample()) 
        else:
            action = policy(obs)  # Custom policy function
            
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        # Track safety violations
        if info.get('violations', 0) > 0:
            violations.append(step)
            
        if done:
            break
            
    return {
        'total_reward': total_reward,
        'steps': step + 1, 
        'violations': violations,
        'final_info': info
    }

# Compare random vs conservative policies
random_results = run_episode(env, 'random')
conservative_results = run_episode(env, 'conservative')

print("Random Policy Results:")
print(f"  Reward: {random_results['total_reward']:.2f}")
print(f"  Violations: {len(random_results['violations'])}")

print("Conservative Policy Results:")
print(f"  Reward: {conservative_results['total_reward']:.2f}")  
print(f"  Violations: {len(conservative_results['violations'])}")
```

## Tutorial 2: Offline Reinforcement Learning

### Loading Historical Data

```python
from grid_fed_rl.algorithms import CQL
from grid_fed_rl.data import GridDataset
import pandas as pd

# Simulate historical grid operation data
def generate_historical_data(env, episodes=100):
    """Generate synthetic historical data."""
    states, actions, rewards, next_states, dones = [], [], [], [], []
    
    for episode in range(episodes):
        obs = env.reset()
        
        for step in range(200):  # Shorter episodes for training data
            # Use rule-based policy for historical data
            action = rule_based_policy(obs)
            next_obs, reward, done, _ = env.step(action)
            
            states.append(obs.copy())
            actions.append(action.copy()) 
            rewards.append(reward)
            next_states.append(next_obs.copy())
            dones.append(done)
            
            obs = next_obs
            if done:
                break
                
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards), 
        'next_states': np.array(next_states),
        'dones': np.array(dones)
    }

def rule_based_policy(obs):
    """Simple rule-based policy for data generation."""
    # Extract voltage magnitudes (first 13 elements for IEEE13Bus)
    voltages = obs[:13]
    action = np.zeros(3)  # Assuming 3 control actions
    
    # Voltage regulation logic
    low_voltage_buses = np.where(voltages < 0.95)[0]
    high_voltage_buses = np.where(voltages > 1.05)[0]
    
    if len(low_voltage_buses) > 0:
        action[0] = 0.1  # Increase voltage regulator
    elif len(high_voltage_buses) > 0:
        action[0] = -0.1  # Decrease voltage regulator
        
    return action

# Generate training data
historical_data = generate_historical_data(env, episodes=50)
print(f"Generated {len(historical_data['states'])} transitions")
```

### Training CQL Algorithm

```python
# Create dataset
dataset = GridDataset(
    states=historical_data['states'],
    actions=historical_data['actions'],
    rewards=historical_data['rewards'],
    next_states=historical_data['next_states'],
    dones=historical_data['dones']
)

# Initialize CQL algorithm
cql = CQL(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dims=[256, 256, 256],
    conservative_weight=5.0,
    learning_rate=3e-4
)

# Training loop
print("Training CQL policy...")
for epoch in range(100):
    batch = dataset.sample_batch(batch_size=256)
    metrics = cql.update(batch)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Q-loss={metrics['q_loss']:.3f}, "
              f"Policy-loss={metrics['policy_loss']:.3f}")

print("Training completed!")
```

### Evaluating Learned Policy

```python
def evaluate_policy(env, policy, episodes=10):
    """Evaluate policy performance."""
    episode_rewards = []
    episode_violations = []
    
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        violations = 0
        
        for step in range(1000):
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            violations += info.get('violations', 0)
            
            if done:
                break
                
        episode_rewards.append(total_reward)
        episode_violations.append(violations)
        
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_violations': np.mean(episode_violations),
        'success_rate': np.mean([v == 0 for v in episode_violations])
    }

# Compare CQL policy with baselines
cql_results = evaluate_policy(env, cql)
random_results = evaluate_policy(env, lambda obs: env.action_space.sample())

print("CQL Policy Performance:")
print(f"  Mean Reward: {cql_results['mean_reward']:.2f} ± {cql_results['std_reward']:.2f}")
print(f"  Success Rate: {cql_results['success_rate']:.1%}")

print("Random Policy Performance:")
print(f"  Mean Reward: {random_results['mean_reward']:.2f} ± {random_results['std_reward']:.2f}")
print(f"  Success Rate: {random_results['success_rate']:.1%}")
```

## Tutorial 3: Federated Learning Setup

### Multi-Utility Federated Training

```python
from grid_fed_rl.federated import FederatedOfflineRL
from grid_fed_rl.privacy import DifferentialPrivacy

# Simulate 5 different utility companies
class UtilityDataset:
    """Simulated utility dataset with local characteristics."""
    
    def __init__(self, utility_id, base_env, size=1000):
        self.utility_id = utility_id
        self.data = self._generate_utility_data(base_env, size)
        
    def _generate_utility_data(self, env, size):
        """Generate data with utility-specific characteristics."""
        # Each utility has different load patterns and preferences
        load_factor = 0.5 + 0.3 * self.utility_id  # Varied load levels
        renewable_factor = 0.1 * self.utility_id    # Different renewable penetration
        
        states, actions, rewards = [], [], []
        
        for i in range(size):
            # Modify environment for this utility's characteristics
            obs = env.reset()
            action = self._utility_policy(obs, load_factor, renewable_factor)
            next_obs, reward, done, _ = env.step(action)
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            
        return {
            'states': np.array(states),
            'actions': np.array(actions), 
            'rewards': np.array(rewards)
        }
        
    def _utility_policy(self, obs, load_factor, renewable_factor):
        """Utility-specific policy reflecting local preferences."""
        action = np.zeros(3)
        voltages = obs[:13]
        
        # Each utility has different voltage tolerance
        voltage_target = 1.0 + 0.02 * (self.utility_id - 2)  # Range 0.96-1.04
        
        voltage_error = np.mean(voltages) - voltage_target
        action[0] = -2.0 * voltage_error  # Voltage regulation
        
        # Renewable utilization preferences
        if renewable_factor > 0.3:
            action[1] = 0.1  # More aggressive renewable dispatch
            
        return np.clip(action, -1, 1)

# Create datasets for 5 utilities
utilities = [UtilityDataset(i, env) for i in range(5)]
print(f"Created datasets for {len(utilities)} utilities")

# Setup federated learning
fed_rl = FederatedOfflineRL(
    algorithm=CQL,
    num_clients=5,
    rounds=50,
    privacy_budget=10.0,  # Differential privacy epsilon
    aggregation='fedavg'
)

# Federated training
print("Starting federated training...")
global_policy = fed_rl.train(
    datasets=[u.data for u in utilities],
    local_epochs=5,
    batch_size=128
)

print("Federated training completed!")
```

### Privacy-Preserving Evaluation

```python
# Test federated policy on each utility's test environment
federated_results = {}

for i, utility in enumerate(utilities):
    # Create test environment matching utility characteristics  
    test_results = evaluate_policy(env, global_policy, episodes=5)
    federated_results[f'utility_{i}'] = test_results
    
    print(f"Utility {i} Results:")
    print(f"  Reward: {test_results['mean_reward']:.2f}")
    print(f"  Success Rate: {test_results['success_rate']:.1%}")

# Compare with individual training (privacy baseline)
individual_results = {}
for i, utility in enumerate(utilities):
    # Train individual CQL for this utility
    individual_cql = CQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    individual_cql.train(utility.data, epochs=50)
    
    results = evaluate_policy(env, individual_cql, episodes=5)
    individual_results[f'utility_{i}'] = results

# Privacy vs Performance Analysis
print("\\nFederated vs Individual Training Comparison:")
for i in range(5):
    fed_reward = federated_results[f'utility_{i}']['mean_reward'] 
    ind_reward = individual_results[f'utility_{i}']['mean_reward']
    
    print(f"Utility {i}: Fed={fed_reward:.2f}, Individual={ind_reward:.2f}, "
          f"Privacy Cost={((ind_reward-fed_reward)/ind_reward*100):.1f}%")
```

## Tutorial 4: Safety-Critical Applications

### Implementing Safety Constraints

```python
from grid_fed_rl.safety import SafetyShield, ConstraintLayer
from grid_fed_rl.controllers import BackupController

# Define safety constraints
def voltage_constraint(state, action, next_state):
    """Check voltage magnitude constraints."""
    voltages = next_state[:13]  # Extract voltage magnitudes
    return np.all((voltages >= 0.95) & (voltages <= 1.05))

def frequency_constraint(state, action, next_state):
    """Check frequency deviation constraints.""" 
    frequency = next_state[13]  # Assuming frequency is at index 13
    return abs(frequency - 60.0) <= 0.5  # ±0.5 Hz limit

def thermal_constraint(state, action, next_state):
    """Check equipment thermal limits."""
    line_loadings = next_state[14:26]  # Line loading percentages
    return np.all(line_loadings <= 0.95)  # 95% loading limit

# Create safety shield
backup_controller = BackupController()  # Rule-based backup
safety_shield = SafetyShield(
    policy=cql,  # Learned policy
    backup_controller=backup_controller,
    constraints=[voltage_constraint, frequency_constraint, thermal_constraint],
    intervention_threshold=0.9  # Intervene if 90% likely to violate
)

# Safe execution loop
def run_safe_episode(env, safe_policy, max_steps=200):
    """Run episode with safety shield active."""
    obs = env.reset()
    interventions = 0
    violations = 0
    
    for step in range(max_steps):
        action, intervened = safe_policy.get_safe_action(obs)
        
        if intervened:
            interventions += 1
            print(f"Step {step}: Safety intervention triggered")
            
        obs, reward, done, info = env.step(action)
        violations += info.get('violations', 0)
        
        if done:
            break
            
    return {
        'steps': step + 1,
        'interventions': interventions,
        'violations': violations,
        'intervention_rate': interventions / (step + 1)
    }

# Compare safe vs unsafe execution
unsafe_results = run_episode(env, cql)
safe_results = run_safe_episode(env, safety_shield)

print("Unsafe CQL Policy:")
print(f"  Violations: {unsafe_results['final_info'].get('violations', 0)}")

print("Safe CQL Policy (with shield):")  
print(f"  Violations: {safe_results['violations']}")
print(f"  Interventions: {safe_results['interventions']}")
print(f"  Intervention Rate: {safe_results['intervention_rate']:.1%}")
```

### Constraint Learning

```python
from grid_fed_rl.algorithms import ConstraintCQL

# CQL with learned safety constraints
constraint_cql = ConstraintCQL(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0], 
    constraint_functions=[voltage_constraint, frequency_constraint, thermal_constraint],
    safety_penalty=100.0,  # High penalty for constraint violations
    conservative_weight=10.0  # Extra conservative for safety
)

# Training with constraint awareness
print("Training constraint-aware CQL...")
for epoch in range(100):
    batch = dataset.sample_batch(256)
    metrics = constraint_cql.update(batch)
    
    if epoch % 20 == 0:
        constraint_violations = metrics.get('constraint_violations', 0)
        print(f"Epoch {epoch}: Q-loss={metrics['q_loss']:.3f}, "
              f"Constraints={constraint_violations}")

# Evaluation
constraint_results = evaluate_policy(env, constraint_cql, episodes=10)
print(f"Constraint-Aware CQL Success Rate: {constraint_results['success_rate']:.1%}")
```

## Tutorial 5: Performance Monitoring and Optimization

### Real-time Monitoring

```python
from grid_fed_rl.utils import PerformanceProfiler, SystemMonitor
import time

# Setup monitoring
profiler = PerformanceProfiler()
monitor = SystemMonitor(
    log_interval=10,  # Log every 10 steps
    metrics=['voltage', 'frequency', 'losses', 'violations']
)

def monitored_episode(env, policy, monitor, max_steps=100):
    """Run episode with comprehensive monitoring."""
    obs = env.reset()
    monitor.reset()
    
    step_times = []
    
    for step in range(max_steps):
        start_time = time.time()
        
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        
        step_time = time.time() - start_time
        step_times.append(step_time)
        
        # Log metrics
        monitor.log_step(step, obs, action, reward, info)
        
        if done:
            break
    
    # Performance summary
    performance_report = {
        'avg_step_time': np.mean(step_times),
        'max_step_time': np.max(step_times),
        'total_episode_time': np.sum(step_times),
        'steps_per_second': len(step_times) / np.sum(step_times)
    }
    
    return performance_report, monitor.get_episode_summary()

# Run monitored evaluation
perf_report, episode_summary = monitored_episode(env, cql, monitor)

print("Performance Report:")
print(f"  Steps per second: {perf_report['steps_per_second']:.1f}")
print(f"  Avg step time: {perf_report['avg_step_time']*1000:.2f}ms")

print("Episode Summary:")
print(f"  Avg voltage: {episode_summary['avg_voltage']:.3f} pu")
print(f"  Total violations: {episode_summary['total_violations']}")
print(f"  System losses: {episode_summary['total_losses']:.2f} kW")
```

### Optimization and Caching

```python
from grid_fed_rl.utils import PerformanceOptimizer

# Setup performance optimization
optimizer = PerformanceOptimizer(
    enable_caching=True,
    cache_size=1000,
    parallel_processing=True,
    num_workers=4
)

# Optimize environment for faster execution
optimized_env = optimizer.optimize_environment(env)

# Benchmark performance improvement
def benchmark_environments(original_env, optimized_env, episodes=5):
    """Compare performance of original vs optimized environments."""
    
    # Benchmark original
    original_times = []
    for _ in range(episodes):
        start = time.time()
        run_episode(original_env, 'random', max_steps=50)
        original_times.append(time.time() - start)
    
    # Benchmark optimized  
    optimized_times = []
    for _ in range(episodes):
        start = time.time()
        run_episode(optimized_env, 'random', max_steps=50)
        optimized_times.append(time.time() - start)
        
    speedup = np.mean(original_times) / np.mean(optimized_times)
    
    return {
        'original_time': np.mean(original_times),
        'optimized_time': np.mean(optimized_times), 
        'speedup': speedup
    }

benchmark_results = benchmark_environments(env, optimized_env)

print("Performance Optimization Results:")
print(f"  Original: {benchmark_results['original_time']:.3f}s per episode")
print(f"  Optimized: {benchmark_results['optimized_time']:.3f}s per episode")
print(f"  Speedup: {benchmark_results['speedup']:.2f}x faster")
```

This tutorial collection covers the major use cases and features of Grid-Fed-RL-Gym, from basic usage to advanced federated learning and safety-critical applications.