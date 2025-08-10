# Grid-Fed-RL-Gym Examples

This document provides practical, runnable examples for common use cases.

## Example 1: Simple Grid Control

```python
#!/usr/bin/env python3
"""Simple grid voltage control example."""

import numpy as np
import matplotlib.pyplot as plt
from grid_fed_rl import GridEnvironment
from grid_fed_rl.feeders import IEEE13Bus

def main():
    # Create environment
    env = GridEnvironment(
        feeder=IEEE13Bus(),
        timestep=1.0,
        episode_length=200,
        stochastic_loads=True
    )
    
    print(f"Environment created with {env.feeder.num_buses} buses")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.shape}")
    
    # Simple proportional controller
    def voltage_controller(obs):
        """Simple P controller for voltage regulation."""
        voltages = obs[:13]  # Bus voltages
        target_voltage = 1.0
        
        # Calculate average voltage error
        voltage_error = np.mean(voltages) - target_voltage
        
        # Proportional control action
        action = np.zeros(env.action_space.shape[0])
        action[0] = -2.0 * voltage_error  # Voltage regulator
        
        return np.clip(action, -1, 1)
    
    # Run episode
    obs = env.reset()
    rewards = []
    voltages = []
    actions = []
    
    for step in range(200):
        action = voltage_controller(obs)
        obs, reward, done, info = env.step(action)
        
        rewards.append(reward)
        voltages.append(np.mean(obs[:13]))  # Average voltage
        actions.append(action[0])
        
        if done:
            print(f"Episode terminated at step {step}")
            break
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Reward vs Time')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(3, 1, 2) 
    plt.plot(voltages)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Target')
    plt.axhline(y=0.95, color='orange', linestyle='--', label='Min')
    plt.axhline(y=1.05, color='orange', linestyle='--', label='Max')
    plt.title('Average Bus Voltage')
    plt.ylabel('Voltage (pu)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(actions)
    plt.title('Control Actions')
    plt.xlabel('Time Step')
    plt.ylabel('Action')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('voltage_control_results.png')
    plt.show()
    
    print(f"Final average voltage: {voltages[-1]:.4f} pu")
    print(f"Total reward: {sum(rewards):.2f}")

if __name__ == "__main__":
    main()
```

## Example 2: Offline RL Training

```python
#!/usr/bin/env python3
"""Offline reinforcement learning training example."""

import numpy as np
import pickle
from grid_fed_rl import GridEnvironment
from grid_fed_rl.feeders import IEEE13Bus
from grid_fed_rl.algorithms import CQL
from grid_fed_rl.data import GridDataset

def generate_expert_data(env, num_episodes=100):
    """Generate expert demonstration data."""
    
    def expert_policy(obs):
        """Expert policy based on optimal power flow principles."""
        voltages = obs[:13]
        line_flows = obs[13:25] if len(obs) > 25 else obs[13:]
        
        action = np.zeros(env.action_space.shape[0])
        
        # Voltage control - maintain 1.0 pu
        voltage_violations = np.sum((voltages < 0.95) | (voltages > 1.05))
        if voltage_violations > 0:
            avg_voltage = np.mean(voltages)
            action[0] = 2.0 * (1.0 - avg_voltage)  # Voltage regulator
        
        # Reactive power control
        if len(action) > 1:
            # Switch capacitor if low voltages
            low_voltage_count = np.sum(voltages < 0.98)
            if low_voltage_count > 2:
                action[1] = 0.5  # Switch in capacitor
            elif np.sum(voltages > 1.02) > 2:
                action[1] = -0.5  # Switch out capacitor
        
        return np.clip(action, -1, 1)
    
    trajectories = []
    
    for episode in range(num_episodes):
        print(f"\\rGenerating episode {episode+1}/{num_episodes}", end="")
        
        obs = env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': []
        }
        
        for step in range(100):  # Shorter episodes for training
            action = expert_policy(obs)
            next_obs, reward, done, _ = env.step(action)
            
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action.copy())
            trajectory['rewards'].append(reward)
            trajectory['next_observations'].append(next_obs.copy())
            trajectory['terminals'].append(done)
            
            obs = next_obs
            if done:
                break
        
        trajectories.append(trajectory)
    
    print("\\nData generation complete!")
    return trajectories

def train_cql_agent(dataset, env):
    """Train CQL agent on offline dataset."""
    
    # Initialize CQL
    cql = CQL(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dims=[256, 256],
        conservative_weight=5.0,
        learning_rate=3e-4,
        batch_size=256
    )
    
    print("Starting CQL training...")
    
    # Training loop
    for epoch in range(200):
        # Sample batch from dataset
        batch = dataset.sample_batch(256)
        
        # Update CQL
        metrics = cql.update(batch)
        
        # Log progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/200:")
            print(f"  Q-loss: {metrics['q_loss']:.4f}")
            print(f"  Policy loss: {metrics['policy_loss']:.4f}")
            print(f"  Conservative loss: {metrics['conservative_loss']:.4f}")
    
    print("Training complete!")
    return cql

def evaluate_policy(env, policy, num_episodes=10):
    """Evaluate trained policy."""
    episode_rewards = []
    episode_lengths = []
    violation_counts = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        violations = 0
        
        for step in range(200):
            action = policy.get_action(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            violations += info.get('violations', 0)
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)
        violation_counts.append(violations)
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_violations': np.mean(violation_counts),
        'success_rate': np.mean([v == 0 for v in violation_counts])
    }
    
    return results

def main():
    # Create environment
    env = GridEnvironment(
        feeder=IEEE13Bus(),
        timestep=1.0,
        episode_length=200
    )
    
    # Generate expert data
    print("Generating expert demonstration data...")
    trajectories = generate_expert_data(env, num_episodes=50)
    
    # Create dataset
    dataset = GridDataset.from_trajectories(trajectories)
    print(f"Dataset created with {len(dataset)} transitions")
    
    # Save dataset
    with open('expert_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print("Dataset saved to expert_dataset.pkl")
    
    # Train CQL agent
    cql_agent = train_cql_agent(dataset, env)
    
    # Save trained agent
    cql_agent.save('cql_agent.pkl')
    print("Trained agent saved to cql_agent.pkl")
    
    # Evaluate performance
    print("\\nEvaluating trained CQL agent...")
    cql_results = evaluate_policy(env, cql_agent)
    
    print("\\nEvaluation Results:")
    print(f"Mean Reward: {cql_results['mean_reward']:.2f} ± {cql_results['std_reward']:.2f}")
    print(f"Mean Episode Length: {cql_results['mean_length']:.1f}")
    print(f"Success Rate: {cql_results['success_rate']:.1%} (no violations)")
    print(f"Mean Violations per Episode: {cql_results['mean_violations']:.2f}")
    
    # Compare with random policy
    print("\\nComparing with random baseline...")
    random_results = evaluate_policy(env, lambda obs: env.action_space.sample())
    
    print("Random Policy Results:")
    print(f"Mean Reward: {random_results['mean_reward']:.2f} ± {random_results['std_reward']:.2f}")
    print(f"Success Rate: {random_results['success_rate']:.1%}")
    
    improvement = (cql_results['mean_reward'] - random_results['mean_reward']) / abs(random_results['mean_reward']) * 100
    print(f"\\nCQL improves reward by {improvement:.1f}% over random policy")

if __name__ == "__main__":
    main()
```

## Example 3: Federated Learning

```python
#!/usr/bin/env python3
"""Federated learning across multiple utilities example."""

import numpy as np
from grid_fed_rl import GridEnvironment
from grid_fed_rl.feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus
from grid_fed_rl.federated import FederatedOfflineRL
from grid_fed_rl.algorithms import CQL
from grid_fed_rl.privacy import DifferentialPrivacy

class UtilitySimulator:
    """Simulates different utility company characteristics."""
    
    def __init__(self, utility_id, feeder_type='IEEE13Bus', load_profile='residential'):
        self.utility_id = utility_id
        self.load_profile = load_profile
        self.privacy_budget = 5.0 + utility_id * 2.0  # Varying privacy requirements
        
        # Create environment based on utility size
        if feeder_type == 'IEEE13Bus':
            self.env = GridEnvironment(feeder=IEEE13Bus())
        elif feeder_type == 'IEEE34Bus': 
            self.env = GridEnvironment(feeder=IEEE34Bus())
        else:
            self.env = GridEnvironment(feeder=IEEE123Bus())
    
    def generate_local_data(self, episodes=20):
        """Generate data reflecting this utility's operational patterns."""
        
        def utility_policy(obs):
            """Utility-specific operational policy."""
            # Each utility has different objectives and constraints
            action = np.zeros(self.env.action_space.shape[0])
            
            voltages = obs[:len(obs)//2]  # Approximate voltage readings
            
            # Utility-specific voltage preferences
            if self.load_profile == 'residential':
                target_voltage = 1.02  # Slightly higher for residential
            elif self.load_profile == 'industrial': 
                target_voltage = 0.98  # Lower for industrial efficiency
            else:
                target_voltage = 1.00  # Standard commercial
            
            # Different risk tolerances
            risk_factor = 0.5 + 0.3 * (self.utility_id / 5)  # 0.5 to 0.8
            voltage_error = np.mean(voltages) - target_voltage
            action[0] = -risk_factor * 3.0 * voltage_error
            
            return np.clip(action, -1, 1)
        
        trajectories = []
        
        for episode in range(episodes):
            obs = self.env.reset()
            trajectory = []
            
            for step in range(50):  # Short episodes for FL
                action = utility_policy(obs)
                next_obs, reward, done, _ = self.env.step(action)
                
                trajectory.append({
                    'state': obs.copy(),
                    'action': action.copy(),
                    'reward': reward,
                    'next_state': next_obs.copy(),
                    'done': done
                })
                
                obs = next_obs
                if done:
                    break
            
            trajectories.append(trajectory)
        
        return trajectories

def setup_utilities():
    """Create diverse set of utility companies."""
    utilities = [
        UtilitySimulator(0, 'IEEE13Bus', 'residential'),
        UtilitySimulator(1, 'IEEE13Bus', 'industrial'), 
        UtilitySimulator(2, 'IEEE34Bus', 'commercial'),
        UtilitySimulator(3, 'IEEE34Bus', 'mixed'),
        UtilitySimulator(4, 'IEEE123Bus', 'urban')
    ]
    
    return utilities

def federated_training_example():
    """Main federated learning workflow."""
    
    print("Setting up federated learning simulation...")
    print("=" * 50)
    
    # Create utilities
    utilities = setup_utilities()
    print(f"Created {len(utilities)} utility participants")
    
    # Generate local data for each utility
    print("\\nGenerating local datasets...")
    utility_datasets = {}
    
    for i, utility in enumerate(utilities):
        print(f"  Utility {i}: {utility.load_profile} profile, "
              f"Privacy budget: {utility.privacy_budget}")
        
        trajectories = utility.generate_local_data(episodes=25)
        utility_datasets[i] = trajectories
    
    # Setup federated learning coordinator
    fed_rl = FederatedOfflineRL(
        algorithm=CQL,
        num_clients=len(utilities),
        rounds=30,
        privacy_budget=8.0,  # Global privacy budget
        aggregation='fedavg',
        secure_aggregation=True
    )
    
    # Configure differential privacy
    privacy_mechanism = DifferentialPrivacy(
        epsilon=8.0,
        delta=1e-5,
        mechanism='gaussian'
    )
    
    print("\\nStarting federated training...")
    print(f"Rounds: {30}, Privacy: ε={8.0}, δ={1e-5}")
    
    # Federated training rounds
    global_policy = None
    
    for round_num in range(30):
        print(f"\\nRound {round_num + 1}/30")
        
        # Each utility trains locally
        local_updates = []
        
        for utility_id, trajectories in utility_datasets.items():
            utility = utilities[utility_id]
            
            # Create local CQL agent
            local_cql = CQL(
                state_dim=utility.env.observation_space.shape[0],
                action_dim=utility.env.action_space.shape[0],
                hidden_dims=[128, 128]  # Smaller for FL
            )
            
            # Load global model if available
            if global_policy is not None:
                local_cql.load_weights(global_policy.get_weights())
            
            # Local training
            for epoch in range(5):  # Few local epochs
                # Sample from local data
                batch = sample_trajectories(trajectories, batch_size=64)
                local_cql.update(batch)
            
            # Add differential privacy noise
            local_weights = local_cql.get_weights()
            noisy_weights = privacy_mechanism.add_noise(
                local_weights, 
                sensitivity=1.0,
                epsilon=utility.privacy_budget / 30  # Per-round budget
            )
            
            local_updates.append(noisy_weights)
        
        # Secure aggregation
        global_weights = fed_rl.aggregate(local_updates)
        
        # Update global policy
        if global_policy is None:
            global_policy = CQL(
                state_dim=utilities[0].env.observation_space.shape[0],
                action_dim=utilities[0].env.action_space.shape[0],
                hidden_dims=[128, 128]
            )
        
        global_policy.load_weights(global_weights)
        
        # Evaluate global policy
        if (round_num + 1) % 10 == 0:
            avg_performance = evaluate_global_policy(global_policy, utilities)
            print(f"  Global policy performance: {avg_performance:.3f}")
    
    print("\\nFederated training complete!")
    
    # Final evaluation
    print("\\nFinal Evaluation:")
    print("=" * 30)
    
    for i, utility in enumerate(utilities):
        performance = evaluate_policy_on_utility(global_policy, utility)
        print(f"Utility {i} ({utility.load_profile}):")
        print(f"  Reward: {performance['reward']:.2f}")
        print(f"  Success Rate: {performance['success_rate']:.1%}")
    
    # Compare with individual training
    print("\\nComparison with Individual Training:")
    individual_results = []
    
    for i, utility in enumerate(utilities):
        individual_cql = train_individual_policy(utility, utility_datasets[i])
        performance = evaluate_policy_on_utility(individual_cql, utility)
        individual_results.append(performance['reward'])
        print(f"Utility {i} Individual: {performance['reward']:.2f}")
    
    # Privacy-utility tradeoff analysis
    federated_avg = np.mean([evaluate_policy_on_utility(global_policy, u)['reward'] 
                            for u in utilities])
    individual_avg = np.mean(individual_results)
    
    privacy_cost = (individual_avg - federated_avg) / individual_avg * 100
    
    print(f"\\nPrivacy-Utility Analysis:")
    print(f"Individual Training Avg: {individual_avg:.2f}")
    print(f"Federated Learning Avg: {federated_avg:.2f}")  
    print(f"Privacy Cost: {privacy_cost:.1f}% performance reduction")
    
    return global_policy

def sample_trajectories(trajectories, batch_size=64):
    """Sample batch from trajectory data."""
    all_transitions = []
    for traj in trajectories:
        all_transitions.extend(traj)
    
    batch_indices = np.random.choice(len(all_transitions), batch_size, replace=True)
    batch = [all_transitions[i] for i in batch_indices]
    
    return {
        'states': np.array([t['state'] for t in batch]),
        'actions': np.array([t['action'] for t in batch]),
        'rewards': np.array([t['reward'] for t in batch]),
        'next_states': np.array([t['next_state'] for t in batch]),
        'dones': np.array([t['done'] for t in batch])
    }

def evaluate_policy_on_utility(policy, utility, episodes=5):
    """Evaluate policy on specific utility environment."""
    rewards = []
    violations = []
    
    for _ in range(episodes):
        obs = utility.env.reset()
        episode_reward = 0
        episode_violations = 0
        
        for _ in range(100):
            action = policy.get_action(obs, deterministic=True)
            obs, reward, done, info = utility.env.step(action)
            episode_reward += reward
            episode_violations += info.get('violations', 0)
            
            if done:
                break
        
        rewards.append(episode_reward)
        violations.append(episode_violations)
    
    return {
        'reward': np.mean(rewards),
        'success_rate': np.mean([v == 0 for v in violations])
    }

def evaluate_global_policy(policy, utilities):
    """Evaluate global policy across all utilities."""
    total_performance = 0
    
    for utility in utilities:
        performance = evaluate_policy_on_utility(policy, utility, episodes=3)
        total_performance += performance['reward']
    
    return total_performance / len(utilities)

def train_individual_policy(utility, trajectories):
    """Train individual policy for comparison."""
    individual_cql = CQL(
        state_dim=utility.env.observation_space.shape[0],
        action_dim=utility.env.action_space.shape[0],
        hidden_dims=[128, 128]
    )
    
    # Train for same total updates as federated
    for epoch in range(150):  # 30 rounds * 5 epochs
        batch = sample_trajectories(trajectories, batch_size=64)
        individual_cql.update(batch)
    
    return individual_cql

if __name__ == "__main__":
    global_policy = federated_training_example()
    
    # Save final global policy
    global_policy.save('federated_global_policy.pkl')
    print("\\nGlobal policy saved to federated_global_policy.pkl")
```

## Example 4: Safety-Critical Control

```python
#!/usr/bin/env python3
"""Safety-critical grid control with constraint enforcement."""

import numpy as np
import matplotlib.pyplot as plt
from grid_fed_rl import GridEnvironment  
from grid_fed_rl.feeders import IEEE13Bus
from grid_fed_rl.safety import SafetyShield, ConstraintMonitor
from grid_fed_rl.algorithms import CQL
import warnings

class SafeGridController:
    """Safety-critical grid controller with multiple protection layers."""
    
    def __init__(self, env):
        self.env = env
        self.safety_shield = None
        self.constraint_monitor = ConstraintMonitor()
        self.emergency_actions_taken = 0
        self.constraint_violations = []
        
    def add_safety_constraints(self):
        """Define and add safety constraints."""
        
        def voltage_magnitude_constraint(state, action, next_state):
            """Voltage magnitude must stay within ANSI C84.1 limits."""
            voltages = next_state[:13]  # Bus voltages
            return np.all((voltages >= 0.95) & (voltages <= 1.05))
        
        def voltage_deviation_constraint(state, action, next_state):
            """Voltage deviation between adjacent buses."""
            voltages = next_state[:13]
            max_deviation = np.max(np.abs(np.diff(voltages)))
            return max_deviation <= 0.03  # 3% max deviation
        
        def frequency_constraint(state, action, next_state):
            """Grid frequency must stay within operational limits.""" 
            # Assuming frequency is encoded in state
            if len(next_state) > 13:
                frequency = next_state[13]  # System frequency
                return abs(frequency - 1.0) <= 0.008  # ±0.5 Hz normalized
            return True
        
        def thermal_constraint(state, action, next_state):
            """Equipment thermal loading limits."""
            if len(next_state) > 14:
                line_loadings = next_state[14:26]  # Line thermal loadings
                return np.all(line_loadings <= 0.95)  # 95% thermal limit
            return True
        
        def rate_of_change_constraint(state, action, next_state):
            """Limit rate of change in control actions."""
            if hasattr(self, 'previous_action'):
                action_change = np.abs(action - self.previous_action)
                return np.all(action_change <= 0.1)  # 10% max change per step
            return True
        
        # Create safety shield with all constraints
        constraints = [
            voltage_magnitude_constraint,
            voltage_deviation_constraint, 
            frequency_constraint,
            thermal_constraint,
            rate_of_change_constraint
        ]
        
        self.safety_shield = SafetyShield(
            constraints=constraints,
            intervention_threshold=0.95,  # Intervene if 95% likely to violate
            backup_controller=self.emergency_controller
        )
        
        print(f"Safety shield created with {len(constraints)} constraints")
    
    def emergency_controller(self, state):
        """Emergency backup controller for safety violations."""
        self.emergency_actions_taken += 1
        
        # Conservative emergency actions
        action = np.zeros(self.env.action_space.shape[0])
        
        voltages = state[:13]
        
        # Emergency voltage regulation
        avg_voltage = np.mean(voltages)
        if avg_voltage < 0.95:
            action[0] = 0.2  # Emergency voltage boost
        elif avg_voltage > 1.05:
            action[0] = -0.2  # Emergency voltage reduction
        
        # Switch off non-critical loads if severe violations
        critical_violations = np.sum((voltages < 0.90) | (voltages > 1.10))
        if critical_violations > 0:
            if len(action) > 1:
                action[1] = -0.5  # Load shedding
        
        print(f"EMERGENCY ACTION: {action}")
        return action
    
    def get_safe_action(self, state, base_policy):
        """Get safety-verified action from base policy."""
        
        # Get base policy action
        base_action = base_policy(state)
        
        # Apply safety shield
        if self.safety_shield is not None:
            safe_action, intervened = self.safety_shield.filter_action(
                state, base_action
            )
            
            if intervened:
                print(f"Safety intervention: {base_action} -> {safe_action}")
            
            # Store for rate-of-change constraint
            self.previous_action = safe_action.copy()
            
            return safe_action, intervened
        
        return base_action, False

def create_aggressive_policy():
    """Create an aggressive policy that may violate constraints."""
    
    def aggressive_policy(state):
        """Aggressive control policy for testing safety systems."""
        action = np.zeros(3)  # Assuming 3 control actions
        
        voltages = state[:13]
        
        # Aggressive voltage control - may cause overshoot
        voltage_error = np.mean(voltages) - 1.0
        action[0] = -5.0 * voltage_error  # High gain, may cause instability
        
        # Aggressive reactive power dispatch
        low_voltage_buses = np.sum(voltages < 1.0)
        if low_voltage_buses > 3:
            action[1] = 0.8  # Large capacitor switching
        
        # Random component to test safety systems
        action[2] = 0.3 * np.random.randn()  # Random action
        
        return np.clip(action, -1, 1)
    
    return aggressive_policy

def run_safety_test(env, controller, policy, episodes=5):
    """Run safety test comparing safe vs unsafe execution."""
    
    results = {
        'safe': {'violations': [], 'rewards': [], 'interventions': []},
        'unsafe': {'violations': [], 'rewards': []}
    }
    
    print("Running safety tests...")
    
    for episode in range(episodes):
        print(f"\\nEpisode {episode + 1}/{episodes}")
        
        # Test 1: Unsafe execution (no safety shield)
        print("  Testing unsafe execution...")
        obs = env.reset()
        unsafe_violations = 0
        unsafe_reward = 0
        
        for step in range(100):
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            unsafe_reward += reward
            unsafe_violations += info.get('violations', 0)
            
            if done:
                break
        
        results['unsafe']['violations'].append(unsafe_violations)
        results['unsafe']['rewards'].append(unsafe_reward)
        
        # Test 2: Safe execution (with safety shield)
        print("  Testing safe execution...")
        obs = env.reset()
        safe_violations = 0
        safe_reward = 0
        interventions = 0
        
        for step in range(100):
            action, intervened = controller.get_safe_action(obs, policy)
            if intervened:
                interventions += 1
                
            obs, reward, done, info = env.step(action) 
            safe_reward += reward
            safe_violations += info.get('violations', 0)
            
            if done:
                break
        
        results['safe']['violations'].append(safe_violations)
        results['safe']['rewards'].append(safe_reward)
        results['safe']['interventions'].append(interventions)
    
    return results

def analyze_safety_results(results):
    """Analyze and display safety test results."""
    
    print("\\n" + "="*50)
    print("SAFETY ANALYSIS RESULTS")
    print("="*50)
    
    unsafe_violations = np.array(results['unsafe']['violations'])
    safe_violations = np.array(results['safe']['violations'])
    interventions = np.array(results['safe']['interventions'])
    
    unsafe_rewards = np.array(results['unsafe']['rewards'])
    safe_rewards = np.array(results['safe']['rewards'])
    
    print("\\nVIOLATION ANALYSIS:")
    print(f"Unsafe Policy:")
    print(f"  Mean violations per episode: {np.mean(unsafe_violations):.1f}")
    print(f"  Max violations in episode: {np.max(unsafe_violations)}")
    print(f"  Episodes with violations: {np.sum(unsafe_violations > 0)}/{len(unsafe_violations)}")
    
    print(f"\\nSafe Policy (with shield):")
    print(f"  Mean violations per episode: {np.mean(safe_violations):.1f}")
    print(f"  Max violations in episode: {np.max(safe_violations)}")
    print(f"  Episodes with violations: {np.sum(safe_violations > 0)}/{len(safe_violations)}")
    
    violation_reduction = (1 - np.mean(safe_violations) / np.mean(unsafe_violations)) * 100
    print(f"  Violation reduction: {violation_reduction:.1f}%")
    
    print("\\nINTERVENTION ANALYSIS:")
    print(f"  Mean interventions per episode: {np.mean(interventions):.1f}")
    print(f"  Intervention rate: {np.mean(interventions)/100*100:.1f}% of actions")
    
    print("\\nPERFORMANCE ANALYSIS:")
    print(f"Unsafe Policy Mean Reward: {np.mean(unsafe_rewards):.2f}")
    print(f"Safe Policy Mean Reward: {np.mean(safe_rewards):.2f}")
    
    if np.mean(unsafe_rewards) != 0:
        performance_cost = (1 - np.mean(safe_rewards) / np.mean(unsafe_rewards)) * 100
        print(f"Performance cost of safety: {performance_cost:.1f}%")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.bar(['Unsafe', 'Safe'], [np.mean(unsafe_violations), np.mean(safe_violations)])
    plt.title('Mean Violations per Episode')
    plt.ylabel('Violations')
    
    plt.subplot(2, 3, 2)
    plt.bar(['Unsafe', 'Safe'], [np.mean(unsafe_rewards), np.mean(safe_rewards)])
    plt.title('Mean Episode Reward')
    plt.ylabel('Reward')
    
    plt.subplot(2, 3, 3)
    plt.hist(interventions, bins=10, alpha=0.7)
    plt.title('Distribution of Interventions')
    plt.xlabel('Interventions per Episode')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 3, 4)
    episodes = range(1, len(unsafe_violations) + 1)
    plt.plot(episodes, unsafe_violations, 'r-o', label='Unsafe', markersize=4)
    plt.plot(episodes, safe_violations, 'g-s', label='Safe', markersize=4)
    plt.title('Violations Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Violations')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(episodes, unsafe_rewards, 'r-o', label='Unsafe', markersize=4)
    plt.plot(episodes, safe_rewards, 'g-s', label='Safe', markersize=4)
    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    plt.subplot(2, 3, 6)
    safety_score = 1 - (safe_violations / np.maximum(unsafe_violations, 1))
    plt.bar(episodes, safety_score)
    plt.title('Safety Improvement Score')
    plt.xlabel('Episode')
    plt.ylabel('Safety Score (0-1)')
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('safety_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main safety-critical control demonstration."""
    
    print("Grid Safety-Critical Control Demo")
    print("=" * 40)
    
    # Create environment
    env = GridEnvironment(
        feeder=IEEE13Bus(),
        timestep=1.0,
        episode_length=100
    )
    
    # Create safety controller
    controller = SafeGridController(env)
    controller.add_safety_constraints()
    
    # Create aggressive test policy
    aggressive_policy = create_aggressive_policy()
    
    print("\\nTesting aggressive policy against safety constraints...")
    
    # Run safety tests
    results = run_safety_test(env, controller, aggressive_policy, episodes=10)
    
    # Analyze results
    analyze_safety_results(results)
    
    print(f"\\nEmergency actions taken: {controller.emergency_actions_taken}")
    
    # Test with trained RL policy if available
    try:
        print("\\nTesting with trained CQL policy...")
        cql_policy = CQL.load('cql_agent.pkl')  # From previous example
        
        def cql_policy_fn(obs):
            return cql_policy.get_action(obs, deterministic=True)
        
        cql_results = run_safety_test(env, controller, cql_policy_fn, episodes=5)
        print("CQL Policy Safety Test:")
        analyze_safety_results(cql_results)
        
    except FileNotFoundError:
        print("No trained CQL policy found. Run offline RL example first.")
    
    print("\\nSafety-critical control demonstration complete!")

if __name__ == "__main__":
    main()
```

These examples provide comprehensive, runnable code for the main use cases of Grid-Fed-RL-Gym, from basic control to advanced safety-critical applications.