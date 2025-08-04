#!/usr/bin/env python3
"""Test grid environment functionality."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_environment_creation():
    """Test creating and running a basic environment."""
    print("Testing environment creation...")
    
    try:
        from grid_fed_rl.environments import GridEnvironment
        from grid_fed_rl.feeders import IEEE13Bus
        
        # Create IEEE 13-bus feeder
        feeder = IEEE13Bus()
        print(f"✓ IEEE13Bus feeder created with {len(feeder.buses)} buses")
        
        # Create environment
        env = GridEnvironment(
            feeder=feeder,
            timestep=1.0,
            episode_length=100,  # Short episode for testing
            stochastic_loads=False,  # Deterministic for testing
            renewable_sources=[],  # No renewables for simplicity
            weather_variation=False
        )
        print(f"✓ Environment created")
        print(f"   - Observation space: {env.observation_space.shape}")
        print(f"   - Action space: {env.action_space.shape}")
        
        return env
    except Exception as e:
        print(f"✗ Environment creation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_environment_steps(env):
    """Test running environment steps."""
    print("\nTesting environment steps...")
    
    try:
        # Reset environment
        obs, info = env.reset(seed=42)
        print(f\"✓ Environment reset, observation shape: {obs.shape}\")
        
        # Run a few steps
        total_reward = 0
        for step in range(10):
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                print(f\"   Step {step}: reward={reward:.3f}, total={total_reward:.3f}\")
            
            if terminated or truncated:
                print(f\"   Episode ended at step {step}\")
                break
                
            obs = next_obs
        
        print(f\"✓ Completed {step + 1} steps, total reward: {total_reward:.3f}\")
        return True
    except Exception as e:
        print(f\"✗ Environment step error: {e}\")
        import traceback
        traceback.print_exc()
        return False

def test_power_flow():
    \"\"\"Test power flow solver.\"\"\"
    print(\"\\nTesting power flow solver...\")
    
    try:
        from grid_fed_rl.environments.power_flow import NewtonRaphsonSolver
        from grid_fed_rl.feeders import SimpleRadialFeeder
        
        # Create simple feeder
        feeder = SimpleRadialFeeder(num_buses=3)
        
        # Create solver
        solver = NewtonRaphsonSolver(tolerance=1e-6, max_iterations=50)
        
        # Simple load/generation scenario
        loads = {2: 1e6, 3: 0.5e6}  # 1MW at bus 2, 0.5MW at bus 3
        generation = {1: 2e6}  # 2MW generation at slack bus
        
        # Solve power flow
        solution = solver.solve(feeder.buses, feeder.lines, loads, generation)
        
        print(f\"✓ Power flow solved in {solution.iterations} iterations\")
        print(f\"   - Converged: {solution.converged}\")
        print(f\"   - Losses: {solution.losses/1e3:.1f} kW\")
        print(f\"   - Voltage range: {solution.bus_voltages.min():.3f} - {solution.bus_voltages.max():.3f} pu\")
        
        return solution.converged
    except Exception as e:
        print(f\"✗ Power flow error: {e}\")
        import traceback
        traceback.print_exc()
        return False

def test_dynamics():
    \"\"\"Test grid dynamics models.\"\"\"
    print(\"\\nTesting dynamics models...\")
    
    try:
        from grid_fed_rl.environments.dynamics import GridDynamics, WeatherData, TimeVaryingLoadModel
        
        # Create dynamics
        dynamics = GridDynamics(base_frequency=60.0)
        
        # Add load model
        load_model = TimeVaryingLoadModel(noise_level=0.1)
        dynamics.add_load_model(\"load_1\", load_model)
        
        # Test load power calculation
        weather = WeatherData(solar_irradiance=800, wind_speed=8.0, temperature=25.0, cloud_cover=0.2)
        
        active_power, reactive_power = dynamics.get_load_power(\"load_1\", 1e6, time=43200)  # Noon
        print(f\"✓ Load power calculated: {active_power/1e3:.1f} kW active, {reactive_power/1e3:.1f} kVAR reactive\")
        
        # Test frequency dynamics
        initial_freq = dynamics.frequency
        dynamics.update_frequency(power_imbalance=0.1, timestep=1.0)  # 100 kW imbalance
        freq_change = dynamics.frequency - initial_freq
        print(f\"✓ Frequency dynamics: {initial_freq:.3f} Hz -> {dynamics.frequency:.3f} Hz (Δ{freq_change:.3f} Hz)\")
        
        return True
    except Exception as e:
        print(f\"✗ Dynamics error: {e}\")
        import traceback
        traceback.print_exc()
        return False

def main():
    \"\"\"Run environment tests.\"\"\"
    print(\"Grid-Fed-RL-Gym Environment Test\")
    print(\"=================================\")
    
    tests = []
    
    # Test environment creation
    env = test_environment_creation()
    if env is not None:
        tests.append(True)
        
        # Test environment steps
        tests.append(test_environment_steps(env))
    else:
        tests.append(False)
        tests.append(False)
    
    # Test power flow
    tests.append(test_power_flow())
    
    # Test dynamics
    tests.append(test_dynamics())
    
    passed = sum(tests)
    total = len(tests)
    
    print(f\"\\nResults: {passed}/{total} tests passed\")
    
    if passed == total:
        print(\"✓ All environment tests passed! System is working correctly.\")
        return 0
    else:
        print(\"✗ Some tests failed.\")
        return 1

if __name__ == \"__main__\":
    sys.exit(main())