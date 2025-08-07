#!/usr/bin/env python3
"""
Comprehensive test of the grid environment implementation.
"""

import numpy as np
from grid_fed_rl.environments.grid_env import GridEnvironment


def test_comprehensive_grid_environment():
    """Test all major functionality of the GridEnvironment."""
    
    # Create a simple mock feeder
    class MockFeeder:
        pass

    # Test different configurations
    configs = [
        {
            'renewable_sources': [],
            'stochastic_loads': False,
            'weather_variation': False
        },
        {
            'renewable_sources': ['solar'],
            'stochastic_loads': True,
            'weather_variation': True
        },
        {
            'renewable_sources': ['solar', 'wind'],
            'stochastic_loads': True,
            'weather_variation': True
        }
    ]
    
    for i, config in enumerate(configs):
        print(f"\n=== Testing Configuration {i+1} ===")
        print(f"Config: {config}")
        
        try:
            # Create environment
            env = GridEnvironment(feeder=MockFeeder(), **config)
            print("✓ Environment created successfully")
            
            # Test spaces
            print(f"  Action space shape: {env.action_space.shape}")
            print(f"  Observation space shape: {env.observation_space.shape}")
            
            # Test reset
            obs, info = env.reset(seed=42)
            print(f"✓ Reset successful - obs shape: {obs.shape}")
            print(f"  Initial info keys: {list(info.keys())}")
            
            # Test multiple steps
            total_reward = 0
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"  Step {step+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
                
                # Check observation validity
                if np.any(~np.isfinite(obs)):
                    print(f"  WARNING: Non-finite values in observation at step {step+1}")
                
                if terminated or truncated:
                    break
                    
            print(f"✓ Multi-step test completed - total reward: {total_reward:.3f}")
            
            # Test render
            env.render()
            print("✓ Render successful")
            
        except Exception as e:
            print(f"✗ Error in configuration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n=== Testing Power Flow Solver Components ===")
    
    # Test individual components
    from grid_fed_rl.environments.power_flow import NewtonRaphsonSolver, PowerFlowSolution
    from grid_fed_rl.environments.dynamics import GridDynamics, WeatherData, TimeVaryingLoadModel, SolarPVModel
    from grid_fed_rl.environments.robust_power_flow import RobustPowerFlowSolver
    from grid_fed_rl.environments.base import Bus, Line
    
    try:
        # Test power flow solver
        solver = NewtonRaphsonSolver()
        print("✓ Newton-Raphson solver created")
        
        # Test robust solver
        robust_solver = RobustPowerFlowSolver()
        print("✓ Robust power flow solver created")
        
        # Test grid dynamics
        dynamics = GridDynamics()
        print("✓ Grid dynamics created")
        
        # Test weather data
        weather = WeatherData(
            solar_irradiance=800.0,
            wind_speed=8.0,
            temperature=25.0,
            cloud_cover=0.3
        )
        print("✓ Weather data created")
        
        # Test models
        load_model = TimeVaryingLoadModel()
        power = load_model.get_power(3600, 1e6)  # 1 hour, 1 MW base
        print(f"✓ Load model test: power = {power}")
        
        solar_model = SolarPVModel()
        solar_power = solar_model.get_power(36000, weather, 1e6)  # 10 AM, 1 MW capacity
        print(f"✓ Solar model test: power = {solar_power:.0f} W")
        
    except Exception as e:
        print(f"✗ Component test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Testing Validation Functions ===")
    
    from grid_fed_rl.utils import validate_action, validate_power_value, validate_voltage
    from grid_fed_rl.environments.base import Box
    
    try:
        # Test action validation
        action_space = Box(low=-1.0, high=1.0, shape=(3,))
        test_action = np.array([0.5, -0.5, 0.0])
        validated = validate_action(test_action, action_space)
        print(f"✓ Action validation: {validated}")
        
        # Test power validation
        power = validate_power_value(1e6)  # 1 MW
        print(f"✓ Power validation: {power}")
        
        # Test voltage validation
        voltage = validate_voltage(1.05)  # 1.05 pu
        print(f"✓ Voltage validation: {voltage}")
        
    except Exception as e:
        print(f"✗ Validation test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== All Tests Completed ===")


if __name__ == "__main__":
    test_comprehensive_grid_environment()