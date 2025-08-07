#!/usr/bin/env python3
"""Quick test of Generation 2 features."""

import sys
import numpy as np
import warnings

def test_error_handling():
    """Test robust error handling."""
    print("🔧 Testing Error Handling...")
    
    try:
        from grid_fed_rl.environments import GridEnvironment
        from grid_fed_rl.feeders import SimpleRadialFeeder
        
        env = GridEnvironment(
            feeder=SimpleRadialFeeder(num_buses=3),
            timestep=1.0,
            episode_length=5,
            safety_penalty=100.0
        )
        
        env.reset()
        
        # Test invalid actions that should be caught and penalized
        invalid_actions = [
            np.array([np.inf]),    # Infinite
            np.array([np.nan]),    # NaN
            np.array([1e10]),      # Very large
        ]
        
        for i, invalid_action in enumerate(invalid_actions):
            obs, reward, done, truncated, info = env.step(invalid_action)
            
            # Should be penalized for invalid action
            if reward < -50 and ("error" in info or "action_invalid" in info):
                print(f"   ✓ Invalid action {i} handled with penalty {reward:.1f}")
            else:
                print(f"   ✗ Invalid action {i} not properly handled: reward={reward}")
                return False
                
        return True
        
    except Exception as e:
        print(f"   ✗ Error handling test failed: {e}")
        return False


def test_ieee_feeders():
    """Test IEEE test feeders."""
    print("⚡ Testing IEEE Test Feeders...")
    
    try:
        from grid_fed_rl.feeders import IEEE13Bus, IEEE34Bus
        from grid_fed_rl.environments import GridEnvironment
        
        # Test IEEE 13-bus
        ieee13 = IEEE13Bus()
        stats = ieee13.get_network_stats()
        
        if stats["num_buses"] != 13:
            print(f"   ✗ IEEE13 should have 13 buses, got {stats['num_buses']}")
            return False
            
        validation_errors = ieee13.validate_network()
        if validation_errors:
            print(f"   ✗ IEEE13 validation failed: {validation_errors}")
            return False
            
        # Test with environment
        env = GridEnvironment(feeder=ieee13, episode_length=3)
        obs, info = env.reset()
        
        for _ in range(2):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
        print("   ✓ IEEE 13-bus feeder functional")
        
        # Test IEEE 34-bus
        ieee34 = IEEE34Bus()
        stats34 = ieee34.get_network_stats()
        
        if stats34["base_voltage_kv"] != 24.9:
            print(f"   ✗ IEEE34 voltage should be 24.9kV, got {stats34['base_voltage_kv']}")
            return False
            
        print("   ✓ IEEE 34-bus feeder functional")
        
        return True
        
    except Exception as e:
        print(f"   ✗ IEEE feeders test failed: {e}")
        return False


def test_core_robustness():
    """Test core system robustness."""
    print("🛡️ Testing Core Robustness...")
    
    try:
        from grid_fed_rl.environments import GridEnvironment
        from grid_fed_rl.feeders import SimpleRadialFeeder
        from grid_fed_rl.utils.validation import validate_action, sanitize_config
        from grid_fed_rl.utils.exceptions import InvalidActionError
        
        # Test config validation
        config = {
            'timestep': '1.5',  # String that should convert to float
            'episode_length': 100,
            'num_clients': 5
        }
        
        sanitized = sanitize_config(config, required_fields=['timestep'])
        if not isinstance(sanitized['timestep'], float):
            print("   ✗ Config sanitization failed")
            return False
            
        print("   ✓ Configuration validation works")
        
        # Test action validation
        from grid_fed_rl.environments.base import Box
        action_space = Box(low=-1.0, high=1.0, shape=(2,))
        
        # Valid action
        valid_action = np.array([0.5, -0.3])
        validated = validate_action(valid_action, action_space)
        if not np.allclose(validated, valid_action):
            print("   ✗ Valid action validation failed")
            return False
            
        # Invalid action (should raise exception)
        try:
            invalid_action = np.array([np.inf, 0.5])
            validate_action(invalid_action, action_space)
            print("   ✗ Invalid action not caught")
            return False
        except InvalidActionError:
            print("   ✓ Invalid action properly caught")
            
        # Test environment robustness
        env = GridEnvironment(
            feeder=SimpleRadialFeeder(num_buses=4),
            episode_length=10,
            safety_penalty=200.0
        )
        
        env.reset()
        
        # Multiple steps with random actions
        total_reward = 0
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
                
        print(f"   ✓ Environment completed 5 steps, total reward: {total_reward:.1f}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Core robustness test failed: {e}")
        return False


def test_power_flow_robustness():
    """Test power flow solver robustness."""
    print("🔋 Testing Power Flow Robustness...")
    
    try:
        from grid_fed_rl.environments.robust_power_flow import RobustPowerFlowSolver
        from grid_fed_rl.environments.base import Bus, Line
        
        # Create test network
        buses = [
            Bus(id=1, voltage_level=4160, bus_type="slack"),
            Bus(id=2, voltage_level=4160, bus_type="pq"),
            Bus(id=3, voltage_level=4160, bus_type="pq")
        ]
        
        lines = [
            Line(id="line_1_2", from_bus=1, to_bus=2, resistance=0.01, reactance=0.02, rating=1e6),
            Line(id="line_2_3", from_bus=2, to_bus=3, resistance=0.015, reactance=0.025, rating=1e6)
        ]
        
        solver = RobustPowerFlowSolver()
        
        # Normal case
        loads = {2: 500e3, 3: 300e3}  # 500kW, 300kW loads
        generation = {1: 0}  # Slack bus provides power
        
        solution = solver.solve(buses, lines, loads, generation)
        
        if not solution.converged:
            print("   ✗ Normal power flow should converge")
            return False
            
        print(f"   ✓ Normal power flow converged in {solution.iterations} iterations")
        
        # Challenging case - high load
        loads = {2: 5e6, 3: 5e6}  # 5MW each - high loading
        solution = solver.solve(buses, lines, loads, generation)
        
        # Should either converge or gracefully handle non-convergence
        if solution.converged:
            print(f"   ✓ High load case converged")
        else:
            print(f"   ✓ High load case handled gracefully (non-convergent)")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Power flow robustness test failed: {e}")
        return False


def main():
    """Run Generation 2 quick tests."""
    print("🔧 Generation 2: Quick Robustness Test")
    print("=" * 45)
    
    # Suppress numpy warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        tests = [
            test_error_handling,
            test_ieee_feeders,
            test_core_robustness,
            test_power_flow_robustness
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                print()
            except Exception as e:
                print(f"   ✗ Test crashed: {e}")
                results.append(False)
                print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("🏁 Generation 2 Quick Test Results")
    print("=" * 35)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All robustness tests passed!")
        print("\nGeneration 2 robust features verified:")
        print("  ✅ Error handling and input validation")
        print("  ✅ IEEE standard test feeders")  
        print("  ✅ Configuration management")
        print("  ✅ Power flow solver robustness")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)