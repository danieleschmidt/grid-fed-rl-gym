#!/usr/bin/env python3
"""
Generation 1 Basic Functionality Test
Tests that core environment works with simple functionality
"""

import numpy as np
import sys
import traceback

def test_basic_imports():
    """Test that core modules can be imported."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.environments.base import BaseGridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        print("✅ Core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_environment_creation():
    """Test that GridEnvironment can be created."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Create simple feeder
        feeder = IEEE13Bus()
        
        # Create environment
        env = GridEnvironment(
            feeder=feeder,
            timestep=1.0,
            episode_length=100,
            stochastic_loads=True,
            renewable_sources=["solar", "wind"]
        )
        
        print("✅ Environment creation successful")
        print(f"   Action space shape: {env.action_space.shape}")
        print(f"   Observation space shape: {env.observation_space.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        traceback.print_exc()
        return False

def test_environment_reset():
    """Test environment reset functionality."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=100)
        
        obs, info = env.reset(seed=42)
        
        print("✅ Environment reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        traceback.print_exc()
        return False

def test_environment_step():
    """Test single environment step."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=100)
        
        obs, info = env.reset(seed=42)
        
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✅ Environment step successful")
        print(f"   Reward: {reward:.2f}")
        print(f"   Terminated: {terminated}")
        print(f"   Power flow converged: {info.get('power_flow_converged', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        traceback.print_exc()
        return False

def test_simple_episode():
    """Test running a short episode."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(
            feeder=feeder, 
            timestep=1.0, 
            episode_length=10,  # Very short episode
            stochastic_loads=False  # Deterministic for consistency
        )
        
        obs, info = env.reset(seed=42)
        total_reward = 0
        steps = 0
        
        for step in range(10):
            action = env.action_space.sample() * 0.1  # Small actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
                
        print("✅ Simple episode successful")
        print(f"   Steps: {steps}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Final voltage range: {info.get('min_voltage', 'N/A'):.3f} - {info.get('max_voltage', 'N/A'):.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Simple episode failed: {e}")
        traceback.print_exc()
        return False

def test_cli_basic():
    """Test basic CLI functionality."""
    try:
        from grid_fed_rl.cli import main
        print("✅ CLI import successful")
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Run all Generation 1 tests."""
    print("=== GENERATION 1: MAKE IT WORK (Basic Functionality) ===\n")
    
    tests = [
        test_basic_imports,
        test_environment_creation,
        test_environment_reset,
        test_environment_step,
        test_simple_episode,
        test_cli_basic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"Running {test.__name__}...")
        if test():
            passed += 1
        print()
    
    print(f"=== GENERATION 1 RESULTS: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 Generation 1 COMPLETE: Basic functionality working!")
        return True
    else:
        print("⚠️ Some tests failed - debugging needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)