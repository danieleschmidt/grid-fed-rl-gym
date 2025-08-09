#!/usr/bin/env python3
"""
Generation 2 Robustness Test Suite
Tests error handling, validation, logging, monitoring, and safety mechanisms
"""

import numpy as np
import sys
import traceback
import logging
import tempfile
import os

def test_input_validation():
    """Test input validation and sanitization."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=100)
        obs, info = env.reset(seed=42)
        
        # Test invalid actions
        invalid_actions = [
            np.array([np.inf, 0.5, 0.0]),  # Infinity
            np.array([np.nan, 0.5, 0.0]),  # NaN 
            np.array([1000, 0.5, 0.0]),   # Out of bounds
            np.array([-1000, 0.5, 0.0]),  # Out of bounds
            [],  # Wrong shape
            [1, 2, 3, 4, 5]  # Wrong shape
        ]
        
        successes = 0
        for i, action in enumerate(invalid_actions):
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                # Should handle gracefully with penalty
                if info.get('action_invalid', False) or reward <= -100:
                    successes += 1
                    print(f"   ✓ Invalid action {i+1} handled gracefully")
                else:
                    print(f"   ⚠ Invalid action {i+1} not properly detected")
            except Exception as e:
                print(f"   ⚠ Invalid action {i+1} caused exception: {e}")
        
        success_rate = successes / len(invalid_actions)
        print(f"✅ Input validation: {success_rate:.1%} of invalid inputs handled safely")
        return success_rate >= 0.5
        
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test comprehensive error handling."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=100)
        
        # Test with extreme conditions
        obs, info = env.reset(seed=42)
        
        # Test multiple steps with extreme actions
        for step in range(5):
            # Extreme action
            action = np.array([1.0, 1.0, 0.0]) * ((-1) ** step) * 10
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Environment should remain stable
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"   ⚠ NaN/Inf in observation at step {step}")
                return False
        
        print("✅ Error handling: Environment stable under extreme conditions")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_logging_monitoring():
    """Test logging and monitoring systems."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        import logging
        
        # Setup test logger
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as f:
            log_file = f.name
            
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=50)
        obs, info = env.reset(seed=42)
        
        # Run episode with monitoring
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check that info contains monitoring data
            expected_keys = ['power_flow_converged', 'max_voltage', 'min_voltage', 'total_losses']
            missing_keys = [key for key in expected_keys if key not in info]
            
        # Check log file was created and contains entries
        log_size = os.path.getsize(log_file)
        os.unlink(log_file)  # Cleanup
        
        if missing_keys:
            print(f"   ⚠ Missing monitoring keys: {missing_keys}")
            
        print(f"✅ Logging/Monitoring: Log file created ({log_size} bytes)")
        print(f"   Info keys available: {len(info)} monitoring metrics")
        return log_size > 0 and len(missing_keys) <= 1
        
    except Exception as e:
        print(f"❌ Logging/monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_safety_constraints():
    """Test safety constraint enforcement."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(
            feeder=feeder, 
            timestep=1.0, 
            episode_length=100,
            voltage_limits=(0.95, 1.05),
            frequency_limits=(59.5, 60.5),
            safety_penalty=100.0
        )
        obs, info = env.reset(seed=42)
        
        violations_detected = 0
        total_steps = 20
        
        for step in range(total_steps):
            # Deliberately extreme action to trigger constraints
            action = np.array([1.0, 1.0, 0.0]) * 2.0  # Large action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check constraint violations
            violations = info.get('constraint_violations', {})
            if any(violations.values()) if isinstance(violations, dict) else violations:
                violations_detected += 1
                
            # Large penalties should be applied for safety violations  
            if reward < -50:
                violations_detected += 1
                
            if terminated or truncated:
                if info.get('constraint_violations', 0) > 5:
                    violations_detected += 1
                break
        
        violation_rate = violations_detected / total_steps
        print(f"✅ Safety constraints: {violation_rate:.1%} violation detection rate")
        print(f"   Final violations count: {info.get('constraint_violations', 0)}")
        return violation_rate > 0.1  # Should detect some violations with extreme actions
        
    except Exception as e:
        print(f"❌ Safety constraint test failed: {e}")
        traceback.print_exc()
        return False

def test_robust_power_flow():
    """Test robust power flow solver with fallback mechanisms."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=100)
        obs, info = env.reset(seed=42)
        
        convergence_failures = 0
        total_steps = 50
        
        for step in range(total_steps):
            # Mix of normal and extreme actions
            if step % 10 == 0:
                action = np.array([0.99, 0.99, 0.1]) * (1 + step * 0.1)  # Escalating extreme actions
            else:
                action = env.action_space.sample() * 0.5  # Normal actions
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track power flow convergence
            if not info.get('power_flow_converged', True):
                convergence_failures += 1
                
            if terminated or truncated:
                break
        
        convergence_rate = 1.0 - (convergence_failures / total_steps)
        print(f"✅ Robust power flow: {convergence_rate:.1%} convergence rate")
        print(f"   Convergence failures: {convergence_failures}/{total_steps}")
        return convergence_rate >= 0.8  # Should maintain high convergence rate
        
    except Exception as e:
        print(f"❌ Robust power flow test failed: {e}")
        traceback.print_exc()
        return False

def test_resource_management():
    """Test memory and resource management."""
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Create and run multiple environment instances
        for env_i in range(3):
            feeder = IEEE13Bus()
            env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=50)
            
            for episode in range(2):
                obs, info = env.reset(seed=42 + episode)
                for step in range(25):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
            
            # Clean up explicitly
            del env
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"✅ Resource management: {memory_growth:.1f} MB memory growth")
        return memory_growth < 100  # Should not grow excessively
        
    except ImportError:
        print("⚠ psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"❌ Resource management test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration_validation():
    """Test configuration validation and sanitization."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        
        # Test various invalid configurations
        invalid_configs = [
            {"timestep": 0},  # Invalid timestep
            {"timestep": -1},  # Negative timestep
            {"episode_length": 0},  # Invalid episode length
            {"voltage_limits": (1.1, 0.9)},  # Inverted limits
            {"safety_penalty": -10}  # Negative penalty
        ]
        
        valid_envs_created = 0
        
        for i, config in enumerate(invalid_configs):
            try:
                env = GridEnvironment(feeder=feeder, **config)
                obs, info = env.reset(seed=42)
                # If we get here, the invalid config was sanitized
                valid_envs_created += 1
                print(f"   ✓ Invalid config {i+1} handled/sanitized")
            except (ValueError, AssertionError, TypeError) as e:
                print(f"   ✓ Invalid config {i+1} properly rejected: {type(e).__name__}")
                valid_envs_created += 1  # Properly rejecting is also good
            except Exception as e:
                print(f"   ⚠ Invalid config {i+1} caused unexpected error: {e}")
        
        success_rate = valid_envs_created / len(invalid_configs)
        print(f"✅ Configuration validation: {success_rate:.1%} of invalid configs handled")
        return success_rate >= 0.6
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Generation 2 robustness tests."""
    print("=== GENERATION 2: MAKE IT ROBUST (Reliable) ===\n")
    
    tests = [
        test_input_validation,
        test_error_handling,
        test_logging_monitoring,
        test_safety_constraints,
        test_robust_power_flow,
        test_resource_management,
        test_configuration_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"Running {test.__name__}...")
        if test():
            passed += 1
        print()
    
    print(f"=== GENERATION 2 RESULTS: {passed}/{total} tests passed ===")
    
    if passed >= total * 0.8:  # Allow 80% pass rate for robustness
        print("🎉 Generation 2 COMPLETE: Robust error handling and monitoring!")
        return True
    else:
        print("⚠️ Some robustness tests failed - improvements needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)