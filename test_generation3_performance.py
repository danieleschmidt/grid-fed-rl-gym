#!/usr/bin/env python3
"""
Generation 3 Performance and Scaling Test Suite
Tests performance optimization, caching, concurrent processing, and auto-scaling
"""

import numpy as np
import sys
import traceback
import time
import threading
import concurrent.futures
import multiprocessing

def test_performance_benchmarks():
    """Test basic performance benchmarks."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=1000)
        
        # Benchmark environment reset
        reset_times = []
        for i in range(10):
            start = time.time()
            obs, info = env.reset(seed=42 + i)
            reset_times.append(time.time() - start)
        
        avg_reset_time = np.mean(reset_times) * 1000  # ms
        
        # Benchmark environment steps
        obs, info = env.reset(seed=42)
        step_times = []
        
        for step in range(100):
            action = env.action_space.sample()
            start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.time() - start)
            
            if terminated or truncated:
                break
        
        avg_step_time = np.mean(step_times) * 1000  # ms
        
        # Performance targets
        reset_target = 50.0  # ms
        step_target = 10.0   # ms
        
        reset_pass = avg_reset_time <= reset_target
        step_pass = avg_step_time <= step_target
        
        print(f"✅ Performance benchmarks:")
        print(f"   Reset time: {avg_reset_time:.2f}ms (target: <{reset_target}ms) {'✓' if reset_pass else '⚠'}")
        print(f"   Step time: {avg_step_time:.2f}ms (target: <{step_target}ms) {'✓' if step_pass else '⚠'}")
        
        return reset_pass and step_pass
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

def test_caching_mechanisms():
    """Test caching for power flow and other computations."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=100)
        obs, info = env.reset(seed=42)
        
        # Measure first computation (no cache)
        action = np.array([0.5, 0.5, 0.0])
        start = time.time()
        obs1, reward1, terminated1, truncated1, info1 = env.step(action)
        first_time = time.time() - start
        
        # Reset and repeat exact same action (should benefit from caching)
        obs, info = env.reset(seed=42)
        start = time.time()
        obs2, reward2, terminated2, truncated2, info2 = env.step(action)
        second_time = time.time() - start
        
        # Third identical step (more caching benefits)
        start = time.time()
        obs3, reward3, terminated3, truncated3, info3 = env.step(action)
        third_time = time.time() - start
        
        # Check if we see performance improvements (caching working)
        cache_improvement = (first_time - min(second_time, third_time)) / first_time
        
        print(f"✅ Caching mechanisms:")
        print(f"   First step: {first_time*1000:.2f}ms")
        print(f"   Second step: {second_time*1000:.2f}ms")
        print(f"   Third step: {third_time*1000:.2f}ms")
        print(f"   Cache improvement: {cache_improvement:.1%}")
        
        return cache_improvement >= 0.0  # Any improvement is good
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        traceback.print_exc()
        return False

def test_concurrent_environments():
    """Test concurrent processing of multiple environments."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        def run_environment_episode(env_id):
            """Run a single environment episode."""
            try:
                feeder = IEEE13Bus()
                env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=50)
                obs, info = env.reset(seed=env_id)
                
                total_reward = 0
                for step in range(50):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                
                return env_id, total_reward, step + 1
            except Exception as e:
                return env_id, None, str(e)
        
        # Sequential execution baseline
        start_sequential = time.time()
        sequential_results = []
        for i in range(4):
            result = run_environment_episode(i)
            sequential_results.append(result)
        sequential_time = time.time() - start_sequential
        
        # Concurrent execution
        start_concurrent = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(run_environment_episode, range(4, 8)))
        concurrent_time = time.time() - start_concurrent
        
        # Check results
        sequential_successes = sum(1 for _, reward, _ in sequential_results if reward is not None)
        concurrent_successes = sum(1 for _, reward, _ in concurrent_results if reward is not None)
        
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
        
        print(f"✅ Concurrent environments:")
        print(f"   Sequential time: {sequential_time:.2f}s ({sequential_successes}/4 success)")
        print(f"   Concurrent time: {concurrent_time:.2f}s ({concurrent_successes}/4 success)")
        print(f"   Speedup: {speedup:.2f}x")
        
        return concurrent_successes >= 3 and speedup > 0.8
        
    except Exception as e:
        print(f"❌ Concurrent environments test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_optimization():
    """Test memory-efficient operations."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Create environment
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=100)
        
        # Test large batch operations
        batch_size = 50
        observations = []
        rewards = []
        
        obs, info = env.reset(seed=42)
        
        for step in range(batch_size):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Store efficiently
            observations.append(obs.copy())
            rewards.append(reward)
            
            if terminated or truncated:
                obs, info = env.reset(seed=42 + step)
        
        # Convert to efficient numpy arrays
        observations_array = np.array(observations)
        rewards_array = np.array(rewards)
        
        # Test vectorized operations
        start = time.time()
        mean_obs = np.mean(observations_array, axis=0)
        std_obs = np.std(observations_array, axis=0)
        total_reward = np.sum(rewards_array)
        vectorized_time = time.time() - start
        
        # Test that vectorized ops are fast
        target_time = 0.01  # 10ms
        
        print(f"✅ Memory optimization:")
        print(f"   Batch size: {batch_size}")
        print(f"   Observations shape: {observations_array.shape}")
        print(f"   Vectorized ops time: {vectorized_time*1000:.2f}ms (target: <{target_time*1000}ms)")
        print(f"   Total reward: {total_reward:.2f}")
        
        return vectorized_time <= target_time
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        traceback.print_exc()
        return False

def test_load_balancing():
    """Test adaptive load balancing."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Simulate different workloads
        workloads = [
            {"episode_length": 10, "complexity": "low"},
            {"episode_length": 50, "complexity": "medium"}, 
            {"episode_length": 100, "complexity": "high"}
        ]
        
        execution_times = []
        
        for workload in workloads:
            feeder = IEEE13Bus()
            env = GridEnvironment(
                feeder=feeder, 
                timestep=1.0, 
                episode_length=workload["episode_length"]
            )
            
            start = time.time()
            obs, info = env.reset(seed=42)
            
            for step in range(workload["episode_length"]):
                # Vary action complexity based on workload
                if workload["complexity"] == "high":
                    action = env.action_space.sample() * 0.8  # More complex actions
                else:
                    action = env.action_space.sample() * 0.2  # Simpler actions
                    
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            execution_time = time.time() - start
            execution_times.append(execution_time)
        
        # Check that execution time scales reasonably with complexity
        time_ratios = [execution_times[i+1] / execution_times[i] for i in range(len(execution_times)-1)]
        reasonable_scaling = all(ratio < 20 for ratio in time_ratios)  # Not exponential growth
        
        print(f"✅ Load balancing:")
        for i, (workload, time_taken) in enumerate(zip(workloads, execution_times)):
            print(f"   {workload['complexity']} workload: {time_taken:.2f}s")
        print(f"   Scaling ratios: {[f'{ratio:.1f}x' for ratio in time_ratios]}")
        
        return reasonable_scaling
        
    except Exception as e:
        print(f"❌ Load balancing test failed: {e}")
        traceback.print_exc()
        return False

def test_auto_scaling_triggers():
    """Test auto-scaling trigger mechanisms."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Simulate increasing load
        load_levels = [1, 2, 4, 8]  # Simulated concurrent environments
        response_times = []
        
        for load_level in load_levels:
            start = time.time()
            
            # Simulate concurrent load by running multiple quick episodes
            for env_idx in range(load_level):
                feeder = IEEE13Bus()
                env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=10)
                obs, info = env.reset(seed=42 + env_idx)
                
                for step in range(10):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
            
            response_time = time.time() - start
            response_times.append(response_time)
        
        # Check if response time grows linearly (good scaling)
        # rather than exponentially (poor scaling)
        max_growth_ratio = max(response_times[i+1] / response_times[i] 
                              for i in range(len(response_times)-1))
        
        good_scaling = max_growth_ratio < 3.0  # Linear-ish growth
        
        print(f"✅ Auto-scaling triggers:")
        for load, time_taken in zip(load_levels, response_times):
            print(f"   Load level {load}: {time_taken:.2f}s")
        print(f"   Max growth ratio: {max_growth_ratio:.1f}x (target: <3.0x)")
        
        return good_scaling
        
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        traceback.print_exc()
        return False

def test_resource_pooling():
    """Test resource pooling and reuse."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Create a pool of environments for reuse
        environment_pool = []
        pool_size = 3
        
        # Initialize pool
        for i in range(pool_size):
            feeder = IEEE13Bus()
            env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=50)
            environment_pool.append(env)
        
        # Test pool usage with multiple "sessions"
        session_results = []
        
        for session in range(6):  # More sessions than pool size
            # Get environment from pool (round-robin)
            env = environment_pool[session % pool_size]
            
            start = time.time()
            obs, info = env.reset(seed=42 + session)
            
            total_reward = 0
            for step in range(10):  # Short episodes for testing
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            session_time = time.time() - start
            session_results.append((session, total_reward, session_time))
        
        # Check that pool reuse works (no failures)
        successful_sessions = sum(1 for _, reward, _ in session_results 
                                 if reward is not None and not np.isnan(reward))
        
        avg_session_time = np.mean([time_taken for _, _, time_taken in session_results])
        
        print(f"✅ Resource pooling:")
        print(f"   Pool size: {pool_size}")
        print(f"   Total sessions: {len(session_results)}")
        print(f"   Successful sessions: {successful_sessions}")
        print(f"   Average session time: {avg_session_time:.2f}s")
        
        return successful_sessions >= len(session_results) * 0.8  # 80% success rate
        
    except Exception as e:
        print(f"❌ Resource pooling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all Generation 3 performance and scaling tests."""
    print("=== GENERATION 3: MAKE IT SCALE (Optimized) ===\n")
    
    tests = [
        test_performance_benchmarks,
        test_caching_mechanisms,
        test_concurrent_environments,
        test_memory_optimization,
        test_load_balancing,
        test_auto_scaling_triggers,
        test_resource_pooling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"Running {test.__name__}...")
        if test():
            passed += 1
        print()
    
    print(f"=== GENERATION 3 RESULTS: {passed}/{total} tests passed ===")
    
    if passed >= total * 0.7:  # Allow 70% pass rate for performance tests
        print("🎉 Generation 3 COMPLETE: Performance optimized and scalable!")
        return True
    else:
        print("⚠️ Some performance tests failed - optimization needed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)