#!/usr/bin/env python3
"""Simplified test of Generation 3 features without external dependencies."""

import sys
import time
import numpy as np
import warnings

def test_distributed_computing():
    """Test distributed computing capabilities."""
    print("🚀 Testing Distributed Computing...")
    
    try:
        from grid_fed_rl.utils.distributed import (
            DistributedExecutor, parallel_map
        )
        
        # Test DistributedExecutor
        executor = DistributedExecutor(num_workers=2)
        executor.start()
        
        # Submit some test tasks
        def test_task(x):
            return x * x
            
        for i in range(5):
            executor.submit_task(f"task_{i}", test_task, i)
            
        # Wait for completion
        success = executor.wait_for_completion(timeout=5.0)
        if not success:
            print("   ✗ Tasks did not complete in time")
            executor.stop()
            return False
            
        completed, failed = executor.get_results()
        executor.stop()
        
        if len(completed) != 5:
            print(f"   ✗ Expected 5 completed tasks, got {len(completed)}")
            return False
            
        print("   ✓ DistributedExecutor functional")
        
        # Test parallel_map
        test_data = list(range(10))
        parallel_results = parallel_map(lambda x: x**2, test_data, num_workers=2)
        expected_results = [x**2 for x in test_data]
        
        if parallel_results != expected_results:
            print("   ✗ parallel_map results incorrect")
            return False
            
        print("   ✓ parallel_map functional")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Distributed computing test failed: {e}")
        return False


def test_optimization_features():
    """Test optimization and caching features."""
    print("⚡ Testing Optimization Features...")
    
    try:
        from grid_fed_rl.utils.optimization import (
            AdaptiveCache, StateCompressor, OptimizationConfig
        )
        
        # Test AdaptiveCache
        cache = AdaptiveCache(initial_size=10)
        
        # Add some items
        for i in range(15):
            cache.put(f"key_{i}", f"value_{i}")
            
        # Test retrieval
        result = cache.get("key_5")
        if result != "value_5":
            print(f"   ✗ Cache retrieval failed: got {result}, expected value_5")
            return False
            
        stats = cache.get_stats()
        if stats["hits"] == 0:
            print("   ✗ Cache not recording hits")
            return False
            
        print("   ✓ AdaptiveCache functional")
        
        # Test StateCompressor
        compressor = StateCompressor()
        
        original_state = np.random.randn(50)
        compressed = compressor.compress_state(original_state)
        decompressed = compressor.decompress_state(compressed)
        
        if not np.allclose(original_state, decompressed):
            print("   ✗ State compression/decompression failed")
            return False
            
        print("   ✓ StateCompressor functional")
        
        # Test OptimizationConfig
        config = OptimizationConfig(
            enable_caching=True,
            enable_state_compression=True,
            cache_size=50
        )
        
        if not config.enable_caching or not config.enable_state_compression:
            print("   ✗ OptimizationConfig not configured correctly")
            return False
            
        print("   ✓ OptimizationConfig functional")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Optimization features test failed: {e}")
        return False


def test_parallel_environments():
    """Test parallel environment execution."""
    print("🔄 Testing Parallel Environment Execution...")
    
    try:
        from grid_fed_rl.utils.distributed import ParallelEnvironmentRunner
        from grid_fed_rl.environments import GridEnvironment
        from grid_fed_rl.feeders import SimpleRadialFeeder
        
        def env_factory(num_buses=3, **kwargs):
            return GridEnvironment(
                feeder=SimpleRadialFeeder(num_buses=num_buses),
                episode_length=3,  # Very short for testing
                **kwargs
            )
            
        # Create parallel runner
        runner = ParallelEnvironmentRunner(
            env_factory=env_factory,
            num_envs=2,  # Reduced for testing
            env_configs=[
                {"num_buses": 3},
                {"num_buses": 4}
            ]
        )
        
        # Initialize environments
        runner.initialize_environments()
        
        if len(runner.environments) != 2:
            print(f"   ✗ Expected 2 environments, got {len(runner.environments)}")
            return False
            
        # Define simple random policy
        def random_policy(obs):
            return np.random.uniform(-1, 1, size=1)
            
        # Run parallel episodes
        episode_data = runner.run_parallel_episodes(
            policy=random_policy,
            episodes_per_env=1,  # Just 1 episode per env
            max_steps_per_episode=3
        )
        
        expected_episodes = 2 * 1  # 2 envs * 1 episode each
        if len(episode_data) != expected_episodes:
            print(f"   ✗ Expected {expected_episodes} episodes, got {len(episode_data)}")
            return False
            
        # Check episode data structure
        for ep_data in episode_data:
            if "total_reward" not in ep_data or "steps" not in ep_data:
                print("   ✗ Episode data missing required fields")
                return False
                
        runner.shutdown()
        print("   ✓ ParallelEnvironmentRunner functional")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Parallel environments test failed: {e}")
        return False


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("💾 Testing Memory Efficiency...")
    
    try:
        from grid_fed_rl.utils.optimization import StateCompressor
        
        # Test memory usage with compression
        compressor = StateCompressor()
        
        # Create state arrays
        large_states = []
        compressed_states = []
        
        for i in range(20):  # Reduced for testing
            state = np.random.randn(100)  # 100-element state
            large_states.append(state)
            compressed_states.append(compressor.compress_state(state))
            
        # Calculate memory usage (rough estimate)
        uncompressed_size = sum(state.nbytes for state in large_states)
        compressed_size = sum(len(comp) for comp in compressed_states)
        
        compression_ratio = compressed_size / uncompressed_size
        
        print(f"   ✓ Uncompressed size: {uncompressed_size / 1024:.1f} KB")
        print(f"   ✓ Compressed size: {compressed_size / 1024:.1f} KB")
        print(f"   ✓ Compression ratio: {compression_ratio:.2f}")
        
        # Test decompression accuracy
        for i, (original, compressed) in enumerate(zip(large_states, compressed_states)):
            decompressed = compressor.decompress_state(compressed)
            
            if not np.allclose(original, decompressed):
                print(f"   ✗ Decompression accuracy failed for state {i}")
                return False
                
        print("   ✓ Compression/decompression accuracy verified")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Memory efficiency test failed: {e}")
        return False


def test_basic_scaling_features():
    """Test basic scaling features."""
    print("📊 Testing Basic Scaling Features...")
    
    try:
        from grid_fed_rl.utils.optimization import OptimizedEnvironmentWrapper, OptimizationConfig
        from grid_fed_rl.environments import GridEnvironment
        from grid_fed_rl.feeders import SimpleRadialFeeder
        
        # Create base environment  
        base_env = GridEnvironment(
            feeder=SimpleRadialFeeder(num_buses=3),
            episode_length=5
        )
        
        config = OptimizationConfig(
            enable_caching=True,
            enable_state_compression=True,
            cache_size=20
        )
        
        opt_env = OptimizedEnvironmentWrapper(base_env, config)
        
        # Run some steps
        obs, _ = opt_env.reset()
        
        for step in range(3):
            action = opt_env.action_space.sample()
            obs, reward, done, truncated, info = opt_env.step(action)
            if done or truncated:
                break
                
        # Get optimization stats
        opt_stats = opt_env.get_optimization_stats()
        
        if opt_stats["total_steps"] == 0:
            print("   ✗ No steps recorded in optimization stats")
            return False
            
        if not opt_stats["cache_enabled"]:
            print("   ✗ Caching not enabled")
            return False
            
        if not opt_stats["compression_enabled"]:
            print("   ✗ Compression not enabled") 
            return False
            
        print("   ✓ OptimizedEnvironmentWrapper functional")
        print(f"   ✓ Completed {opt_stats['total_steps']} steps")
        print(f"   ✓ Cache enabled: {opt_stats['cache_enabled']}")
        print(f"   ✓ Compression enabled: {opt_stats['compression_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Basic scaling features test failed: {e}")
        return False


def main():
    """Run Generation 3 scaling tests (simplified)."""
    print("🚀 Generation 3: Testing Core Scaling Features")
    print("=" * 50)
    
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        tests = [
            test_distributed_computing,
            test_optimization_features,
            test_parallel_environments,
            test_memory_efficiency,
            test_basic_scaling_features
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
    
    print("🏁 Generation 3 Core Scaling Test Results")
    print("=" * 45)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All core scaling tests passed!")
        print("\nGeneration 3 core scaling features verified:")
        print("  ✅ Distributed computing framework")
        print("  ✅ Advanced optimization features")
        print("  ✅ Parallel environment execution")
        print("  ✅ Memory efficiency improvements") 
        print("  ✅ Basic scaling capabilities")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)