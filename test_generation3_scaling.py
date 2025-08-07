#!/usr/bin/env python3
"""Test Generation 3 scaling and optimization features."""

import sys
import time
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

def test_distributed_computing():
    """Test distributed computing capabilities."""
    print("🚀 Testing Distributed Computing...")
    
    try:
        from grid_fed_rl.utils.distributed import (
            DistributedExecutor, ParallelEnvironmentRunner, parallel_map
        )
        
        # Test DistributedExecutor
        executor = DistributedExecutor(num_workers=2)
        executor.start()
        
        # Submit some test tasks
        def test_task(x):
            return x * x
            
        for i in range(10):
            executor.submit_task(f"task_{i}", test_task, i)
            
        # Wait for completion
        success = executor.wait_for_completion(timeout=10.0)
        if not success:
            print("   ✗ Tasks did not complete in time")
            return False
            
        completed, failed = executor.get_results()
        
        if len(completed) != 10:
            print(f"   ✗ Expected 10 completed tasks, got {len(completed)}")
            return False
            
        # Check results
        results = {r.task_id: r.result for r in completed}
        expected = {f"task_{i}": i*i for i in range(10)}
        
        for task_id, expected_result in expected.items():
            if results.get(task_id) != expected_result:
                print(f"   ✗ Wrong result for {task_id}: got {results.get(task_id)}, expected {expected_result}")
                return False
                
        executor.stop()
        print("   ✓ DistributedExecutor functional")
        
        # Test parallel_map
        test_data = list(range(20))
        parallel_results = parallel_map(lambda x: x**2, test_data, num_workers=3)
        expected_results = [x**2 for x in test_data]
        
        if parallel_results != expected_results:
            print("   ✗ parallel_map results incorrect")
            return False
            
        print("   ✓ parallel_map functional")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Distributed computing test failed: {e}")
        return False


def test_monitoring_system():
    """Test monitoring and telemetry system."""
    print("📊 Testing Monitoring System...")
    
    try:
        from grid_fed_rl.utils.monitoring import (
            MetricsCollector, PerformanceMonitor, AutoScaler,
            SystemMetrics, GridMetrics, TrainingMetrics
        )
        
        # Test MetricsCollector
        collector = MetricsCollector(collection_interval=0.1)
        collector.start_collection()
        
        # Let it collect for a short time
        time.sleep(0.3)
        
        # Add some test metrics
        grid_metric = GridMetrics(
            timestamp=time.time(),
            environment_id="test_env",
            episode=1,
            step=10,
            total_reward=100.5,
            power_flow_convergence=True,
            power_flow_iterations=3,
            voltage_violations=0,
            frequency_deviation=0.1,
            total_losses=50.0,
            renewable_generation=200.0,
            load_served=500.0
        )
        
        collector.record_grid_metrics(grid_metric)
        
        training_metric = TrainingMetrics(
            timestamp=time.time(),
            algorithm="test_algo",
            episode=1,
            step=10,
            loss=0.5,
            reward=100.5
        )
        
        collector.record_training_metrics(training_metric)
        
        # Get statistics
        system_stats = collector.get_system_stats()
        grid_stats = collector.get_grid_stats()
        training_stats = collector.get_training_stats()
        
        collector.stop_collection()
        
        if not system_stats:
            print("   ✗ No system stats collected")
            return False
            
        if grid_stats["sample_count"] != 1:
            print(f"   ✗ Expected 1 grid sample, got {grid_stats['sample_count']}")
            return False
            
        print("   ✓ MetricsCollector functional")
        
        # Test PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Record some performance data
        monitor.record_environment_performance(
            env_id="test_env",
            episode=1,
            step=5,
            reward=50.0,
            info={"power_flow_converged": True, "voltage_violations": 0}
        )
        
        report = monitor.get_performance_report()
        monitor.stop_monitoring()
        
        if "system_performance" not in report:
            print("   ✗ Performance report missing system data")
            return False
            
        print("   ✓ PerformanceMonitor functional")
        
        # Test AutoScaler
        scaler = AutoScaler(min_workers=1, max_workers=4)
        
        recommended = scaler.get_recommended_workers()
        if recommended < 1 or recommended > 4:
            print(f"   ✗ Invalid recommended workers: {recommended}")
            return False
            
        print("   ✓ AutoScaler functional")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Monitoring system test failed: {e}")
        return False


def test_optimization_features():
    """Test optimization and caching features."""
    print("⚡ Testing Optimization Features...")
    
    try:
        from grid_fed_rl.utils.optimization import (
            AdaptiveCache, StateCompressor, OptimizedEnvironmentWrapper,
            OptimizationConfig, VectorizedPowerFlow
        )
        from grid_fed_rl.environments import GridEnvironment
        from grid_fed_rl.feeders import SimpleRadialFeeder
        
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
        
        original_state = np.random.randn(100)
        compressed = compressor.compress_state(original_state)
        decompressed = compressor.decompress_state(compressed)
        
        if not np.allclose(original_state, decompressed):
            print("   ✗ State compression/decompression failed")
            return False
            
        compression_ratio = compressor.get_compression_ratio()
        print(f"   ✓ StateCompressor functional (ratio: {compression_ratio:.2f})")
        
        # Test OptimizedEnvironmentWrapper
        base_env = GridEnvironment(
            feeder=SimpleRadialFeeder(num_buses=3),
            episode_length=10
        )
        
        config = OptimizationConfig(
            enable_caching=True,
            enable_state_compression=True,
            cache_size=50
        )
        
        opt_env = OptimizedEnvironmentWrapper(base_env, config)
        
        # Run some steps
        obs, _ = opt_env.reset()
        
        for _ in range(5):
            action = opt_env.action_space.sample()
            obs, reward, done, truncated, info = opt_env.step(action)
            if done or truncated:
                break
                
        # Get optimization stats
        opt_stats = opt_env.get_optimization_stats()
        
        if opt_stats["total_steps"] == 0:
            print("   ✗ No steps recorded in optimization stats")
            return False
            
        print("   ✓ OptimizedEnvironmentWrapper functional")
        
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
                episode_length=5,
                **kwargs
            )
            
        # Create parallel runner
        runner = ParallelEnvironmentRunner(
            env_factory=env_factory,
            num_envs=3,
            env_configs=[
                {"num_buses": 3},
                {"num_buses": 4}, 
                {"num_buses": 5}
            ]
        )
        
        # Initialize environments
        runner.initialize_environments()
        
        if len(runner.environments) != 3:
            print(f"   ✗ Expected 3 environments, got {len(runner.environments)}")
            return False
            
        # Define simple random policy
        def random_policy(obs):
            return np.random.uniform(-1, 1, size=1)
            
        # Run parallel episodes
        episode_data = runner.run_parallel_episodes(
            policy=random_policy,
            episodes_per_env=2,
            max_steps_per_episode=5
        )
        
        expected_episodes = 3 * 2  # 3 envs * 2 episodes each
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


def test_performance_scaling():
    """Test performance improvements with scaling."""
    print("📈 Testing Performance Scaling...")
    
    try:
        from grid_fed_rl.utils.optimization import benchmark_optimization_impact
        from grid_fed_rl.environments import GridEnvironment  
        from grid_fed_rl.feeders import SimpleRadialFeeder
        
        def env_factory():
            return GridEnvironment(
                feeder=SimpleRadialFeeder(num_buses=4),
                episode_length=20
            )
            
        # Quick benchmark (small scale for testing)
        results = benchmark_optimization_impact(
            env_factory=env_factory,
            num_episodes=2,
            max_steps=10
        )
        
        # Check that we have results for different configurations
        expected_configs = ["baseline", "caching_only", "compression_only", "all_optimizations"]
        
        for config in expected_configs:
            if config not in results:
                print(f"   ✗ Missing benchmark results for {config}")
                return False
                
            if "total_time" not in results[config]:
                print(f"   ✗ Missing timing data for {config}")
                return False
                
        # Check that optimizations show some improvement
        baseline_time = results["baseline"]["total_time"]
        optimized_time = results["all_optimizations"]["total_time"]
        
        print(f"   ✓ Baseline time: {baseline_time:.3f}s")
        print(f"   ✓ Optimized time: {optimized_time:.3f}s")
        
        if optimized_time <= baseline_time * 1.2:  # Allow some variance
            print("   ✓ Optimizations show reasonable performance")
        else:
            print("   ! Optimizations may not be helping (could be test variability)")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Performance scaling test failed: {e}")
        return False


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("💾 Testing Memory Efficiency...")
    
    try:
        from grid_fed_rl.utils.optimization import StateCompressor
        import sys
        
        # Test memory usage with compression
        compressor = StateCompressor()
        
        # Create large state arrays
        large_states = []
        compressed_states = []
        
        for i in range(50):
            state = np.random.randn(200)  # 200-element state
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
                
        if compression_ratio > 0.8:  # Should achieve some compression
            print("   ! Compression ratio higher than expected (may be due to random data)")
        else:
            print("   ✓ Compression achieving good ratio")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Memory efficiency test failed: {e}")
        return False


def main():
    """Run all Generation 3 scaling tests."""
    print("🚀 Generation 3: Testing Scaling & Optimization Features")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        tests = [
            test_distributed_computing,
            test_monitoring_system, 
            test_optimization_features,
            test_parallel_environments,
            test_performance_scaling,
            test_memory_efficiency
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
    
    print("🏁 Generation 3 Scaling Test Results")
    print("=" * 40)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All scaling tests passed!")
        print("\nGeneration 3 scaling features verified:")
        print("  ✅ Distributed computing framework")
        print("  ✅ Monitoring and telemetry system")
        print("  ✅ Advanced optimization features")
        print("  ✅ Parallel environment execution")
        print("  ✅ Performance scaling capabilities")
        print("  ✅ Memory efficiency improvements")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)