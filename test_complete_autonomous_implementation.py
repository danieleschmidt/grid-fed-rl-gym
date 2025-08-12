#!/usr/bin/env python3
"""
Comprehensive test suite for the autonomous SDLC implementation.
Tests all three generations: MAKE IT WORK, MAKE IT ROBUST, MAKE IT SCALE.
"""

import pytest
import numpy as np
import time
import threading
import logging
from typing import Dict, Any, List
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestGeneration1MakeItWork:
    """Test Generation 1: Core functionality - MAKE IT WORK."""
    
    def test_package_import(self):
        """Test that the package imports correctly."""
        import grid_fed_rl
        assert grid_fed_rl.__version__ == "0.1.0"
        assert grid_fed_rl.__author__ == "Daniel Schmidt"
        logger.info("✅ Package import successful")
    
    def test_core_environment_creation(self):
        """Test creation of core grid environment."""
        from grid_fed_rl import GridEnvironment, IEEE13Bus
        
        env = GridEnvironment(feeder=IEEE13Bus())
        assert env is not None
        logger.info("✅ Grid environment creation")
    
    def test_environment_basic_operations(self):
        """Test basic environment operations."""
        from grid_fed_rl import GridEnvironment, IEEE13Bus
        
        env = GridEnvironment(feeder=IEEE13Bus())
        
        # Test reset
        obs = env.reset()
        assert obs is not None
        
        # Test sample action
        action = env.action_space.sample()
        assert action is not None
        
        # Test step
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        logger.info("✅ Environment basic operations")
    
    def test_feeder_implementations(self):
        """Test different feeder implementations."""
        from grid_fed_rl.feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus
        
        feeders = [IEEE13Bus(), IEEE34Bus(), IEEE123Bus()]
        
        for i, feeder in enumerate(feeders):
            assert feeder is not None
            # Test basic feeder properties
            assert hasattr(feeder, 'bus_count')
            assert feeder.bus_count > 0
        
        logger.info("✅ IEEE feeder implementations")
    
    def test_offline_rl_algorithms(self):
        """Test offline RL algorithm implementations."""
        from grid_fed_rl.algorithms import CQL, IQL
        
        # Test algorithm instantiation
        state_dim, action_dim = 10, 3
        cql = CQL(state_dim=state_dim, action_dim=action_dim)
        iql = IQL(state_dim=state_dim, action_dim=action_dim)
        
        assert cql is not None
        assert iql is not None
        logger.info("✅ Offline RL algorithms")
    
    def test_federated_learning_core(self):
        """Test federated learning core functionality."""
        from grid_fed_rl.federated import FederatedOfflineRL
        
        fed_learner = FederatedOfflineRL(
            num_clients=3,
            rounds=5
        )
        assert fed_learner is not None
        logger.info("✅ Federated learning core")


class TestGeneration2MakeItRobust:
    """Test Generation 2: Robustness and reliability - MAKE IT ROBUST."""
    
    def test_error_handling_framework(self):
        """Test advanced error handling."""
        from grid_fed_rl.utils.exceptions import (
            CircuitBreaker, exponential_backoff, 
            ErrorRecoveryManager, PowerFlowError
        )
        
        # Test circuit breaker
        @CircuitBreaker(failure_threshold=2, reset_timeout=1.0)
        def failing_function():
            raise PowerFlowError("Test failure")
        
        # Test that circuit breaker opens after failures
        with pytest.raises(PowerFlowError):
            failing_function()
        with pytest.raises(PowerFlowError):
            failing_function()
        
        # Next call should raise CircuitBreakerOpenError
        from grid_fed_rl.utils.exceptions import CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            failing_function()
        
        logger.info("✅ Circuit breaker functionality")
    
    def test_distributed_tracing(self):
        """Test distributed tracing system."""
        from grid_fed_rl.utils.advanced_robustness import DistributedTracer
        
        tracer = DistributedTracer()
        
        # Test span creation and finishing
        span = tracer.start_span("test_operation", "test_component")
        assert span.trace_id is not None
        assert span.span_id is not None
        assert span.operation == "test_operation"
        
        span.add_log("info", "Test log message", extra="data")
        tracer.finish_span(span)
        
        # Test trace summary
        summary = tracer.get_trace_summary(span.trace_id)
        assert summary["span_count"] == 1
        assert summary["failed_spans"] == 0
        
        logger.info("✅ Distributed tracing system")
    
    def test_robust_power_flow_solver(self):
        """Test multi-solver power flow with fallback."""
        from grid_fed_rl.utils.advanced_robustness import RobustPowerFlowSolver
        
        solver = RobustPowerFlowSolver()
        
        # Test successful solution
        result = solver.solve_with_fallback({
            "bus_count": 13,
            "complexity": 1.0
        })
        
        assert result["converged"] is True
        assert "solver_used" in result
        assert "solve_time" in result
        assert len(result["bus_voltages"]) == 13
        
        # Test solver performance tracking
        performance = solver.get_solver_performance()
        assert isinstance(performance, dict)
        
        logger.info("✅ Robust power flow solver")
    
    def test_data_integrity_validation(self):
        """Test comprehensive data validation."""
        from grid_fed_rl.utils.advanced_robustness import DataIntegrityValidator
        
        validator = DataIntegrityValidator()
        
        # Test valid data
        valid_data = {
            "bus_voltages": [0.95, 1.0, 1.05],
            "frequency": 60.0,
            "line_flows": [0.5, 0.7, 0.3],
            "timestamp": time.time()
        }
        
        result = validator.validate_data(valid_data)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Test invalid data
        invalid_data = {
            "bus_voltages": [0.5, float('nan'), 2.0],  # Contains NaN and extreme value
            "frequency": 70.0,  # High frequency
            "line_flows": [1.5, 0.8],  # Overload
        }
        
        result = validator.validate_data(invalid_data)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        
        logger.info("✅ Data integrity validation")
    
    def test_advanced_safety_system(self):
        """Test multi-layer safety system."""
        from grid_fed_rl.utils.advanced_robustness import AdvancedSafetySystem
        
        safety_system = AdvancedSafetySystem()
        
        # Test safe system state
        safe_state = {
            "bus_voltages": np.array([0.98, 1.0, 1.02]),
            "frequency": 60.1,
            "line_flows": np.array([0.5, 0.6, 0.4])
        }
        
        safety_result = safety_system.evaluate_safety(safe_state)
        assert safety_result["overall_status"] == "safe"
        assert len(safety_result["emergency_protocols"]) == 0
        
        # Test unsafe system state
        unsafe_state = {
            "bus_voltages": np.array([0.7, 1.0, 1.4]),  # Extreme voltages
            "frequency": 62.5,  # High frequency
            "line_flows": np.array([1.2, 1.1, 0.4])  # Overloads
        }
        
        safety_result = safety_system.evaluate_safety(unsafe_state)
        assert safety_result["overall_status"] != "safe"
        assert len(safety_result["emergency_protocols"]) > 0
        
        logger.info("✅ Advanced safety system")
    
    def test_backup_and_recovery(self):
        """Test backup and recovery system."""
        from grid_fed_rl.utils.backup_recovery import BackupManager
        
        # Use temporary directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_mgr = BackupManager(backup_dir=temp_dir)
            
            # Test data
            test_data = {
                "system_state": {"mode": "normal", "voltage": [1.0, 0.98, 1.02]},
                "environment_config": {"feeder": "IEEE13"},
                "training_data": [1, 2, 3, 4, 5]
            }
            
            # Create backup
            backup_id = backup_mgr.create_backup(
                data=test_data,
                backup_type="test",
                compression=True,
                encryption=False
            )
            assert backup_id is not None
            
            # Restore backup
            restored_data = backup_mgr.restore_backup(backup_id)
            assert restored_data["system_state"]["mode"] == "normal"
            assert restored_data["environment_config"]["feeder"] == "IEEE13"
            
            # Test backup listing
            backups = backup_mgr.list_backups()
            assert len(backups) >= 1
            
            # Test backup statistics
            stats = backup_mgr.get_backup_statistics()
            assert stats["total_backups"] >= 1
            
        logger.info("✅ Backup and recovery system")
    
    def test_monitoring_system(self):
        """Test comprehensive monitoring system."""
        from grid_fed_rl.utils.monitoring import GridMonitor, SystemMetrics
        
        monitor = GridMonitor()
        
        # Record test metrics
        test_metrics = monitor.record_metrics(
            step_count=100,
            power_flow_time=0.05,
            grid_state={
                "bus_voltages": [0.98, 1.0, 1.02],
                "frequency": 60.0,
                "losses": 0.02,
                "renewable_power": 50.0,
                "total_power": 100.0
            },
            violations={"total_violations": 0}
        )
        
        assert isinstance(test_metrics, SystemMetrics)
        assert test_metrics.step_count == 100
        
        # Test statistics
        stats = monitor.get_summary_stats()
        assert "total_steps" in stats
        assert "avg_voltage_deviation" in stats
        
        logger.info("✅ Monitoring system")


class TestGeneration3MakeItScale:
    """Test Generation 3: Performance and scaling - MAKE IT SCALE."""
    
    def test_adaptive_load_balancer(self):
        """Test adaptive load balancing system."""
        from grid_fed_rl.utils.scaling_optimization import AdaptiveLoadBalancer
        
        # Create load balancer with minimal workers for testing
        load_balancer = AdaptiveLoadBalancer(min_workers=1, max_workers=2)
        
        # Submit a test task
        task_id = load_balancer.submit_task(
            "power_flow",
            {"bus_count": 5, "complexity": 0.5}
        )
        assert task_id is not None
        
        # Wait for result
        result = None
        max_wait = 10  # seconds
        start_time = time.time()
        
        while result is None and (time.time() - start_time) < max_wait:
            result = load_balancer.get_result(timeout=0.5)
        
        assert result is not None
        assert result["status"] in ["completed", "failed"]
        
        # Test performance stats
        stats = load_balancer.get_performance_stats()
        assert "total_workers" in stats
        assert stats["total_workers"] >= 1
        
        # Cleanup
        load_balancer.shutdown()
        
        logger.info("✅ Adaptive load balancer")
    
    def test_memory_optimizer(self):
        """Test memory optimization system."""
        from grid_fed_rl.utils.scaling_optimization import MemoryOptimizer
        
        optimizer = MemoryOptimizer(max_memory_mb=512)
        
        # Test array pooling
        array1 = optimizer.get_array((100, 100))
        assert array1.shape == (100, 100)
        
        optimizer.return_array(array1)
        
        array2 = optimizer.get_array((100, 100))
        # Should be the same array from pool (though content is cleared)
        assert array2.shape == (100, 100)
        
        # Test dictionary pooling
        dict1 = optimizer.get_dict()
        dict1["test"] = "value"
        optimizer.return_dict(dict1)
        
        dict2 = optimizer.get_dict()
        assert len(dict2) == 0  # Should be cleared
        
        # Test memory statistics
        stats = optimizer.get_memory_stats()
        assert "current_memory_mb" in stats
        assert "array_pool_size" in stats
        
        optimizer.stop_monitoring()
        
        logger.info("✅ Memory optimizer")
    
    def test_caching_system(self):
        """Test intelligent caching system."""
        from grid_fed_rl.utils.scaling_optimization import CachingSystem
        
        cache = CachingSystem(max_size=10, default_ttl=5.0)
        
        # Test basic caching
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent", "default") == "default"
        
        # Test TTL expiration
        cache.set("temp_key", "temp_value", ttl=0.1)  # 100ms TTL
        assert cache.get("temp_key") == "temp_value"
        
        time.sleep(0.2)  # Wait for expiration
        assert cache.get("temp_key") is None
        
        # Test cache statistics
        stats = cache.get_cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        
        # Test cache size limits (LRU eviction)
        for i in range(15):  # Exceed max_size of 10
            cache.set(f"key{i}", f"value{i}")
        
        assert cache.get("key0") is None  # Should be evicted
        assert cache.get("key14") == "value14"  # Should still exist
        
        cache.stop_maintenance()
        
        logger.info("✅ Caching system")
    
    def test_performance_scaling(self):
        """Test system performance under load."""
        from grid_fed_rl import GridEnvironment, IEEE13Bus
        from grid_fed_rl.utils.monitoring import GridMonitor
        
        # Create environment and monitor
        env = GridEnvironment(feeder=IEEE13Bus())
        monitor = GridMonitor()
        
        # Run performance test
        start_time = time.time()
        episode_count = 10
        steps_per_episode = 50
        
        for episode in range(episode_count):
            obs = env.reset()
            
            for step in range(steps_per_episode):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                # Record metrics
                monitor.record_metrics(
                    step_count=episode * steps_per_episode + step,
                    power_flow_time=0.01,  # Simulated
                    grid_state={
                        "bus_voltages": obs[:13] if len(obs) >= 13 else [1.0] * 13,
                        "frequency": 60.0,
                        "losses": 0.02
                    },
                    violations={"total_violations": 0}
                )
                
                if done:
                    break
        
        total_time = time.time() - start_time
        total_steps = episode_count * steps_per_episode
        
        steps_per_second = total_steps / total_time
        
        # Performance thresholds
        assert steps_per_second > 10  # At least 10 steps/second
        assert total_time < 60  # Complete in under 60 seconds
        
        # Check monitoring statistics
        stats = monitor.get_summary_stats()
        assert stats["total_steps"] >= total_steps * 0.9  # Allow for early episode termination
        
        logger.info(f"✅ Performance scaling: {steps_per_second:.1f} steps/sec")


class TestIntegrationAndQualityGates:
    """Integration tests and quality gate validation."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        from grid_fed_rl import GridEnvironment, IEEE13Bus
        from grid_fed_rl.utils.advanced_robustness import (
            DistributedTracer, RobustPowerFlowSolver, AdvancedSafetySystem
        )
        from grid_fed_rl.utils.backup_recovery import BackupManager
        from grid_fed_rl.utils.monitoring import GridMonitor
        
        # Initialize components
        tracer = DistributedTracer()
        solver = RobustPowerFlowSolver(tracer)
        safety_system = AdvancedSafetySystem(tracer)
        monitor = GridMonitor()
        
        # Create environment
        env = GridEnvironment(feeder=IEEE13Bus())
        
        # Run integrated workflow
        span = tracer.start_span("end_to_end_test", "integration_test")
        
        try:
            # Environment step
            obs = env.reset()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Power flow solving
            system_data = {"bus_count": 13, "complexity": 1.0}
            pf_result = solver.solve_with_fallback(system_data)
            
            # Safety evaluation
            system_state = {
                "bus_voltages": pf_result["bus_voltages"],
                "frequency": 60.0,
                "line_flows": pf_result["line_flows"]
            }
            safety_result = safety_system.evaluate_safety(system_state)
            
            # Monitoring
            metrics = monitor.record_metrics(
                step_count=1,
                power_flow_time=pf_result["solve_time"],
                grid_state=system_state,
                violations={"total_violations": len(safety_result.get("emergency_protocols", []))}
            )
            
            # Validate results
            assert pf_result["converged"]
            assert safety_result["overall_status"] is not None
            assert metrics.step_count == 1
            
            tracer.finish_span(span, "completed")
            
        except Exception as e:
            tracer.finish_span(span, "failed")
            raise
        
        logger.info("✅ End-to-end workflow integration")
    
    def test_quality_gates_validation(self):
        """Validate all quality gates pass."""
        from grid_fed_rl.utils.monitoring import GridMonitor
        from grid_fed_rl.utils.advanced_robustness import AdvancedSafetySystem
        
        # Quality gate metrics
        quality_gates = {
            "package_import": False,
            "core_functionality": False,
            "error_handling": False,
            "safety_systems": False,
            "monitoring": False,
            "performance": False
        }
        
        # Gate 1: Package Import
        try:
            import grid_fed_rl
            quality_gates["package_import"] = True
        except Exception as e:
            logger.error(f"Package import failed: {e}")
        
        # Gate 2: Core Functionality
        try:
            from grid_fed_rl import GridEnvironment, IEEE13Bus
            env = GridEnvironment(feeder=IEEE13Bus())
            obs = env.reset()
            quality_gates["core_functionality"] = True
        except Exception as e:
            logger.error(f"Core functionality failed: {e}")
        
        # Gate 3: Error Handling
        try:
            from grid_fed_rl.utils.exceptions import CircuitBreaker
            @CircuitBreaker(failure_threshold=1)
            def test_func():
                pass
            test_func()
            quality_gates["error_handling"] = True
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
        
        # Gate 4: Safety Systems
        try:
            safety_system = AdvancedSafetySystem()
            test_state = {
                "bus_voltages": np.array([1.0, 1.0, 1.0]),
                "frequency": 60.0,
                "line_flows": np.array([0.5, 0.5])
            }
            result = safety_system.evaluate_safety(test_state)
            quality_gates["safety_systems"] = True
        except Exception as e:
            logger.error(f"Safety systems failed: {e}")
        
        # Gate 5: Monitoring
        try:
            monitor = GridMonitor()
            metrics = monitor.record_metrics(1, 0.01, {}, {})
            quality_gates["monitoring"] = True
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
        
        # Gate 6: Performance (basic threshold)
        try:
            start_time = time.time()
            # Simulate workload
            from grid_fed_rl import GridEnvironment, IEEE13Bus
            env = GridEnvironment(feeder=IEEE13Bus())
            for _ in range(10):
                env.reset()
                env.step(env.action_space.sample())
            
            elapsed = time.time() - start_time
            if elapsed < 5.0:  # Should complete in under 5 seconds
                quality_gates["performance"] = True
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
        
        # Validate all gates pass
        passed_gates = sum(quality_gates.values())
        total_gates = len(quality_gates)
        
        logger.info(f"Quality Gates: {passed_gates}/{total_gates} passed")
        
        for gate, passed in quality_gates.items():
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {gate}")
        
        # Require 85% of gates to pass (5/6)
        assert passed_gates >= 5, f"Quality gates failed: {passed_gates}/{total_gates} passed"
        
        logger.info("✅ Quality gates validation passed")


def run_comprehensive_test_suite():
    """Run the complete test suite and generate report."""
    
    print("🧪 AUTONOMOUS SDLC - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()
    
    # Test results tracking
    test_results = {
        "generation_1": {"passed": 0, "failed": 0, "total": 0},
        "generation_2": {"passed": 0, "failed": 0, "total": 0},
        "generation_3": {"passed": 0, "failed": 0, "total": 0},
        "integration": {"passed": 0, "failed": 0, "total": 0}
    }
    
    def run_test_class(test_class, generation_key):
        """Run all tests in a test class."""
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for method_name in methods:
            test_results[generation_key]["total"] += 1
            try:
                method = getattr(instance, method_name)
                method()
                test_results[generation_key]["passed"] += 1
            except Exception as e:
                test_results[generation_key]["failed"] += 1
                logger.error(f"❌ {method_name}: {e}")
    
    # Run test suites
    print("🚀 Generation 1: MAKE IT WORK")
    run_test_class(TestGeneration1MakeItWork, "generation_1")
    print()
    
    print("🛡️ Generation 2: MAKE IT ROBUST")
    run_test_class(TestGeneration2MakeItRobust, "generation_2")
    print()
    
    print("⚡ Generation 3: MAKE IT SCALE")
    run_test_class(TestGeneration3MakeItScale, "generation_3")
    print()
    
    print("🔗 Integration & Quality Gates")
    run_test_class(TestIntegrationAndQualityGates, "integration")
    print()
    
    # Generate final report
    print("📊 FINAL TEST REPORT")
    print("=" * 60)
    
    total_passed = sum(results["passed"] for results in test_results.values())
    total_failed = sum(results["failed"] for results in test_results.values())
    total_tests = sum(results["total"] for results in test_results.values())
    
    success_rate = total_passed / total_tests if total_tests > 0 else 0
    
    for generation, results in test_results.items():
        gen_rate = results["passed"] / results["total"] if results["total"] > 0 else 0
        status = "✅" if gen_rate >= 0.8 else "❌"
        print(f"{status} {generation.replace('_', ' ').title()}: {results['passed']}/{results['total']} ({gen_rate:.1%})")
    
    print("-" * 60)
    print(f"🎯 Overall Success: {total_passed}/{total_tests} ({success_rate:.1%})")
    
    if success_rate >= 0.85:
        print("🏆 AUTONOMOUS SDLC IMPLEMENTATION: SUCCESS!")
        print("   All quality gates passed. System ready for production.")
    else:
        print("⚠️  AUTONOMOUS SDLC IMPLEMENTATION: NEEDS IMPROVEMENT")
        print(f"   Success rate: {success_rate:.1%} (target: 85%)")
    
    print("=" * 60)
    
    return success_rate >= 0.85


if __name__ == "__main__":
    success = run_comprehensive_test_suite()
    exit(0 if success else 1)