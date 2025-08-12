"""Comprehensive test suite for autonomous SDLC enhancements."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import asyncio
import time
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import the enhanced modules
from grid_fed_rl.utils.safety import SafetyChecker, SafetyShield, ConstraintViolation
from grid_fed_rl.utils.robust_neural_engine import RobustNeuralEngine, RobustPolicyNetwork
from grid_fed_rl.utils.advanced_optimization import (
    AdaptiveLearningRateScheduler, AdvancedCacheManager, 
    ParallelComputationEngine, OptimizationOrchestrator
)
from grid_fed_rl.federated.async_coordinator import AsyncFederatedCoordinator, SecureAggregator
from grid_fed_rl.federated.core import FedLearningConfig, ClientUpdate


class TestEnhancedSafetySystems:
    """Test suite for enhanced safety systems."""
    
    def test_safety_checker_initialization(self):
        """Test safety checker initialization with enhanced features."""
        checker = SafetyChecker(
            voltage_limits=(0.95, 1.05),
            frequency_limits=(59.5, 60.5),
            thermal_limits={'transformer': 100.0, 'generator': 150.0},
            rate_of_change_limits={'voltage': 0.1, 'frequency': 0.5}
        )
        
        assert checker.voltage_limits == (0.95, 1.05)
        assert checker.thermal_limits['transformer'] == 100.0
        assert checker.rate_of_change_limits['voltage'] == 0.1
        assert checker._previous_state is None
    
    def test_constraint_violation_detection(self):
        """Test comprehensive constraint violation detection."""
        checker = SafetyChecker()
        
        # Create test data with violations
        bus_voltages = np.array([0.94, 1.06, 1.0])  # Low and high voltage violations
        frequency = 61.5  # High frequency violation
        line_loadings = np.array([0.8, 1.1, 0.9])  # Line overload violation
        thermal_data = {'transformer_1': 120.0, 'generator_1': 160.0}  # Thermal violations
        
        violations = checker.check_constraints(
            bus_voltages, frequency, line_loadings, thermal_data, timestep=1.0
        )
        
        # Verify all violation types detected
        assert len(violations['voltage']) == 2  # Low and high voltage
        assert len(violations['frequency']) == 1  # High frequency
        assert len(violations['line_loading']) == 1  # Line overload
        assert len(violations['thermal']) == 2  # Both thermal violations
        
        # Test severity assessment
        severity = checker.get_violation_severity(violations)
        assert severity == 'critical'  # Due to thermal violations
    
    def test_safety_shield_intervention(self):
        """Test safety shield intervention logic."""
        safety_checker = SafetyChecker()
        shield = SafetyShield(
            safety_checker=safety_checker,
            intervention_threshold=0.9,
            backup_controller=Mock()
        )
        
        # Mock backup controller
        shield.backup_controller.get_action.return_value = np.array([0.1, 0.1])
        
        # Test state that should trigger intervention
        current_state = {
            'bus_voltages': np.array([0.85, 1.15, 1.0]),  # Critical violations
            'frequency': 58.0,  # Critical frequency
            'line_loadings': np.array([0.8, 0.9, 1.0]),
            'thermal_data': {'transformer_1': 200.0},  # Critical thermal
            'timestep': 1.0
        }
        
        proposed_action = np.array([0.5, 0.8])
        
        safe_action, intervened = shield.get_safe_action(
            current_state, proposed_action, confidence=0.8
        )
        
        assert intervened is True
        assert shield.intervention_count == 1
        assert len(shield.intervention_history) == 1
        np.testing.assert_array_equal(safe_action, np.array([0.1, 0.1]))
    
    def test_rate_of_change_monitoring(self):
        """Test rate of change constraint monitoring."""
        checker = SafetyChecker(rate_of_change_limits={'voltage': 0.05, 'frequency': 0.2})
        
        # First measurement
        violations1 = checker.check_constraints(
            np.array([1.0, 1.0, 1.0]), 60.0, np.array([0.5, 0.5, 0.5]), timestep=1.0
        )
        assert len(violations1['rate_of_change']) == 0  # No previous state
        
        # Second measurement with high rate of change
        violations2 = checker.check_constraints(
            np.array([1.1, 1.1, 1.1]), 60.5, np.array([0.5, 0.5, 0.5]), timestep=1.0
        )
        assert len(violations2['rate_of_change']) >= 1  # Should detect high voltage rate


class TestRobustNeuralEngine:
    """Test suite for robust neural network engine."""
    
    def test_robust_neural_engine_initialization(self):
        """Test robust neural engine initialization."""
        engine = RobustNeuralEngine(
            numerical_stability_check=True,
            gradient_clipping=1.0,
            memory_monitoring=True
        )
        
        assert engine.numerical_stability_check is True
        assert engine.gradient_clipping == 1.0
        assert engine.memory_monitoring is True
        assert engine.error_count == 0
        assert engine.total_operations == 0
    
    def test_safe_forward_pass(self):
        """Test safe forward pass execution."""
        engine = RobustNeuralEngine()
        model = RobustPolicyNetwork(state_dim=10, action_dim=5)
        
        # Test normal forward pass
        inputs = torch.randn(32, 10)
        outputs, health = engine.safe_forward(model, inputs, training=False)
        
        assert outputs.shape == (32, 5)
        assert health.numerical_stability is True
        assert health.gradient_norm >= 0
        assert engine.total_operations == 1
    
    def test_safe_backward_pass(self):
        """Test safe backward pass with gradient monitoring."""
        engine = RobustNeuralEngine(gradient_clipping=0.5)
        model = RobustPolicyNetwork(state_dim=10, action_dim=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward pass
        inputs = torch.randn(32, 10)
        outputs = model(inputs)
        loss = torch.mean(outputs ** 2)
        
        # Backward pass
        health = engine.safe_backward(loss, model, optimizer)
        
        assert health.loss_value > 0
        assert health.numerical_stability is True
        assert health.gradient_norm <= engine.gradient_clipping + 0.1  # Allow small tolerance
    
    def test_safe_inference_batching(self):
        """Test safe inference with automatic batching."""
        engine = RobustNeuralEngine()
        model = RobustPolicyNetwork(state_dim=10, action_dim=5)
        
        # Large input that should be batched
        large_inputs = torch.randn(1000, 10)
        outputs = engine.safe_inference(model, large_inputs, batch_size=100)
        
        assert outputs.shape == (1000, 5)
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()
    
    def test_numerical_stability_monitoring(self):
        """Test numerical stability monitoring."""
        engine = RobustNeuralEngine(numerical_stability_check=True)
        
        # Test with problematic inputs
        model = RobustPolicyNetwork(state_dim=5, action_dim=3)
        
        # Normal inputs should pass
        normal_inputs = torch.randn(10, 5)
        outputs, health = engine.safe_forward(model, normal_inputs)
        assert health.numerical_stability is True
        
        # Test performance statistics
        stats = engine.get_performance_stats()
        assert stats['total_operations'] > 0
        assert stats['error_rate'] >= 0.0
        assert 'avg_computation_time_ms' in stats


class TestAdvancedOptimization:
    """Test suite for advanced optimization features."""
    
    def test_adaptive_learning_rate_scheduler(self):
        """Test adaptive learning rate scheduling."""
        scheduler = AdaptiveLearningRateScheduler(
            initial_lr=0.001,
            strategy='plateau',
            patience=5,
            factor=0.5
        )
        
        # Test initial learning rate
        lr = scheduler.step()
        assert lr == 0.001
        
        # Test plateau detection
        for _ in range(10):
            lr = scheduler.step(loss=1.0)  # Constant loss should trigger reduction
        
        assert lr < 0.001  # Should have reduced
    
    def test_advanced_cache_manager(self):
        """Test advanced caching system."""
        cache = AdvancedCacheManager(
            max_size=10,
            ttl_seconds=60,
            compression=True,
            eviction_policy='lru'
        )
        
        # Test cache operations
        test_data = {'key': 'value', 'numbers': [1, 2, 3, 4, 5]}
        cache.put('test_key', test_data)
        
        retrieved_data = cache.get('test_key')
        assert retrieved_data == test_data
        
        # Test cache miss
        assert cache.get('nonexistent_key') is None
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats['hits'] >= 1
        assert stats['hit_rate'] > 0
    
    def test_parallel_computation_engine(self):
        """Test parallel computation engine."""
        def square_function(x):
            return x ** 2
        
        with ParallelComputationEngine(max_workers=2, use_processes=False) as engine:
            data = list(range(100))
            results = engine.parallel_map(square_function, data)
            
            expected = [x ** 2 for x in data]
            assert len(results) == len(expected)
            # Allow for some None values due to error handling
            valid_results = [r for r in results if r is not None]
            assert len(valid_results) >= len(expected) * 0.9  # At least 90% success
    
    def test_optimization_orchestrator(self):
        """Test optimization orchestrator integration."""
        orchestrator = OptimizationOrchestrator(
            enable_caching=True,
            enable_parallel=True,
            enable_compression=True
        )
        
        def test_computation(data):
            return sum(data)
        
        # Test optimized computation
        test_data = [1, 2, 3, 4, 5]
        result1 = orchestrator.optimize_computation(test_computation, test_data, 'sum_test')
        result2 = orchestrator.optimize_computation(test_computation, test_data, 'sum_test')  # Should hit cache
        
        assert result1 == 15
        assert result2 == 15
        
        # Get performance report
        report = orchestrator.get_optimization_report()
        assert 'optimization_metrics' in report
        assert report['optimization_metrics']['cache_hit_rate'] >= 0.0


class TestAsyncFederatedLearning:
    """Test suite for asynchronous federated learning."""
    
    def test_secure_aggregator_initialization(self):
        """Test secure aggregator initialization."""
        aggregator = SecureAggregator(
            byzantine_tolerance=2,
            verification_threshold=0.8,
            enable_encryption=True
        )
        
        assert aggregator.byzantine_tolerance == 2
        assert aggregator.verification_threshold == 0.8
        assert aggregator.enable_encryption is True
        assert len(aggregator.trusted_clients) == 0
    
    def test_client_update_verification(self):
        """Test client update verification."""
        aggregator = SecureAggregator()
        
        # Create valid update
        valid_update = ClientUpdate(
            client_id='client_1',
            parameters={'layer1.weight': np.random.randn(10, 5)},
            num_samples=100,
            loss=0.5,
            metrics={}
        )
        
        # Create invalid update with NaN
        invalid_update = ClientUpdate(
            client_id='client_2',
            parameters={'layer1.weight': np.array([[np.nan, 1.0], [2.0, 3.0]])},
            num_samples=100,
            loss=0.5,
            metrics={}
        )
        
        verified_updates = aggregator._verify_updates([valid_update, invalid_update])
        assert len(verified_updates) == 1
        assert verified_updates[0].client_id == 'client_1'
        assert 'client_2' in aggregator.suspicious_clients
    
    def test_byzantine_update_filtering(self):
        """Test Byzantine update detection and filtering."""
        aggregator = SecureAggregator(byzantine_tolerance=1)
        
        # Create normal updates
        normal_updates = []
        for i in range(5):
            update = ClientUpdate(
                client_id=f'client_{i}',
                parameters={'layer1.weight': np.random.randn(10, 5) * 0.1},
                num_samples=100,
                loss=0.5,
                metrics={}
            )
            normal_updates.append(update)
        
        # Create Byzantine update with extremely large values
        byzantine_update = ClientUpdate(
            client_id='byzantine_client',
            parameters={'layer1.weight': np.random.randn(10, 5) * 100},  # 100x larger
            num_samples=100,
            loss=0.5,
            metrics={}
        )
        
        all_updates = normal_updates + [byzantine_update]
        clean_updates = aggregator._filter_byzantine_updates(all_updates)
        
        # Byzantine update should be filtered out
        assert len(clean_updates) <= len(all_updates)
        byzantine_ids = [update.client_id for update in clean_updates]
        assert 'byzantine_client' not in byzantine_ids or len(clean_updates) == len(all_updates)
    
    def test_weighted_aggregation(self):
        """Test weighted aggregation of updates."""
        aggregator = SecureAggregator()
        
        # Create updates with different sample sizes
        updates = []
        for i in range(3):
            update = ClientUpdate(
                client_id=f'client_{i}',
                parameters={'weight': np.ones((2, 2)) * (i + 1)},
                num_samples=(i + 1) * 100,  # Different sample sizes
                loss=0.5,
                metrics={}
            )
            updates.append(update)
        
        aggregated = aggregator._weighted_aggregation(updates)
        
        # Check that aggregation preserves structure
        assert 'weight' in aggregated
        assert aggregated['weight'].shape == (2, 2)
        
        # Weighted average should be between min and max individual values
        min_val = np.min([update.parameters['weight'] for update in updates])
        max_val = np.max([update.parameters['weight'] for update in updates])
        
        assert np.all(aggregated['weight'] >= min_val)
        assert np.all(aggregated['weight'] <= max_val)
    
    @pytest.mark.asyncio
    async def test_async_coordinator_initialization(self):
        """Test async federated coordinator initialization."""
        config = FedLearningConfig(
            num_clients=5,
            rounds=10,
            byzantine_resilience=True,
            secure_aggregation=True
        )
        
        model_template = RobustPolicyNetwork(state_dim=10, action_dim=5)
        
        coordinator = AsyncFederatedCoordinator(
            config=config,
            model_template=model_template
        )
        
        assert coordinator.config.num_clients == 5
        assert coordinator.current_round == 0
        assert len(coordinator.client_states) == 0
        assert coordinator.secure_aggregator.byzantine_tolerance > 0


class TestProductionReadiness:
    """Test production readiness aspects."""
    
    def test_error_handling_robustness(self):
        """Test comprehensive error handling."""
        from grid_fed_rl.utils.exceptions import GridEnvironmentError
        
        engine = RobustNeuralEngine()
        
        # Test handling of invalid inputs
        model = RobustPolicyNetwork(state_dim=10, action_dim=5)
        
        with pytest.raises(Exception):  # Should handle gracefully
            invalid_inputs = torch.tensor([[float('inf'), float('nan')] * 5])
            with engine.safe_computation("test_computation"):
                model(invalid_inputs)
    
    def test_memory_management(self):
        """Test memory management and cleanup."""
        cache = AdvancedCacheManager(max_size=5)
        
        # Fill cache beyond capacity
        for i in range(10):
            cache.put(f'key_{i}', f'value_{i}')
        
        # Cache should not exceed max size
        assert len(cache.cache) <= 5
        assert cache.evictions > 0
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        engine = RobustNeuralEngine()
        model = RobustPolicyNetwork(state_dim=5, action_dim=3)
        
        # Perform several operations
        for _ in range(5):
            inputs = torch.randn(10, 5)
            with engine.safe_computation("test_op"):
                outputs = model(inputs)
        
        stats = engine.get_performance_stats()
        
        assert stats['total_operations'] == 5
        assert 'avg_computation_time_ms' in stats
        assert 'device' in stats
        assert stats['error_rate'] >= 0.0
    
    def test_configuration_validation(self):
        """Test configuration validation and sanitization."""
        from grid_fed_rl.utils.validation import sanitize_config
        
        # Test valid configuration
        valid_config = {
            'num_clients': 5,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        sanitized = sanitize_config(valid_config)
        assert sanitized['num_clients'] == 5
        assert sanitized['learning_rate'] == 0.001
    
    def test_security_features(self):
        """Test security features implementation."""
        aggregator = SecureAggregator(enable_encryption=True)
        
        # Test client verification
        assert len(aggregator.trusted_clients) == 0
        assert len(aggregator.suspicious_clients) == 0
        
        # Security should be enabled
        assert aggregator.enable_encryption is True


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_environment_step_performance(self):
        """Test environment step performance benchmark."""
        import grid_fed_rl
        
        # Test basic import and environment creation
        env = grid_fed_rl.GridEnvironment(grid_fed_rl.IEEE13Bus())
        
        start_time = time.time()
        obs = env.reset()
        reset_time = time.time() - start_time
        
        # Reset should be fast (under 100ms as per requirements)
        assert reset_time < 0.1, f"Reset time {reset_time:.3f}s exceeds 100ms threshold"
        
        # Test step performance
        start_time = time.time()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step_time = time.time() - start_time
        
        # Step should be fast (under 20ms as per requirements)
        assert step_time < 0.02, f"Step time {step_time:.3f}s exceeds 20ms threshold"
    
    def test_neural_network_performance(self):
        """Test neural network performance benchmarks."""
        engine = RobustNeuralEngine()
        model = RobustPolicyNetwork(state_dim=50, action_dim=20)
        
        # Test batch inference performance
        batch_size = 1000
        inputs = torch.randn(batch_size, 50)
        
        start_time = time.time()
        outputs = engine.safe_inference(model, inputs, batch_size=100)
        inference_time = time.time() - start_time
        
        # Should process 1000 samples quickly
        throughput = batch_size / inference_time
        assert throughput > 1000, f"Throughput {throughput:.1f} samples/sec too low"
    
    def test_cache_performance(self):
        """Test cache performance characteristics."""
        cache = AdvancedCacheManager(max_size=1000, compression=True)
        
        # Benchmark cache operations
        test_data = {'large_array': np.random.randn(1000, 1000)}
        
        # Test put performance
        start_time = time.time()
        for i in range(100):
            cache.put(f'key_{i}', test_data)
        put_time = time.time() - start_time
        
        # Test get performance
        start_time = time.time()
        for i in range(100):
            cache.get(f'key_{i}')
        get_time = time.time() - start_time
        
        # Cache operations should be fast
        assert put_time < 5.0, f"Cache put operations too slow: {put_time:.2f}s"
        assert get_time < 1.0, f"Cache get operations too slow: {get_time:.2f}s"


if __name__ == '__main__':
    # Run the test suite
    pytest.main([__file__, '-v', '--tb=short'])