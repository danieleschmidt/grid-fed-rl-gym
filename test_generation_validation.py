"""Comprehensive validation test for all three generations of the autonomous SDLC."""

import sys
import time
import traceback
from typing import Dict, Any, List, Tuple


def test_generation_1_basic_functionality() -> Dict[str, Any]:
    """Test Generation 1: Basic functionality working."""
    test_results = {
        'name': 'Generation 1: Basic Functionality',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        # Test core imports
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        from grid_fed_rl.environments.grid_env import GridEnvironment
        test_results['details'].append('✅ Core module imports successful')
        test_results['passed'] += 1
        
        # Test environment creation
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder)
        test_results['details'].append('✅ Environment creation successful')
        test_results['passed'] += 1
        
        # Test environment reset
        obs, info = env.reset()
        assert len(obs) > 0, "Observation should not be empty"
        test_results['details'].append(f'✅ Environment reset successful (obs_len={len(obs)})')
        test_results['passed'] += 1
        
        # Test environment step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        test_results['details'].append(f'✅ Environment step successful (reward={reward:.2f})')
        test_results['passed'] += 1
        
        # Test multiple steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()
        test_results['details'].append('✅ Multi-step execution successful')
        test_results['passed'] += 1
        
        # Test action/observation spaces
        assert hasattr(env.action_space, 'sample'), "Action space should have sample method"
        assert hasattr(env.observation_space, 'shape'), "Observation space should have shape"
        test_results['details'].append('✅ Action/observation spaces valid')
        test_results['passed'] += 1
        
    except Exception as e:
        test_results['details'].append(f'❌ Generation 1 test failed: {e}')
        test_results['failed'] += 1
        
    return test_results


def test_generation_2_robust_features() -> Dict[str, Any]:
    """Test Generation 2: Robust error handling and validation."""
    test_results = {
        'name': 'Generation 2: Robust Features',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        # Test robust validation imports
        from grid_fed_rl.utils.robust_validation import global_validator, global_error_handler
        test_results['details'].append('✅ Robust validation modules imported')
        test_results['passed'] += 1
        
        # Test enhanced logging
        from grid_fed_rl.utils.enhanced_logging import grid_logger, performance_monitor
        test_results['details'].append('✅ Enhanced logging modules imported')
        test_results['passed'] += 1
        
        # Test health monitoring
        from grid_fed_rl.utils.health_monitoring import system_health, system_watchdog
        test_results['details'].append('✅ Health monitoring modules imported')
        test_results['passed'] += 1
        
        # Test security hardening
        from grid_fed_rl.utils.security_hardening import input_sanitizer, security_monitor
        test_results['details'].append('✅ Security hardening modules imported')
        test_results['passed'] += 1
        
        # Test validation functionality
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        env = GridEnvironment(feeder=IEEE13Bus())
        
        # Test action validation
        valid_action = [0.5]
        validation_result = global_validator.validate_action(valid_action, env.action_space)
        assert validation_result.is_valid, "Valid action should pass validation"
        test_results['details'].append('✅ Action validation working')
        test_results['passed'] += 1
        
        # Test invalid action handling
        invalid_action = [float('inf')]
        validation_result = global_validator.validate_action(invalid_action, env.action_space)
        assert not validation_result.is_valid, "Invalid action should fail validation"
        test_results['details'].append('✅ Invalid action detection working')
        test_results['passed'] += 1
        
        # Test health monitoring
        health_report = system_health.get_health_report()
        assert 'overall_status' in health_report, "Health report should include overall status"
        test_results['details'].append(f'✅ Health monitoring working (status={health_report["overall_status"]})')
        test_results['passed'] += 1
        
        # Test error handling
        error_summary = global_error_handler.get_error_summary()
        assert 'total_errors' in error_summary, "Error summary should include total errors"
        test_results['details'].append('✅ Error handling system operational')
        test_results['passed'] += 1
        
    except Exception as e:
        test_results['details'].append(f'❌ Generation 2 test failed: {e}')
        test_results['failed'] += 1
        
    return test_results


def test_generation_3_optimization() -> Dict[str, Any]:
    """Test Generation 3: Performance optimization and scaling."""
    test_results = {
        'name': 'Generation 3: Optimization',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        # Test performance optimization imports
        from grid_fed_rl.utils.performance_optimization import (
            power_flow_cache, adaptive_optimizer, memory_optimizer, performance_profiler
        )
        test_results['details'].append('✅ Performance optimization modules imported')
        test_results['passed'] += 1
        
        # Test auto-scaling imports
        from grid_fed_rl.utils.auto_scaling import (
            worker_pool, resource_manager, load_balancer
        )
        test_results['details'].append('✅ Auto-scaling modules imported')
        test_results['passed'] += 1
        
        # Test parallel environment imports
        from grid_fed_rl.utils.parallel_environment import (
            VectorizedEnvironment, async_env_manager, default_parallel_config
        )
        test_results['details'].append('✅ Parallel environment modules imported')
        test_results['passed'] += 1
        
        # Test caching functionality
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        env = GridEnvironment(feeder=IEEE13Bus())
        env.reset()
        
        # Run steps to populate cache
        for i in range(3):
            action = env.action_space.sample()
            env.step(action)
            
        cache_stats = power_flow_cache.get_stats()
        assert cache_stats['total_requests'] > 0, "Cache should have recorded requests"
        test_results['details'].append(f'✅ Caching system working (requests={cache_stats["total_requests"]})')
        test_results['passed'] += 1
        
        # Test performance profiling
        perf_report = performance_profiler.get_performance_report()
        test_results['details'].append('✅ Performance profiling operational')
        test_results['passed'] += 1
        
        # Test resource management
        resource_report = resource_manager.get_resource_report()
        assert 'current_usage' in resource_report, "Resource report should include current usage"
        test_results['details'].append('✅ Resource management operational')
        test_results['passed'] += 1
        
        # Test worker pool
        pool_status = worker_pool.get_pool_status()
        assert pool_status['total_workers'] >= 1, "Worker pool should have at least one worker"
        test_results['details'].append(f'✅ Worker pool operational (workers={pool_status["total_workers"]})')
        test_results['passed'] += 1
        
        # Test load balancer
        lb_stats = load_balancer.get_load_balancer_stats()
        test_results['details'].append('✅ Load balancer operational')
        test_results['passed'] += 1
        
    except Exception as e:
        test_results['details'].append(f'❌ Generation 3 test failed: {e}')
        test_results['failed'] += 1
        
    return test_results


def test_integration_scenarios() -> Dict[str, Any]:
    """Test integration scenarios across all generations."""
    test_results = {
        'name': 'Integration Scenarios',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Test full integration scenario
        env = GridEnvironment(feeder=IEEE13Bus())
        
        # Episode with error handling and monitoring
        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if done or truncated:
                break
                
        test_results['details'].append(f'✅ Full episode completed ({step_count} steps, reward={total_reward:.2f})')
        test_results['passed'] += 1
        
        # Test constraint handling
        assert 'constraint_violations' in info, "Info should include constraint violations"
        test_results['details'].append('✅ Constraint monitoring working')
        test_results['passed'] += 1
        
        # Test power flow convergence
        assert 'power_flow_converged' in info, "Info should include power flow convergence"
        test_results['details'].append(f'✅ Power flow working (converged={info["power_flow_converged"]})')
        test_results['passed'] += 1
        
        # Test performance monitoring integration
        from grid_fed_rl.utils.health_monitoring import system_health
        health_report = system_health.get_health_report()
        
        simulation_speed = health_report['metrics'].get('simulation_speed', {})
        if 'value' in simulation_speed:
            test_results['details'].append(f'✅ Performance monitoring integrated (speed={simulation_speed["value"]:.2f}ms)')
        else:
            test_results['details'].append('✅ Performance monitoring structure ready')
        test_results['passed'] += 1
        
    except Exception as e:
        test_results['details'].append(f'❌ Integration test failed: {e}')
        test_results['failed'] += 1
        
    return test_results


def test_edge_cases_and_stress() -> Dict[str, Any]:
    """Test edge cases and stress scenarios."""
    test_results = {
        'name': 'Edge Cases and Stress Tests',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Test invalid actions
        env = GridEnvironment(feeder=IEEE13Bus())
        env.reset()
        
        # Test extremely large action
        try:
            large_action = [1e6]
            obs, reward, done, truncated, info = env.step(large_action)
            test_results['details'].append('✅ Large action handled gracefully')
            test_results['passed'] += 1
        except Exception as e:
            test_results['details'].append(f'❌ Large action handling failed: {e}')
            test_results['failed'] += 1
            
        # Test NaN action  
        try:
            nan_action = [float('nan')]
            obs, reward, done, truncated, info = env.step(nan_action)
            test_results['details'].append('✅ NaN action handled gracefully')
            test_results['passed'] += 1
        except Exception as e:
            test_results['details'].append(f'❌ NaN action handling failed: {e}')
            test_results['failed'] += 1
            
        # Test rapid reset/step cycles
        try:
            for i in range(5):
                env.reset()
                env.step(env.action_space.sample())
            test_results['details'].append('✅ Rapid reset/step cycles handled')
            test_results['passed'] += 1
        except Exception as e:
            test_results['details'].append(f'❌ Rapid cycles failed: {e}')
            test_results['failed'] += 1
            
        # Test long episodes
        try:
            env.reset()
            for i in range(50):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    env.reset()
            test_results['details'].append('✅ Long episode execution successful')
            test_results['passed'] += 1
        except Exception as e:
            test_results['details'].append(f'❌ Long episode failed: {e}')
            test_results['failed'] += 1
            
    except Exception as e:
        test_results['details'].append(f'❌ Edge case test setup failed: {e}')
        test_results['failed'] += 1
        
    return test_results


def run_comprehensive_quality_gates() -> Dict[str, Any]:
    """Run comprehensive quality gates validation."""
    print("🛡️ RUNNING COMPREHENSIVE QUALITY GATES VALIDATION")
    print("=" * 60)
    
    start_time = time.time()
    all_results = []
    
    # Run all test suites
    test_suites = [
        test_generation_1_basic_functionality,
        test_generation_2_robust_features,
        test_generation_3_optimization,
        test_integration_scenarios,
        test_edge_cases_and_stress
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_suite in test_suites:
        print(f"\\nRunning {test_suite.__name__}...")
        try:
            results = test_suite()
            all_results.append(results)
            
            print(f"📊 {results['name']}")
            print(f"   Passed: {results['passed']} | Failed: {results['failed']}")
            
            for detail in results['details']:
                print(f"   {detail}")
                
            total_passed += results['passed']
            total_failed += results['failed']
            
        except Exception as e:
            print(f"❌ Test suite {test_suite.__name__} crashed: {e}")
            traceback.print_exc()
            total_failed += 1
            
    duration = time.time() - start_time
    
    # Summary
    print("\\n" + "=" * 60)
    print("📈 QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")
    print(f"Duration: {duration:.2f} seconds")
    
    overall_status = "PASSED" if total_failed == 0 else "FAILED"
    print(f"\\n🎯 OVERALL STATUS: {overall_status}")
    
    return {
        'overall_status': overall_status,
        'total_passed': total_passed,
        'total_failed': total_failed,
        'success_rate': total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0,
        'duration': duration,
        'detailed_results': all_results
    }


if __name__ == "__main__":
    results = run_comprehensive_quality_gates()
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] == 'PASSED' else 1
    sys.exit(exit_code)