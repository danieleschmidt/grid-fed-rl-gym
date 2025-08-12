"""Quick validation test for autonomous enhancements."""

import sys
import os
import time

def test_package_imports():
    """Test that enhanced package imports work correctly."""
    try:
        import grid_fed_rl
        print("✅ Core package import successful")
        
        # Test version
        print(f"📦 Package version: {grid_fed_rl.__version__}")
        
        # Test basic environment creation
        env = grid_fed_rl.GridEnvironment(grid_fed_rl.IEEE13Bus())
        print("✅ Basic environment creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_enhanced_modules():
    """Test enhanced module imports."""
    success_count = 0
    total_tests = 0
    
    # Test safety module
    total_tests += 1
    try:
        from grid_fed_rl.utils.safety import SafetyChecker, SafetyShield
        print("✅ Enhanced safety module import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Safety module import failed: {e}")
    
    # Test robust neural engine
    total_tests += 1
    try:
        from grid_fed_rl.utils.robust_neural_engine import RobustNeuralEngine
        print("✅ Robust neural engine module import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Robust neural engine import failed: {e}")
    
    # Test advanced optimization
    total_tests += 1
    try:
        from grid_fed_rl.utils.advanced_optimization import OptimizationOrchestrator
        print("✅ Advanced optimization module import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Advanced optimization import failed: {e}")
    
    # Test async federated learning
    total_tests += 1
    try:
        from grid_fed_rl.federated.async_coordinator import AsyncFederatedCoordinator
        print("✅ Async federated learning module import successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Async federated learning import failed: {e}")
    
    return success_count, total_tests

def test_performance_benchmark():
    """Quick performance benchmark."""
    try:
        import grid_fed_rl
        
        # Test environment reset performance
        env = grid_fed_rl.GridEnvironment(grid_fed_rl.IEEE13Bus())
        
        start_time = time.time()
        obs = env.reset()
        reset_time = time.time() - start_time
        
        print(f"🏃‍♂️ Environment reset time: {reset_time*1000:.1f}ms")
        
        # Test step performance
        start_time = time.time()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        step_time = time.time() - start_time
        
        print(f"🏃‍♂️ Environment step time: {step_time*1000:.1f}ms")
        
        # Performance criteria (from requirements)
        reset_ok = reset_time < 0.1  # Under 100ms
        step_ok = step_time < 0.02   # Under 20ms
        
        if reset_ok and step_ok:
            print("✅ Performance benchmarks PASSED")
            return True
        else:
            print(f"❌ Performance benchmarks FAILED (reset: {reset_ok}, step: {step_ok})")
            return False
            
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def test_safety_systems():
    """Test basic safety system functionality."""
    try:
        from grid_fed_rl.utils.safety import SafetyChecker
        import numpy as np
        
        # Create safety checker
        checker = SafetyChecker()
        
        # Test with safe values
        safe_violations = checker.check_constraints(
            bus_voltages=np.array([1.0, 1.0, 1.0]),
            frequency=60.0,
            line_loadings=np.array([0.5, 0.5, 0.5])
        )
        
        # Test with unsafe values
        unsafe_violations = checker.check_constraints(
            bus_voltages=np.array([0.8, 1.2, 1.0]),  # Violations
            frequency=62.0,  # Violation
            line_loadings=np.array([1.5, 0.5, 0.5])  # Violation
        )
        
        safe_total = sum(len(v) for v in safe_violations.values())
        unsafe_total = sum(len(v) for v in unsafe_violations.values())
        
        if safe_total == 0 and unsafe_total > 0:
            print("✅ Safety constraint detection working correctly")
            return True
        else:
            print(f"❌ Safety system test failed (safe: {safe_total}, unsafe: {unsafe_total})")
            return False
            
    except Exception as e:
        print(f"❌ Safety system test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Autonomous SDLC Enhancement Validation")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Package imports
    total_tests += 1
    if test_package_imports():
        tests_passed += 1
    print()
    
    # Test 2: Enhanced modules
    success_count, module_tests = test_enhanced_modules()
    tests_passed += success_count
    total_tests += module_tests
    print()
    
    # Test 3: Performance benchmark
    total_tests += 1
    if test_performance_benchmark():
        tests_passed += 1
    print()
    
    # Test 4: Safety systems
    total_tests += 1
    if test_safety_systems():
        tests_passed += 1
    print()
    
    # Final results
    print("=" * 50)
    print(f"📊 Validation Results: {tests_passed}/{total_tests} tests passed")
    
    success_rate = (tests_passed / total_tests) * 100
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 85:
        print("🎉 AUTONOMOUS ENHANCEMENTS VALIDATION PASSED!")
        print("✅ Quality gate threshold (85%) exceeded")
        return True
    else:
        print("❌ Validation failed - below 85% threshold")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)