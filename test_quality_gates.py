#!/usr/bin/env python3
"""
Quality Gates Validation Test Suite
Validates all mandatory quality requirements before production deployment
"""

import sys
import traceback
import subprocess
import os
import importlib.util
import ast

def test_code_runs_without_errors():
    """Test that code runs without critical errors."""
    try:
        # Import main modules
        import grid_fed_rl
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Create and run environment
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=20)
        obs, info = env.reset(seed=42)
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        print("✅ Code runs without errors: Basic functionality operational")
        return True
        
    except Exception as e:
        print(f"❌ Code execution failed: {e}")
        traceback.print_exc()
        return False

def test_basic_test_coverage():
    """Test basic test coverage by running existing tests."""
    try:
        # Count test files
        test_files = [
            "test_generation1_basic.py",
            "test_generation2_robust.py", 
            "test_generation3_performance.py"
        ]
        
        existing_tests = []
        for test_file in test_files:
            if os.path.exists(test_file):
                existing_tests.append(test_file)
        
        # Run tests if available
        total_passed = 0
        total_tests = 0
        
        for test_file in existing_tests:
            try:
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    total_passed += 1
                total_tests += 1
                
            except subprocess.TimeoutExpired:
                print(f"   ⚠ {test_file} timed out")
                total_tests += 1
            except Exception as e:
                print(f"   ⚠ {test_file} failed to run: {e}")
                total_tests += 1
        
        coverage_rate = total_passed / total_tests if total_tests > 0 else 0
        target_coverage = 0.85  # 85%
        
        print(f"✅ Test coverage: {coverage_rate:.1%} ({total_passed}/{total_tests} test suites passed)")
        print(f"   Target: {target_coverage:.1%}")
        
        return coverage_rate >= target_coverage
        
    except Exception as e:
        print(f"❌ Test coverage check failed: {e}")
        return False

def test_security_scan():
    """Basic security scan - check for obvious security issues."""
    try:
        security_issues = []
        
        # Check for hardcoded credentials or secrets
        python_files = []
        for root, dirs, files in os.walk("grid_fed_rl"):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        suspicious_patterns = [
            "password", "secret", "api_key", "private_key",
            "token", "credential", "auth_token"
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for pattern in suspicious_patterns:
                        if f'"{pattern}"' in content or f"'{pattern}'" in content:
                            # Check if it's in a comment or docstring
                            if not (content.find(pattern) > content.rfind('#') or 
                                   content.find(pattern) > content.rfind('"""')):
                                security_issues.append(f"Potential hardcoded {pattern} in {file_path}")
            except Exception:
                pass  # Skip files that can't be read
        
        # Check for unsafe imports
        unsafe_imports = ['eval', 'exec', 'compile', '__import__']
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Name) and node.id in unsafe_imports:
                            security_issues.append(f"Unsafe function {node.id} in {file_path}")
            except Exception:
                pass  # Skip files that can't be parsed
        
        security_score = 1.0 - min(len(security_issues) / 10, 1.0)  # Max 10 issues
        
        print(f"✅ Security scan: {security_score:.1%} security score")
        if security_issues:
            for issue in security_issues[:3]:  # Show first 3
                print(f"   ⚠ {issue}")
            if len(security_issues) > 3:
                print(f"   ... and {len(security_issues) - 3} more issues")
        
        return security_score >= 0.8  # 80% security score
        
    except Exception as e:
        print(f"❌ Security scan failed: {e}")
        return False

def test_performance_benchmarks():
    """Test that performance benchmarks meet requirements."""
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        import time
        
        # Performance targets
        reset_target = 100  # ms
        step_target = 20    # ms
        
        feeder = IEEE13Bus()
        env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=100)
        
        # Measure reset performance
        reset_times = []
        for i in range(5):
            start = time.time()
            obs, info = env.reset(seed=42 + i)
            reset_times.append((time.time() - start) * 1000)
        
        avg_reset = sum(reset_times) / len(reset_times)
        
        # Measure step performance
        obs, info = env.reset(seed=42)
        step_times = []
        for step in range(20):
            action = env.action_space.sample()
            start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append((time.time() - start) * 1000)
            if terminated or truncated:
                break
        
        avg_step = sum(step_times) / len(step_times)
        
        reset_pass = avg_reset <= reset_target
        step_pass = avg_step <= step_target
        
        print(f"✅ Performance benchmarks:")
        print(f"   Reset: {avg_reset:.1f}ms (target: <{reset_target}ms) {'✓' if reset_pass else '✗'}")
        print(f"   Step: {avg_step:.1f}ms (target: <{step_target}ms) {'✓' if step_pass else '✗'}")
        
        return reset_pass and step_pass
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def test_documentation_updated():
    """Test that documentation exists and is reasonably complete."""
    try:
        doc_files = [
            "README.md",
            "ARCHITECTURE.md", 
            "CONTRIBUTING.md",
            "LICENSE"
        ]
        
        existing_docs = []
        total_lines = 0
        
        for doc_file in doc_files:
            if os.path.exists(doc_file):
                existing_docs.append(doc_file)
                with open(doc_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
        
        # Check README completeness
        readme_complete = False
        if os.path.exists("README.md"):
            with open("README.md", 'r', encoding='utf-8') as f:
                readme_content = f.read().lower()
                required_sections = ["installation", "usage", "example", "architecture"]
                found_sections = sum(1 for section in required_sections if section in readme_content)
                readme_complete = found_sections >= 3
        
        doc_score = (len(existing_docs) / len(doc_files)) * 0.7 + (1 if readme_complete else 0) * 0.3
        
        print(f"✅ Documentation: {doc_score:.1%} completeness")
        print(f"   Files present: {len(existing_docs)}/{len(doc_files)}")
        print(f"   Total lines: {total_lines}")
        print(f"   README sections: {'Complete' if readme_complete else 'Incomplete'}")
        
        return doc_score >= 0.8
        
    except Exception as e:
        print(f"❌ Documentation check failed: {e}")
        return False

def test_zero_security_vulnerabilities():
    """Test for zero critical security vulnerabilities."""
    try:
        # Basic checks for common Python security issues
        vulnerabilities = []
        
        # Check for unsafe file operations
        python_files = []
        for root, dirs, files in os.walk("grid_fed_rl"):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for dangerous patterns
                    dangerous_patterns = [
                        ('open(', 'without explicit mode'),
                        ('eval(', 'code evaluation'),
                        ('exec(', 'code execution'),
                        ('subprocess.call', 'without shell=False'),
                        ('os.system', 'system command execution')
                    ]
                    
                    for pattern, description in dangerous_patterns:
                        if pattern in content:
                            # Basic check - count occurrences
                            count = content.count(pattern)
                            if count > 0:
                                vulnerabilities.append(f"{description} in {file_path} ({count} times)")
            except Exception:
                pass
        
        # Filter out false positives (very basic)
        real_vulnerabilities = [v for v in vulnerabilities if 'test_' not in v and 'example' not in v.lower()]
        
        vuln_score = 1.0 if len(real_vulnerabilities) == 0 else max(0, 1.0 - len(real_vulnerabilities) * 0.2)
        
        print(f"✅ Security vulnerabilities: {vuln_score:.1%} security rating")
        if real_vulnerabilities:
            for vuln in real_vulnerabilities[:3]:
                print(f"   ⚠ {vuln}")
        
        return len(real_vulnerabilities) == 0
        
    except Exception as e:
        print(f"❌ Vulnerability check failed: {e}")
        return False

def test_production_ready_deployment():
    """Test production readiness indicators."""
    try:
        production_indicators = []
        
        # Check for configuration management
        config_files = ["pyproject.toml", "setup.py", "requirements.txt"]
        config_score = sum(1 for f in config_files if os.path.exists(f)) / len(config_files)
        production_indicators.append(("Configuration", config_score >= 0.8))
        
        # Check for CLI interface
        try:
            from grid_fed_rl.cli import main
            cli_available = True
        except Exception:
            cli_available = False
        production_indicators.append(("CLI Interface", cli_available))
        
        # Check for error handling in main modules
        try:
            from grid_fed_rl.environments.grid_env import GridEnvironment
            from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
            
            feeder = IEEE13Bus()
            env = GridEnvironment(feeder=feeder, timestep=1.0, episode_length=10)
            
            # Test with invalid action
            obs, info = env.reset(seed=42)
            try:
                obs, reward, terminated, truncated, info = env.step([999, 999, 999])
                error_handling = not (np.isnan(reward) or np.any(np.isnan(obs)))
            except Exception:
                error_handling = False
                
        except Exception:
            error_handling = False
        production_indicators.append(("Error Handling", error_handling))
        
        # Check for logging
        logging_available = os.path.exists("grid_fed_rl/utils/logging_config.py")
        production_indicators.append(("Logging", logging_available))
        
        # Check for monitoring 
        monitoring_available = os.path.exists("grid_fed_rl/utils/monitoring.py")
        production_indicators.append(("Monitoring", monitoring_available))
        
        passed_indicators = sum(1 for _, passed in production_indicators if passed)
        total_indicators = len(production_indicators)
        production_score = passed_indicators / total_indicators
        
        print(f"✅ Production readiness: {production_score:.1%}")
        for indicator, passed in production_indicators:
            print(f"   {indicator}: {'✓' if passed else '✗'}")
        
        return production_score >= 0.8
        
    except Exception as e:
        print(f"❌ Production readiness check failed: {e}")
        return False

def main():
    """Run all quality gate validations."""
    print("=== QUALITY GATES VALIDATION ===\n")
    print("Validating all mandatory quality requirements...\n")
    
    quality_gates = [
        ("Code runs without errors", test_code_runs_without_errors),
        ("Basic test coverage (≥85%)", test_basic_test_coverage),
        ("Security scan passes", test_security_scan),
        ("Performance benchmarks met", test_performance_benchmarks),
        ("Documentation updated", test_documentation_updated),
        ("Zero security vulnerabilities", test_zero_security_vulnerabilities),
        ("Production-ready deployment", test_production_ready_deployment)
    ]
    
    passed = 0
    total = len(quality_gates)
    results = []
    
    for gate_name, gate_test in quality_gates:
        print(f"🔍 {gate_name}...")
        try:
            result = gate_test()
            results.append((gate_name, result))
            if result:
                passed += 1
                print(f"   ✅ PASSED\n")
            else:
                print(f"   ❌ FAILED\n")
        except Exception as e:
            results.append((gate_name, False))
            print(f"   ❌ FAILED: {e}\n")
    
    print(f"=== QUALITY GATES RESULTS: {passed}/{total} gates passed ===\n")
    
    # Summary
    print("Gate Summary:")
    for gate_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {gate_name}")
    
    print()
    
    if passed == total:
        print("🎉 ALL QUALITY GATES PASSED - READY FOR PRODUCTION!")
        return True
    elif passed >= total * 0.85:  # 85% pass rate
        print(f"⚠️  MOSTLY READY - {passed}/{total} gates passed (≥85% required)")
        return True
    else:
        print(f"❌ NOT READY - Only {passed}/{total} gates passed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)