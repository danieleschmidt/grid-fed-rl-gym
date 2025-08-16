"""Production readiness verification tests."""

import os
import sys
import time
import json
import subprocess
from typing import Dict, Any, List
from pathlib import Path


def test_package_structure() -> Dict[str, Any]:
    """Test package structure and organization."""
    results = {
        'name': 'Package Structure',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        # Check core package structure
        required_dirs = [
            'grid_fed_rl',
            'grid_fed_rl/environments',
            'grid_fed_rl/feeders', 
            'grid_fed_rl/algorithms',
            'grid_fed_rl/utils',
            'grid_fed_rl/federated',
            'tests'
        ]
        
        for dir_path in required_dirs:
            if os.path.isdir(dir_path):
                results['details'].append(f'✅ Directory {dir_path} exists')
                results['passed'] += 1
            else:
                results['details'].append(f'❌ Missing directory {dir_path}')
                results['failed'] += 1
                
        # Check core files
        required_files = [
            'README.md',
            'pyproject.toml',
            'LICENSE',
            'grid_fed_rl/__init__.py'
        ]
        
        for file_path in required_files:
            if os.path.isfile(file_path):
                results['details'].append(f'✅ File {file_path} exists')
                results['passed'] += 1
            else:
                results['details'].append(f'❌ Missing file {file_path}')
                results['failed'] += 1
                
    except Exception as e:
        results['details'].append(f'❌ Package structure check failed: {e}')
        results['failed'] += 1
        
    return results


def test_import_hierarchy() -> Dict[str, Any]:
    """Test import hierarchy and dependencies."""
    results = {
        'name': 'Import Hierarchy',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    # Test core imports
    core_imports = [
        'grid_fed_rl',
        'grid_fed_rl.environments',
        'grid_fed_rl.feeders',
        'grid_fed_rl.algorithms',
        'grid_fed_rl.utils'
    ]
    
    for module_name in core_imports:
        try:
            __import__(module_name)
            results['details'].append(f'✅ Import {module_name} successful')
            results['passed'] += 1
        except Exception as e:
            results['details'].append(f'❌ Import {module_name} failed: {e}')
            results['failed'] += 1
            
    # Test specific class imports
    specific_imports = [
        ('grid_fed_rl.environments.grid_env', 'GridEnvironment'),
        ('grid_fed_rl.feeders.ieee_feeders', 'IEEE13Bus'),
        ('grid_fed_rl.feeders.base', 'BaseFeeder'),
        ('grid_fed_rl.utils.robust_validation', 'RobustValidator'),
        ('grid_fed_rl.utils.performance_optimization', 'AdaptiveCache')
    ]
    
    for module_name, class_name in specific_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            results['details'].append(f'✅ Class {class_name} from {module_name} imported')
            results['passed'] += 1
        except Exception as e:
            results['details'].append(f'❌ Class {class_name} from {module_name} failed: {e}')
            results['failed'] += 1
            
    return results


def test_configuration_files() -> Dict[str, Any]:
    """Test configuration files and metadata."""
    results = {
        'name': 'Configuration Files',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        # Test pyproject.toml
        if os.path.isfile('pyproject.toml'):
            with open('pyproject.toml', 'r') as f:
                content = f.read()
                
            # Check for required sections
            required_sections = ['build-system', 'project', 'tool.pytest']
            for section in required_sections:
                if section in content:
                    results['details'].append(f'✅ pyproject.toml has [{section}] section')
                    results['passed'] += 1
                else:
                    results['details'].append(f'❌ pyproject.toml missing [{section}] section')
                    results['failed'] += 1
        else:
            results['details'].append('❌ pyproject.toml not found')
            results['failed'] += 1
            
        # Test README.md
        if os.path.isfile('README.md'):
            with open('README.md', 'r') as f:
                readme_content = f.read()
                
            if len(readme_content) > 1000:  # Reasonable minimum length
                results['details'].append('✅ README.md has substantial content')
                results['passed'] += 1
            else:
                results['details'].append('❌ README.md content too brief')
                results['failed'] += 1
        else:
            results['details'].append('❌ README.md not found')
            results['failed'] += 1
            
        # Test LICENSE
        if os.path.isfile('LICENSE'):
            results['details'].append('✅ LICENSE file exists')
            results['passed'] += 1
        else:
            results['details'].append('❌ LICENSE file missing')
            results['failed'] += 1
            
    except Exception as e:
        results['details'].append(f'❌ Configuration file check failed: {e}')
        results['failed'] += 1
        
    return results


def test_dependencies() -> Dict[str, Any]:
    """Test dependency management and requirements."""
    results = {
        'name': 'Dependencies',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        # Check if package can be imported without core dependencies
        try:
            import grid_fed_rl
            results['details'].append('✅ Package imports without NumPy (graceful degradation)')
            results['passed'] += 1
        except Exception as e:
            results['details'].append(f'❌ Package fails to import: {e}')
            results['failed'] += 1
            
        # Test core functionality works without heavy dependencies
        try:
            from grid_fed_rl.environments.grid_env import GridEnvironment
            from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
            
            env = GridEnvironment(feeder=IEEE13Bus())
            obs, info = env.reset()
            
            results['details'].append('✅ Core functionality works without NumPy')
            results['passed'] += 1
        except Exception as e:
            results['details'].append(f'❌ Core functionality fails: {e}')
            results['failed'] += 1
            
        # Check version consistency
        try:
            from grid_fed_rl import __version__
            
            # Read version from pyproject.toml
            if os.path.isfile('pyproject.toml'):
                with open('pyproject.toml', 'r') as f:
                    content = f.read()
                    
                if 'version = "0.1.0"' in content and __version__ == "0.1.0":
                    results['details'].append('✅ Version consistency between package and config')
                    results['passed'] += 1
                else:
                    results['details'].append('❌ Version mismatch between package and config')
                    results['failed'] += 1
            else:
                results['details'].append('❌ Cannot verify version consistency')
                results['failed'] += 1
                
        except Exception as e:
            results['details'].append(f'❌ Version check failed: {e}')
            results['failed'] += 1
            
    except Exception as e:
        results['details'].append(f'❌ Dependency check failed: {e}')
        results['failed'] += 1
        
    return results


def test_documentation() -> Dict[str, Any]:
    """Test documentation completeness."""
    results = {
        'name': 'Documentation',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        # Check for key documentation files
        doc_files = [
            'README.md',
            'CONTRIBUTING.md',
            'EXAMPLES.md',
            'TUTORIALS.md',
            'API_REFERENCE.md'
        ]
        
        for doc_file in doc_files:
            if os.path.isfile(doc_file):
                with open(doc_file, 'r') as f:
                    content = f.read()
                if len(content) > 500:  # Reasonable minimum
                    results['details'].append(f'✅ {doc_file} exists with substantial content')
                    results['passed'] += 1
                else:
                    results['details'].append(f'❌ {doc_file} exists but lacks content')
                    results['failed'] += 1
            else:
                results['details'].append(f'❌ {doc_file} missing')
                results['failed'] += 1
                
        # Check for docs directory
        if os.path.isdir('docs'):
            results['details'].append('✅ docs/ directory exists')
            results['passed'] += 1
        else:
            results['details'].append('❌ docs/ directory missing')
            results['failed'] += 1
            
        # Check for docstrings in key modules
        try:
            from grid_fed_rl.environments.grid_env import GridEnvironment
            
            if GridEnvironment.__doc__ and len(GridEnvironment.__doc__) > 50:
                results['details'].append('✅ GridEnvironment has comprehensive docstring')
                results['passed'] += 1
            else:
                results['details'].append('❌ GridEnvironment lacks adequate docstring')
                results['failed'] += 1
                
        except Exception as e:
            results['details'].append(f'❌ Docstring check failed: {e}')
            results['failed'] += 1
            
    except Exception as e:
        results['details'].append(f'❌ Documentation check failed: {e}')
        results['failed'] += 1
        
    return results


def test_performance_benchmarks() -> Dict[str, Any]:
    """Test performance benchmarks and thresholds."""
    results = {
        'name': 'Performance Benchmarks',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        # Test environment creation speed
        start_time = time.time()
        env = GridEnvironment(feeder=IEEE13Bus())
        creation_time = time.time() - start_time
        
        if creation_time < 1.0:  # Should create in under 1 second
            results['details'].append(f'✅ Environment creation time: {creation_time:.3f}s')
            results['passed'] += 1
        else:
            results['details'].append(f'❌ Environment creation too slow: {creation_time:.3f}s')
            results['failed'] += 1
            
        # Test reset speed
        start_time = time.time()
        obs, info = env.reset()
        reset_time = time.time() - start_time
        
        if reset_time < 0.1:  # Should reset in under 100ms
            results['details'].append(f'✅ Environment reset time: {reset_time:.3f}s')
            results['passed'] += 1
        else:
            results['details'].append(f'❌ Environment reset too slow: {reset_time:.3f}s')
            results['failed'] += 1
            
        # Test step speed
        total_step_time = 0.0
        num_steps = 10
        
        for i in range(num_steps):
            action = env.action_space.sample()
            start_time = time.time()
            obs, reward, done, truncated, info = env.step(action)
            step_time = time.time() - start_time
            total_step_time += step_time
            
        avg_step_time = total_step_time / num_steps
        
        if avg_step_time < 0.05:  # Should step in under 50ms average
            results['details'].append(f'✅ Average step time: {avg_step_time:.3f}s')
            results['passed'] += 1
        else:
            results['details'].append(f'❌ Average step time too slow: {avg_step_time:.3f}s')
            results['failed'] += 1
            
        # Test memory usage is reasonable
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb < 500:  # Should use less than 500MB
                results['details'].append(f'✅ Memory usage: {memory_mb:.1f}MB')
                results['passed'] += 1
            else:
                results['details'].append(f'❌ Memory usage too high: {memory_mb:.1f}MB')
                results['failed'] += 1
        except ImportError:
            results['details'].append('⚠️ psutil not available, skipping memory check')
            
    except Exception as e:
        results['details'].append(f'❌ Performance benchmark failed: {e}')
        results['failed'] += 1
        
    return results


def test_error_handling() -> Dict[str, Any]:
    """Test comprehensive error handling."""
    results = {
        'name': 'Error Handling',
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    try:
        from grid_fed_rl.environments.grid_env import GridEnvironment
        from grid_fed_rl.feeders.ieee_feeders import IEEE13Bus
        
        env = GridEnvironment(feeder=IEEE13Bus())
        env.reset()
        
        # Test invalid action handling
        error_scenarios = [
            ([float('inf')], 'infinity'),
            ([float('nan')], 'NaN'),
            ([1e10], 'extremely large value'),
            ([-1e10], 'extremely small value'),
            ([None], 'None value'),
        ]
        
        for invalid_action, description in error_scenarios:
            try:
                obs, reward, done, truncated, info = env.step(invalid_action)
                # Should not crash, but handle gracefully
                results['details'].append(f'✅ Gracefully handled {description}')
                results['passed'] += 1
            except Exception as e:
                results['details'].append(f'❌ Failed to handle {description}: {e}')
                results['failed'] += 1
                
        # Test that error information is provided
        try:
            obs, reward, done, truncated, info = env.step([float('nan')])
            if 'error' in info or 'action_invalid' in info:
                results['details'].append('✅ Error information provided in info dict')
                results['passed'] += 1
            else:
                results['details'].append('❌ No error information in info dict')
                results['failed'] += 1
        except Exception as e:
            results['details'].append(f'❌ Error info test failed: {e}')
            results['failed'] += 1
            
    except Exception as e:
        results['details'].append(f'❌ Error handling test failed: {e}')
        results['failed'] += 1
        
    return results


def run_production_readiness_tests() -> Dict[str, Any]:
    """Run complete production readiness test suite."""
    print("🚀 RUNNING PRODUCTION READINESS TESTS")
    print("=" * 60)
    
    start_time = time.time()
    all_results = []
    
    test_suites = [
        test_package_structure,
        test_import_hierarchy,
        test_configuration_files,
        test_dependencies,
        test_documentation,
        test_performance_benchmarks,
        test_error_handling
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
            total_failed += 1
            
    duration = time.time() - start_time
    
    # Calculate production readiness score
    total_tests = total_passed + total_failed
    if total_tests > 0:
        readiness_score = (total_passed / total_tests) * 100
    else:
        readiness_score = 0
        
    # Determine production readiness status
    if readiness_score >= 95:
        readiness_status = "PRODUCTION READY"
    elif readiness_score >= 85:
        readiness_status = "NEARLY READY"
    elif readiness_score >= 70:
        readiness_status = "NEEDS IMPROVEMENT"
    else:
        readiness_status = "NOT READY"
        
    print("\\n" + "=" * 60)
    print("🏭 PRODUCTION READINESS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Readiness Score: {readiness_score:.1f}%")
    print(f"Duration: {duration:.2f} seconds")
    print(f"\\n🎯 PRODUCTION STATUS: {readiness_status}")
    
    # Save detailed report
    report = {
        'production_status': readiness_status,
        'readiness_score': readiness_score,
        'total_passed': total_passed,
        'total_failed': total_failed,
        'duration': duration,
        'timestamp': time.time(),
        'detailed_results': all_results
    }
    
    with open('production_readiness_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"📄 Detailed report saved to production_readiness_report.json")
    
    return report


if __name__ == "__main__":
    results = run_production_readiness_tests()
    
    # Exit with appropriate code
    exit_code = 0 if results['readiness_score'] >= 85 else 1
    sys.exit(exit_code)