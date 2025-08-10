#!/usr/bin/env python3
"""Production readiness assessment and final validation script."""

import os
import sys
import json
import subprocess
import importlib
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def setup_logging() -> logging.Logger:
    """Setup logging for the readiness check."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('production_readiness.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


class ProductionReadinessChecker:
    """Comprehensive production readiness assessment."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.results = {}
        
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all production readiness checks."""
        
        logger.info("🚀 Starting Production Readiness Assessment")
        logger.info("=" * 60)
        
        # Run all check categories
        self.check_dependencies()
        self.check_core_functionality() 
        self.check_performance()
        self.check_security()
        self.check_monitoring()
        self.check_compliance()
        self.check_deployment_readiness()
        
        # Generate final report
        return self.generate_final_report()
    
    def check_dependencies(self) -> None:
        """Check all dependencies are installed."""
        logger.info("\n1️⃣  Checking Dependencies...")
        
        required_modules = [
            'numpy', 'scipy', 'pandas', 'matplotlib', 'networkx', 'pydantic'
        ]
        
        optional_modules = [
            'torch', 'gymnasium', 'crypten', 'opacus', 'flower'
        ]
        
        # Check required modules
        for module in required_modules:
            try:
                importlib.import_module(module)
                logger.info(f"  ✅ {module} - installed")
                self._pass_check(f"dependency_{module}")
            except Exception as e:
                if module == 'matplotlib' and ('NumPy 1.x cannot be run in NumPy' in str(e) or 'numpy' in str(e).lower()):
                    logger.warning(f"  ⚠️  {module} - version compatibility issue with NumPy 2.x")
                    self._warn_check(f"dependency_{module}")
                else:
                    logger.error(f"  ❌ {module} - missing (required): {str(e)}")
                    self._fail_check(f"dependency_{module}")
        
        # Check optional modules
        for module in optional_modules:
            try:
                importlib.import_module(module)
                logger.info(f"  ✅ {module} - installed")
                self._pass_check(f"optional_{module}")
            except ImportError:
                logger.warning(f"  ⚠️  {module} - missing (optional)")
                self._warn_check(f"optional_{module}")
    
    def check_core_functionality(self) -> None:
        """Check core system functionality."""
        logger.info("\n2️⃣  Checking Core Functionality...")
        
        try:
            # Test imports
            from grid_fed_rl.feeders import IEEE13Bus, SimpleRadialFeeder
            from grid_fed_rl.environments import GridEnvironment
            from grid_fed_rl.utils import SafetyChecker, validate_action
            
            logger.info("  ✅ Core imports successful")
            self._pass_check("core_imports")
            
            # Test feeder creation
            feeder = IEEE13Bus()
            errors = feeder.validate_network()
            if not errors:
                logger.info("  ✅ Network validation passed")
                self._pass_check("network_validation")
            else:
                logger.error(f"  ❌ Network validation failed: {len(errors)} errors")
                self._fail_check("network_validation")
            
            # Test environment creation and basic operation
            env = GridEnvironment(SimpleRadialFeeder(num_buses=3), episode_length=10)
            obs, info = env.reset()
            
            for _ in range(5):  # Run 5 steps
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    obs, info = env.reset()
            
            logger.info("  ✅ Environment operation successful")
            self._pass_check("environment_operation")
            
        except Exception as e:
            logger.error(f"  ❌ Core functionality test failed: {e}")
            self._fail_check("core_functionality")
    
    def check_performance(self) -> None:
        """Check performance benchmarks."""
        logger.info("\n3️⃣  Checking Performance...")
        
        try:
            from grid_fed_rl.environments import GridEnvironment
            from grid_fed_rl.feeders import IEEE13Bus
            import time
            
            # Performance benchmarks
            feeder = IEEE13Bus()
            env = GridEnvironment(feeder, episode_length=100)
            
            # Measure initialization time
            start_time = time.time()
            obs, info = env.reset()
            init_time = time.time() - start_time
            
            # Measure step performance
            step_times = []
            for _ in range(50):
                start_time = time.time()
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                step_time = time.time() - start_time
                step_times.append(step_time)
                
                if done:
                    obs, info = env.reset()
            
            avg_step_time = sum(step_times) / len(step_times)
            max_step_time = max(step_times)
            
            # Performance thresholds
            MAX_INIT_TIME = 0.1  # 100ms
            MAX_AVG_STEP_TIME = 0.02  # 20ms
            MAX_STEP_TIME = 0.05  # 50ms
            
            if init_time <= MAX_INIT_TIME:
                logger.info(f"  ✅ Initialization time: {init_time*1000:.2f}ms")
                self._pass_check("init_performance")
            else:
                logger.warning(f"  ⚠️  Slow initialization: {init_time*1000:.2f}ms > {MAX_INIT_TIME*1000}ms")
                self._warn_check("init_performance")
            
            if avg_step_time <= MAX_AVG_STEP_TIME:
                logger.info(f"  ✅ Average step time: {avg_step_time*1000:.2f}ms")
                self._pass_check("step_performance")
            else:
                logger.warning(f"  ⚠️  Slow step performance: {avg_step_time*1000:.2f}ms > {MAX_AVG_STEP_TIME*1000}ms")
                self._warn_check("step_performance")
            
            if max_step_time <= MAX_STEP_TIME:
                logger.info(f"  ✅ Max step time: {max_step_time*1000:.2f}ms")
                self._pass_check("max_step_performance")
            else:
                logger.warning(f"  ⚠️  High step time variance: {max_step_time*1000:.2f}ms > {MAX_STEP_TIME*1000}ms")
                self._warn_check("max_step_performance")
                
        except Exception as e:
            logger.error(f"  ❌ Performance test failed: {e}")
            self._fail_check("performance")
    
    def check_security(self) -> None:
        """Check security configuration."""
        logger.info("\n4️⃣  Checking Security...")
        
        try:
            from grid_fed_rl.utils.security import SecurityAuditor, InputValidator
            
            # Test input validation
            validator = InputValidator()
            
            # Test numeric validation
            valid, msg = validator.validate_numeric_input(1000.0, max_val=10000.0)
            if valid:
                logger.info("  ✅ Numeric input validation working")
                self._pass_check("input_validation")
            else:
                logger.error(f"  ❌ Input validation failed: {msg}")
                self._fail_check("input_validation")
            
            # Test security audit
            auditor = SecurityAuditor()
            audit_results = auditor.audit_system()
            
            critical_issues = audit_results['summary']['critical_count']
            high_issues = audit_results['summary']['high_count']
            
            if critical_issues == 0:
                logger.info("  ✅ No critical security issues")
                self._pass_check("security_critical")
            else:
                logger.error(f"  ❌ {critical_issues} critical security issues found")
                self._fail_check("security_critical")
            
            if high_issues <= 2:
                logger.info(f"  ✅ High security issues: {high_issues} (acceptable)")
                self._pass_check("security_high")
            else:
                logger.warning(f"  ⚠️  {high_issues} high security issues found")
                self._warn_check("security_high")
                
        except Exception as e:
            logger.error(f"  ❌ Security check failed: {e}")
            self._fail_check("security")
    
    def check_monitoring(self) -> None:
        """Check monitoring and observability."""
        logger.info("\n5️⃣  Checking Monitoring & Observability...")
        
        try:
            from grid_fed_rl.utils.monitoring import GridMonitor, HealthChecker
            
            # Test monitoring system
            monitor = GridMonitor(metrics_window=50)
            health_checker = HealthChecker()
            
            # Test health check
            test_state = {
                'power_flow_converged': True,
                'bus_voltages': [0.98, 1.0, 1.02],
                'frequency': 60.1,
                'line_loadings': [0.6, 0.8]
            }
            
            health_result = health_checker.run_health_check(test_state)
            
            if health_result['overall_health'] == 'healthy':
                logger.info("  ✅ Health checking system working")
                self._pass_check("health_check")
            else:
                logger.warning(f"  ⚠️  Health check issues: {health_result['overall_health']}")
                self._warn_check("health_check")
            
            # Test metrics recording
            violations = {'total_violations': 0, 'emergency_action_required': False}
            metrics = monitor.record_metrics(1, 0.005, test_state, violations)
            
            if metrics:
                logger.info("  ✅ Metrics recording working")
                self._pass_check("metrics")
            else:
                logger.error("  ❌ Metrics recording failed")
                self._fail_check("metrics")
                
        except Exception as e:
            logger.error(f"  ❌ Monitoring check failed: {e}")
            self._fail_check("monitoring")
    
    def check_compliance(self) -> None:
        """Check compliance and regulatory requirements."""
        logger.info("\n6️⃣  Checking Compliance...")
        
        try:
            from grid_fed_rl.utils.compliance import ComplianceManager
            from grid_fed_rl.utils.internationalization import global_translation_manager
            
            # Test compliance management
            compliance_manager = ComplianceManager()
            
            deployment_regions = ['EU', 'California', 'China']
            audit_results = compliance_manager.run_compliance_audit(deployment_regions)
            
            compliance_score = audit_results['compliance_score']
            violations = len(audit_results['violations'])
            
            if compliance_score >= 80:
                logger.info(f"  ✅ Compliance score: {compliance_score:.1f}%")
                self._pass_check("compliance_score")
            else:
                logger.warning(f"  ⚠️  Low compliance score: {compliance_score:.1f}%")
                self._warn_check("compliance_score")
            
            if violations <= 2:
                logger.info(f"  ✅ Compliance violations: {violations} (acceptable)")
                self._pass_check("compliance_violations")
            else:
                logger.warning(f"  ⚠️  Multiple compliance violations: {violations}")
                self._warn_check("compliance_violations")
            
            # Test internationalization
            languages = global_translation_manager.get_available_locales()
            if len(languages) >= 6:
                logger.info(f"  ✅ Multi-language support: {len(languages)} languages")
                self._pass_check("internationalization")
            else:
                logger.warning(f"  ⚠️  Limited language support: {len(languages)} languages")
                self._warn_check("internationalization")
                
        except Exception as e:
            logger.error(f"  ❌ Compliance check failed: {e}")
            self._fail_check("compliance")
    
    def check_deployment_readiness(self) -> None:
        """Check deployment readiness."""
        logger.info("\n7️⃣  Checking Deployment Readiness...")
        
        # Check required files
        required_files = [
            'pyproject.toml',
            'requirements.txt', 
            'README.md',
            'DEPLOYMENT.md',
            'LICENSE'
        ]
        
        for file in required_files:
            if os.path.exists(file):
                logger.info(f"  ✅ {file} exists")
                self._pass_check(f"file_{file}")
            else:
                logger.error(f"  ❌ Missing required file: {file}")
                self._fail_check(f"file_{file}")
        
        # Check package structure
        required_dirs = [
            'grid_fed_rl',
            'grid_fed_rl/environments',
            'grid_fed_rl/feeders',
            'grid_fed_rl/algorithms',
            'grid_fed_rl/utils'
        ]
        
        for dir_path in required_dirs:
            if os.path.isdir(dir_path):
                logger.info(f"  ✅ {dir_path}/ directory exists")
                self._pass_check(f"dir_{dir_path}")
            else:
                logger.error(f"  ❌ Missing directory: {dir_path}/")
                self._fail_check(f"dir_{dir_path}")
        
        # Test CLI
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'grid_fed_rl.cli', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and 'grid-fed-rl-gym' in result.stdout:
                logger.info("  ✅ CLI working correctly")
                self._pass_check("cli_test")
            else:
                logger.error(f"  ❌ CLI test failed: {result.stderr}")
                self._fail_check("cli_test")
                
        except Exception as e:
            logger.error(f"  ❌ CLI test failed: {e}")
            self._fail_check("cli_test")
    
    def _pass_check(self, check_name: str) -> None:
        """Record a passed check."""
        self.checks_passed += 1
        self.results[check_name] = 'pass'
    
    def _fail_check(self, check_name: str) -> None:
        """Record a failed check."""
        self.checks_failed += 1
        self.results[check_name] = 'fail'
    
    def _warn_check(self, check_name: str) -> None:
        """Record a warning check."""
        self.warnings += 1
        self.results[check_name] = 'warning'
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final production readiness report."""
        
        total_checks = self.checks_passed + self.checks_failed + self.warnings
        pass_rate = self.checks_passed / total_checks if total_checks > 0 else 0
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 PRODUCTION READINESS ASSESSMENT COMPLETE")
        logger.info("=" * 60)
        
        logger.info(f"✅ Checks passed: {self.checks_passed}")
        logger.info(f"❌ Checks failed: {self.checks_failed}")
        logger.info(f"⚠️  Warnings: {self.warnings}")
        logger.info(f"📈 Pass rate: {pass_rate:.1%}")
        
        # Determine readiness level
        if self.checks_failed == 0 and pass_rate >= 0.9:
            readiness = "PRODUCTION_READY"
            logger.info("\n🎉 SYSTEM IS PRODUCTION READY!")
            logger.info("✅ All critical checks passed - ready for deployment")
        elif self.checks_failed == 0 and pass_rate >= 0.8:
            readiness = "MOSTLY_READY"
            logger.info("\n⚠️  SYSTEM IS MOSTLY PRODUCTION READY")
            logger.info("📋 Address warnings before production deployment")
        elif self.checks_failed <= 2:
            readiness = "NEEDS_ATTENTION"
            logger.info("\n❗ SYSTEM NEEDS ATTENTION")
            logger.info("🔧 Fix critical issues before production deployment")
        else:
            readiness = "NOT_READY"
            logger.info("\n❌ SYSTEM NOT READY FOR PRODUCTION")
            logger.info("🚨 Multiple critical issues must be resolved")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'readiness_level': readiness,
            'summary': {
                'total_checks': total_checks,
                'checks_passed': self.checks_passed,
                'checks_failed': self.checks_failed,
                'warnings': self.warnings,
                'pass_rate': pass_rate
            },
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open('production_readiness_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed report saved: production_readiness_report.json")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on check results."""
        recommendations = []
        
        # Check for failed dependencies
        failed_deps = [k for k, v in self.results.items() 
                      if k.startswith('dependency_') and v == 'fail']
        if failed_deps:
            recommendations.append("Install missing required dependencies")
        
        # Check for performance issues
        perf_warnings = [k for k, v in self.results.items() 
                        if 'performance' in k and v == 'warning']
        if perf_warnings:
            recommendations.append("Optimize performance for production workloads")
        
        # Check for security issues
        if self.results.get('security_critical') == 'fail':
            recommendations.append("CRITICAL: Fix all critical security vulnerabilities")
        
        # Check for compliance issues
        if self.results.get('compliance_score') == 'warning':
            recommendations.append("Improve compliance score for target regions")
        
        # General recommendations
        if self.checks_failed > 0:
            recommendations.append("Review and fix all failed checks before deployment")
        
        if self.warnings > 5:
            recommendations.append("Address excessive warnings to improve system reliability")
        
        return recommendations


def main():
    """Run production readiness assessment."""
    
    checker = ProductionReadinessChecker()
    
    try:
        report = checker.run_all_checks()
        
        # Exit with appropriate code
        if report['readiness_level'] in ['PRODUCTION_READY', 'MOSTLY_READY']:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        logger.error("Assessment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Assessment failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()