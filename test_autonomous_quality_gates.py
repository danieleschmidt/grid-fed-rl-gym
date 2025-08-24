"""Comprehensive quality gates for autonomous SDLC implementation."""

import sys
import time
import json
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of quality gate execution."""
    name: str
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]
    execution_time: float
    errors: List[str]
    warnings: List[str]


class AutonomousQualityGates:
    """Comprehensive quality gate system for autonomous SDLC."""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.overall_score = 0.0
        
    def test_imports_and_basic_functionality(self) -> QualityGateResult:
        """Test that all core modules can be imported and basic functions work."""
        
        start_time = time.time()
        errors = []
        warnings = []
        details = {}
        
        try:
            # Core framework imports
            import grid_fed_rl
            details["version"] = grid_fed_rl.__version__
            
            # Test basic environment creation
            try:
                from grid_fed_rl import run_quick_demo
                demo_result = run_quick_demo()
                details["quick_demo"] = demo_result["success"]
                
                if demo_result["success"]:
                    perf_summary = demo_result["performance_summary"]
                    details["stability_rate"] = perf_summary["stability_rate"]
                    details["performance_score"] = perf_summary["performance_score"]
            except Exception as e:
                errors.append(f"Quick demo failed: {e}")
            
            # Test monitoring
            try:
                from grid_fed_rl.monitoring import run_health_check
                health = run_health_check()
                details["health_status"] = health["status"]
                # Convert health summary to JSON-serializable format
                health_summary = health["summary"]
                details["health_summary"] = {
                    key: (value.value if hasattr(value, 'value') else value)
                    for key, value in health_summary.items()
                    if key != 'detailed_results'  # Skip complex nested objects
                }
            except Exception as e:
                errors.append(f"Health check failed: {e}")
            
            # Test validation
            try:
                from grid_fed_rl.validation import validate_grid_input
                validation_result = validate_grid_input({"timestep": 1.0})
                details["validation_working"] = validation_result["valid"]
            except Exception as e:
                errors.append(f"Validation failed: {e}")
            
            # Test error handling
            try:
                from grid_fed_rl.error_handling import RobustExecutor
                executor = RobustExecutor()
                test_result = executor.execute(lambda: "test_success")
                details["error_handling_working"] = test_result.success
            except Exception as e:
                errors.append(f"Error handling failed: {e}")
            
            # Test performance components
            try:
                from grid_fed_rl.performance import get_cache_stats
                cache_stats = get_cache_stats()
                details["caching_working"] = isinstance(cache_stats, dict)
            except Exception as e:
                errors.append(f"Performance caching failed: {e}")
            
            # Test optimization
            try:
                from grid_fed_rl.optimization import optimize_function
                
                def simple_objective(params):
                    return -params["x"]**2
                
                result = optimize_function(simple_objective, {"x": (-2.0, 2.0)}, max_iterations=5)
                details["optimization_working"] = result is not None
            except Exception as e:
                errors.append(f"Optimization failed: {e}")
            
        except Exception as e:
            errors.append(f"Core import failed: {e}")
        
        execution_time = time.time() - start_time
        
        # Calculate score
        total_components = 6  # Core, demo, monitoring, validation, error handling, performance, optimization
        working_components = sum([
            details.get("quick_demo", False),
            details.get("health_status") != "failed",
            details.get("validation_working", False),
            details.get("error_handling_working", False), 
            details.get("caching_working", False),
            details.get("optimization_working", False)
        ])
        
        score = (working_components / total_components) * 100
        passed = len(errors) == 0 and score >= 85
        
        return QualityGateResult(
            name="Imports and Basic Functionality",
            passed=passed,
            score=score,
            details=details,
            execution_time=execution_time,
            errors=errors,
            warnings=warnings
        )
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        
        logger.info("🚀 Starting Autonomous Quality Gates Execution")
        overall_start = time.time()
        
        # Define quality gates (simplified for now)
        gates = [
            ("Imports and Basic Functionality", self.test_imports_and_basic_functionality)
        ]
        
        # Run each gate
        for gate_name, gate_func in gates:
            logger.info(f"🔍 Executing {gate_name}...")
            
            try:
                result = gate_func()
                self.results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                logger.info(f"   {status} - Score: {result.score:.1f}/100 ({result.execution_time:.2f}s)")
                
                if result.errors:
                    for error in result.errors:
                        logger.error(f"   ERROR: {error}")
                
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning(f"   WARNING: {warning}")
                        
            except Exception as e:
                logger.error(f"   💥 EXCEPTION: Quality gate failed to execute: {e}")
                self.results.append(QualityGateResult(
                    name=gate_name,
                    passed=False,
                    score=0.0,
                    details={"exception": str(e)},
                    execution_time=0.0,
                    errors=[str(e)],
                    warnings=[]
                ))
        
        # Calculate overall results
        total_execution_time = time.time() - overall_start
        
        if self.results:
            self.overall_score = sum(r.score for r in self.results) / len(self.results)
            gates_passed = sum(1 for r in self.results if r.passed)
            gates_failed = len(self.results) - gates_passed
        else:
            self.overall_score = 0.0
            gates_passed = 0
            gates_failed = len(gates)
        
        # Generate comprehensive report
        report = {
            "timestamp": time.time(),
            "overall_status": "PASSED" if self.overall_score >= 80 and gates_failed == 0 else "FAILED",
            "overall_score": self.overall_score,
            "execution_time_seconds": total_execution_time,
            "summary": {
                "total_gates": len(gates),
                "gates_passed": gates_passed, 
                "gates_failed": gates_failed,
                "success_rate": f"{(gates_passed/len(gates)*100):.1f}%" if gates else "0%"
            },
            "detailed_results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "details": result.details
                }
                for result in self.results
            ]
        }
        
        # Log final results
        logger.info("🏁 Quality Gates Execution Complete")
        logger.info(f"📊 Overall Status: {report['overall_status']}")
        logger.info(f"📈 Overall Score: {self.overall_score:.1f}/100")
        logger.info(f"⏱️ Total Time: {total_execution_time:.2f}s")
        logger.info(f"🎯 Success Rate: {report['summary']['success_rate']}")
        
        return report


def main():
    """Main entry point for quality gates execution."""
    
    # Create and run quality gates
    quality_gates = AutonomousQualityGates()
    report = quality_gates.run_all_gates()
    
    # Save detailed report
    report_file = "quality_gates_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if report["overall_status"] == "PASSED":
        print("🎉 All quality gates passed!")
        return 0
    else:
        print("💥 Some quality gates failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())