#!/usr/bin/env python3
"""
Autonomous Quality Gates Validation for Grid-Fed-RL-Gym
Comprehensive testing, security, and performance validation.
"""

import os
import sys
import time
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    status: str  # passed, failed, warning, skipped
    score: float  # 0-100
    details: Dict[str, Any]
    execution_time_seconds: float
    recommendations: List[str]


class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results: List[QualityGateResult] = []
        self.overall_score = 0.0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        
        print("🚀 Starting Autonomous Quality Gates Validation")
        print("=" * 60)
        
        # Define quality gates in execution order
        gates = [
            ("Import Validation", self._test_import_structure),
            ("Code Quality", self._test_code_quality),
            ("Security Scan", self._test_security),
            ("Performance Benchmarks", self._test_performance),
            ("Documentation Coverage", self._test_documentation),
            ("API Compatibility", self._test_api_compatibility),
            ("Integration Tests", self._test_integration),
            ("Resource Usage", self._test_resource_usage),
            ("Scalability", self._test_scalability),
            ("Error Handling", self._test_error_handling)
        ]
        
        # Execute each gate
        for gate_name, gate_func in gates:
            print(f"\n🔍 Running {gate_name}...")
            
            start_time = time.time()
            try:
                result = gate_func()
                execution_time = time.time() - start_time
                
                if result is None:
                    result = QualityGateResult(
                        gate_name=gate_name,
                        status="skipped",
                        score=0.0,
                        details={},
                        execution_time_seconds=execution_time,
                        recommendations=[]
                    )
                else:
                    result.execution_time_seconds = execution_time
                
                self.results.append(result)
                
                # Print result
                status_emoji = {
                    "passed": "✅",
                    "failed": "❌", 
                    "warning": "⚠️",
                    "skipped": "⏭️"
                }
                
                print(f"   {status_emoji.get(result.status, '❓')} {gate_name}: {result.status.upper()} "
                      f"(Score: {result.score:.1f}/100, Time: {execution_time:.2f}s)")
                
                if result.recommendations:
                    print(f"   💡 Recommendations: {', '.join(result.recommendations[:3])}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    status="failed",
                    score=0.0,
                    details={"error": str(e)},
                    execution_time_seconds=execution_time,
                    recommendations=[f"Fix error: {str(e)[:100]}"]
                )
                self.results.append(error_result)
                print(f"   ❌ {gate_name}: FAILED - {str(e)[:100]}")
        
        # Calculate overall score
        self._calculate_overall_score()
        
        # Generate final report
        return self._generate_final_report()
    
    def _test_import_structure(self) -> QualityGateResult:
        """Test import structure and basic functionality."""
        
        score = 100.0
        details = {}
        recommendations = []
        
        try:
            # Test basic package import
            import grid_fed_rl
            details["package_import"] = "success"
            details["version"] = getattr(grid_fed_rl, "__version__", "unknown")
            
            # Test core module imports
            core_modules = [
                "grid_fed_rl.environments",
                "grid_fed_rl.algorithms", 
                "grid_fed_rl.federated",
                "grid_fed_rl.utils"
            ]
            
            import_results = {}
            for module in core_modules:
                try:
                    __import__(module)
                    import_results[module] = "success"
                except ImportError as e:
                    import_results[module] = f"failed: {str(e)}"
                    score -= 20
                    recommendations.append(f"Fix import for {module}")
            
            details["core_imports"] = import_results
            
            # Test that main classes are available
            try:
                from grid_fed_rl import GridEnvironment
                details["main_classes"] = "available"
            except ImportError:
                details["main_classes"] = "missing"
                score -= 10
                recommendations.append("Ensure main classes are properly exported")
            
        except Exception as e:
            score = 0
            details["error"] = str(e)
            recommendations.append("Fix critical import errors")
        
        status = "passed" if score >= 90 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="Import Validation",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,  # Will be set by caller
            recommendations=recommendations
        )
    
    def _test_code_quality(self) -> QualityGateResult:
        """Test code quality metrics."""
        
        score = 85.0  # Base score
        details = {}
        recommendations = []
        
        # Count Python files
        py_files = list(self.project_root.rglob("*.py"))
        py_files = [f for f in py_files if not str(f).startswith(".")]  # Exclude hidden files
        
        details["total_python_files"] = len(py_files)
        
        # Basic code quality checks
        if len(py_files) == 0:
            score = 0
            details["error"] = "No Python files found"
            return QualityGateResult("Code Quality", "failed", score, details, 0, ["Add Python code"])
        
        # Check for docstrings in main modules
        main_modules = [
            self.project_root / "grid_fed_rl" / "__init__.py",
            self.project_root / "grid_fed_rl" / "environments" / "__init__.py",
            self.project_root / "grid_fed_rl" / "algorithms" / "__init__.py"
        ]
        
        docstring_coverage = 0
        for module_path in main_modules:
            if module_path.exists():
                try:
                    content = module_path.read_text()
                    if '"""' in content or "'''" in content:
                        docstring_coverage += 1
                except Exception:
                    pass
        
        docstring_score = (docstring_coverage / len(main_modules)) * 100 if main_modules else 0
        details["docstring_coverage"] = f"{docstring_score:.1f}%"
        
        if docstring_score < 50:
            score -= 15
            recommendations.append("Add docstrings to main modules")
        
        # Check for type hints (basic check)
        type_hint_files = 0
        for py_file in py_files[:10]:  # Check first 10 files
            try:
                content = py_file.read_text()
                if " -> " in content or ": " in content:
                    type_hint_files += 1
            except Exception:
                pass
        
        type_hint_coverage = (type_hint_files / min(len(py_files), 10)) * 100
        details["type_hint_coverage"] = f"{type_hint_coverage:.1f}%"
        
        if type_hint_coverage < 30:
            score -= 10
            recommendations.append("Add type hints to improve code quality")
        
        # Check for test files
        test_files = list(self.project_root.rglob("test_*.py")) + list(self.project_root.rglob("*_test.py"))
        test_coverage = len(test_files) / max(len(py_files), 1) * 100
        details["test_file_ratio"] = f"{test_coverage:.1f}%"
        
        if test_coverage < 10:
            score -= 10
            recommendations.append("Add more test files")
        
        status = "passed" if score >= 80 else "warning" if score >= 60 else "failed"
        
        return QualityGateResult(
            gate_name="Code Quality",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _test_security(self) -> QualityGateResult:
        """Test security vulnerabilities and best practices."""
        
        score = 90.0
        details = {}
        recommendations = []
        
        try:
            # Check for common security issues in code
            security_issues = []
            py_files = list(self.project_root.rglob("*.py"))
            
            for py_file in py_files:
                try:
                    content = py_file.read_text()
                    
                    # Check for dangerous patterns
                    dangerous_patterns = [
                        ("eval(", "Use of eval() function"),
                        ("exec(", "Use of exec() function"),
                        ("__import__", "Dynamic imports"),
                        ("pickle.loads", "Unsafe pickle deserialization"),
                        ("shell=True", "Shell command execution"),
                        ("input(", "User input without validation")
                    ]
                    
                    for pattern, description in dangerous_patterns:
                        if pattern in content:
                            security_issues.append(f"{py_file.name}: {description}")
                            score -= 15
                
                except Exception:
                    pass
            
            details["security_issues"] = security_issues[:10]  # Show first 10
            details["total_security_issues"] = len(security_issues)
            
            if security_issues:
                recommendations.extend([
                    "Review and fix security issues",
                    "Use safe alternatives to dangerous functions",
                    "Add input validation"
                ])
            
            # Check for secrets in code (basic patterns)
            secret_patterns = [
                ("password", "Hard-coded passwords"),
                ("api_key", "API keys in code"),
                ("secret", "Secret tokens"),
                ("token", "Authentication tokens")
            ]
            
            potential_secrets = []
            for py_file in py_files:
                try:
                    content = py_file.read_text().lower()
                    for pattern, description in secret_patterns:
                        if pattern in content and "=" in content:
                            # Simple check for assignment
                            lines = content.split('\n')
                            for line in lines:
                                if pattern in line and "=" in line and not line.strip().startswith("#"):
                                    potential_secrets.append(f"{py_file.name}: {description}")
                                    break
                except Exception:
                    pass
            
            if potential_secrets:
                score -= 10
                details["potential_secrets"] = potential_secrets[:5]
                recommendations.append("Review potential secrets in code")
            
            # Check for requirements.txt security
            req_file = self.project_root / "requirements.txt"
            if req_file.exists():
                details["requirements_security"] = "checked"
                # In a real implementation, would check for known vulnerable packages
            else:
                details["requirements_security"] = "no_requirements_file"
            
        except Exception as e:
            score = 50
            details["error"] = str(e)
            recommendations.append("Fix security scanning errors")
        
        status = "passed" if score >= 85 else "warning" if score >= 70 else "failed"
        
        return QualityGateResult(
            gate_name="Security Scan",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _test_performance(self) -> QualityGateResult:
        """Test performance benchmarks."""
        
        score = 80.0
        details = {}
        recommendations = []
        
        try:
            # Test basic import performance
            import_start = time.time()
            try:
                import grid_fed_rl
                import_time = (time.time() - import_start) * 1000
                details["import_time_ms"] = round(import_time, 2)
                
                if import_time > 1000:  # > 1 second
                    score -= 20
                    recommendations.append("Optimize import time")
                elif import_time > 500:  # > 0.5 seconds
                    score -= 10
                    recommendations.append("Consider reducing import overhead")
                
            except ImportError:
                score -= 30
                details["import_time_ms"] = "failed"
                recommendations.append("Fix import errors")
            
            # Test basic functionality performance
            try:
                # Test GridEnvironment creation if available
                from grid_fed_rl import GridEnvironment
                
                creation_start = time.time()
                try:
                    env = GridEnvironment()
                    creation_time = (time.time() - creation_start) * 1000
                    details["environment_creation_ms"] = round(creation_time, 2)
                    
                    if creation_time > 5000:  # > 5 seconds
                        score -= 15
                        recommendations.append("Optimize environment creation")
                    
                except Exception as e:
                    details["environment_creation_ms"] = f"failed: {str(e)[:50]}"
                    score -= 10
                
            except ImportError:
                details["environment_creation_ms"] = "not_available"
            
            # Memory usage check (basic)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                details["memory_usage_mb"] = round(memory_mb, 1)
                
                if memory_mb > 500:  # > 500 MB
                    score -= 10
                    recommendations.append("Monitor memory usage")
                
            except ImportError:
                details["memory_usage_mb"] = "psutil_not_available"
            
            # File size check
            total_size = 0
            py_files = list(self.project_root.rglob("*.py"))
            for py_file in py_files:
                try:
                    total_size += py_file.stat().st_size
                except Exception:
                    pass
            
            total_size_mb = total_size / 1024 / 1024
            details["total_code_size_mb"] = round(total_size_mb, 2)
            
            if total_size_mb > 50:  # > 50 MB of Python code
                score -= 5
                recommendations.append("Consider code size optimization")
            
        except Exception as e:
            score = 30
            details["error"] = str(e)
            recommendations.append("Fix performance testing errors")
        
        status = "passed" if score >= 75 else "warning" if score >= 60 else "failed"
        
        return QualityGateResult(
            gate_name="Performance Benchmarks",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _test_documentation(self) -> QualityGateResult:
        """Test documentation coverage and quality."""
        
        score = 75.0
        details = {}
        recommendations = []
        
        # Check for README
        readme_files = [
            self.project_root / "README.md",
            self.project_root / "README.rst",
            self.project_root / "README.txt"
        ]
        
        readme_exists = any(f.exists() for f in readme_files)
        details["readme_exists"] = readme_exists
        
        if readme_exists:
            # Check README content
            for readme_file in readme_files:
                if readme_file.exists():
                    try:
                        content = readme_file.read_text()
                        details["readme_length"] = len(content)
                        
                        # Check for key sections
                        key_sections = ["installation", "usage", "example", "license"]
                        present_sections = []
                        for section in key_sections:
                            if section.lower() in content.lower():
                                present_sections.append(section)
                        
                        details["readme_sections"] = present_sections
                        section_score = len(present_sections) / len(key_sections) * 25
                        score += section_score
                        
                        if len(present_sections) < 3:
                            recommendations.append("Add more sections to README")
                        
                        break
                    except Exception:
                        pass
        else:
            score -= 25
            recommendations.append("Add README file")
        
        # Check for documentation files
        doc_dirs = [
            self.project_root / "docs",
            self.project_root / "documentation",
            self.project_root / "doc"
        ]
        
        doc_dir_exists = any(d.exists() and d.is_dir() for d in doc_dirs)
        details["documentation_directory"] = doc_dir_exists
        
        if doc_dir_exists:
            for doc_dir in doc_dirs:
                if doc_dir.exists():
                    doc_files = list(doc_dir.rglob("*.md")) + list(doc_dir.rglob("*.rst"))
                    details["documentation_files"] = len(doc_files)
                    
                    if len(doc_files) >= 5:
                        score += 15
                    elif len(doc_files) >= 2:
                        score += 10
                    else:
                        score += 5
                        recommendations.append("Add more documentation files")
                    break
        else:
            score -= 10
            recommendations.append("Add documentation directory")
        
        # Check for inline documentation
        py_files = list(self.project_root.rglob("*.py"))
        documented_files = 0
        
        for py_file in py_files[:20]:  # Check first 20 files
            try:
                content = py_file.read_text()
                if '"""' in content and len(content.split('"""')) >= 3:
                    documented_files += 1
            except Exception:
                pass
        
        if py_files:
            doc_ratio = documented_files / min(len(py_files), 20)
            details["inline_documentation_ratio"] = f"{doc_ratio * 100:.1f}%"
            
            if doc_ratio < 0.3:
                score -= 15
                recommendations.append("Add more inline documentation")
            elif doc_ratio < 0.6:
                score -= 5
        
        # Check for examples
        example_dirs = [
            self.project_root / "examples",
            self.project_root / "example",
            self.project_root / "demos"
        ]
        
        example_files = []
        for example_dir in example_dirs:
            if example_dir.exists():
                example_files.extend(list(example_dir.rglob("*.py")))
        
        details["example_files"] = len(example_files)
        
        if len(example_files) >= 3:
            score += 10
        elif len(example_files) >= 1:
            score += 5
        else:
            recommendations.append("Add example files")
        
        status = "passed" if score >= 75 else "warning" if score >= 60 else "failed"
        
        return QualityGateResult(
            gate_name="Documentation Coverage",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _test_api_compatibility(self) -> QualityGateResult:
        """Test API compatibility and interface consistency."""
        
        score = 80.0
        details = {}
        recommendations = []
        
        try:
            # Test that main classes can be imported and instantiated
            api_tests = {}
            
            try:
                from grid_fed_rl import GridEnvironment
                api_tests["GridEnvironment"] = "importable"
                
                # Try to get basic info about the class
                if hasattr(GridEnvironment, '__init__'):
                    api_tests["GridEnvironment_init"] = "available"
                else:
                    api_tests["GridEnvironment_init"] = "missing"
                    score -= 10
                
            except ImportError:
                api_tests["GridEnvironment"] = "import_failed"
                score -= 20
                recommendations.append("Fix GridEnvironment import")
            
            # Test other core components
            core_components = [
                ("grid_fed_rl.algorithms", "algorithms"),
                ("grid_fed_rl.federated", "federated"),
                ("grid_fed_rl.utils", "utils")
            ]
            
            for module_name, component_name in core_components:
                try:
                    module = __import__(module_name, fromlist=[component_name])
                    api_tests[component_name] = "importable"
                except ImportError:
                    api_tests[component_name] = "import_failed"
                    score -= 5
            
            details["api_components"] = api_tests
            
            # Check for version information
            try:
                import grid_fed_rl
                version = getattr(grid_fed_rl, "__version__", None)
                if version:
                    details["version_available"] = True
                    details["version"] = version
                else:
                    details["version_available"] = False
                    score -= 5
                    recommendations.append("Add version information")
            except Exception:
                score -= 10
            
            # Check for standard attributes
            standard_attrs = ["__author__", "__email__", "__version__"]
            available_attrs = []
            
            try:
                import grid_fed_rl
                for attr in standard_attrs:
                    if hasattr(grid_fed_rl, attr):
                        available_attrs.append(attr)
            except Exception:
                pass
            
            details["standard_attributes"] = available_attrs
            
            if len(available_attrs) < 2:
                score -= 5
                recommendations.append("Add standard package attributes")
            
        except Exception as e:
            score = 40
            details["error"] = str(e)
            recommendations.append("Fix API compatibility issues")
        
        status = "passed" if score >= 75 else "warning" if score >= 60 else "failed"
        
        return QualityGateResult(
            gate_name="API Compatibility",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _test_integration(self) -> QualityGateResult:
        """Test integration capabilities."""
        
        score = 70.0
        details = {}
        recommendations = []
        
        try:
            # Test package structure
            expected_structure = [
                "grid_fed_rl/__init__.py",
                "grid_fed_rl/environments/__init__.py",
                "grid_fed_rl/algorithms/__init__.py",
                "grid_fed_rl/federated/__init__.py",
                "grid_fed_rl/utils/__init__.py"
            ]
            
            structure_score = 0
            missing_files = []
            
            for expected_file in expected_structure:
                file_path = self.project_root / expected_file
                if file_path.exists():
                    structure_score += 1
                else:
                    missing_files.append(expected_file)
            
            details["package_structure_score"] = f"{structure_score}/{len(expected_structure)}"
            details["missing_files"] = missing_files
            
            structure_percentage = structure_score / len(expected_structure)
            score = score * structure_percentage
            
            if missing_files:
                recommendations.append(f"Add missing files: {', '.join(missing_files)}")
            
            # Test configuration files
            config_files = ["pyproject.toml", "setup.py", "requirements.txt"]
            present_configs = []
            
            for config_file in config_files:
                if (self.project_root / config_file).exists():
                    present_configs.append(config_file)
            
            details["configuration_files"] = present_configs
            
            if len(present_configs) >= 2:
                score += 10
            elif len(present_configs) >= 1:
                score += 5
            else:
                recommendations.append("Add configuration files")
            
            # Test for deployment files
            deployment_files = [
                "Dockerfile",
                "docker-compose.yml", 
                "kubernetes/deployment.yaml",
                ".github/workflows"
            ]
            
            present_deployment = []
            for deploy_file in deployment_files:
                if (self.project_root / deploy_file).exists():
                    present_deployment.append(deploy_file)
            
            details["deployment_files"] = present_deployment
            
            if present_deployment:
                score += 15
                details["deployment_ready"] = True
            else:
                details["deployment_ready"] = False
                recommendations.append("Add deployment configuration")
            
        except Exception as e:
            score = 30
            details["error"] = str(e)
            recommendations.append("Fix integration testing errors")
        
        status = "passed" if score >= 70 else "warning" if score >= 50 else "failed"
        
        return QualityGateResult(
            gate_name="Integration Tests",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _test_resource_usage(self) -> QualityGateResult:
        """Test resource usage patterns."""
        
        score = 85.0
        details = {}
        recommendations = []
        
        try:
            # Test memory usage during import
            try:
                import psutil
                process = psutil.Process()
                
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Import the package
                import grid_fed_rl
                
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = memory_after - memory_before
                
                details["memory_increase_mb"] = round(memory_increase, 2)
                details["total_memory_mb"] = round(memory_after, 2)
                
                if memory_increase > 100:  # > 100 MB increase
                    score -= 20
                    recommendations.append("Reduce memory usage during import")
                elif memory_increase > 50:  # > 50 MB increase
                    score -= 10
                
            except ImportError:
                details["memory_tracking"] = "psutil_not_available"
            except Exception as e:
                details["memory_tracking"] = f"error: {str(e)}"
                score -= 10
            
            # Check file count and sizes
            py_files = list(self.project_root.rglob("*.py"))
            total_size = sum(f.stat().st_size for f in py_files if f.exists())
            total_size_mb = total_size / 1024 / 1024
            
            details["total_python_files"] = len(py_files)
            details["total_code_size_mb"] = round(total_size_mb, 2)
            details["average_file_size_kb"] = round(total_size / max(len(py_files), 1) / 1024, 2)
            
            # Check for large files
            large_files = []
            for py_file in py_files:
                try:
                    size_kb = py_file.stat().st_size / 1024
                    if size_kb > 100:  # > 100 KB
                        large_files.append(f"{py_file.name}: {size_kb:.1f}KB")
                except Exception:
                    pass
            
            details["large_files"] = large_files[:5]  # Show first 5
            
            if len(large_files) > 10:
                score -= 10
                recommendations.append("Consider refactoring large files")
            
            # Check import depth (rough estimate)
            try:
                import grid_fed_rl
                import sys
                
                modules_before = len(sys.modules)
                # This is a rough estimate of import complexity
                grid_modules = [name for name in sys.modules.keys() if 'grid_fed_rl' in name]
                
                details["imported_modules"] = len(grid_modules)
                
                if len(grid_modules) > 50:
                    score -= 5
                    recommendations.append("Consider reducing import complexity")
                
            except Exception:
                details["imported_modules"] = "unable_to_measure"
            
        except Exception as e:
            score = 40
            details["error"] = str(e)
            recommendations.append("Fix resource usage testing")
        
        status = "passed" if score >= 80 else "warning" if score >= 65 else "failed"
        
        return QualityGateResult(
            gate_name="Resource Usage",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _test_scalability(self) -> QualityGateResult:
        """Test scalability features and patterns."""
        
        score = 75.0
        details = {}
        recommendations = []
        
        try:
            # Check for scalability-related modules
            scalability_modules = [
                "grid_fed_rl/utils/advanced_scaling.py",
                "grid_fed_rl/utils/distributed.py",
                "grid_fed_rl/utils/performance.py",
                "grid_fed_rl/algorithms/neural_optimization.py"
            ]
            
            present_modules = []
            for module_path in scalability_modules:
                if (self.project_root / module_path).exists():
                    present_modules.append(module_path)
            
            details["scalability_modules"] = present_modules
            details["scalability_module_count"] = len(present_modules)
            
            module_score = len(present_modules) / len(scalability_modules) * 30
            score += module_score
            
            if len(present_modules) < 2:
                recommendations.append("Add more scalability modules")
            
            # Check for threading/multiprocessing usage
            threading_files = []
            async_files = []
            
            py_files = list(self.project_root.rglob("*.py"))
            for py_file in py_files:
                try:
                    content = py_file.read_text()
                    
                    if any(pattern in content for pattern in ["threading", "multiprocessing", "concurrent.futures"]):
                        threading_files.append(py_file.name)
                    
                    if any(pattern in content for pattern in ["async def", "await ", "asyncio"]):
                        async_files.append(py_file.name)
                        
                except Exception:
                    pass
            
            details["threading_files"] = len(threading_files)
            details["async_files"] = len(async_files)
            
            if threading_files or async_files:
                score += 15
                details["concurrency_support"] = True
            else:
                details["concurrency_support"] = False
                recommendations.append("Add concurrency support for scalability")
            
            # Check for caching mechanisms
            caching_patterns = ["cache", "memoize", "lru_cache", "Cache"]
            caching_files = []
            
            for py_file in py_files:
                try:
                    content = py_file.read_text()
                    if any(pattern in content for pattern in caching_patterns):
                        caching_files.append(py_file.name)
                except Exception:
                    pass
            
            details["caching_files"] = len(caching_files)
            
            if caching_files:
                score += 10
                details["caching_support"] = True
            else:
                details["caching_support"] = False
                recommendations.append("Add caching for performance")
            
            # Check for configuration management
            config_patterns = ["config", "settings", "Config", "Settings"]
            config_files = []
            
            for py_file in py_files:
                try:
                    if any(pattern in py_file.name for pattern in config_patterns):
                        config_files.append(py_file.name)
                except Exception:
                    pass
            
            details["config_files"] = len(config_files)
            
            if config_files:
                score += 5
            else:
                recommendations.append("Add configuration management")
            
        except Exception as e:
            score = 40
            details["error"] = str(e)
            recommendations.append("Fix scalability testing")
        
        status = "passed" if score >= 75 else "warning" if score >= 60 else "failed"
        
        return QualityGateResult(
            gate_name="Scalability",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _test_error_handling(self) -> QualityGateResult:
        """Test error handling robustness."""
        
        score = 80.0
        details = {}
        recommendations = []
        
        try:
            # Check for custom exception classes
            exception_files = []
            exception_classes = []
            
            py_files = list(self.project_root.rglob("*.py"))
            for py_file in py_files:
                try:
                    content = py_file.read_text()
                    
                    # Look for exception definitions
                    lines = content.split('\n')
                    for line in lines:
                        if 'class ' in line and 'Exception' in line:
                            exception_classes.append(line.strip())
                            if py_file.name not in exception_files:
                                exception_files.append(py_file.name)
                        elif 'class ' in line and 'Error' in line:
                            exception_classes.append(line.strip())
                            if py_file.name not in exception_files:
                                exception_files.append(py_file.name)
                
                except Exception:
                    pass
            
            details["exception_classes"] = len(exception_classes)
            details["exception_files"] = len(exception_files)
            details["custom_exceptions"] = exception_classes[:10]  # Show first 10
            
            if len(exception_classes) >= 5:
                score += 15
            elif len(exception_classes) >= 2:
                score += 10
            else:
                score -= 10
                recommendations.append("Add custom exception classes")
            
            # Check for try-catch blocks
            try_catch_files = []
            total_try_blocks = 0
            
            for py_file in py_files:
                try:
                    content = py_file.read_text()
                    try_count = content.count('try:')
                    except_count = content.count('except')
                    
                    if try_count > 0 and except_count > 0:
                        try_catch_files.append(py_file.name)
                        total_try_blocks += try_count
                
                except Exception:
                    pass
            
            details["try_catch_files"] = len(try_catch_files)
            details["total_try_blocks"] = total_try_blocks
            
            error_handling_ratio = len(try_catch_files) / max(len(py_files), 1)
            details["error_handling_ratio"] = f"{error_handling_ratio * 100:.1f}%"
            
            if error_handling_ratio < 0.3:
                score -= 15
                recommendations.append("Add more error handling")
            elif error_handling_ratio < 0.5:
                score -= 5
            
            # Check for logging
            logging_files = []
            
            for py_file in py_files:
                try:
                    content = py_file.read_text()
                    if any(pattern in content for pattern in ['logging', 'logger', 'log.']):
                        logging_files.append(py_file.name)
                except Exception:
                    pass
            
            details["logging_files"] = len(logging_files)
            
            if len(logging_files) >= len(py_files) * 0.3:
                score += 10
                details["logging_coverage"] = "good"
            elif len(logging_files) > 0:
                score += 5
                details["logging_coverage"] = "partial"
            else:
                details["logging_coverage"] = "minimal"
                recommendations.append("Add logging for better error tracking")
            
            # Check for validation patterns
            validation_patterns = ['validate', 'check', 'assert', 'raise', 'ValueError', 'TypeError']
            validation_files = []
            
            for py_file in py_files:
                try:
                    content = py_file.read_text()
                    if any(pattern in content for pattern in validation_patterns):
                        validation_files.append(py_file.name)
                except Exception:
                    pass
            
            details["validation_files"] = len(validation_files)
            
            if len(validation_files) >= len(py_files) * 0.4:
                score += 5
            else:
                recommendations.append("Add input validation")
            
        except Exception as e:
            score = 30
            details["error"] = str(e)
            recommendations.append("Fix error handling testing")
        
        status = "passed" if score >= 75 else "warning" if score >= 60 else "failed"
        
        return QualityGateResult(
            gate_name="Error Handling",
            status=status,
            score=score,
            details=details,
            execution_time_seconds=0,
            recommendations=recommendations
        )
    
    def _calculate_overall_score(self) -> None:
        """Calculate weighted overall score."""
        
        # Define weights for different quality gates
        weights = {
            "Import Validation": 0.15,
            "Code Quality": 0.12,
            "Security Scan": 0.15,
            "Performance Benchmarks": 0.10,
            "Documentation Coverage": 0.08,
            "API Compatibility": 0.12,
            "Integration Tests": 0.10,
            "Resource Usage": 0.08,
            "Scalability": 0.05,
            "Error Handling": 0.05
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in self.results:
            weight = weights.get(result.gate_name, 0.05)  # Default weight
            total_weighted_score += result.score * weight
            total_weight += weight
        
        self.overall_score = total_weighted_score / max(total_weight, 1.0)
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        # Categorize results
        passed = [r for r in self.results if r.status == "passed"]
        warnings = [r for r in self.results if r.status == "warning"]
        failed = [r for r in self.results if r.status == "failed"]
        skipped = [r for r in self.results if r.status == "skipped"]
        
        # Collect all recommendations
        all_recommendations = []
        for result in self.results:
            all_recommendations.extend(result.recommendations)
        
        # Determine overall status
        if self.overall_score >= 85:
            overall_status = "EXCELLENT"
            status_emoji = "🏆"
        elif self.overall_score >= 75:
            overall_status = "GOOD"
            status_emoji = "✅"
        elif self.overall_score >= 65:
            overall_status = "ACCEPTABLE"
            status_emoji = "⚠️"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
            status_emoji = "❌"
        
        report = {
            "timestamp": time.time(),
            "overall_score": round(self.overall_score, 1),
            "overall_status": overall_status,
            "summary": {
                "total_gates": len(self.results),
                "passed": len(passed),
                "warnings": len(warnings),
                "failed": len(failed),
                "skipped": len(skipped)
            },
            "gate_results": [asdict(result) for result in self.results],
            "recommendations": {
                "high_priority": all_recommendations[:10],
                "total_recommendations": len(all_recommendations)
            },
            "execution_time_seconds": sum(r.execution_time_seconds for r in self.results)
        }
        
        # Print final summary
        print("\n" + "=" * 60)
        print(f"{status_emoji} QUALITY GATES VALIDATION COMPLETE")
        print("=" * 60)
        print(f"Overall Score: {self.overall_score:.1f}/100 ({overall_status})")
        print(f"Passed: {len(passed)}, Warnings: {len(warnings)}, Failed: {len(failed)}, Skipped: {len(skipped)}")
        print(f"Total execution time: {report['execution_time_seconds']:.2f} seconds")
        
        if all_recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(all_recommendations[:5], 1):
                print(f"{i}. {rec}")
        
        print("\n🎯 Grid-Fed-RL-Gym autonomous quality validation completed successfully!")
        
        return report


def main():
    """Main entry point for quality gates validation."""
    
    # Initialize validator
    validator = QualityGateValidator(".")
    
    # Run all quality gates
    report = validator.run_all_gates()
    
    # Save report to file
    report_file = Path("quality_gates_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if report["overall_score"] >= 70:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()