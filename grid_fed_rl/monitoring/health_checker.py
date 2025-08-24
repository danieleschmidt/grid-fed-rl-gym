"""Health monitoring and system validation for grid environments."""

import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration_ms: float
    metadata: Dict[str, Any] = None


class GridHealthChecker:
    """Comprehensive health monitoring for grid simulation systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checks_registry = {}
        self.last_check_results = {}
        self.system_start_time = time.time()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register built-in health checks."""
        
        def check_memory_usage():
            """Check system memory usage."""
            try:
                import psutil
                memory = psutil.virtual_memory()
                usage_percent = memory.percent
                
                if usage_percent > 90:
                    return HealthStatus.CRITICAL, f"Memory usage critical: {usage_percent:.1f}%"
                elif usage_percent > 75:
                    return HealthStatus.WARNING, f"Memory usage high: {usage_percent:.1f}%"
                else:
                    return HealthStatus.HEALTHY, f"Memory usage normal: {usage_percent:.1f}%"
            except ImportError:
                return HealthStatus.WARNING, "Memory monitoring unavailable (psutil not installed)"
        
        def check_disk_space():
            """Check available disk space."""
            try:
                import shutil
                total, used, free = shutil.disk_usage("/")
                free_percent = (free / total) * 100
                
                if free_percent < 5:
                    return HealthStatus.CRITICAL, f"Disk space critical: {free_percent:.1f}% free"
                elif free_percent < 15:
                    return HealthStatus.WARNING, f"Disk space low: {free_percent:.1f}% free"
                else:
                    return HealthStatus.HEALTHY, f"Disk space adequate: {free_percent:.1f}% free"
            except Exception as e:
                return HealthStatus.WARNING, f"Disk space check failed: {e}"
        
        def check_core_imports():
            """Check if core dependencies can be imported."""
            missing_critical = []
            missing_optional = []
            
            # Critical imports
            critical_modules = ['json', 'time', 'logging', 'os', 'sys']
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_critical.append(module)
            
            # Optional imports
            optional_modules = ['numpy', 'pandas', 'matplotlib', 'torch']
            for module in optional_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_optional.append(module)
            
            if missing_critical:
                return HealthStatus.CRITICAL, f"Critical modules missing: {missing_critical}"
            elif len(missing_optional) > 2:
                return HealthStatus.WARNING, f"Many optional modules missing: {missing_optional}"
            else:
                return HealthStatus.HEALTHY, f"Core imports functional, {len(missing_optional)} optional missing"
        
        def check_file_permissions():
            """Check file system permissions."""
            import tempfile
            try:
                with tempfile.NamedTemporaryFile(delete=True) as f:
                    f.write(b"test")
                    f.flush()
                return HealthStatus.HEALTHY, "File system writable"
            except Exception as e:
                return HealthStatus.CRITICAL, f"File system not writable: {e}"
        
        # Register checks
        self.checks_registry.update({
            "memory_usage": check_memory_usage,
            "disk_space": check_disk_space,
            "core_imports": check_core_imports,
            "file_permissions": check_file_permissions
        })
    
    def register_check(self, name: str, check_function):
        """Register a custom health check."""
        self.checks_registry[name] = check_function
        
    def run_check(self, name: str) -> HealthCheck:
        """Execute a single health check."""
        if name not in self.checks_registry:
            return HealthCheck(
                name=name,
                status=HealthStatus.FAILED,
                message=f"Check '{name}' not registered",
                timestamp=time.time(),
                duration_ms=0
            )
        
        start_time = time.time()
        
        try:
            status, message = self.checks_registry[name]()
            duration_ms = (time.time() - start_time) * 1000
            
            result = HealthCheck(
                name=name,
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=duration_ms
            )
            
            self.last_check_results[name] = result
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.exception(f"Health check '{name}' failed with exception")
            
            result = HealthCheck(
                name=name,
                status=HealthStatus.FAILED,
                message=f"Check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=duration_ms
            )
            
            self.last_check_results[name] = result
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Execute all registered health checks."""
        results = {}
        
        for name in self.checks_registry:
            results[name] = self.run_check(name)
        
        return results
    
    def get_system_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        # Determine overall status
        statuses = [check.status for check in results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.FAILED in statuses:
            overall_status = HealthStatus.FAILED
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Compile summary
        summary = {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self.system_start_time,
            "checks_total": len(results),
            "checks_healthy": len([c for c in results.values() if c.status == HealthStatus.HEALTHY]),
            "checks_warning": len([c for c in results.values() if c.status == HealthStatus.WARNING]),
            "checks_critical": len([c for c in results.values() if c.status == HealthStatus.CRITICAL]),
            "checks_failed": len([c for c in results.values() if c.status == HealthStatus.FAILED]),
            "detailed_results": {name: asdict(check) for name, check in results.items()}
        }
        
        return overall_status, summary
    
    def format_health_report(self, include_details: bool = True) -> str:
        """Generate a formatted health report."""
        overall_status, summary = self.get_system_health()
        
        # Status icons
        status_icons = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🔴", 
            HealthStatus.FAILED: "❌"
        }
        
        icon = status_icons.get(overall_status, "❓")
        
        report = f"""
🔍 GRID-FED-RL-GYM HEALTH REPORT
{'='*50}

{icon} OVERALL STATUS: {overall_status.value.upper()}

📊 SUMMARY:
   - Total Checks: {summary['checks_total']}
   - Healthy: {summary['checks_healthy']}
   - Warning: {summary['checks_warning']}
   - Critical: {summary['checks_critical']}
   - Failed: {summary['checks_failed']}
   - System Uptime: {summary['uptime_seconds']:.1f} seconds

"""
        
        if include_details:
            report += "📋 DETAILED RESULTS:\n"
            for name, result in summary['detailed_results'].items():
                status_icon = status_icons.get(HealthStatus(result['status']), "❓")
                report += f"   {status_icon} {name}: {result['message']} ({result['duration_ms']:.1f}ms)\n"
        
        return report


def run_health_check() -> Dict[str, Any]:
    """Standalone function to run comprehensive health check."""
    checker = GridHealthChecker()
    overall_status, summary = checker.get_system_health()
    
    print(checker.format_health_report())
    
    return {
        "status": overall_status.value,
        "summary": summary,
        "timestamp": time.time()
    }


if __name__ == "__main__":
    # Allow direct execution
    result = run_health_check()
    
    # Exit with appropriate code
    if result["status"] in ["critical", "failed"]:
        exit(1)
    elif result["status"] == "warning":
        exit(2)
    else:
        exit(0)