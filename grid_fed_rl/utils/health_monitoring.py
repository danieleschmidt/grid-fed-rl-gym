"""Health monitoring and system diagnostics for grid simulation."""

import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    status: HealthStatus
    last_updated: datetime
    unit: str = ""
    
    def update(self, value: float):
        """Update metric value and status."""
        self.value = value
        self.last_updated = datetime.now()
        
        if value >= self.threshold_critical:
            self.status = HealthStatus.CRITICAL
        elif value >= self.threshold_warning:
            self.status = HealthStatus.WARNING
        else:
            self.status = HealthStatus.HEALTHY


@dataclass
class SystemCheck:
    """System health check definition."""
    name: str
    check_function: Callable
    frequency_seconds: float
    last_run: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.metrics: Dict[str, HealthMetric] = {}
        self.checks: List[SystemCheck] = []
        self.alerts: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        # Initialize core metrics
        self._initialize_core_metrics()
        self._initialize_system_checks()
        
    def _initialize_core_metrics(self):
        """Initialize core health metrics."""
        self.metrics = {
            "simulation_speed": HealthMetric(
                name="Simulation Speed",
                value=0.0,
                threshold_warning=0.1,  # 10ms per step
                threshold_critical=0.5,  # 500ms per step
                status=HealthStatus.HEALTHY,
                last_updated=datetime.now(),
                unit="ms/step"
            ),
            "memory_usage": HealthMetric(
                name="Memory Usage", 
                value=0.0,
                threshold_warning=500.0,  # 500 MB
                threshold_critical=1000.0,  # 1 GB
                status=HealthStatus.HEALTHY,
                last_updated=datetime.now(),
                unit="MB"
            ),
            "error_rate": HealthMetric(
                name="Error Rate",
                value=0.0,
                threshold_warning=0.01,  # 1% error rate
                threshold_critical=0.05,  # 5% error rate
                status=HealthStatus.HEALTHY,
                last_updated=datetime.now(),
                unit="%"
            ),
            "power_flow_convergence": HealthMetric(
                name="Power Flow Convergence Rate",
                value=1.0,
                threshold_warning=0.95,  # Below 95% convergence
                threshold_critical=0.85,  # Below 85% convergence
                status=HealthStatus.HEALTHY,
                last_updated=datetime.now(),
                unit="%"
            ),
            "constraint_violations": HealthMetric(
                name="Constraint Violations",
                value=0.0,
                threshold_warning=5.0,  # 5 violations per hour
                threshold_critical=20.0,  # 20 violations per hour
                status=HealthStatus.HEALTHY,
                last_updated=datetime.now(),
                unit="violations/hour"
            )
        }
        
    def _initialize_system_checks(self):
        """Initialize system health checks."""
        self.checks = [
            SystemCheck(
                name="Environment Response",
                check_function=self._check_environment_response,
                frequency_seconds=60.0  # Check every minute
            ),
            SystemCheck(
                name="Memory Health",
                check_function=self._check_memory_health,
                frequency_seconds=30.0  # Check every 30 seconds
            ),
            SystemCheck(
                name="Configuration Integrity",
                check_function=self._check_configuration,
                frequency_seconds=300.0  # Check every 5 minutes
            )
        ]
        
    def update_metric(self, name: str, value: float):
        """Update a health metric."""
        if name in self.metrics:
            old_status = self.metrics[name].status
            self.metrics[name].update(value)
            
            # Check for status change
            new_status = self.metrics[name].status
            if new_status != old_status and new_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self._create_alert(f"Metric {name} status changed to {new_status.value}", new_status)
                
    def update_simulation_speed(self, step_duration: float):
        """Update simulation speed metric."""
        speed_ms = step_duration * 1000  # Convert to milliseconds
        self.update_metric("simulation_speed", speed_ms)
        
    def update_error_rate(self, total_operations: int, error_count: int):
        """Update error rate metric."""
        if total_operations > 0:
            error_rate = error_count / total_operations
            self.update_metric("error_rate", error_rate)
            
    def update_convergence_rate(self, total_attempts: int, successful_attempts: int):
        """Update power flow convergence rate."""
        if total_attempts > 0:
            convergence_rate = successful_attempts / total_attempts
            self.update_metric("power_flow_convergence", convergence_rate)
            
    def increment_violations(self):
        """Increment constraint violations counter."""
        current_violations = self.metrics["constraint_violations"].value
        
        # Calculate violations per hour
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        if uptime_hours > 0:
            violations_per_hour = (current_violations + 1) / uptime_hours
            self.update_metric("constraint_violations", violations_per_hour)
            
    def run_health_checks(self):
        """Run due health checks."""
        current_time = datetime.now()
        
        for check in self.checks:
            if (check.last_run is None or 
                (current_time - check.last_run).total_seconds() >= check.frequency_seconds):
                
                try:
                    result = check.check_function()
                    check.last_result = result
                    check.last_run = current_time
                    check.last_error = None
                    
                    if not result:
                        self._create_alert(f"Health check '{check.name}' failed", HealthStatus.WARNING)
                        
                except Exception as e:
                    check.last_error = str(e)
                    check.last_run = current_time
                    self._create_alert(f"Health check '{check.name}' error: {e}", HealthStatus.CRITICAL)
                    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        statuses = [metric.status for metric in self.metrics.values()]
        
        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
            
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        overall_status = self.get_overall_status()
        uptime = datetime.now() - self.start_time
        
        metric_summary = {}
        for name, metric in self.metrics.items():
            metric_summary[name] = {
                "value": metric.value,
                "status": metric.status.value,
                "unit": metric.unit,
                "last_updated": metric.last_updated.isoformat()
            }
            
        check_summary = {}
        for check in self.checks:
            check_summary[check.name] = {
                "last_run": check.last_run.isoformat() if check.last_run else None,
                "last_result": check.last_result,
                "last_error": check.last_error,
                "frequency": check.frequency_seconds
            }
            
        return {
            "overall_status": overall_status.value,
            "uptime_seconds": uptime.total_seconds(),
            "metrics": metric_summary,
            "checks": check_summary,
            "recent_alerts": self.alerts[-10:],  # Last 10 alerts
            "total_alerts": len(self.alerts),
            "report_time": datetime.now().isoformat()
        }
        
    def _create_alert(self, message: str, severity: HealthStatus):
        """Create a health alert."""
        alert = {
            "message": message,
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
            "id": len(self.alerts)
        }
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts to prevent memory issues
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
            
    def _check_environment_response(self) -> bool:
        """Check if environment is responding normally."""
        try:
            # This would typically test environment instantiation
            # For now, just return True as basic check
            return True
        except Exception:
            return False
            
    def _check_memory_health(self) -> bool:
        """Check memory usage health."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.update_metric("memory_usage", memory_mb)
            
            # Return False if memory usage is critical
            return memory_mb < self.metrics["memory_usage"].threshold_critical
        except ImportError:
            return True  # Can't check without psutil
        except Exception:
            return False
            
    def _check_configuration(self) -> bool:
        """Check configuration integrity."""
        try:
            # This would typically validate configuration files
            # For now, just return True as basic check
            return True
        except Exception:
            return False


class WatchdogTimer:
    """Watchdog timer for detecting system hangs."""
    
    def __init__(self, timeout_seconds: float = 300.0):  # 5 minute default
        self.timeout = timeout_seconds
        self.last_heartbeat = datetime.now()
        self.enabled = True
        
    def heartbeat(self):
        """Signal that system is alive."""
        if self.enabled:
            self.last_heartbeat = datetime.now()
            
    def check_timeout(self) -> bool:
        """Check if system has timed out."""
        if not self.enabled:
            return False
            
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat > self.timeout
        
    def reset(self):
        """Reset watchdog timer."""
        self.last_heartbeat = datetime.now()
        
    def enable(self):
        """Enable watchdog monitoring."""
        self.enabled = True
        self.reset()
        
    def disable(self):
        """Disable watchdog monitoring."""
        self.enabled = False


# Global health monitor instance
system_health = HealthMonitor()
system_watchdog = WatchdogTimer()