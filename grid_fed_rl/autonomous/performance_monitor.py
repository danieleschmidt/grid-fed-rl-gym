"""
Performance monitoring and metric collection for autonomous execution.
"""

import time
import json
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = None

class MetricCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_metrics_per_type: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        self._lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, unit: str = "", metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value, 
            unit=unit,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
        
    def get_metrics(self, name: str, limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Get metrics by name."""
        with self._lock:
            metrics = list(self.metrics[name])
            if limit:
                return metrics[-limit:]
            return metrics
            
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get statistical summary of metrics."""
        metrics = self.get_metrics(name)
        if not metrics:
            return {"error": "No metrics found"}
            
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values), 
            "mean": sum(values) / len(values),
            "latest": values[-1],
            "unit": metrics[-1].unit,
            "timespan_seconds": metrics[-1].timestamp - metrics[0].timestamp if len(metrics) > 1 else 0
        }
        
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all collected metrics."""
        summary = {}
        with self._lock:
            for name in self.metrics:
                summary[name] = self.get_metric_summary(name)
        return summary

class PerformanceTracker:
    """Tracks system performance during autonomous execution."""
    
    def __init__(self, collector: MetricCollector = None):
        self.collector = collector or MetricCollector()
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = 1.0  # seconds
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring."""
        if self._monitoring:
            return
            
        self._monitor_interval = interval
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self._monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.collector.record_metric("cpu_usage", cpu_percent, "percent")
            
            # Memory metrics  
            memory = psutil.virtual_memory()
            self.collector.record_metric("memory_usage", memory.percent, "percent")
            self.collector.record_metric("memory_available", memory.available / 1024**3, "GB")
            
            # Process-specific metrics
            process = psutil.Process()
            proc_memory = process.memory_info()
            self.collector.record_metric("process_memory_rss", proc_memory.rss / 1024**2, "MB")
            self.collector.record_metric("process_memory_vms", proc_memory.vms / 1024**2, "MB")
            
            # Thread count
            self.collector.record_metric("thread_count", process.num_threads(), "count")
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            
    def measure_execution_time(self, operation_name: str):
        """Context manager to measure execution time."""
        return ExecutionTimer(self.collector, operation_name)
        
    def track_function_performance(self, func: Callable, name: str = None) -> Callable:
        """Decorator to track function performance."""
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                raise
            finally:
                execution_time = time.time() - start_time
                self.collector.record_metric(
                    f"function_execution_time_{func_name}",
                    execution_time * 1000,  # Convert to milliseconds
                    "ms",
                    {"success": success}
                )
            return result
        return wrapper
        
    def generate_report(self, output_path: Path = None) -> Dict[str, Any]:
        """Generate performance report."""
        summary = self.collector.get_all_metrics_summary()
        
        report = {
            "report_timestamp": time.time(),
            "metrics_summary": summary,
            "performance_insights": self._analyze_performance(summary)
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report
        
    def _analyze_performance(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics and provide insights."""
        insights = {
            "warnings": [],
            "recommendations": [],
            "overall_health": "good"
        }
        
        # Analyze CPU usage
        if "cpu_usage" in summary and summary["cpu_usage"].get("mean", 0) > 80:
            insights["warnings"].append("High CPU usage detected")
            insights["recommendations"].append("Consider optimizing CPU-intensive operations")
            
        # Analyze memory usage
        if "memory_usage" in summary and summary["memory_usage"].get("mean", 0) > 85:
            insights["warnings"].append("High memory usage detected")
            insights["recommendations"].append("Monitor memory leaks and optimize memory allocation")
            
        # Analyze execution times
        slow_functions = []
        for metric_name, metric_data in summary.items():
            if "function_execution_time" in metric_name and metric_data.get("mean", 0) > 5000:  # 5 seconds
                func_name = metric_name.replace("function_execution_time_", "")
                slow_functions.append(func_name)
                
        if slow_functions:
            insights["warnings"].append(f"Slow functions detected: {', '.join(slow_functions)}")
            insights["recommendations"].append("Optimize slow function execution")
            
        # Determine overall health
        if len(insights["warnings"]) > 3:
            insights["overall_health"] = "poor"
        elif len(insights["warnings"]) > 1:
            insights["overall_health"] = "fair"
            
        return insights

class ExecutionTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, collector: MetricCollector, operation_name: str):
        self.collector = collector
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_metric(
                f"execution_time_{self.operation_name}",
                duration * 1000,  # Convert to milliseconds
                "ms",
                {"success": exc_type is None}
            )