"""Advanced performance monitoring and benchmarking system with regression detection."""

import asyncio
import time
import threading
import logging
import json
import sqlite3
import psutil
import statistics
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path
import functools
import inspect
import traceback
from datetime import datetime, timedelta
import concurrent.futures
import pickle
import gzip
import hashlib

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None
    sns = None
    PLOTTING_AVAILABLE = False

try:
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    scipy = None
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""
    UNIT = "unit"
    INTEGRATION = "integration"
    LOAD = "load"
    STRESS = "stress"
    PERFORMANCE = "performance"
    REGRESSION = "regression"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: float
    metric_type: MetricType
    name: str
    value: float
    unit: str
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'metric_type': self.metric_type.value,
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'context': self.context,
            'tags': self.tags
        }


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""
    benchmark_name: str
    category: BenchmarkCategory
    execution_time: float
    success: bool
    metrics: List[PerformanceMetric] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add a performance metric to the result."""
        self.metrics.append(metric)
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get value of a specific metric."""
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric.value
        return None


@dataclass
class PerformanceAlert:
    """Performance alert."""
    timestamp: float
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


class StatisticalAnalyzer:
    """Statistical analysis for performance metrics."""
    
    def __init__(self):
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add metric to statistical analysis."""
        key = f"{metric.name}:{metric.metric_type.value}"
        self.metric_history[key].append(metric.value)
    
    def calculate_statistics(self, metric_key: str, window_size: Optional[int] = None) -> Dict[str, float]:
        """Calculate statistical measures for a metric."""
        if metric_key not in self.metric_history:
            return {}
        
        values = list(self.metric_history[metric_key])
        if window_size:
            values = values[-window_size:]
        
        if not values:
            return {}
        
        stats = {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }
        
        # Add percentiles
        if len(values) >= 4:
            stats.update({
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            })
        
        # Add coefficient of variation
        if stats['mean'] != 0:
            stats['cv'] = stats['std_dev'] / stats['mean']
        
        return stats
    
    def detect_anomalies(self, metric_key: str, threshold_std: float = 2.0) -> List[Tuple[int, float]]:
        """Detect anomalies using statistical methods."""
        if metric_key not in self.metric_history:
            return []
        
        values = list(self.metric_history[metric_key])
        if len(values) < 10:
            return []
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean) / std_dev if std_dev > 0 else 0
            if z_score > threshold_std:
                anomalies.append((i, value))
        
        return anomalies
    
    def calculate_trend(self, metric_key: str, window_size: int = 50) -> Optional[float]:
        """Calculate trend slope for a metric."""
        if metric_key not in self.metric_history:
            return None
        
        values = list(self.metric_history[metric_key])[-window_size:]
        if len(values) < 10:
            return None
        
        # Simple linear regression
        x = list(range(len(values)))
        if SCIPY_AVAILABLE:
            slope, _, _, _, _ = scipy.stats.linregress(x, values)
            return slope
        else:
            # Manual calculation
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope


class RegressionDetector:
    """Detects performance regressions."""
    
    def __init__(self, sensitivity: float = 0.1, window_size: int = 100):
        self.sensitivity = sensitivity  # 10% threshold for regression detection
        self.window_size = window_size
        self.baselines: Dict[str, float] = {}
        
    def set_baseline(self, metric_key: str, baseline_value: float) -> None:
        """Set baseline value for regression detection."""
        self.baselines[metric_key] = baseline_value
    
    def detect_regression(
        self, 
        metric_key: str, 
        current_stats: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Detect if there's a performance regression."""
        if metric_key not in self.baselines:
            return None
        
        baseline = self.baselines[metric_key]
        current_mean = current_stats.get('mean')
        
        if current_mean is None:
            return None
        
        # Calculate percentage change
        if baseline != 0:
            change_percent = (current_mean - baseline) / baseline
        else:
            change_percent = 1.0 if current_mean > 0 else 0.0
        
        # Determine if this is a regression (depends on metric type)
        is_regression = False
        regression_type = "improvement"
        
        # For latency, throughput metrics
        if abs(change_percent) > self.sensitivity:
            if "latency" in metric_key.lower() or "response_time" in metric_key.lower():
                # Higher latency is worse
                is_regression = change_percent > 0
                regression_type = "regression" if is_regression else "improvement"
            elif "throughput" in metric_key.lower() or "rate" in metric_key.lower():
                # Lower throughput is worse
                is_regression = change_percent < 0
                regression_type = "regression" if is_regression else "improvement"
            else:
                # Generic - significant change is noted
                is_regression = True
                regression_type = "significant_change"
        
        if is_regression or abs(change_percent) > self.sensitivity:
            return {
                'metric_key': metric_key,
                'baseline': baseline,
                'current': current_mean,
                'change_percent': change_percent * 100,
                'is_regression': is_regression,
                'regression_type': regression_type,
                'severity': 'high' if abs(change_percent) > 0.2 else 'medium'
            }
        
        return None
    
    def analyze_trend_regression(
        self,
        metric_key: str,
        trend_slope: float,
        current_stats: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze trend for potential regressions."""
        if abs(trend_slope) < 0.001:  # Minimal slope
            return None
        
        # Determine if trend indicates regression
        is_negative_trend = False
        
        if "latency" in metric_key.lower() or "response_time" in metric_key.lower():
            # Increasing latency trend is bad
            is_negative_trend = trend_slope > 0
        elif "throughput" in metric_key.lower() or "rate" in metric_key.lower():
            # Decreasing throughput trend is bad
            is_negative_trend = trend_slope < 0
        elif "error" in metric_key.lower():
            # Increasing error rate is bad
            is_negative_trend = trend_slope > 0
        
        if is_negative_trend:
            return {
                'metric_key': metric_key,
                'trend_slope': trend_slope,
                'trend_type': 'negative',
                'severity': 'high' if abs(trend_slope) > 0.01 else 'medium',
                'prediction': 'Performance degradation trend detected'
            }
        
        return None


class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self):
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules: List[Dict[str, Any]] = []
        self.subscribers: List[Callable[[PerformanceAlert], None]] = []
        
    def add_alert_rule(
        self,
        metric_pattern: str,
        threshold: float,
        comparison: str = "greater",
        level: AlertLevel = AlertLevel.WARNING,
        cooldown: float = 300.0  # 5 minutes
    ) -> None:
        """Add an alert rule."""
        rule = {
            'metric_pattern': metric_pattern,
            'threshold': threshold,
            'comparison': comparison,  # greater, less, equal
            'level': level,
            'cooldown': cooldown,
            'last_triggered': 0.0
        }
        self.alert_rules.append(rule)
    
    def check_alerts(self, metric: PerformanceMetric) -> None:
        """Check if metric triggers any alerts."""
        current_time = time.time()
        
        for rule in self.alert_rules:
            # Check if metric matches pattern
            if rule['metric_pattern'] not in metric.name:
                continue
            
            # Check cooldown
            if current_time - rule['last_triggered'] < rule['cooldown']:
                continue
            
            # Check threshold
            triggered = False
            if rule['comparison'] == 'greater' and metric.value > rule['threshold']:
                triggered = True
            elif rule['comparison'] == 'less' and metric.value < rule['threshold']:
                triggered = True
            elif rule['comparison'] == 'equal' and abs(metric.value - rule['threshold']) < 0.001:
                triggered = True
            
            if triggered:
                alert = PerformanceAlert(
                    timestamp=current_time,
                    level=rule['level'],
                    metric_name=metric.name,
                    current_value=metric.value,
                    threshold_value=rule['threshold'],
                    message=f"Metric {metric.name} triggered alert: {metric.value} {rule['comparison']} {rule['threshold']}",
                    context=metric.context.copy()
                )
                
                self.alerts.append(alert)
                rule['last_triggered'] = current_time
                
                # Notify subscribers
                for subscriber in self.subscribers:
                    try:
                        subscriber(alert)
                    except Exception as e:
                        logger.error(f"Alert notification failed: {e}")
    
    def subscribe_to_alerts(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Subscribe to alert notifications."""
        self.subscribers.append(callback)
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_timestamp: float) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.timestamp == alert_timestamp:
                alert.resolved = True
                alert.resolution_time = time.time()
                return True
        return False


class PerformanceProfiler:
    """Advanced performance profiler with automatic instrumentation."""
    
    def __init__(self, enable_detailed_profiling: bool = False):
        self.enable_detailed_profiling = enable_detailed_profiling
        self.function_metrics: Dict[str, List[float]] = defaultdict(list)
        self.call_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.active_calls: Dict[int, Dict[str, Any]] = {}
        
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator for profiling function execution."""
        def decorator(func: Callable) -> Callable:
            function_name = func_name or f"{func.__module__}.{func.__qualname__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                call_id = id(threading.current_thread()) + hash(time.time())
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss
                
                # Record call start
                self.active_calls[call_id] = {
                    'function': function_name,
                    'start_time': start_time,
                    'start_memory': start_memory,
                    'thread_id': threading.get_ident()
                }
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.perf_counter()
                    end_memory = psutil.Process().memory_info().rss
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Record metrics
                    self.function_metrics[function_name].append(execution_time)
                    
                    # Record detailed metrics if enabled
                    if self.enable_detailed_profiling:
                        self._record_detailed_metrics(
                            function_name, execution_time, memory_delta, success, error
                        )
                    
                    # Clean up active call
                    self.active_calls.pop(call_id, None)
                
                return result
            
            return wrapper
        return decorator
    
    def _record_detailed_metrics(
        self,
        function_name: str,
        execution_time: float,
        memory_delta: int,
        success: bool,
        error: Optional[str]
    ) -> None:
        """Record detailed profiling metrics."""
        # This would integrate with the main monitoring system
        pass
    
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get statistics for a specific function."""
        if function_name not in self.function_metrics:
            return {}
        
        times = self.function_metrics[function_name]
        
        return {
            'call_count': len(times),
            'total_time': sum(times),
            'average_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'median_time': statistics.median(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0.0
        }
    
    def get_top_functions(self, metric: str = 'total_time', limit: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """Get top functions by specified metric."""
        function_stats = []
        
        for func_name in self.function_metrics:
            stats = self.get_function_stats(func_name)
            if stats:
                function_stats.append((func_name, stats))
        
        # Sort by specified metric
        if metric in ['total_time', 'average_time', 'max_time', 'call_count']:
            function_stats.sort(key=lambda x: x[1].get(metric, 0), reverse=True)
        
        return function_stats[:limit]


class BenchmarkSuite:
    """Comprehensive benchmark suite with automated execution."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.benchmarks: Dict[str, Callable] = {}
        self.benchmark_configs: Dict[str, Dict[str, Any]] = {}
        self.results_history: List[BenchmarkResult] = []
        
        # Storage for persistence
        self.storage_path = Path(storage_path) if storage_path else Path("benchmark_results.db")
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Initialize benchmark results storage."""
        try:
            self.conn = sqlite3.connect(str(self.storage_path), check_same_thread=False)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    execution_time REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    metrics_json TEXT,
                    metadata_json TEXT,
                    error_message TEXT
                )
            """)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize benchmark storage: {e}")
            self.conn = None
    
    def register_benchmark(
        self,
        name: str,
        func: Callable,
        category: BenchmarkCategory = BenchmarkCategory.PERFORMANCE,
        **config
    ) -> None:
        """Register a benchmark function."""
        self.benchmarks[name] = func
        self.benchmark_configs[name] = {
            'category': category,
            **config
        }
    
    def benchmark(
        self,
        name: Optional[str] = None,
        category: BenchmarkCategory = BenchmarkCategory.PERFORMANCE,
        **config
    ):
        """Decorator to register benchmark functions."""
        def decorator(func: Callable) -> Callable:
            benchmark_name = name or func.__name__
            self.register_benchmark(benchmark_name, func, category, **config)
            return func
        return decorator
    
    def run_benchmark(self, benchmark_name: str, **kwargs) -> BenchmarkResult:
        """Run a specific benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not found")
        
        func = self.benchmarks[benchmark_name]
        config = self.benchmark_configs[benchmark_name]
        category = config['category']
        
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            category=category,
            execution_time=0.0,
            success=False,
            timestamp=time.time()
        )
        
        try:
            # Execute benchmark
            benchmark_result = func(**kwargs)
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            result.execution_time = end_time - start_time
            result.success = True
            
            # Add default metrics
            result.add_metric(PerformanceMetric(
                timestamp=time.time(),
                metric_type=MetricType.LATENCY,
                name=f"{benchmark_name}_execution_time",
                value=result.execution_time,
                unit="seconds"
            ))
            
            result.add_metric(PerformanceMetric(
                timestamp=time.time(),
                metric_type=MetricType.MEMORY_USAGE,
                name=f"{benchmark_name}_memory_delta",
                value=(end_memory - start_memory) / (1024 * 1024),
                unit="MB"
            ))
            
            # Add custom metrics if returned by benchmark
            if isinstance(benchmark_result, dict):
                for key, value in benchmark_result.items():
                    if isinstance(value, (int, float)):
                        result.add_metric(PerformanceMetric(
                            timestamp=time.time(),
                            metric_type=MetricType.CUSTOM,
                            name=f"{benchmark_name}_{key}",
                            value=float(value),
                            unit="units"
                        ))
            
        except Exception as e:
            end_time = time.perf_counter()
            result.execution_time = end_time - start_time
            result.success = False
            result.error_message = str(e)
            logger.error(f"Benchmark {benchmark_name} failed: {e}")
        
        # Store result
        self.results_history.append(result)
        self._store_result(result)
        
        return result
    
    def run_all_benchmarks(self, category_filter: Optional[BenchmarkCategory] = None) -> List[BenchmarkResult]:
        """Run all registered benchmarks."""
        results = []
        
        for benchmark_name, config in self.benchmark_configs.items():
            if category_filter and config['category'] != category_filter:
                continue
            
            logger.info(f"Running benchmark: {benchmark_name}")
            result = self.run_benchmark(benchmark_name)
            results.append(result)
        
        return results
    
    def run_benchmark_suite(
        self,
        iterations: int = 3,
        warmup_iterations: int = 1
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmark suite with multiple iterations."""
        suite_results = defaultdict(list)
        
        for benchmark_name in self.benchmarks:
            logger.info(f"Running benchmark suite for: {benchmark_name}")
            
            # Warmup iterations
            for _ in range(warmup_iterations):
                self.run_benchmark(benchmark_name)
            
            # Actual benchmark iterations
            for i in range(iterations):
                result = self.run_benchmark(benchmark_name)
                suite_results[benchmark_name].append(result)
                logger.info(f"Iteration {i+1}/{iterations}: {result.execution_time:.4f}s")
        
        return dict(suite_results)
    
    def _store_result(self, result: BenchmarkResult) -> None:
        """Store benchmark result to database."""
        if not self.conn:
            return
        
        try:
            metrics_json = json.dumps([metric.to_dict() for metric in result.metrics])
            metadata_json = json.dumps(result.metadata)
            
            self.conn.execute("""
                INSERT INTO benchmark_results 
                (benchmark_name, category, timestamp, execution_time, success, 
                 metrics_json, metadata_json, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.benchmark_name,
                result.category.value,
                result.timestamp,
                result.execution_time,
                result.success,
                metrics_json,
                metadata_json,
                result.error_message
            ))
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store benchmark result: {e}")
    
    def get_benchmark_history(
        self,
        benchmark_name: str,
        days: int = 30
    ) -> List[BenchmarkResult]:
        """Get historical results for a benchmark."""
        if not self.conn:
            return []
        
        cutoff_time = time.time() - (days * 24 * 3600)
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM benchmark_results 
                WHERE benchmark_name = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (benchmark_name, cutoff_time))
            
            results = []
            for row in cursor.fetchall():
                # Reconstruct BenchmarkResult from database row
                # This is simplified - in practice would fully reconstruct the object
                pass
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get benchmark history: {e}")
            return []
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.results_history:
            return {'error': 'No benchmark results available'}
        
        report = {
            'summary': {
                'total_benchmarks': len(set(r.benchmark_name for r in self.results_history)),
                'total_executions': len(self.results_history),
                'success_rate': sum(1 for r in self.results_history if r.success) / len(self.results_history),
                'average_execution_time': sum(r.execution_time for r in self.results_history) / len(self.results_history)
            },
            'benchmarks': {},
            'timestamp': time.time()
        }
        
        # Group results by benchmark
        benchmark_groups = defaultdict(list)
        for result in self.results_history:
            benchmark_groups[result.benchmark_name].append(result)
        
        for benchmark_name, results in benchmark_groups.items():
            execution_times = [r.execution_time for r in results if r.success]
            
            if execution_times:
                report['benchmarks'][benchmark_name] = {
                    'executions': len(results),
                    'success_rate': sum(1 for r in results if r.success) / len(results),
                    'avg_execution_time': statistics.mean(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times),
                    'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
                }
        
        return report


class PerformanceMonitoringSystem:
    """Main performance monitoring and benchmarking system."""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        enable_detailed_profiling: bool = False
    ):
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.statistical_analyzer = StatisticalAnalyzer()
        self.regression_detector = RegressionDetector()
        self.alert_manager = AlertManager()
        self.profiler = PerformanceProfiler(enable_detailed_profiling)
        self.benchmark_suite = BenchmarkSuite(storage_path)
        
        # Background processing
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Default alert rules
        self._setup_default_alerts()
        
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules."""
        self.alert_manager.add_alert_rule(
            "response_time", 1000.0, "greater", AlertLevel.WARNING
        )
        self.alert_manager.add_alert_rule(
            "error_rate", 0.05, "greater", AlertLevel.ERROR
        )
        self.alert_manager.add_alert_rule(
            "cpu_usage", 90.0, "greater", AlertLevel.CRITICAL
        )
        self.alert_manager.add_alert_rule(
            "memory_usage", 85.0, "greater", AlertLevel.WARNING
        )
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Process metrics buffer
                while self.metrics_buffer:
                    try:
                        metric = self.metrics_buffer.popleft()
                        
                        # Add to statistical analyzer
                        self.statistical_analyzer.add_metric(metric)
                        
                        # Check for alerts
                        self.alert_manager.check_alerts(metric)
                        
                        # Check for regressions
                        self._check_for_regressions(metric)
                        
                    except IndexError:
                        break
                
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
    
    def _check_for_regressions(self, metric: PerformanceMetric) -> None:
        """Check for performance regressions."""
        metric_key = f"{metric.name}:{metric.metric_type.value}"
        
        # Get current statistics
        stats = self.statistical_analyzer.calculate_statistics(metric_key)
        if not stats:
            return
        
        # Check for regression against baseline
        regression_result = self.regression_detector.detect_regression(metric_key, stats)
        if regression_result:
            alert = PerformanceAlert(
                timestamp=time.time(),
                level=AlertLevel.WARNING if regression_result['severity'] == 'medium' else AlertLevel.ERROR,
                metric_name=metric.name,
                current_value=regression_result['current'],
                threshold_value=regression_result['baseline'],
                message=f"Performance regression detected: {regression_result['change_percent']:.1f}% change",
                context=regression_result
            )
            self.alert_manager.alerts.append(alert)
        
        # Check for trend-based regression
        trend = self.statistical_analyzer.calculate_trend(metric_key)
        if trend:
            trend_regression = self.regression_detector.analyze_trend_regression(
                metric_key, trend, stats
            )
            if trend_regression:
                alert = PerformanceAlert(
                    timestamp=time.time(),
                    level=AlertLevel.WARNING,
                    metric_name=metric.name,
                    current_value=stats['mean'],
                    threshold_value=0.0,
                    message=f"Negative performance trend detected: {trend_regression['prediction']}",
                    context=trend_regression
                )
                self.alert_manager.alerts.append(alert)
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        self.metrics_buffer.append(metric)
    
    def record_simple_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.CUSTOM,
        unit: str = "units",
        **context
    ) -> None:
        """Record a simple performance metric."""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_type=metric_type,
            name=name,
            value=value,
            unit=unit,
            context=context
        )
        self.record_metric(metric)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'monitoring_active': self.monitoring_active,
            'metrics_buffer_size': len(self.metrics_buffer),
            'total_metrics_processed': len(self.statistical_analyzer.metric_history),
            'active_alerts_count': len(active_alerts),
            'alert_levels': {
                level.value: len([a for a in active_alerts if a.level == level])
                for level in AlertLevel
            },
            'benchmark_summary': self.benchmark_suite.generate_benchmark_report(),
            'top_functions': self.profiler.get_top_functions(),
            'timestamp': time.time()
        }
    
    def generate_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.alert_manager.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        report = {
            'period': {
                'days': days,
                'start_time': cutoff_time,
                'end_time': time.time()
            },
            'summary': {
                'total_alerts': len(recent_alerts),
                'critical_alerts': len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL]),
                'error_alerts': len([a for a in recent_alerts if a.level == AlertLevel.ERROR]),
                'warning_alerts': len([a for a in recent_alerts if a.level == AlertLevel.WARNING])
            },
            'metric_statistics': {},
            'regression_analysis': {},
            'benchmark_results': self.benchmark_suite.generate_benchmark_report(),
            'recommendations': self._generate_recommendations(),
            'timestamp': time.time()
        }
        
        # Add metric statistics
        for metric_key in self.statistical_analyzer.metric_history.keys():
            stats = self.statistical_analyzer.calculate_statistics(metric_key)
            anomalies = self.statistical_analyzer.detect_anomalies(metric_key)
            trend = self.statistical_analyzer.calculate_trend(metric_key)
            
            report['metric_statistics'][metric_key] = {
                'statistics': stats,
                'anomaly_count': len(anomalies),
                'trend_slope': trend,
                'trend_direction': 'increasing' if trend and trend > 0 else 'decreasing' if trend and trend < 0 else 'stable'
            }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check for high error rates
        error_metrics = [
            key for key in self.statistical_analyzer.metric_history.keys()
            if 'error' in key.lower()
        ]
        
        for metric_key in error_metrics:
            stats = self.statistical_analyzer.calculate_statistics(metric_key)
            if stats and stats.get('mean', 0) > 0.01:  # > 1% error rate
                recommendations.append(f"High error rate detected in {metric_key}: {stats['mean']:.2%}")
        
        # Check for high latency
        latency_metrics = [
            key for key in self.statistical_analyzer.metric_history.keys()
            if 'latency' in key.lower() or 'response_time' in key.lower()
        ]
        
        for metric_key in latency_metrics:
            stats = self.statistical_analyzer.calculate_statistics(metric_key)
            if stats and stats.get('p95', 0) > 1.0:  # P95 > 1 second
                recommendations.append(f"High latency P95 in {metric_key}: {stats['p95']:.3f}s")
        
        return recommendations
    
    # Decorator methods for easy instrumentation
    def monitor(
        self,
        metric_name: Optional[str] = None,
        metric_type: MetricType = MetricType.LATENCY
    ):
        """Decorator for automatic performance monitoring."""
        def decorator(func: Callable) -> Callable:
            name = metric_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    execution_time = time.perf_counter() - start_time
                    
                    # Record metric
                    self.record_simple_metric(
                        name=f"{name}_execution_time",
                        value=execution_time,
                        metric_type=metric_type,
                        unit="seconds",
                        success=success
                    )
                
                return result
            
            return wrapper
        return decorator


# Global monitoring system
global_monitoring = PerformanceMonitoringSystem()


def monitor_performance(
    metric_name: Optional[str] = None,
    metric_type: MetricType = MetricType.LATENCY
):
    """Global decorator for performance monitoring."""
    return global_monitoring.monitor(metric_name, metric_type)


def record_metric(name: str, value: float, metric_type: MetricType = MetricType.CUSTOM, **context):
    """Record a performance metric globally."""
    global_monitoring.record_simple_metric(name, value, metric_type, **context)


def get_performance_report() -> Dict[str, Any]:
    """Get global performance report."""
    return global_monitoring.generate_performance_report()