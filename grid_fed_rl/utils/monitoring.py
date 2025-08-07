"""Monitoring, telemetry, and observability utilities."""

import time
import psutil
import threading
import json
import os
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging

from .exceptions import GridEnvironmentError


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int


@dataclass
class GridMetrics:
    """Grid-specific performance metrics."""
    timestamp: float
    environment_id: str
    episode: int
    step: int
    total_reward: float
    power_flow_convergence: bool
    power_flow_iterations: int
    voltage_violations: int
    frequency_deviation: float
    total_losses: float
    renewable_generation: float
    load_served: float


@dataclass
class TrainingMetrics:
    """ML training metrics."""
    timestamp: float
    algorithm: str
    episode: int
    step: int
    loss: float
    reward: float
    exploration_rate: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None


class MetricsCollector:
    """Collect and aggregate various metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.is_collecting = False
        self.collection_thread = None
        
        # Metric storage
        self.system_metrics: deque = deque(maxlen=3600)  # 1 hour of data
        self.grid_metrics: deque = deque(maxlen=10000)   # 10k samples
        self.training_metrics: deque = deque(maxlen=10000)
        
        # Aggregated statistics
        self.metric_aggregates: Dict[str, Dict] = defaultdict(dict)
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def start_collection(self) -> None:
        """Start background metric collection."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info(f"Started metrics collection (interval: {self.collection_interval}s)")
        
    def stop_collection(self) -> None:
        """Stop background metric collection."""
        self.is_collecting = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
            
        self.logger.info("Stopped metrics collection")
        
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.is_collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                with self.lock:
                    self.system_metrics.append(system_metrics)
                    
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                
            time.sleep(self.collection_interval)
            
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage (root partition)
        disk = psutil.disk_usage('/')
        
        # Network stats
        network = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv
        )
        
    def record_grid_metrics(self, metrics: GridMetrics) -> None:
        """Record grid-specific metrics."""
        with self.lock:
            self.grid_metrics.append(metrics)
            
    def record_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Record training metrics."""
        with self.lock:
            self.training_metrics.append(metrics)
            
    def get_system_stats(self, window_minutes: int = 10) -> Dict[str, Any]:
        """Get aggregated system statistics."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self.lock:
            recent_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
            
        if not recent_metrics:
            return {}
            
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            "window_minutes": window_minutes,
            "sample_count": len(recent_metrics),
            "cpu_percent": {
                "mean": np.mean(cpu_values),
                "std": np.std(cpu_values),
                "min": np.min(cpu_values),
                "max": np.max(cpu_values)
            },
            "memory_percent": {
                "mean": np.mean(memory_values),
                "std": np.std(memory_values),
                "min": np.min(memory_values),
                "max": np.max(memory_values)
            },
            "memory_used_mb": recent_metrics[-1].memory_used_mb if recent_metrics else 0,
            "disk_usage_percent": recent_metrics[-1].disk_usage_percent if recent_metrics else 0
        }
        
    def get_grid_stats(self, environment_id: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated grid performance statistics."""
        with self.lock:
            if environment_id:
                metrics = [m for m in self.grid_metrics if m.environment_id == environment_id]
            else:
                metrics = list(self.grid_metrics)
                
        if not metrics:
            return {}
            
        # Calculate statistics
        rewards = [m.total_reward for m in metrics]
        convergence_rate = np.mean([m.power_flow_convergence for m in metrics])
        avg_iterations = np.mean([m.power_flow_iterations for m in metrics])
        violation_rate = np.mean([m.voltage_violations > 0 for m in metrics])
        
        return {
            "sample_count": len(metrics),
            "rewards": {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "min": np.min(rewards),
                "max": np.max(rewards)
            },
            "power_flow_convergence_rate": convergence_rate,
            "average_power_flow_iterations": avg_iterations,
            "voltage_violation_rate": violation_rate,
            "average_losses": np.mean([m.total_losses for m in metrics]),
            "average_renewable_generation": np.mean([m.renewable_generation for m in metrics]),
            "load_served_ratio": np.mean([m.load_served for m in metrics])
        }
        
    def get_training_stats(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated training statistics."""
        with self.lock:
            if algorithm:
                metrics = [m for m in self.training_metrics if m.algorithm == algorithm]
            else:
                metrics = list(self.training_metrics)
                
        if not metrics:
            return {}
            
        losses = [m.loss for m in metrics if m.loss is not None]
        rewards = [m.reward for m in metrics]
        
        stats = {
            "sample_count": len(metrics),
            "rewards": {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "min": np.min(rewards),
                "max": np.max(rewards)
            }
        }
        
        if losses:
            stats["losses"] = {
                "mean": np.mean(losses),
                "std": np.std(losses),
                "min": np.min(losses),
                "max": np.max(losses)
            }
            
        # Learning rates if available
        learning_rates = [m.learning_rate for m in metrics if m.learning_rate is not None]
        if learning_rates:
            stats["learning_rate"] = {
                "mean": np.mean(learning_rates),
                "current": learning_rates[-1]
            }
            
        return stats
        
    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """Export collected metrics to file."""
        with self.lock:
            data = {
                "system_metrics": [asdict(m) for m in self.system_metrics],
                "grid_metrics": [asdict(m) for m in self.grid_metrics],
                "training_metrics": [asdict(m) for m in self.training_metrics],
                "export_timestamp": time.time()
            }
            
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        self.logger.info(f"Exported metrics to {filepath}")


class PerformanceMonitor:
    """Monitor performance of grid environments and algorithms."""
    
    def __init__(
        self,
        alert_thresholds: Optional[Dict[str, float]] = None,
        alert_callback: Optional[Callable] = None
    ):
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 80.0,
            "power_flow_failure_rate": 0.1,  # 10% failure rate
            "voltage_violation_rate": 0.05   # 5% violation rate
        }
        
        self.alert_callback = alert_callback
        self.metrics_collector = MetricsCollector()
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history: List[Dict] = []
        
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.metrics_collector.start_collection()
        self.logger.info("Started performance monitoring")
        
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.metrics_collector.stop_collection()
        self.logger.info("Stopped performance monitoring")
        
    def record_environment_performance(
        self,
        env_id: str,
        episode: int,
        step: int,
        reward: float,
        info: Dict[str, Any]
    ) -> None:
        """Record environment performance metrics."""
        metrics = GridMetrics(
            timestamp=time.time(),
            environment_id=env_id,
            episode=episode,
            step=step,
            total_reward=reward,
            power_flow_convergence=info.get("power_flow_converged", True),
            power_flow_iterations=info.get("power_flow_iterations", 1),
            voltage_violations=info.get("voltage_violations", 0),
            frequency_deviation=abs(info.get("frequency", 60.0) - 60.0),
            total_losses=info.get("total_losses", 0.0),
            renewable_generation=info.get("renewable_generation", 0.0),
            load_served=info.get("load_served", 0.0)
        )
        
        self.metrics_collector.record_grid_metrics(metrics)
        
        # Check for alerts
        self._check_grid_alerts(metrics)
        
    def record_training_performance(
        self,
        algorithm: str,
        episode: int,
        step: int,
        loss: float,
        reward: float,
        **kwargs
    ) -> None:
        """Record training performance metrics."""
        metrics = TrainingMetrics(
            timestamp=time.time(),
            algorithm=algorithm,
            episode=episode,
            step=step,
            loss=loss,
            reward=reward,
            exploration_rate=kwargs.get("exploration_rate"),
            learning_rate=kwargs.get("learning_rate"),
            gradient_norm=kwargs.get("gradient_norm")
        )
        
        self.metrics_collector.record_training_metrics(metrics)
        
    def _check_grid_alerts(self, metrics: GridMetrics) -> None:
        """Check grid metrics against alert thresholds."""
        alerts = []
        
        if not metrics.power_flow_convergence:
            alerts.append({
                "type": "power_flow_failure",
                "environment": metrics.environment_id,
                "episode": metrics.episode,
                "step": metrics.step,
                "message": "Power flow failed to converge"
            })
            
        if metrics.voltage_violations > 0:
            alerts.append({
                "type": "voltage_violation",
                "environment": metrics.environment_id,
                "violations": metrics.voltage_violations,
                "message": f"{metrics.voltage_violations} voltage violations detected"
            })
            
        if metrics.frequency_deviation > 0.5:  # 0.5 Hz deviation
            alerts.append({
                "type": "frequency_deviation",
                "environment": metrics.environment_id,
                "deviation": metrics.frequency_deviation,
                "message": f"Frequency deviation: {metrics.frequency_deviation:.2f} Hz"
            })
            
        # Process alerts
        for alert in alerts:
            alert["timestamp"] = metrics.timestamp
            self.alert_history.append(alert)
            
            if self.alert_callback:
                self.alert_callback(alert)
            else:
                self.logger.warning(f"Grid Alert: {alert['message']}")
                
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        system_stats = self.metrics_collector.get_system_stats(window_minutes=10)
        grid_stats = self.metrics_collector.get_grid_stats()
        training_stats = self.metrics_collector.get_training_stats()
        
        recent_alerts = [a for a in self.alert_history if time.time() - a["timestamp"] < 3600]
        
        return {
            "report_timestamp": time.time(),
            "system_performance": system_stats,
            "grid_performance": grid_stats,
            "training_performance": training_stats,
            "recent_alerts": recent_alerts,
            "alert_summary": {
                "total_alerts_1h": len(recent_alerts),
                "power_flow_failures": len([a for a in recent_alerts if a["type"] == "power_flow_failure"]),
                "voltage_violations": len([a for a in recent_alerts if a["type"] == "voltage_violation"]),
                "frequency_deviations": len([a for a in recent_alerts if a["type"] == "frequency_deviation"])
            }
        }
        
    def export_report(self, filepath: str) -> None:
        """Export performance report to file."""
        report = self.get_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Exported performance report to {filepath}")


class AutoScaler:
    """Automatic scaling based on performance metrics."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        cooldown_period: float = 300.0  # 5 minutes
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.current_workers = min_workers
        self.last_scale_time = 0.0
        
        self.metrics_collector = MetricsCollector()
        self.scaling_history: List[Dict] = []
        
        self.logger = logging.getLogger(__name__)
        
    def should_scale_up(self) -> bool:
        """Check if should scale up based on metrics."""
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False
            
        if self.current_workers >= self.max_workers:
            return False
            
        system_stats = self.metrics_collector.get_system_stats(window_minutes=2)
        if not system_stats:
            return False
            
        cpu_usage = system_stats["cpu_percent"]["mean"]
        memory_usage = system_stats["memory_percent"]["mean"]
        
        if cpu_usage > self.scale_up_threshold or memory_usage > self.scale_up_threshold:
            self.logger.info(f"Scale up triggered: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%")
            return True
            
        return False
        
    def should_scale_down(self) -> bool:
        """Check if should scale down based on metrics."""
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False
            
        if self.current_workers <= self.min_workers:
            return False
            
        system_stats = self.metrics_collector.get_system_stats(window_minutes=5)
        if not system_stats:
            return False
            
        cpu_usage = system_stats["cpu_percent"]["mean"]
        memory_usage = system_stats["memory_percent"]["mean"]
        
        if cpu_usage < self.scale_down_threshold and memory_usage < self.scale_down_threshold:
            self.logger.info(f"Scale down triggered: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%")
            return True
            
        return False
        
    def scale_up(self) -> int:
        """Scale up by one worker."""
        new_count = min(self.current_workers + 1, self.max_workers)
        
        if new_count > self.current_workers:
            self.current_workers = new_count
            self.last_scale_time = time.time()
            
            self.scaling_history.append({
                "timestamp": time.time(),
                "action": "scale_up",
                "old_count": self.current_workers - 1,
                "new_count": self.current_workers
            })
            
            self.logger.info(f"Scaled up to {self.current_workers} workers")
            
        return self.current_workers
        
    def scale_down(self) -> int:
        """Scale down by one worker."""
        new_count = max(self.current_workers - 1, self.min_workers)
        
        if new_count < self.current_workers:
            self.current_workers = new_count
            self.last_scale_time = time.time()
            
            self.scaling_history.append({
                "timestamp": time.time(),
                "action": "scale_down", 
                "old_count": self.current_workers + 1,
                "new_count": self.current_workers
            })
            
            self.logger.info(f"Scaled down to {self.current_workers} workers")
            
        return self.current_workers
        
    def get_recommended_workers(self) -> int:
        """Get recommended number of workers based on current metrics."""
        if self.should_scale_up():
            return min(self.current_workers + 1, self.max_workers)
        elif self.should_scale_down():
            return max(self.current_workers - 1, self.min_workers)
        else:
            return self.current_workers
            
    def get_scaling_history(self) -> List[Dict]:
        """Get scaling history."""
        return self.scaling_history.copy()


def create_monitoring_dashboard(
    monitor: PerformanceMonitor,
    output_dir: str = "monitoring_dashboard"
) -> str:
    """Create a simple HTML monitoring dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate performance report
    report = monitor.get_performance_report()
    
    # Create simple HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grid-Fed-RL Performance Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric-card {{ 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px; 
                background-color: #f9f9f9;
            }}
            .alert {{ 
                color: red; 
                font-weight: bold; 
            }}
            .good {{ 
                color: green; 
            }}
            pre {{ 
                background-color: #f0f0f0; 
                padding: 10px; 
                border-radius: 3px; 
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <h1>Grid-Fed-RL Performance Dashboard</h1>
        <p>Generated at: {time.ctime(report['report_timestamp'])}</p>
        
        <div class="metric-card">
            <h2>System Performance</h2>
            <p>CPU Usage: {report['system_performance'].get('cpu_percent', {}).get('mean', 'N/A'):.1f}%</p>
            <p>Memory Usage: {report['system_performance'].get('memory_percent', {}).get('mean', 'N/A'):.1f}%</p>
            <p>Disk Usage: {report['system_performance'].get('disk_usage_percent', 'N/A'):.1f}%</p>
        </div>
        
        <div class="metric-card">
            <h2>Grid Performance</h2>
            <p>Power Flow Convergence Rate: {report['grid_performance'].get('power_flow_convergence_rate', 'N/A'):.2%}</p>
            <p>Voltage Violation Rate: {report['grid_performance'].get('voltage_violation_rate', 'N/A'):.2%}</p>
            <p>Average Reward: {report['grid_performance'].get('rewards', {}).get('mean', 'N/A'):.2f}</p>
        </div>
        
        <div class="metric-card">
            <h2>Recent Alerts ({len(report['recent_alerts'])})</h2>
            <ul>
    """
    
    for alert in report['recent_alerts'][-10:]:  # Show last 10 alerts
        html_content += f"<li class='alert'>{alert['type']}: {alert['message']}</li>"
    
    html_content += """
            </ul>
        </div>
        
        <div class="metric-card">
            <h2>Full Report (JSON)</h2>
            <pre>{}</pre>
        </div>
    </body>
    </html>
    """.format(json.dumps(report, indent=2, default=str))
    
    dashboard_path = os.path.join(output_dir, "dashboard.html")
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
        
    # Also save raw JSON report
    json_path = os.path.join(output_dir, "report.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    return dashboard_path