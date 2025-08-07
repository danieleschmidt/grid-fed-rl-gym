```python
"""Monitoring, telemetry, and observability utilities for grid operations."""

import time
import psutil
import threading
import json
import os
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from datetime import datetime
import csv

from .exceptions import GridEnvironmentError

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: float
    step_count: int
    power_flow_time_ms: float
    constraint_violations: int
    safety_interventions: int
    average_voltage: float
    frequency_deviation: float
    total_losses: float
    renewable_utilization: float
    # Additional system resource metrics
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_available_mb: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemMetrics':
        return cls(**data)


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


class GridMonitor:
    """Comprehensive grid monitoring system."""
    
    def __init__(
        self,
        metrics_window: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None,
        collection_interval: float = 1.0
    ):
        self.metrics_window = metrics_window
        self.collection_interval = collection_interval
        self.alert_thresholds = alert_thresholds or {
            'voltage_deviation': 0.1,  # ±10%
            'frequency_deviation': 1.0,  # ±1 Hz
            'line_loading': 0.9,  # 90%
            'power_flow_time': 100.0,  # 100ms
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
        }
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=metrics_window)
        self.alerts_history: List[Dict[str, Any]] = []
        self.grid_metrics: deque = deque(maxlen=10000)
        self.training_metrics: deque = deque(maxlen=10000)
        
        # Real-time counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # Performance tracking
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        
        # Background collection
        self.is_collecting = False
        self.collection_thread = None
        self.lock = threading.Lock()
        
    def start_collection(self) -> None:
        """Start background metric collection."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"Started metrics collection (interval: {self.collection_interval}s)")
        
    def stop_collection(self) -> None:
        """Stop background metric collection."""
        self.is_collecting = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
            
        logger.info("Stopped metrics collection")
        
    def _collection_loop(self) -> None:
        """Main collection loop for system metrics."""
        while self.is_collecting:
            try:
                # Collect system resource metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                # Add to latest metrics if available
                with self.lock:
                    if self.metrics_history:
                        latest = self.metrics_history[-1]
                        latest.cpu_percent = cpu_percent
                        latest.memory_percent = memory.percent
                        latest.memory_used_mb = memory.used / (1024 * 1024)
                        latest.memory_available_mb = memory.available / (1024 * 1024)
                        latest.disk_usage_percent = disk.percent
                        latest.network_bytes_sent = network.bytes_sent
                        latest.network_bytes_recv = network.bytes_recv
                        
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                
            time.sleep(self.collection_interval)
        
    def record_metrics(
        self,
        step_count: int,
        power_flow_time: float,
        grid_state: Dict[str, Any],
        violations: Dict[str, Any]
    ) -> SystemMetrics:
        """Record system metrics for monitoring."""
        
        current_time = time.time()
        
        # Calculate derived metrics
        voltage_data = grid_state.get('bus_voltages', [1.0])
        if not isinstance(voltage_data, (list, np.ndarray)):
            voltage_data = [voltage_data]  # Convert single value to list
        voltage_array = np.array(voltage_data)
        avg_voltage = np.mean(voltage_array)
        voltage_deviation = np.max(np.abs(voltage_array - 1.0))
        
        frequency = grid_state.get('frequency', 60.0)
        frequency_deviation = abs(frequency - 60.0)
        
        total_losses = grid_state.get('losses', 0.0)
        renewable_power = grid_state.get('renewable_power', 0.0)
        total_power = grid_state.get('total_power', 1.0)
        renewable_utilization = renewable_power / max(total_power, 1e-6)
        
        # Create metrics object
        metrics = SystemMetrics(
            timestamp=current_time,
            step_count=step_count,
            power_flow_time_ms=power_flow_time * 1000,
            constraint_violations=violations.get('total_violations', 0),
            safety_interventions=int(violations.get('emergency_action_required', False)),
            average_voltage=avg_voltage,
            frequency_deviation=frequency_deviation,
            total_losses=total_losses,
            renewable_utilization=renewable_utilization
        )
        
        # Store metrics
        with self.lock:
            self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Update counters
        self.counters['total_steps'] += 1
        if violations.get('total_violations', 0) > 0:
            self.counters['violation_episodes'] += 1
        
        self.last_metrics_time = current_time
        
        return metrics
    
    def record_grid_metrics(self, metrics: GridMetrics) -> None:
        """Record grid-specific metrics."""
        with self.lock:
            self.grid_metrics.append(metrics)
            
    def record_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Record training metrics."""
        with self.lock:
            self.training_metrics.append(metrics)
    
    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check metrics against thresholds and generate alerts."""
        
        alerts = []
        
        # Voltage alerts
        voltage_dev = abs(metrics.average_voltage - 1.0)
        if voltage_dev > self.alert_thresholds['voltage_deviation']:
            alerts.append({
                'type': 'voltage_deviation',
                'severity': 'warning' if voltage_dev < 0.15 else 'critical',
                'value': voltage_dev,
                'threshold': self.alert_thresholds['voltage_deviation'],
                'message': f'Average voltage deviation: {voltage_dev:.3f} pu'
            })
        
        # Frequency alerts
        if metrics.frequency_deviation > self.alert_thresholds['frequency_deviation']:
            alerts.append({
                'type': 'frequency_deviation',
                'severity': 'warning' if metrics.frequency_deviation < 2.0 else 'critical',
                'value': metrics.frequency_deviation,
                'threshold': self.alert_thresholds['frequency_deviation'],
                'message': f'Frequency deviation: {metrics.frequency_deviation:.2f} Hz'
            })
        
        # Performance alerts
        if metrics.power_flow_time_ms > self.alert_thresholds['power_flow_time']:
            alerts.append({
                'type': 'performance',
                'severity': 'warning',
                'value': metrics.power_flow_time_ms,
                'threshold': self.alert_thresholds['power_flow_time'],
                'message': f'Slow power flow: {metrics.power_flow_time_ms:.1f} ms'
            })
        
        # System resource alerts
        if metrics.cpu_percent and metrics.cpu_percent > self.alert_thresholds.get('cpu_percent', 90):
            alerts.append({
                'type': 'system_resource',
                'severity': 'warning',
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent'],
                'message': f'High CPU usage: {metrics.cpu_percent:.1f}%'
            })
        
        # Safety alerts
        if metrics.safety_interventions > 0:
            alerts.append({
                'type': 'safety',
                'severity': 'critical',
                'value': metrics.safety_interventions,
                'message': 'Safety intervention triggered'
            })
        
        # Store and log alerts
        for alert in alerts:
            alert['timestamp'] = metrics.timestamp
            alert['step'] = metrics.step_count
            self.alerts_history.append(alert)
            
            # Log alert
            log_level = logging.WARNING if alert['severity'] == 'warning' else logging.CRITICAL
            logger.log(log_level, f"ALERT: {alert['message']}")
    
    def get_recent_metrics(self, n: int = 100) -> List[SystemMetrics]:
        """Get the most recent n metrics."""
        with self.lock:
            return list(self.metrics_history)[-n:]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics over the monitoring window."""
        with self.lock:
            metrics_list = list(self.metrics_history)
        
        if not metrics_list:
            return {}
        
        # Extract arrays for statistics
        voltage_devs = [abs(m.average_voltage - 1.0) for m in metrics_list]
        freq_devs = [m.frequency_deviation for m in metrics_list]
        power_flow_times = [m.power_flow_time_ms for m in metrics_list]
        violations = [m.constraint_violations for m in metrics_list]
        
        current_time = time.time()
        uptime = current_time - self.start_time
        
        stats = {
            'uptime_seconds': uptime,
            'total_steps': self.counters['total_steps'],
            'violation_rate': self.counters['violation_episodes'] / max(1, self.counters['total_steps']),
            'avg_voltage_deviation': np.mean(voltage_devs),
            'max_voltage_deviation': np.max(voltage_devs),
            'avg_frequency_deviation': np.mean(freq_devs),
            'max_frequency_deviation': np.max(freq_devs),
            'avg_power_flow_time_ms': np.mean(power_flow_times),
            'max_power_flow_time_ms': np.max(power_flow_times),
            'total_violations': sum(violations),
            'alert_count': len(self.alerts_history),
            'critical_alerts': len([a for a in self.alerts_history if a['severity'] == 'critical']),
            'recent_renewable_utilization': metrics_list[-1].renewable_utilization if metrics_list else 0.0
        }
        
        # Add system resource stats if available
        cpu_values = [m.cpu_percent for m in metrics_list if m.cpu_percent is not None]
        if cpu_values:
            stats['cpu_percent'] = {
                'mean': np.mean(cpu_values),
                'std': np.std(cpu_values),
                'min': np.min(cpu_values),
                'max': np.max(cpu_values)
            }
        
        memory_values = [m.memory_percent for m in metrics_list if m.memory_percent is not None]
        if memory_values:
            stats['memory_percent'] = {
                'mean': np.mean(memory_values),
                'std': np.std(memory_values),
                'min': np.min(memory_values),
                'max': np.max(memory_values)
            }
        
        return stats
    
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
    
    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """Export metrics to file."""
        
        if format.lower() == 'json':
            with self.lock:
                data = {
                    'summary': self.get_summary_stats(),
                    'metrics': [m.to_dict() for m in self.metrics_history],
                    'grid_metrics': [asdict(m) for m in self.grid_metrics],
                    'training_metrics': [asdict(m) for m in self.training_metrics],
                    'alerts': self.alerts_history[-100:],  # Last 100 alerts
                    'export_timestamp': time.time()
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format.lower() == 'csv':
            with open(filepath, 'w', newline='') as f:
                if self.metrics_history:
                    writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].to_dict().keys())
                    writer.writeheader()
                    for metrics in self.metrics_history:
                        writer.writerow(metrics.to_dict())
        
        logger.info(f"Metrics exported to {filepath}")
    
    def reset(self) -> None:
        """Reset monitoring state."""
        with self.lock:
            self.metrics_history.clear()
            self.grid_metrics.clear()
            self.training_metrics.clear()
            self.alerts_history.clear()
            self.counters.clear()
            self.timers.clear()
            self.start_time = time.time()
        logger.info("Monitor reset")


class HealthChecker:
    """System health monitoring and diagnostics."""
    
    def __init__(self):
        self.health_checks = {
            'power_flow_convergence': self._check_power_flow,
            'voltage_stability': self._check_voltage_stability,
            'frequency_stability': self._check_frequency_stability,
            'system_loading': self._check_system_loading,
            'data_quality': self._check_data_quality
        }
        
        self.last_check_results = {}
        
    def run_health_check(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive health check."""
        
        results = {
            'timestamp': time.time(),
            'overall_health': 'healthy',
            'checks': {},
            'recommendations': []
        }
        
        # Run all health checks
        failed_checks = 0
        warning_checks = 0
        
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = check_func(system_state)
                results['checks'][check_name] = check_result
                
                if check_result['status'] == 'failed':
                    failed_checks += 1
                elif check_result['status'] == 'warning':
                    warning_checks += 1
                    
                # Add recommendations
                if 'recommendation' in check_result:
                    results['recommendations'].append(check_result['recommendation'])
                    
            except Exception as e:
                results['checks'][check_name] = {
                    'status': 'error',
                    'message': f'Health check failed: {e}'
                }
                failed_checks += 1
        
        # Determine overall health
        if failed_checks > 0:
            results['overall_health'] = 'unhealthy'
        elif warning_checks > 2:
            results['overall_health'] = 'degraded'
        
        self.last_check_results = results
        return results
    
    def _check_power_flow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check power flow convergence and quality."""
        converged = state.get('power_flow_converged', True)
        max_mismatch = state.get('max_mismatch', 0.0)
        iterations = state.get('power_flow_iterations', 1)
        
        if not converged:
            return {
                'status': 'failed',
                'message': 'Power flow failed to converge',
                'recommendation': 'Check load/generation balance and network parameters'
            }
        elif max_mismatch > 1e-3:
            return {
                'status': 'warning',
                'message': f'High power flow mismatch: {max_mismatch}',
                'recommendation': 'Consider tightening convergence tolerance'
            }
        elif iterations > 20:
            return {
                'status': 'warning', 
                'message': f'Power flow required {iterations} iterations',
                'recommendation': 'Check for numerical conditioning issues'
            }
        else:
            return {'status': 'healthy', 'message': 'Power flow converging normally'}
    
    def _check_voltage_stability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check voltage stability margins."""
        voltages = np.array(state.get('bus_voltages', [1.0]))
        
        min_voltage = np.min(voltages)
        max_voltage = np.max(voltages)
        voltage_spread = max_voltage - min_voltage
        
        if min_voltage < 0.9 or max_voltage > 1.1:
            return {
                'status': 'failed',
                'message': f'Voltage limits exceeded: {min_voltage:.3f} - {max_voltage:.3f} pu',
                'recommendation': 'Implement voltage control measures'
            }
        elif voltage_spread > 0.15:
            return {
                'status': 'warning',
                'message': f'Large voltage spread: {voltage_spread:.3f} pu',
                'recommendation': 'Consider reactive power optimization'
            }
        else:
            return {'status': 'healthy', 'message': 'Voltage profile within acceptable range'}
    
    def _check_frequency_stability(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check frequency stability."""
        frequency = state.get('frequency', 60.0)
        freq_deviation = abs(frequency - 60.0)
        
        if freq_deviation > 2.0:
            return {
                'status': 'failed',
                'message': f'Large frequency deviation: {freq_deviation:.2f} Hz',
                'recommendation': 'Check generation/load balance and governor response'
            }
        elif freq_deviation > 0.5:
            return {
                'status': 'warning',
                'message': f'Frequency deviation: {freq_deviation:.2f} Hz',
                'recommendation': 'Monitor generation/load balance'
            }
        else:
            return {'status': 'healthy', 'message': 'Frequency within normal range'}
    
    def _check_system_loading(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check system loading levels."""
        loadings = np.array(state.get('line_loadings', [0.5]))
        
        max_loading = np.max(loadings)
        overloaded_lines = np.sum(loadings > 1.0)
        heavily_loaded = np.sum(loadings > 0.8)
        
        if overloaded_lines > 0:
            return {
                'status': 'failed',
                'message': f'{overloaded_lines} lines overloaded',
                'recommendation': 'Implement load shedding or generation redispatch'
            }
        elif heavily_loaded > len(loadings) // 2:
            return {
                'status': 'warning',
                'message': f'{heavily_loaded} lines heavily loaded (>80%)',
                'recommendation': 'Monitor system loading and prepare contingency plans'
            }
        else:
            return {'status': 'healthy', 'message': 'System loading within acceptable limits'}
    
    def _check_data_quality(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check data quality and consistency."""
        issues = []
        
        # Check for NaN or infinite values
        for key, value in state.items():
            if isinstance(value, (list, np.ndarray)):
                if not np.all(np.isfinite(value)):
                    issues.append(f"Non-finite values in {key}")
            elif isinstance(value, (int, float)):
                if not np.isfinite(value):
                    issues.append(f"Non-finite value in {key}")
        
        # Check for reasonable ranges
        voltages = np.array(state.get('bus_voltages', [1.0]))
        if np.any(voltages < 0) or np.any(voltages > 5.0):
            issues.append("Unreasonable voltage values detected")
        
        frequency = state.get('frequency', 60.0)
        if frequency < 30.0 or frequency > 100.0:
            issues.append(f"Unreasonable frequency value: {frequency}")
        
        if issues:
            return {
                'status': 'warning',
                'message': f"Data quality issues: {'; '.join(issues)}",
                'recommendation': 'Check measurement systems and data processing'
            }
        else:
            return {'status': 'healthy', 'message': 'Data quality acceptable'}


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
        self.grid_monitor = GridMonitor()
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_history: List[Dict] = []
        
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        self.grid_monitor.start_collection()
        self.logger.info("Started performance monitoring")
        
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.grid_monitor.stop_collection()
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
        
        self.grid_monitor.record_grid_metrics(metrics)
        
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
        
        self.grid_monitor.record_training_metrics(metrics)
        
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
        system_stats = self.grid_monitor.get_summary_stats()
        grid_stats = self.grid_monitor.get_grid_stats()
        training_stats = self.grid_monitor.get_training_stats()
        
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
        
        self.grid_monitor = GridMonitor()
        self.scaling_history: List[Dict] = []
        
        self.logger = logging.getLogger(__name__)
        
    def should_scale_up(self) -> bool:
        """Check if should scale up based on metrics."""
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False
            
        if self.current_workers >= self.max_workers:
            return False
            
        system_stats = self.grid_monitor.get_summary_stats()
        if not system_stats:
            return False
            
        cpu_stats = system_stats.get("cpu_percent", {})
        memory_stats = system_stats.get("memory_percent", {})
        
        cpu_usage = cpu_stats.get("mean", 0)
        memory_usage = memory_stats.get("mean", 0)
        
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
            
        system_stats = self.grid_monitor.get_summary_stats()
        if not system_stats:
            return False
            
        cpu_stats = system_stats.get("cpu_percent", {})
        memory_stats = system_stats.get("memory_percent", {})
        
        cpu_usage = cpu_stats.get("mean", 0)
        memory_usage = memory_stats.get("mean", 0)
        
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
    monitor: Union[PerformanceMonitor, GridMonitor],
    output_dir: str = "monitoring_dashboard"
) -> str:
    """Create a simple HTML monitoring dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate performance report
    if isinstance(monitor, PerformanceMonitor):
        report = monitor.get_performance_report()
    else:
        # For GridMonitor, construct a report
        report = {
            "report_timestamp": time.time(),
            "system_performance": monitor.get_summary_stats(),
            "grid_performance": monitor.get_grid_stats(),
            "training_performance": monitor.get_training_stats(),
            "recent_alerts": monitor.alerts_history[-100:],
            "alert_summary": {
                "total_alerts_1h": len([a for a in monitor.alerts_history 
                                       if time.time() - a["timestamp"] < 3600])
            }
        }
    
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
            <p>Uptime: {report['system_performance'].get('uptime_seconds', 0) / 3600:.1f} hours</p>
        </div>
        
        <div class="metric-card">
            <h2>Grid Performance</h2>
            <p>Power Flow Convergence Rate: {report['grid_performance'].get('power_flow_convergence_rate', 'N/A'):.2%}</p>
            <p>Voltage Violation Rate: {report['grid_performance'].get('voltage_violation_rate', 'N/A'):.2%}</p>
            <p>Average Reward: {report['grid_performance'].get('rewards', {}).get('mean', 'N/A'):.2f}</p>
        </div>
        
        <div class="metric-card">
            <h2>Recent Alerts ({len(report.get('recent_alerts', []))})</h2>
            <ul>
    """
    
    for alert in report.get('recent_alerts', [])[-10:]:  # Show last 10 alerts
        html_content += f"<li class='alert'>{alert.get('type', 'unknown')}: {alert.get('message', 'No message')}</li>"
    
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


# Global monitor instances
global_monitor = GridMonitor()
global_health_checker = HealthChecker()
```
