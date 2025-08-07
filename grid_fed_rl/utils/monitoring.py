"""Monitoring and observability utilities for grid operations."""

import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from datetime import datetime

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
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemMetrics':
        return cls(**data)


class GridMonitor:
    """Comprehensive grid monitoring system."""
    
    def __init__(
        self,
        metrics_window: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        self.metrics_window = metrics_window
        self.alert_thresholds = alert_thresholds or {
            'voltage_deviation': 0.1,  # ±10%
            'frequency_deviation': 1.0,  # ±1 Hz
            'line_loading': 0.9,  # 90%
            'power_flow_time': 100.0,  # 100ms
        }
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=metrics_window)
        self.alerts_history: List[Dict[str, Any]] = []
        
        # Real-time counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # Performance tracking
        self.start_time = time.time()
        self.last_metrics_time = time.time()
        
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
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Update counters
        self.counters['total_steps'] += 1
        if violations.get('total_violations', 0) > 0:
            self.counters['violation_episodes'] += 1
        
        self.last_metrics_time = current_time
        
        return metrics
    
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
        return list(self.metrics_history)[-n:]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics over the monitoring window."""
        if not self.metrics_history:
            return {}
        
        metrics_list = list(self.metrics_history)
        
        # Extract arrays for statistics
        voltage_devs = [abs(m.average_voltage - 1.0) for m in metrics_list]
        freq_devs = [m.frequency_deviation for m in metrics_list]
        power_flow_times = [m.power_flow_time_ms for m in metrics_list]
        violations = [m.constraint_violations for m in metrics_list]
        
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
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
    
    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """Export metrics to file."""
        
        if format.lower() == 'json':
            data = {
                'summary': self.get_summary_stats(),
                'metrics': [m.to_dict() for m in self.metrics_history],
                'alerts': self.alerts_history[-100:]  # Last 100 alerts
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format.lower() == 'csv':
            import csv
            
            with open(filepath, 'w', newline='') as f:
                if self.metrics_history:
                    writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].to_dict().keys())
                    writer.writeheader()
                    for metrics in self.metrics_history:
                        writer.writerow(metrics.to_dict())
        
        logger.info(f"Metrics exported to {filepath}")
    
    def reset(self) -> None:
        """Reset monitoring state."""
        self.metrics_history.clear()
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


# Global monitor instance
global_monitor = GridMonitor()
global_health_checker = HealthChecker()