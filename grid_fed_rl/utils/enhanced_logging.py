"""Enhanced logging system for grid operations."""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
import json


class GridFormatter(logging.Formatter):
    """Custom formatter for grid simulation logs."""
    
    def __init__(self):
        super().__init__()
        
    def format(self, record):
        # Add grid-specific context
        if hasattr(record, 'step'):
            step_info = f"[Step {record.step}]"
        else:
            step_info = ""
            
        if hasattr(record, 'component'):
            component_info = f"[{record.component}]"
        else:
            component_info = ""
            
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        return f"{timestamp} {record.levelname:8} {step_info}{component_info} {record.getMessage()}"


class GridLogger:
    """Enhanced logger for grid simulation events."""
    
    def __init__(self, name: str = "grid_fed_rl", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # Console handler with custom formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(GridFormatter())
        self.logger.addHandler(console_handler)
        
        # Performance tracking
        self.performance_log = []
        self.error_log = []
        
    def log_simulation_start(self, config: Dict[str, Any]):
        """Log simulation initialization."""
        self.logger.info(f"Starting grid simulation with config: {config}")
        
    def log_step(self, step: int, component: str, message: str, level: int = logging.INFO):
        """Log step-specific information."""
        extra = {'step': step, 'component': component}
        self.logger.log(level, message, extra=extra)
        
    def log_power_flow(self, step: int, converged: bool, iterations: int, losses: float):
        """Log power flow results."""
        status = "CONVERGED" if converged else "FAILED"
        message = f"Power flow {status} in {iterations} iterations, losses: {losses:.3f} MW"
        self.log_step(step, "PowerFlow", message, logging.INFO if converged else logging.WARNING)
        
    def log_constraint_violation(self, step: int, violation_type: str, details: Dict[str, Any]):
        """Log constraint violations."""
        message = f"Constraint violation: {violation_type} - {details}"
        self.log_step(step, "Safety", message, logging.WARNING)
        
    def log_performance(self, step: int, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics."""
        perf_data = {
            'step': step,
            'metric': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        }
        self.performance_log.append(perf_data)
        
        if len(self.performance_log) % 100 == 0:  # Log summary every 100 entries
            self.logger.info(f"Performance logged: {len(self.performance_log)} metrics")
            
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log errors with full context."""
        error_data = {
            'error': str(error),
            'error_type': type(error).__name__,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.error_log.append(error_data)
        
        self.logger.error(f"Error: {error} | Context: {context}")
        
    def log_safety_intervention(self, step: int, intervention_type: str, reason: str):
        """Log safety system interventions."""
        message = f"Safety intervention: {intervention_type} - Reason: {reason}"
        self.log_step(step, "Safety", message, logging.CRITICAL)
        
    def log_episode_summary(self, episode: int, total_reward: float, steps: int, violations: int):
        """Log episode completion summary."""
        message = f"Episode {episode} completed: {steps} steps, reward: {total_reward:.2f}, violations: {violations}"
        self.logger.info(message)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.performance_log:
            return {"message": "No performance data collected"}
            
        metrics = {}
        for entry in self.performance_log:
            metric = entry['metric']
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(entry['value'])
            
        summary = {}
        for metric, values in metrics.items():
            summary[metric] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values)
            }
            
        return {
            'total_entries': len(self.performance_log),
            'metrics': summary,
            'error_count': len(self.error_log)
        }
        
    def export_logs(self, filename: str):
        """Export logs to file."""
        log_data = {
            'performance': self.performance_log,
            'errors': self.error_log,
            'summary': self.get_performance_summary(),
            'export_time': datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(log_data, f, indent=2)
            self.logger.info(f"Logs exported to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger: GridLogger):
        self.logger = logger
        self.start_times = {}
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
        
    def end_timer(self, operation: str, step: int = None):
        """End timing and log duration."""
        if operation not in self.start_times:
            return
            
        duration = (datetime.now() - self.start_times[operation]).total_seconds()
        del self.start_times[operation]
        
        if step is not None:
            self.logger.log_performance(step, f"{operation}_duration", duration, "seconds")
        else:
            self.logger.logger.info(f"{operation} completed in {duration:.3f} seconds")
            
    def log_memory_usage(self, step: int):
        """Log current memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.log_performance(step, "memory_usage", memory_mb, "MB")
        except ImportError:
            pass  # psutil not available
            
    def log_cpu_usage(self, step: int):
        """Log current CPU usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.logger.log_performance(step, "cpu_usage", cpu_percent, "%")
        except ImportError:
            pass  # psutil not available


# Global logger instance
grid_logger = GridLogger()
performance_monitor = PerformanceMonitor(grid_logger)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Recreate global logger with new level
    global grid_logger, performance_monitor
    grid_logger = GridLogger(level=log_level)
    performance_monitor = PerformanceMonitor(grid_logger)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(GridFormatter())
        grid_logger.logger.addHandler(file_handler)
        
    grid_logger.logger.info(f"Logging initialized at level {level}")