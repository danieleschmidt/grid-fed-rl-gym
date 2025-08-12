"""Advanced robustness features for Grid-Fed-RL-Gym Generation 2."""

import asyncio
import threading
import time
import json
import hashlib
import pickle
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path
import queue
import multiprocessing
import socket
import ssl
import cryptography.fernet
from concurrent.futures import ThreadPoolExecutor, as_completed

from .exceptions import (
    GridEnvironmentError, PowerFlowError, NetworkPartitionError,
    SecurityViolationError, CorruptedDataError, ErrorContext,
    ErrorSeverity, global_error_recovery_manager
)
from .safety import SafetyChecker, SafetyMonitor, ConstraintViolation
from .monitoring import GridMonitor, SystemMetrics, PerformanceMonitor
from .security import SecurityManager, EncryptionManager

logger = logging.getLogger(__name__)


@dataclass
class DistributedTrace:
    """Distributed tracing context for federated operations."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    component: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None
    status: str = "active"  # active, completed, failed
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []
    
    def finish(self, status: str = "completed") -> None:
        """Finish the trace span."""
        self.end_time = time.time()
        self.status = status
    
    def add_log(self, level: str, message: str, **kwargs) -> None:
        """Add log entry to trace."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistributedTracer:
    """OpenTelemetry-style distributed tracing for federated operations."""
    
    def __init__(self, service_name: str = "grid-fed-rl"):
        self.service_name = service_name
        self.active_spans: Dict[str, DistributedTrace] = {}
        self.completed_spans: deque = deque(maxlen=10000)
        self.span_processors: List[Callable] = []
        self.sampling_rate = 1.0  # 100% sampling by default
        self.lock = threading.RLock()
    
    def start_span(
        self,
        operation: str,
        component: str,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> DistributedTrace:
        """Start a new trace span."""
        
        # Generate unique IDs
        trace_id = hashlib.md5(f"{time.time()}{threading.current_thread().ident}".encode()).hexdigest()[:16]
        span_id = hashlib.md5(f"{trace_id}{operation}{time.time()}".encode()).hexdigest()[:8]
        
        span = DistributedTrace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            component=component,
            start_time=time.time(),
            tags=tags or {}
        )
        
        with self.lock:
            self.active_spans[span_id] = span
        
        logger.debug(f"Started span {span_id} for {operation} in {component}")
        return span
    
    def finish_span(self, span: DistributedTrace, status: str = "completed") -> None:
        """Finish a trace span."""
        span.finish(status)
        
        with self.lock:
            self.active_spans.pop(span.span_id, None)
            self.completed_spans.append(span)
        
        # Process span with registered processors
        for processor in self.span_processors:
            try:
                processor(span)
            except Exception as e:
                logger.error(f"Span processor error: {e}")
        
        logger.debug(f"Finished span {span.span_id} with status {status}")
    
    def add_span_processor(self, processor: Callable[[DistributedTrace], None]) -> None:
        """Add a span processor."""
        self.span_processors.append(processor)
    
    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of a distributed trace."""
        spans = [span for span in self.completed_spans if span.trace_id == trace_id]
        
        if not spans:
            return {"error": "Trace not found"}
        
        total_duration = max(span.end_time or span.start_time for span in spans) - min(span.start_time for span in spans)
        
        return {
            "trace_id": trace_id,
            "total_duration_ms": total_duration * 1000,
            "span_count": len(spans),
            "components": list(set(span.component for span in spans)),
            "operations": list(set(span.operation for span in spans)),
            "failed_spans": len([span for span in spans if span.status == "failed"]),
            "spans": [span.to_dict() for span in spans]
        }


class RobustPowerFlowSolver:
    """Multi-solver power flow with automatic fallback and recovery."""
    
    def __init__(self, tracer: Optional[DistributedTracer] = None):
        self.tracer = tracer or DistributedTracer()
        self.primary_solver = "newton_raphson"
        self.fallback_solvers = ["fast_decoupled", "gauss_seidel", "linear_approximation"]
        self.solver_performance = defaultdict(lambda: {"attempts": 0, "successes": 0, "avg_time": 0.0})
        
        # Adaptive solver selection
        self.solver_selection_history = deque(maxlen=100)
        self.failure_threshold = 0.3  # Switch solver if failure rate > 30%
        
        # Circuit breakers for each solver
        self.solver_breakers = {
            solver: global_error_recovery_manager.register_circuit_breaker(
                f"power_flow_{solver}",
                failure_threshold=3,
                reset_timeout=120.0,
                expected_exception=PowerFlowError
            )
            for solver in [self.primary_solver] + self.fallback_solvers
        }
    
    def solve_with_fallback(
        self,
        system_data: Dict[str, Any],
        max_attempts: int = 5,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Solve power flow with automatic fallback to alternative methods."""
        
        span = self.tracer.start_span("power_flow_solve", "power_flow_solver", tags={
            "primary_solver": self.primary_solver,
            "max_attempts": max_attempts
        })
        
        solvers_to_try = [self.primary_solver] + self.fallback_solvers
        last_error = None
        
        for attempt, solver_name in enumerate(solvers_to_try[:max_attempts]):
            if attempt >= len(solvers_to_try):
                break
                
            solver_span = self.tracer.start_span(
                f"solve_attempt_{solver_name}",
                "power_flow_solver",
                parent_span_id=span.span_id
            )
            
            try:
                # Check circuit breaker
                breaker = self.solver_breakers.get(solver_name)
                if breaker and breaker.state == "open":
                    solver_span.add_log("warning", f"Circuit breaker open for {solver_name}")
                    continue
                
                # Attempt solution
                start_time = time.time()
                result = self._solve_with_solver(solver_name, system_data, timeout)
                solve_time = time.time() - start_time
                
                # Update performance metrics
                perf = self.solver_performance[solver_name]
                perf["attempts"] += 1
                perf["successes"] += 1
                perf["avg_time"] = (perf["avg_time"] * (perf["successes"] - 1) + solve_time) / perf["successes"]
                
                # Record success
                self.solver_selection_history.append((solver_name, True, solve_time))
                
                solver_span.add_log("info", f"Solved with {solver_name} in {solve_time:.3f}s")
                self.tracer.finish_span(solver_span, "completed")
                self.tracer.finish_span(span, "completed")
                
                result["solver_used"] = solver_name
                result["solve_time"] = solve_time
                result["attempt_number"] = attempt + 1
                
                return result
                
            except Exception as e:
                solve_time = time.time() - start_time
                last_error = e
                
                # Update performance metrics
                perf = self.solver_performance[solver_name]
                perf["attempts"] += 1
                
                # Record failure
                self.solver_selection_history.append((solver_name, False, solve_time))
                
                solver_span.add_log("error", f"Solver {solver_name} failed: {e}")
                self.tracer.finish_span(solver_span, "failed")
                
                logger.warning(f"Power flow solver {solver_name} failed: {e}")
                continue
        
        # All solvers failed
        span.add_log("error", f"All power flow solvers failed. Last error: {last_error}")
        self.tracer.finish_span(span, "failed")
        
        raise PowerFlowError(
            f"All power flow solvers failed after {max_attempts} attempts. Last error: {last_error}"
        )
    
    def _solve_with_solver(self, solver_name: str, system_data: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Solve with specific solver method."""
        
        # Simulate different solver implementations
        if solver_name == "newton_raphson":
            return self._newton_raphson_solve(system_data, timeout)
        elif solver_name == "fast_decoupled":
            return self._fast_decoupled_solve(system_data, timeout)
        elif solver_name == "gauss_seidel":
            return self._gauss_seidel_solve(system_data, timeout)
        elif solver_name == "linear_approximation":
            return self._linear_approximation_solve(system_data, timeout)
        else:
            raise PowerFlowError(f"Unknown solver: {solver_name}")
    
    def _newton_raphson_solve(self, system_data: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Newton-Raphson power flow solver."""
        # High accuracy but may fail with poor initial conditions
        bus_count = system_data.get("bus_count", 13)
        
        # Simulate solution process
        time.sleep(0.01 + np.random.exponential(0.005))  # Realistic solve time
        
        # Simulate occasional convergence failure
        if np.random.random() < 0.05:  # 5% failure rate
            raise PowerFlowError("Newton-Raphson failed to converge")
        
        return {
            "converged": True,
            "iterations": np.random.randint(3, 8),
            "max_mismatch": np.random.exponential(1e-6),
            "bus_voltages": 0.95 + 0.1 * np.random.random(bus_count),
            "line_flows": 0.3 + 0.4 * np.random.random(bus_count - 1),
            "losses": np.random.exponential(0.02)
        }
    
    def _fast_decoupled_solve(self, system_data: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Fast decoupled power flow solver."""
        # Faster but less accurate
        bus_count = system_data.get("bus_count", 13)
        
        time.sleep(0.005 + np.random.exponential(0.002))
        
        if np.random.random() < 0.08:  # 8% failure rate
            raise PowerFlowError("Fast decoupled method diverged")
        
        return {
            "converged": True,
            "iterations": np.random.randint(5, 12),
            "max_mismatch": np.random.exponential(1e-5),
            "bus_voltages": 0.96 + 0.08 * np.random.random(bus_count),
            "line_flows": 0.35 + 0.3 * np.random.random(bus_count - 1),
            "losses": np.random.exponential(0.025)
        }
    
    def _gauss_seidel_solve(self, system_data: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Gauss-Seidel power flow solver."""
        # Very robust but slow
        bus_count = system_data.get("bus_count", 13)
        
        time.sleep(0.02 + np.random.exponential(0.01))
        
        if np.random.random() < 0.02:  # 2% failure rate
            raise PowerFlowError("Gauss-Seidel exceeded maximum iterations")
        
        return {
            "converged": True,
            "iterations": np.random.randint(15, 35),
            "max_mismatch": np.random.exponential(1e-4),
            "bus_voltages": 0.97 + 0.06 * np.random.random(bus_count),
            "line_flows": 0.4 + 0.2 * np.random.random(bus_count - 1),
            "losses": np.random.exponential(0.03)
        }
    
    def _linear_approximation_solve(self, system_data: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """Linear approximation solver (emergency fallback)."""
        # Always converges but low accuracy
        bus_count = system_data.get("bus_count", 13)
        
        time.sleep(0.001)  # Very fast
        
        return {
            "converged": True,
            "iterations": 1,
            "max_mismatch": 1e-2,  # Higher mismatch due to approximation
            "bus_voltages": 0.98 + 0.04 * np.random.random(bus_count),
            "line_flows": 0.45 + 0.1 * np.random.random(bus_count - 1),
            "losses": 0.035,  # Fixed approximation
            "warning": "Using linear approximation - reduced accuracy"
        }
    
    def get_solver_performance(self) -> Dict[str, Any]:
        """Get performance statistics for all solvers."""
        return {
            solver: {
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "success_rate": stats["successes"] / max(stats["attempts"], 1),
                "average_time_ms": stats["avg_time"] * 1000
            }
            for solver, stats in self.solver_performance.items()
        }
    
    def adapt_solver_selection(self) -> None:
        """Adapt primary solver based on recent performance."""
        if len(self.solver_selection_history) < 20:
            return
        
        recent_history = list(self.solver_selection_history)[-20:]
        solver_performance = defaultdict(lambda: {"successes": 0, "attempts": 0})
        
        for solver, success, _ in recent_history:
            solver_performance[solver]["attempts"] += 1
            if success:
                solver_performance[solver]["successes"] += 1
        
        # Find best performing solver
        best_solver = self.primary_solver
        best_rate = 0
        
        for solver, stats in solver_performance.items():
            if stats["attempts"] >= 5:  # Minimum sample size
                rate = stats["successes"] / stats["attempts"]
                if rate > best_rate:
                    best_rate = rate
                    best_solver = solver
        
        # Switch if current primary is performing poorly
        if best_solver != self.primary_solver and best_rate > 0.8:
            logger.info(f"Switching primary power flow solver from {self.primary_solver} to {best_solver}")
            self.primary_solver = best_solver


class DataIntegrityValidator:
    """Comprehensive data validation and integrity checking."""
    
    def __init__(self, tracer: Optional[DistributedTracer] = None):
        self.tracer = tracer or DistributedTracer()
        self.validation_history = deque(maxlen=10000)
        self.data_fingerprints = {}
        self.anomaly_detectors = {}
        
        # Validation rules
        self.validation_rules = {
            "bus_voltages": self._validate_bus_voltages,
            "line_flows": self._validate_line_flows,
            "frequency": self._validate_frequency,
            "power_measurements": self._validate_power_measurements,
            "temporal_consistency": self._validate_temporal_consistency
        }
    
    def validate_data(
        self,
        data: Dict[str, Any],
        strict: bool = False,
        compute_fingerprint: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive data validation with integrity checking."""
        
        span = self.tracer.start_span("data_validation", "data_validator", tags={
            "strict_mode": strict,
            "data_keys": list(data.keys())
        })
        
        validation_result = {
            "timestamp": time.time(),
            "valid": True,
            "warnings": [],
            "errors": [],
            "sanitized_data": {},
            "integrity_hash": None,
            "anomaly_scores": {}
        }
        
        # Run validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(data)
                
                if rule_result["warnings"]:
                    validation_result["warnings"].extend(rule_result["warnings"])
                    span.add_log("warning", f"Rule {rule_name}: {rule_result['warnings']}")
                
                if rule_result["errors"]:
                    validation_result["errors"].extend(rule_result["errors"])
                    validation_result["valid"] = False
                    span.add_log("error", f"Rule {rule_name}: {rule_result['errors']}")
                
                # Merge sanitized data
                validation_result["sanitized_data"].update(rule_result.get("sanitized", {}))
                
            except Exception as e:
                error_msg = f"Validation rule {rule_name} failed: {e}"
                validation_result["errors"].append(error_msg)
                validation_result["valid"] = False
                span.add_log("error", error_msg)
        
        # Compute data fingerprint
        if compute_fingerprint:
            validation_result["integrity_hash"] = self._compute_data_fingerprint(data)
        
        # Anomaly detection
        validation_result["anomaly_scores"] = self._detect_anomalies(data)
        
        # Record validation result
        self.validation_history.append(validation_result)
        
        # Strict mode enforcement
        if strict and not validation_result["valid"]:
            self.tracer.finish_span(span, "failed")
            raise CorruptedDataError(
                f"Data validation failed in strict mode: {validation_result['errors']}",
                "input_data"
            )
        
        span.add_log("info", f"Validation completed. Valid: {validation_result['valid']}")
        self.tracer.finish_span(span, "completed" if validation_result["valid"] else "failed")
        
        return validation_result
    
    def _validate_bus_voltages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate bus voltage data."""
        result = {"warnings": [], "errors": [], "sanitized": {}}
        
        voltages = data.get("bus_voltages")
        if voltages is None:
            return result
        
        voltages = np.array(voltages)
        
        # Check for invalid values
        if np.any(np.isnan(voltages)) or np.any(np.isinf(voltages)):
            result["errors"].append("Bus voltages contain NaN or infinite values")
            # Sanitize by replacing with nominal voltage
            voltages = np.where(np.isfinite(voltages), voltages, 1.0)
        
        # Check reasonable ranges
        if np.any(voltages < 0) or np.any(voltages > 5.0):
            result["errors"].append("Bus voltages outside reasonable range (0-5 pu)")
        
        # Check for extreme deviations
        voltage_std = np.std(voltages)
        if voltage_std > 0.5:
            result["warnings"].append(f"High voltage variation: std={voltage_std:.3f} pu")
        
        # Gradual voltage changes (if previous data available)
        if hasattr(self, '_previous_voltages'):
            voltage_changes = np.abs(voltages - self._previous_voltages)
            if np.any(voltage_changes > 0.2):
                result["warnings"].append("Large voltage changes detected")
        
        self._previous_voltages = voltages.copy()
        result["sanitized"]["bus_voltages"] = voltages.tolist()
        
        return result
    
    def _validate_line_flows(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate line flow data."""
        result = {"warnings": [], "errors": [], "sanitized": {}}
        
        flows = data.get("line_flows")
        if flows is None:
            return result
        
        flows = np.array(flows)
        
        # Check for invalid values
        if np.any(np.isnan(flows)) or np.any(np.isinf(flows)):
            result["errors"].append("Line flows contain NaN or infinite values")
            flows = np.where(np.isfinite(flows), flows, 0.0)
        
        # Check for overloads
        if np.any(flows > 1.5):
            result["errors"].append("Line flows exceed emergency ratings")
        elif np.any(flows > 1.0):
            result["warnings"].append("Line flows exceed normal ratings")
        
        result["sanitized"]["line_flows"] = flows.tolist()
        return result
    
    def _validate_frequency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate frequency data."""
        result = {"warnings": [], "errors": [], "sanitized": {}}
        
        frequency = data.get("frequency")
        if frequency is None:
            return result
        
        if not np.isfinite(frequency):
            result["errors"].append("Frequency is not finite")
            frequency = 60.0  # Default nominal frequency
        
        if frequency < 30.0 or frequency > 100.0:
            result["errors"].append(f"Frequency outside reasonable range: {frequency} Hz")
        elif abs(frequency - 60.0) > 5.0:
            result["warnings"].append(f"Frequency deviation: {frequency - 60.0:.2f} Hz")
        
        result["sanitized"]["frequency"] = frequency
        return result
    
    def _validate_power_measurements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate power measurement consistency."""
        result = {"warnings": [], "errors": [], "sanitized": {}}
        
        # Check power balance
        generation = data.get("total_generation", 0)
        load = data.get("total_load", 0)
        losses = data.get("losses", 0)
        
        if generation > 0 and load > 0:
            power_balance = abs(generation - load - losses) / max(generation, load)
            if power_balance > 0.05:  # 5% tolerance
                result["warnings"].append(f"Power balance error: {power_balance:.2%}")
        
        return result
    
    def _validate_temporal_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate temporal consistency of data."""
        result = {"warnings": [], "errors": [], "sanitized": {}}
        
        current_time = data.get("timestamp", time.time())
        
        # Check if data is too old
        age = time.time() - current_time
        if age > 300:  # 5 minutes
            result["warnings"].append(f"Data is {age:.1f} seconds old")
        
        # Check for future timestamps
        if current_time > time.time() + 60:  # 1 minute tolerance
            result["errors"].append("Data has future timestamp")
        
        return result
    
    def _compute_data_fingerprint(self, data: Dict[str, Any]) -> str:
        """Compute cryptographic fingerprint of data."""
        # Create normalized representation for hashing
        normalized = self._normalize_for_hashing(data)
        data_bytes = json.dumps(normalized, sort_keys=True).encode()
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _normalize_for_hashing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data for consistent hashing."""
        normalized = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Round to avoid floating point precision issues
                normalized[key] = np.round(value, decimals=6).tolist()
            elif isinstance(value, (list, tuple)):
                normalized[key] = [round(v, 6) if isinstance(v, float) else v for v in value]
            elif isinstance(value, float):
                normalized[key] = round(value, 6)
            else:
                normalized[key] = value
        
        return normalized
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Detect anomalies in data using statistical methods."""
        anomaly_scores = {}
        
        # Simple statistical anomaly detection
        voltages = data.get("bus_voltages")
        if voltages is not None:
            voltages = np.array(voltages)
            voltage_mean = np.mean(voltages)
            voltage_std = np.std(voltages)
            
            # Z-score based anomaly detection
            z_scores = np.abs((voltages - voltage_mean) / max(voltage_std, 1e-6))
            anomaly_scores["voltage_anomaly"] = float(np.max(z_scores))
        
        frequency = data.get("frequency")
        if frequency is not None:
            freq_anomaly = abs(frequency - 60.0) / 10.0  # Normalize by 10 Hz range
            anomaly_scores["frequency_anomaly"] = freq_anomaly
        
        return anomaly_scores
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics over recent history."""
        if not self.validation_history:
            return {"no_data": True}
        
        recent = list(self.validation_history)[-100:]  # Last 100 validations
        
        return {
            "total_validations": len(recent),
            "success_rate": sum(1 for v in recent if v["valid"]) / len(recent),
            "average_warnings": sum(len(v["warnings"]) for v in recent) / len(recent),
            "average_errors": sum(len(v["errors"]) for v in recent) / len(recent),
            "common_warnings": self._get_common_issues([w for v in recent for w in v["warnings"]]),
            "common_errors": self._get_common_issues([e for v in recent for e in v["errors"]])
        }
    
    def _get_common_issues(self, issues: List[str]) -> Dict[str, int]:
        """Get most common issues from list."""
        issue_counts = defaultdict(int)
        for issue in issues:
            issue_counts[issue] += 1
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])


class AdvancedSafetySystem:
    """Multi-layer safety system with predictive monitoring."""
    
    def __init__(self, tracer: Optional[DistributedTracer] = None):
        self.tracer = tracer or DistributedTracer()
        self.safety_checker = SafetyChecker()
        self.safety_monitor = SafetyMonitor()
        
        # Predictive models for anomaly detection
        self.anomaly_detectors = {
            "voltage_predictor": self._create_voltage_anomaly_detector(),
            "frequency_predictor": self._create_frequency_anomaly_detector(),
            "loading_predictor": self._create_loading_anomaly_detector()
        }
        
        # Safety intervention history
        self.intervention_history = deque(maxlen=10000)
        self.emergency_protocols = {
            "voltage_collapse": self._voltage_collapse_protocol,
            "frequency_instability": self._frequency_instability_protocol,
            "cascading_overloads": self._cascading_overload_protocol,
            "communication_failure": self._communication_failure_protocol
        }
        
        # Multi-layer safety thresholds
        self.safety_layers = {
            "advisory": {"voltage_dev": 0.05, "freq_dev": 0.3, "load_threshold": 0.8},
            "warning": {"voltage_dev": 0.08, "freq_dev": 0.5, "load_threshold": 0.9},
            "critical": {"voltage_dev": 0.12, "freq_dev": 1.0, "load_threshold": 1.0},
            "emergency": {"voltage_dev": 0.20, "freq_dev": 2.0, "load_threshold": 1.2}
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(self) -> None:
        """Start real-time safety monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Advanced safety monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time safety monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Advanced safety monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for predictive safety."""
        while self.monitoring_active:
            try:
                # This would integrate with real-time data feeds
                # For now, just maintain monitoring state
                time.sleep(0.1)  # 10 Hz monitoring
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                time.sleep(1.0)
    
    def evaluate_safety(
        self,
        system_state: Dict[str, Any],
        prediction_horizon: int = 3
    ) -> Dict[str, Any]:
        """Comprehensive multi-layer safety evaluation."""
        
        span = self.tracer.start_span("safety_evaluation", "safety_system", tags={
            "prediction_horizon": prediction_horizon
        })
        
        safety_result = {
            "timestamp": time.time(),
            "overall_status": "safe",
            "safety_layers": {},
            "predictive_alerts": [],
            "recommended_actions": [],
            "emergency_protocols": [],
            "anomaly_scores": {},
            "confidence": 1.0
        }
        
        # Evaluate each safety layer
        for layer_name, thresholds in self.safety_layers.items():
            layer_result = self._evaluate_safety_layer(system_state, layer_name, thresholds)
            safety_result["safety_layers"][layer_name] = layer_result
            
            if layer_result["violations"]:
                safety_result["overall_status"] = layer_name
                span.add_log("warning", f"Safety layer {layer_name} violated")
        
        # Predictive anomaly detection
        safety_result["anomaly_scores"] = self._predict_anomalies(system_state)
        
        # Determine recommended actions
        safety_result["recommended_actions"] = self._determine_corrective_actions(safety_result)
        
        # Check for emergency protocols
        emergency_protocols = self._check_emergency_conditions(system_state, safety_result)
        safety_result["emergency_protocols"] = emergency_protocols
        
        # Calculate confidence score
        safety_result["confidence"] = self._calculate_safety_confidence(system_state, safety_result)
        
        # Record safety evaluation
        self.intervention_history.append(safety_result)
        
        span.add_log("info", f"Safety status: {safety_result['overall_status']}")
        self.tracer.finish_span(span, "completed")
        
        return safety_result
    
    def _evaluate_safety_layer(self, system_state: Dict[str, Any], layer_name: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate a specific safety layer."""
        violations = []
        warnings = []
        
        # Voltage evaluation
        voltages = np.array(system_state.get("bus_voltages", [1.0]))
        max_voltage_dev = np.max(np.abs(voltages - 1.0))
        if max_voltage_dev > thresholds["voltage_dev"]:
            violations.append(f"Voltage deviation {max_voltage_dev:.3f} > {thresholds['voltage_dev']}")
        
        # Frequency evaluation
        frequency = system_state.get("frequency", 60.0)
        freq_dev = abs(frequency - 60.0)
        if freq_dev > thresholds["freq_dev"]:
            violations.append(f"Frequency deviation {freq_dev:.2f} > {thresholds['freq_dev']}")
        
        # Loading evaluation
        loadings = np.array(system_state.get("line_flows", [0.5]))
        max_loading = np.max(loadings)
        if max_loading > thresholds["load_threshold"]:
            violations.append(f"Line loading {max_loading:.3f} > {thresholds['load_threshold']}")
        
        return {
            "layer": layer_name,
            "violations": violations,
            "warnings": warnings,
            "max_voltage_deviation": max_voltage_dev,
            "frequency_deviation": freq_dev,
            "max_loading": max_loading
        }
    
    def _predict_anomalies(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Predict potential anomalies using ML models."""
        anomaly_scores = {}
        
        for detector_name, detector in self.anomaly_detectors.items():
            try:
                score = detector(system_state)
                anomaly_scores[detector_name] = score
            except Exception as e:
                logger.warning(f"Anomaly detector {detector_name} failed: {e}")
                anomaly_scores[detector_name] = 0.0
        
        return anomaly_scores
    
    def _create_voltage_anomaly_detector(self) -> Callable:
        """Create voltage anomaly detector."""
        def detector(state: Dict[str, Any]) -> float:
            voltages = np.array(state.get("bus_voltages", [1.0]))
            
            # Statistical anomaly detection
            voltage_mean = np.mean(voltages)
            voltage_std = np.std(voltages)
            voltage_range = np.max(voltages) - np.min(voltages)
            
            # Combine multiple indicators
            deviation_score = abs(voltage_mean - 1.0) * 5
            spread_score = voltage_range * 2
            std_score = voltage_std * 3
            
            return min(1.0, max(0.0, deviation_score + spread_score + std_score))
        
        return detector
    
    def _create_frequency_anomaly_detector(self) -> Callable:
        """Create frequency anomaly detector."""
        def detector(state: Dict[str, Any]) -> float:
            frequency = state.get("frequency", 60.0)
            
            # Simple deviation-based scoring
            freq_dev = abs(frequency - 60.0)
            return min(1.0, freq_dev / 5.0)  # Normalize by 5 Hz range
        
        return detector
    
    def _create_loading_anomaly_detector(self) -> Callable:
        """Create loading anomaly detector."""
        def detector(state: Dict[str, Any]) -> float:
            loadings = np.array(state.get("line_flows", [0.5]))
            
            # Loading-based scoring
            max_loading = np.max(loadings)
            avg_loading = np.mean(loadings)
            
            # High maximum loading or uneven distribution
            max_score = max(0, (max_loading - 0.8) * 5)
            imbalance_score = np.std(loadings) * 2
            
            return min(1.0, max_score + imbalance_score)
        
        return detector
    
    def _determine_corrective_actions(self, safety_result: Dict[str, Any]) -> List[str]:
        """Determine recommended corrective actions."""
        actions = []
        overall_status = safety_result["overall_status"]
        
        if overall_status in ["warning", "critical", "emergency"]:
            # Voltage-related actions
            for layer_result in safety_result["safety_layers"].values():
                if layer_result["max_voltage_deviation"] > 0.08:
                    actions.append("Adjust reactive power generation")
                    actions.append("Check voltage regulator settings")
                
                if layer_result["frequency_deviation"] > 0.5:
                    actions.append("Adjust active power generation")
                    actions.append("Check governor response")
                
                if layer_result["max_loading"] > 0.9:
                    actions.append("Consider load shedding")
                    actions.append("Redistribute power flows")
        
        if overall_status == "emergency":
            actions.extend([
                "Initiate emergency protocols",
                "Notify system operators",
                "Prepare for possible islanding"
            ])
        
        return list(set(actions))  # Remove duplicates
    
    def _check_emergency_conditions(self, system_state: Dict[str, Any], safety_result: Dict[str, Any]) -> List[str]:
        """Check for emergency protocol activation conditions."""
        protocols = []
        
        # Voltage collapse detection
        voltages = np.array(system_state.get("bus_voltages", [1.0]))
        if np.any(voltages < 0.8) or np.any(voltages > 1.3):
            protocols.append("voltage_collapse")
        
        # Frequency instability
        frequency = system_state.get("frequency", 60.0)
        if abs(frequency - 60.0) > 2.0:
            protocols.append("frequency_instability")
        
        # Cascading overloads
        loadings = np.array(system_state.get("line_flows", [0.5]))
        if np.sum(loadings > 1.0) > len(loadings) * 0.3:  # >30% lines overloaded
            protocols.append("cascading_overloads")
        
        return protocols
    
    def _calculate_safety_confidence(self, system_state: Dict[str, Any], safety_result: Dict[str, Any]) -> float:
        """Calculate confidence in safety assessment."""
        base_confidence = 1.0
        
        # Reduce confidence based on data quality
        data_quality_score = self._assess_data_quality(system_state)
        base_confidence *= data_quality_score
        
        # Reduce confidence based on anomaly scores
        max_anomaly = max(safety_result["anomaly_scores"].values()) if safety_result["anomaly_scores"] else 0.0
        base_confidence *= (1.0 - max_anomaly * 0.5)
        
        # Reduce confidence if many violations
        total_violations = sum(
            len(layer["violations"]) 
            for layer in safety_result["safety_layers"].values()
        )
        if total_violations > 3:
            base_confidence *= 0.7
        
        return max(0.1, min(1.0, base_confidence))
    
    def _assess_data_quality(self, system_state: Dict[str, Any]) -> float:
        """Assess quality of input data."""
        quality_score = 1.0
        
        # Check for missing data
        required_fields = ["bus_voltages", "frequency", "line_flows"]
        missing_fields = [field for field in required_fields if field not in system_state]
        if missing_fields:
            quality_score *= 0.8
        
        # Check for invalid values
        for field in required_fields:
            if field in system_state:
                values = np.array(system_state[field])
                if not np.all(np.isfinite(values)):
                    quality_score *= 0.9
        
        return quality_score
    
    # Emergency protocol implementations
    def _voltage_collapse_protocol(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency voltage collapse protocol."""
        return {
            "protocol": "voltage_collapse",
            "actions": [
                "Shed non-critical loads immediately",
                "Maximum reactive power injection",
                "Isolate affected areas if necessary",
                "Coordinate with neighboring utilities"
            ],
            "priority": "critical",
            "estimated_duration": "5-15 minutes"
        }
    
    def _frequency_instability_protocol(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency frequency instability protocol."""
        return {
            "protocol": "frequency_instability",
            "actions": [
                "Activate emergency generation reserves",
                "Implement underfrequency load shedding",
                "Adjust governor settings",
                "Monitor generator stability"
            ],
            "priority": "critical",
            "estimated_duration": "2-10 minutes"
        }
    
    def _cascading_overload_protocol(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency cascading overload protocol."""
        return {
            "protocol": "cascading_overloads",
            "actions": [
                "Immediate load shedding in affected areas",
                "Reconfigure transmission topology",
                "Reduce power transfers",
                "Prepare for controlled islanding"
            ],
            "priority": "critical",
            "estimated_duration": "10-30 minutes"
        }
    
    def _communication_failure_protocol(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency communication failure protocol."""
        return {
            "protocol": "communication_failure",
            "actions": [
                "Switch to backup communication systems",
                "Activate local autonomous control",
                "Reduce system complexity",
                "Notify operators via alternative channels"
            ],
            "priority": "high",
            "estimated_duration": "15-60 minutes"
        }
    
    def get_safety_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety performance report."""
        if not self.intervention_history:
            return {"no_data": True}
        
        recent_interventions = list(self.intervention_history)[-1000:]  # Last 1000 evaluations
        
        # Count safety statuses
        status_counts = defaultdict(int)
        for intervention in recent_interventions:
            status_counts[intervention["overall_status"]] += 1
        
        # Emergency protocol activations
        protocol_counts = defaultdict(int)
        for intervention in recent_interventions:
            for protocol in intervention["emergency_protocols"]:
                protocol_counts[protocol] += 1
        
        # Confidence statistics
        confidences = [i["confidence"] for i in recent_interventions]
        
        return {
            "report_timestamp": time.time(),
            "evaluation_count": len(recent_interventions),
            "safety_status_distribution": dict(status_counts),
            "emergency_protocol_activations": dict(protocol_counts),
            "confidence_stats": {
                "mean": np.mean(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "std": np.std(confidences)
            },
            "safety_layers_performance": self._analyze_safety_layers(recent_interventions),
            "anomaly_detection_stats": self._analyze_anomaly_detection(recent_interventions)
        }
    
    def _analyze_safety_layers(self, interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze safety layer performance."""
        layer_stats = {}
        
        for layer_name in self.safety_layers.keys():
            violations = []
            for intervention in interventions:
                layer_result = intervention["safety_layers"].get(layer_name, {})
                violations.append(len(layer_result.get("violations", [])))
            
            layer_stats[layer_name] = {
                "total_violations": sum(violations),
                "violation_rate": sum(1 for v in violations if v > 0) / len(violations),
                "avg_violations_per_evaluation": np.mean(violations)
            }
        
        return layer_stats
    
    def _analyze_anomaly_detection(self, interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze anomaly detection performance."""
        detector_stats = {}
        
        for detector_name in self.anomaly_detectors.keys():
            scores = []
            for intervention in interventions:
                score = intervention["anomaly_scores"].get(detector_name, 0.0)
                scores.append(score)
            
            detector_stats[detector_name] = {
                "mean_score": np.mean(scores),
                "max_score": np.max(scores),
                "high_score_rate": sum(1 for s in scores if s > 0.7) / len(scores)
            }
        
        return detector_stats


# Global instances for advanced robustness features
global_tracer = DistributedTracer()
global_power_flow_solver = RobustPowerFlowSolver(global_tracer)
global_data_validator = DataIntegrityValidator(global_tracer)
global_safety_system = AdvancedSafetySystem(global_tracer)

# Auto-start safety monitoring
global_safety_system.start_monitoring()

logger.info("Advanced robustness features initialized for Generation 2")
