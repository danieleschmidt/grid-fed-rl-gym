"""Utilities for grid-fed-rl-gym."""

from .validation import validate_action, validate_network_parameters, sanitize_config
from .exceptions import PowerFlowError, InvalidActionError, SafetyLimitExceededError
from .logging_config import setup_logging
# Performance and safety imports - graceful failure
try:
    from .performance import global_profiler, global_cache
    performance_monitor = global_profiler
    measure_time = global_profiler.profile
except ImportError:
    # Create dummy performance monitor
    class DummyProfiler:
        def profile(self, name=None):
            def decorator(func):
                return func
            return decorator
    performance_monitor = DummyProfiler()
    measure_time = performance_monitor.profile

try:
    from .safety import SafetyChecker, ConstraintViolation
except ImportError:
    # Create dummy classes if safety module not available
    class SafetyChecker:
        def check_constraints(self, *args, **kwargs):
            return {}
    class ConstraintViolation:
        pass

__all__ = [
    "validate_action",
    "validate_network_parameters", 
    "sanitize_config",
    "PowerFlowError",
    "InvalidActionError",
    "SafetyLimitExceededError",
    "setup_logging",
    "performance_monitor",
    "measure_time", 
    "SafetyChecker",
    "ConstraintViolation"
]