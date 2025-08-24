"""Error handling and fault tolerance module."""

from .robust_executor import (
    RobustExecutor, ExecutionResult, RetryStrategy, ErrorSeverity,
    CircuitBreaker, CircuitBreakerState, robust_execution
)

__all__ = [
    "RobustExecutor",
    "ExecutionResult", 
    "RetryStrategy",
    "ErrorSeverity",
    "CircuitBreaker",
    "CircuitBreakerState",
    "robust_execution"
]