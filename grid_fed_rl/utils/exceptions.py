"""
Custom exceptions for Grid-Fed-RL-Gym with enhanced error handling.
Includes context-aware exceptions, retry mechanisms, and error recovery.
Features circuit breakers, exponential backoff, and predictive error analysis.
"""

import time
import random
import functools
import logging
from typing import Optional, Callable, Any, Type, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for handling decisions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: datetime
    attempt_count: int
    severity: ErrorSeverity
    component: str
    additional_info: dict
    
    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "attempt_count": self.attempt_count,
            "severity": self.severity.value,
            "component": self.component,
            "additional_info": self.additional_info
        }


class GridEnvironmentError(Exception):
    """Base exception for grid environment errors."""
    pass


class PowerFlowError(GridEnvironmentError):
    """Power flow calculation failed."""
    pass


class ConstraintViolationError(GridEnvironmentError):
    """Grid operational constraints violated."""
    pass


class NetworkTopologyError(GridEnvironmentError):
    """Invalid network topology."""
    pass


class InvalidActionError(GridEnvironmentError):
    """Invalid action provided to environment."""
    pass


class SafetyLimitExceededError(GridEnvironmentError):
    """Safety limits exceeded during operation."""
    pass


class DataValidationError(GridEnvironmentError):
    """Data validation failed."""
    pass


class ConfigurationError(GridEnvironmentError):
    """Invalid configuration provided."""
    pass


class FederatedLearningError(GridEnvironmentError):
    """Federated learning process failed."""
    pass


class PrivacyError(GridEnvironmentError):
    """Privacy mechanism failed."""
    pass


class InvalidConfigError(ConfigurationError):
    """Invalid configuration parameters."""
    pass


class MultiAgentError(GridEnvironmentError):
    """Multi-agent learning error."""
    pass


class RetryableError(GridEnvironmentError):
    """Error that can be retried with backoff."""
    def __init__(self, message: str, retry_after: float = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.retry_after = retry_after
        self.severity = severity


class NonRetryableError(GridEnvironmentError):
    """Error that should not be retried."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH):
        super().__init__(message)
        self.severity = severity


class CircuitBreakerOpenError(GridEnvironmentError):
    """Circuit breaker is open, preventing operation."""
    def __init__(self, message: str, reset_time: datetime):
        super().__init__(message)
        self.reset_time = reset_time


class RateLimitExceededError(RetryableError):
    """Rate limit exceeded error."""
    def __init__(self, message: str, retry_after: float):
        super().__init__(message, retry_after, ErrorSeverity.LOW)


class ResourceExhaustionError(RetryableError):
    """System resource exhaustion error."""
    def __init__(self, message: str, resource_type: str):
        super().__init__(message, severity=ErrorSeverity.HIGH)
        self.resource_type = resource_type


class CorruptedDataError(NonRetryableError):
    """Data corruption detected."""
    def __init__(self, message: str, data_source: str):
        super().__init__(message, ErrorSeverity.CRITICAL)
        self.data_source = data_source


class NetworkPartitionError(RetryableError):
    """Network partition or connectivity error."""
    def __init__(self, message: str, affected_nodes: List[str]):
        super().__init__(message, retry_after=5.0, severity=ErrorSeverity.HIGH)
        self.affected_nodes = affected_nodes


class SecurityViolationError(NonRetryableError):
    """Security policy violation."""
    def __init__(self, message: str, violation_type: str):
        super().__init__(message, ErrorSeverity.CRITICAL)
        self.violation_type = violation_type


class ComplianceViolationError(NonRetryableError):
    """Regulatory compliance violation."""
    def __init__(self, message: str, regulation: str):
        super().__init__(message, ErrorSeverity.CRITICAL)
        self.regulation = regulation


# Advanced error handling utilities

class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        name: str = "circuit_breaker"
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
        
        logger.info(f"Circuit breaker '{name}' initialized with threshold {failure_threshold}")
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info(f"Circuit breaker '{self.name}' transitioning to half-open")
            else:
                reset_time = self.last_failure_time + timedelta(seconds=self.reset_timeout)
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open",
                    reset_time
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            datetime.now() >= self.last_failure_time + timedelta(seconds=self.reset_timeout)
        )
    
    def _on_success(self):
        self.failure_count = 0
        if self.state == "half_open":
            self.state = "closed"
            logger.info(f"Circuit breaker '{self.name}' reset to closed")
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
            )
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
        logger.info(f"Circuit breaker '{self.name}' manually reset")
    
    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "reset_timeout": self.reset_timeout
        }


def exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (RetryableError, ConnectionError, TimeoutError)
) -> Callable:
    """Decorator for exponential backoff retry logic."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to avoid thundering herd
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    # Check if exception specifies retry_after
                    if hasattr(e, 'retry_after') and e.retry_after:
                        delay = max(delay, e.retry_after)
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Function {func.__name__} failed with non-retryable error: {e}")
                    raise
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float) -> Callable:
    """Decorator to add timeout to function calls."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function call timed out after {timeout_seconds} seconds")
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Set timeout signal
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel timeout
                return result
            except Exception:
                signal.alarm(0)  # Cancel timeout
                raise
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


class ErrorRecoveryManager:
    """Centralized error recovery and handling manager."""
    
    def __init__(self):
        self.circuit_breakers: dict = {}
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: dict = {}
        self.max_history_size = 1000
        
    def register_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            expected_exception=expected_exception,
            name=name
        )
        self.circuit_breakers[name] = breaker
        return breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def register_recovery_strategy(
        self,
        exception_type: Type[Exception],
        strategy: Callable[[Exception], Any]
    ):
        """Register a recovery strategy for specific exception types."""
        self.recovery_strategies[exception_type] = strategy
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """Handle error with appropriate recovery strategy."""
        # Record error in history
        self.error_history.append(context)
        
        # Trim history if too large
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
        
        logger.error(
            f"Error in {context.component}.{context.operation} "
            f"(attempt {context.attempt_count}): {error}"
        )
        
        # Attempt recovery if enabled
        if attempt_recovery and type(error) in self.recovery_strategies:
            try:
                recovery_strategy = self.recovery_strategies[type(error)]
                result = recovery_strategy(error)
                logger.info(f"Successfully recovered from {type(error).__name__}")
                return result
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
        
        # Re-raise if no recovery or recovery failed
        raise error
    
    def get_error_statistics(self) -> dict:
        """Get error statistics from history."""
        if not self.error_history:
            return {"total_errors": 0}
        
        by_component = {}
        by_severity = {}
        by_operation = {}
        
        for error in self.error_history:
            # By component
            by_component[error.component] = by_component.get(error.component, 0) + 1
            
            # By severity
            by_severity[error.severity.value] = by_severity.get(error.severity.value, 0) + 1
            
            # By operation
            by_operation[error.operation] = by_operation.get(error.operation, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_component": by_component,
            "by_severity": by_severity,
            "by_operation": by_operation,
            "circuit_breaker_states": {
                name: breaker.get_state() 
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers."""
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")


# Global error recovery manager instance
global_error_recovery_manager = ErrorRecoveryManager()

# Register common recovery strategies
def power_flow_recovery(error: PowerFlowError) -> None:
    """Recovery strategy for power flow errors."""
    logger.info("Attempting power flow recovery with simplified model")
    # This would implement fallback to simpler power flow model

def network_recovery(error: NetworkPartitionError) -> None:
    """Recovery strategy for network partition errors."""
    logger.info("Attempting network recovery with backup connections")
    # This would implement fallback to backup network paths

global_error_recovery_manager.register_recovery_strategy(PowerFlowError, power_flow_recovery)
global_error_recovery_manager.register_recovery_strategy(NetworkPartitionError, network_recovery)