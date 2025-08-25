"""Robust execution framework with comprehensive error handling."""

import time
import logging
import traceback
import functools
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class RetryStrategy(Enum):
    """Retry strategies for failed operations."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExecutionResult:
    """Result of robust execution."""
    success: bool
    result: Any
    error: Optional[Exception]
    error_message: str
    execution_time: float
    retry_count: int
    severity: ErrorSeverity
    metadata: Dict[str, Any]


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Failing, reject all calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class RobustExecutor:
    """Comprehensive error handling and execution framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
    def _calculate_delay(self, retry_count: int, strategy: RetryStrategy, base_delay: float = 1.0) -> float:
        """Calculate retry delay based on strategy."""
        if strategy == RetryStrategy.NONE:
            return 0.0
        elif strategy == RetryStrategy.LINEAR:
            return base_delay * retry_count
        elif strategy == RetryStrategy.EXPONENTIAL:
            return base_delay * (2 ** retry_count)
        elif strategy == RetryStrategy.FIBONACCI:
            if retry_count <= 1:
                return base_delay
            a, b = 1, 1
            for _ in range(retry_count - 1):
                a, b = b, a + b
            return base_delay * b
        else:
            return base_delay
    
    def _classify_error(self, error: Exception) -> ErrorSeverity:
        """Classify error severity."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        critical_patterns = [
            "memory", "disk", "permission", "security", "authentication",
            "outofmemory", "stackoverflow", "segmentation"
        ]
        
        # High severity errors  
        high_patterns = [
            "connection", "network", "timeout", "database", "file"
        ]
        
        # Medium severity errors
        medium_patterns = [
            "validation", "format", "parse", "conversion"
        ]
        
        # Check patterns
        for pattern in critical_patterns:
            if pattern in error_message or pattern in error_type.lower():
                return ErrorSeverity.CRITICAL
                
        for pattern in high_patterns:
            if pattern in error_message or pattern in error_type.lower():
                return ErrorSeverity.HIGH
                
        for pattern in medium_patterns:
            if pattern in error_message or pattern in error_type.lower():
                return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _should_retry(self, error: Exception, severity: ErrorSeverity, retry_count: int, max_retries: int) -> bool:
        """Determine if operation should be retried."""
        if retry_count >= max_retries:
            return False
            
        # Don't retry critical errors
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        # Don't retry certain error types
        non_retryable = [
            ValueError, TypeError, AttributeError, 
            KeyError, IndexError, ImportError,
            SyntaxError, NameError
        ]
        
        if any(isinstance(error, err_type) for err_type in non_retryable):
            return False
        
        return True
    
    def _record_error(self, error: Exception, context: Dict[str, Any]):
        """Record error in history."""
        error_record = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": self._classify_error(error).value,
            "context": context,
            "traceback": traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        
        # Trim history if too large
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def execute(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        circuit_breaker_key: Optional[str] = None,
        fallback_func: Optional[Callable] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute function with comprehensive error handling."""
        
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        # Check circuit breaker
        if circuit_breaker_key:
            if circuit_breaker_key not in self.circuit_breakers:
                self.circuit_breakers[circuit_breaker_key] = CircuitBreaker()
                
            breaker = self.circuit_breakers[circuit_breaker_key]
            
            if not breaker.can_execute():
                error_msg = f"Circuit breaker open for {circuit_breaker_key}"
                return ExecutionResult(
                    success=False,
                    result=None,
                    error=Exception(error_msg),
                    error_message=error_msg,
                    execution_time=time.time() - start_time,
                    retry_count=0,
                    severity=ErrorSeverity.HIGH,
                    metadata={"circuit_breaker_state": breaker.state.value}
                )
        
        while retry_count <= max_retries:
            try:
                # Execute with timeout if specified
                if timeout:
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Function execution timed out after {timeout}s")
                    
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))
                
                # Execute function
                result = func(*args, **kwargs)
                
                if timeout:
                    signal.alarm(0)  # Cancel alarm
                
                # Record success with circuit breaker
                if circuit_breaker_key:
                    self.circuit_breakers[circuit_breaker_key].record_success()
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=True,
                    result=result,
                    error=None,
                    error_message="",
                    execution_time=execution_time,
                    retry_count=retry_count,
                    severity=ErrorSeverity.LOW,
                    metadata={"function": func.__name__}
                )
                
            except Exception as error:
                if timeout:
                    try:
                        import signal
                        signal.alarm(0)  # Cancel alarm
                    except:
                        pass
                
                last_error = error
                severity = self._classify_error(error)
                
                # Record error
                context = {
                    "function": func.__name__,
                    "retry_count": retry_count,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                self._record_error(error, context)
                
                # Record failure with circuit breaker
                if circuit_breaker_key:
                    self.circuit_breakers[circuit_breaker_key].record_failure()
                
                # Check if should retry
                if not self._should_retry(error, severity, retry_count, max_retries):
                    break
                
                retry_count += 1
                
                # Log retry attempt
                self.logger.warning(
                    f"Function {func.__name__} failed (attempt {retry_count}), "
                    f"retrying in {retry_delay}s: {error}"
                )
                
                # Wait before retry
                if retry_count <= max_retries:
                    delay = self._calculate_delay(retry_count, retry_strategy, retry_delay)
                    if delay > 0:
                        time.sleep(delay)
        
        # All retries exhausted, try fallback
        if fallback_func:
            try:
                fallback_result = fallback_func(*args, **kwargs)
                
                return ExecutionResult(
                    success=True,
                    result=fallback_result,
                    error=last_error,
                    error_message=f"Primary function failed, used fallback: {last_error}",
                    execution_time=time.time() - start_time,
                    retry_count=retry_count,
                    severity=ErrorSeverity.MEDIUM,
                    metadata={"used_fallback": True, "primary_error": str(last_error)}
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback function also failed: {fallback_error}")
        
        # Final failure
        execution_time = time.time() - start_time
        severity = self._classify_error(last_error) if last_error else ErrorSeverity.HIGH
        
        return ExecutionResult(
            success=False,
            result=None,
            error=last_error,
            error_message=str(last_error) if last_error else "Unknown error",
            execution_time=execution_time,
            retry_count=retry_count,
            severity=severity,
            metadata={"max_retries_exhausted": True}
        )
    
    def get_error_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of recent errors."""
        recent_errors = self.error_history[-last_n:] if last_n else self.error_history
        
        if not recent_errors:
            return {"total_errors": 0, "error_types": {}, "severities": {}}
        
        # Count by type
        error_types = {}
        severities = {}
        
        for error in recent_errors:
            error_type = error["error_type"]
            severity = error["severity"]
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            severities[severity] = severities.get(severity, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "error_types": error_types,
            "severities": severities,
            "most_recent": recent_errors[-1] if recent_errors else None,
            "time_range": {
                "start": recent_errors[0]["timestamp"] if recent_errors else None,
                "end": recent_errors[-1]["timestamp"] if recent_errors else None
            }
        }


def robust_execution(
    max_retries: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None,
    circuit_breaker_key: Optional[str] = None,
    fallback_func: Optional[Callable] = None
):
    """Decorator for robust function execution."""
    
    executor = RobustExecutor()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = executor.execute(
                func, *args,
                max_retries=max_retries,
                retry_strategy=retry_strategy,
                retry_delay=retry_delay,
                timeout=timeout,
                circuit_breaker_key=circuit_breaker_key,
                fallback_func=fallback_func,
                **kwargs
            )
            
            if result.success:
                return result.result
            else:
                raise result.error
        
        # Add execution info to wrapper
        wrapper._executor = executor
        wrapper._execution_stats = lambda: executor.get_error_summary()
        
        return wrapper
    
    return decorator


if __name__ == "__main__":
    # Test robust execution
    executor = RobustExecutor()
    
    def flaky_function(fail_rate: float = 0.5):
        import random
        if random.random() < fail_rate:
            raise ConnectionError("Simulated network error")
        return {"status": "success", "data": "test_data"}
    
    # Test with retries
    result = executor.execute(
        flaky_function,
        fail_rate=0.3,
        max_retries=5,
        retry_strategy=RetryStrategy.EXPONENTIAL
    )
    
    print("Execution Result:")
    print(f"Success: {result.success}")
    print(f"Result: {result.result}")
    print(f"Retries: {result.retry_count}")
    print(f"Time: {result.execution_time:.3f}s")
    
    print("\nError Summary:")
    print(json.dumps(executor.get_error_summary(), indent=2))