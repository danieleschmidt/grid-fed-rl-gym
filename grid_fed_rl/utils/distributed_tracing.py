"""Distributed tracing system for federated grid control operations."""

import time
import uuid
import threading
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict, deque
import functools

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Types of spans in distributed tracing."""
    CLIENT = "client"
    SERVER = "server"
    INTERNAL = "internal"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Span completion status."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """Context information for distributed tracing spans."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = None
    
    def __post_init__(self):
        if self.baggage is None:
            self.baggage = {}


@dataclass
class Span:
    """Individual span in a distributed trace."""
    context: SpanContext
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None
    status: SpanStatus = SpanStatus.OK
    kind: SpanKind = SpanKind.INTERNAL
    component: str = "grid_fed_rl"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []
    
    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span with status."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the span."""
        self.tags[key] = value
    
    def log_event(self, event: str, **kwargs):
        """Log an event on the span."""
        log_entry = {
            "timestamp": time.time(),
            "event": event,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def log_error(self, error: Exception):
        """Log an error on the span."""
        self.status = SpanStatus.ERROR
        self.log_event("error", 
                      error_type=type(error).__name__,
                      error_message=str(error))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status.value,
            "kind": self.kind.value,
            "component": self.component
        }


class TraceCollector:
    """Collects and manages distributed trace spans."""
    
    def __init__(self, max_spans: int = 10000):
        self.max_spans = max_spans
        self.spans: Dict[str, List[Span]] = defaultdict(list)
        self.completed_traces: deque = deque(maxlen=1000)
        self.active_spans: Dict[str, Span] = {}
        self.lock = threading.Lock()
    
    def add_span(self, span: Span):
        """Add a span to the collector."""
        with self.lock:
            trace_id = span.context.trace_id
            self.spans[trace_id].append(span)
            
            if span.end_time is None:
                # Active span
                self.active_spans[span.context.span_id] = span
            else:
                # Completed span
                if span.context.span_id in self.active_spans:
                    del self.active_spans[span.context.span_id]
                
                # Check if trace is complete
                self._check_trace_completion(trace_id)
            
            # Cleanup old spans
            total_spans = sum(len(span_list) for span_list in self.spans.values())
            if total_spans > self.max_spans:
                self._cleanup_old_spans()
    
    def _check_trace_completion(self, trace_id: str):
        """Check if a trace is complete and move to completed traces."""
        trace_spans = self.spans[trace_id]
        
        # Check if all spans in trace are completed
        if all(span.end_time is not None for span in trace_spans):
            # Move to completed traces
            trace_data = {
                "trace_id": trace_id,
                "spans": [span.to_dict() for span in trace_spans],
                "start_time": min(span.start_time for span in trace_spans),
                "end_time": max(span.end_time for span in trace_spans),
                "duration": max(span.end_time for span in trace_spans) - min(span.start_time for span in trace_spans),
                "total_spans": len(trace_spans),
                "errors": sum(1 for span in trace_spans if span.status == SpanStatus.ERROR)
            }
            
            self.completed_traces.append(trace_data)
            
            # Remove from active spans collection
            del self.spans[trace_id]
    
    def _cleanup_old_spans(self):
        """Clean up old spans to prevent memory leaks."""
        current_time = time.time()
        old_threshold = current_time - 3600  # 1 hour old
        
        traces_to_remove = []
        for trace_id, span_list in self.spans.items():
            # Remove traces where all spans are old
            if all(span.start_time < old_threshold for span in span_list):
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.spans[trace_id]
            logger.debug(f"Cleaned up old trace: {trace_id}")
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific trace by ID."""
        with self.lock:
            # Check active traces
            if trace_id in self.spans:
                return {
                    "trace_id": trace_id,
                    "spans": [span.to_dict() for span in self.spans[trace_id]],
                    "status": "active"
                }
            
            # Check completed traces
            for trace in self.completed_traces:
                if trace["trace_id"] == trace_id:
                    return trace
        
        return None
    
    def get_active_traces(self) -> List[Dict[str, Any]]:
        """Get all active traces."""
        with self.lock:
            active_traces = []
            for trace_id, span_list in self.spans.items():
                active_traces.append({
                    "trace_id": trace_id,
                    "span_count": len(span_list),
                    "start_time": min(span.start_time for span in span_list),
                    "active_spans": sum(1 for span in span_list if span.end_time is None),
                    "components": list(set(span.component for span in span_list))
                })
            return active_traces
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        with self.lock:
            active_traces = len(self.spans)
            total_active_spans = sum(len(spans) for spans in self.spans.values())
            completed_traces = len(self.completed_traces)
            
            # Calculate average trace duration from completed traces
            if completed_traces > 0:
                avg_duration = sum(trace["duration"] for trace in self.completed_traces) / completed_traces
                error_rate = sum(trace["errors"] for trace in self.completed_traces) / sum(trace["total_spans"] for trace in self.completed_traces)
            else:
                avg_duration = 0.0
                error_rate = 0.0
            
            return {
                "active_traces": active_traces,
                "total_active_spans": total_active_spans,
                "completed_traces": completed_traces,
                "average_trace_duration": avg_duration,
                "error_rate": error_rate,
                "memory_usage_spans": total_active_spans + completed_traces * 10  # Rough estimate
            }


class DistributedTracer:
    """Main distributed tracing system."""
    
    def __init__(self, service_name: str = "grid_fed_rl", collector: Optional[TraceCollector] = None):
        self.service_name = service_name
        self.collector = collector or TraceCollector()
        self.local_context = threading.local()
        
        # Configuration
        self.sampling_rate = 1.0  # Sample all traces by default
        self.enabled = True
        
    def start_span(
        self, 
        operation_name: str, 
        child_of: Optional[SpanContext] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        if not self.enabled:
            # Return no-op span if tracing disabled
            return self._create_noop_span()
        
        # Check sampling
        if not self._should_sample():
            return self._create_noop_span()
        
        # Generate span context
        if child_of is None:
            # Root span
            trace_id = self._generate_trace_id()
            parent_span_id = None
            baggage = {}
        else:
            # Child span
            trace_id = child_of.trace_id
            parent_span_id = child_of.span_id
            baggage = child_of.baggage.copy()
        
        span_id = self._generate_span_id()
        
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )
        
        span = Span(
            context=context,
            operation_name=operation_name,
            start_time=time.time(),
            kind=kind,
            component=self.service_name,
            tags=tags or {}
        )
        
        # Set default tags
        span.set_tag("service.name", self.service_name)
        span.set_tag("thread.id", threading.current_thread().ident)
        
        # Add to collector
        self.collector.add_span(span)
        
        # Set as active span
        self._set_active_span(span)
        
        logger.debug(f"Started span {operation_name} ({span_id}) in trace {trace_id}")
        
        return span
    
    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK):
        """Finish a span."""
        if span and hasattr(span, 'finish'):
            span.finish(status)
            self.collector.add_span(span)  # Update in collector
            self._clear_active_span(span)
            logger.debug(f"Finished span {span.operation_name} ({span.context.span_id})")
    
    def get_active_span(self) -> Optional[Span]:
        """Get the currently active span."""
        return getattr(self.local_context, 'active_span', None)
    
    def _set_active_span(self, span: Span):
        """Set the active span for current thread."""
        self.local_context.active_span = span
    
    def _clear_active_span(self, span: Span):
        """Clear active span if it matches."""
        if hasattr(self.local_context, 'active_span') and self.local_context.active_span == span:
            self.local_context.active_span = None
    
    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        import random
        return random.random() < self.sampling_rate
    
    def _generate_trace_id(self) -> str:
        """Generate a new trace ID."""
        return str(uuid.uuid4())
    
    def _generate_span_id(self) -> str:
        """Generate a new span ID."""
        return str(uuid.uuid4())[:8]
    
    def _create_noop_span(self) -> 'NoOpSpan':
        """Create a no-op span when tracing is disabled."""
        return NoOpSpan()
    
    @contextmanager
    def trace(
        self, 
        operation_name: str, 
        kind: SpanKind = SpanKind.INTERNAL,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing operations."""
        span = self.start_span(operation_name, kind=kind, tags=tags)
        try:
            yield span
        except Exception as e:
            span.log_error(e)
            self.finish_span(span, SpanStatus.ERROR)
            raise
        else:
            self.finish_span(span, SpanStatus.OK)
    
    def inject_context(self, span_context: SpanContext) -> Dict[str, str]:
        """Inject span context into carrier for cross-service communication."""
        return {
            "x-trace-id": span_context.trace_id,
            "x-span-id": span_context.span_id,
            "x-parent-span-id": span_context.parent_span_id or "",
            "x-baggage": json.dumps(span_context.baggage)
        }
    
    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract span context from carrier."""
        try:
            trace_id = carrier.get("x-trace-id")
            span_id = carrier.get("x-span-id")
            parent_span_id = carrier.get("x-parent-span-id") or None
            baggage_str = carrier.get("x-baggage", "{}")
            
            if not trace_id or not span_id:
                return None
            
            baggage = json.loads(baggage_str)
            
            return SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                baggage=baggage
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to extract trace context from carrier")
            return None
    
    def set_sampling_rate(self, rate: float):
        """Set the sampling rate (0.0 to 1.0)."""
        self.sampling_rate = max(0.0, min(1.0, rate))
        logger.info(f"Sampling rate set to {self.sampling_rate}")
    
    def enable_tracing(self, enabled: bool = True):
        """Enable or disable tracing."""
        self.enabled = enabled
        logger.info(f"Tracing {'enabled' if enabled else 'disabled'}")


class NoOpSpan:
    """No-operation span for when tracing is disabled."""
    
    def __init__(self):
        self.context = None
    
    def finish(self, status=None):
        pass
    
    def set_tag(self, key, value):
        pass
    
    def log_event(self, event, **kwargs):
        pass
    
    def log_error(self, error):
        pass


def trace_federated_operation(operation_name: str, component: str = None):
    """Decorator for tracing federated operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get tracer from args (if first arg is an object with tracer)
            tracer = None
            if args and hasattr(args[0], 'tracer'):
                tracer = args[0].tracer
            else:
                tracer = global_tracer
            
            tags = {
                "function.name": func.__name__,
                "function.module": func.__module__
            }
            
            if component:
                tags["component"] = component
            
            with tracer.trace(operation_name, tags=tags) as span:
                # Add function arguments as tags (be careful with sensitive data)
                if len(args) > 1:  # Skip 'self' parameter
                    span.set_tag("args.count", len(args) - 1)
                
                if kwargs:
                    span.set_tag("kwargs.count", len(kwargs))
                    # Add non-sensitive kwargs
                    safe_kwargs = {k: str(v)[:100] for k, v in kwargs.items() 
                                 if not any(sensitive in k.lower() 
                                          for sensitive in ['password', 'secret', 'key', 'token'])}
                    for k, v in safe_kwargs.items():
                        span.set_tag(f"kwargs.{k}", v)
                
                # Execute function
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    span.set_tag("execution.time_ms", execution_time * 1000)
                    span.set_tag("execution.success", True)
                    
                    return result
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    span.set_tag("execution.time_ms", execution_time * 1000)
                    span.set_tag("execution.success", False)
                    span.set_tag("error.type", type(e).__name__)
                    raise
        
        return wrapper
    return decorator


class FederatedTracingMixin:
    """Mixin class to add distributed tracing capabilities to federated components."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = DistributedTracer(service_name=self.__class__.__name__)
        self.client_tracers: Dict[str, DistributedTracer] = {}
    
    def get_client_tracer(self, client_id: str) -> DistributedTracer:
        """Get or create a tracer for a specific client."""
        if client_id not in self.client_tracers:
            self.client_tracers[client_id] = DistributedTracer(
                service_name=f"{self.__class__.__name__}.client.{client_id}",
                collector=self.tracer.collector  # Share the same collector
            )
        return self.client_tracers[client_id]
    
    def trace_operation(self, operation_name: str, **kwargs):
        """Context manager for tracing operations."""
        return self.tracer.trace(operation_name, **kwargs)
    
    def get_tracing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracing statistics."""
        base_stats = self.tracer.collector.get_statistics()
        
        # Add client-specific statistics
        client_stats = {}
        for client_id, client_tracer in self.client_tracers.items():
            client_stats[client_id] = {
                "service_name": client_tracer.service_name,
                "sampling_rate": client_tracer.sampling_rate,
                "enabled": client_tracer.enabled
            }
        
        return {
            **base_stats,
            "client_tracers": client_stats,
            "total_tracers": 1 + len(self.client_tracers)
        }


# Global tracer instance
global_tracer = DistributedTracer()

# Configure global tracer
def configure_global_tracer(service_name: str = "grid_fed_rl", sampling_rate: float = 1.0):
    """Configure the global tracer."""
    global global_tracer
    global_tracer = DistributedTracer(service_name=service_name)
    global_tracer.set_sampling_rate(sampling_rate)
    logger.info(f"Global tracer configured: service={service_name}, sampling_rate={sampling_rate}")


# Example usage decorators for common operations
def trace_power_flow(func):
    """Decorator specifically for power flow operations."""
    return trace_federated_operation("power_flow_solve", "power_flow")(func)

def trace_federated_training(func):
    """Decorator specifically for federated training operations."""
    return trace_federated_operation("federated_training", "federated_learning")(func)

def trace_safety_check(func):
    """Decorator specifically for safety checking operations."""
    return trace_federated_operation("safety_check", "safety")(func)