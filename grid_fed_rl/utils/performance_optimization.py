"""Advanced performance optimization and scaling for grid operations."""

import time
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
from functools import wraps, lru_cache
import json
import pickle
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_adaptive_optimization: bool = True
    profiling_enabled: bool = True
    memory_limit_mb: int = 1000


class AdaptiveCache:
    """High-performance adaptive caching system."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
                
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with LRU eviction."""
        with self.lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
                
            self.cache[key] = value
            self.access_times[key] = current_time
            self.access_counts[key] += 1
            
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
            
        # Find LRU item considering both recency and frequency
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.access_times[k] * (1 + self.access_counts[k] * 0.1))
        
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.access_counts[lru_key]
        
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class PowerFlowCache:
    """Specialized cache for power flow results."""
    
    def __init__(self, max_size: int = 500):
        self.cache = AdaptiveCache(max_size)
        self.tolerance = 1e-6
        
    def _generate_key(self, buses: List, lines: List, loads: Dict, generation: Dict) -> str:
        """Generate cache key from grid state."""
        try:
            # Create hash from grid configuration
            state_data = {
                'num_buses': len(buses),
                'num_lines': len(lines),
                'loads': sorted(loads.items()),
                'generation': sorted(generation.items())
            }
            
            # Simple hash function
            key_str = json.dumps(state_data, sort_keys=True, separators=(',', ':'))
            return str(hash(key_str))
            
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            return f"fallback_{time.time()}"
            
    def get_solution(self, buses: List, lines: List, loads: Dict, generation: Dict) -> Optional[Any]:
        """Get cached power flow solution."""
        key = self._generate_key(buses, lines, loads, generation)
        return self.cache.get(key)
        
    def store_solution(self, buses: List, lines: List, loads: Dict, generation: Dict, solution: Any) -> None:
        """Store power flow solution in cache."""
        key = self._generate_key(buses, lines, loads, generation)
        self.cache.put(key, solution)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class ConcurrentProcessor:
    """Concurrent processing for grid operations."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.active_tasks = 0
        self.task_queue = deque()
        self.results = {}
        self.lock = threading.Lock()
        
    def submit_batch(self, operations: List[Callable], args_list: List[Tuple]) -> List[Any]:
        """Submit batch of operations for concurrent processing."""
        if len(operations) != len(args_list):
            raise ValueError("Operations and arguments lists must have same length")
            
        results = []
        
        if len(operations) == 1 or self.max_workers == 1:
            # Single-threaded execution for small batches
            for op, args in zip(operations, args_list):
                results.append(op(*args))
        else:
            # Multi-threaded execution
            import concurrent.futures
            
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(op, *args) for op, args in zip(operations, args_list)]
                    results = [future.result(timeout=30) for future in futures]
            except ImportError:
                # Fallback to sequential processing
                for op, args in zip(operations, args_list):
                    results.append(op(*args))
            except Exception as e:
                logger.error(f"Concurrent processing failed: {e}")
                # Fallback to sequential processing
                for op, args in zip(operations, args_list):
                    results.append(op(*args))
                    
        return results
        
    def process_grid_states(self, states: List[Dict], processor_func: Callable) -> List[Any]:
        """Process multiple grid states concurrently."""
        operations = [processor_func] * len(states)
        args_list = [(state,) for state in states]
        return self.submit_batch(operations, args_list)


class AdaptiveOptimizer:
    """Adaptive optimization based on performance metrics."""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        self.optimization_parameters = {
            'cache_size': 1000,
            'parallel_threshold': 4,
            'batch_size': 10
        }
        self.adaptation_interval = 100  # Adapt every 100 operations
        self.operation_count = 0
        
    def record_performance(self, operation: str, duration: float, success: bool):
        """Record performance metric."""
        self.performance_history.append({
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })
        self.operation_count += 1
        
        # Trigger adaptation periodically
        if self.operation_count % self.adaptation_interval == 0:
            self.adapt_parameters()
            
    def adapt_parameters(self):
        """Adapt optimization parameters based on performance."""
        if not self.performance_history:
            return
            
        # Analyze recent performance
        recent_ops = list(self.performance_history)[-100:]  # Last 100 operations
        
        # Calculate average duration and success rate
        avg_duration = sum(op['duration'] for op in recent_ops) / len(recent_ops)
        success_rate = sum(1 for op in recent_ops if op['success']) / len(recent_ops)
        
        # Adapt cache size based on performance
        if avg_duration > 0.1 and success_rate > 0.95:  # Slow but reliable
            self.optimization_parameters['cache_size'] = min(2000, 
                                                           self.optimization_parameters['cache_size'] * 1.2)
        elif avg_duration < 0.01:  # Very fast, can reduce cache
            self.optimization_parameters['cache_size'] = max(500, 
                                                           self.optimization_parameters['cache_size'] * 0.9)
            
        # Adapt parallel processing threshold
        parallel_benefit = self._calculate_parallel_benefit(recent_ops)
        if parallel_benefit > 1.5:  # Good speedup
            self.optimization_parameters['parallel_threshold'] = max(2, 
                                                                   self.optimization_parameters['parallel_threshold'] - 1)
        elif parallel_benefit < 1.1:  # Poor speedup
            self.optimization_parameters['parallel_threshold'] = min(10, 
                                                                   self.optimization_parameters['parallel_threshold'] + 1)
            
        logger.info(f"Adapted parameters: {self.optimization_parameters}")
        
    def _calculate_parallel_benefit(self, operations: List[Dict]) -> float:
        """Calculate benefit of parallel processing."""
        # Simplified calculation - in practice would compare parallel vs sequential times
        batch_ops = [op for op in operations if 'batch' in op['operation']]
        single_ops = [op for op in operations if 'batch' not in op['operation']]
        
        if not batch_ops or not single_ops:
            return 1.0
            
        avg_batch_time = sum(op['duration'] for op in batch_ops) / len(batch_ops)
        avg_single_time = sum(op['duration'] for op in single_ops) / len(single_ops)
        
        # Estimate speedup (simplified)
        return avg_single_time / avg_batch_time if avg_batch_time > 0 else 1.0
        
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get current optimal parameters."""
        return self.optimization_parameters.copy()


class MemoryOptimizer:
    """Memory usage optimization and monitoring."""
    
    def __init__(self, limit_mb: int = 1000):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.tracked_objects = {}
        self.cleanup_callbacks = []
        
    def track_object(self, obj_id: str, obj: Any) -> None:
        """Track object for memory management."""
        try:
            import sys
            size = sys.getsizeof(obj)
            self.tracked_objects[obj_id] = {
                'object': obj,
                'size': size,
                'created': time.time()
            }
        except Exception as e:
            logger.warning(f"Failed to track object {obj_id}: {e}")
            
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'tracked_objects': len(self.tracked_objects),
                'tracked_size_mb': sum(obj['size'] for obj in self.tracked_objects.values()) / 1024 / 1024
            }
        except ImportError:
            return {'message': 'psutil not available for memory monitoring'}
        except Exception as e:
            logger.error(f"Memory usage check failed: {e}")
            return {'error': str(e)}
            
    def cleanup_if_needed(self) -> None:
        """Cleanup memory if usage is high."""
        usage = self.get_memory_usage()
        
        if isinstance(usage, dict) and 'rss_mb' in usage:
            if usage['rss_mb'] > self.limit_bytes / 1024 / 1024:
                logger.warning(f"High memory usage: {usage['rss_mb']:.1f} MB")
                self._perform_cleanup()
                
    def _perform_cleanup(self) -> None:
        """Perform memory cleanup."""
        # Remove oldest tracked objects first
        if self.tracked_objects:
            sorted_objects = sorted(self.tracked_objects.items(), 
                                  key=lambda x: x[1]['created'])
            
            # Remove oldest 25% of objects
            num_to_remove = max(1, len(sorted_objects) // 4)
            for obj_id, _ in sorted_objects[:num_to_remove]:
                del self.tracked_objects[obj_id]
                
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
                
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Memory cleanup completed")


class PerformanceProfiler:
    """Advanced performance profiling and analysis."""
    
    def __init__(self):
        self.profiles = {}
        self.current_contexts = {}
        self.lock = threading.Lock()
        
    def profile(self, name: str):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    duration = time.time() - start_time
                    self._record_profile(name, duration, success, error)
                return result
            return wrapper
        return decorator
        
    def start_context(self, name: str) -> str:
        """Start profiling context."""
        context_id = f"{name}_{int(time.time() * 1000000)}"
        self.current_contexts[context_id] = {
            'name': name,
            'start_time': time.time()
        }
        return context_id
        
    def end_context(self, context_id: str, success: bool = True, error: str = None) -> None:
        """End profiling context."""
        if context_id in self.current_contexts:
            context = self.current_contexts[context_id]
            duration = time.time() - context['start_time']
            self._record_profile(context['name'], duration, success, error)
            del self.current_contexts[context_id]
            
    def _record_profile(self, name: str, duration: float, success: bool, error: str = None):
        """Record profiling data."""
        with self.lock:
            if name not in self.profiles:
                self.profiles[name] = {
                    'call_count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'success_count': 0,
                    'error_count': 0,
                    'recent_errors': deque(maxlen=10)
                }
                
            profile = self.profiles[name]
            profile['call_count'] += 1
            profile['total_time'] += duration
            profile['min_time'] = min(profile['min_time'], duration)
            profile['max_time'] = max(profile['max_time'], duration)
            
            if success:
                profile['success_count'] += 1
            else:
                profile['error_count'] += 1
                if error:
                    profile['recent_errors'].append(error)
                    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {}
        
        with self.lock:
            for name, profile in self.profiles.items():
                avg_time = profile['total_time'] / profile['call_count'] if profile['call_count'] > 0 else 0
                success_rate = profile['success_count'] / profile['call_count'] if profile['call_count'] > 0 else 0
                
                report[name] = {
                    'call_count': profile['call_count'],
                    'avg_time_ms': avg_time * 1000,
                    'min_time_ms': profile['min_time'] * 1000 if profile['min_time'] != float('inf') else 0,
                    'max_time_ms': profile['max_time'] * 1000,
                    'total_time_s': profile['total_time'],
                    'success_rate': success_rate,
                    'error_count': profile['error_count'],
                    'recent_errors': list(profile['recent_errors'])
                }
                
        return report


# Global optimization components
performance_config = PerformanceConfig()
power_flow_cache = PowerFlowCache()
concurrent_processor = ConcurrentProcessor()
adaptive_optimizer = AdaptiveOptimizer()
memory_optimizer = MemoryOptimizer()
performance_profiler = PerformanceProfiler()