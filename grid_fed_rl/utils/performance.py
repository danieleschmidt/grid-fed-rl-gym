"""Performance optimization utilities."""

import time
import functools
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple
import logging
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache implementation for power flow results."""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: Any) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hits += 1
            return value
        else:
            self.misses += 1
            return None
            
    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        if key in self.cache:
            # Update existing
            self.cache.pop(key)
        elif len(self.cache) >= self.maxsize:
            # Remove oldest
            self.cache.popitem(last=False)
            
        self.cache[key] = value
        
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'maxsize': self.maxsize
        }


class PowerFlowCache:
    """Specialized cache for power flow solutions."""
    
    def __init__(self, maxsize: int = 64, tolerance: float = 1e-4):
        self.cache = LRUCache(maxsize)
        self.tolerance = tolerance
        
    def _hash_state(self, loads: Dict, generation: Dict, buses: list, lines: list) -> str:
        """Create hash key for power flow state."""
        # Simple hash based on power values and network structure
        load_sum = sum(loads.values()) if loads else 0
        gen_sum = sum(generation.values()) if generation else 0
        n_buses = len(buses)
        n_lines = len(lines)
        
        # Round to tolerance for cache hits with similar values
        load_rounded = round(load_sum / self.tolerance) * self.tolerance
        gen_rounded = round(gen_sum / self.tolerance) * self.tolerance
        
        return f"{n_buses}_{n_lines}_{load_rounded}_{gen_rounded}"
        
    def get(self, loads: Dict, generation: Dict, buses: list, lines: list):
        """Get cached power flow solution."""
        key = self._hash_state(loads, generation, buses, lines)
        return self.cache.get(key)
        
    def put(self, loads: Dict, generation: Dict, buses: list, lines: list, solution):
        """Cache power flow solution."""
        key = self._hash_state(loads, generation, buses, lines)
        self.cache.put(key, solution)
        
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'hits': self.cache.hits,
            'misses': self.cache.misses,
            'size': len(self.cache.cache),
            'maxsize': self.cache.maxsize
        }


class PerformanceProfiler:
    """Profile function execution times."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
        
    def profile(self, func_name: str = None):
        """Decorator to profile function execution."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    self.timings[name].append(execution_time)
                    self.call_counts[name] += 1
                    
            return wrapper
        return decorator
        
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get profiling statistics."""
        stats = {}
        for func_name, times in self.timings.items():
            if times:
                stats[func_name] = {
                    'calls': self.call_counts[func_name],
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times)
                }
        return stats
        
    def reset(self):
        """Reset profiling data."""
        self.timings.clear()
        self.call_counts.clear()
        
    def print_stats(self, sort_by: str = 'total_time', top_n: int = 10):
        """Print profiling statistics."""
        stats = self.get_stats()
        sorted_stats = sorted(
            stats.items(), 
            key=lambda x: x[1][sort_by], 
            reverse=True
        )
        
        print(f"\\nPerformance Profile (Top {top_n} by {sort_by}):")
        print("-" * 80)
        print(f"{'Function':<40} {'Calls':<8} {'Total (s)':<12} {'Avg (ms)':<12} {'Max (ms)':<12}")
        print("-" * 80)
        
        for func_name, data in sorted_stats[:top_n]:
            print(f"{func_name:<40} {data['calls']:<8} {data['total_time']:<12.4f} "
                  f"{data['avg_time']*1000:<12.2f} {data['max_time']*1000:<12.2f}")


def memoize_with_ttl(ttl_seconds: float = 300.0):
    """Memoization decorator with time-to-live."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached result is still valid
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
                else:
                    # Remove expired entry
                    del cache[key]
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            
            # Cleanup old entries periodically
            if len(cache) > 100:  # Arbitrary cleanup threshold
                cutoff_time = current_time - ttl_seconds
                expired_keys = [
                    k for k, (_, timestamp) in cache.items() 
                    if timestamp < cutoff_time
                ]
                for k in expired_keys:
                    del cache[k]
            
            return result
            
        wrapper._cache = cache  # For debugging/inspection
        return wrapper
    return decorator


class BatchProcessor:
    """Process operations in batches for efficiency."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        
    def process_batch(self, items: list, processor_func: Callable) -> list:
        """Process items in batches."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = processor_func(batch)
            results.extend(batch_results)
            
        return results
        
    def vectorized_process(self, data: np.ndarray, func: Callable) -> np.ndarray:
        """Apply vectorized processing where possible."""
        if hasattr(func, '__vectorized__'):
            return func(data)
        else:
            # Fallback to element-wise processing
            return np.array([func(item) for item in data])


class MemoryPool:
    """Memory pool for efficient array allocation."""
    
    def __init__(self, max_arrays: int = 100):
        self.pools = defaultdict(list)  # shape -> list of arrays
        self.max_arrays = max_arrays
        self.allocations = 0
        self.reuses = 0
        
    def get_array(self, shape: Tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """Get array from pool or allocate new one."""
        key = (shape, dtype)
        
        if self.pools[key]:
            array = self.pools[key].pop()
            array.fill(0)  # Clear contents
            self.reuses += 1
            return array
        else:
            self.allocations += 1
            return np.zeros(shape, dtype=dtype)
            
    def return_array(self, array: np.ndarray) -> None:
        """Return array to pool."""
        if array.size == 0:
            return
            
        key = (array.shape, array.dtype)
        
        if len(self.pools[key]) < self.max_arrays:
            self.pools[key].append(array)
            
    def stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        total_pooled = sum(len(arrays) for arrays in self.pools.values())
        total_allocations = self.allocations + self.reuses
        reuse_rate = self.reuses / total_allocations if total_allocations > 0 else 0
        
        return {
            'total_allocations': self.allocations,
            'total_reuses': self.reuses,
            'reuse_rate': reuse_rate,
            'pooled_arrays': total_pooled,
            'pool_types': len(self.pools)
        }


# Global instances for easy access
global_profiler = PerformanceProfiler()
global_cache = PowerFlowCache()
global_memory_pool = MemoryPool()