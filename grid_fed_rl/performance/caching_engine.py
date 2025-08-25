"""Advanced caching engine for performance optimization."""

import time
import threading
import hashlib
import pickle
import json
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict
import logging


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used  
    FIFO = "fifo"        # First In, First Out
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on access patterns


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    avg_access_time: float = 0.0
    last_cleanup: float = 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    last_access: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class AdaptiveCache:
    """High-performance adaptive cache with multiple strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: Optional[float] = 3600,  # 1 hour
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        cleanup_interval: float = 300  # 5 minutes
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.cleanup_interval = cleanup_interval
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.metrics = CacheMetrics()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Adaptive strategy parameters
        self.access_patterns: Dict[str, List[float]] = {}
        self.strategy_performance: Dict[CacheStrategy, float] = {}
        self.current_strategy = strategy
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                return len(str(value).encode('utf-8'))
            elif isinstance(value, (list, tuple, dict)):
                return len(json.dumps(value, default=str).encode('utf-8'))
            else:
                return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return len(str(value).encode('utf-8'))
    
    def _generate_key(self, key: Union[str, Tuple[Any, ...]], args: Tuple = (), kwargs: Dict = None) -> str:
        """Generate cache key from function arguments."""
        if isinstance(key, str):
            base_key = key
        else:
            base_key = str(key)
        
        if args or kwargs:
            # Include arguments in key
            arg_string = json.dumps({
                'args': args,
                'kwargs': kwargs or {}
            }, default=str, sort_keys=True)
            
            # Hash for shorter keys
            arg_hash = hashlib.md5(arg_string.encode()).hexdigest()[:8]
            return f"{base_key}:{arg_hash}"
        
        return base_key
    
    def _should_evict_lru(self) -> Optional[str]:
        """Find LRU candidate for eviction."""
        if not self.cache:
            return None
        
        oldest_key = None
        oldest_time = float('inf')
        
        for key, entry in self.cache.items():
            if entry.last_access < oldest_time:
                oldest_time = entry.last_access
                oldest_key = key
        
        return oldest_key
    
    def _should_evict_lfu(self) -> Optional[str]:
        """Find LFU candidate for eviction."""
        if not self.cache:
            return None
        
        least_key = None
        least_count = float('inf')
        
        for key, entry in self.cache.items():
            if entry.access_count < least_count:
                least_count = entry.access_count
                least_key = key
        
        return least_key
    
    def _should_evict_fifo(self) -> Optional[str]:
        """Find FIFO candidate for eviction."""
        if not self.cache:
            return None
        
        oldest_key = None
        oldest_time = float('inf')
        
        for key, entry in self.cache.items():
            if entry.timestamp < oldest_time:
                oldest_time = entry.timestamp
                oldest_key = key
        
        return oldest_key
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on access patterns."""
        # Analyze access patterns and choose best strategy
        current_time = time.time()
        
        # Score different strategies
        strategy_scores = {}
        
        for strategy in [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.FIFO]:
            if strategy == CacheStrategy.LRU:
                candidate = self._should_evict_lru()
            elif strategy == CacheStrategy.LFU:
                candidate = self._should_evict_lfu()
            else:
                candidate = self._should_evict_fifo()
            
            if candidate and candidate in self.cache:
                entry = self.cache[candidate]
                
                # Score based on recency, frequency, and predicted future access
                recency_score = current_time - entry.last_access
                frequency_score = 1.0 / (entry.access_count + 1)
                
                # Predict future access based on pattern
                future_score = 0.0
                if candidate in self.access_patterns:
                    intervals = self.access_patterns[candidate]
                    if len(intervals) >= 2:
                        avg_interval = sum(intervals) / len(intervals)
                        time_since_last = current_time - entry.last_access
                        future_score = max(0, 1 - (time_since_last / avg_interval))
                
                # Combined score (higher means more likely to evict)
                strategy_scores[strategy] = recency_score + frequency_score - future_score
        
        # Choose strategy with highest score
        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            
            if best_strategy == CacheStrategy.LRU:
                return self._should_evict_lru()
            elif best_strategy == CacheStrategy.LFU:
                return self._should_evict_lfu()
            else:
                return self._should_evict_fifo()
        
        return self._should_evict_lru()  # Fallback
    
    def _evict_entries(self):
        """Evict entries based on current strategy."""
        with self.lock:
            while (len(self.cache) >= self.max_size or 
                   self.metrics.size_bytes >= self.max_memory_bytes):
                
                if not self.cache:
                    break
                
                # Find eviction candidate
                if self.strategy == CacheStrategy.LRU:
                    evict_key = self._should_evict_lru()
                elif self.strategy == CacheStrategy.LFU:
                    evict_key = self._should_evict_lfu()
                elif self.strategy == CacheStrategy.FIFO:
                    evict_key = self._should_evict_fifo()
                elif self.strategy == CacheStrategy.ADAPTIVE:
                    evict_key = self._adaptive_eviction()
                else:
                    evict_key = next(iter(self.cache))  # Remove first
                
                if evict_key and evict_key in self.cache:
                    entry = self.cache[evict_key]
                    self.metrics.size_bytes -= entry.size_bytes
                    del self.cache[evict_key]
                    self.metrics.evictions += 1
                else:
                    break  # Safety break
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.cache[key]
                self.metrics.size_bytes -= entry.size_bytes
                del self.cache[key]
                self.metrics.evictions += 1
    
    def _cleanup_worker(self):
        """Background cleanup thread."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
                self.metrics.last_cleanup = time.time()
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        start_time = time.time()
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self.metrics.size_bytes -= entry.size_bytes
                    del self.cache[key]
                    self.metrics.misses += 1
                    return default
                
                # Update access info
                current_time = time.time()
                entry.last_access = current_time
                entry.access_count += 1
                
                # Track access pattern
                if key not in self.access_patterns:
                    self.access_patterns[key] = []
                
                patterns = self.access_patterns[key]
                if len(patterns) > 0:
                    interval = current_time - patterns[-1]
                    patterns.append(interval)
                    
                    # Keep only recent patterns
                    if len(patterns) > 10:
                        patterns.pop(0)
                else:
                    patterns.append(current_time)
                
                # Move to end for LRU
                self.cache.move_to_end(key)
                
                self.metrics.hits += 1
                access_time = time.time() - start_time
                
                # Update average access time
                total_accesses = self.metrics.hits + self.metrics.misses
                self.metrics.avg_access_time = (
                    (self.metrics.avg_access_time * (total_accesses - 1) + access_time) 
                    / total_accesses
                )
                
                return entry.value
            else:
                self.metrics.misses += 1
                return default
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value into cache."""
        with self.lock:
            current_time = time.time()
            size_bytes = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.metrics.size_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=current_time,
                last_access=current_time,
                access_count=1,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Evict if necessary
            self.cache[key] = entry
            self.metrics.size_bytes += size_bytes
            self._evict_entries()
    
    def cached_function(
        self, 
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None
    ) -> Callable:
        """Decorator for caching function results."""
        
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                result = self.get(cache_key)
                
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.put(cache_key, result, ttl)
                
                return result
            
            wrapper._cache = self
            wrapper._original_func = func
            return wrapper
        
        return decorator
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries matching pattern."""
        with self.lock:
            if pattern is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                self.access_patterns.clear()
                self.metrics.size_bytes = 0
                return count
            
            # Pattern matching
            invalidated = []
            for key in self.cache:
                if pattern in key:
                    invalidated.append(key)
            
            for key in invalidated:
                entry = self.cache[key]
                self.metrics.size_bytes -= entry.size_bytes
                del self.cache[key]
                
                if key in self.access_patterns:
                    del self.access_patterns[key]
            
            return len(invalidated)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = (
                self.metrics.hits / (self.metrics.hits + self.metrics.misses)
                if (self.metrics.hits + self.metrics.misses) > 0 else 0
            )
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": self.metrics.size_bytes / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "evictions": self.metrics.evictions,
                "avg_access_time_ms": self.metrics.avg_access_time * 1000,
                "strategy": self.strategy.value,
                "last_cleanup": self.metrics.last_cleanup
            }


# Global cache instance
_global_cache = AdaptiveCache(max_size=5000, max_memory_mb=500)


def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Global cache decorator."""
    return _global_cache.cached_function(ttl=ttl, key_func=key_func)


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_stats()


def clear_cache(pattern: Optional[str] = None) -> int:
    """Clear global cache."""
    return _global_cache.invalidate(pattern)


if __name__ == "__main__":
    # Test caching performance
    cache = AdaptiveCache(max_size=100, max_memory_mb=10)
    
    @cache.cached_function(ttl=60)
    def expensive_computation(n: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return n * n
    
    # Test cache performance
    start = time.time()
    
    # First call (cache miss)
    result1 = expensive_computation(5)
    print(f"First call: {result1} (time: {time.time() - start:.3f}s)")
    
    # Second call (cache hit)
    start = time.time()
    result2 = expensive_computation(5)
    print(f"Second call: {result2} (time: {time.time() - start:.3f}s)")
    
    # Print stats
    stats = cache.get_stats()
    print(f"\nCache Stats:")
    print(f"Hit rate: {stats['hit_rate']:.1%}")
    print(f"Entries: {stats['size']}")
    print(f"Memory: {stats['memory_usage_mb']:.2f} MB")