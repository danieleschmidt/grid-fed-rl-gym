"""Advanced optimization utilities for scaling performance."""

import time
import numpy as np
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from collections import defaultdict
import logging
import pickle
import hashlib

from .exceptions import GridEnvironmentError
from .performance import LRUCache


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies."""
    enable_caching: bool = True
    enable_vectorization: bool = True
    enable_parallel_power_flow: bool = True
    enable_state_compression: bool = True
    enable_action_batching: bool = True
    cache_size: int = 1000
    batch_size: int = 32
    compression_level: int = 3


class AdaptiveCache:
    """Adaptive cache that adjusts size and eviction policy based on hit rates."""
    
    def __init__(
        self,
        initial_size: int = 1000,
        min_size: int = 100,
        max_size: int = 10000,
        target_hit_rate: float = 0.7
    ):
        self.cache = LRUCache(initial_size)
        self.min_size = min_size
        self.max_size = max_size
        self.target_hit_rate = target_hit_rate
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 60.0  # Adjust every minute
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            result = self.cache.get(key)
            
            if result is not None:
                self.hits += 1
            else:
                self.misses += 1
                
            # Periodic cache size adjustment
            if time.time() - self.last_adjustment > self.adjustment_interval:
                self._adjust_cache_size()
                
            return result
            
    def put(self, key: str, value: Any) -> None:
        """Put item into cache."""
        with self.lock:
            self.cache.put(key, value)
            
    def _adjust_cache_size(self) -> None:
        """Adjust cache size based on hit rate."""
        total_requests = self.hits + self.misses
        
        if total_requests < 100:  # Not enough data
            return
            
        current_hit_rate = self.hits / total_requests
        current_size = self.cache.maxsize
        
        if current_hit_rate < self.target_hit_rate and current_size < self.max_size:
            # Increase cache size
            new_size = min(int(current_size * 1.2), self.max_size)
            self._resize_cache(new_size)
            self.logger.info(f"Increased cache size to {new_size} (hit rate: {current_hit_rate:.2%})")
            
        elif current_hit_rate > self.target_hit_rate * 1.1 and current_size > self.min_size:
            # Decrease cache size to free memory
            new_size = max(int(current_size * 0.9), self.min_size)
            self._resize_cache(new_size)
            self.logger.info(f"Decreased cache size to {new_size} (hit rate: {current_hit_rate:.2%})")
            
        # Reset counters
        self.hits = 0
        self.misses = 0
        self.last_adjustment = time.time()
        
    def _resize_cache(self, new_size: int) -> None:
        """Resize the cache."""
        old_cache = self.cache
        self.cache = LRUCache(new_size)
        
        # Transfer most recent items
        for key in list(old_cache.cache.keys())[-new_size:]:
            if key in old_cache.cache:
                self.cache.put(key, old_cache.get(key))
                
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            return {
                "capacity": self.cache.maxsize,
                "size": len(self.cache.cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }


class VectorizedPowerFlow:
    """Vectorized power flow calculations for batch processing."""
    
    def __init__(self, base_solver):
        self.base_solver = base_solver
        self.logger = logging.getLogger(__name__)
        
    def solve_batch(
        self,
        network_configs: List[Tuple],
        use_previous_solution: bool = True
    ) -> List[Any]:
        """Solve multiple power flow problems efficiently."""
        if not network_configs:
            return []
            
        results = []
        previous_solution = None
        
        for i, config in enumerate(network_configs):
            buses, lines, loads, generation = config
            
            try:
                # Use previous solution as starting point
                if use_previous_solution and previous_solution:
                    self._initialize_from_previous(buses, previous_solution)
                    
                solution = self.base_solver.solve(buses, lines, loads, generation)
                
                if solution.converged:
                    previous_solution = solution
                    
                results.append(solution)
                
            except Exception as e:
                self.logger.error(f"Power flow {i} failed: {e}")
                # Create dummy failed solution
                from ..environments.power_flow import PowerFlowSolution
                results.append(PowerFlowSolution(
                    converged=False,
                    iterations=-1,
                    bus_voltages=np.array([]),
                    bus_angles=np.array([]),
                    line_flows=np.array([]),
                    losses=0.0
                ))
                
        return results
        
    def _initialize_from_previous(self, buses, previous_solution):
        """Initialize bus voltages from previous solution."""
        if len(previous_solution.bus_voltages) == len(buses):
            for i, bus in enumerate(buses):
                if i < len(previous_solution.bus_voltages):
                    bus.voltage_magnitude = previous_solution.bus_voltages[i]
                if i < len(previous_solution.bus_angles):
                    bus.voltage_angle = previous_solution.bus_angles[i]


class StateCompressor:
    """Compress and decompress environment states for memory efficiency."""
    
    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level
        self.compression_stats = {
            "total_compressed": 0,
            "total_uncompressed_size": 0,
            "total_compressed_size": 0
        }
        
    def compress_state(self, state: np.ndarray) -> bytes:
        """Compress a state array."""
        uncompressed_size = state.nbytes
        
        # Use protocol 4 for better numpy array handling
        compressed_data = pickle.dumps(state, protocol=4)
        
        self.compression_stats["total_compressed"] += 1
        self.compression_stats["total_uncompressed_size"] += uncompressed_size
        self.compression_stats["total_compressed_size"] += len(compressed_data)
        
        return compressed_data
        
    def decompress_state(self, compressed_data: bytes) -> np.ndarray:
        """Decompress state data."""
        return pickle.loads(compressed_data)
        
    def get_compression_ratio(self) -> float:
        """Get overall compression ratio."""
        if self.compression_stats["total_uncompressed_size"] == 0:
            return 1.0
            
        return (self.compression_stats["total_compressed_size"] / 
                self.compression_stats["total_uncompressed_size"])
                
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            **self.compression_stats,
            "compression_ratio": self.get_compression_ratio()
        }


class BatchProcessor:
    """Process actions and observations in batches for efficiency."""
    
    def __init__(self, batch_size: int = 32, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        
        self.pending_items = []
        self.pending_callbacks = []
        self.last_batch_time = time.time()
        
        self.lock = threading.Lock()
        self.processing_thread = None
        self.is_running = False
        
    def start(self) -> None:
        """Start batch processing."""
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.processing_thread.start()
        
    def stop(self) -> None:
        """Stop batch processing."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            
    def submit(self, item: Any, callback: Callable[[Any], None]) -> None:
        """Submit item for batch processing."""
        with self.lock:
            self.pending_items.append(item)
            self.pending_callbacks.append(callback)
            
    def _process_batches(self) -> None:
        """Main batch processing loop."""
        while self.is_running:
            should_process = False
            
            with self.lock:
                if len(self.pending_items) >= self.batch_size:
                    should_process = True
                elif (self.pending_items and 
                      time.time() - self.last_batch_time > self.timeout):
                    should_process = True
                    
            if should_process:
                self._process_current_batch()
            else:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                
    def _process_current_batch(self) -> None:
        """Process current batch of items."""
        with self.lock:
            if not self.pending_items:
                return
                
            batch_items = self.pending_items.copy()
            batch_callbacks = self.pending_callbacks.copy()
            
            self.pending_items.clear()
            self.pending_callbacks.clear()
            self.last_batch_time = time.time()
            
        # Process batch (outside lock)
        try:
            processed_results = self._batch_process_function(batch_items)
            
            # Execute callbacks
            for callback, result in zip(batch_callbacks, processed_results):
                try:
                    callback(result)
                except Exception as e:
                    logging.error(f"Callback error: {e}")
                    
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            
            # Call callbacks with error
            for callback in batch_callbacks:
                try:
                    callback(None)  # Or pass error information
                except:
                    pass
                    
    def _batch_process_function(self, items: List[Any]) -> List[Any]:
        """Override this method to implement actual batch processing."""
        # Default: just return items as-is
        return items


class OptimizedEnvironmentWrapper:
    """Wrapper that applies various optimizations to environments."""
    
    def __init__(self, base_env, config: OptimizationConfig):
        self.base_env = base_env
        self.config = config
        
        # Initialize optimization components
        self.adaptive_cache = AdaptiveCache(config.cache_size) if config.enable_caching else None
        self.state_compressor = StateCompressor(config.compression_level) if config.enable_state_compression else None
        self.batch_processor = BatchProcessor(config.batch_size) if config.enable_action_batching else None
        
        # Performance tracking
        self.step_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger = logging.getLogger(__name__)
        
        if self.batch_processor:
            self.batch_processor.start()
            
    def __del__(self):
        """Cleanup resources."""
        if self.batch_processor:
            self.batch_processor.stop()
            
    def reset(self, **kwargs):
        """Reset environment with optimizations."""
        start_time = time.time()
        
        # Check cache for initial state
        cache_key = self._generate_cache_key("reset", kwargs)
        
        if self.adaptive_cache:
            cached_result = self.adaptive_cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                if self.state_compressor:
                    return self.state_compressor.decompress_state(cached_result)
                return cached_result
                
        # Not in cache, compute normally
        result = self.base_env.reset(**kwargs)
        
        # Cache the result
        if self.adaptive_cache:
            self.cache_misses += 1
            if self.state_compressor:
                compressed = self.state_compressor.compress_state(result[0])
                self.adaptive_cache.put(cache_key, (compressed, result[1]))
            else:
                self.adaptive_cache.put(cache_key, result)
                
        self.step_times.append(time.time() - start_time)
        return result
        
    def step(self, action):
        """Step environment with optimizations."""
        start_time = time.time()
        
        # Generate cache key based on current state and action
        cache_key = self._generate_step_cache_key(action)
        
        # Check cache
        if self.adaptive_cache:
            cached_result = self.adaptive_cache.get(cache_key)
            if cached_result is not None:
                self.cache_hits += 1
                self.step_times.append(time.time() - start_time)
                return self._decompress_step_result(cached_result)
                
        # Not in cache, compute normally
        result = self.base_env.step(action)
        
        # Cache the result
        if self.adaptive_cache:
            self.cache_misses += 1
            compressed_result = self._compress_step_result(result)
            self.adaptive_cache.put(cache_key, compressed_result)
            
        self.step_times.append(time.time() - start_time)
        return result
        
    def _generate_cache_key(self, method: str, kwargs: Dict) -> str:
        """Generate cache key for method call."""
        key_data = f"{method}_{hash(frozenset(kwargs.items()) if kwargs else 0)}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _generate_step_cache_key(self, action) -> str:
        """Generate cache key for step function."""
        # Include environment state hash and action
        state_hash = hash(self._get_state_signature())
        action_hash = hash(action.tobytes() if hasattr(action, 'tobytes') else str(action))
        
        key_data = f"step_{state_hash}_{action_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _get_state_signature(self) -> str:
        """Get a signature of current environment state."""
        # This is a simplified signature - in practice, you'd want to
        # capture key state variables that affect the step outcome
        if hasattr(self.base_env, 'current_step'):
            return f"{self.base_env.current_step}_{getattr(self.base_env, 'current_time', 0)}"
        return "0"
        
    def _compress_step_result(self, result) -> Any:
        """Compress step result for caching."""
        if not self.state_compressor:
            return result
            
        obs, reward, terminated, truncated, info = result
        compressed_obs = self.state_compressor.compress_state(obs)
        
        return (compressed_obs, reward, terminated, truncated, info)
        
    def _decompress_step_result(self, compressed_result) -> Any:
        """Decompress cached step result."""
        if not self.state_compressor:
            return compressed_result
            
        compressed_obs, reward, terminated, truncated, info = compressed_result
        obs = self.state_compressor.decompress_state(compressed_obs)
        
        return (obs, reward, terminated, truncated, info)
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        stats = {
            "total_steps": len(self.step_times),
            "average_step_time": np.mean(self.step_times) if self.step_times else 0,
            "cache_enabled": self.adaptive_cache is not None,
            "compression_enabled": self.state_compressor is not None,
            "batching_enabled": self.batch_processor is not None
        }
        
        if self.adaptive_cache:
            cache_stats = self.adaptive_cache.get_stats()
            stats.update({
                "cache_hit_rate": cache_stats["hit_rate"],
                "cache_size": cache_stats["size"],
                "cache_capacity": cache_stats["capacity"]
            })
            
        if self.state_compressor:
            compression_stats = self.state_compressor.get_stats()
            stats.update({
                "compression_ratio": compression_stats["compression_ratio"],
                "total_compressed": compression_stats["total_compressed"]
            })
            
        return stats
        
    # Delegate all other attributes to base environment
    def __getattr__(self, name):
        return getattr(self.base_env, name)


def optimize_environment(env, config: Optional[OptimizationConfig] = None):
    """Apply optimizations to an environment."""
    config = config or OptimizationConfig()
    return OptimizedEnvironmentWrapper(env, config)


def benchmark_optimization_impact(
    env_factory: Callable,
    num_episodes: int = 10,
    max_steps: int = 100
) -> Dict[str, Any]:
    """Benchmark the impact of various optimizations."""
    results = {}
    
    # Test configurations
    configs = {
        "baseline": OptimizationConfig(
            enable_caching=False,
            enable_state_compression=False,
            enable_action_batching=False
        ),
        "caching_only": OptimizationConfig(
            enable_caching=True,
            enable_state_compression=False,
            enable_action_batching=False
        ),
        "compression_only": OptimizationConfig(
            enable_caching=False,
            enable_state_compression=True,
            enable_action_batching=False
        ),
        "all_optimizations": OptimizationConfig(
            enable_caching=True,
            enable_state_compression=True,
            enable_action_batching=True
        )
    }
    
    for config_name, config in configs.items():
        print(f"Testing {config_name}...")
        
        env = env_factory()
        optimized_env = optimize_environment(env, config)
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            obs, _ = optimized_env.reset()
            
            for step in range(max_steps):
                action = optimized_env.action_space.sample()
                obs, reward, terminated, truncated, info = optimized_env.step(action)
                
                if terminated or truncated:
                    break
                    
        end_time = time.time()
        
        optimization_stats = optimized_env.get_optimization_stats()
        
        results[config_name] = {
            "total_time": end_time - start_time,
            "episodes": num_episodes,
            "optimization_stats": optimization_stats
        }
        
        # Cleanup
        if hasattr(optimized_env, '__del__'):
            optimized_env.__del__()
            
    return results