"""Performance optimization utilities for high-scale grid simulation."""

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from functools import wraps, lru_cache
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""
    enable_vectorization: bool = True
    enable_caching: bool = True
    enable_parallel: bool = True
    max_workers: int = 4
    cache_size: int = 128
    batch_size: int = 32
    

class VectorizedOperations:
    """Vectorized operations for batch processing."""
    
    @staticmethod
    def batch_power_flow(
        bus_voltages_batch: np.ndarray,
        admittance_matrix: np.ndarray,
        power_injections_batch: np.ndarray
    ) -> np.ndarray:
        """Compute power flow for multiple scenarios simultaneously."""
        # Vectorized power flow calculation
        # Shape: (batch_size, n_buses)
        batch_size = bus_voltages_batch.shape[0]
        n_buses = admittance_matrix.shape[0]
        
        # Broadcast admittance matrix for batch operation
        Y_batch = np.broadcast_to(admittance_matrix, (batch_size, n_buses, n_buses))
        
        # Vectorized matrix multiplication: batch_matmul(Y, V)
        currents_batch = np.matmul(Y_batch, bus_voltages_batch[..., None]).squeeze(-1)
        
        # Power calculation: S = V * conj(I)
        powers_batch = bus_voltages_batch * np.conj(currents_batch)
        
        return powers_batch
    
    @staticmethod
    def batch_constraint_check(
        voltages_batch: np.ndarray,
        voltage_limits: tuple,
        frequencies_batch: np.ndarray,
        frequency_limits: tuple
    ) -> np.ndarray:
        """Check constraints for batch of system states."""
        batch_size = voltages_batch.shape[0]
        
        # Vectorized voltage constraint checking
        v_low, v_high = voltage_limits
        voltage_violations = np.logical_or(
            np.any(voltages_batch < v_low, axis=1),
            np.any(voltages_batch > v_high, axis=1)
        )
        
        # Vectorized frequency constraint checking
        f_low, f_high = frequency_limits
        frequency_violations = np.logical_or(
            frequencies_batch < f_low,
            frequencies_batch > f_high
        )
        
        # Combine violations
        violations = np.logical_or(voltage_violations, frequency_violations)
        
        return violations.astype(int)
    
    @staticmethod 
    def batch_reward_calculation(
        voltages_batch: np.ndarray,
        frequencies_batch: np.ndarray,
        losses_batch: np.ndarray,
        loadings_batch: np.ndarray,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """Calculate rewards for batch of states."""
        if weights is None:
            weights = {'voltage': 10.0, 'frequency': 20.0, 'losses': 0.1, 'loading': 50.0}
        
        batch_size = voltages_batch.shape[0]
        rewards = np.zeros(batch_size)
        
        # Vectorized voltage deviation penalty
        voltage_deviations = np.sum(np.abs(voltages_batch - 1.0), axis=1)
        rewards -= weights['voltage'] * voltage_deviations
        
        # Frequency deviation penalty
        freq_deviations = np.abs(frequencies_batch - 60.0)
        rewards -= weights['frequency'] * freq_deviations
        
        # Losses penalty
        rewards -= weights['losses'] * losses_batch
        
        # Line loading penalty (for overloads > 0.8)
        overload_mask = loadings_batch > 0.8
        overload_penalties = np.sum(overload_mask * (loadings_batch - 0.8), axis=1)
        rewards -= weights['loading'] * overload_penalties
        
        return rewards


class ParallelProcessor:
    """Parallel processing for multi-environment simulation."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def parallel_environment_step(
        self,
        environments: List,
        actions: List[np.ndarray],
        timeout: float = 10.0
    ) -> List[tuple]:
        """Execute environment steps in parallel."""
        
        futures = []
        for env, action in zip(environments, actions):
            future = self.executor.submit(env.step, action)
            futures.append(future)
        
        results = []
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel step failed: {e}")
                # Return safe fallback result
                results.append((np.zeros(1), -1000.0, True, False, {'error': str(e)}))
        
        return results
    
    def parallel_power_flow_solve(
        self,
        scenarios: List[Dict[str, Any]],
        solver_func: Callable,
        timeout: float = 5.0
    ) -> List[Any]:
        """Solve multiple power flow scenarios in parallel."""
        
        futures = []
        for scenario in scenarios:
            future = self.executor.submit(solver_func, **scenario)
            futures.append(future)
        
        results = []
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel power flow failed: {e}")
                results.append(None)
        
        return results
    
    def close(self):
        """Clean up thread pool."""
        self.executor.shutdown(wait=True)


class AdaptiveCache:
    """Adaptive caching system that adjusts based on hit rates."""
    
    def __init__(self, initial_size: int = 128, min_size: int = 32, max_size: int = 1024):
        self.cache = {}
        self.access_counts = {}
        self.hit_counts = {}
        self.miss_counts = {}
        
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        
        self.total_hits = 0
        self.total_misses = 0
        
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with adaptive sizing."""
        with self._lock:
            if key in self.cache:
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
                self.total_hits += 1
                return self.cache[key]
            else:
                self.miss_counts[key] = self.miss_counts.get(key, 0) + 1
                self.total_misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with automatic cleanup."""
        with self._lock:
            # If cache is full, remove least accessed items
            if len(self.cache) >= self.current_size:
                self._cleanup_cache()
            
            self.cache[key] = value
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            # Adapt cache size based on hit rate
            self._adapt_cache_size()
    
    def _cleanup_cache(self) -> None:
        """Remove least frequently accessed items."""
        # Sort by access count (ascending) and remove bottom 25%
        sorted_items = sorted(self.access_counts.items(), key=lambda x: x[1])
        items_to_remove = max(1, len(sorted_items) // 4)
        
        for key, _ in sorted_items[:items_to_remove]:
            if key in self.cache:
                del self.cache[key]
                del self.access_counts[key]
                if key in self.hit_counts:
                    del self.hit_counts[key]
                if key in self.miss_counts:
                    del self.miss_counts[key]
    
    def _adapt_cache_size(self) -> None:
        """Adapt cache size based on hit rate."""
        total_requests = self.total_hits + self.total_misses
        
        if total_requests > 100:  # Need sufficient data
            hit_rate = self.total_hits / total_requests
            
            if hit_rate > 0.8 and self.current_size < self.max_size:
                # High hit rate - increase cache size
                self.current_size = min(self.max_size, int(self.current_size * 1.2))
                logger.debug(f"Increased cache size to {self.current_size} (hit rate: {hit_rate:.2%})")
            elif hit_rate < 0.5 and self.current_size > self.min_size:
                # Low hit rate - decrease cache size
                self.current_size = max(self.min_size, int(self.current_size * 0.8))
                logger.debug(f"Decreased cache size to {self.current_size} (hit rate: {hit_rate:.2%})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.total_hits + self.total_misses
        hit_rate = self.total_hits / total if total > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'current_size': self.current_size,
            'items_cached': len(self.cache),
            'most_accessed': max(self.access_counts.items(), key=lambda x: x[1]) if self.access_counts else None
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.vectorized_ops = VectorizedOperations()
        self.parallel_processor = ParallelProcessor(self.config.max_workers) if self.config.enable_parallel else None
        self.adaptive_cache = AdaptiveCache(self.config.cache_size) if self.config.enable_caching else None
        
        # Performance monitoring
        self.optimization_stats = {
            'vectorized_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_tasks': 0,
            'total_speedup': 1.0
        }
        
    def optimize_environment_batch(
        self,
        environments: List,
        actions: List[np.ndarray]
    ) -> List[tuple]:
        """Optimize batch environment processing."""
        
        if self.config.enable_parallel and len(environments) > 1:
            # Use parallel processing
            results = self.parallel_processor.parallel_environment_step(environments, actions)
            self.optimization_stats['parallel_tasks'] += len(environments)
            return results
        else:
            # Sequential processing
            results = []
            for env, action in zip(environments, actions):
                result = env.step(action)
                results.append(result)
            return results
    
    def optimize_power_flow_batch(
        self,
        power_flow_scenarios: List[Dict[str, Any]],
        solver_function: Callable
    ) -> List[Any]:
        """Optimize batch power flow solving."""
        
        if self.config.enable_caching:
            # Check cache first
            cached_results = []
            uncached_scenarios = []
            uncached_indices = []
            
            for i, scenario in enumerate(power_flow_scenarios):
                cache_key = self._generate_cache_key(scenario)
                cached_result = self.adaptive_cache.get(cache_key)
                
                if cached_result is not None:
                    cached_results.append((i, cached_result))
                    self.optimization_stats['cache_hits'] += 1
                else:
                    uncached_scenarios.append(scenario)
                    uncached_indices.append(i)
                    self.optimization_stats['cache_misses'] += 1
            
            # Solve uncached scenarios
            if uncached_scenarios:
                if self.config.enable_parallel and len(uncached_scenarios) > 1:
                    uncached_results = self.parallel_processor.parallel_power_flow_solve(
                        uncached_scenarios, solver_function
                    )
                    self.optimization_stats['parallel_tasks'] += len(uncached_scenarios)
                else:
                    uncached_results = [solver_function(**scenario) for scenario in uncached_scenarios]
                
                # Cache results
                for scenario, result in zip(uncached_scenarios, uncached_results):
                    if result is not None:  # Only cache successful results
                        cache_key = self._generate_cache_key(scenario)
                        self.adaptive_cache.put(cache_key, result)
            else:
                uncached_results = []
            
            # Combine results in original order
            all_results = [None] * len(power_flow_scenarios)
            
            # Fill in cached results
            for i, result in cached_results:
                all_results[i] = result
            
            # Fill in newly computed results
            for i, result in zip(uncached_indices, uncached_results):
                all_results[i] = result
            
            return all_results
        
        else:
            # No caching - direct computation
            if self.config.enable_parallel and len(power_flow_scenarios) > 1:
                return self.parallel_processor.parallel_power_flow_solve(
                    power_flow_scenarios, solver_function
                )
            else:
                return [solver_function(**scenario) for scenario in power_flow_scenarios]
    
    def optimize_constraint_checking(
        self,
        states_batch: List[Dict[str, Any]],
        voltage_limits: tuple,
        frequency_limits: tuple
    ) -> np.ndarray:
        """Optimize constraint checking for multiple states."""
        
        if self.config.enable_vectorization and len(states_batch) > 1:
            # Vectorized constraint checking
            voltages_batch = np.array([state.get('bus_voltages', [1.0]) for state in states_batch])
            frequencies_batch = np.array([state.get('frequency', 60.0) for state in states_batch])
            
            violations = self.vectorized_ops.batch_constraint_check(
                voltages_batch, voltage_limits, frequencies_batch, frequency_limits
            )
            
            self.optimization_stats['vectorized_operations'] += len(states_batch)
            return violations
        else:
            # Sequential constraint checking
            violations = []
            for state in states_batch:
                voltages = np.array(state.get('bus_voltages', [1.0]))
                frequency = state.get('frequency', 60.0)
                
                v_violation = np.any(voltages < voltage_limits[0]) or np.any(voltages > voltage_limits[1])
                f_violation = frequency < frequency_limits[0] or frequency > frequency_limits[1]
                
                violations.append(int(v_violation or f_violation))
            
            return np.array(violations)
    
    def _generate_cache_key(self, scenario: Dict[str, Any]) -> str:
        """Generate cache key for power flow scenario."""
        # Simple hash based on key parameters
        key_parts = []
        
        for key in ['loads', 'generation', 'bus_count', 'line_count']:
            if key in scenario:
                if isinstance(scenario[key], dict):
                    key_parts.append(f"{key}:{hash(tuple(sorted(scenario[key].items())))}")
                elif isinstance(scenario[key], (list, tuple)):
                    key_parts.append(f"{key}:{hash(tuple(scenario[key]))}")
                else:
                    key_parts.append(f"{key}:{scenario[key]}")
        
        return "|".join(key_parts)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get performance optimization statistics."""
        stats = self.optimization_stats.copy()
        
        if self.adaptive_cache:
            cache_stats = self.adaptive_cache.get_stats()
            stats.update({f"cache_{k}": v for k, v in cache_stats.items()})
        
        # Calculate estimated speedup
        total_ops = (
            stats['vectorized_operations'] +
            stats['parallel_tasks'] +
            max(1, stats['cache_hits'])
        )
        
        if total_ops > 0:
            # Rough speedup estimation
            vectorization_speedup = min(4.0, stats['vectorized_operations'] / max(1, total_ops)) * 3.0
            parallelization_speedup = min(self.config.max_workers, stats['parallel_tasks'] / max(1, total_ops)) * 2.0
            cache_speedup = (stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses'])) * 10.0
            
            stats['estimated_speedup'] = 1.0 + vectorization_speedup + parallelization_speedup + cache_speedup
        
        return stats
    
    def close(self):
        """Clean up resources."""
        if self.parallel_processor:
            self.parallel_processor.close()


# Global optimizer instance
global_optimizer = PerformanceOptimizer()