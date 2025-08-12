"""Advanced optimization techniques for grid control and federated learning."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import hashlib

from .exceptions import GridEnvironmentError
from .monitoring import SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance."""
    convergence_rate: float
    computation_efficiency: float
    memory_efficiency: float
    communication_overhead: float
    cache_hit_rate: float
    parallelization_speedup: float


class AdaptiveLearningRateScheduler:
    """Advanced learning rate scheduler with multiple strategies."""
    
    def __init__(
        self,
        initial_lr: float = 1e-3,
        strategy: str = 'plateau',
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        warmup_steps: int = 0
    ):
        self.initial_lr = initial_lr
        self.strategy = strategy
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        
        self.current_lr = initial_lr
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = deque(maxlen=patience * 2)
        
    def step(self, loss: Optional[float] = None) -> float:
        """Update learning rate based on strategy."""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            self.current_lr = self.initial_lr * (self.step_count / self.warmup_steps)
        elif self.strategy == 'plateau' and loss is not None:
            self._plateau_step(loss)
        elif self.strategy == 'cosine':
            self._cosine_step()
        elif self.strategy == 'exponential':
            self._exponential_step()
        elif self.strategy == 'adaptive':
            self._adaptive_step(loss)
        
        return max(self.current_lr, self.min_lr)
    
    def _plateau_step(self, loss: float):
        """Reduce LR on plateau."""
        self.loss_history.append(loss)
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self.current_lr *= self.factor
            self.patience_counter = 0
            logger.info(f"Reduced learning rate to {self.current_lr:.6f}")
    
    def _cosine_step(self):
        """Cosine annealing schedule."""
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                         (1 + np.cos(np.pi * self.step_count / self.warmup_steps)) / 2
    
    def _exponential_step(self):
        """Exponential decay schedule."""
        self.current_lr = self.initial_lr * (self.factor ** (self.step_count / self.patience))
    
    def _adaptive_step(self, loss: Optional[float]):
        """Adaptive learning rate based on gradient information."""
        if loss is not None and len(self.loss_history) > 1:
            recent_improvement = np.mean(list(self.loss_history)[-5:]) - \
                               np.mean(list(self.loss_history)[-10:-5]) if len(self.loss_history) >= 10 else 0
            
            if recent_improvement > 0:  # Improving
                self.current_lr *= 1.01  # Slight increase
            elif recent_improvement < -1e-4:  # Stagnating
                self.current_lr *= 0.99  # Slight decrease


class AdvancedCacheManager:
    """High-performance caching system for grid computations."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        compression: bool = True,
        eviction_policy: str = 'lru'
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.compression = compression
        self.eviction_policy = eviction_policy
        
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.cache_lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with thread safety."""
        with self.cache_lock:
            if key in self.cache:
                entry_time, data = self.cache[key]
                
                # Check TTL
                if time.time() - entry_time > self.ttl_seconds:
                    del self.cache[key]
                    del self.access_times[key]
                    self.misses += 1
                    return None
                
                # Update access patterns
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                self.hits += 1
                
                return self._decompress(data) if self.compression else data
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with eviction if necessary."""
        with self.cache_lock:
            # Compress if enabled
            data = self._compress(value) if self.compression else value
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_one()
            
            # Store item
            self.cache[key] = (time.time(), data)
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
    
    def _evict_one(self):
        """Evict one item based on eviction policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == 'lru':
            # Least recently used
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        elif self.eviction_policy == 'lfu':
            # Least frequently used
            oldest_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        else:
            # FIFO
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
        
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        del self.access_counts[oldest_key]
        self.evictions += 1
    
    def _compress(self, data: Any) -> bytes:
        """Compress data for storage."""
        import pickle
        import gzip
        return gzip.compress(pickle.dumps(data))
    
    def _decompress(self, data: bytes) -> Any:
        """Decompress data from storage."""
        import pickle
        import gzip
        return pickle.loads(gzip.decompress(data))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }


class ParallelComputationEngine:
    """Advanced parallel computation engine for grid operations."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        chunk_size: Optional[int] = None
    ):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.chunk_size = chunk_size or max(1, 1000 // self.max_workers)
        
        self.executor = None
        self.active_futures = []
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info(f"Initialized ParallelComputationEngine with {self.max_workers} workers")
    
    def __enter__(self):
        """Context manager entry."""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def parallel_map(
        self,
        func: Callable,
        data: List[Any],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Execute function in parallel over data chunks."""
        start_time = time.time()
        
        try:
            # Split data into chunks
            chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
            
            # Submit tasks
            futures = []
            for chunk in chunks:
                future = self.executor.submit(self._process_chunk, func, chunk)
                futures.append(future)
                self.active_futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    chunk_results = future.result(timeout=timeout)
                    results.extend(chunk_results)
                    self.performance_metrics['completed_tasks'] += 1
                except Exception as e:
                    logger.error(f"Parallel computation failed: {e}")
                    self.performance_metrics['failed_tasks'] += 1
                    # Continue with other chunks
                finally:
                    self.active_futures.remove(future)
            
            execution_time = time.time() - start_time
            self.performance_metrics['total_tasks'] += len(chunks)
            self.performance_metrics['avg_execution_time'] = \
                (self.performance_metrics['avg_execution_time'] + execution_time) / 2
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel map execution failed: {e}")
            raise GridEnvironmentError(f"Parallel computation error: {e}")
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a single chunk of data."""
        results = []
        for item in chunk:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                results.append(None)  # Or some default value
        return results
    
    async def async_parallel_map(
        self,
        async_func: Callable,
        data: List[Any],
        max_concurrent: int = 10
    ) -> List[Any]:
        """Asynchronous parallel execution."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_async_func(item):
            async with semaphore:
                return await async_func(item)
        
        tasks = [bounded_async_func(item) for item in data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Async task failed: {result}")
                valid_results.append(None)
            else:
                valid_results.append(result)
        
        return valid_results


class ModelCompressionEngine:
    """Advanced model compression for federated learning."""
    
    def __init__(
        self,
        compression_ratio: float = 0.1,
        quantization_bits: int = 8,
        sparsity_threshold: float = 1e-5
    ):
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
        self.sparsity_threshold = sparsity_threshold
        
    def compress_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        method: str = 'topk'
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Compress gradients using various methods."""
        compressed_gradients = {}
        compression_info = {}
        
        for name, grad in gradients.items():
            if method == 'topk':
                compressed_grad, info = self._topk_compression(grad)
            elif method == 'quantization':
                compressed_grad, info = self._quantization_compression(grad)
            elif method == 'sparsification':
                compressed_grad, info = self._sparsification_compression(grad)
            else:
                compressed_grad, info = grad, {'method': 'none'}
            
            compressed_gradients[name] = compressed_grad
            compression_info[name] = info
        
        return compressed_gradients, compression_info
    
    def _topk_compression(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Top-k sparsification compression."""
        flat_tensor = tensor.flatten()
        k = max(1, int(len(flat_tensor) * self.compression_ratio))
        
        # Get top-k values by magnitude
        _, top_indices = torch.topk(torch.abs(flat_tensor), k)
        
        # Create sparse representation
        compressed = torch.zeros_like(flat_tensor)
        compressed[top_indices] = flat_tensor[top_indices]
        
        compression_info = {
            'method': 'topk',
            'k': k,
            'sparsity': 1 - k / len(flat_tensor),
            'original_size': tensor.numel(),
            'compressed_size': k
        }
        
        return compressed.reshape(tensor.shape), compression_info
    
    def _quantization_compression(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Quantization compression."""
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Quantize to specified bits
        levels = 2 ** self.quantization_bits
        scale = (max_val - min_val) / (levels - 1)
        
        quantized = torch.round((tensor - min_val) / scale) * scale + min_val
        
        compression_info = {
            'method': 'quantization',
            'bits': self.quantization_bits,
            'scale': scale.item(),
            'min_val': min_val.item(),
            'max_val': max_val.item()
        }
        
        return quantized, compression_info
    
    def _sparsification_compression(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Threshold-based sparsification."""
        mask = torch.abs(tensor) > self.sparsity_threshold
        compressed = tensor * mask
        
        sparsity_ratio = (tensor.numel() - mask.sum().item()) / tensor.numel()
        
        compression_info = {
            'method': 'sparsification',
            'threshold': self.sparsity_threshold,
            'sparsity_ratio': sparsity_ratio,
            'remaining_elements': mask.sum().item()
        }
        
        return compressed, compression_info


class OptimizationOrchestrator:
    """Orchestrates all optimization techniques."""
    
    def __init__(
        self,
        enable_caching: bool = True,
        enable_parallel: bool = True,
        enable_compression: bool = True,
        cache_size: int = 1000,
        max_workers: int = None
    ):
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.enable_compression = enable_compression
        
        # Initialize components
        self.cache_manager = AdvancedCacheManager(max_size=cache_size) if enable_caching else None
        self.lr_scheduler = AdaptiveLearningRateScheduler()
        self.compression_engine = ModelCompressionEngine() if enable_compression else None
        
        # Performance tracking
        self.optimization_metrics = OptimizationMetrics(
            convergence_rate=0.0,
            computation_efficiency=1.0,
            memory_efficiency=1.0,
            communication_overhead=0.0,
            cache_hit_rate=0.0,
            parallelization_speedup=1.0
        )
        
        logger.info("OptimizationOrchestrator initialized with all components")
    
    def optimize_computation(
        self,
        func: Callable,
        data: Any,
        cache_key: Optional[str] = None
    ) -> Any:
        """Optimize a computation with caching and performance tracking."""
        start_time = time.time()
        
        # Try cache first
        if self.enable_caching and cache_key and self.cache_manager:
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute computation
        try:
            result = func(data)
            
            # Cache result
            if self.enable_caching and cache_key and self.cache_manager:
                self.cache_manager.put(cache_key, result)
            
            # Update metrics
            computation_time = time.time() - start_time
            self._update_performance_metrics(computation_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized computation failed: {e}")
            raise GridEnvironmentError(f"Computation optimization error: {e}")
    
    def _update_performance_metrics(self, computation_time: float):
        """Update performance metrics."""
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            self.optimization_metrics.cache_hit_rate = cache_stats['hit_rate']
        
        # Update computation efficiency (inverse of time)
        self.optimization_metrics.computation_efficiency = min(1.0, 1.0 / (computation_time + 1e-6))
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization performance report."""
        report = {
            'optimization_metrics': {
                'convergence_rate': self.optimization_metrics.convergence_rate,
                'computation_efficiency': self.optimization_metrics.computation_efficiency,
                'memory_efficiency': self.optimization_metrics.memory_efficiency,
                'cache_hit_rate': self.optimization_metrics.cache_hit_rate,
                'parallelization_speedup': self.optimization_metrics.parallelization_speedup
            }
        }
        
        if self.cache_manager:
            report['cache_stats'] = self.cache_manager.get_stats()
        
        return report