"""
Advanced scaling and performance optimization for Grid-Fed-RL-Gym.
Implements adaptive load balancing, distributed processing, and intelligent caching.
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import threading
import queue
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import pickle
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions."""
    timestamp: float
    cpu_utilization: float
    memory_usage: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    queue_depth: int
    active_workers: int
    error_rate: float


class IntelligentCache:
    """High-performance caching with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 10000, default_ttl: float = 3600.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._expiry_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU tracking."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Check expiry
                if current_time > self._expiry_times.get(key, 0):
                    self._remove_key(key)
                    self.misses += 1
                    return None
                
                # Update access time for LRU
                self._access_times[key] = current_time
                self.hits += 1
                return self._cache[key]['value']
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with optional TTL."""
        with self._lock:
            current_time = time.time()
            ttl = ttl or self.default_ttl
            
            # Evict expired entries if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = {'value': value, 'size': self._estimate_size(value)}
            self._access_times[key] = current_time
            self._expiry_times[key] = current_time + ttl
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all tracking structures."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._expiry_times.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self._access_times:
            return
            
        # Find oldest accessed key
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove_key(oldest_key)
        self.evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value."""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1000  # Default estimate
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._expiry_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        with self._lock:
            total_size = sum(item['size'] for item in self._cache.values())
            
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'total_memory_bytes': total_size
        }


class AdaptiveLoadBalancer:
    """Adaptive load balancer with performance-based routing."""
    
    def __init__(self, initial_workers: int = 4):
        self.workers: List[Dict[str, Any]] = []
        self.worker_stats: Dict[int, PerformanceMetrics] = {}
        self.request_queue = queue.Queue()
        self.result_futures: Dict[str, concurrent.futures.Future] = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.load_prediction_window = deque(maxlen=100)
        
        # Scaling parameters
        self.min_workers = 1
        self.max_workers = mp.cpu_count() * 2
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.scale_cooldown = 30.0  # seconds
        self.last_scale_time = 0.0
        
        # Initialize workers
        self._initialize_workers(initial_workers)
        
    def _initialize_workers(self, count: int) -> None:
        """Initialize worker pool."""
        self.executor = ThreadPoolExecutor(max_workers=count, thread_name_prefix="grid_worker")
        
        for i in range(count):
            worker_info = {
                'id': i,
                'active_tasks': 0,
                'completed_tasks': 0,
                'avg_latency': 0.0,
                'error_count': 0,
                'last_activity': time.time()
            }
            self.workers.append(worker_info)
        
        logger.info(f"Initialized {count} workers for adaptive load balancing")
    
    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task with intelligent worker selection."""
        # Select best worker based on current load
        worker_id = self._select_optimal_worker()
        
        # Update worker stats
        if worker_id < len(self.workers):
            self.workers[worker_id]['active_tasks'] += 1
            self.workers[worker_id]['last_activity'] = time.time()
        
        # Submit to executor
        future = self.executor.submit(self._execute_with_monitoring, func, worker_id, *args, **kwargs)
        
        # Store request ID for tracking
        request_id = f"{id(future)}_{time.time()}"
        self.result_futures[request_id] = future
        
        return future
    
    def _select_optimal_worker(self) -> int:
        """Select optimal worker based on current load and performance."""
        if not self.workers:
            return 0
        
        # Calculate worker scores (lower is better)
        worker_scores = []
        
        for worker in self.workers:
            # Base score from active tasks
            score = worker['active_tasks'] * 10
            
            # Penalty for high error rate
            error_rate = worker['error_count'] / max(worker['completed_tasks'], 1)
            score += error_rate * 50
            
            # Penalty for high latency
            score += worker['avg_latency'] * 100
            
            # Bonus for recent activity (worker warmth)
            time_since_activity = time.time() - worker['last_activity']
            if time_since_activity > 60:  # Cold worker
                score += 20
            
            worker_scores.append(score)
        
        # Return worker with lowest score
        return worker_scores.index(min(worker_scores))
    
    def _execute_with_monitoring(self, func: Callable, worker_id: int, *args, **kwargs) -> Any:
        """Execute function with performance monitoring."""
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            
            # Update success metrics
            execution_time = time.time() - start_time
            self._update_worker_stats(worker_id, execution_time, success=True)
            
            return result
            
        except Exception as e:
            # Update error metrics
            execution_time = time.time() - start_time
            self._update_worker_stats(worker_id, execution_time, success=False)
            raise
        
        finally:
            # Decrement active task count
            if worker_id < len(self.workers):
                self.workers[worker_id]['active_tasks'] = max(0, self.workers[worker_id]['active_tasks'] - 1)
    
    def _update_worker_stats(self, worker_id: int, execution_time: float, success: bool) -> None:
        """Update worker performance statistics."""
        if worker_id >= len(self.workers):
            return
        
        worker = self.workers[worker_id]
        
        # Update completion count
        worker['completed_tasks'] += 1
        
        # Update error count
        if not success:
            worker['error_count'] += 1
        
        # Update average latency (exponential moving average)
        alpha = 0.1  # Smoothing factor
        worker['avg_latency'] = (1 - alpha) * worker['avg_latency'] + alpha * execution_time
        
        # Record system-wide metrics
        self._record_performance_metrics()
    
    def _record_performance_metrics(self) -> None:
        """Record system-wide performance metrics."""
        current_time = time.time()
        
        # Calculate aggregated metrics
        total_active = sum(w['active_tasks'] for w in self.workers)
        total_completed = sum(w['completed_tasks'] for w in self.workers)
        total_errors = sum(w['error_count'] for w in self.workers)
        avg_latency = np.mean([w['avg_latency'] for w in self.workers if w['avg_latency'] > 0])
        
        error_rate = total_errors / max(total_completed, 1)
        
        metrics = PerformanceMetrics(
            timestamp=current_time,
            cpu_utilization=0.0,  # Would be filled by system monitor
            memory_usage=0.0,     # Would be filled by system monitor
            throughput=total_completed / max(current_time - getattr(self, 'start_time', current_time), 1),
            latency_p50=avg_latency,
            latency_p95=avg_latency * 1.5,  # Approximation
            latency_p99=avg_latency * 2.0,  # Approximation
            queue_depth=total_active,
            active_workers=len(self.workers),
            error_rate=error_rate
        )
        
        self.performance_history.append(metrics)
        
        # Check for scaling needs
        self._evaluate_scaling_needs(metrics)
    
    def _evaluate_scaling_needs(self, metrics: PerformanceMetrics) -> None:
        """Evaluate if scaling up or down is needed."""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        # Calculate current utilization
        utilization = metrics.queue_depth / max(metrics.active_workers, 1)
        
        # Scale up if high utilization and low error rate
        if (utilization > self.scale_up_threshold and 
            metrics.active_workers < self.max_workers and
            metrics.error_rate < 0.1):
            self._scale_up()
            
        # Scale down if low utilization
        elif (utilization < self.scale_down_threshold and 
              metrics.active_workers > self.min_workers):
            self._scale_down()
    
    def _scale_up(self) -> None:
        """Add a worker to the pool."""
        new_worker_id = len(self.workers)
        
        worker_info = {
            'id': new_worker_id,
            'active_tasks': 0,
            'completed_tasks': 0,
            'avg_latency': 0.0,
            'error_count': 0,
            'last_activity': time.time()
        }
        self.workers.append(worker_info)
        
        # Recreate executor with more workers
        old_executor = self.executor
        self.executor = ThreadPoolExecutor(max_workers=len(self.workers), thread_name_prefix="grid_worker")
        
        # Shutdown old executor gracefully
        old_executor.shutdown(wait=False)
        
        self.last_scale_time = time.time()
        logger.info(f"Scaled up to {len(self.workers)} workers")
    
    def _scale_down(self) -> None:
        """Remove a worker from the pool."""
        if len(self.workers) <= self.min_workers:
            return
        
        # Remove least active worker
        least_active_idx = min(
            range(len(self.workers)), 
            key=lambda i: self.workers[i]['active_tasks']
        )
        
        self.workers.pop(least_active_idx)
        
        # Recreate executor with fewer workers
        old_executor = self.executor
        self.executor = ThreadPoolExecutor(max_workers=len(self.workers), thread_name_prefix="grid_worker")
        
        # Shutdown old executor gracefully
        old_executor.shutdown(wait=False)
        
        self.last_scale_time = time.time()
        logger.info(f"Scaled down to {len(self.workers)} workers")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.performance_history:
            return {}
        
        recent_metrics = list(self.performance_history)[-10:]
        
        return {
            'current_workers': len(self.workers),
            'total_completed_tasks': sum(w['completed_tasks'] for w in self.workers),
            'total_errors': sum(w['error_count'] for w in self.workers),
            'average_latency': np.mean([m.latency_p50 for m in recent_metrics]),
            'current_queue_depth': recent_metrics[-1].queue_depth if recent_metrics else 0,
            'throughput': recent_metrics[-1].throughput if recent_metrics else 0,
            'error_rate': recent_metrics[-1].error_rate if recent_metrics else 0,
            'worker_details': self.workers.copy()
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the load balancer."""
        self.executor.shutdown(wait=True)
        logger.info("Load balancer shutdown complete")


class DistributedTaskManager:
    """Distributed task management with fault tolerance."""
    
    def __init__(self, node_id: str = "main"):
        self.node_id = node_id
        self.load_balancer = AdaptiveLoadBalancer()
        self.cache = IntelligentCache()
        
        # Task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.failed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Performance optimization
        self.task_dependency_graph: Dict[str, List[str]] = {}
        self.execution_pipeline = queue.Queue()
        
        # Fault tolerance
        self.heartbeat_interval = 30.0
        self.task_timeout = 300.0  # 5 minutes
        self.max_retries = 3
        
        self._start_heartbeat()
    
    def submit_computation(
        self, 
        task_id: str,
        func: Callable,
        *args,
        dependencies: Optional[List[str]] = None,
        cache_key: Optional[str] = None,
        priority: int = 0,
        **kwargs
    ) -> str:
        """Submit computation task with dependency tracking."""
        
        # Check cache first
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for task {task_id}")
                return cached_result
        
        # Create task metadata
        task_info = {
            'id': task_id,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'dependencies': dependencies or [],
            'cache_key': cache_key,
            'priority': priority,
            'submitted_at': time.time(),
            'status': 'pending',
            'retries': 0,
            'node_id': self.node_id
        }
        
        self.active_tasks[task_id] = task_info
        
        # Add to dependency graph
        if dependencies:
            self.task_dependency_graph[task_id] = dependencies
        
        # Check if ready to execute
        if self._are_dependencies_satisfied(task_id):
            self._schedule_task(task_id)
        
        return task_id
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all task dependencies are completed."""
        dependencies = self.task_dependency_graph.get(task_id, [])
        
        for dep_id in dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    def _schedule_task(self, task_id: str) -> None:
        """Schedule task for execution."""
        if task_id not in self.active_tasks:
            return
        
        task_info = self.active_tasks[task_id]
        task_info['status'] = 'scheduled'
        task_info['scheduled_at'] = time.time()
        
        # Submit to load balancer
        future = self.load_balancer.submit_task(
            self._execute_task_with_monitoring,
            task_id
        )
        
        task_info['future'] = future
        
        # Set up completion callback
        future.add_done_callback(lambda f: self._handle_task_completion(task_id, f))
    
    def _execute_task_with_monitoring(self, task_id: str) -> Any:
        """Execute task with comprehensive monitoring."""
        task_info = self.active_tasks.get(task_id)
        if not task_info:
            raise ValueError(f"Task {task_id} not found")
        
        start_time = time.time()
        task_info['status'] = 'running'
        task_info['started_at'] = start_time
        
        try:
            # Execute the actual function
            result = task_info['func'](*task_info['args'], **task_info['kwargs'])
            
            # Cache result if cache key provided
            if task_info['cache_key']:
                self.cache.set(task_info['cache_key'], result)
            
            execution_time = time.time() - start_time
            task_info['execution_time'] = execution_time
            task_info['status'] = 'completed'
            task_info['result'] = result
            
            logger.debug(f"Task {task_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            task_info['execution_time'] = execution_time
            task_info['status'] = 'failed'
            task_info['error'] = str(e)
            task_info['retries'] = task_info.get('retries', 0) + 1
            
            logger.error(f"Task {task_id} failed after {execution_time:.2f}s: {e}")
            
            # Retry if under limit
            if task_info['retries'] < self.max_retries:
                logger.info(f"Retrying task {task_id} (attempt {task_info['retries'] + 1})")
                return self._execute_task_with_monitoring(task_id)
            
            raise
    
    def _handle_task_completion(self, task_id: str, future: concurrent.futures.Future) -> None:
        """Handle task completion and schedule dependent tasks."""
        task_info = self.active_tasks.get(task_id)
        if not task_info:
            return
        
        try:
            result = future.result()
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task_info
            self.active_tasks.pop(task_id, None)
            
            # Schedule dependent tasks
            for dependent_task_id, dependencies in self.task_dependency_graph.items():
                if task_id in dependencies and self._are_dependencies_satisfied(dependent_task_id):
                    self._schedule_task(dependent_task_id)
            
        except Exception as e:
            # Move to failed tasks
            task_info['final_error'] = str(e)
            self.failed_tasks[task_id] = task_info
            self.active_tasks.pop(task_id, None)
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result with optional timeout."""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]['result']
        
        if task_id in self.failed_tasks:
            error_msg = self.failed_tasks[task_id].get('final_error', 'Unknown error')
            raise RuntimeError(f"Task {task_id} failed: {error_msg}")
        
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        # Wait for completion
        task_info = self.active_tasks[task_id]
        future = task_info.get('future')
        
        if future:
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Task {task_id} timed out after {timeout}s")
        
        raise RuntimeError(f"Task {task_id} not yet scheduled")
    
    def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring for fault detection."""
        def heartbeat_loop():
            while True:
                try:
                    self._cleanup_stale_tasks()
                    self._log_performance_metrics()
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
        
        heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_thread.start()
    
    def _cleanup_stale_tasks(self) -> None:
        """Clean up tasks that have been running too long."""
        current_time = time.time()
        stale_tasks = []
        
        for task_id, task_info in self.active_tasks.items():
            if task_info['status'] == 'running':
                started_at = task_info.get('started_at', current_time)
                if current_time - started_at > self.task_timeout:
                    stale_tasks.append(task_id)
        
        for task_id in stale_tasks:
            logger.warning(f"Cancelling stale task {task_id}")
            task_info = self.active_tasks[task_id]
            future = task_info.get('future')
            if future:
                future.cancel()
            
            task_info['status'] = 'timeout'
            self.failed_tasks[task_id] = task_info
            self.active_tasks.pop(task_id, None)
    
    def _log_performance_metrics(self) -> None:
        """Log current performance metrics."""
        stats = self.get_comprehensive_stats()
        logger.info(
            f"Node {self.node_id}: "
            f"Active={stats['active_tasks']} "
            f"Completed={stats['completed_tasks']} "
            f"Failed={stats['failed_tasks']} "
            f"Cache hit rate={stats['cache_hit_rate']:.2%}"
        )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        cache_stats = self.cache.get_stats()
        load_balancer_stats = self.load_balancer.get_performance_stats()
        
        return {
            'node_id': self.node_id,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'cache_hit_rate': cache_stats.get('hit_rate', 0),
            'cache_size': cache_stats.get('size', 0),
            'cache_memory_mb': cache_stats.get('total_memory_bytes', 0) / (1024 * 1024),
            'load_balancer': load_balancer_stats,
            'uptime_seconds': time.time() - getattr(self, 'start_time', time.time())
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the task manager."""
        logger.info(f"Shutting down distributed task manager {self.node_id}")
        
        # Wait for active tasks to complete
        for task_id, task_info in self.active_tasks.items():
            future = task_info.get('future')
            if future:
                try:
                    future.result(timeout=10.0)  # Wait up to 10 seconds
                except Exception:
                    future.cancel()
        
        self.load_balancer.shutdown()
        self.cache.clear()


# Global distributed task manager
global_task_manager = DistributedTaskManager()


def distributed_compute(
    task_id: str = None,
    cache_key: str = None,
    dependencies: List[str] = None,
    priority: int = 0
) -> Callable:
    """Decorator for distributed computation with automatic task management."""
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            # Generate task ID if not provided
            nonlocal task_id
            if task_id is None:
                func_name = func.__name__
                args_hash = hashlib.md5(str(args).encode()).hexdigest()[:8]
                task_id = f"{func_name}_{args_hash}_{time.time()}"
            
            # Generate cache key if not provided
            nonlocal cache_key
            if cache_key is None:
                cache_key = f"{func.__name__}_{hashlib.md5(str(args).encode()).hexdigest()}"
            
            # Submit to distributed task manager
            submitted_task_id = global_task_manager.submit_computation(
                task_id=task_id,
                func=func,
                *args,
                dependencies=dependencies,
                cache_key=cache_key,
                priority=priority,
                **kwargs
            )
            
            # Return result (will wait for completion)
            return global_task_manager.get_task_result(submitted_task_id)
        
        return wrapper
    return decorator


def parallel_map(
    func: Callable,
    items: List[Any],
    max_workers: Optional[int] = None,
    chunk_size: Optional[int] = None
) -> List[Any]:
    """Parallel map with automatic chunking and load balancing."""
    
    if not items:
        return []
    
    max_workers = max_workers or min(len(items), mp.cpu_count() * 2)
    chunk_size = chunk_size or max(1, len(items) // max_workers)
    
    # Create chunks
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    # Submit chunk processing tasks
    task_ids = []
    for i, chunk in enumerate(chunks):
        task_id = f"parallel_map_chunk_{i}_{time.time()}"
        global_task_manager.submit_computation(
            task_id=task_id,
            func=lambda chunk_data: [func(item) for item in chunk_data],
            chunk
        )
        task_ids.append(task_id)
    
    # Collect results
    results = []
    for task_id in task_ids:
        chunk_results = global_task_manager.get_task_result(task_id)
        results.extend(chunk_results)
    
    return results