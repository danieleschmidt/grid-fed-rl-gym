"""Advanced scaling and optimization for Grid-Fed-RL-Gym Generation 3."""

import asyncio
import multiprocessing as mp
import threading
import concurrent.futures
import time
import numpy as np
import logging
import psutil
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref
import gc
import joblib
from pathlib import Path
import pickle
import redis
import zmq

from .exceptions import GridEnvironmentError, ResourceExhaustionError
from .monitoring import GridMonitor, SystemMetrics
from .advanced_robustness import DistributedTracer

logger = logging.getLogger(__name__)


@dataclass
class WorkerMetrics:
    """Performance metrics for worker processes."""
    worker_id: str
    cpu_usage: float
    memory_usage_mb: float
    tasks_completed: int
    tasks_failed: int
    average_task_time: float
    last_activity: datetime
    status: str  # active, idle, busy, error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "last_activity": self.last_activity.isoformat()
        }


class AdaptiveLoadBalancer:
    """Intelligent load balancing with performance optimization."""
    
    def __init__(
        self,
        min_workers: int = 2,
        max_workers: int = None,
        scaling_factor: float = 1.5,
        load_threshold: float = 0.8
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers or mp.cpu_count() * 2
        self.scaling_factor = scaling_factor
        self.load_threshold = load_threshold
        
        # Worker pool management
        self.worker_pool: Dict[str, mp.Process] = {}
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Load balancing state
        self.current_load = 0.0
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = True
        self.last_scaling_time = 0
        self.scaling_cooldown = 30  # seconds
        
        self._initialize_workers()
        
    def _initialize_workers(self) -> None:
        """Initialize the minimum number of workers."""
        for i in range(self.min_workers):
            worker_id = f"worker_{i}"
            self._start_worker(worker_id)
    
    def _start_worker(self, worker_id: str) -> None:
        """Start a new worker process."""
        try:
            worker_process = mp.Process(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=True
            )
            worker_process.start()
            
            self.worker_pool[worker_id] = worker_process
            self.worker_metrics[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                cpu_usage=0.0,
                memory_usage_mb=0.0,
                tasks_completed=0,
                tasks_failed=0,
                average_task_time=0.0,
                last_activity=datetime.now(),
                status="active"
            )
            
            logger.info(f"Started worker: {worker_id}")
            
        except Exception as e:
            logger.error(f"Failed to start worker {worker_id}: {e}")
    
    def _worker_loop(self, worker_id: str) -> None:
        """Main loop for worker processes."""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get task from queue with timeout
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Update worker status
                self._update_worker_status(worker_id, "busy")
                
                # Execute task
                start_time = time.time()
                try:
                    result = self._execute_task(task)
                    execution_time = time.time() - start_time
                    
                    # Send result
                    self.result_queue.put({
                        "task_id": task["task_id"],
                        "result": result,
                        "execution_time": execution_time,
                        "worker_id": worker_id,
                        "status": "completed"
                    })
                    
                    self._update_worker_metrics(worker_id, execution_time, success=True)
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    self.result_queue.put({
                        "task_id": task["task_id"],
                        "error": str(e),
                        "execution_time": execution_time,
                        "worker_id": worker_id,
                        "status": "failed"
                    })
                    
                    self._update_worker_metrics(worker_id, execution_time, success=False)
                
                # Mark task as done
                self.task_queue.task_done()
                self._update_worker_status(worker_id, "active")
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self._update_worker_status(worker_id, "error")
                time.sleep(1)
    
    def _execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task."""
        task_type = task.get("type")
        task_data = task.get("data", {})
        
        if task_type == "power_flow":
            return self._execute_power_flow(task_data)
        elif task_type == "optimization":
            return self._execute_optimization(task_data)
        elif task_type == "simulation":
            return self._execute_simulation(task_data)
        else:
            raise GridEnvironmentError(f"Unknown task type: {task_type}")
    
    def _execute_power_flow(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute power flow calculation."""
        # Simulate power flow computation
        bus_count = data.get("bus_count", 13)
        complexity = data.get("complexity", 1.0)
        
        # Simulate computational load
        compute_time = 0.01 * complexity * (1 + np.random.exponential(0.5))
        time.sleep(compute_time)
        
        return {
            "bus_voltages": 0.95 + 0.1 * np.random.random(bus_count),
            "line_flows": 0.3 + 0.4 * np.random.random(bus_count - 1),
            "converged": np.random.random() > 0.05,  # 5% failure rate
            "iterations": np.random.randint(3, 15),
            "computation_time": compute_time
        }
    
    def _execute_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization task."""
        # Simulate optimization
        variables = data.get("variables", 10)
        iterations = data.get("iterations", 100)
        
        compute_time = 0.001 * variables * iterations
        time.sleep(compute_time)
        
        return {
            "optimal_solution": np.random.random(variables),
            "objective_value": np.random.exponential(10),
            "converged": True,
            "iterations_used": iterations,
            "computation_time": compute_time
        }
    
    def _execute_simulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simulation task."""
        # Simulate environment step
        steps = data.get("steps", 1)
        complexity = data.get("complexity", 1.0)
        
        compute_time = 0.005 * steps * complexity
        time.sleep(compute_time)
        
        return {
            "states": [np.random.random(10) for _ in range(steps)],
            "rewards": np.random.normal(0, 1, steps),
            "done": np.random.random() < 0.01,  # 1% episode end
            "computation_time": compute_time
        }
    
    def submit_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Submit a task for execution."""
        task_id = f"task_{time.time()}_{np.random.randint(1000)}"
        
        task = {
            "task_id": task_id,
            "type": task_type,
            "data": task_data,
            "priority": priority,
            "submitted_time": time.time()
        }
        
        self.task_queue.put(task)
        self.pending_tasks += 1
        
        # Check if we need to scale
        if self.auto_scaling_enabled:
            self._check_auto_scaling()
        
        return task_id
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get a completed task result."""
        try:
            result = self.result_queue.get(timeout=timeout)
            
            if result["status"] == "completed":
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1
            
            self.pending_tasks = max(0, self.pending_tasks - 1)
            return result
            
        except queue.Empty:
            return None
    
    def _check_auto_scaling(self) -> None:
        """Check if we need to scale workers up or down."""
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_scaling_time < self.scaling_cooldown:
            return
        
        # Calculate current load
        active_workers = len([w for w in self.worker_metrics.values() if w.status in ["active", "busy"]])
        queue_size = self.task_queue.qsize()
        
        if active_workers > 0:
            load_ratio = queue_size / active_workers
        else:
            load_ratio = float('inf')
        
        # Scale up if overloaded
        if (load_ratio > self.load_threshold and 
            len(self.worker_pool) < self.max_workers):
            
            new_workers = min(
                int(len(self.worker_pool) * self.scaling_factor) - len(self.worker_pool),
                self.max_workers - len(self.worker_pool)
            )
            
            for i in range(new_workers):
                worker_id = f"worker_{len(self.worker_pool)}_{current_time}"
                self._start_worker(worker_id)
            
            self.last_scaling_time = current_time
            self.scaling_history.append({
                "time": current_time,
                "action": "scale_up",
                "workers_added": new_workers,
                "total_workers": len(self.worker_pool),
                "load_ratio": load_ratio
            })
            
            logger.info(f"Scaled up: added {new_workers} workers (total: {len(self.worker_pool)})")
        
        # Scale down if underutilized
        elif (load_ratio < self.load_threshold * 0.3 and 
              len(self.worker_pool) > self.min_workers and
              queue_size < self.min_workers):
            
            workers_to_remove = min(
                len(self.worker_pool) - int(len(self.worker_pool) / self.scaling_factor),
                len(self.worker_pool) - self.min_workers
            )
            
            # Remove least active workers
            worker_activity = [(wid, metrics.last_activity) for wid, metrics in self.worker_metrics.items()]
            worker_activity.sort(key=lambda x: x[1])
            
            for i in range(workers_to_remove):
                worker_id = worker_activity[i][0]
                self._stop_worker(worker_id)
            
            self.last_scaling_time = current_time
            self.scaling_history.append({
                "time": current_time,
                "action": "scale_down",
                "workers_removed": workers_to_remove,
                "total_workers": len(self.worker_pool),
                "load_ratio": load_ratio
            })
            
            logger.info(f"Scaled down: removed {workers_to_remove} workers (total: {len(self.worker_pool)})")
    
    def _stop_worker(self, worker_id: str) -> None:
        """Stop a worker process."""
        if worker_id in self.worker_pool:
            try:
                self.worker_pool[worker_id].terminate()
                self.worker_pool[worker_id].join(timeout=5.0)
                del self.worker_pool[worker_id]
                del self.worker_metrics[worker_id]
                logger.info(f"Stopped worker: {worker_id}")
            except Exception as e:
                logger.error(f"Failed to stop worker {worker_id}: {e}")
    
    def _update_worker_status(self, worker_id: str, status: str) -> None:
        """Update worker status."""
        if worker_id in self.worker_metrics:
            self.worker_metrics[worker_id].status = status
            self.worker_metrics[worker_id].last_activity = datetime.now()
    
    def _update_worker_metrics(self, worker_id: str, execution_time: float, success: bool) -> None:
        """Update worker performance metrics."""
        if worker_id in self.worker_metrics:
            metrics = self.worker_metrics[worker_id]
            
            if success:
                metrics.tasks_completed += 1
            else:
                metrics.tasks_failed += 1
            
            # Update rolling average execution time
            total_tasks = metrics.tasks_completed + metrics.tasks_failed
            if total_tasks > 1:
                metrics.average_task_time = (
                    metrics.average_task_time * (total_tasks - 1) + execution_time
                ) / total_tasks
            else:
                metrics.average_task_time = execution_time
            
            metrics.last_activity = datetime.now()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        active_workers = len([w for w in self.worker_metrics.values() if w.status in ["active", "busy"]])
        busy_workers = len([w for w in self.worker_metrics.values() if w.status == "busy"])
        
        total_completed = sum(w.tasks_completed for w in self.worker_metrics.values())
        total_failed = sum(w.tasks_failed for w in self.worker_metrics.values())
        
        if total_completed > 0:
            avg_task_time = sum(w.average_task_time * w.tasks_completed for w in self.worker_metrics.values()) / total_completed
        else:
            avg_task_time = 0.0
        
        return {
            "total_workers": len(self.worker_pool),
            "active_workers": active_workers,
            "busy_workers": busy_workers,
            "utilization": busy_workers / max(active_workers, 1),
            "pending_tasks": self.pending_tasks,
            "queue_size": self.task_queue.qsize(),
            "completed_tasks": total_completed,
            "failed_tasks": total_failed,
            "success_rate": total_completed / max(total_completed + total_failed, 1),
            "average_task_time": avg_task_time,
            "scaling_events": len(self.scaling_history)
        }
    
    def shutdown(self) -> None:
        """Shutdown all workers and cleanup."""
        logger.info("Shutting down load balancer...")
        
        for worker_id in list(self.worker_pool.keys()):
            self._stop_worker(worker_id)
        
        # Clear queues
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Load balancer shutdown complete")


class MemoryOptimizer:
    """Advanced memory optimization and garbage collection."""
    
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.memory_threshold = 0.8  # 80% of max memory
        
        # Memory tracking
        self.memory_usage_history = deque(maxlen=1000)
        self.gc_events = deque(maxlen=100)
        
        # Object pools
        self.array_pool = defaultdict(list)  # Pool of numpy arrays by shape
        self.dict_pool = []  # Pool of dictionaries
        
        # Weak references for large objects
        self.large_object_registry = weakref.WeakValueDictionary()
        
        # Memory monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.start_monitoring()
    
    def start_monitoring(self) -> None:
        """Start memory monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Memory optimizer monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for memory optimization."""
        while self.monitoring_active:
            try:
                # Get current memory usage
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Record usage
                self.memory_usage_history.append({
                    "timestamp": time.time(),
                    "memory_mb": memory_mb,
                    "memory_percent": memory_mb / self.max_memory_mb
                })
                
                # Check if we need to free memory
                if memory_mb > self.max_memory_mb * self.memory_threshold:
                    self._aggressive_cleanup()
                
                # Regular maintenance
                if len(self.memory_usage_history) % 10 == 0:
                    self._maintenance_cleanup()
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(10.0)
    
    def _aggressive_cleanup(self) -> None:
        """Aggressive memory cleanup when threshold exceeded."""
        logger.warning("Memory threshold exceeded, performing aggressive cleanup")
        
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        # Clear object pools
        self.array_pool.clear()
        self.dict_pool.clear()
        
        # Force garbage collection multiple times
        for i in range(3):
            collected = gc.collect()
            if i == 0:
                logger.debug(f"GC pass {i+1}: collected {collected} objects")
        
        # Clear weak references
        self.large_object_registry.clear()
        
        final_memory = self._get_memory_usage()
        cleanup_time = time.time() - start_time
        memory_freed = initial_memory - final_memory
        
        self.gc_events.append({
            "timestamp": start_time,
            "type": "aggressive",
            "memory_freed_mb": memory_freed,
            "cleanup_time": cleanup_time,
            "objects_collected": collected if 'collected' in locals() else 0
        })
        
        logger.info(f"Aggressive cleanup freed {memory_freed:.1f} MB in {cleanup_time:.2f}s")
    
    def _maintenance_cleanup(self) -> None:
        """Regular maintenance cleanup."""
        start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        # Trim object pools
        for shape, arrays in self.array_pool.items():
            if len(arrays) > 10:  # Keep max 10 arrays per shape
                self.array_pool[shape] = arrays[-10:]
        
        if len(self.dict_pool) > 50:  # Keep max 50 dicts
            self.dict_pool = self.dict_pool[-50:]
        
        # Gentle garbage collection
        collected = gc.collect(0)  # Only generation 0
        
        final_memory = self._get_memory_usage()
        cleanup_time = time.time() - start_time
        memory_freed = initial_memory - final_memory
        
        if memory_freed > 1.0:  # Only log if significant memory freed
            self.gc_events.append({
                "timestamp": start_time,
                "type": "maintenance",
                "memory_freed_mb": memory_freed,
                "cleanup_time": cleanup_time,
                "objects_collected": collected
            })
    
    def get_array(self, shape: Tuple[int, ...], dtype=np.float64) -> np.ndarray:
        """Get array from pool or create new one."""
        pool_key = (shape, str(dtype))
        
        if pool_key in self.array_pool and self.array_pool[pool_key]:
            array = self.array_pool[pool_key].pop()
            array.fill(0)  # Clear existing data
            return array
        else:
            return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray) -> None:
        """Return array to pool for reuse."""
        if array.size > 1000:  # Only pool large arrays
            pool_key = (array.shape, str(array.dtype))
            if len(self.array_pool[pool_key]) < 5:  # Max 5 per shape
                self.array_pool[pool_key].append(array)
    
    def get_dict(self) -> Dict[str, Any]:
        """Get dictionary from pool or create new one."""
        if self.dict_pool:
            return self.dict_pool.pop()
        else:
            return {}
    
    def return_dict(self, d: Dict[str, Any]) -> None:
        """Return dictionary to pool for reuse."""
        d.clear()
        if len(self.dict_pool) < 50:
            self.dict_pool.append(d)
    
    def register_large_object(self, obj_id: str, obj: Any) -> None:
        """Register large object for monitoring."""
        self.large_object_registry[obj_id] = obj
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memory_usage_history:
            return {"no_data": True}
        
        current_memory = self._get_memory_usage()
        recent_usage = list(self.memory_usage_history)[-10:]  # Last 10 readings
        
        return {
            "current_memory_mb": current_memory,
            "max_memory_mb": self.max_memory_mb,
            "memory_utilization": current_memory / self.max_memory_mb,
            "average_memory_mb": sum(u["memory_mb"] for u in recent_usage) / len(recent_usage),
            "array_pool_size": sum(len(arrays) for arrays in self.array_pool.values()),
            "dict_pool_size": len(self.dict_pool),
            "large_objects_tracked": len(self.large_object_registry),
            "gc_events": len(self.gc_events),
            "recent_gc_events": list(self.gc_events)[-5:]  # Last 5 GC events
        }


class CachingSystem:
    """Intelligent caching with LRU and TTL support."""
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: float = 3600,  # 1 hour
        redis_config: Optional[Dict[str, Any]] = None
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Local cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order = deque()  # For LRU
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
        
        # Redis distributed cache (optional)
        self.redis_client = None
        if redis_config:
            try:
                self.redis_client = redis.Redis(**redis_config)
                self.redis_client.ping()  # Test connection
                logger.info("Redis distributed cache connected")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        
        # Cache maintenance
        self.maintenance_active = False
        self.maintenance_thread = None
        self.start_maintenance()
    
    def start_maintenance(self) -> None:
        """Start cache maintenance thread."""
        if self.maintenance_active:
            return
        
        self.maintenance_active = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
    
    def stop_maintenance(self) -> None:
        """Stop cache maintenance."""
        self.maintenance_active = False
        if self.maintenance_thread and self.maintenance_thread.is_alive():
            self.maintenance_thread.join(timeout=5.0)
    
    def _maintenance_loop(self) -> None:
        """Cache maintenance loop."""
        while self.maintenance_active:
            try:
                # Clean expired entries
                self._cleanup_expired()
                
                # Enforce size limits
                if len(self.cache) > self.max_size:
                    self._evict_lru(len(self.cache) - self.max_size)
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Cache maintenance error: {e}")
                time.sleep(60)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        # Check local cache first
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if time.time() < entry["expires_at"]:
                # Update access order for LRU
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
                self.access_order.append(key)
                
                self.cache_stats["hits"] += 1
                return entry["value"]
            else:
                # Expired
                del self.cache[key]
                self.cache_stats["expired"] += 1
        
        # Check Redis if available
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value is not None:
                    deserialized_value = pickle.loads(value)
                    
                    # Cache locally for faster future access
                    self.set(key, deserialized_value, ttl=self.default_ttl // 2)
                    
                    self.cache_stats["hits"] += 1
                    return deserialized_value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Cache miss
        self.cache_stats["misses"] += 1
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        # Store in local cache
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                serialized_value = pickle.dumps(value)
                self.redis_client.setex(key, int(ttl), serialized_value)
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
        
        # Enforce size limit
        if len(self.cache) > self.max_size:
            self._evict_lru(1)
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        deleted = False
        
        # Remove from local cache
        if key in self.cache:
            del self.cache[key]
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            deleted = True
        
        # Remove from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                deleted = True
            except Exception as e:
                logger.warning(f"Redis cache delete error: {e}")
        
        return deleted
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
        
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis cache clear error: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time >= entry["expires_at"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.cache_stats["expired"] += 1
    
    def _evict_lru(self, count: int) -> None:
        """Evict least recently used entries."""
        for _ in range(min(count, len(self.access_order))):
            if self.access_order:
                lru_key = self.access_order.popleft()
                if lru_key in self.cache:
                    del self.cache[lru_key]
                    self.cache_stats["evictions"] += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(total_requests, 1)
        
        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "redis_enabled": self.redis_client is not None
        }


class DistributedComputeManager:
    """Manage distributed computation across multiple nodes."""
    
    def __init__(
        self,
        node_configs: Optional[List[Dict[str, Any]]] = None,
        communication_backend: str = "zmq"
    ):
        self.node_configs = node_configs or []
        self.communication_backend = communication_backend
        
        # Node management
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.node_status: Dict[str, str] = {}  # active, busy, error, offline
        
        # Task distribution
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.pending_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Communication
        if communication_backend == "zmq":
            self.context = zmq.Context()
            self.socket = None
        
        # Performance tracking
        self.node_metrics: Dict[str, Dict[str, Any]] = {}
        self.distribution_history = deque(maxlen=1000)
        
        self._initialize_nodes()
    
    def _initialize_nodes(self) -> None:
        """Initialize compute nodes."""
        for node_config in self.node_configs:
            node_id = node_config.get("node_id", f"node_{len(self.nodes)}")
            
            self.nodes[node_id] = {
                "config": node_config,
                "capabilities": node_config.get("capabilities", ["general"]),
                "max_concurrent_tasks": node_config.get("max_tasks", 4),
                "current_tasks": 0,
                "total_completed": 0,
                "total_failed": 0
            }
            
            self.node_status[node_id] = "offline"
            self.node_metrics[node_id] = {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "task_completion_rate": 0.0,
                "average_task_time": 0.0,
                "last_heartbeat": 0.0
            }
            
            logger.info(f"Initialized node: {node_id}")
    
    def connect_nodes(self) -> None:
        """Establish connections to compute nodes."""
        if self.communication_backend == "zmq":
            self.socket = self.context.socket(zmq.DEALER)
            
            for node_id, node in self.nodes.items():
                address = node["config"].get("address", "tcp://localhost:5555")
                try:
                    self.socket.connect(address)
                    self.node_status[node_id] = "active"
                    logger.info(f"Connected to node {node_id} at {address}")
                except Exception as e:
                    logger.error(f"Failed to connect to node {node_id}: {e}")
                    self.node_status[node_id] = "error"
    
    def submit_distributed_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        priority: int = 0
    ) -> str:
        """Submit task for distributed execution."""
        task_id = f"dist_task_{time.time()}_{np.random.randint(10000)}"
        
        task = {
            "task_id": task_id,
            "type": task_type,
            "data": task_data,
            "required_capabilities": required_capabilities or ["general"],
            "priority": priority,
            "submitted_time": time.time(),
            "status": "queued"
        }
        
        # Find suitable node
        suitable_node = self._select_node(task)
        
        if suitable_node:
            task["assigned_node"] = suitable_node
            task["status"] = "assigned"
            
            # Send task to node
            self._send_task_to_node(task, suitable_node)
            
            self.pending_tasks[task_id] = task
            
            # Update node state
            self.nodes[suitable_node]["current_tasks"] += 1
            if self.nodes[suitable_node]["current_tasks"] >= self.nodes[suitable_node]["max_concurrent_tasks"]:
                self.node_status[suitable_node] = "busy"
            
            logger.debug(f"Distributed task {task_id} assigned to node {suitable_node}")
        else:
            # Queue task for later
            self.task_queue.put(task)
            logger.debug(f"Task {task_id} queued (no suitable nodes available)")
        
        return task_id
    
    def _select_node(self, task: Dict[str, Any]) -> Optional[str]:
        """Select best node for task execution."""
        required_caps = set(task["required_capabilities"])
        
        # Filter nodes by capabilities and availability
        suitable_nodes = []
        
        for node_id, node in self.nodes.items():
            if (self.node_status[node_id] == "active" and
                node["current_tasks"] < node["max_concurrent_tasks"] and
                required_caps.issubset(set(node["capabilities"]))):
                
                # Calculate node score (lower is better)
                load_factor = node["current_tasks"] / node["max_concurrent_tasks"]
                performance_factor = 1.0 / max(self.node_metrics[node_id]["task_completion_rate"], 0.1)
                
                score = load_factor + performance_factor
                suitable_nodes.append((node_id, score))
        
        if suitable_nodes:
            # Select node with best score
            suitable_nodes.sort(key=lambda x: x[1])
            return suitable_nodes[0][0]
        
        return None
    
    def _send_task_to_node(self, task: Dict[str, Any], node_id: str) -> None:
        """Send task to specific node."""
        if self.communication_backend == "zmq" and self.socket:
            try:
                message = {
                    "type": "task",
                    "node_id": node_id,
                    "task": task
                }
                
                self.socket.send_json(message, zmq.NOBLOCK)
                
            except Exception as e:
                logger.error(f"Failed to send task to node {node_id}: {e}")
                self.node_status[node_id] = "error"
    
    def check_results(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Check for completed distributed tasks."""
        results = []
        
        if self.communication_backend == "zmq" and self.socket:
            try:
                while True:
                    message = self.socket.recv_json(zmq.NOBLOCK)
                    
                    if message["type"] == "result":
                        task_id = message["task_id"]
                        result = message["result"]
                        node_id = message["node_id"]
                        
                        # Update node state
                        if node_id in self.nodes:
                            self.nodes[node_id]["current_tasks"] = max(0, self.nodes[node_id]["current_tasks"] - 1)
                            
                            if result.get("status") == "completed":
                                self.nodes[node_id]["total_completed"] += 1
                            else:
                                self.nodes[node_id]["total_failed"] += 1
                            
                            # Update node status
                            if self.node_status[node_id] == "busy" and self.nodes[node_id]["current_tasks"] < self.nodes[node_id]["max_concurrent_tasks"]:
                                self.node_status[node_id] = "active"
                        
                        # Clean up pending task
                        if task_id in self.pending_tasks:
                            del self.pending_tasks[task_id]
                        
                        results.append({
                            "task_id": task_id,
                            "node_id": node_id,
                            "result": result
                        })
                        
                    elif message["type"] == "heartbeat":
                        node_id = message["node_id"]
                        metrics = message["metrics"]
                        
                        # Update node metrics
                        if node_id in self.node_metrics:
                            self.node_metrics[node_id].update(metrics)
                            self.node_metrics[node_id]["last_heartbeat"] = time.time()
                        
                        # Update node status
                        if node_id in self.node_status and self.node_status[node_id] == "offline":
                            self.node_status[node_id] = "active"
                            
            except zmq.Again:
                # No messages available
                pass
            except Exception as e:
                logger.error(f"Error checking distributed results: {e}")
        
        return results
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get distributed cluster statistics."""
        total_nodes = len(self.nodes)
        active_nodes = len([n for n in self.node_status.values() if n == "active"])
        busy_nodes = len([n for n in self.node_status.values() if n == "busy"])
        error_nodes = len([n for n in self.node_status.values() if n == "error"])
        
        total_capacity = sum(node["max_concurrent_tasks"] for node in self.nodes.values())
        current_load = sum(node["current_tasks"] for node in self.nodes.values())
        
        total_completed = sum(node["total_completed"] for node in self.nodes.values())
        total_failed = sum(node["total_failed"] for node in self.nodes.values())
        
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "busy_nodes": busy_nodes,
            "error_nodes": error_nodes,
            "offline_nodes": total_nodes - active_nodes - busy_nodes - error_nodes,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "utilization": current_load / max(total_capacity, 1),
            "pending_tasks": len(self.pending_tasks),
            "queued_tasks": self.task_queue.qsize(),
            "total_completed": total_completed,
            "total_failed": total_failed,
            "success_rate": total_completed / max(total_completed + total_failed, 1),
            "node_details": {
                node_id: {
                    "status": self.node_status[node_id],
                    "current_tasks": node["current_tasks"],
                    "max_tasks": node["max_concurrent_tasks"],
                    "completed": node["total_completed"],
                    "failed": node["total_failed"],
                    "metrics": self.node_metrics[node_id]
                }
                for node_id, node in self.nodes.items()
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown distributed compute manager."""
        logger.info("Shutting down distributed compute manager...")
        
        if self.socket:
            # Send shutdown message to all nodes
            for node_id in self.nodes:
                try:
                    message = {
                        "type": "shutdown",
                        "node_id": node_id
                    }
                    self.socket.send_json(message, zmq.NOBLOCK)
                except Exception as e:
                    logger.error(f"Failed to send shutdown to node {node_id}: {e}")
            
            self.socket.close()
        
        if self.context:
            self.context.term()
        
        logger.info("Distributed compute manager shutdown complete")


# Global instances for scaling optimization
global_load_balancer = AdaptiveLoadBalancer()
global_memory_optimizer = MemoryOptimizer()
global_cache = CachingSystem()
global_compute_manager = DistributedComputeManager()

logger.info("Scaling and optimization systems initialized for Generation 3")
