"""Auto-scaling and load balancing for grid simulation."""

import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Auto-scaling modes."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    mode: ScalingMode = ScalingMode.BALANCED
    min_workers: int = 1
    max_workers: int = 8
    target_utilization: float = 0.7
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown: float = 60.0  # seconds
    scale_down_cooldown: float = 300.0  # seconds
    monitoring_window: int = 100  # operations


@dataclass
class WorkerMetrics:
    """Metrics for a worker instance."""
    id: str
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_duration: float = 0.0
    last_activity: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    status: str = "idle"  # idle, busy, overloaded, error


class WorkerPool:
    """Managed pool of workers with auto-scaling."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.workers: Dict[str, WorkerMetrics] = {}
        self.task_queue = deque()
        self.performance_history = deque(maxlen=config.monitoring_window)
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.lock = threading.RLock()
        
        # Initialize minimum workers
        for i in range(config.min_workers):
            self._add_worker(f"worker_{i}")
            
    def submit_task(self, task: Callable, *args, **kwargs) -> str:
        """Submit task to worker pool."""
        task_id = f"task_{int(time.time() * 1000000)}"
        
        with self.lock:
            self.task_queue.append({
                'id': task_id,
                'task': task,
                'args': args,
                'kwargs': kwargs,
                'submitted_time': time.time()
            })
            
        # Trigger scaling check
        self._check_scaling_needs()
        
        return task_id
        
    def execute_pending_tasks(self) -> List[Any]:
        """Execute pending tasks using available workers."""
        results = []
        
        with self.lock:
            available_workers = [w for w in self.workers.values() if w.status in ["idle", "busy"]]
            
            while self.task_queue and available_workers:
                task_data = self.task_queue.popleft()
                worker = self._select_worker(available_workers)
                
                if worker:
                    result = self._execute_task(worker, task_data)
                    results.append(result)
                    
        return results
        
    def _add_worker(self, worker_id: str) -> None:
        """Add new worker to pool."""
        self.workers[worker_id] = WorkerMetrics(
            id=worker_id,
            last_activity=time.time()
        )
        logger.info(f"Added worker {worker_id}, total workers: {len(self.workers)}")
        
    def _remove_worker(self, worker_id: str) -> None:
        """Remove worker from pool."""
        if worker_id in self.workers and len(self.workers) > self.config.min_workers:
            del self.workers[worker_id]
            logger.info(f"Removed worker {worker_id}, total workers: {len(self.workers)}")
            
    def _select_worker(self, available_workers: List[WorkerMetrics]) -> Optional[WorkerMetrics]:
        """Select best worker using load balancing strategy."""
        if not available_workers:
            return None
            
        # Simple least-loaded strategy
        return min(available_workers, key=lambda w: w.active_tasks)
        
    def _execute_task(self, worker: WorkerMetrics, task_data: Dict) -> Any:
        """Execute task on selected worker."""
        start_time = time.time()
        worker.active_tasks += 1
        worker.status = "busy"
        
        try:
            task = task_data['task']
            args = task_data['args']
            kwargs = task_data['kwargs']
            
            result = task(*args, **kwargs)
            
            worker.completed_tasks += 1
            success = True
            
        except Exception as e:
            logger.error(f"Task execution failed on {worker.id}: {e}")
            worker.failed_tasks += 1
            result = None
            success = False
            
        finally:
            duration = time.time() - start_time
            worker.active_tasks = max(0, worker.active_tasks - 1)
            worker.last_activity = time.time()
            
            # Update average duration
            total_tasks = worker.completed_tasks + worker.failed_tasks
            if total_tasks > 0:
                worker.avg_task_duration = ((worker.avg_task_duration * (total_tasks - 1)) + duration) / total_tasks
                
            # Update status
            if worker.active_tasks == 0:
                worker.status = "idle"
            elif worker.active_tasks > 5:  # Threshold for overloaded
                worker.status = "overloaded"
                
            # Record performance
            self.performance_history.append({
                'worker_id': worker.id,
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })
            
        return result
        
    def _check_scaling_needs(self) -> None:
        """Check if scaling is needed and perform scaling."""
        current_time = time.time()
        
        # Calculate current utilization
        total_workers = len(self.workers)
        active_workers = sum(1 for w in self.workers.values() if w.active_tasks > 0)
        queue_size = len(self.task_queue)
        
        utilization = active_workers / total_workers if total_workers > 0 else 0
        
        # Scale up conditions
        scale_up_needed = (
            (utilization > self.config.scale_up_threshold or queue_size > total_workers * 2) and
            total_workers < self.config.max_workers and
            (current_time - self.last_scale_up) > self.config.scale_up_cooldown
        )
        
        # Scale down conditions
        scale_down_needed = (
            utilization < self.config.scale_down_threshold and
            queue_size == 0 and
            total_workers > self.config.min_workers and
            (current_time - self.last_scale_down) > self.config.scale_down_cooldown
        )
        
        if scale_up_needed:
            self._scale_up()
        elif scale_down_needed:
            self._scale_down()
            
    def _scale_up(self) -> None:
        """Scale up worker pool."""
        new_worker_id = f"worker_{len(self.workers)}"
        self._add_worker(new_worker_id)
        self.last_scale_up = time.time()
        
    def _scale_down(self) -> None:
        """Scale down worker pool."""
        # Find least active worker
        idle_workers = [w for w in self.workers.values() if w.status == "idle"]
        if idle_workers:
            worker_to_remove = min(idle_workers, key=lambda w: w.last_activity)
            self._remove_worker(worker_to_remove.id)
            self.last_scale_down = time.time()
            
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        total_workers = len(self.workers)
        active_workers = sum(1 for w in self.workers.values() if w.active_tasks > 0)
        idle_workers = sum(1 for w in self.workers.values() if w.status == "idle")
        overloaded_workers = sum(1 for w in self.workers.values() if w.status == "overloaded")
        
        total_completed = sum(w.completed_tasks for w in self.workers.values())
        total_failed = sum(w.failed_tasks for w in self.workers.values())
        
        avg_duration = 0.0
        if self.performance_history:
            recent_successful = [p for p in self.performance_history if p['success']]
            if recent_successful:
                avg_duration = sum(p['duration'] for p in recent_successful) / len(recent_successful)
                
        return {
            'total_workers': total_workers,
            'active_workers': active_workers,
            'idle_workers': idle_workers,
            'overloaded_workers': overloaded_workers,
            'queue_size': len(self.task_queue),
            'utilization': active_workers / total_workers if total_workers > 0 else 0,
            'total_completed_tasks': total_completed,
            'total_failed_tasks': total_failed,
            'success_rate': total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 0,
            'avg_task_duration_ms': avg_duration * 1000,
            'last_scale_up': self.last_scale_up,
            'last_scale_down': self.last_scale_down
        }


class ResourceManager:
    """Manage computational resources and allocation."""
    
    def __init__(self):
        self.resource_limits = {
            'max_memory_mb': 2000,
            'max_cpu_cores': 8,
            'max_concurrent_environments': 10
        }
        self.current_usage = {
            'memory_mb': 0,
            'cpu_cores': 0,
            'active_environments': 0
        }
        self.resource_history = deque(maxlen=1000)
        self.lock = threading.Lock()
        
    def allocate_resources(self, request: Dict[str, float]) -> bool:
        """Attempt to allocate resources."""
        with self.lock:
            # Check if resources are available
            if (self.current_usage['memory_mb'] + request.get('memory_mb', 0) > self.resource_limits['max_memory_mb'] or
                self.current_usage['cpu_cores'] + request.get('cpu_cores', 0) > self.resource_limits['max_cpu_cores'] or
                self.current_usage['active_environments'] + request.get('environments', 0) > self.resource_limits['max_concurrent_environments']):
                return False
                
            # Allocate resources
            for resource, amount in request.items():
                if resource in self.current_usage:
                    self.current_usage[resource] += amount
                    
            self.resource_history.append({
                'timestamp': time.time(),
                'action': 'allocate',
                'request': request.copy(),
                'current_usage': self.current_usage.copy()
            })
            
            return True
            
    def release_resources(self, release: Dict[str, float]) -> None:
        """Release allocated resources."""
        with self.lock:
            for resource, amount in release.items():
                if resource in self.current_usage:
                    self.current_usage[resource] = max(0, self.current_usage[resource] - amount)
                    
            self.resource_history.append({
                'timestamp': time.time(),
                'action': 'release',
                'release': release.copy(),
                'current_usage': self.current_usage.copy()
            })
            
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        utilization = {}
        for resource, current in self.current_usage.items():
            limit = self.resource_limits.get(f'max_{resource}', 1)
            utilization[resource] = current / limit if limit > 0 else 0
            
        return utilization
        
    def optimize_allocation(self) -> None:
        """Optimize resource allocation based on usage patterns."""
        if len(self.resource_history) < 100:
            return
            
        # Analyze recent usage patterns
        recent_usage = list(self.resource_history)[-100:]
        
        # Calculate average utilization
        avg_utilization = {}
        for resource in self.current_usage.keys():
            values = [entry['current_usage'][resource] for entry in recent_usage]
            avg_utilization[resource] = sum(values) / len(values)
            
        # Adjust limits based on usage
        for resource, avg_usage in avg_utilization.items():
            limit_key = f'max_{resource}'
            if limit_key in self.resource_limits:
                current_limit = self.resource_limits[limit_key]
                
                # If consistently using less than 50% of limit, reduce limit
                if avg_usage < current_limit * 0.5:
                    new_limit = max(current_limit * 0.8, avg_usage * 1.5)
                    self.resource_limits[limit_key] = new_limit
                    logger.info(f"Reduced {resource} limit to {new_limit}")
                    
                # If consistently using more than 90% of limit, increase limit
                elif avg_usage > current_limit * 0.9:
                    new_limit = min(current_limit * 1.2, current_limit + 1000)  # Cap increases
                    self.resource_limits[limit_key] = new_limit
                    logger.info(f"Increased {resource} limit to {new_limit}")
                    
    def get_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive resource usage report."""
        utilization = self.get_resource_utilization()
        
        return {
            'current_usage': self.current_usage.copy(),
            'resource_limits': self.resource_limits.copy(),
            'utilization': utilization,
            'total_allocations': len([h for h in self.resource_history if h['action'] == 'allocate']),
            'total_releases': len([h for h in self.resource_history if h['action'] == 'release']),
            'history_length': len(self.resource_history),
            'report_time': time.time()
        }


class LoadBalancer:
    """Load balancer for distributing work across resources."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.endpoints = {}
        self.request_counts = {}
        self.response_times = {}
        self.health_status = {}
        self.weights = {}
        
    def register_endpoint(self, endpoint_id: str, weight: float = 1.0) -> None:
        """Register new endpoint for load balancing."""
        self.endpoints[endpoint_id] = {
            'weight': weight,
            'active_requests': 0,
            'total_requests': 0,
            'total_response_time': 0.0,
            'last_request_time': 0.0
        }
        self.weights[endpoint_id] = weight
        self.health_status[endpoint_id] = True
        logger.info(f"Registered endpoint {endpoint_id} with weight {weight}")
        
    def select_endpoint(self) -> Optional[str]:
        """Select endpoint based on load balancing strategy."""
        healthy_endpoints = [eid for eid, status in self.health_status.items() if status]
        
        if not healthy_endpoints:
            return None
            
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted_selection(healthy_endpoints)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_selection(healthy_endpoints)
        else:
            return healthy_endpoints[0] if healthy_endpoints else None
            
    def _round_robin_selection(self, endpoints: List[str]) -> str:
        """Round-robin endpoint selection."""
        # Simple round-robin based on total requests
        return min(endpoints, key=lambda e: self.endpoints[e]['total_requests'])
        
    def _least_loaded_selection(self, endpoints: List[str]) -> str:
        """Least loaded endpoint selection."""
        return min(endpoints, key=lambda e: self.endpoints[e]['active_requests'])
        
    def _weighted_selection(self, endpoints: List[str]) -> str:
        """Weighted endpoint selection."""
        # Select based on weight and current load
        def weight_score(endpoint_id):
            endpoint = self.endpoints[endpoint_id]
            load = endpoint['active_requests'] + 1
            return load / self.weights[endpoint_id]
            
        return min(endpoints, key=weight_score)
        
    def _adaptive_selection(self, endpoints: List[str]) -> str:
        """Adaptive endpoint selection based on performance."""
        def performance_score(endpoint_id):
            endpoint = self.endpoints[endpoint_id]
            
            # Calculate average response time
            avg_response_time = (endpoint['total_response_time'] / endpoint['total_requests'] 
                               if endpoint['total_requests'] > 0 else 1.0)
            
            # Combine load and performance
            load_factor = endpoint['active_requests'] + 1
            time_factor = avg_response_time
            
            return load_factor * time_factor / self.weights[endpoint_id]
            
        return min(endpoints, key=performance_score)
        
    def record_request_start(self, endpoint_id: str) -> None:
        """Record request start."""
        if endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            endpoint['active_requests'] += 1
            endpoint['last_request_time'] = time.time()
            
    def record_request_end(self, endpoint_id: str, duration: float, success: bool = True) -> None:
        """Record request completion."""
        if endpoint_id in self.endpoints:
            endpoint = self.endpoints[endpoint_id]
            endpoint['active_requests'] = max(0, endpoint['active_requests'] - 1)
            endpoint['total_requests'] += 1
            endpoint['total_response_time'] += duration
            
            # Update health status based on recent performance
            if not success:
                self.health_status[endpoint_id] = False
                logger.warning(f"Endpoint {endpoint_id} marked as unhealthy")
            elif success and not self.health_status[endpoint_id]:
                self.health_status[endpoint_id] = True
                logger.info(f"Endpoint {endpoint_id} marked as healthy")
                
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        stats = {
            'strategy': self.strategy.value,
            'total_endpoints': len(self.endpoints),
            'healthy_endpoints': sum(1 for status in self.health_status.values() if status),
            'endpoints': {}
        }
        
        for endpoint_id, endpoint in self.endpoints.items():
            avg_response_time = (endpoint['total_response_time'] / endpoint['total_requests'] 
                               if endpoint['total_requests'] > 0 else 0)
            
            stats['endpoints'][endpoint_id] = {
                'weight': self.weights[endpoint_id],
                'active_requests': endpoint['active_requests'],
                'total_requests': endpoint['total_requests'],
                'avg_response_time_ms': avg_response_time * 1000,
                'healthy': self.health_status[endpoint_id]
            }
            
        return stats


# Global scaling components
default_scaling_config = ScalingConfig()
worker_pool = WorkerPool(default_scaling_config)
resource_manager = ResourceManager()
load_balancer = LoadBalancer()