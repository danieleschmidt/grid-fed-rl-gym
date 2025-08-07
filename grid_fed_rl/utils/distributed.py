"""Distributed computing and parallel processing utilities."""

import multiprocessing as mp
import concurrent.futures
import threading
import queue
import time
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from .exceptions import GridEnvironmentError
from .performance import PerformanceProfiler


@dataclass
class TaskResult:
    """Result of a distributed task."""
    task_id: str
    result: Any
    execution_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class WorkerStatus:
    """Status of a distributed worker."""
    worker_id: str
    is_active: bool
    tasks_completed: int
    average_task_time: float
    last_heartbeat: float


class TaskQueue:
    """Thread-safe task queue for distributed processing."""
    
    def __init__(self, maxsize: int = 0):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        self.completed_tasks = []
        self.failed_tasks = []
        
    def put_task(self, task_id: str, func: Callable, args: Tuple = (), kwargs: Dict = None) -> None:
        """Add a task to the queue."""
        task = {
            'id': task_id,
            'func': func,
            'args': args or (),
            'kwargs': kwargs or {},
            'timestamp': time.time()
        }
        self.queue.put(task, block=False)
        
    def get_task(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """Get a task from the queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def task_done(self) -> None:
        """Mark task as done."""
        self.queue.task_done()
        
    def add_result(self, result: TaskResult) -> None:
        """Add task result."""
        with self.lock:
            if result.success:
                self.completed_tasks.append(result)
            else:
                self.failed_tasks.append(result)
                
    def get_results(self) -> Tuple[List[TaskResult], List[TaskResult]]:
        """Get completed and failed task results."""
        with self.lock:
            return self.completed_tasks.copy(), self.failed_tasks.copy()
            
    def clear_results(self) -> None:
        """Clear stored results."""
        with self.lock:
            self.completed_tasks.clear()
            self.failed_tasks.clear()


class DistributedWorker:
    """Worker process for distributed task execution."""
    
    def __init__(self, worker_id: str, task_queue: TaskQueue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.is_running = False
        self.tasks_completed = 0
        self.total_execution_time = 0.0
        self.logger = logging.getLogger(f"Worker-{worker_id}")
        
    def run(self) -> None:
        """Main worker loop."""
        self.is_running = True
        self.logger.info(f"Worker {self.worker_id} started")
        
        while self.is_running:
            task = self.task_queue.get_task(timeout=1.0)
            
            if task is None:
                continue  # No task available, keep waiting
                
            # Execute task
            start_time = time.time()
            try:
                result = task['func'](*task['args'], **task['kwargs'])
                execution_time = time.time() - start_time
                
                task_result = TaskResult(
                    task_id=task['id'],
                    result=result,
                    execution_time=execution_time,
                    success=True
                )
                
                self.tasks_completed += 1
                self.total_execution_time += execution_time
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Task {task['id']} failed: {e}")
                
                task_result = TaskResult(
                    task_id=task['id'],
                    result=None,
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
                
            # Store result
            self.task_queue.add_result(task_result)
            self.task_queue.task_done()
            
        self.logger.info(f"Worker {self.worker_id} stopped")
        
    def stop(self) -> None:
        """Stop the worker."""
        self.is_running = False
        
    def get_status(self) -> WorkerStatus:
        """Get worker status."""
        avg_time = (self.total_execution_time / max(self.tasks_completed, 1))
        
        return WorkerStatus(
            worker_id=self.worker_id,
            is_active=self.is_running,
            tasks_completed=self.tasks_completed,
            average_task_time=avg_time,
            last_heartbeat=time.time()
        )


class DistributedExecutor:
    """Distributed task executor with automatic load balancing."""
    
    def __init__(
        self,
        num_workers: int = None,
        max_queue_size: int = 1000,
        worker_timeout: float = 30.0
    ):
        self.num_workers = num_workers or mp.cpu_count()
        self.max_queue_size = max_queue_size
        self.worker_timeout = worker_timeout
        
        self.task_queue = TaskQueue(maxsize=max_queue_size)
        self.workers: List[DistributedWorker] = []
        self.worker_threads: List[threading.Thread] = []
        
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> None:
        """Start the distributed executor."""
        if self.is_running:
            return
            
        self.logger.info(f"Starting distributed executor with {self.num_workers} workers")
        
        # Create and start workers
        for i in range(self.num_workers):
            worker = DistributedWorker(f"worker-{i}", self.task_queue)
            thread = threading.Thread(target=worker.run, daemon=True)
            
            self.workers.append(worker)
            self.worker_threads.append(thread)
            thread.start()
            
        self.is_running = True
        
    def stop(self, timeout: float = 10.0) -> None:
        """Stop the distributed executor."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping distributed executor")
        
        # Stop all workers
        for worker in self.workers:
            worker.stop()
            
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=timeout)
            
        self.is_running = False
        self.logger.info("Distributed executor stopped")
        
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> None:
        """Submit a task for execution."""
        if not self.is_running:
            raise GridEnvironmentError("Executor is not running")
            
        try:
            self.task_queue.put_task(task_id, func, args, kwargs)
        except queue.Full:
            raise GridEnvironmentError("Task queue is full")
            
    def submit_batch(
        self,
        tasks: List[Tuple[str, Callable, Tuple, Dict]]
    ) -> None:
        """Submit multiple tasks as a batch."""
        for task_id, func, args, kwargs in tasks:
            self.submit_task(task_id, func, *args, **kwargs)
            
    def get_results(self, clear: bool = True) -> Tuple[List[TaskResult], List[TaskResult]]:
        """Get completed and failed task results."""
        completed, failed = self.task_queue.get_results()
        
        if clear:
            self.task_queue.clear_results()
            
        return completed, failed
        
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all submitted tasks to complete."""
        if timeout is not None:
            end_time = time.time() + timeout
            
        while True:
            if timeout is not None and time.time() > end_time:
                return False
                
            if self.task_queue.queue.empty():
                break
                
            time.sleep(0.1)
            
        return True
        
    def get_worker_status(self) -> List[WorkerStatus]:
        """Get status of all workers."""
        return [worker.get_status() for worker in self.workers]
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        completed, failed = self.get_results(clear=False)
        
        if not completed:
            return {
                "total_tasks": len(failed),
                "completed_tasks": 0,
                "failed_tasks": len(failed),
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "throughput": 0.0
            }
            
        execution_times = [result.execution_time for result in completed]
        
        return {
            "total_tasks": len(completed) + len(failed),
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "success_rate": len(completed) / (len(completed) + len(failed)),
            "average_execution_time": np.mean(execution_times),
            "min_execution_time": np.min(execution_times),
            "max_execution_time": np.max(execution_times),
            "throughput": len(completed) / np.sum(execution_times) if execution_times else 0.0
        }


class ParallelEnvironmentRunner:
    """Run multiple environments in parallel for batch training."""
    
    def __init__(
        self,
        env_factory: Callable,
        num_envs: int,
        env_configs: Optional[List[Dict]] = None
    ):
        self.env_factory = env_factory
        self.num_envs = num_envs
        self.env_configs = env_configs or [{}] * num_envs
        
        self.executor = DistributedExecutor(num_workers=num_envs)
        self.environments = []
        self.episode_data = []
        
        self.logger = logging.getLogger(__name__)
        
    def initialize_environments(self) -> None:
        """Initialize all parallel environments."""
        self.logger.info(f"Initializing {self.num_envs} parallel environments")
        
        self.executor.start()
        
        # Create environments in parallel
        tasks = []
        for i in range(self.num_envs):
            task_id = f"init_env_{i}"
            config = self.env_configs[i]
            tasks.append((task_id, self._create_environment, (i, config), {}))
            
        self.executor.submit_batch(tasks)
        self.executor.wait_for_completion(timeout=60.0)
        
        # Collect results
        completed, failed = self.executor.get_results()
        
        if failed:
            error_msgs = [result.error for result in failed]
            raise GridEnvironmentError(f"Failed to initialize environments: {error_msgs}")
            
        # Sort environments by index
        env_results = [(int(result.task_id.split('_')[-1]), result.result) for result in completed]
        env_results.sort(key=lambda x: x[0])
        
        self.environments = [env for _, env in env_results]
        self.logger.info(f"Successfully initialized {len(self.environments)} environments")
        
    def _create_environment(self, env_index: int, config: Dict) -> Any:
        """Create a single environment."""
        return self.env_factory(**config)
        
    def run_parallel_episodes(
        self,
        policy: Callable,
        episodes_per_env: int = 1,
        max_steps_per_episode: int = 1000
    ) -> List[Dict[str, Any]]:
        """Run episodes in parallel across all environments."""
        if not self.environments:
            raise GridEnvironmentError("Environments not initialized")
            
        self.logger.info(f"Running {episodes_per_env} episodes per environment in parallel")
        
        # Submit episode tasks
        tasks = []
        for env_idx in range(len(self.environments)):
            for ep_idx in range(episodes_per_env):
                task_id = f"episode_{env_idx}_{ep_idx}"
                tasks.append((
                    task_id, 
                    self._run_single_episode,
                    (env_idx, policy, max_steps_per_episode),
                    {}
                ))
                
        self.executor.submit_batch(tasks)
        self.executor.wait_for_completion(timeout=300.0)  # 5 minute timeout
        
        # Collect episode data
        completed, failed = self.executor.get_results()
        
        episode_data = []
        for result in completed:
            if result.success:
                episode_data.append(result.result)
            else:
                self.logger.warning(f"Episode {result.task_id} failed: {result.error}")
                
        self.logger.info(f"Completed {len(episode_data)} episodes successfully")
        return episode_data
        
    def _run_single_episode(
        self,
        env_index: int,
        policy: Callable,
        max_steps: int
    ) -> Dict[str, Any]:
        """Run a single episode in one environment."""
        env = self.environments[env_index]
        
        obs, _ = env.reset()
        episode_data = {
            "env_index": env_index,
            "observations": [obs.copy()],
            "actions": [],
            "rewards": [],
            "infos": [],
            "total_reward": 0.0,
            "steps": 0,
            "success": False
        }
        
        for step in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_data["actions"].append(action.copy())
            episode_data["rewards"].append(reward)
            episode_data["observations"].append(obs.copy())
            episode_data["infos"].append(info.copy())
            episode_data["total_reward"] += reward
            episode_data["steps"] = step + 1
            
            if terminated or truncated:
                episode_data["success"] = True
                break
                
        return episode_data
        
    def shutdown(self) -> None:
        """Shutdown the parallel runner."""
        self.executor.stop()
        self.environments.clear()
        

def parallel_power_flow_batch(
    solver,
    network_configs: List[Tuple],
    num_workers: int = None
) -> List[Any]:
    """Solve multiple power flow problems in parallel."""
    if not network_configs:
        return []
        
    executor = DistributedExecutor(num_workers=num_workers)
    executor.start()
    
    try:
        # Submit all power flow tasks
        tasks = []
        for i, (buses, lines, loads, generation) in enumerate(network_configs):
            task_id = f"powerflow_{i}"
            tasks.append((
                task_id,
                solver.solve,
                (buses, lines, loads, generation),
                {}
            ))
            
        executor.submit_batch(tasks)
        executor.wait_for_completion(timeout=120.0)  # 2 minute timeout
        
        # Collect results
        completed, failed = executor.get_results()
        
        # Sort results by task index
        results = [None] * len(network_configs)
        for result in completed:
            idx = int(result.task_id.split('_')[-1])
            results[idx] = result.result
            
        # Handle failed tasks
        for result in failed:
            idx = int(result.task_id.split('_')[-1])
            # Return a failed solution object
            from ..environments.power_flow import PowerFlowSolution
            results[idx] = PowerFlowSolution(
                converged=False,
                iterations=-1,
                bus_voltages=np.array([]),
                bus_angles=np.array([]),
                line_flows=np.array([]),
                losses=0.0
            )
            
        return results
        
    finally:
        executor.stop()


# Utility functions for common parallel operations

def parallel_map(func: Callable, items: List[Any], num_workers: int = None) -> List[Any]:
    """Apply function to items in parallel."""
    executor = DistributedExecutor(num_workers=num_workers)
    executor.start()
    
    try:
        tasks = [(f"task_{i}", func, (item,), {}) for i, item in enumerate(items)]
        executor.submit_batch(tasks)
        executor.wait_for_completion()
        
        completed, failed = executor.get_results()
        
        # Sort results by task index
        results = [None] * len(items)
        for result in completed:
            idx = int(result.task_id.split('_')[-1])
            results[idx] = result.result
            
        return results
        
    finally:
        executor.stop()


def benchmark_distributed_performance(
    func: Callable,
    test_data: List[Any],
    worker_counts: List[int] = [1, 2, 4, 8]
) -> Dict[str, Any]:
    """Benchmark performance with different worker counts."""
    results = {}
    
    for num_workers in worker_counts:
        print(f"Testing with {num_workers} workers...")
        
        start_time = time.time()
        parallel_results = parallel_map(func, test_data, num_workers=num_workers)
        end_time = time.time()
        
        results[f"workers_{num_workers}"] = {
            "execution_time": end_time - start_time,
            "throughput": len(test_data) / (end_time - start_time),
            "results_count": len([r for r in parallel_results if r is not None])
        }
        
    return results