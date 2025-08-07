```python
"""Distributed computing utilities for large-scale grid simulation."""

import json
import time
import pickle
import hashlib
import multiprocessing as mp
import concurrent.futures
import threading
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np

from .exceptions import GridEnvironmentError
from .performance import PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass  
class TaskConfig:
    """Configuration for distributed tasks."""
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = 300.0  # 5 minutes
    retry_count: int = 3
    

@dataclass
class TaskResult:
    """Result from distributed task execution."""
    task_id: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None
    worker_id: Optional[str] = None


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


class WorkerPool:
    """Process pool for distributed task execution."""
    
    def __init__(self, max_workers: int = 8, worker_timeout: float = 300.0):
        self.max_workers = max_workers
        self.worker_timeout = worker_timeout
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Performance metrics
        self.task_stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'avg_execution_time': 0.0
        }
        
    def submit_task(self, task_config: TaskConfig, worker_func: Callable) -> str:
        """Submit task for distributed execution."""
        
        future = self.executor.submit(
            self._execute_task_wrapper,
            task_config,
            worker_func
        )
        
        self.active_tasks[task_config.task_id] = {
            'future': future,
            'config': task_config,
            'submit_time': time.time()
        }
        
        self.task_stats['total_submitted'] += 1
        
        return task_config.task_id
    
    def submit_batch(self, task_configs: List[TaskConfig], worker_func: Callable) -> List[str]:
        """Submit multiple tasks as a batch."""
        task_ids = []
        
        for config in task_configs:
            task_id = self.submit_task(config, worker_func)
            task_ids.append(task_id)
            
        return task_ids
    
    def get_results(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """Get results from completed tasks."""
        
        results = []
        completed_task_ids = []
        
        for task_id, task_info in self.active_tasks.items():
            future = task_info['future']
            config = task_info['config']
            
            if future.done():
                try:
                    result = future.result(timeout=timeout)
                    
                    task_result = TaskResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        execution_time=time.time() - task_info['submit_time']
                    )
                    
                    results.append(task_result)
                    self.completed_tasks[task_id] = task_result
                    self.task_stats['total_completed'] += 1
                    
                except Exception as e:
                    task_result = TaskResult(
                        task_id=task_id,
                        success=False,
                        result=None,
                        execution_time=time.time() - task_info['submit_time'],
                        error_message=str(e)
                    )
                    
                    results.append(task_result)
                    self.failed_tasks[task_id] = task_result
                    self.task_stats['total_failed'] += 1
                    
                completed_task_ids.append(task_id)
        
        # Remove completed tasks from active list
        for task_id in completed_task_ids:
            del self.active_tasks[task_id]
            
        # Update average execution time
        if self.task_stats['total_completed'] > 0:
            total_time = sum(
                result.execution_time for result in self.completed_tasks.values()
            )
            self.task_stats['avg_execution_time'] = total_time / self.task_stats['total_completed']
        
        return results
    
    def wait_for_completion(self, task_ids: List[str] = None, timeout: float = 300.0) -> List[TaskResult]:
        """Wait for specific tasks or all active tasks to complete."""
        
        if task_ids is None:
            target_tasks = list(self.active_tasks.keys())
        else:
            target_tasks = task_ids
            
        start_time = time.time()
        all_results = []
        
        while target_tasks and (time.time() - start_time) < timeout:
            # Check for new results
            new_results = self.get_results(timeout=1.0)
            all_results.extend(new_results)
            
            # Remove completed tasks from target list
            completed_ids = [r.task_id for r in new_results]
            target_tasks = [tid for tid in target_tasks if tid not in completed_ids]
            
            if target_tasks:
                time.sleep(0.1)  # Brief pause
        
        return all_results
    
    @staticmethod
    def _execute_task_wrapper(task_config: TaskConfig, worker_func: Callable) -> Any:
        """Wrapper for task execution with error handling."""
        
        start_time = time.time()
        
        try:
            # Execute the actual task
            result = worker_func(task_config.parameters)
            
            execution_time = time.time() - start_time
            logger.debug(f"Task {task_config.task_id} completed in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task_config.task_id} failed after {execution_time:.3f}s: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            'max_workers': self.max_workers,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            **self.task_stats
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pool."""
        self.executor.shutdown(wait=wait)


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
                    success=True,
                    worker_id=self.worker_id
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
                    error_message=str(e),
                    worker_id=self.worker_id
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


class DistributedGridSimulator:
    """Distributed grid simulation coordinator."""
    
    def __init__(self, max_workers: int = 8):
        self.worker_pool = WorkerPool(max_workers=max_workers)
        
        # Simulation state
        self.simulation_configs = {}
        self.results_cache = {}
        
    def run_distributed_simulation(
        self,
        simulation_configs: List[Dict[str, Any]],
        simulation_steps: int = 100
    ) -> List[Dict[str, Any]]:
        """Run multiple grid simulations in parallel."""
        
        # Create task configurations
        task_configs = []
        for i, config in enumerate(simulation_configs):
            task_config = TaskConfig(
                task_id=f"sim_{i}",
                task_type="grid_simulation",
                parameters={
                    'config': config,
                    'steps': simulation_steps,
                    'seed': i  # Different seed for each simulation
                }
            )
            task_configs.append(task_config)
        
        # Submit tasks
        logger.info(f"Starting {len(task_configs)} distributed simulations")
        task_ids = self.worker_pool.submit_batch(task_configs, self._run_single_simulation)
        
        # Wait for completion
        results = self.worker_pool.wait_for_completion(task_ids, timeout=600.0)  # 10 minutes
        
        # Process results
        simulation_results = []
        for result in results:
            if result.success:
                simulation_results.append({
                    'task_id': result.task_id,
                    'success': True,
                    'data': result.result,
                    'execution_time': result.execution_time
                })
            else:
                simulation_results.append({
                    'task_id': result.task_id,
                    'success': False,
                    'error': result.error_message,
                    'execution_time': result.execution_time
                })
        
        return simulation_results
    
    def run_distributed_parameter_sweep(
        self,
        base_config: Dict[str, Any],
        parameter_ranges: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Run parameter sweep across multiple workers."""
        
        # Generate parameter combinations
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Create simulation configs
        simulation_configs = []
        for i, param_combo in enumerate(param_combinations):
            config = base_config.copy()
            
            # Apply parameter values
            for param_name, param_value in zip(param_names, param_combo):
                config[param_name] = param_value
            
            config['parameter_combo_id'] = i
            simulation_configs.append(config)
        
        logger.info(f"Running parameter sweep with {len(simulation_configs)} combinations")
        
        # Run distributed simulations
        return self.run_distributed_simulation(simulation_configs)
    
    @staticmethod
    def _run_single_simulation(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single grid simulation (executed in worker process)."""
        
        # Import here to avoid pickle issues
        import sys
        import numpy as np
        
        # Set random seed for reproducibility
        np.random.seed(parameters.get('seed', 42))
        
        config = parameters['config']
        steps = parameters['steps']
        
        # Simulate grid environment (simplified for demonstration)
        simulation_data = {
            'config': config,
            'steps': steps,
            'voltages': [],
            'frequencies': [],
            'rewards': [],
            'violations': 0
        }
        
        # Simulate time series data
        for step in range(steps):
            # Generate realistic grid data
            base_voltage = 1.0
            voltage_noise = np.random.normal(0, 0.02)
            voltage = base_voltage + voltage_noise
            
            base_frequency = 60.0
            frequency_noise = np.random.normal(0, 0.1)
            frequency = base_frequency + frequency_noise
            
            # Simple reward calculation
            voltage_penalty = abs(voltage - 1.0) * 10
            frequency_penalty = abs(frequency - 60.0) * 20
            reward = 100 - voltage_penalty - frequency_penalty
            
            # Check violations
            if voltage < 0.95 or voltage > 1.05:
                simulation_data['violations'] += 1
            if frequency < 59.5 or frequency > 60.5:
                simulation_data['violations'] += 1
            
            simulation_data['voltages'].append(voltage)
            simulation_data['frequencies'].append(frequency)
            simulation_data['rewards'].append(reward)
        
        # Calculate summary statistics
        simulation_data['avg_voltage'] = np.mean(simulation_data['voltages'])
        simulation_data['avg_frequency'] = np.mean(simulation_data['frequencies'])
        simulation_data['total_reward'] = sum(simulation_data['rewards'])
        simulation_data['violation_rate'] = simulation_data['violations'] / steps
        
        return simulation_data
    
    def shutdown(self):
        """Shutdown distributed simulator."""
        self.worker_pool.shutdown()


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
            error_msgs = [result.error_message for result in failed]
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
                self.logger.warning(f"Episode {result.task_id} failed: {result.error_message}")
                
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


class DistributedCache:
    """Distributed caching system for sharing results across processes."""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.local_cache = {}
        
        # Simple file-based caching for process sharing
        self.cache_dir = "/tmp/grid_fed_rl_cache"
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        
        # Check local cache first
        if key in self.local_cache:
            return self.local_cache[key]
        
        # Check file cache
        cache_file = f"{self.cache_dir}/{hashlib.md5(key.encode()).hexdigest()}.pkl"
        
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                
            # Update local cache
            if len(self.local_cache) < self.cache_size:
                self.local_cache[key] = result
                
            return result
            
        except (FileNotFoundError, pickle.PickleError):
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Cache result."""
        
        # Update local cache
        if len(self.local_cache) < self.cache_size:
            self.local_cache[key] = value
        
        # Update file cache
        cache_file = f"{self.cache_dir}/{hashlib.md5(key.encode()).hexdigest()}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def clear(self) -> None:
        """Clear all caches."""
        self.local_cache.clear()
        
        import os
        import glob
        
        cache_files = glob.glob(f"{self.cache_dir}/*.pkl")
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
            except OSError:
                pass


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


# Global instances
global_distributed_cache = DistributedCache()
global_simulator = None  # Created on demand to avoid resource usage
```
