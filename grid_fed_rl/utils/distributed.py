"""Distributed computing utilities for large-scale grid simulation."""

import json
import time
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import numpy as np

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


# Global instances
global_distributed_cache = DistributedCache()
global_simulator = None  # Created on demand to avoid resource usage