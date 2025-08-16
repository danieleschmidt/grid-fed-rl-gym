"""Parallel environment execution for high-performance simulation."""

import time
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel environment execution."""
    max_parallel_envs: int = 4
    batch_size: int = 10
    enable_vectorization: bool = True
    enable_async_execution: bool = True
    result_timeout: float = 30.0  # seconds
    enable_result_caching: bool = True


class EnvironmentBatch:
    """Batch of environments for parallel execution."""
    
    def __init__(self, environments: List[Any], config: ParallelConfig):
        self.environments = environments
        self.config = config
        self.results = {}
        self.lock = threading.Lock()
        
    def reset_all(self, seeds: Optional[List[int]] = None) -> List[Tuple[Any, Dict]]:
        """Reset all environments in parallel."""
        if not self.config.enable_vectorization or len(self.environments) == 1:
            # Sequential execution
            results = []
            for i, env in enumerate(self.environments):
                seed = seeds[i] if seeds and i < len(seeds) else None
                obs, info = env.reset(seed=seed)
                results.append((obs, info))
            return results
            
        # Parallel execution
        def reset_env(env_idx: int) -> Tuple[int, Tuple[Any, Dict]]:
            env = self.environments[env_idx]
            seed = seeds[env_idx] if seeds and env_idx < len(seeds) else None
            try:
                obs, info = env.reset(seed=seed)
                return env_idx, (obs, info)
            except Exception as e:
                logger.error(f"Environment {env_idx} reset failed: {e}")
                return env_idx, (None, {'error': str(e)})
                
        results = [None] * len(self.environments)
        
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_envs) as executor:
            future_to_idx = {executor.submit(reset_env, i): i for i in range(len(self.environments))}
            
            for future in as_completed(future_to_idx, timeout=self.config.result_timeout):
                try:
                    env_idx, result = future.result()
                    results[env_idx] = result
                except Exception as e:
                    env_idx = future_to_idx[future]
                    logger.error(f"Environment {env_idx} reset timeout/error: {e}")
                    results[env_idx] = (None, {'error': str(e)})
                    
        return results
        
    def step_all(self, actions: List[Any]) -> List[Tuple[Any, float, bool, bool, Dict]]:
        """Execute step on all environments in parallel."""
        if len(actions) != len(self.environments):
            raise ValueError("Number of actions must match number of environments")
            
        if not self.config.enable_vectorization or len(self.environments) == 1:
            # Sequential execution
            results = []
            for env, action in zip(self.environments, actions):
                obs, reward, done, truncated, info = env.step(action)
                results.append((obs, reward, done, truncated, info))
            return results
            
        # Parallel execution
        def step_env(env_idx: int) -> Tuple[int, Tuple[Any, float, bool, bool, Dict]]:
            env = self.environments[env_idx]
            action = actions[env_idx]
            try:
                obs, reward, done, truncated, info = env.step(action)
                return env_idx, (obs, reward, done, truncated, info)
            except Exception as e:
                logger.error(f"Environment {env_idx} step failed: {e}")
                return env_idx, (None, 0.0, True, True, {'error': str(e)})
                
        results = [None] * len(self.environments)
        
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_envs) as executor:
            future_to_idx = {executor.submit(step_env, i): i for i in range(len(self.environments))}
            
            for future in as_completed(future_to_idx, timeout=self.config.result_timeout):
                try:
                    env_idx, result = future.result()
                    results[env_idx] = result
                except Exception as e:
                    env_idx = future_to_idx[future]
                    logger.error(f"Environment {env_idx} step timeout/error: {e}")
                    results[env_idx] = (None, 0.0, True, True, {'error': str(e)})
                    
        return results
        
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get statistics for the environment batch."""
        return {
            'num_environments': len(self.environments),
            'parallel_enabled': self.config.enable_vectorization,
            'max_workers': self.config.max_parallel_envs,
            'batch_size': self.config.batch_size
        }


class AsyncEnvironmentManager:
    """Asynchronous environment management for high-throughput simulation."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.environment_pools = {}
        self.active_tasks = defaultdict(int)
        self.completed_tasks = defaultdict(int)
        self.task_history = defaultdict(list)
        self.executor = ThreadPoolExecutor(max_workers=config.max_parallel_envs * 2)
        self.lock = threading.Lock()
        
    def register_environment_pool(self, pool_id: str, environments: List[Any]) -> None:
        """Register a pool of environments for async execution."""
        self.environment_pools[pool_id] = EnvironmentBatch(environments, self.config)
        logger.info(f"Registered environment pool '{pool_id}' with {len(environments)} environments")
        
    def submit_batch_reset(self, pool_id: str, seeds: Optional[List[int]] = None) -> str:
        """Submit batch reset operation asynchronously."""
        if pool_id not in self.environment_pools:
            raise ValueError(f"Unknown environment pool: {pool_id}")
            
        task_id = f"reset_{pool_id}_{int(time.time() * 1000000)}"
        
        def execute_reset():
            start_time = time.time()
            try:
                batch = self.environment_pools[pool_id]
                results = batch.reset_all(seeds)
                duration = time.time() - start_time
                
                with self.lock:
                    self.active_tasks[pool_id] -= 1
                    self.completed_tasks[pool_id] += 1
                    self.task_history[pool_id].append({
                        'task_id': task_id,
                        'operation': 'reset',
                        'duration': duration,
                        'success': True,
                        'timestamp': time.time()
                    })
                    
                return {'task_id': task_id, 'results': results, 'duration': duration}
            except Exception as e:
                duration = time.time() - start_time
                with self.lock:
                    self.active_tasks[pool_id] -= 1
                    self.task_history[pool_id].append({
                        'task_id': task_id,
                        'operation': 'reset',
                        'duration': duration,
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                raise
                
        with self.lock:
            self.active_tasks[pool_id] += 1
            
        future = self.executor.submit(execute_reset)
        return task_id
        
    def submit_batch_step(self, pool_id: str, actions: List[Any]) -> str:
        """Submit batch step operation asynchronously."""
        if pool_id not in self.environment_pools:
            raise ValueError(f"Unknown environment pool: {pool_id}")
            
        task_id = f"step_{pool_id}_{int(time.time() * 1000000)}"
        
        def execute_step():
            start_time = time.time()
            try:
                batch = self.environment_pools[pool_id]
                results = batch.step_all(actions)
                duration = time.time() - start_time
                
                with self.lock:
                    self.active_tasks[pool_id] -= 1
                    self.completed_tasks[pool_id] += 1
                    self.task_history[pool_id].append({
                        'task_id': task_id,
                        'operation': 'step',
                        'duration': duration,
                        'success': True,
                        'timestamp': time.time()
                    })
                    
                return {'task_id': task_id, 'results': results, 'duration': duration}
            except Exception as e:
                duration = time.time() - start_time
                with self.lock:
                    self.active_tasks[pool_id] -= 1
                    self.task_history[pool_id].append({
                        'task_id': task_id,
                        'operation': 'step',
                        'duration': duration,
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                raise
                
        with self.lock:
            self.active_tasks[pool_id] += 1
            
        future = self.executor.submit(execute_step)
        return task_id
        
    def get_pool_status(self, pool_id: str) -> Dict[str, Any]:
        """Get status of environment pool."""
        if pool_id not in self.environment_pools:
            return {'error': f'Unknown pool: {pool_id}'}
            
        with self.lock:
            recent_tasks = self.task_history[pool_id][-10:] if self.task_history[pool_id] else []
            
            successful_tasks = [t for t in recent_tasks if t['success']]
            failed_tasks = [t for t in recent_tasks if not t['success']]
            
            avg_duration = (sum(t['duration'] for t in successful_tasks) / len(successful_tasks) 
                          if successful_tasks else 0.0)
            
            return {
                'pool_id': pool_id,
                'num_environments': len(self.environment_pools[pool_id].environments),
                'active_tasks': self.active_tasks[pool_id],
                'completed_tasks': self.completed_tasks[pool_id],
                'total_tasks': len(self.task_history[pool_id]),
                'success_rate': len(successful_tasks) / len(recent_tasks) if recent_tasks else 0.0,
                'avg_duration_ms': avg_duration * 1000,
                'recent_failures': len(failed_tasks),
                'last_task_time': recent_tasks[-1]['timestamp'] if recent_tasks else None
            }
            
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall async manager status."""
        with self.lock:
            total_active = sum(self.active_tasks.values())
            total_completed = sum(self.completed_tasks.values())
            total_pools = len(self.environment_pools)
            
            pool_statuses = {}
            for pool_id in self.environment_pools:
                pool_statuses[pool_id] = self.get_pool_status(pool_id)
                
            return {
                'total_pools': total_pools,
                'total_active_tasks': total_active,
                'total_completed_tasks': total_completed,
                'executor_max_workers': self.config.max_parallel_envs * 2,
                'pools': pool_statuses,
                'timestamp': time.time()
            }
            
    def shutdown(self) -> None:
        """Shutdown the async environment manager."""
        self.executor.shutdown(wait=True)
        logger.info("Async environment manager shutdown complete")


class VectorizedEnvironment:
    """Vectorized environment wrapper for batch operations."""
    
    def __init__(self, environment_factory: Callable, num_envs: int, config: ParallelConfig):
        self.num_envs = num_envs
        self.config = config
        
        # Create environments
        self.environments = []
        for i in range(num_envs):
            try:
                env = environment_factory()
                self.environments.append(env)
            except Exception as e:
                logger.error(f"Failed to create environment {i}: {e}")
                raise
                
        # Create batch manager
        self.batch = EnvironmentBatch(self.environments, config)
        
        # Performance tracking
        self.step_count = 0
        self.total_step_time = 0.0
        self.reset_count = 0
        self.total_reset_time = 0.0
        
    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[List[Any], List[Dict]]:
        """Reset all environments and return observations and infos."""
        start_time = time.time()
        
        results = self.batch.reset_all(seeds)
        observations = [result[0] for result in results]
        infos = [result[1] for result in results]
        
        duration = time.time() - start_time
        self.reset_count += 1
        self.total_reset_time += duration
        
        # Performance optimization tracking
        try:
            from .performance_optimization import adaptive_optimizer
            adaptive_optimizer.record_performance("vectorized_reset", duration, all(obs is not None for obs in observations))
        except ImportError:
            pass
            
        return observations, infos
        
    def step(self, actions: List[Any]) -> Tuple[List[Any], List[float], List[bool], List[bool], List[Dict]]:
        """Step all environments and return results."""
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
            
        start_time = time.time()
        
        results = self.batch.step_all(actions)
        observations = [result[0] for result in results]
        rewards = [result[1] for result in results]
        dones = [result[2] for result in results]
        truncateds = [result[3] for result in results]
        infos = [result[4] for result in results]
        
        duration = time.time() - start_time
        self.step_count += 1
        self.total_step_time += duration
        
        # Performance optimization tracking
        try:
            from .performance_optimization import adaptive_optimizer
            adaptive_optimizer.record_performance("vectorized_step", duration, all(obs is not None for obs in observations))
        except ImportError:
            pass
            
        return observations, rewards, dones, truncateds, infos
        
    def close(self) -> None:
        """Close all environments."""
        for env in self.environments:
            try:
                env.close()
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")
                
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_step_time = self.total_step_time / self.step_count if self.step_count > 0 else 0.0
        avg_reset_time = self.total_reset_time / self.reset_count if self.reset_count > 0 else 0.0
        
        return {
            'num_environments': self.num_envs,
            'total_steps': self.step_count,
            'total_resets': self.reset_count,
            'avg_step_time_ms': avg_step_time * 1000,
            'avg_reset_time_ms': avg_reset_time * 1000,
            'steps_per_second': self.num_envs / avg_step_time if avg_step_time > 0 else 0.0,
            'parallel_enabled': self.config.enable_vectorization,
            'max_workers': self.config.max_parallel_envs
        }


# Global parallel processing components
default_parallel_config = ParallelConfig()
async_env_manager = AsyncEnvironmentManager(default_parallel_config)