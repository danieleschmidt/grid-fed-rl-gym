"""Parallel and concurrent processing for grid simulations."""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Iterator
import logging
import queue
import asyncio
from dataclasses import dataclass
from enum import Enum


class ProcessingStrategy(Enum):
    """Processing strategies for different workloads."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"         # I/O bound tasks
    MULTIPROCESS = "multiprocess" # CPU bound tasks  
    ASYNC = "async"               # Async I/O tasks
    ADAPTIVE = "adaptive"         # Auto-select based on workload


@dataclass
class ProcessingResult:
    """Result of parallel processing operation."""
    task_id: str
    result: Any
    execution_time: float
    worker_id: Optional[str] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = None


class WorkloadAnalyzer:
    """Analyze workload characteristics to choose optimal processing strategy."""
    
    def __init__(self):
        self.cpu_cores = mp.cpu_count()
        self.logger = logging.getLogger(__name__)
    
    def analyze_task(self, func: Callable, sample_args: List[Any] = None) -> ProcessingStrategy:
        """Analyze task characteristics to recommend processing strategy."""
        
        # Basic heuristics
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        # I/O intensive indicators
        io_patterns = [
            'read', 'write', 'download', 'upload', 'fetch', 'request',
            'query', 'http', 'api', 'database', 'file', 'network'
        ]
        
        # CPU intensive indicators  
        cpu_patterns = [
            'compute', 'calculate', 'solve', 'optimize', 'matrix',
            'simulation', 'numeric', 'algorithm', 'process', 'analyze'
        ]
        
        # Async patterns
        async_patterns = [
            'async', 'await', 'coroutine', 'aiohttp', 'websocket'
        ]
        
        func_lower = func_name.lower()
        
        # Check for async patterns
        for pattern in async_patterns:
            if pattern in func_lower:
                return ProcessingStrategy.ASYNC
        
        # Check for I/O patterns
        io_score = sum(1 for pattern in io_patterns if pattern in func_lower)
        cpu_score = sum(1 for pattern in cpu_patterns if pattern in func_lower)
        
        if io_score > cpu_score:
            return ProcessingStrategy.THREADED
        elif cpu_score > io_score:
            return ProcessingStrategy.MULTIPROCESS
        
        # Performance test with sample
        if sample_args:
            try:
                return self._benchmark_strategies(func, sample_args)
            except Exception as e:
                self.logger.warning(f"Benchmark failed: {e}")
        
        # Default fallback
        return ProcessingStrategy.THREADED
    
    def _benchmark_strategies(self, func: Callable, sample_args: List[Any]) -> ProcessingStrategy:
        """Benchmark different strategies with sample data."""
        
        if len(sample_args) < 2:
            return ProcessingStrategy.THREADED
        
        sample_size = min(4, len(sample_args))
        test_args = sample_args[:sample_size]
        
        strategies_to_test = [
            ProcessingStrategy.SEQUENTIAL,
            ProcessingStrategy.THREADED,
        ]
        
        # Only test multiprocessing if we have enough work
        if len(sample_args) >= self.cpu_cores:
            strategies_to_test.append(ProcessingStrategy.MULTIPROCESS)
        
        best_strategy = ProcessingStrategy.THREADED
        best_time = float('inf')
        
        for strategy in strategies_to_test:
            try:
                start_time = time.time()
                
                if strategy == ProcessingStrategy.SEQUENTIAL:
                    for args in test_args:
                        func(args) if not isinstance(args, tuple) else func(*args)
                
                elif strategy == ProcessingStrategy.THREADED:
                    with ThreadPoolExecutor(max_workers=min(4, len(test_args))) as executor:
                        futures = []
                        for args in test_args:
                            if isinstance(args, tuple):
                                futures.append(executor.submit(func, *args))
                            else:
                                futures.append(executor.submit(func, args))
                        
                        for future in as_completed(futures):
                            future.result()  # Wait for completion
                
                elif strategy == ProcessingStrategy.MULTIPROCESS:
                    with ProcessPoolExecutor(max_workers=min(self.cpu_cores, len(test_args))) as executor:
                        futures = []
                        for args in test_args:
                            if isinstance(args, tuple):
                                futures.append(executor.submit(func, *args))
                            else:
                                futures.append(executor.submit(func, args))
                        
                        for future in as_completed(futures):
                            future.result()  # Wait for completion
                
                elapsed = time.time() - start_time
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_strategy = strategy
                    
            except Exception as e:
                self.logger.warning(f"Strategy {strategy} failed in benchmark: {e}")
                continue
        
        return best_strategy


class ParallelProcessor:
    """High-performance parallel processor with adaptive strategies."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE,
        timeout: Optional[float] = None
    ):
        self.max_workers = max_workers or mp.cpu_count()
        self.strategy = strategy
        self.timeout = timeout
        self.workload_analyzer = WorkloadAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.execution_stats: Dict[str, List[float]] = {}
        self.strategy_performance: Dict[ProcessingStrategy, List[float]] = {}
    
    def _execute_sequential(
        self, 
        func: Callable, 
        tasks: List[Any],
        task_ids: List[str]
    ) -> List[ProcessingResult]:
        """Execute tasks sequentially."""
        results = []
        
        for i, (task, task_id) in enumerate(zip(tasks, task_ids)):
            start_time = time.time()
            
            try:
                if isinstance(task, tuple):
                    result = func(*task)
                else:
                    result = func(task)
                
                execution_time = time.time() - start_time
                
                results.append(ProcessingResult(
                    task_id=task_id,
                    result=result,
                    execution_time=execution_time,
                    worker_id="main",
                    metadata={"strategy": "sequential", "task_index": i}
                ))
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                results.append(ProcessingResult(
                    task_id=task_id,
                    result=None,
                    execution_time=execution_time,
                    error=e,
                    worker_id="main",
                    metadata={"strategy": "sequential", "task_index": i}
                ))
        
        return results
    
    def _execute_threaded(
        self,
        func: Callable,
        tasks: List[Any],
        task_ids: List[str]
    ) -> List[ProcessingResult]:
        """Execute tasks using thread pool."""
        results = []
        worker_count = min(self.max_workers, len(tasks))
        
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            future_to_task = {}
            
            for i, (task, task_id) in enumerate(zip(tasks, task_ids)):
                start_time = time.time()
                
                if isinstance(task, tuple):
                    future = executor.submit(func, *task)
                else:
                    future = executor.submit(func, task)
                
                future_to_task[future] = {
                    'task_id': task_id,
                    'start_time': start_time,
                    'task_index': i
                }
            
            # Collect results
            for future in as_completed(future_to_task, timeout=self.timeout):
                task_info = future_to_task[future]
                execution_time = time.time() - task_info['start_time']
                
                try:
                    result = future.result()
                    
                    results.append(ProcessingResult(
                        task_id=task_info['task_id'],
                        result=result,
                        execution_time=execution_time,
                        worker_id=f"thread-{threading.get_ident()}",
                        metadata={"strategy": "threaded", "task_index": task_info['task_index']}
                    ))
                    
                except Exception as e:
                    results.append(ProcessingResult(
                        task_id=task_info['task_id'],
                        result=None,
                        execution_time=execution_time,
                        error=e,
                        worker_id=f"thread-{threading.get_ident()}",
                        metadata={"strategy": "threaded", "task_index": task_info['task_index']}
                    ))
        
        return results
    
    def _execute_multiprocess(
        self,
        func: Callable,
        tasks: List[Any], 
        task_ids: List[str]
    ) -> List[ProcessingResult]:
        """Execute tasks using process pool."""
        results = []
        worker_count = min(self.max_workers, len(tasks))
        
        # Multiprocessing requires serializable functions and data
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                # Submit all tasks
                future_to_task = {}
                
                for i, (task, task_id) in enumerate(zip(tasks, task_ids)):
                    start_time = time.time()
                    
                    if isinstance(task, tuple):
                        future = executor.submit(func, *task)
                    else:
                        future = executor.submit(func, task)
                    
                    future_to_task[future] = {
                        'task_id': task_id,
                        'start_time': start_time,
                        'task_index': i
                    }
                
                # Collect results
                for future in as_completed(future_to_task, timeout=self.timeout):
                    task_info = future_to_task[future]
                    execution_time = time.time() - task_info['start_time']
                    
                    try:
                        result = future.result()
                        
                        results.append(ProcessingResult(
                            task_id=task_info['task_id'],
                            result=result,
                            execution_time=execution_time,
                            worker_id=f"process-{mp.current_process().pid}",
                            metadata={"strategy": "multiprocess", "task_index": task_info['task_index']}
                        ))
                        
                    except Exception as e:
                        results.append(ProcessingResult(
                            task_id=task_info['task_id'],
                            result=None,
                            execution_time=execution_time,
                            error=e,
                            worker_id=f"process-{mp.current_process().pid}",
                            metadata={"strategy": "multiprocess", "task_index": task_info['task_index']}
                        ))
                        
        except Exception as e:
            self.logger.warning(f"Multiprocessing failed: {e}, falling back to threading")
            return self._execute_threaded(func, tasks, task_ids)
        
        return results
    
    async def _execute_async(
        self,
        async_func: Callable,
        tasks: List[Any],
        task_ids: List[str]
    ) -> List[ProcessingResult]:
        """Execute async tasks concurrently."""
        results = []
        
        async def execute_single_task(task, task_id: str, index: int):
            start_time = time.time()
            
            try:
                if isinstance(task, tuple):
                    result = await async_func(*task)
                else:
                    result = await async_func(task)
                
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task_id,
                    result=result,
                    execution_time=execution_time,
                    worker_id=f"async-{id(asyncio.current_task())}",
                    metadata={"strategy": "async", "task_index": index}
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                return ProcessingResult(
                    task_id=task_id,
                    result=None,
                    execution_time=execution_time,
                    error=e,
                    worker_id=f"async-{id(asyncio.current_task())}",
                    metadata={"strategy": "async", "task_index": index}
                )
        
        # Create coroutines
        coroutines = [
            execute_single_task(task, task_id, i)
            for i, (task, task_id) in enumerate(zip(tasks, task_ids))
        ]
        
        # Execute with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def bounded_task(coro):
            async with semaphore:
                return await coro
        
        bounded_coroutines = [bounded_task(coro) for coro in coroutines]
        
        # Wait for completion
        if self.timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*bounded_coroutines, return_exceptions=True),
                timeout=self.timeout
            )
        else:
            results = await asyncio.gather(*bounded_coroutines, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ProcessingResult(
                    task_id=task_ids[i],
                    result=None,
                    execution_time=0.0,
                    error=result,
                    metadata={"strategy": "async", "task_index": i}
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def process_batch(
        self,
        func: Callable,
        tasks: List[Any],
        task_ids: Optional[List[str]] = None,
        strategy: Optional[ProcessingStrategy] = None
    ) -> List[ProcessingResult]:
        """Process batch of tasks with specified or adaptive strategy."""
        
        if not tasks:
            return []
        
        # Generate task IDs if not provided
        if task_ids is None:
            task_ids = [f"task_{i}" for i in range(len(tasks))]
        elif len(task_ids) != len(tasks):
            raise ValueError("task_ids length must match tasks length")
        
        # Determine strategy
        if strategy is None:
            strategy = self.strategy
            
        if strategy == ProcessingStrategy.ADAPTIVE:
            strategy = self.workload_analyzer.analyze_task(func, tasks[:min(4, len(tasks))])
        
        # Record strategy performance
        start_time = time.time()
        
        # Execute based on strategy
        try:
            if strategy == ProcessingStrategy.SEQUENTIAL:
                results = self._execute_sequential(func, tasks, task_ids)
            elif strategy == ProcessingStrategy.THREADED:
                results = self._execute_threaded(func, tasks, task_ids)
            elif strategy == ProcessingStrategy.MULTIPROCESS:
                results = self._execute_multiprocess(func, tasks, task_ids)
            elif strategy == ProcessingStrategy.ASYNC:
                # Handle async execution
                if asyncio.iscoroutinefunction(func):
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    results = loop.run_until_complete(
                        self._execute_async(func, tasks, task_ids)
                    )
                else:
                    self.logger.warning("Function is not async, falling back to threaded")
                    results = self._execute_threaded(func, tasks, task_ids)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Record performance
            total_time = time.time() - start_time
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = []
            self.strategy_performance[strategy].append(total_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed with strategy {strategy}: {e}")
            # Fallback to sequential
            if strategy != ProcessingStrategy.SEQUENTIAL:
                return self._execute_sequential(func, tasks, task_ids)
            raise
    
    def process_stream(
        self,
        func: Callable,
        task_stream: Iterator[Any],
        buffer_size: int = 100,
        strategy: Optional[ProcessingStrategy] = None
    ) -> Iterator[ProcessingResult]:
        """Process stream of tasks with buffering."""
        
        buffer = []
        task_counter = 0
        
        for task in task_stream:
            buffer.append(task)
            task_counter += 1
            
            if len(buffer) >= buffer_size:
                # Process buffer
                task_ids = [f"stream_task_{i}" for i in range(task_counter - len(buffer), task_counter)]
                results = self.process_batch(func, buffer, task_ids, strategy)
                
                for result in results:
                    yield result
                
                buffer.clear()
        
        # Process remaining tasks in buffer
        if buffer:
            task_ids = [f"stream_task_{i}" for i in range(task_counter - len(buffer), task_counter)]
            results = self.process_batch(func, buffer, task_ids, strategy)
            
            for result in results:
                yield result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "strategies_tested": list(self.strategy_performance.keys()),
            "strategy_performance": {}
        }
        
        for strategy, times in self.strategy_performance.items():
            if times:
                stats["strategy_performance"][strategy.value] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }
        
        return stats


def parallel_map(
    func: Callable,
    tasks: List[Any],
    strategy: str = "adaptive",
    max_workers: Optional[int] = None
) -> List[Any]:
    """Convenient parallel map function."""
    
    processor = ParallelProcessor(
        max_workers=max_workers,
        strategy=ProcessingStrategy(strategy)
    )
    
    results = processor.process_batch(func, tasks)
    
    # Extract just the results, raise exceptions
    final_results = []
    for result in results:
        if result.error:
            raise result.error
        final_results.append(result.result)
    
    return final_results


if __name__ == "__main__":
    # Test parallel processing
    def cpu_intensive_task(n: int) -> int:
        """Simulate CPU intensive work."""
        result = 0
        for i in range(n * 100000):
            result += i * i
        return result
    
    def io_intensive_task(delay: float) -> str:
        """Simulate I/O intensive work."""
        time.sleep(delay)
        return f"completed after {delay}s"
    
    # Test different strategies
    processor = ParallelProcessor()
    
    # CPU intensive tasks
    cpu_tasks = [1000 + i * 100 for i in range(8)]
    print("Testing CPU intensive tasks...")
    
    start = time.time()
    cpu_results = processor.process_batch(cpu_intensive_task, cpu_tasks)
    cpu_time = time.time() - start
    
    print(f"CPU tasks completed in {cpu_time:.2f}s")
    print(f"Success rate: {sum(1 for r in cpu_results if not r.error) / len(cpu_results):.1%}")
    
    # I/O intensive tasks  
    io_tasks = [0.1] * 10
    print("\nTesting I/O intensive tasks...")
    
    start = time.time()
    io_results = processor.process_batch(io_intensive_task, io_tasks)
    io_time = time.time() - start
    
    print(f"I/O tasks completed in {io_time:.2f}s")
    print(f"Success rate: {sum(1 for r in io_results if not r.error) / len(io_results):.1%}")
    
    # Performance stats
    print("\nPerformance Stats:")
    stats = processor.get_performance_stats()
    for strategy, perf in stats["strategy_performance"].items():
        print(f"  {strategy}: {perf['avg_time']:.3f}s avg ({perf['count']} runs)")