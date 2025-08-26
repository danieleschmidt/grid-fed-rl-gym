"""Enhanced parallel processing with GPU acceleration, async patterns, and intelligent work distribution."""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import queue
import logging
import psutil
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import weakref
import functools
import inspect
from contextlib import asynccontextmanager, contextmanager

try:
    import torch
    import cupy as cp
    GPU_AVAILABLE = torch.cuda.is_available() or cp.cuda.is_available()
    CUDA_DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
except ImportError:
    torch = None
    cp = None
    GPU_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    ASYNC_IO = "async_io"
    GPU_PARALLEL = "gpu_parallel"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"


class WorkloadType(Enum):
    """Types of computational workloads."""
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MEMORY_BOUND = "memory_bound"
    GPU_SUITABLE = "gpu_suitable"
    NETWORK_BOUND = "network_bound"
    MIXED = "mixed"


class ResourceType(Enum):
    """Available computational resources."""
    CPU_CORE = "cpu_core"
    GPU_DEVICE = "gpu_device"
    MEMORY_GB = "memory_gb"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DISK_IO = "disk_io"


@dataclass
class GPUInfo:
    """Information about available GPU resources."""
    device_id: int
    name: str
    memory_total: int
    memory_available: int
    compute_capability: Tuple[int, int]
    multi_processor_count: int
    is_available: bool = True
    current_utilization: float = 0.0
    temperature: Optional[float] = None


@dataclass
class TaskMetrics:
    """Metrics for task execution."""
    task_id: str
    execution_time: float
    resource_usage: Dict[str, float]
    success: bool
    error_message: Optional[str] = None
    worker_id: Optional[str] = None
    processing_mode: Optional[ProcessingMode] = None
    memory_peak: float = 0.0
    gpu_memory_used: float = 0.0


@dataclass
class WorkerPool:
    """Configuration for a worker pool."""
    pool_type: ProcessingMode
    max_workers: int
    current_workers: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    resource_allocation: Dict[ResourceType, float] = field(default_factory=dict)
    executor: Optional[Any] = None


class GPUManager:
    """Manages GPU resources and workload distribution."""
    
    def __init__(self):
        self.gpu_devices: List[GPUInfo] = []
        self.device_queues: Dict[int, queue.Queue] = {}
        self.device_locks: Dict[int, threading.Lock] = {}
        self.device_usage_history: Dict[int, deque] = {}
        
        self._discover_gpu_devices()
        self._initialize_device_management()
    
    def _discover_gpu_devices(self) -> None:
        """Discover available GPU devices."""
        if not GPU_AVAILABLE:
            logger.info("No GPU support available")
            return
        
        try:
            if torch and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory
                    memory_available = memory_total - torch.cuda.memory_allocated(i)
                    
                    gpu_info = GPUInfo(
                        device_id=i,
                        name=props.name,
                        memory_total=memory_total,
                        memory_available=memory_available,
                        compute_capability=(props.major, props.minor),
                        multi_processor_count=props.multi_processor_count
                    )
                    
                    self.gpu_devices.append(gpu_info)
                    logger.info(f"Discovered GPU {i}: {props.name}")
            
            elif cp and cp.cuda.is_available():
                # CuPy GPU discovery
                for i in range(cp.cuda.runtime.getDeviceCount()):
                    with cp.cuda.Device(i):
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        memory_total = props['totalGlobalMem']
                        memory_available = memory_total  # Simplified
                        
                        gpu_info = GPUInfo(
                            device_id=i,
                            name=props['name'].decode(),
                            memory_total=memory_total,
                            memory_available=memory_available,
                            compute_capability=(props['major'], props['minor']),
                            multi_processor_count=props['multiProcessorCount']
                        )
                        
                        self.gpu_devices.append(gpu_info)
                        logger.info(f"Discovered GPU {i}: {props['name'].decode()}")
        
        except Exception as e:
            logger.error(f"GPU discovery failed: {e}")
    
    def _initialize_device_management(self) -> None:
        """Initialize GPU device management structures."""
        for gpu in self.gpu_devices:
            device_id = gpu.device_id
            self.device_queues[device_id] = queue.Queue()
            self.device_locks[device_id] = threading.Lock()
            self.device_usage_history[device_id] = deque(maxlen=1000)
    
    def get_optimal_device(self, memory_required: int = 0) -> Optional[int]:
        """Get the optimal GPU device for a task."""
        if not self.gpu_devices:
            return None
        
        # Find device with most available memory and lowest utilization
        best_device = None
        best_score = -1
        
        for gpu in self.gpu_devices:
            if not gpu.is_available:
                continue
            
            if memory_required > 0 and gpu.memory_available < memory_required:
                continue
            
            # Score based on available memory and utilization
            memory_score = gpu.memory_available / gpu.memory_total
            utilization_score = 1.0 - gpu.current_utilization
            
            combined_score = (memory_score + utilization_score) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_device = gpu.device_id
        
        return best_device
    
    def execute_gpu_task(self, device_id: int, func: Callable, *args, **kwargs) -> Any:
        """Execute a task on a specific GPU device."""
        if device_id not in self.device_locks:
            raise ValueError(f"Invalid GPU device ID: {device_id}")
        
        with self.device_locks[device_id]:
            try:
                # Set device context
                if torch and torch.cuda.is_available():
                    with torch.cuda.device(device_id):
                        result = func(*args, **kwargs)
                elif cp and cp.cuda.is_available():
                    with cp.cuda.Device(device_id):
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Update usage stats
                self.device_usage_history[device_id].append({
                    'timestamp': time.time(),
                    'success': True
                })
                
                return result
                
            except Exception as e:
                self.device_usage_history[device_id].append({
                    'timestamp': time.time(),
                    'success': False,
                    'error': str(e)
                })
                raise
    
    def update_device_stats(self) -> None:
        """Update GPU device statistics."""
        for gpu in self.gpu_devices:
            try:
                if torch and torch.cuda.is_available():
                    gpu.memory_available = gpu.memory_total - torch.cuda.memory_allocated(gpu.device_id)
                    gpu.current_utilization = torch.cuda.utilization(gpu.device_id) / 100.0
                    
                elif cp and cp.cuda.is_available():
                    meminfo = cp.cuda.MemoryInfo()
                    gpu.memory_available = meminfo.free
                    
            except Exception as e:
                logger.warning(f"Failed to update GPU {gpu.device_id} stats: {e}")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU statistics."""
        self.update_device_stats()
        
        return {
            'total_devices': len(self.gpu_devices),
            'available_devices': len([g for g in self.gpu_devices if g.is_available]),
            'devices': [
                {
                    'device_id': gpu.device_id,
                    'name': gpu.name,
                    'memory_total_mb': gpu.memory_total / (1024 * 1024),
                    'memory_available_mb': gpu.memory_available / (1024 * 1024),
                    'memory_utilization': 1.0 - (gpu.memory_available / gpu.memory_total),
                    'compute_capability': f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
                    'current_utilization': gpu.current_utilization,
                    'is_available': gpu.is_available
                }
                for gpu in self.gpu_devices
            ]
        }


class WorkloadAnalyzer:
    """Analyzes workload characteristics to optimize processing strategy."""
    
    def __init__(self):
        self.execution_history = deque(maxlen=10000)
        self.workload_patterns = defaultdict(list)
        self.performance_cache = {}
        
    def analyze_function(self, func: Callable, sample_args: Optional[List] = None) -> WorkloadType:
        """Analyze function to determine workload type."""
        # Check function signature and source for patterns
        func_name = func.__name__.lower()
        
        # GPU-suitable patterns
        gpu_patterns = [
            'matrix', 'matmul', 'dot', 'conv', 'fft', 'tensor',
            'neural', 'deep', 'cuda', 'gpu', 'parallel', 'vector'
        ]
        
        # CPU-bound patterns
        cpu_patterns = [
            'compute', 'calculate', 'process', 'algorithm', 'optimize',
            'solve', 'iterate', 'recursive', 'complex'
        ]
        
        # I/O-bound patterns
        io_patterns = [
            'read', 'write', 'load', 'save', 'fetch', 'download',
            'upload', 'request', 'query', 'database', 'file'
        ]
        
        # Network-bound patterns
        network_patterns = [
            'http', 'api', 'rest', 'websocket', 'tcp', 'udp',
            'socket', 'network', 'distributed', 'remote'
        ]
        
        # Score different patterns
        gpu_score = sum(1 for pattern in gpu_patterns if pattern in func_name)
        cpu_score = sum(1 for pattern in cpu_patterns if pattern in func_name)
        io_score = sum(1 for pattern in io_patterns if pattern in func_name)
        network_score = sum(1 for pattern in network_patterns if pattern in func_name)
        
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)
        if is_async:
            return WorkloadType.IO_BOUND
        
        # Determine workload type based on scores
        if gpu_score > 0 and GPU_AVAILABLE:
            return WorkloadType.GPU_SUITABLE
        elif network_score > max(cpu_score, io_score):
            return WorkloadType.NETWORK_BOUND
        elif io_score > cpu_score:
            return WorkloadType.IO_BOUND
        elif cpu_score > 0:
            return WorkloadType.CPU_BOUND
        else:
            return WorkloadType.MIXED
    
    def recommend_processing_mode(
        self, 
        workload_type: WorkloadType,
        batch_size: int = 1,
        available_resources: Optional[Dict[ResourceType, float]] = None
    ) -> ProcessingMode:
        """Recommend optimal processing mode based on workload analysis."""
        
        if workload_type == WorkloadType.GPU_SUITABLE and GPU_AVAILABLE:
            return ProcessingMode.GPU_PARALLEL
        elif workload_type == WorkloadType.IO_BOUND:
            return ProcessingMode.ASYNC_IO if batch_size < 100 else ProcessingMode.THREADED
        elif workload_type == WorkloadType.CPU_BOUND:
            return ProcessingMode.MULTIPROCESS if batch_size > 10 else ProcessingMode.THREADED
        elif workload_type == WorkloadType.NETWORK_BOUND:
            return ProcessingMode.ASYNC_IO
        elif workload_type == WorkloadType.MEMORY_BOUND:
            return ProcessingMode.SEQUENTIAL
        else:
            # Mixed workload - choose based on batch size and resources
            if batch_size > 50:
                return ProcessingMode.MULTIPROCESS
            elif batch_size > 10:
                return ProcessingMode.THREADED
            else:
                return ProcessingMode.SEQUENTIAL
    
    def record_execution(
        self, 
        func_name: str, 
        workload_type: WorkloadType, 
        processing_mode: ProcessingMode,
        execution_time: float,
        success: bool,
        batch_size: int = 1
    ) -> None:
        """Record execution metrics for learning."""
        record = {
            'timestamp': time.time(),
            'func_name': func_name,
            'workload_type': workload_type.value,
            'processing_mode': processing_mode.value,
            'execution_time': execution_time,
            'success': success,
            'batch_size': batch_size,
            'throughput': batch_size / execution_time if execution_time > 0 else 0
        }
        
        self.execution_history.append(record)
        self.workload_patterns[func_name].append(record)
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights from execution history."""
        if not self.execution_history:
            return {}
        
        # Analyze performance by processing mode
        mode_performance = defaultdict(list)
        for record in self.execution_history:
            mode_performance[record['processing_mode']].append(record['throughput'])
        
        # Calculate average throughput by mode
        mode_avg_throughput = {
            mode: np.mean(throughputs) for mode, throughputs in mode_performance.items()
        }
        
        # Find best performing mode
        best_mode = max(mode_avg_throughput.items(), key=lambda x: x[1])[0] if mode_avg_throughput else None
        
        return {
            'total_executions': len(self.execution_history),
            'mode_performance': mode_avg_throughput,
            'best_performing_mode': best_mode,
            'success_rate': sum(1 for r in self.execution_history if r['success']) / len(self.execution_history),
            'average_execution_time': np.mean([r['execution_time'] for r in self.execution_history])
        }


class AsyncWorkQueue:
    """Asynchronous work queue for I/O-bound tasks."""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = set()
        self.completed_tasks = 0
        self.failed_tasks = 0
        
    async def execute(self, coro_func: Callable, *args, **kwargs) -> Any:
        """Execute async function with concurrency control."""
        async with self.semaphore:
            task = asyncio.create_task(coro_func(*args, **kwargs))
            self.active_tasks.add(task)
            
            try:
                result = await task
                self.completed_tasks += 1
                return result
            except Exception as e:
                self.failed_tasks += 1
                raise
            finally:
                self.active_tasks.discard(task)
    
    async def map_async(
        self, 
        coro_func: Callable, 
        items: List[Any],
        return_exceptions: bool = False
    ) -> List[Any]:
        """Async map over items with concurrency control."""
        tasks = [self.execute(coro_func, item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'max_concurrent': self.max_concurrent,
            'success_rate': self.completed_tasks / max(self.completed_tasks + self.failed_tasks, 1)
        }


class EnhancedParallelProcessor:
    """Enhanced parallel processor with GPU acceleration and intelligent scheduling."""
    
    def __init__(
        self, 
        max_cpu_workers: Optional[int] = None,
        max_async_concurrent: int = 100,
        enable_gpu: bool = True,
        enable_distributed: bool = False
    ):
        self.max_cpu_workers = max_cpu_workers or mp.cpu_count()
        self.max_async_concurrent = max_async_concurrent
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.enable_distributed = enable_distributed and RAY_AVAILABLE
        
        # Initialize components
        self.gpu_manager = GPUManager() if self.enable_gpu else None
        self.workload_analyzer = WorkloadAnalyzer()
        self.async_queue = AsyncWorkQueue(max_async_concurrent)
        
        # Worker pools
        self.worker_pools: Dict[ProcessingMode, WorkerPool] = {}
        self._initialize_worker_pools()
        
        # Performance tracking
        self.execution_metrics = deque(maxlen=10000)
        self.resource_monitor = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitoring_active = True
        self.resource_monitor.start()
        
        # Distributed processing
        if self.enable_distributed:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                logger.info("Ray distributed processing initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Ray: {e}")
                self.enable_distributed = False
    
    def _initialize_worker_pools(self) -> None:
        """Initialize worker pools for different processing modes."""
        # Thread pool for I/O-bound tasks
        self.worker_pools[ProcessingMode.THREADED] = WorkerPool(
            pool_type=ProcessingMode.THREADED,
            max_workers=min(50, self.max_cpu_workers * 4),
            executor=concurrent.futures.ThreadPoolExecutor(
                max_workers=min(50, self.max_cpu_workers * 4),
                thread_name_prefix="parallel-io"
            )
        )
        
        # Process pool for CPU-bound tasks
        self.worker_pools[ProcessingMode.MULTIPROCESS] = WorkerPool(
            pool_type=ProcessingMode.MULTIPROCESS,
            max_workers=self.max_cpu_workers,
            executor=concurrent.futures.ProcessPoolExecutor(max_workers=self.max_cpu_workers)
        )
    
    def _monitor_resources(self) -> None:
        """Monitor system resources continuously."""
        while self.monitoring_active:
            try:
                # Update GPU stats
                if self.gpu_manager:
                    self.gpu_manager.update_device_stats()
                
                # Monitor CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Update worker pool stats
                for pool in self.worker_pools.values():
                    if hasattr(pool.executor, '_threads'):
                        # ThreadPoolExecutor
                        pool.current_workers = len(pool.executor._threads)
                    elif hasattr(pool.executor, '_processes'):
                        # ProcessPoolExecutor
                        pool.current_workers = len(pool.executor._processes)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10)
    
    async def process_async(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None,
        processing_mode: Optional[ProcessingMode] = None
    ) -> List[Any]:
        """Process items asynchronously with optimal strategy selection."""
        
        if not items:
            return []
        
        # Analyze workload if not specified
        if processing_mode is None:
            workload_type = self.workload_analyzer.analyze_function(func)
            processing_mode = self.workload_analyzer.recommend_processing_mode(
                workload_type, len(items)
            )
        
        start_time = time.time()
        
        try:
            if processing_mode == ProcessingMode.ASYNC_IO:
                if not inspect.iscoroutinefunction(func):
                    # Wrap sync function for async execution
                    async def async_wrapper(item):
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, func, item)
                    results = await self.async_queue.map_async(async_wrapper, items)
                else:
                    results = await self.async_queue.map_async(func, items)
            
            elif processing_mode == ProcessingMode.GPU_PARALLEL and self.gpu_manager:
                results = await self._process_gpu_parallel(func, items)
            
            elif processing_mode == ProcessingMode.DISTRIBUTED and self.enable_distributed:
                results = await self._process_distributed(func, items)
            
            else:
                # Use thread/process pools
                pool = self.worker_pools.get(processing_mode)
                if not pool or not pool.executor:
                    # Fallback to sequential
                    results = [func(item) for item in items]
                else:
                    loop = asyncio.get_event_loop()
                    futures = [
                        loop.run_in_executor(pool.executor, func, item) 
                        for item in items
                    ]
                    results = await asyncio.gather(*futures)
            
            # Record performance metrics
            execution_time = time.time() - start_time
            self._record_execution_metrics(
                func.__name__, processing_mode, execution_time, 
                len(items), True
            )
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution_metrics(
                func.__name__, processing_mode, execution_time,
                len(items), False, str(e)
            )
            raise
    
    def process_sync(
        self,
        func: Callable,
        items: List[Any],
        processing_mode: Optional[ProcessingMode] = None,
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Synchronous wrapper for async processing."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.process_async(func, items, chunk_size, processing_mode)
        )
    
    async def _process_gpu_parallel(self, func: Callable, items: List[Any]) -> List[Any]:
        """Process items using GPU parallel execution."""
        if not self.gpu_manager:
            raise RuntimeError("GPU manager not available")
        
        # Get optimal GPU device
        device_id = self.gpu_manager.get_optimal_device()
        if device_id is None:
            raise RuntimeError("No available GPU devices")
        
        # Execute on GPU
        def gpu_batch_func(batch_items):
            return [func(item) for item in batch_items]
        
        # Split into batches for GPU processing
        batch_size = min(len(items), 1000)  # Adjust based on GPU memory
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        results = []
        for batch in batches:
            batch_result = await asyncio.get_event_loop().run_in_executor(
                None, self.gpu_manager.execute_gpu_task, device_id, gpu_batch_func, batch
            )
            results.extend(batch_result)
        
        return results
    
    async def _process_distributed(self, func: Callable, items: List[Any]) -> List[Any]:
        """Process items using distributed computing (Ray)."""
        if not self.enable_distributed:
            raise RuntimeError("Distributed processing not enabled")
        
        # Create Ray remote function
        @ray.remote
        def ray_func(item):
            return func(item)
        
        # Submit tasks
        futures = [ray_func.remote(item) for item in items]
        
        # Get results
        results = await asyncio.get_event_loop().run_in_executor(
            None, ray.get, futures
        )
        
        return results
    
    def _record_execution_metrics(
        self,
        func_name: str,
        processing_mode: ProcessingMode,
        execution_time: float,
        batch_size: int,
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """Record execution metrics for analysis."""
        
        metrics = TaskMetrics(
            task_id=f"{func_name}_{int(time.time() * 1000000)}",
            execution_time=execution_time,
            resource_usage={
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent
            },
            success=success,
            error_message=error_message,
            processing_mode=processing_mode
        )
        
        self.execution_metrics.append(metrics)
        
        # Update workload analyzer
        workload_type = self.workload_analyzer.analyze_function(lambda: None)  # Simplified
        self.workload_analyzer.record_execution(
            func_name, workload_type, processing_mode, execution_time, success, batch_size
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        stats = {
            'worker_pools': {
                mode.value: {
                    'max_workers': pool.max_workers,
                    'current_workers': pool.current_workers,
                    'active_tasks': pool.active_tasks,
                    'completed_tasks': pool.completed_tasks,
                    'failed_tasks': pool.failed_tasks,
                    'success_rate': pool.completed_tasks / max(pool.completed_tasks + pool.failed_tasks, 1)
                }
                for mode, pool in self.worker_pools.items()
            },
            'async_queue': self.async_queue.get_status(),
            'system_resources': {
                'cpu_count': mp.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'memory_percent': psutil.virtual_memory().percent
            },
            'execution_metrics': {
                'total_executions': len(self.execution_metrics),
                'avg_execution_time': np.mean([m.execution_time for m in self.execution_metrics]) if self.execution_metrics else 0,
                'success_rate': sum(1 for m in self.execution_metrics if m.success) / max(len(self.execution_metrics), 1)
            },
            'workload_insights': self.workload_analyzer.get_performance_insights()
        }
        
        # Add GPU stats if available
        if self.gpu_manager:
            stats['gpu'] = self.gpu_manager.get_gpu_stats()
        
        return stats
    
    def shutdown(self) -> None:
        """Gracefully shutdown the processor."""
        logger.info("Shutting down enhanced parallel processor...")
        
        self.monitoring_active = False
        
        # Shutdown worker pools
        for pool in self.worker_pools.values():
            if pool.executor:
                pool.executor.shutdown(wait=True)
        
        # Shutdown Ray if enabled
        if self.enable_distributed and ray.is_initialized():
            ray.shutdown()
        
        logger.info("Enhanced parallel processor shutdown complete")


# Global processor instance
global_processor = EnhancedParallelProcessor()


def parallel_map(
    func: Callable,
    items: List[Any],
    processing_mode: Optional[ProcessingMode] = None,
    chunk_size: Optional[int] = None
) -> List[Any]:
    """Convenient parallel map function."""
    return global_processor.process_sync(func, items, processing_mode, chunk_size)


async def parallel_map_async(
    func: Callable,
    items: List[Any],
    processing_mode: Optional[ProcessingMode] = None,
    chunk_size: Optional[int] = None
) -> List[Any]:
    """Convenient async parallel map function."""
    return await global_processor.process_async(func, items, chunk_size, processing_mode)


def gpu_parallel(func: Callable) -> Callable:
    """Decorator to mark function for GPU parallel execution."""
    @functools.wraps(func)
    def wrapper(items: List[Any]) -> List[Any]:
        return global_processor.process_sync(func, items, ProcessingMode.GPU_PARALLEL)
    
    return wrapper


def async_parallel(max_concurrent: int = 100):
    """Decorator for async parallel execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(items: List[Any]) -> List[Any]:
            queue = AsyncWorkQueue(max_concurrent)
            return await queue.map_async(func, items)
        
        return wrapper
    return decorator


def get_processing_stats() -> Dict[str, Any]:
    """Get global processing statistics."""
    return global_processor.get_performance_stats()