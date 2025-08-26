"""
Enhanced Performance Optimization and Scaling System for Grid-Fed-RL-Gym.

This module provides comprehensive performance optimization capabilities including:
- Multi-level caching with intelligent invalidation
- Enhanced parallel processing with GPU acceleration
- Auto-scaling with container orchestration
- Performance monitoring and benchmarking
- Memory and resource management

Usage Examples:
    # Basic caching
    from grid_fed_rl.performance import cached_multi_level, invalidate_cache
    
    @cached_multi_level(ttl=300, tags={'power_flow'})
    def compute_power_flow(grid_state):
        # Expensive computation
        return result
    
    # Parallel processing
    from grid_fed_rl.performance import parallel_map, ProcessingMode
    
    results = parallel_map(
        compute_function, 
        data_items, 
        processing_mode=ProcessingMode.GPU_PARALLEL
    )
    
    # Auto-scaling
    from grid_fed_rl.performance import create_kubernetes_auto_scaler
    
    auto_scaler = create_kubernetes_auto_scaler(
        image="grid-fed-rl:latest",
        service_name="grid-simulation",
        min_instances=2,
        max_instances=20
    )
    auto_scaler.start_auto_scaling()
    
    # Performance monitoring
    from grid_fed_rl.performance import monitor_performance, get_performance_report
    
    @monitor_performance()
    def my_function():
        # Function will be automatically monitored
        pass
    
    # Resource management
    from grid_fed_rl.performance import managed_database_connection
    
    with managed_database_connection('main_db') as conn:
        # Use connection from pool
        pass
"""

# Import all major components
try:
    from .multi_level_cache import (
        MultiLevelCache,
        CacheConfiguration,
        CacheLevel,
        CompressionType,
        InvalidationStrategy,
        cached_multi_level,
        invalidate_cache,
        get_cache_stats as get_multi_cache_stats,
        global_multi_cache
    )
    MULTI_LEVEL_CACHE_AVAILABLE = True
except ImportError as e:
    MULTI_LEVEL_CACHE_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Multi-level cache not available: {e}")

try:
    from .enhanced_parallel_processor import (
        EnhancedParallelProcessor,
        ProcessingMode,
        WorkloadType,
        GPUManager,
        parallel_map as enhanced_parallel_map,
        parallel_map_async,
        gpu_parallel,
        async_parallel,
        get_processing_stats,
        global_processor
    )
    ENHANCED_PARALLEL_AVAILABLE = True
except ImportError as e:
    ENHANCED_PARALLEL_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Enhanced parallel processing not available: {e}")

try:
    from .auto_scaling_system import (
        AutoScalingSystem,
        ContainerOrchestrator,
        ContainerSpec,
        ScalingThresholds,
        OrchestrationPlatform,
        ScalingPolicy,
        create_docker_auto_scaler,
        create_kubernetes_auto_scaler,
        initialize_auto_scaling,
        get_auto_scaling_status,
        global_auto_scaler
    )
    AUTO_SCALING_AVAILABLE = True
except ImportError as e:
    AUTO_SCALING_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Auto-scaling not available: {e}")

try:
    from .monitoring_benchmarking import (
        PerformanceMonitoringSystem,
        BenchmarkSuite,
        MetricType,
        AlertLevel,
        BenchmarkCategory,
        PerformanceMetric,
        BenchmarkResult,
        monitor_performance,
        record_metric,
        get_performance_report,
        global_monitoring
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    MONITORING_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Performance monitoring not available: {e}")

try:
    from .resource_management import (
        ResourceManager,
        MemoryTracker,
        ConnectionPool,
        DatabaseConnectionPool,
        RedisConnectionPool,
        HTTPConnectionPool,
        MemoryPool,
        PoolConfig,
        ResourceType,
        managed_database_connection,
        get_database_pool,
        get_memory_buffer,
        track_memory_usage,
        get_resource_status,
        global_resource_manager
    )
    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    RESOURCE_MANAGEMENT_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Resource management not available: {e}")

# Legacy imports from existing modules
from .caching_engine import (
    AdaptiveCache,
    cached,
    clear_cache,
    get_cache_stats,
    CacheStrategy
)

from .parallel_processor import (
    ParallelProcessor,
    ProcessingStrategy,
    ProcessingResult,
    parallel_map
)

# Integration and convenience functions
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import functools

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Global performance system configuration."""
    # Caching configuration
    enable_multi_level_cache: bool = True
    cache_l1_size: int = 1000
    cache_l2_size: int = 5000
    cache_l3_max_disk_mb: int = 2048
    cache_default_ttl: float = 3600.0
    
    # Parallel processing configuration
    enable_enhanced_parallel: bool = True
    enable_gpu_acceleration: bool = True
    max_cpu_workers: Optional[int] = None
    max_async_concurrent: int = 100
    
    # Auto-scaling configuration
    enable_auto_scaling: bool = False
    auto_scaling_platform: str = "docker"
    min_instances: int = 1
    max_instances: int = 10
    
    # Monitoring configuration
    enable_performance_monitoring: bool = True
    enable_detailed_profiling: bool = False
    monitoring_storage_path: Optional[str] = None
    
    # Resource management configuration
    enable_resource_management: bool = True
    memory_tracking_interval: float = 10.0
    connection_pool_size: int = 10


class PerformanceManager:
    """Central manager for all performance optimization systems."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.initialized = False
        self.systems_status: Dict[str, bool] = {}
        
    def initialize(self) -> None:
        """Initialize all performance systems."""
        if self.initialized:
            logger.warning("Performance manager already initialized")
            return
        
        logger.info("Initializing Grid-Fed-RL performance optimization systems...")
        
        try:
            # Initialize caching system
            if self.config.enable_multi_level_cache and MULTI_LEVEL_CACHE_AVAILABLE:
                self._initialize_caching()
                self.systems_status['caching'] = True
                logger.info("Multi-level caching system initialized")
            
            # Initialize parallel processing
            if self.config.enable_enhanced_parallel and ENHANCED_PARALLEL_AVAILABLE:
                self._initialize_parallel_processing()
                self.systems_status['parallel_processing'] = True
                logger.info("Enhanced parallel processing system initialized")
            
            # Initialize auto-scaling
            if self.config.enable_auto_scaling and AUTO_SCALING_AVAILABLE:
                self._initialize_auto_scaling()
                self.systems_status['auto_scaling'] = True
                logger.info("Auto-scaling system initialized")
            
            # Initialize monitoring
            if self.config.enable_performance_monitoring and MONITORING_AVAILABLE:
                self._initialize_monitoring()
                self.systems_status['monitoring'] = True
                logger.info("Performance monitoring system initialized")
            
            # Initialize resource management
            if self.config.enable_resource_management and RESOURCE_MANAGEMENT_AVAILABLE:
                self._initialize_resource_management()
                self.systems_status['resource_management'] = True
                logger.info("Resource management system initialized")
            
            self.initialized = True
            logger.info("Grid-Fed-RL performance optimization systems fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance systems: {e}")
            raise
    
    def _initialize_caching(self) -> None:
        """Initialize multi-level caching system."""
        # The global cache is already initialized
        if hasattr(global_multi_cache, '_configured'):
            return
        global_multi_cache._configured = True
    
    def _initialize_parallel_processing(self) -> None:
        """Initialize enhanced parallel processing."""
        # The global processor is already initialized
        if self.config.max_cpu_workers and ENHANCED_PARALLEL_AVAILABLE:
            global_processor.max_cpu_workers = self.config.max_cpu_workers
    
    def _initialize_auto_scaling(self) -> None:
        """Initialize auto-scaling system."""
        logger.info(f"Auto-scaling configured for platform: {self.config.auto_scaling_platform}")
    
    def _initialize_monitoring(self) -> None:
        """Initialize performance monitoring."""
        if MONITORING_AVAILABLE:
            global_monitoring.start_monitoring()
    
    def _initialize_resource_management(self) -> None:
        """Initialize resource management."""
        if RESOURCE_MANAGEMENT_AVAILABLE:
            global_resource_manager.start_resource_management()
            global_resource_manager.memory_tracker.sampling_interval = self.config.memory_tracking_interval
    
    def shutdown(self) -> None:
        """Shutdown all performance systems."""
        if not self.initialized:
            return
        
        logger.info("Shutting down performance optimization systems...")
        
        try:
            # Shutdown monitoring
            if self.systems_status.get('monitoring') and MONITORING_AVAILABLE:
                global_monitoring.stop_monitoring()
            
            # Shutdown resource management
            if self.systems_status.get('resource_management') and RESOURCE_MANAGEMENT_AVAILABLE:
                global_resource_manager.stop_resource_management()
            
            # Shutdown parallel processing
            if self.systems_status.get('parallel_processing') and ENHANCED_PARALLEL_AVAILABLE:
                global_processor.shutdown()
            
            # Shutdown caching
            if self.systems_status.get('caching') and MULTI_LEVEL_CACHE_AVAILABLE:
                global_multi_cache.shutdown()
            
            # Shutdown auto-scaling
            if (self.systems_status.get('auto_scaling') and AUTO_SCALING_AVAILABLE and 
                'global_auto_scaler' in globals() and global_auto_scaler):
                global_auto_scaler.stop_auto_scaling()
            
            self.initialized = False
            logger.info("Performance optimization systems shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during performance systems shutdown: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all performance systems."""
        status = {
            'initialized': self.initialized,
            'systems_active': self.systems_status.copy(),
            'systems_available': {
                'multi_level_cache': MULTI_LEVEL_CACHE_AVAILABLE,
                'enhanced_parallel': ENHANCED_PARALLEL_AVAILABLE,
                'auto_scaling': AUTO_SCALING_AVAILABLE,
                'monitoring': MONITORING_AVAILABLE,
                'resource_management': RESOURCE_MANAGEMENT_AVAILABLE
            },
            'timestamp': time.time()
        }
        
        if self.initialized:
            try:
                # Get caching stats
                if self.systems_status.get('caching') and MULTI_LEVEL_CACHE_AVAILABLE:
                    status['caching'] = get_multi_cache_stats()
                
                # Get parallel processing stats
                if self.systems_status.get('parallel_processing') and ENHANCED_PARALLEL_AVAILABLE:
                    status['parallel_processing'] = get_processing_stats()
                
                # Get monitoring stats
                if self.systems_status.get('monitoring') and MONITORING_AVAILABLE:
                    status['monitoring'] = global_monitoring.get_system_status()
                
                # Get resource management stats
                if self.systems_status.get('resource_management') and RESOURCE_MANAGEMENT_AVAILABLE:
                    status['resource_management'] = get_resource_status()
                
                # Get auto-scaling stats
                if (self.systems_status.get('auto_scaling') and AUTO_SCALING_AVAILABLE and
                    'get_auto_scaling_status' in globals()):
                    status['auto_scaling'] = get_auto_scaling_status()
                
            except Exception as e:
                status['error'] = f"Error retrieving system status: {e}"
        
        return status


# Global performance manager instance
global_performance_manager = PerformanceManager()


def initialize_performance_systems(config: Optional[PerformanceConfig] = None) -> None:
    """Initialize all performance optimization systems."""
    if config:
        global global_performance_manager
        global_performance_manager = PerformanceManager(config)
    
    global_performance_manager.initialize()


def shutdown_performance_systems() -> None:
    """Shutdown all performance optimization systems."""
    global_performance_manager.shutdown()


def get_comprehensive_status() -> Dict[str, Any]:
    """Get comprehensive status of all performance systems."""
    return global_performance_manager.get_system_status()


# Convenience decorators and functions that integrate multiple systems
def optimized(
    cache_ttl: Optional[float] = None,
    cache_tags: Optional[set] = None,
    enable_monitoring: bool = True,
    enable_profiling: bool = False
):
    """
    Decorator that applies multiple performance optimizations.
    
    Args:
        cache_ttl: Cache time-to-live in seconds
        cache_tags: Cache tags for invalidation
        enable_monitoring: Enable performance monitoring
        enable_profiling: Enable detailed profiling
    """
    def decorator(func):
        # Apply caching
        if cache_ttl is not None and MULTI_LEVEL_CACHE_AVAILABLE:
            func = cached_multi_level(ttl=cache_ttl, tags=cache_tags)(func)
        elif cache_ttl is not None:
            # Fallback to basic caching
            func = cached(func)
        
        # Apply monitoring
        if enable_monitoring and MONITORING_AVAILABLE:
            func = monitor_performance()(func)
        
        # Apply memory tracking
        if RESOURCE_MANAGEMENT_AVAILABLE:
            func = track_memory_usage()(func)
        
        return func
    
    return decorator


def high_performance_compute(
    data: List[Any],
    compute_func: Callable,
    enable_caching: bool = True,
    cache_ttl: float = 300.0,
    processing_mode = None  # Use string to avoid import issues
) -> List[Any]:
    """
    High-performance computation with automatic optimization.
    
    Args:
        data: Input data list
        compute_func: Function to apply to each data item
        enable_caching: Enable result caching
        cache_ttl: Cache TTL in seconds
        processing_mode: Specific processing mode to use
    
    Returns:
        List of computation results
    """
    # Apply caching if enabled
    if enable_caching and MULTI_LEVEL_CACHE_AVAILABLE:
        compute_func = cached_multi_level(ttl=cache_ttl)(compute_func)
    elif enable_caching:
        compute_func = cached(compute_func)
    
    # Apply monitoring
    if MONITORING_AVAILABLE:
        compute_func = monitor_performance()(compute_func)
    
    # Use parallel processing
    if ENHANCED_PARALLEL_AVAILABLE and processing_mode:
        return enhanced_parallel_map(compute_func, data, processing_mode)
    else:
        # Fallback to basic parallel processing
        return parallel_map(compute_func, data)


# Export lists based on availability
__all__ = [
    # Core classes
    'PerformanceManager',
    'PerformanceConfig',
    
    # Legacy compatibility (always available)
    'AdaptiveCache',
    'ParallelProcessor',
    'cached',
    'clear_cache',
    'get_cache_stats',
    'CacheStrategy',
    'ProcessingStrategy',
    'ProcessingResult',
    'parallel_map',
    
    # Integration functions
    'initialize_performance_systems',
    'shutdown_performance_systems',
    'get_comprehensive_status',
    'optimized',
    'high_performance_compute',
    'global_performance_manager'
]

# Add enhanced features to exports if available
if MULTI_LEVEL_CACHE_AVAILABLE:
    __all__.extend([
        'MultiLevelCache',
        'CacheConfiguration', 
        'CacheLevel',
        'cached_multi_level',
        'invalidate_cache',
        'global_multi_cache'
    ])

if ENHANCED_PARALLEL_AVAILABLE:
    __all__.extend([
        'EnhancedParallelProcessor',
        'ProcessingMode',
        'WorkloadType',
        'gpu_parallel',
        'async_parallel',
        'get_processing_stats',
        'global_processor'
    ])

if AUTO_SCALING_AVAILABLE:
    __all__.extend([
        'AutoScalingSystem',
        'create_docker_auto_scaler',
        'create_kubernetes_auto_scaler',
        'get_auto_scaling_status'
    ])

if MONITORING_AVAILABLE:
    __all__.extend([
        'PerformanceMonitoringSystem',
        'BenchmarkSuite',
        'monitor_performance',
        'record_metric',
        'get_performance_report',
        'global_monitoring'
    ])

if RESOURCE_MANAGEMENT_AVAILABLE:
    __all__.extend([
        'ResourceManager',
        'managed_database_connection',
        'get_database_pool',
        'get_memory_buffer',
        'track_memory_usage',
        'get_resource_status',
        'global_resource_manager'
    ])