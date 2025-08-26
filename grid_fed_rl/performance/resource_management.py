"""Advanced memory and resource management with connection pooling and leak detection."""

import asyncio
import time
import threading
import logging
import gc
import weakref
import psutil
import sqlite3
import socket
import functools
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, ContextManager
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from contextlib import contextmanager, asynccontextmanager
import concurrent.futures
from pathlib import Path
import tempfile
import mmap
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    pymongo = None
    MONGODB_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of managed resources."""
    DATABASE_CONNECTION = "database_connection"
    HTTP_CONNECTION = "http_connection"
    FILE_HANDLE = "file_handle"
    MEMORY_BUFFER = "memory_buffer"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    SOCKET = "socket"
    CACHE_ENTRY = "cache_entry"
    CUSTOM = "custom"


class PoolStrategy(Enum):
    """Connection pooling strategies."""
    FIFO = "fifo"          # First In, First Out
    LIFO = "lifo"          # Last In, First Out (stack-like)
    LEAST_USED = "least_used"    # Least recently used
    ROUND_ROBIN = "round_robin"   # Round-robin selection
    ADAPTIVE = "adaptive"   # Adaptive based on performance


@dataclass
class ResourceMetrics:
    """Metrics for resource usage."""
    resource_id: str
    resource_type: ResourceType
    created_at: float
    last_used: float
    use_count: int
    total_time_active: float
    peak_memory_usage: int = 0
    error_count: int = 0
    is_active: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent_used: float
    available_mb: float
    swap_used_mb: float = 0.0
    gc_count: Dict[int, int] = field(default_factory=dict)
    top_objects: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class PoolConfig:
    """Configuration for resource pools."""
    min_connections: int = 1
    max_connections: int = 10
    idle_timeout: float = 300.0  # 5 minutes
    max_lifetime: float = 3600.0  # 1 hour
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0
    strategy: PoolStrategy = PoolStrategy.LIFO
    auto_scaling: bool = True
    monitoring_enabled: bool = True


class MemoryTracker:
    """Tracks memory usage and detects leaks."""
    
    def __init__(self, sampling_interval: float = 10.0, history_size: int = 1000):
        self.sampling_interval = sampling_interval
        self.memory_history = deque(maxlen=history_size)
        self.object_tracker: Dict[type, int] = defaultdict(int)
        self.tracked_objects: Dict[int, weakref.ref] = {}
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Leak detection
        self.baseline_memory = 0.0
        self.leak_threshold = 100.0  # MB
        self.suspected_leaks: List[Dict[str, Any]] = []
        
    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.baseline_memory = self._get_current_memory().rss_mb
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Memory tracking started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect memory stats
                stats = self._get_current_memory()
                self.memory_history.append(stats)
                
                # Check for memory leaks
                self._check_for_leaks(stats)
                
                # Update object tracking
                self._update_object_tracking()
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.sampling_interval)
    
    def _get_current_memory(self) -> MemoryStats:
        """Get current memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        # Get garbage collection stats
        gc_stats = {i: gc.get_count()[i] if i < len(gc.get_count()) else 0 for i in range(3)}
        
        return MemoryStats(
            timestamp=time.time(),
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent_used=memory_percent,
            available_mb=virtual_memory.available / (1024 * 1024),
            swap_used_mb=swap_memory.used / (1024 * 1024),
            gc_count=gc_stats
        )
    
    def _check_for_leaks(self, current_stats: MemoryStats) -> None:
        """Check for potential memory leaks."""
        if len(self.memory_history) < 10:
            return
        
        # Check if memory usage is consistently increasing
        recent_memory = [stats.rss_mb for stats in list(self.memory_history)[-10:]]
        
        if len(recent_memory) < 2:
            return
        
        # Simple trend detection
        trend = (recent_memory[-1] - recent_memory[0]) / len(recent_memory)
        
        # If memory is increasing consistently and significantly
        if (trend > 1.0 and  # > 1MB per sample on average
            current_stats.rss_mb - self.baseline_memory > self.leak_threshold):
            
            leak_info = {
                'detected_at': time.time(),
                'memory_growth_mb': current_stats.rss_mb - self.baseline_memory,
                'trend_mb_per_sample': trend,
                'current_memory_mb': current_stats.rss_mb,
                'gc_stats': current_stats.gc_count,
                'top_objects': self._get_top_objects()
            }
            
            self.suspected_leaks.append(leak_info)
            logger.warning(f"Potential memory leak detected: {leak_info['memory_growth_mb']:.1f}MB growth")
    
    def _update_object_tracking(self) -> None:
        """Update object count tracking."""
        # Track object counts by type
        for obj_type in [dict, list, tuple, str, int, float]:
            count = len(gc.get_objects())  # Simplified - would filter by type in practice
            self.object_tracker[obj_type] = count
    
    def _get_top_objects(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top objects by count/memory usage."""
        # This is a simplified implementation
        # In practice, would use more sophisticated object tracking
        return [("dict", self.object_tracker.get(dict, 0)),
                ("list", self.object_tracker.get(list, 0)),
                ("tuple", self.object_tracker.get(tuple, 0))][:limit]
    
    def track_object(self, obj: Any, identifier: Optional[str] = None) -> str:
        """Track a specific object for leak detection."""
        obj_id = identifier or f"obj_{id(obj)}"
        self.tracked_objects[hash(obj_id)] = weakref.ref(obj)
        return obj_id
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return stats."""
        collected = {}
        for generation in range(3):
            collected[f"gen_{generation}"] = gc.collect(generation)
        
        logger.info(f"Forced GC collected: {collected}")
        return collected
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        if not self.memory_history:
            return {'error': 'No memory data available'}
        
        current_stats = self.memory_history[-1]
        memory_values = [stats.rss_mb for stats in self.memory_history]
        
        return {
            'current_memory_mb': current_stats.rss_mb,
            'baseline_memory_mb': self.baseline_memory,
            'memory_growth_mb': current_stats.rss_mb - self.baseline_memory,
            'peak_memory_mb': max(memory_values),
            'average_memory_mb': sum(memory_values) / len(memory_values),
            'memory_trend_mb_per_min': self._calculate_memory_trend(),
            'gc_stats': current_stats.gc_count,
            'suspected_leaks_count': len(self.suspected_leaks),
            'tracked_objects_count': len([ref for ref in self.tracked_objects.values() if ref() is not None]),
            'monitoring_active': self.monitoring_active,
            'timestamp': current_stats.timestamp
        }
    
    def _calculate_memory_trend(self) -> float:
        """Calculate memory usage trend."""
        if len(self.memory_history) < 2:
            return 0.0
        
        memory_values = [stats.rss_mb for stats in self.memory_history]
        time_values = [stats.timestamp for stats in self.memory_history]
        
        # Simple linear regression
        n = len(memory_values)
        if n < 2:
            return 0.0
        
        sum_x = sum(time_values)
        sum_y = sum(memory_values)
        sum_xy = sum(time_values[i] * memory_values[i] for i in range(n))
        sum_x2 = sum(t * t for t in time_values)
        
        # Slope in MB per second, converted to MB per minute
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope * 60  # Convert to per minute
        
        return 0.0


class ConnectionPool:
    """Generic connection pool with advanced features."""
    
    def __init__(
        self,
        connection_factory: Callable,
        config: PoolConfig,
        resource_type: ResourceType = ResourceType.CUSTOM
    ):
        self.connection_factory = connection_factory
        self.config = config
        self.resource_type = resource_type
        
        # Pool state
        self.available_connections: deque = deque()
        self.active_connections: Dict[str, Any] = {}
        self.connection_metrics: Dict[str, ResourceMetrics] = {}
        
        # Synchronization
        self.pool_lock = threading.RLock()
        self.condition = threading.Condition(self.pool_lock)
        
        # Health monitoring
        self.health_check_thread: Optional[threading.Thread] = None
        self.health_monitoring_active = False
        
        # Statistics
        self.pool_stats = {
            'connections_created': 0,
            'connections_destroyed': 0,
            'connections_reused': 0,
            'total_requests': 0,
            'failed_requests': 0,
            'average_wait_time': 0.0
        }
        
        # Initialize minimum connections
        self._initialize_minimum_connections()
        
        # Start health monitoring
        if config.monitoring_enabled:
            self._start_health_monitoring()
    
    def _initialize_minimum_connections(self) -> None:
        """Initialize minimum number of connections."""
        with self.pool_lock:
            for _ in range(self.config.min_connections):
                try:
                    conn = self._create_connection()
                    if conn:
                        self.available_connections.append(conn)
                except Exception as e:
                    logger.error(f"Failed to create initial connection: {e}")
    
    def _create_connection(self) -> Any:
        """Create a new connection."""
        try:
            connection = self.connection_factory()
            conn_id = f"conn_{id(connection)}"
            
            # Track metrics
            self.connection_metrics[conn_id] = ResourceMetrics(
                resource_id=conn_id,
                resource_type=self.resource_type,
                created_at=time.time(),
                last_used=time.time(),
                use_count=0,
                total_time_active=0.0
            )
            
            self.pool_stats['connections_created'] += 1
            logger.debug(f"Created new connection: {conn_id}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self.pool_stats['failed_requests'] += 1
            return None
    
    def _destroy_connection(self, connection: Any) -> None:
        """Destroy a connection."""
        try:
            conn_id = f"conn_{id(connection)}"
            
            # Clean up connection
            if hasattr(connection, 'close'):
                connection.close()
            elif hasattr(connection, 'disconnect'):
                connection.disconnect()
            
            # Remove metrics
            if conn_id in self.connection_metrics:
                del self.connection_metrics[conn_id]
            
            self.pool_stats['connections_destroyed'] += 1
            logger.debug(f"Destroyed connection: {conn_id}")
            
        except Exception as e:
            logger.error(f"Error destroying connection: {e}")
    
    def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if connection is healthy."""
        try:
            # Generic health check - specific implementations should override
            if hasattr(connection, 'ping'):
                return connection.ping()
            elif hasattr(connection, 'is_connected'):
                return connection.is_connected()
            else:
                # Assume healthy if no explicit health check
                return True
        except Exception:
            return False
    
    def _start_health_monitoring(self) -> None:
        """Start health monitoring thread."""
        if self.health_monitoring_active:
            return
        
        self.health_monitoring_active = True
        self.health_check_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        self.health_check_thread.start()
    
    def _health_monitoring_loop(self) -> None:
        """Health monitoring loop."""
        while self.health_monitoring_active:
            try:
                self._perform_health_checks()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.config.health_check_interval)
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on connections."""
        with self.pool_lock:
            # Check available connections
            healthy_connections = deque()
            
            while self.available_connections:
                conn = self.available_connections.popleft()
                if self._is_connection_healthy(conn):
                    healthy_connections.append(conn)
                else:
                    self._destroy_connection(conn)
            
            self.available_connections = healthy_connections
            
            # Check active connections
            unhealthy_active = []
            for conn_id, conn in self.active_connections.items():
                if not self._is_connection_healthy(conn):
                    unhealthy_active.append(conn_id)
            
            for conn_id in unhealthy_active:
                conn = self.active_connections.pop(conn_id)
                self._destroy_connection(conn)
    
    @contextmanager
    def get_connection(self, timeout: Optional[float] = None):
        """Get a connection from the pool."""
        start_time = time.time()
        connection = None
        
        try:
            # Get connection with timeout
            connection = self._acquire_connection(timeout)
            if not connection:
                raise TimeoutError("Failed to acquire connection within timeout")
            
            # Update statistics
            wait_time = time.time() - start_time
            self.pool_stats['total_requests'] += 1
            
            # Update average wait time
            total_wait = self.pool_stats['average_wait_time'] * (self.pool_stats['total_requests'] - 1)
            self.pool_stats['average_wait_time'] = (total_wait + wait_time) / self.pool_stats['total_requests']
            
            yield connection
            
        except Exception as e:
            self.pool_stats['failed_requests'] += 1
            logger.error(f"Connection error: {e}")
            raise
        finally:
            if connection:
                self._release_connection(connection)
    
    def _acquire_connection(self, timeout: Optional[float] = None) -> Any:
        """Acquire a connection from the pool."""
        deadline = time.time() + timeout if timeout else None
        
        with self.condition:
            while True:
                # Try to get available connection
                if self.available_connections:
                    connection = self.available_connections.popleft()
                    conn_id = f"conn_{id(connection)}"
                    
                    # Move to active connections
                    self.active_connections[conn_id] = connection
                    
                    # Update metrics
                    if conn_id in self.connection_metrics:
                        metrics = self.connection_metrics[conn_id]
                        metrics.last_used = time.time()
                        metrics.use_count += 1
                        metrics.is_active = True
                        self.pool_stats['connections_reused'] += 1
                    
                    return connection
                
                # Try to create new connection if under max limit
                if len(self.active_connections) < self.config.max_connections:
                    connection = self._create_connection()
                    if connection:
                        conn_id = f"conn_{id(connection)}"
                        self.active_connections[conn_id] = connection
                        
                        if conn_id in self.connection_metrics:
                            self.connection_metrics[conn_id].is_active = True
                        
                        return connection
                
                # Wait for connection to become available
                if deadline and time.time() >= deadline:
                    return None
                
                wait_time = (deadline - time.time()) if deadline else self.config.idle_timeout
                if not self.condition.wait(wait_time):
                    return None  # Timeout
    
    def _release_connection(self, connection: Any) -> None:
        """Release a connection back to the pool."""
        with self.condition:
            conn_id = f"conn_{id(connection)}"
            
            # Remove from active connections
            if conn_id in self.active_connections:
                del self.active_connections[conn_id]
            
            # Update metrics
            if conn_id in self.connection_metrics:
                metrics = self.connection_metrics[conn_id]
                metrics.is_active = False
                metrics.total_time_active += time.time() - metrics.last_used
            
            # Check if connection should be destroyed
            should_destroy = False
            
            if conn_id in self.connection_metrics:
                metrics = self.connection_metrics[conn_id]
                connection_age = time.time() - metrics.created_at
                
                if connection_age > self.config.max_lifetime:
                    should_destroy = True
                elif not self._is_connection_healthy(connection):
                    should_destroy = True
            
            if should_destroy:
                self._destroy_connection(connection)
            else:
                # Return to available pool
                if self.config.strategy == PoolStrategy.LIFO:
                    self.available_connections.append(connection)
                else:  # FIFO or others
                    self.available_connections.appendleft(connection)
            
            # Notify waiting threads
            self.condition.notify()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.pool_lock:
            active_count = len(self.active_connections)
            available_count = len(self.available_connections)
            total_count = active_count + available_count
            
            return {
                'active_connections': active_count,
                'available_connections': available_count,
                'total_connections': total_count,
                'max_connections': self.config.max_connections,
                'min_connections': self.config.min_connections,
                'utilization': active_count / max(self.config.max_connections, 1),
                'pool_stats': self.pool_stats.copy(),
                'health_monitoring_active': self.health_monitoring_active,
                'config': {
                    'strategy': self.config.strategy.value,
                    'idle_timeout': self.config.idle_timeout,
                    'max_lifetime': self.config.max_lifetime
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown the connection pool."""
        logger.info("Shutting down connection pool")
        
        # Stop health monitoring
        self.health_monitoring_active = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
        
        with self.pool_lock:
            # Close all connections
            while self.available_connections:
                conn = self.available_connections.popleft()
                self._destroy_connection(conn)
            
            for conn in self.active_connections.values():
                self._destroy_connection(conn)
            
            self.active_connections.clear()
            self.connection_metrics.clear()


class DatabaseConnectionPool(ConnectionPool):
    """Specialized connection pool for database connections."""
    
    def __init__(self, database_url: str, config: PoolConfig):
        def sqlite_factory():
            return sqlite3.connect(database_url, check_same_thread=False)
        
        super().__init__(sqlite_factory, config, ResourceType.DATABASE_CONNECTION)
    
    def _is_connection_healthy(self, connection: sqlite3.Connection) -> bool:
        """Check if SQLite connection is healthy."""
        try:
            connection.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False


class RedisConnectionPool(ConnectionPool):
    """Redis connection pool."""
    
    def __init__(self, redis_config: Dict[str, Any], config: PoolConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        def redis_factory():
            return redis.Redis(**redis_config)
        
        super().__init__(redis_factory, config, ResourceType.DATABASE_CONNECTION)
    
    def _is_connection_healthy(self, connection) -> bool:
        """Check if Redis connection is healthy."""
        try:
            connection.ping()
            return True
        except Exception:
            return False


class HTTPConnectionPool(ConnectionPool):
    """HTTP connection pool using aiohttp."""
    
    def __init__(self, config: PoolConfig):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp not available")
        
        def http_factory():
            return aiohttp.ClientSession()
        
        super().__init__(http_factory, config, ResourceType.HTTP_CONNECTION)
    
    def _is_connection_healthy(self, connection) -> bool:
        """Check if HTTP session is healthy."""
        return not connection.closed


class MemoryPool:
    """Memory buffer pool for efficient memory management."""
    
    def __init__(self, buffer_sizes: List[int], pool_size_per_buffer: int = 10):
        self.buffer_pools: Dict[int, deque] = {}
        self.allocated_buffers: Dict[int, Any] = {}
        self.pool_lock = threading.RLock()
        self.allocation_stats = defaultdict(int)
        
        # Initialize buffer pools
        for size in buffer_sizes:
            self.buffer_pools[size] = deque()
            # Pre-allocate some buffers
            for _ in range(pool_size_per_buffer):
                buffer = bytearray(size)
                self.buffer_pools[size].append(buffer)
    
    @contextmanager
    def get_buffer(self, size: int):
        """Get a memory buffer from the pool."""
        buffer = None
        actual_size = None
        
        try:
            with self.pool_lock:
                # Find the smallest buffer that fits
                available_sizes = [s for s in self.buffer_pools.keys() if s >= size]
                
                if available_sizes:
                    actual_size = min(available_sizes)
                    if self.buffer_pools[actual_size]:
                        buffer = self.buffer_pools[actual_size].popleft()
                        buffer_id = id(buffer)
                        self.allocated_buffers[buffer_id] = (buffer, actual_size)
                        self.allocation_stats[actual_size] += 1
                
                # Create new buffer if none available
                if buffer is None:
                    buffer = bytearray(size)
                    actual_size = size
                    buffer_id = id(buffer)
                    self.allocated_buffers[buffer_id] = (buffer, actual_size)
            
            yield buffer
            
        finally:
            if buffer is not None:
                with self.pool_lock:
                    buffer_id = id(buffer)
                    if buffer_id in self.allocated_buffers:
                        _, original_size = self.allocated_buffers[buffer_id]
                        del self.allocated_buffers[buffer_id]
                        
                        # Return to appropriate pool
                        if original_size in self.buffer_pools and len(self.buffer_pools[original_size]) < 20:
                            # Clear buffer before returning
                            buffer[:] = b'\x00' * len(buffer)
                            self.buffer_pools[original_size].append(buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.pool_lock:
            total_allocated = len(self.allocated_buffers)
            total_available = sum(len(pool) for pool in self.buffer_pools.values())
            
            return {
                'allocated_buffers': total_allocated,
                'available_buffers': total_available,
                'buffer_sizes': list(self.buffer_pools.keys()),
                'allocation_stats': dict(self.allocation_stats),
                'pool_sizes': {size: len(pool) for size, pool in self.buffer_pools.items()}
            }


class ResourceManager:
    """Main resource management system."""
    
    def __init__(self):
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.memory_tracker = MemoryTracker()
        self.memory_pool = MemoryPool([1024, 4096, 16384, 65536])  # Common buffer sizes
        self.resource_registry: Dict[str, Any] = {}
        self.cleanup_callbacks: List[Callable] = []
        
        # Background cleanup
        self.cleanup_active = False
        self.cleanup_thread: Optional[threading.Thread] = None
        
    def start_resource_management(self) -> None:
        """Start resource management systems."""
        self.memory_tracker.start_monitoring()
        
        self.cleanup_active = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Resource management started")
    
    def stop_resource_management(self) -> None:
        """Stop resource management systems."""
        self.cleanup_active = False
        self.memory_tracker.stop_monitoring()
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        # Shutdown all connection pools
        for pool in self.connection_pools.values():
            pool.shutdown()
        
        logger.info("Resource management stopped")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.cleanup_active:
            try:
                # Run cleanup callbacks
                for callback in self.cleanup_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Cleanup callback error: {e}")
                
                # Force garbage collection periodically
                if len(self.memory_tracker.memory_history) % 60 == 0:  # Every 10 minutes
                    self.memory_tracker.force_garbage_collection()
                
                time.sleep(10)  # Cleanup every 10 seconds
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                time.sleep(30)
    
    def create_database_pool(
        self,
        name: str,
        database_url: str,
        config: Optional[PoolConfig] = None
    ) -> DatabaseConnectionPool:
        """Create a database connection pool."""
        config = config or PoolConfig()
        pool = DatabaseConnectionPool(database_url, config)
        self.connection_pools[name] = pool
        return pool
    
    def create_redis_pool(
        self,
        name: str,
        redis_config: Dict[str, Any],
        config: Optional[PoolConfig] = None
    ) -> RedisConnectionPool:
        """Create a Redis connection pool."""
        config = config or PoolConfig()
        pool = RedisConnectionPool(redis_config, config)
        self.connection_pools[name] = pool
        return pool
    
    def create_http_pool(
        self,
        name: str,
        config: Optional[PoolConfig] = None
    ) -> HTTPConnectionPool:
        """Create an HTTP connection pool."""
        config = config or PoolConfig()
        pool = HTTPConnectionPool(config)
        self.connection_pools[name] = pool
        return pool
    
    def get_connection_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name."""
        return self.connection_pools.get(name)
    
    def register_resource(self, name: str, resource: Any) -> None:
        """Register a resource for management."""
        self.resource_registry[name] = resource
        
        # Track in memory monitor if it's a significant object
        self.memory_tracker.track_object(resource, name)
    
    def get_resource(self, name: str) -> Any:
        """Get a registered resource."""
        return self.resource_registry.get(name)
    
    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add a cleanup callback."""
        self.cleanup_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive resource management status."""
        return {
            'memory_tracker': self.memory_tracker.get_memory_report(),
            'memory_pool': self.memory_pool.get_stats(),
            'connection_pools': {
                name: pool.get_pool_stats() 
                for name, pool in self.connection_pools.items()
            },
            'registered_resources': len(self.resource_registry),
            'cleanup_callbacks': len(self.cleanup_callbacks),
            'resource_management_active': self.cleanup_active,
            'timestamp': time.time()
        }


# Global resource manager
global_resource_manager = ResourceManager()


def get_database_pool(name: str) -> Optional[ConnectionPool]:
    """Get database connection pool."""
    return global_resource_manager.get_connection_pool(name)


def get_memory_buffer(size: int):
    """Get memory buffer from global pool."""
    return global_resource_manager.memory_pool.get_buffer(size)


def track_memory_usage():
    """Decorator to track memory usage of functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Track object for potential leaks
            result = func(*args, **kwargs)
            if result is not None:
                global_resource_manager.memory_tracker.track_object(result, func.__name__)
            return result
        return wrapper
    return decorator


def get_resource_status() -> Dict[str, Any]:
    """Get global resource management status."""
    return global_resource_manager.get_system_status()


# Context managers for easy resource management
@contextmanager
def managed_database_connection(pool_name: str):
    """Context manager for database connections."""
    pool = get_database_pool(pool_name)
    if not pool:
        raise ValueError(f"Database pool '{pool_name}' not found")
    
    with pool.get_connection() as conn:
        yield conn


@asynccontextmanager
async def managed_http_session(pool_name: str):
    """Async context manager for HTTP sessions."""
    pool = global_resource_manager.get_connection_pool(pool_name)
    if not pool or not isinstance(pool, HTTPConnectionPool):
        raise ValueError(f"HTTP pool '{pool_name}' not found")
    
    # This would need to be adapted for async usage
    # For now, this is a placeholder
    async with aiohttp.ClientSession() as session:
        yield session