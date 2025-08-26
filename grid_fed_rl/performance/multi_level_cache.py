"""Multi-level caching system with intelligent cache invalidation and adaptive sizing."""

import time
import threading
import hashlib
import pickle
import json
import asyncio
import sqlite3
import os
import tempfile
import psutil
import redis
import lz4
import zstd
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import OrderedDict, defaultdict, deque
import logging
from pathlib import Path
import weakref
import concurrent.futures
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Multi-level cache hierarchy."""
    L1_MEMORY = "l1_memory"      # Hot data, fastest access
    L2_MEMORY = "l2_memory"      # Warm data, fast access
    L3_DISK = "l3_disk"          # Cold data, medium access
    L4_DISTRIBUTED = "l4_distributed"  # Shared cache across nodes
    L5_PERSISTENT = "l5_persistent"    # Long-term persistent cache


class CompressionType(Enum):
    """Compression algorithms for cache entries."""
    NONE = "none"
    LZ4 = "lz4"         # Fast compression/decompression
    ZSTD = "zstd"       # High compression ratio
    GZIP = "gzip"       # Standard compression
    PICKLE = "pickle"   # Python object serialization


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""
    TIME_BASED = "time_based"       # TTL-based invalidation
    VERSION_BASED = "version_based" # Version number invalidation
    TAG_BASED = "tag_based"         # Tag-based invalidation
    DEPENDENCY_BASED = "dependency_based"  # Dependency graph invalidation
    PATTERN_BASED = "pattern_based" # Pattern matching invalidation


@dataclass
class CacheConfiguration:
    """Configuration for multi-level cache."""
    # Memory cache settings
    l1_max_size: int = 1000
    l1_max_memory_mb: int = 100
    l2_max_size: int = 5000
    l2_max_memory_mb: int = 500
    
    # Disk cache settings
    l3_max_size: int = 50000
    l3_max_disk_mb: int = 2048
    l3_cache_dir: Optional[str] = None
    
    # Distributed cache settings
    l4_enabled: bool = False
    l4_redis_config: Optional[Dict[str, Any]] = None
    
    # Persistent cache settings
    l5_enabled: bool = False
    l5_db_path: Optional[str] = None
    
    # General settings
    compression: CompressionType = CompressionType.LZ4
    default_ttl: float = 3600.0
    invalidation_strategy: InvalidationStrategy = InvalidationStrategy.TIME_BASED
    enable_write_through: bool = True
    enable_write_back: bool = False
    sync_interval: float = 300.0  # 5 minutes
    
    # Performance settings
    enable_metrics: bool = True
    enable_background_sync: bool = True
    max_concurrent_operations: int = 10
    adaptive_sizing: bool = True


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    last_access: float
    access_count: int
    ttl: Optional[float]
    size_bytes: int
    compression_type: CompressionType = CompressionType.NONE
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    level: CacheLevel = CacheLevel.L1_MEMORY
    checksum: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def update_access(self) -> None:
        """Update access metadata."""
        self.last_access = time.time()
        self.access_count += 1
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for data integrity."""
        data = pickle.dumps(self.value)
        return hashlib.sha256(data).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify data integrity using checksum."""
        if not self.checksum:
            return True
        return self.checksum == self.calculate_checksum()


@dataclass
class CacheMetrics:
    """Comprehensive cache performance metrics."""
    # Basic metrics
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    avg_access_time: float = 0.0
    last_cleanup: float = 0.0
    
    # Level-specific metrics
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    l4_hits: int = 0
    l5_hits: int = 0
    
    l1_misses: int = 0
    l2_misses: int = 0
    l3_misses: int = 0
    l4_misses: int = 0
    l5_misses: int = 0
    
    # Operation metrics
    invalidations: int = 0
    write_through_operations: int = 0
    write_back_operations: int = 0
    background_syncs: int = 0
    compression_saves_bytes: int = 0
    
    # Performance metrics
    compression_ratio: float = 1.0
    hit_rate_by_level: Dict[str, float] = field(default_factory=dict)
    avg_access_time_by_level: Dict[str, float] = field(default_factory=dict)
    cache_efficiency_score: float = 0.0
    
    def calculate_hit_rates(self):
        """Calculate hit rates for all levels."""
        levels = [
            ("l1", self.l1_hits, self.l1_misses),
            ("l2", self.l2_hits, self.l2_misses),
            ("l3", self.l3_hits, self.l3_misses),
            ("l4", self.l4_hits, self.l4_misses),
            ("l5", self.l5_hits, self.l5_misses)
        ]
        
        for level_name, hits, misses in levels:
            total = hits + misses
            self.hit_rate_by_level[level_name] = hits / total if total > 0 else 0.0


class CompressionManager:
    """Manages compression/decompression operations."""
    
    def __init__(self):
        self.compressors = {
            CompressionType.LZ4: self._lz4_compress,
            CompressionType.ZSTD: self._zstd_compress,
            CompressionType.GZIP: self._gzip_compress,
            CompressionType.PICKLE: self._pickle_compress
        }
        
        self.decompressors = {
            CompressionType.LZ4: self._lz4_decompress,
            CompressionType.ZSTD: self._zstd_decompress,
            CompressionType.GZIP: self._gzip_decompress,
            CompressionType.PICKLE: self._pickle_decompress
        }
    
    def compress(self, data: Any, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return pickle.dumps(data)
        
        compressor = self.compressors.get(compression_type, self._pickle_compress)
        return compressor(data)
    
    def decompress(self, data: bytes, compression_type: CompressionType) -> Any:
        """Decompress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return pickle.loads(data)
        
        decompressor = self.decompressors.get(compression_type, self._pickle_decompress)
        return decompressor(data)
    
    def _lz4_compress(self, data: Any) -> bytes:
        """LZ4 compression."""
        try:
            pickled_data = pickle.dumps(data)
            return lz4.compress(pickled_data)
        except ImportError:
            logger.warning("LZ4 not available, falling back to pickle")
            return pickle.dumps(data)
    
    def _lz4_decompress(self, data: bytes) -> Any:
        """LZ4 decompression."""
        try:
            decompressed_data = lz4.decompress(data)
            return pickle.loads(decompressed_data)
        except ImportError:
            return pickle.loads(data)
    
    def _zstd_compress(self, data: Any) -> bytes:
        """ZSTD compression."""
        try:
            pickled_data = pickle.dumps(data)
            return zstd.compress(pickled_data)
        except ImportError:
            logger.warning("ZSTD not available, falling back to pickle")
            return pickle.dumps(data)
    
    def _zstd_decompress(self, data: bytes) -> Any:
        """ZSTD decompression."""
        try:
            decompressed_data = zstd.decompress(data)
            return pickle.loads(decompressed_data)
        except ImportError:
            return pickle.loads(data)
    
    def _gzip_compress(self, data: Any) -> bytes:
        """GZIP compression."""
        import gzip
        pickled_data = pickle.dumps(data)
        return gzip.compress(pickled_data)
    
    def _gzip_decompress(self, data: bytes) -> Any:
        """GZIP decompression."""
        import gzip
        decompressed_data = gzip.decompress(data)
        return pickle.loads(decompressed_data)
    
    def _pickle_compress(self, data: Any) -> bytes:
        """Pickle serialization (no compression)."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _pickle_decompress(self, data: bytes) -> Any:
        """Pickle deserialization."""
        return pickle.loads(data)


class InvalidationManager:
    """Manages cache invalidation strategies."""
    
    def __init__(self, strategy: InvalidationStrategy):
        self.strategy = strategy
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> keys
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # key -> dependents
        self.version_map: Dict[str, int] = {}  # key -> version
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)  # pattern -> keys
    
    def register_entry(self, entry: CacheEntry) -> None:
        """Register cache entry for invalidation tracking."""
        # Tag-based indexing
        for tag in entry.tags:
            self.tag_index[tag].add(entry.key)
        
        # Dependency tracking
        for dep in entry.dependencies:
            self.dependency_graph[dep].add(entry.key)
        
        # Version tracking
        self.version_map[entry.key] = entry.version
        
        # Pattern indexing (simple prefix matching)
        parts = entry.key.split(":")
        for i in range(len(parts)):
            pattern = ":".join(parts[:i+1])
            self.pattern_index[pattern].add(entry.key)
    
    def unregister_entry(self, key: str, entry: Optional[CacheEntry] = None) -> None:
        """Unregister cache entry from invalidation tracking."""
        # Remove from tag index
        if entry:
            for tag in entry.tags:
                self.tag_index[tag].discard(key)
        
        # Remove from dependency graph
        self.dependency_graph.pop(key, None)
        for dependents in self.dependency_graph.values():
            dependents.discard(key)
        
        # Remove from version map
        self.version_map.pop(key, None)
        
        # Remove from pattern index
        for pattern_keys in self.pattern_index.values():
            pattern_keys.discard(key)
    
    def get_invalidation_candidates(
        self, 
        key: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        pattern: Optional[str] = None,
        dependencies: Optional[Set[str]] = None
    ) -> Set[str]:
        """Get keys that should be invalidated based on criteria."""
        candidates = set()
        
        # Tag-based invalidation
        if tags:
            for tag in tags:
                candidates.update(self.tag_index[tag])
        
        # Pattern-based invalidation
        if pattern:
            candidates.update(self.pattern_index[pattern])
        
        # Dependency-based invalidation
        if dependencies:
            for dep in dependencies:
                candidates.update(self.dependency_graph[dep])
        
        # Single key invalidation
        if key:
            candidates.add(key)
            # Also invalidate dependents
            candidates.update(self.dependency_graph[key])
        
        return candidates
    
    def should_invalidate(self, entry: CacheEntry) -> bool:
        """Check if entry should be invalidated based on strategy."""
        if self.strategy == InvalidationStrategy.TIME_BASED:
            return entry.is_expired()
        elif self.strategy == InvalidationStrategy.VERSION_BASED:
            current_version = self.version_map.get(entry.key, entry.version)
            return current_version > entry.version
        else:
            return entry.is_expired()


class MultiLevelCache:
    """Advanced multi-level cache with intelligent management."""
    
    def __init__(self, config: CacheConfiguration):
        self.config = config
        self.metrics = CacheMetrics()
        self.compression_manager = CompressionManager()
        self.invalidation_manager = InvalidationManager(config.invalidation_strategy)
        
        # Cache levels
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()  # Hot data
        self.l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()  # Warm data
        self.l3_cache: Dict[str, str] = {}  # Disk cache (key -> file path)
        self.l4_client: Optional[redis.Redis] = None  # Distributed cache
        self.l5_db: Optional[sqlite3.Connection] = None  # Persistent cache
        
        # Thread synchronization
        self.lock = threading.RLock()
        self.background_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_concurrent_operations,
            thread_name_prefix="cache-bg"
        )
        
        # Cache directories
        self.cache_dir = Path(config.l3_cache_dir or tempfile.gettempdir()) / "grid_fed_rl_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache levels
        self._initialize_distributed_cache()
        self._initialize_persistent_cache()
        
        # Background tasks
        self.background_sync_active = config.enable_background_sync
        if self.background_sync_active:
            self._start_background_sync()
        
        logger.info("Multi-level cache system initialized")
    
    def _initialize_distributed_cache(self) -> None:
        """Initialize distributed cache (Redis)."""
        if not self.config.l4_enabled or not self.config.l4_redis_config:
            return
        
        try:
            self.l4_client = redis.Redis(**self.config.l4_redis_config)
            self.l4_client.ping()  # Test connection
            logger.info("Distributed cache (Redis) connected")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.l4_client = None
    
    def _initialize_persistent_cache(self) -> None:
        """Initialize persistent cache (SQLite)."""
        if not self.config.l5_enabled:
            return
        
        try:
            db_path = self.config.l5_db_path or str(self.cache_dir / "persistent_cache.db")
            self.l5_db = sqlite3.connect(db_path, check_same_thread=False)
            
            # Create cache table
            self.l5_db.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    last_access REAL,
                    access_count INTEGER,
                    ttl REAL,
                    size_bytes INTEGER,
                    compression_type TEXT,
                    version INTEGER,
                    tags TEXT,
                    checksum TEXT
                )
            """)
            self.l5_db.commit()
            logger.info("Persistent cache (SQLite) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize persistent cache: {e}")
            self.l5_db = None
    
    def _start_background_sync(self) -> None:
        """Start background synchronization task."""
        def sync_loop():
            while self.background_sync_active:
                try:
                    self._background_sync()
                    time.sleep(self.config.sync_interval)
                except Exception as e:
                    logger.error(f"Background sync error: {e}")
                    time.sleep(60)  # Wait a minute on error
        
        sync_thread = threading.Thread(target=sync_loop, daemon=True)
        sync_thread.start()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with multi-level lookup."""
        start_time = time.time()
        
        try:
            # Level 1: Hot memory cache
            with self.lock:
                if key in self.l1_cache:
                    entry = self.l1_cache[key]
                    if not entry.is_expired() and entry.verify_integrity():
                        entry.update_access()
                        self.l1_cache.move_to_end(key)  # Update LRU order
                        self.metrics.l1_hits += 1
                        self.metrics.hits += 1
                        return entry.value
                    else:
                        # Expired or corrupted, remove
                        del self.l1_cache[key]
                        self.invalidation_manager.unregister_entry(key, entry)
            
            # Level 2: Warm memory cache
            with self.lock:
                if key in self.l2_cache:
                    entry = self.l2_cache[key]
                    if not entry.is_expired() and entry.verify_integrity():
                        entry.update_access()
                        # Promote to L1 if frequently accessed
                        if entry.access_count > 3:
                            self._promote_to_l1(key, entry)
                        self.metrics.l2_hits += 1
                        self.metrics.hits += 1
                        return entry.value
                    else:
                        del self.l2_cache[key]
                        self.invalidation_manager.unregister_entry(key, entry)
            
            # Level 3: Disk cache
            if key in self.l3_cache:
                try:
                    file_path = self.l3_cache[key]
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            data = f.read()
                        
                        # Deserialize entry
                        entry = pickle.loads(data)
                        if not entry.is_expired() and entry.verify_integrity():
                            entry.update_access()
                            # Promote to L2
                            self._promote_to_l2(key, entry)
                            self.metrics.l3_hits += 1
                            self.metrics.hits += 1
                            return entry.value
                        else:
                            # Expired, remove
                            os.remove(file_path)
                            del self.l3_cache[key]
                except Exception as e:
                    logger.warning(f"Disk cache read error for {key}: {e}")
            
            # Level 4: Distributed cache
            if self.l4_client:
                try:
                    data = self.l4_client.get(key)
                    if data:
                        entry = pickle.loads(data)
                        if not entry.is_expired() and entry.verify_integrity():
                            entry.update_access()
                            # Cache in L2 for faster future access
                            self._store_in_l2(key, entry)
                            self.metrics.l4_hits += 1
                            self.metrics.hits += 1
                            return entry.value
                except Exception as e:
                    logger.warning(f"Distributed cache read error for {key}: {e}")
            
            # Level 5: Persistent cache
            if self.l5_db:
                try:
                    cursor = self.l5_db.cursor()
                    cursor.execute("SELECT * FROM cache_entries WHERE key = ?", (key,))
                    row = cursor.fetchone()
                    if row:
                        # Reconstruct entry
                        _, value_blob, timestamp, last_access, access_count, ttl, size_bytes, \
                        compression_type, version, tags_str, checksum = row
                        
                        value = self.compression_manager.decompress(
                            value_blob, CompressionType(compression_type)
                        )
                        
                        entry = CacheEntry(
                            key=key,
                            value=value,
                            timestamp=timestamp,
                            last_access=last_access,
                            access_count=access_count,
                            ttl=ttl,
                            size_bytes=size_bytes,
                            compression_type=CompressionType(compression_type),
                            version=version,
                            tags=set(tags_str.split(',')) if tags_str else set(),
                            level=CacheLevel.L5_PERSISTENT,
                            checksum=checksum
                        )
                        
                        if not entry.is_expired() and entry.verify_integrity():
                            entry.update_access()
                            # Cache in L2 for faster future access
                            self._store_in_l2(key, entry)
                            self.metrics.l5_hits += 1
                            self.metrics.hits += 1
                            return entry.value
                        else:
                            # Expired, remove
                            cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                            self.l5_db.commit()
                except Exception as e:
                    logger.warning(f"Persistent cache read error for {key}: {e}")
            
            # Cache miss
            self.metrics.misses += 1
            return default
            
        finally:
            access_time = time.time() - start_time
            total_accesses = self.metrics.hits + self.metrics.misses
            if total_accesses > 0:
                self.metrics.avg_access_time = (
                    (self.metrics.avg_access_time * (total_accesses - 1) + access_time)
                    / total_accesses
                )
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
        level: Optional[CacheLevel] = None
    ) -> None:
        """Set value in cache with multi-level storage."""
        ttl = ttl or self.config.default_ttl
        tags = tags or set()
        dependencies = dependencies or set()
        level = level or CacheLevel.L1_MEMORY
        
        # Create cache entry
        compressed_data = self.compression_manager.compress(value, self.config.compression)
        size_bytes = len(compressed_data)
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            last_access=time.time(),
            access_count=1,
            ttl=ttl,
            size_bytes=size_bytes,
            compression_type=self.config.compression,
            tags=tags,
            dependencies=dependencies,
            level=level,
            checksum=hashlib.sha256(compressed_data).hexdigest()
        )
        
        # Store in appropriate level
        if level == CacheLevel.L1_MEMORY:
            self._store_in_l1(key, entry)
        elif level == CacheLevel.L2_MEMORY:
            self._store_in_l2(key, entry)
        elif level == CacheLevel.L3_DISK:
            self._store_in_l3(key, entry)
        
        # Write-through to lower levels if enabled
        if self.config.enable_write_through:
            self.background_executor.submit(self._write_through, key, entry)
        
        # Register for invalidation tracking
        self.invalidation_manager.register_entry(entry)
    
    def _store_in_l1(self, key: str, entry: CacheEntry) -> None:
        """Store entry in L1 cache."""
        with self.lock:
            # Check size limits and evict if necessary
            while (len(self.l1_cache) >= self.config.l1_max_size or
                   self._get_l1_memory_usage() + entry.size_bytes > self.config.l1_max_memory_mb * 1024 * 1024):
                if not self.l1_cache:
                    break
                lru_key, lru_entry = self.l1_cache.popitem(last=False)
                # Demote to L2
                self._store_in_l2(lru_key, lru_entry)
                self.metrics.evictions += 1
            
            self.l1_cache[key] = entry
            entry.level = CacheLevel.L1_MEMORY
    
    def _store_in_l2(self, key: str, entry: CacheEntry) -> None:
        """Store entry in L2 cache."""
        with self.lock:
            # Check size limits and evict if necessary
            while (len(self.l2_cache) >= self.config.l2_max_size or
                   self._get_l2_memory_usage() + entry.size_bytes > self.config.l2_max_memory_mb * 1024 * 1024):
                if not self.l2_cache:
                    break
                lru_key, lru_entry = self.l2_cache.popitem(last=False)
                # Demote to L3
                self._store_in_l3(lru_key, lru_entry)
                self.metrics.evictions += 1
            
            self.l2_cache[key] = entry
            entry.level = CacheLevel.L2_MEMORY
    
    def _store_in_l3(self, key: str, entry: CacheEntry) -> None:
        """Store entry in L3 disk cache."""
        try:
            # Create file path
            safe_key = hashlib.md5(key.encode()).hexdigest()
            file_path = self.cache_dir / f"l3_{safe_key}.cache"
            
            # Serialize entry
            data = pickle.dumps(entry)
            
            # Write to disk
            with open(file_path, 'wb') as f:
                f.write(data)
            
            self.l3_cache[key] = str(file_path)
            entry.level = CacheLevel.L3_DISK
        except Exception as e:
            logger.warning(f"Failed to store {key} in L3 cache: {e}")
    
    def _promote_to_l1(self, key: str, entry: CacheEntry) -> None:
        """Promote entry from L2 to L1."""
        with self.lock:
            if key in self.l2_cache:
                del self.l2_cache[key]
            self._store_in_l1(key, entry)
    
    def _promote_to_l2(self, key: str, entry: CacheEntry) -> None:
        """Promote entry from L3+ to L2."""
        self._store_in_l2(key, entry)
    
    def _write_through(self, key: str, entry: CacheEntry) -> None:
        """Write-through to lower cache levels."""
        try:
            # Write to distributed cache
            if self.l4_client:
                try:
                    data = pickle.dumps(entry)
                    self.l4_client.setex(key, int(entry.ttl or self.config.default_ttl), data)
                    self.metrics.write_through_operations += 1
                except Exception as e:
                    logger.warning(f"Write-through to distributed cache failed for {key}: {e}")
            
            # Write to persistent cache
            if self.l5_db:
                try:
                    cursor = self.l5_db.cursor()
                    compressed_value = self.compression_manager.compress(
                        entry.value, entry.compression_type
                    )
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, value, timestamp, last_access, access_count, ttl, size_bytes,
                         compression_type, version, tags, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, compressed_value, entry.timestamp, entry.last_access,
                        entry.access_count, entry.ttl, entry.size_bytes,
                        entry.compression_type.value, entry.version,
                        ','.join(entry.tags), entry.checksum
                    ))
                    self.l5_db.commit()
                    self.metrics.write_through_operations += 1
                except Exception as e:
                    logger.warning(f"Write-through to persistent cache failed for {key}: {e}")
                    
        except Exception as e:
            logger.error(f"Write-through operation failed for {key}: {e}")
    
    def invalidate(
        self,
        key: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        pattern: Optional[str] = None,
        dependencies: Optional[Set[str]] = None
    ) -> int:
        """Intelligent cache invalidation."""
        candidates = self.invalidation_manager.get_invalidation_candidates(
            key=key, tags=tags, pattern=pattern, dependencies=dependencies
        )
        
        invalidated_count = 0
        
        with self.lock:
            for candidate_key in candidates:
                # Remove from all levels
                removed = False
                
                # L1 cache
                if candidate_key in self.l1_cache:
                    entry = self.l1_cache.pop(candidate_key)
                    self.invalidation_manager.unregister_entry(candidate_key, entry)
                    removed = True
                
                # L2 cache
                if candidate_key in self.l2_cache:
                    entry = self.l2_cache.pop(candidate_key)
                    self.invalidation_manager.unregister_entry(candidate_key, entry)
                    removed = True
                
                # L3 cache
                if candidate_key in self.l3_cache:
                    try:
                        file_path = self.l3_cache.pop(candidate_key)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove L3 cache file for {candidate_key}: {e}")
                    removed = True
                
                # L4 distributed cache
                if self.l4_client:
                    try:
                        self.l4_client.delete(candidate_key)
                    except Exception as e:
                        logger.warning(f"Failed to remove from distributed cache {candidate_key}: {e}")
                
                # L5 persistent cache
                if self.l5_db:
                    try:
                        cursor = self.l5_db.cursor()
                        cursor.execute("DELETE FROM cache_entries WHERE key = ?", (candidate_key,))
                        self.l5_db.commit()
                    except Exception as e:
                        logger.warning(f"Failed to remove from persistent cache {candidate_key}: {e}")
                
                if removed:
                    invalidated_count += 1
        
        self.metrics.invalidations += invalidated_count
        return invalidated_count
    
    def _background_sync(self) -> None:
        """Background synchronization between cache levels."""
        try:
            # Cleanup expired entries
            self._cleanup_expired()
            
            # Optimize cache distribution
            self._optimize_cache_distribution()
            
            # Update metrics
            self.metrics.calculate_hit_rates()
            self.metrics.background_syncs += 1
            
        except Exception as e:
            logger.error(f"Background sync error: {e}")
    
    def _cleanup_expired(self) -> None:
        """Clean up expired entries from all levels."""
        current_time = time.time()
        
        with self.lock:
            # L1 cleanup
            expired_l1 = [k for k, v in self.l1_cache.items() if v.is_expired()]
            for key in expired_l1:
                entry = self.l1_cache.pop(key)
                self.invalidation_manager.unregister_entry(key, entry)
            
            # L2 cleanup
            expired_l2 = [k for k, v in self.l2_cache.items() if v.is_expired()]
            for key in expired_l2:
                entry = self.l2_cache.pop(key)
                self.invalidation_manager.unregister_entry(key, entry)
            
            # L3 cleanup (in background)
            def cleanup_l3():
                expired_l3 = []
                for key, file_path in list(self.l3_cache.items()):
                    try:
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                entry = pickle.loads(f.read())
                            if entry.is_expired():
                                expired_l3.append((key, file_path))
                    except Exception as e:
                        logger.warning(f"Error checking L3 entry {key}: {e}")
                        expired_l3.append((key, file_path))
                
                for key, file_path in expired_l3:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        self.l3_cache.pop(key, None)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup L3 entry {key}: {e}")
            
            self.background_executor.submit(cleanup_l3)
    
    def _optimize_cache_distribution(self) -> None:
        """Optimize distribution of data across cache levels."""
        # Move frequently accessed L2 items to L1
        with self.lock:
            promote_candidates = []
            for key, entry in list(self.l2_cache.items()):
                if entry.access_count > 5 and len(self.l1_cache) < self.config.l1_max_size * 0.8:
                    promote_candidates.append((key, entry))
            
            for key, entry in promote_candidates[:10]:  # Limit promotions
                del self.l2_cache[key]
                self._store_in_l1(key, entry)
    
    def _get_l1_memory_usage(self) -> int:
        """Get current L1 cache memory usage in bytes."""
        return sum(entry.size_bytes for entry in self.l1_cache.values())
    
    def _get_l2_memory_usage(self) -> int:
        """Get current L2 cache memory usage in bytes."""
        return sum(entry.size_bytes for entry in self.l2_cache.values())
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        self.metrics.calculate_hit_rates()
        
        # Calculate efficiency score
        total_hits = self.metrics.hits
        total_requests = self.metrics.hits + self.metrics.misses
        hit_rate = total_hits / max(total_requests, 1)
        
        # Weight by access speed (L1 is fastest)
        weighted_hits = (
            self.metrics.l1_hits * 1.0 +    # Fastest
            self.metrics.l2_hits * 0.9 +    # Fast
            self.metrics.l3_hits * 0.7 +    # Medium
            self.metrics.l4_hits * 0.5 +    # Slower
            self.metrics.l5_hits * 0.3      # Slowest
        )
        
        efficiency_score = weighted_hits / max(total_hits, 1) if total_hits > 0 else 0
        self.metrics.cache_efficiency_score = efficiency_score
        
        return {
            "metrics": asdict(self.metrics),
            "configuration": asdict(self.config),
            "level_sizes": {
                "l1": len(self.l1_cache),
                "l2": len(self.l2_cache),
                "l3": len(self.l3_cache),
                "l4": "redis_connected" if self.l4_client else "disabled",
                "l5": "sqlite_connected" if self.l5_db else "disabled"
            },
            "memory_usage": {
                "l1_mb": self._get_l1_memory_usage() / (1024 * 1024),
                "l2_mb": self._get_l2_memory_usage() / (1024 * 1024),
                "process_mb": psutil.Process().memory_info().rss / (1024 * 1024)
            },
            "performance": {
                "hit_rate": hit_rate,
                "efficiency_score": efficiency_score,
                "avg_access_time_ms": self.metrics.avg_access_time * 1000
            }
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the cache system."""
        logger.info("Shutting down multi-level cache...")
        
        self.background_sync_active = False
        
        if self.background_executor:
            self.background_executor.shutdown(wait=True)
        
        if self.l4_client:
            try:
                self.l4_client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
        
        if self.l5_db:
            try:
                self.l5_db.close()
            except Exception as e:
                logger.warning(f"Error closing SQLite connection: {e}")
        
        logger.info("Multi-level cache shutdown complete")


# Global multi-level cache instance
_default_config = CacheConfiguration()
global_multi_cache = MultiLevelCache(_default_config)


def cached_multi_level(
    ttl: Optional[float] = None,
    tags: Optional[Set[str]] = None,
    level: Optional[CacheLevel] = None
) -> Callable:
    """Decorator for multi-level caching."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{func.__module__}.{func.__name__}"
            args_str = json.dumps([str(arg) for arg in args], sort_keys=True)
            kwargs_str = json.dumps(kwargs, sort_keys=True)
            cache_key = f"{func_name}:{hashlib.md5((args_str + kwargs_str).encode()).hexdigest()}"
            
            # Try to get from cache
            result = global_multi_cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            global_multi_cache.set(cache_key, result, ttl=ttl, tags=tags, level=level)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(
    tags: Optional[Set[str]] = None,
    pattern: Optional[str] = None
) -> int:
    """Invalidate cache entries by tags or pattern."""
    return global_multi_cache.invalidate(tags=tags, pattern=pattern)


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return global_multi_cache.get_comprehensive_stats()