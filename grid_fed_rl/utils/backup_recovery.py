"""Backup and recovery systems for Grid-Fed-RL-Gym."""

import os
import json
import pickle
import sqlite3
import gzip
import shutil
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import hashlib
import boto3
import cryptography.fernet
from collections import defaultdict, deque

from .exceptions import GridEnvironmentError, CorruptedDataError
from .security import EncryptionManager

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for backup files."""
    backup_id: str
    timestamp: datetime
    backup_type: str  # full, incremental, checkpoint
    data_types: List[str]
    file_size: int
    compression_ratio: float
    encryption_enabled: bool
    checksum: str
    source_system: str
    retention_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupMetadata':
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class BackupManager:
    """Comprehensive backup and recovery manager."""
    
    def __init__(
        self,
        backup_dir: str = "./backups",
        max_local_backups: int = 50,
        encryption_key: Optional[bytes] = None,
        cloud_config: Optional[Dict[str, str]] = None
    ):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_local_backups = max_local_backups
        self.encryption_manager = EncryptionManager() if encryption_key else None
        
        # Cloud storage configuration
        self.cloud_config = cloud_config or {}
        self.s3_client = None
        if self.cloud_config.get("aws_access_key_id"):
            self._setup_cloud_storage()
        
        # Backup tracking
        self.backup_index = self._load_backup_index()
        self.active_backups: Dict[str, threading.Thread] = {}
        
        # Automatic backup scheduling
        self.auto_backup_enabled = False
        self.auto_backup_thread = None
        self.backup_intervals = {
            "critical_data": 300,      # 5 minutes
            "training_checkpoints": 1800,  # 30 minutes
            "system_config": 3600,    # 1 hour
            "logs": 7200              # 2 hours
        }
        
        # Recovery point tracking
        self.recovery_points = deque(maxlen=1000)
        
        logger.info(f"Backup manager initialized with directory: {self.backup_dir}")
    
    def _setup_cloud_storage(self) -> None:
        """Setup cloud storage client."""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.cloud_config["aws_access_key_id"],
                aws_secret_access_key=self.cloud_config["aws_secret_access_key"],
                region_name=self.cloud_config.get("region", "us-east-1")
            )
            logger.info("Cloud storage configured successfully")
        except Exception as e:
            logger.error(f"Failed to setup cloud storage: {e}")
            self.s3_client = None
    
    def create_backup(
        self,
        data: Dict[str, Any],
        backup_type: str = "full",
        data_types: Optional[List[str]] = None,
        compression: bool = True,
        encryption: bool = True,
        upload_to_cloud: bool = False
    ) -> str:
        """Create a comprehensive backup."""
        
        backup_id = self._generate_backup_id()
        timestamp = datetime.now()
        
        logger.info(f"Creating {backup_type} backup with ID: {backup_id}")
        
        try:
            # Prepare backup data
            backup_data = self._prepare_backup_data(data, data_types or [])
            
            # Serialize data
            serialized_data = self._serialize_backup_data(backup_data)
            
            # Compress if enabled
            if compression:
                serialized_data = self._compress_data(serialized_data)
                compression_ratio = len(serialized_data) / len(pickle.dumps(backup_data))
            else:
                compression_ratio = 1.0
            
            # Encrypt if enabled
            if encryption and self.encryption_manager:
                serialized_data = self.encryption_manager.encrypt_data(serialized_data)
            
            # Calculate checksum
            checksum = hashlib.sha256(serialized_data).hexdigest()
            
            # Save to local file
            backup_filename = f"backup_{backup_id}.bak"
            backup_path = self.backup_dir / backup_filename
            
            with open(backup_path, 'wb') as f:
                f.write(serialized_data)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type=backup_type,
                data_types=data_types or list(backup_data.keys()),
                file_size=len(serialized_data),
                compression_ratio=compression_ratio,
                encryption_enabled=encryption and self.encryption_manager is not None,
                checksum=checksum,
                source_system=os.getenv('HOSTNAME', 'unknown'),
                retention_days=self._get_retention_days(backup_type)
            )
            
            # Save metadata
            metadata_path = self.backup_dir / f"backup_{backup_id}.meta"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update backup index
            self.backup_index[backup_id] = metadata
            self._save_backup_index()
            
            # Upload to cloud if enabled
            if upload_to_cloud and self.s3_client:
                self._upload_to_cloud(backup_id, backup_path, metadata_path)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            logger.info(f"Backup {backup_id} created successfully ({len(serialized_data)} bytes)")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise GridEnvironmentError(f"Backup creation failed: {e}")
    
    def restore_backup(
        self,
        backup_id: str,
        verify_checksum: bool = True,
        partial_restore: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Restore data from backup."""
        
        logger.info(f"Restoring backup: {backup_id}")
        
        try:
            # Load metadata
            if backup_id not in self.backup_index:
                # Try to load from file
                metadata_path = self.backup_dir / f"backup_{backup_id}.meta"
                if not metadata_path.exists():
                    raise GridEnvironmentError(f"Backup {backup_id} not found")
                
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                metadata = BackupMetadata.from_dict(metadata_dict)
                self.backup_index[backup_id] = metadata
            else:
                metadata = self.backup_index[backup_id]
            
            # Load backup file
            backup_path = self.backup_dir / f"backup_{backup_id}.bak"
            if not backup_path.exists():
                # Try to download from cloud
                if self.s3_client:
                    self._download_from_cloud(backup_id, backup_path)
                else:
                    raise GridEnvironmentError(f"Backup file {backup_id} not found locally")
            
            with open(backup_path, 'rb') as f:
                backup_data = f.read()
            
            # Verify checksum
            if verify_checksum:
                current_checksum = hashlib.sha256(backup_data).hexdigest()
                if current_checksum != metadata.checksum:
                    raise CorruptedDataError(
                        f"Backup {backup_id} checksum mismatch",
                        str(backup_path)
                    )
            
            # Decrypt if needed
            if metadata.encryption_enabled and self.encryption_manager:
                backup_data = self.encryption_manager.decrypt_data(backup_data)
            
            # Decompress if needed
            if metadata.compression_ratio < 0.99:  # Was compressed
                backup_data = self._decompress_data(backup_data)
            
            # Deserialize data
            restored_data = self._deserialize_backup_data(backup_data)
            
            # Partial restore if requested
            if partial_restore:
                restored_data = {
                    key: value for key, value in restored_data.items()
                    if key in partial_restore
                }
            
            logger.info(f"Backup {backup_id} restored successfully")
            return restored_data
            
        except Exception as e:
            logger.error(f"Failed to restore backup {backup_id}: {e}")
            raise GridEnvironmentError(f"Backup restoration failed: {e}")
    
    def list_backups(
        self,
        backup_type: Optional[str] = None,
        data_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[BackupMetadata]:
        """List available backups with filtering."""
        
        backups = list(self.backup_index.values())
        
        # Apply filters
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        if data_type:
            backups = [b for b in backups if data_type in b.data_types]
        
        if since:
            backups = [b for b in backups if b.timestamp >= since]
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        return backups
    
    def delete_backup(self, backup_id: str, delete_from_cloud: bool = True) -> bool:
        """Delete a backup."""
        
        try:
            # Remove from index
            if backup_id in self.backup_index:
                del self.backup_index[backup_id]
                self._save_backup_index()
            
            # Delete local files
            backup_path = self.backup_dir / f"backup_{backup_id}.bak"
            metadata_path = self.backup_dir / f"backup_{backup_id}.meta"
            
            backup_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            
            # Delete from cloud
            if delete_from_cloud and self.s3_client:
                try:
                    bucket = self.cloud_config.get("bucket", "grid-fed-rl-backups")
                    self.s3_client.delete_object(Bucket=bucket, Key=f"backups/{backup_id}.bak")
                    self.s3_client.delete_object(Bucket=bucket, Key=f"backups/{backup_id}.meta")
                except Exception as e:
                    logger.warning(f"Failed to delete cloud backup {backup_id}: {e}")
            
            logger.info(f"Backup {backup_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def start_auto_backup(
        self,
        data_source_callback: Callable[[], Dict[str, Any]],
        backup_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start automatic backup process."""
        
        if self.auto_backup_enabled:
            logger.warning("Auto backup already running")
            return
        
        self.auto_backup_enabled = True
        self.data_source_callback = data_source_callback
        self.backup_config = backup_config or {}
        
        self.auto_backup_thread = threading.Thread(target=self._auto_backup_loop, daemon=True)
        self.auto_backup_thread.start()
        
        logger.info("Automatic backup started")
    
    def stop_auto_backup(self) -> None:
        """Stop automatic backup process."""
        
        self.auto_backup_enabled = False
        if self.auto_backup_thread and self.auto_backup_thread.is_alive():
            self.auto_backup_thread.join(timeout=10.0)
        
        logger.info("Automatic backup stopped")
    
    def _auto_backup_loop(self) -> None:
        """Main loop for automatic backups."""
        
        last_backup_times = defaultdict(float)
        
        while self.auto_backup_enabled:
            try:
                current_time = time.time()
                
                for data_type, interval in self.backup_intervals.items():
                    if current_time - last_backup_times[data_type] >= interval:
                        # Get data from callback
                        try:
                            data = self.data_source_callback()
                            
                            # Filter data for this backup type
                            filtered_data = self._filter_data_for_type(data, data_type)
                            
                            if filtered_data:  # Only backup if there's data
                                backup_id = self.create_backup(
                                    filtered_data,
                                    backup_type="incremental",
                                    data_types=[data_type],
                                    **self.backup_config
                                )
                                
                                last_backup_times[data_type] = current_time
                                logger.debug(f"Auto backup created for {data_type}: {backup_id}")
                                
                        except Exception as e:
                            logger.error(f"Auto backup failed for {data_type}: {e}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto backup loop error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def create_recovery_point(
        self,
        system_state: Dict[str, Any],
        description: str = "",
        critical: bool = False
    ) -> str:
        """Create a recovery point for quick restoration."""
        
        recovery_id = self._generate_backup_id()
        timestamp = datetime.now()
        
        recovery_point = {
            "recovery_id": recovery_id,
            "timestamp": timestamp,
            "description": description,
            "critical": critical,
            "system_state": system_state,
            "checksum": hashlib.sha256(json.dumps(system_state, sort_keys=True).encode()).hexdigest()
        }
        
        self.recovery_points.append(recovery_point)
        
        # Save critical recovery points to disk
        if critical:
            recovery_path = self.backup_dir / f"recovery_{recovery_id}.json"
            with open(recovery_path, 'w') as f:
                json.dump(recovery_point, f, indent=2, default=str)
        
        logger.info(f"Recovery point created: {recovery_id} - {description}")
        return recovery_id
    
    def restore_from_recovery_point(self, recovery_id: str) -> Dict[str, Any]:
        """Restore system from a recovery point."""
        
        # Search in memory first
        for rp in self.recovery_points:
            if rp["recovery_id"] == recovery_id:
                logger.info(f"Restoring from recovery point: {recovery_id}")
                return rp["system_state"]
        
        # Search on disk
        recovery_path = self.backup_dir / f"recovery_{recovery_id}.json"
        if recovery_path.exists():
            with open(recovery_path, 'r') as f:
                recovery_point = json.load(f)
            
            logger.info(f"Restoring from disk recovery point: {recovery_id}")
            return recovery_point["system_state"]
        
        raise GridEnvironmentError(f"Recovery point {recovery_id} not found")
    
    def get_recovery_points(self, critical_only: bool = False) -> List[Dict[str, Any]]:
        """Get list of available recovery points."""
        points = list(self.recovery_points)
        
        if critical_only:
            points = [p for p in points if p["critical"]]
        
        return sorted(points, key=lambda x: x["timestamp"], reverse=True)
    
    def verify_backup_integrity(self, backup_id: Optional[str] = None) -> Dict[str, Any]:
        """Verify integrity of backups."""
        
        if backup_id:
            backups_to_check = [backup_id] if backup_id in self.backup_index else []
        else:
            backups_to_check = list(self.backup_index.keys())
        
        results = {
            "total_checked": len(backups_to_check),
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for bid in backups_to_check:
            metadata = self.backup_index[bid]
            backup_path = self.backup_dir / f"backup_{bid}.bak"
            
            result = {
                "backup_id": bid,
                "status": "unknown",
                "issues": []
            }
            
            # Check file exists
            if not backup_path.exists():
                result["status"] = "failed"
                result["issues"].append("Backup file missing")
            else:
                # Check file size
                actual_size = backup_path.stat().st_size
                if actual_size != metadata.file_size:
                    result["status"] = "failed"
                    result["issues"].append(f"Size mismatch: expected {metadata.file_size}, got {actual_size}")
                
                # Check checksum
                try:
                    with open(backup_path, 'rb') as f:
                        actual_checksum = hashlib.sha256(f.read()).hexdigest()
                    
                    if actual_checksum != metadata.checksum:
                        result["status"] = "failed"
                        result["issues"].append("Checksum mismatch")
                    else:
                        result["status"] = "passed"
                        
                except Exception as e:
                    result["status"] = "failed"
                    result["issues"].append(f"Checksum verification failed: {e}")
            
            if result["status"] == "passed":
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            results["details"].append(result)
        
        logger.info(f"Backup integrity check completed: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get comprehensive backup statistics."""
        
        if not self.backup_index:
            return {"no_backups": True}
        
        backups = list(self.backup_index.values())
        
        # Basic statistics
        total_size = sum(b.file_size for b in backups)
        avg_compression = sum(b.compression_ratio for b in backups) / len(backups)
        
        # By type
        type_stats = defaultdict(lambda: {"count": 0, "size": 0})
        for backup in backups:
            type_stats[backup.backup_type]["count"] += 1
            type_stats[backup.backup_type]["size"] += backup.file_size
        
        # Recent activity
        now = datetime.now()
        recent_backups = [b for b in backups if (now - b.timestamp).days <= 7]
        
        return {
            "total_backups": len(backups),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "average_compression_ratio": avg_compression,
            "encrypted_backups": sum(1 for b in backups if b.encryption_enabled),
            "by_type": dict(type_stats),
            "recent_activity_7d": {
                "count": len(recent_backups),
                "size_mb": sum(b.file_size for b in recent_backups) / (1024 * 1024)
            },
            "oldest_backup": min(b.timestamp for b in backups).isoformat(),
            "newest_backup": max(b.timestamp for b in backups).isoformat()
        }
    
    # Helper methods
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(f"{time.time()}{os.getpid()}".encode()).hexdigest()[:6]
        return f"{timestamp}_{random_suffix}"
    
    def _prepare_backup_data(self, data: Dict[str, Any], data_types: List[str]) -> Dict[str, Any]:
        """Prepare data for backup."""
        if not data_types:
            return data
        
        # Filter data based on types
        filtered_data = {}
        for data_type in data_types:
            if data_type in data:
                filtered_data[data_type] = data[data_type]
        
        return filtered_data or data
    
    def _serialize_backup_data(self, data: Dict[str, Any]) -> bytes:
        """Serialize backup data."""
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_backup_data(self, data: bytes) -> Dict[str, Any]:
        """Deserialize backup data."""
        return pickle.loads(data)
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data, compresslevel=6)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress gzipped data."""
        return gzip.decompress(data)
    
    def _get_retention_days(self, backup_type: str) -> int:
        """Get retention period for backup type."""
        retention_map = {
            "full": 90,
            "incremental": 30,
            "checkpoint": 7,
            "emergency": 365
        }
        return retention_map.get(backup_type, 30)
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy."""
        now = datetime.now()
        to_delete = []
        
        for backup_id, metadata in self.backup_index.items():
            age_days = (now - metadata.timestamp).days
            if age_days > metadata.retention_days:
                to_delete.append(backup_id)
        
        # Keep minimum number of recent backups
        if len(self.backup_index) - len(to_delete) < 10:
            # Sort by timestamp and keep 10 most recent
            sorted_backups = sorted(
                self.backup_index.items(),
                key=lambda x: x[1].timestamp,
                reverse=True
            )
            to_keep = [bid for bid, _ in sorted_backups[:10]]
            to_delete = [bid for bid in to_delete if bid not in to_keep]
        
        # Delete old backups
        for backup_id in to_delete:
            self.delete_backup(backup_id)
        
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old backups")
    
    def _load_backup_index(self) -> Dict[str, BackupMetadata]:
        """Load backup index from disk."""
        index_path = self.backup_dir / "backup_index.json"
        
        if not index_path.exists():
            return {}
        
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            index = {}
            for backup_id, metadata_dict in index_data.items():
                index[backup_id] = BackupMetadata.from_dict(metadata_dict)
            
            return index
            
        except Exception as e:
            logger.error(f"Failed to load backup index: {e}")
            return {}
    
    def _save_backup_index(self) -> None:
        """Save backup index to disk."""
        index_path = self.backup_dir / "backup_index.json"
        
        try:
            index_data = {
                backup_id: metadata.to_dict()
                for backup_id, metadata in self.backup_index.items()
            }
            
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup index: {e}")
    
    def _upload_to_cloud(self, backup_id: str, backup_path: Path, metadata_path: Path) -> None:
        """Upload backup to cloud storage."""
        if not self.s3_client:
            return
        
        try:
            bucket = self.cloud_config.get("bucket", "grid-fed-rl-backups")
            
            # Upload backup file
            self.s3_client.upload_file(
                str(backup_path),
                bucket,
                f"backups/{backup_id}.bak"
            )
            
            # Upload metadata
            self.s3_client.upload_file(
                str(metadata_path),
                bucket,
                f"backups/{backup_id}.meta"
            )
            
            logger.info(f"Backup {backup_id} uploaded to cloud storage")
            
        except Exception as e:
            logger.error(f"Failed to upload backup {backup_id} to cloud: {e}")
    
    def _download_from_cloud(self, backup_id: str, local_path: Path) -> None:
        """Download backup from cloud storage."""
        if not self.s3_client:
            raise GridEnvironmentError("Cloud storage not configured")
        
        try:
            bucket = self.cloud_config.get("bucket", "grid-fed-rl-backups")
            
            self.s3_client.download_file(
                bucket,
                f"backups/{backup_id}.bak",
                str(local_path)
            )
            
            logger.info(f"Backup {backup_id} downloaded from cloud storage")
            
        except Exception as e:
            logger.error(f"Failed to download backup {backup_id} from cloud: {e}")
            raise GridEnvironmentError(f"Cloud backup download failed: {e}")
    
    def _filter_data_for_type(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Filter data for specific backup type."""
        type_filters = {
            "critical_data": ["system_state", "safety_config", "emergency_protocols"],
            "training_checkpoints": ["model_weights", "optimizer_state", "training_metrics"],
            "system_config": ["environment_config", "algorithm_config", "network_topology"],
            "logs": ["system_logs", "error_logs", "performance_logs"]
        }
        
        if data_type not in type_filters:
            return data
        
        filtered_data = {}
        for key in type_filters[data_type]:
            if key in data:
                filtered_data[key] = data[key]
        
        return filtered_data


class DisasterRecoveryManager:
    """Comprehensive disaster recovery management."""
    
    def __init__(
        self,
        backup_manager: BackupManager,
        recovery_config: Optional[Dict[str, Any]] = None
    ):
        self.backup_manager = backup_manager
        self.recovery_config = recovery_config or {}
        
        # Recovery procedures
        self.recovery_procedures = {
            "system_crash": self._system_crash_recovery,
            "data_corruption": self._data_corruption_recovery,
            "network_failure": self._network_failure_recovery,
            "storage_failure": self._storage_failure_recovery,
            "complete_failure": self._complete_failure_recovery
        }
        
        # Recovery metrics
        self.recovery_history = deque(maxlen=1000)
        self.rto_target = recovery_config.get("rto_minutes", 30)  # Recovery Time Objective
        self.rpo_target = recovery_config.get("rpo_minutes", 15)  # Recovery Point Objective
    
    def initiate_recovery(
        self,
        failure_type: str,
        last_known_good_timestamp: Optional[datetime] = None,
        recovery_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Initiate disaster recovery procedure."""
        
        recovery_start = datetime.now()
        logger.critical(f"Initiating disaster recovery for: {failure_type}")
        
        recovery_result = {
            "recovery_id": self._generate_recovery_id(),
            "failure_type": failure_type,
            "start_time": recovery_start,
            "target_recovery_time": last_known_good_timestamp,
            "status": "in_progress",
            "steps_completed": [],
            "issues_encountered": [],
            "data_recovered": {},
            "system_status": "unknown"
        }
        
        try:
            # Execute recovery procedure
            if failure_type in self.recovery_procedures:
                procedure_result = self.recovery_procedures[failure_type](
                    last_known_good_timestamp,
                    recovery_options or {}
                )
                
                recovery_result.update(procedure_result)
                recovery_result["status"] = "completed"
                recovery_result["system_status"] = "operational"
                
            else:
                # Generic recovery
                recovery_result.update(self._generic_recovery(
                    last_known_good_timestamp,
                    recovery_options or {}
                ))
                recovery_result["status"] = "completed"
                recovery_result["system_status"] = "operational"
        
        except Exception as e:
            recovery_result["status"] = "failed"
            recovery_result["error"] = str(e)
            recovery_result["system_status"] = "failed"
            logger.error(f"Recovery failed: {e}")
        
        finally:
            recovery_result["end_time"] = datetime.now()
            recovery_result["total_duration"] = (
                recovery_result["end_time"] - recovery_start
            ).total_seconds()
            
            # Record recovery attempt
            self.recovery_history.append(recovery_result)
            
            # Log results
            if recovery_result["status"] == "completed":
                logger.info(
                    f"Recovery completed in {recovery_result['total_duration']:.1f}s "
                    f"for {failure_type}"
                )
            else:
                logger.error(
                    f"Recovery failed after {recovery_result['total_duration']:.1f}s "
                    f"for {failure_type}"
                )
        
        return recovery_result
    
    def _system_crash_recovery(
        self,
        target_time: Optional[datetime],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from system crash."""
        
        steps = []
        recovered_data = {}
        
        # Step 1: Find most recent backup
        backups = self.backup_manager.list_backups(
            backup_type="full",
            since=target_time - timedelta(hours=24) if target_time else None
        )
        
        if not backups:
            raise GridEnvironmentError("No suitable backups found for recovery")
        
        latest_backup = backups[0]
        steps.append(f"Selected backup: {latest_backup.backup_id}")
        
        # Step 2: Restore system state
        restored_data = self.backup_manager.restore_backup(latest_backup.backup_id)
        recovered_data["system_state"] = restored_data.get("system_state", {})
        steps.append("System state restored")
        
        # Step 3: Restore configuration
        if "environment_config" in restored_data:
            recovered_data["environment_config"] = restored_data["environment_config"]
            steps.append("Environment configuration restored")
        
        # Step 4: Check for incremental backups
        incremental_backups = self.backup_manager.list_backups(
            backup_type="incremental",
            since=latest_backup.timestamp
        )
        
        for inc_backup in incremental_backups[:5]:  # Last 5 incremental
            try:
                inc_data = self.backup_manager.restore_backup(inc_backup.backup_id)
                # Merge incremental data
                for key, value in inc_data.items():
                    if key not in recovered_data:
                        recovered_data[key] = value
                steps.append(f"Applied incremental backup: {inc_backup.backup_id}")
            except Exception as e:
                steps.append(f"Failed to apply incremental {inc_backup.backup_id}: {e}")
        
        return {
            "steps_completed": steps,
            "data_recovered": recovered_data
        }
    
    def _data_corruption_recovery(
        self,
        target_time: Optional[datetime],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from data corruption."""
        
        steps = []
        recovered_data = {}
        
        # Step 1: Verify backup integrity
        integrity_results = self.backup_manager.verify_backup_integrity()
        good_backups = [
            detail["backup_id"] for detail in integrity_results["details"]
            if detail["status"] == "passed"
        ]
        
        if not good_backups:
            raise GridEnvironmentError("No uncorrupted backups available")
        
        steps.append(f"Found {len(good_backups)} uncorrupted backups")
        
        # Step 2: Select best backup
        backup_candidates = self.backup_manager.list_backups()
        best_backup = None
        
        for backup in backup_candidates:
            if backup.backup_id in good_backups:
                best_backup = backup
                break
        
        if not best_backup:
            raise GridEnvironmentError("No suitable uncorrupted backup found")
        
        steps.append(f"Selected uncorrupted backup: {best_backup.backup_id}")
        
        # Step 3: Restore from clean backup
        restored_data = self.backup_manager.restore_backup(
            best_backup.backup_id,
            verify_checksum=True
        )
        
        recovered_data = restored_data
        steps.append("Data restored from uncorrupted backup")
        
        # Step 4: Validate restored data
        validation_errors = self._validate_restored_data(recovered_data)
        if validation_errors:
            steps.append(f"Data validation warnings: {validation_errors}")
        else:
            steps.append("Restored data validation passed")
        
        return {
            "steps_completed": steps,
            "data_recovered": recovered_data
        }
    
    def _network_failure_recovery(
        self,
        target_time: Optional[datetime],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from network failure."""
        
        steps = []
        recovered_data = {}
        
        # Step 1: Switch to local-only mode
        steps.append("Switched to local-only operation mode")
        
        # Step 2: Use local recovery points
        recovery_points = self.backup_manager.get_recovery_points(critical_only=True)
        
        if recovery_points:
            latest_rp = recovery_points[0]
            recovered_data = self.backup_manager.restore_from_recovery_point(
                latest_rp["recovery_id"]
            )
            steps.append(f"Restored from recovery point: {latest_rp['recovery_id']}")
        
        # Step 3: Initialize basic services
        recovered_data["network_mode"] = "local_only"
        recovered_data["federated_learning_enabled"] = False
        steps.append("Configured for local-only operation")
        
        return {
            "steps_completed": steps,
            "data_recovered": recovered_data
        }
    
    def _storage_failure_recovery(
        self,
        target_time: Optional[datetime],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from storage failure."""
        
        steps = []
        recovered_data = {}
        
        # Step 1: Check cloud backups
        if self.backup_manager.s3_client:
            try:
                # List cloud backups
                bucket = self.backup_manager.cloud_config.get("bucket", "grid-fed-rl-backups")
                response = self.backup_manager.s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix="backups/"
                )
                
                cloud_backups = [obj["Key"] for obj in response.get("Contents", [])]
                steps.append(f"Found {len(cloud_backups)} cloud backups")
                
                # Download most recent backup
                if cloud_backups:
                    # Extract backup IDs and find most recent
                    backup_ids = [
                        key.split("/")[-1].replace(".bak", "")
                        for key in cloud_backups
                        if key.endswith(".bak")
                    ]
                    
                    if backup_ids:
                        latest_backup_id = sorted(backup_ids)[-1]  # Assume timestamp ordering
                        
                        # Download and restore
                        backup_path = self.backup_manager.backup_dir / f"backup_{latest_backup_id}.bak"
                        self.backup_manager._download_from_cloud(latest_backup_id, backup_path)
                        
                        recovered_data = self.backup_manager.restore_backup(latest_backup_id)
                        steps.append(f"Restored from cloud backup: {latest_backup_id}")
                        
            except Exception as e:
                steps.append(f"Cloud recovery failed: {e}")
                raise GridEnvironmentError(f"Storage failure recovery failed: {e}")
        
        else:
            raise GridEnvironmentError("No cloud storage configured for recovery")
        
        return {
            "steps_completed": steps,
            "data_recovered": recovered_data
        }
    
    def _complete_failure_recovery(
        self,
        target_time: Optional[datetime],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from complete system failure."""
        
        steps = []
        recovered_data = {}
        
        # Comprehensive recovery combining all strategies
        try:
            # Try cloud recovery first
            cloud_result = self._storage_failure_recovery(target_time, options)
            steps.extend(cloud_result["steps_completed"])
            recovered_data.update(cloud_result["data_recovered"])
            
        except Exception:
            # Fallback to local recovery
            try:
                crash_result = self._system_crash_recovery(target_time, options)
                steps.extend(crash_result["steps_completed"])
                recovered_data.update(crash_result["data_recovered"])
                
            except Exception:
                # Ultimate fallback - minimal system
                steps.append("Initializing minimal system configuration")
                recovered_data = self._create_minimal_system_config()
        
        return {
            "steps_completed": steps,
            "data_recovered": recovered_data
        }
    
    def _generic_recovery(
        self,
        target_time: Optional[datetime],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic recovery procedure."""
        
        # Try most common recovery approach
        return self._system_crash_recovery(target_time, options)
    
    def _validate_restored_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate restored data integrity."""
        
        issues = []
        
        # Check for required fields
        required_fields = ["system_state", "environment_config"]
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")
        
        # Check data types
        if "system_state" in data:
            state = data["system_state"]
            if not isinstance(state, dict):
                issues.append("System state is not a dictionary")
        
        return issues
    
    def _create_minimal_system_config(self) -> Dict[str, Any]:
        """Create minimal system configuration for emergency recovery."""
        
        return {
            "system_state": {
                "mode": "emergency",
                "safety_mode": True,
                "federated_learning_enabled": False,
                "monitoring_enabled": True
            },
            "environment_config": {
                "feeder_type": "IEEE13Bus",
                "safety_limits": {
                    "voltage_range": [0.9, 1.1],
                    "frequency_range": [59.5, 60.5]
                }
            }
        }
    
    def _generate_recovery_id(self) -> str:
        """Generate unique recovery ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"recovery_{timestamp}"
    
    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get disaster recovery performance metrics."""
        
        if not self.recovery_history:
            return {"no_recoveries": True}
        
        recoveries = list(self.recovery_history)
        successful = [r for r in recoveries if r["status"] == "completed"]
        
        if not successful:
            return {
                "total_attempts": len(recoveries),
                "success_rate": 0.0,
                "no_successful_recoveries": True
            }
        
        # Calculate metrics
        recovery_times = [r["total_duration"] for r in successful]
        avg_recovery_time = sum(recovery_times) / len(recovery_times)
        
        return {
            "total_attempts": len(recoveries),
            "successful_recoveries": len(successful),
            "success_rate": len(successful) / len(recoveries),
            "average_recovery_time_seconds": avg_recovery_time,
            "rto_compliance": sum(1 for rt in recovery_times if rt <= self.rto_target * 60) / len(recovery_times),
            "recovery_by_type": self._analyze_recovery_by_type(recoveries),
            "recent_performance": self._analyze_recent_performance(recoveries[-10:])  # Last 10
        }
    
    def _analyze_recovery_by_type(self, recoveries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze recovery performance by failure type."""
        
        by_type = defaultdict(lambda: {"attempts": 0, "successes": 0, "avg_time": 0})
        
        for recovery in recoveries:
            failure_type = recovery["failure_type"]
            by_type[failure_type]["attempts"] += 1
            
            if recovery["status"] == "completed":
                by_type[failure_type]["successes"] += 1
                # Update rolling average
                current_avg = by_type[failure_type]["avg_time"]
                current_successes = by_type[failure_type]["successes"]
                new_time = recovery["total_duration"]
                by_type[failure_type]["avg_time"] = (
                    current_avg * (current_successes - 1) + new_time
                ) / current_successes
        
        # Calculate success rates
        for stats in by_type.values():
            stats["success_rate"] = stats["successes"] / stats["attempts"]
        
        return dict(by_type)
    
    def _analyze_recent_performance(self, recent_recoveries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recent recovery performance."""
        
        if not recent_recoveries:
            return {"no_recent_recoveries": True}
        
        successful = [r for r in recent_recoveries if r["status"] == "completed"]
        
        return {
            "count": len(recent_recoveries),
            "success_rate": len(successful) / len(recent_recoveries),
            "avg_time_seconds": sum(r["total_duration"] for r in successful) / max(len(successful), 1),
            "failure_types": list(set(r["failure_type"] for r in recent_recoveries))
        }


# Global instances
global_backup_manager = BackupManager()
global_disaster_recovery = DisasterRecoveryManager(global_backup_manager)

logger.info("Backup and recovery systems initialized")
