"""Enhanced monitoring system with real-time alerting, dashboards, and persistence."""

import time
import threading
import json
import os
import logging
import socket
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import csv
from enum import Enum
from contextlib import contextmanager

# Optional dependencies with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
    # Mock psutil functionality
    class MockPsutil:
        def cpu_percent(self): return 50.0
        def virtual_memory(self): 
            class Memory: 
                percent = 60.0
                available = 8 * 1024**3
            return Memory()
        def disk_usage(self, path): 
            class Disk: 
                percent = 70.0
                free = 100 * 1024**3
            return Disk()
    psutil = MockPsutil()

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    sqlite3 = None

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    from urllib.request import urlopen
    from urllib.parse import urlencode
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False

from .exceptions import GridEnvironmentError, RetryableError, exponential_backoff
from .distributed_tracing import global_tracer, trace_federated_operation
from .monitoring import SystemMetrics, GridMetrics, TrainingMetrics

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    SMS = "sms"
    DASHBOARD = "dashboard"


@dataclass
class AlertRule:
    """Configuration for alert rules."""
    name: str
    metric_name: str
    condition: str  # ">", "<", "==", "!=", ">=", "<="
    threshold: float
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_seconds: int = 300  # 5 minutes default
    enabled: bool = True
    description: str = ""
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if the rule triggers an alert."""
        if not self.enabled:
            return False
            
        if self.condition == ">":
            return value > self.threshold
        elif self.condition == "<":
            return value < self.threshold
        elif self.condition == ">=":
            return value >= self.threshold
        elif self.condition == "<=":
            return value <= self.threshold
        elif self.condition == "==":
            return abs(value - self.threshold) < 1e-6
        elif self.condition == "!=":
            return abs(value - self.threshold) >= 1e-6
        else:
            return False


@dataclass
class Alert:
    """Alert instance."""
    rule_name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    metric_name: str
    value: float
    threshold: float
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp.isoformat() if self.resolved_timestamp else None,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_timestamp": self.acknowledged_timestamp.isoformat() if self.acknowledged_timestamp else None
        }


class AlertManager:
    """Manages alert rules, evaluation, and delivery."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.last_triggered: Dict[str, datetime] = {}
        self.notification_handlers: Dict[AlertChannel, Callable] = {}
        
        # Configure notification handlers
        self._setup_notification_handlers()
        
        # Default alert rules
        self._setup_default_rules()
        
        logger.info("Alert manager initialized")
    
    def _setup_default_rules(self):
        """Setup default alert rules for grid operations."""
        default_rules = [
            AlertRule(
                name="high_voltage_violation",
                metric_name="max_voltage_deviation",
                condition=">",
                threshold=0.15,  # 15% deviation
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
                description="High voltage deviation detected"
            ),
            AlertRule(
                name="critical_voltage_violation",
                metric_name="max_voltage_deviation",
                condition=">",
                threshold=0.25,  # 25% deviation
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.DASHBOARD],
                description="Critical voltage deviation - immediate attention required"
            ),
            AlertRule(
                name="high_frequency_deviation",
                metric_name="max_frequency_deviation",
                condition=">",
                threshold=1.5,  # 1.5 Hz
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
                description="High frequency deviation detected"
            ),
            AlertRule(
                name="power_flow_convergence_failure",
                metric_name="power_flow_convergence_rate",
                condition="<",
                threshold=0.90,  # Below 90%
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
                description="Power flow convergence rate below acceptable threshold"
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_percent_mean",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                description="High CPU usage detected"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_percent_mean",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                description="High memory usage - potential resource exhaustion"
            ),
            AlertRule(
                name="emergency_safety_intervention",
                metric_name="safety_interventions_per_hour",
                condition=">",
                threshold=5.0,
                severity=AlertSeverity.EMERGENCY,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK, AlertChannel.DASHBOARD],
                description="Multiple safety interventions - system may be unstable"
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def _setup_notification_handlers(self):
        """Setup notification handlers for different channels."""
        self.notification_handlers[AlertChannel.LOG] = self._send_log_alert
        self.notification_handlers[AlertChannel.EMAIL] = self._send_email_alert
        self.notification_handlers[AlertChannel.WEBHOOK] = self._send_webhook_alert
        self.notification_handlers[AlertChannel.SLACK] = self._send_slack_alert
        self.notification_handlers[AlertChannel.DASHBOARD] = self._send_dashboard_alert
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    @trace_federated_operation("evaluate_metrics", "monitoring")
    def evaluate_metrics(self, metrics: Dict[str, Any]):
        """Evaluate metrics against all alert rules."""
        current_time = datetime.now()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Extract metric value
            metric_value = self._extract_metric_value(metrics, rule.metric_name)
            
            if metric_value is None:
                continue
            
            # Check cooldown
            if rule_name in self.last_triggered:
                time_since_last = (current_time - self.last_triggered[rule_name]).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    continue
            
            # Evaluate rule
            if rule.evaluate(metric_value):
                self._trigger_alert(rule, metric_value, current_time)
            else:
                # Check if we should resolve an active alert
                if rule_name in self.active_alerts:
                    self._resolve_alert(rule_name, current_time)
    
    def _extract_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from nested metrics dictionary."""
        # Handle nested metric names like "cpu_percent.mean"
        if "." in metric_name:
            parts = metric_name.split(".")
            value = metrics
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return float(value) if isinstance(value, (int, float)) else None
        
        # Direct metric access
        if metric_name in metrics:
            value = metrics[metric_name]
            return float(value) if isinstance(value, (int, float)) else None
        
        return None
    
    def _trigger_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """Trigger an alert."""
        alert = Alert(
            rule_name=rule.name,
            message=f"{rule.description or rule.name}: {rule.metric_name} = {value:.2f} {rule.condition} {rule.threshold}",
            severity=rule.severity,
            timestamp=timestamp,
            metric_name=rule.metric_name,
            value=value,
            threshold=rule.threshold
        )
        
        # Add to active alerts
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        self.last_triggered[rule.name] = timestamp
        
        # Send notifications
        for channel in rule.channels:
            try:
                if channel in self.notification_handlers:
                    self.notification_handlers[channel](alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
        
        logger.warning(f"ALERT TRIGGERED: {alert.message}")
    
    def _resolve_alert(self, rule_name: str, timestamp: datetime):
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_timestamp = timestamp
            
            # Remove from active alerts
            del self.active_alerts[rule_name]
            
            logger.info(f"ALERT RESOLVED: {rule_name}")
    
    def acknowledge_alert(self, rule_name: str, acknowledged_by: str):
        """Acknowledge an alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_timestamp = datetime.now()
            
            logger.info(f"Alert acknowledged: {rule_name} by {acknowledged_by}")
    
    def _send_log_alert(self, alert: Alert):
        """Send alert to log."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
    
    @exponential_backoff(max_retries=3, base_delay=1.0)
    def _send_email_alert(self, alert: Alert):
        """Send alert via email."""
        email_config = self.config.get('email', {})
        
        if not email_config.get('enabled', False):
            return
        
        smtp_server = email_config.get('smtp_server', 'localhost')
        smtp_port = email_config.get('smtp_port', 587)
        username = email_config.get('username')
        password = email_config.get('password')
        from_email = email_config.get('from_email', 'grid-fed-rl@localhost')
        to_emails = email_config.get('to_emails', [])
        
        if not to_emails:
            logger.warning("No email recipients configured")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = ', '.join(to_emails)
        msg['Subject'] = f"[{alert.severity.value.upper()}] Grid-Fed-RL Alert: {alert.rule_name}"
        
        body = f"""
        Alert Details:
        - Rule: {alert.rule_name}
        - Severity: {alert.severity.value.upper()}
        - Metric: {alert.metric_name}
        - Value: {alert.value:.2f}
        - Threshold: {alert.threshold}
        - Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        - Message: {alert.message}
        
        This is an automated alert from the Grid-Fed-RL monitoring system.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.rule_name}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise RetryableError(f"Email delivery failed: {e}")
    
    @exponential_backoff(max_retries=3, base_delay=0.5)
    def _send_webhook_alert(self, alert: Alert):
        """Send alert via webhook."""
        webhook_config = self.config.get('webhook', {})
        
        if not webhook_config.get('enabled', False):
            return
        
        webhook_url = webhook_config.get('url')
        if not webhook_url:
            return
        
        # Prepare payload
        payload = {
            'alert': alert.to_dict(),
            'timestamp': alert.timestamp.isoformat(),
            'service': 'grid-fed-rl'
        }
        
        # Send webhook
        try:
            import urllib.request
            import urllib.parse
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(webhook_url, data=data)
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info(f"Webhook alert sent for {alert.rule_name}")
                else:
                    logger.warning(f"Webhook returned status {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            raise RetryableError(f"Webhook delivery failed: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack."""
        # Placeholder for Slack integration
        logger.info(f"Slack alert (placeholder): {alert.message}")
    
    def _send_dashboard_alert(self, alert: Alert):
        """Send alert to dashboard."""
        # Dashboard alerts are handled by storing in active_alerts
        pass
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return list(self.alert_history)[-limit:]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_hour = now - timedelta(hours=1)
        
        recent_alerts = [a for a in self.alert_history if a.timestamp >= last_24h]
        recent_alerts_1h = [a for a in self.alert_history if a.timestamp >= last_hour]
        
        by_severity = defaultdict(int)
        for alert in recent_alerts:
            by_severity[alert.severity.value] += 1
        
        return {
            "active_alerts": len(self.active_alerts),
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
            "alerts_last_24h": len(recent_alerts),
            "alerts_last_hour": len(recent_alerts_1h),
            "alerts_by_severity_24h": dict(by_severity),
            "mean_time_to_resolve": self._calculate_mttr()
        }
    
    def _calculate_mttr(self) -> float:
        """Calculate mean time to resolution for resolved alerts."""
        resolved_alerts = [a for a in self.alert_history if a.resolved and a.resolved_timestamp]
        
        if not resolved_alerts:
            return 0.0
        
        total_time = sum(
            (alert.resolved_timestamp - alert.timestamp).total_seconds()
            for alert in resolved_alerts
        )
        
        return total_time / len(resolved_alerts)


class PersistentMetricsStore:
    """Persistent storage for metrics using SQLite."""
    
    def __init__(self, db_path: str = "grid_metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with tables."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    step_count INTEGER,
                    power_flow_time_ms REAL,
                    constraint_violations INTEGER,
                    safety_interventions INTEGER,
                    average_voltage REAL,
                    frequency_deviation REAL,
                    total_losses REAL,
                    renewable_utilization REAL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_mb REAL,
                    memory_available_mb REAL,
                    disk_usage_percent REAL,
                    network_bytes_sent INTEGER,
                    network_bytes_recv INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT,
                    message TEXT,
                    severity TEXT,
                    timestamp TEXT,
                    metric_name TEXT,
                    value REAL,
                    threshold REAL,
                    resolved BOOLEAN,
                    resolved_timestamp TEXT,
                    acknowledged BOOLEAN,
                    acknowledged_by TEXT,
                    acknowledged_timestamp TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def store_metrics(self, metrics: SystemMetrics):
        """Store metrics in database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO system_metrics (
                    timestamp, step_count, power_flow_time_ms, constraint_violations,
                    safety_interventions, average_voltage, frequency_deviation, total_losses,
                    renewable_utilization, cpu_percent, memory_percent, memory_used_mb,
                    memory_available_mb, disk_usage_percent, network_bytes_sent, network_bytes_recv
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.step_count, metrics.power_flow_time_ms,
                metrics.constraint_violations, metrics.safety_interventions,
                metrics.average_voltage, metrics.frequency_deviation, metrics.total_losses,
                metrics.renewable_utilization, metrics.cpu_percent, metrics.memory_percent,
                metrics.memory_used_mb, metrics.memory_available_mb, metrics.disk_usage_percent,
                metrics.network_bytes_sent, metrics.network_bytes_recv
            ))
    
    def store_alert(self, alert: Alert):
        """Store alert in database."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO alerts (
                    rule_name, message, severity, timestamp, metric_name, value,
                    threshold, resolved, resolved_timestamp, acknowledged,
                    acknowledged_by, acknowledged_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.rule_name, alert.message, alert.severity.value,
                alert.timestamp.isoformat(), alert.metric_name, alert.value,
                alert.threshold, alert.resolved,
                alert.resolved_timestamp.isoformat() if alert.resolved_timestamp else None,
                alert.acknowledged, alert.acknowledged_by,
                alert.acknowledged_timestamp.isoformat() if alert.acknowledged_timestamp else None
            ))
    
    def get_metrics_range(self, start_time: float, end_time: float, limit: int = 1000) -> List[Dict]:
        """Get metrics within time range."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM system_metrics 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (start_time, end_time, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to prevent database growth."""
        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        
        with self._get_connection() as conn:
            # Delete old metrics
            cursor = conn.execute(
                "DELETE FROM system_metrics WHERE timestamp < ?", (cutoff_time,)
            )
            metrics_deleted = cursor.rowcount
            
            # Delete old resolved alerts
            cursor = conn.execute(
                "DELETE FROM alerts WHERE timestamp < ? AND resolved = 1", 
                (datetime.fromtimestamp(cutoff_time).isoformat(),)
            )
            alerts_deleted = cursor.rowcount
            
        logger.info(f"Cleaned up {metrics_deleted} old metrics and {alerts_deleted} old alerts")


class RealTimeDashboard:
    """Real-time dashboard for monitoring grid operations."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.dashboard_data = {
            "system_status": "unknown",
            "active_alerts": [],
            "key_metrics": {},
            "performance_trends": {},
            "last_updated": None
        }
        self.subscribers: Set[Callable] = set()
        self.running = False
        self.update_thread = None
    
    def subscribe(self, callback: Callable[[Dict], None]):
        """Subscribe to dashboard updates."""
        self.subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from dashboard updates."""
        self.subscribers.discard(callback)
    
    def update_data(self, metrics: Dict[str, Any], alerts: List[Alert]):
        """Update dashboard data."""
        # Determine system status
        if any(alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] for alert in alerts):
            system_status = "critical"
        elif any(alert.severity == AlertSeverity.ERROR for alert in alerts):
            system_status = "error"
        elif any(alert.severity == AlertSeverity.WARNING for alert in alerts):
            system_status = "warning"
        elif alerts:
            system_status = "info"
        else:
            system_status = "healthy"
        
        # Extract key metrics
        key_metrics = {
            "voltage_status": self._get_voltage_status(metrics),
            "frequency_status": self._get_frequency_status(metrics),
            "power_flow_status": self._get_power_flow_status(metrics),
            "system_load": self._get_system_load(metrics),
            "renewable_utilization": metrics.get("recent_renewable_utilization", 0.0)
        }
        
        # Update dashboard data
        self.dashboard_data.update({
            "system_status": system_status,
            "active_alerts": [alert.to_dict() for alert in alerts],
            "key_metrics": key_metrics,
            "last_updated": datetime.now().isoformat()
        })
        
        # Notify subscribers
        self._notify_subscribers()
    
    def _get_voltage_status(self, metrics: Dict) -> Dict[str, Any]:
        """Extract voltage status from metrics."""
        max_deviation = metrics.get("max_voltage_deviation", 0.0)
        avg_deviation = metrics.get("avg_voltage_deviation", 0.0)
        
        if max_deviation > 0.2:
            status = "critical"
        elif max_deviation > 0.1:
            status = "warning"
        else:
            status = "good"
        
        return {
            "status": status,
            "max_deviation": max_deviation,
            "avg_deviation": avg_deviation
        }
    
    def _get_frequency_status(self, metrics: Dict) -> Dict[str, Any]:
        """Extract frequency status from metrics."""
        max_deviation = metrics.get("max_frequency_deviation", 0.0)
        avg_deviation = metrics.get("avg_frequency_deviation", 0.0)
        
        if max_deviation > 2.0:
            status = "critical"
        elif max_deviation > 1.0:
            status = "warning"
        else:
            status = "good"
        
        return {
            "status": status,
            "max_deviation": max_deviation,
            "avg_deviation": avg_deviation
        }
    
    def _get_power_flow_status(self, metrics: Dict) -> Dict[str, Any]:
        """Extract power flow status from metrics."""
        avg_time = metrics.get("avg_power_flow_time_ms", 0.0)
        max_time = metrics.get("max_power_flow_time_ms", 0.0)
        
        if max_time > 500:
            status = "slow"
        elif avg_time > 100:
            status = "moderate"
        else:
            status = "fast"
        
        return {
            "status": status,
            "avg_time_ms": avg_time,
            "max_time_ms": max_time
        }
    
    def _get_system_load(self, metrics: Dict) -> Dict[str, Any]:
        """Extract system load from metrics."""
        cpu_stats = metrics.get("cpu_percent", {})
        memory_stats = metrics.get("memory_percent", {})
        
        cpu_mean = cpu_stats.get("mean", 0) if isinstance(cpu_stats, dict) else 0
        memory_mean = memory_stats.get("mean", 0) if isinstance(memory_stats, dict) else 0
        
        if cpu_mean > 90 or memory_mean > 90:
            status = "high"
        elif cpu_mean > 70 or memory_mean > 70:
            status = "moderate"
        else:
            status = "low"
        
        return {
            "status": status,
            "cpu_percent": cpu_mean,
            "memory_percent": memory_mean
        }
    
    def _notify_subscribers(self):
        """Notify all subscribers of data updates."""
        for callback in self.subscribers.copy():  # Copy to avoid modification during iteration
            try:
                callback(self.dashboard_data)
            except Exception as e:
                logger.error(f"Error notifying dashboard subscriber: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data.copy()


# Integration with existing GridMonitor
class EnhancedGridMonitor:
    """Enhanced grid monitor with alerting and dashboard capabilities."""
    
    def __init__(
        self,
        base_monitor,
        alert_config: Optional[Dict[str, Any]] = None,
        enable_persistence: bool = True,
        enable_dashboard: bool = True
    ):
        self.base_monitor = base_monitor
        self.alert_manager = AlertManager(alert_config)
        self.enable_persistence = enable_persistence
        self.enable_dashboard = enable_dashboard
        
        # Initialize persistence
        if self.enable_persistence:
            self.metrics_store = PersistentMetricsStore()
        
        # Initialize dashboard
        if self.enable_dashboard:
            self.dashboard = RealTimeDashboard()
        
        # Override base monitor's record_metrics to include alerting
        self._wrap_base_monitor()
        
        logger.info("Enhanced grid monitor initialized")
    
    def _wrap_base_monitor(self):
        """Wrap base monitor methods to add alerting."""
        original_record_metrics = self.base_monitor.record_metrics
        
        def enhanced_record_metrics(*args, **kwargs):
            # Call original method
            metrics = original_record_metrics(*args, **kwargs)
            
            # Store in persistent storage
            if self.enable_persistence:
                try:
                    self.metrics_store.store_metrics(metrics)
                except Exception as e:
                    logger.error(f"Failed to store metrics: {e}")
            
            # Get summary stats for alerting
            try:
                summary_stats = self.base_monitor.get_summary_stats()
                self.alert_manager.evaluate_metrics(summary_stats)
                
                # Update dashboard
                if self.enable_dashboard:
                    active_alerts = self.alert_manager.get_active_alerts()
                    self.dashboard.update_data(summary_stats, active_alerts)
                
            except Exception as e:
                logger.error(f"Failed to evaluate alerts: {e}")
            
            return metrics
        
        # Replace method
        self.base_monitor.record_metrics = enhanced_record_metrics
    
    def get_alert_manager(self) -> AlertManager:
        """Get the alert manager."""
        return self.alert_manager
    
    def get_dashboard(self) -> Optional[RealTimeDashboard]:
        """Get the dashboard."""
        return self.dashboard if self.enable_dashboard else None
    
    def get_metrics_store(self) -> Optional[PersistentMetricsStore]:
        """Get the metrics store."""
        return self.metrics_store if self.enable_persistence else None
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Cleanup old data."""
        if self.enable_persistence:
            self.metrics_store.cleanup_old_data(days_to_keep)