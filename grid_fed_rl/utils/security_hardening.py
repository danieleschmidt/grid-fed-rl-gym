"""Security hardening and data protection for grid operations."""

import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    max_action_deviation: float = 2.0  # Maximum allowed action deviation
    max_consecutive_errors: int = 5     # Maximum consecutive errors before lockout
    input_validation_strict: bool = True
    log_security_events: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 1000


class InputSanitizer:
    """Sanitize and validate inputs for security."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.failed_attempts = {}
        
    def sanitize_action(self, action: Any) -> Any:
        """Sanitize action input to prevent injection attacks."""
        try:
            # Convert to safe numeric format
            if hasattr(action, '__iter__') and not isinstance(action, str):
                # Handle iterable actions
                sanitized = []
                for item in action:
                    if isinstance(item, (int, float)):
                        # Clamp to reasonable range
                        safe_item = max(-100.0, min(100.0, float(item)))
                        sanitized.append(safe_item)
                    else:
                        logger.warning(f"Non-numeric action item: {item}, replacing with 0.0")
                        sanitized.append(0.0)
                return sanitized
            else:
                # Handle single value actions
                if isinstance(action, (int, float)):
                    return max(-100.0, min(100.0, float(action)))
                else:
                    logger.warning(f"Non-numeric action: {action}, replacing with 0.0")
                    return 0.0
                    
        except Exception as e:
            logger.error(f"Action sanitization failed: {e}")
            return 0.0  # Safe default
            
    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration parameters."""
        sanitized = {}
        
        # Whitelist of allowed config keys
        allowed_keys = {
            'timestep', 'episode_length', 'voltage_limits', 'frequency_limits',
            'safety_penalty', 'stochastic_loads', 'renewable_sources', 
            'weather_variation'
        }
        
        for key, value in config.items():
            if key in allowed_keys:
                # Validate specific parameters
                if key == 'timestep' and isinstance(value, (int, float)):
                    sanitized[key] = max(0.001, min(10.0, float(value)))
                elif key == 'episode_length' and isinstance(value, int):
                    sanitized[key] = max(1, min(1000000, int(value)))
                elif key == 'voltage_limits' and isinstance(value, (list, tuple)) and len(value) == 2:
                    sanitized[key] = (max(0.5, min(0.9, float(value[0]))), 
                                    max(1.1, min(2.0, float(value[1]))))
                elif key == 'safety_penalty' and isinstance(value, (int, float)):
                    sanitized[key] = max(0.0, min(1000.0, float(value)))
                else:
                    sanitized[key] = value
            else:
                logger.warning(f"Unauthorized config key ignored: {key}")
                
        return sanitized
        
    def validate_network_data(self, data: Dict[str, Any]) -> bool:
        """Validate network topology data for security."""
        try:
            # Check for required fields
            required_fields = ['buses', 'lines']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False
                    
            # Validate data types and ranges
            if 'buses' in data:
                buses = data['buses']
                if not isinstance(buses, list) or len(buses) > 1000:  # Prevent memory attacks
                    logger.error("Invalid bus data")
                    return False
                    
            if 'lines' in data:
                lines = data['lines'] 
                if not isinstance(lines, list) or len(lines) > 2000:  # Prevent memory attacks
                    logger.error("Invalid line data")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Network data validation failed: {e}")
            return False


class AccessController:
    """Control access to sensitive operations."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.access_log = []
        self.rate_limits = {}
        
    def check_rate_limit(self, operation: str, client_id: str = "default") -> bool:
        """Check if operation is within rate limits."""
        if not self.policy.enable_rate_limiting:
            return True
            
        current_time = __import__('time').time()
        key = f"{client_id}:{operation}"
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute window
        if key in self.rate_limits:
            self.rate_limits[key] = [t for t in self.rate_limits[key] if t > cutoff_time]
        else:
            self.rate_limits[key] = []
            
        # Check limit
        if len(self.rate_limits[key]) >= self.policy.max_requests_per_minute:
            logger.warning(f"Rate limit exceeded for {operation} by {client_id}")
            return False
            
        # Record this request
        self.rate_limits[key].append(current_time)
        return True
        
    def log_access(self, operation: str, client_id: str, success: bool, details: str = ""):
        """Log access attempt."""
        if self.policy.log_security_events:
            log_entry = {
                'timestamp': __import__('datetime').datetime.now().isoformat(),
                'operation': operation,
                'client_id': client_id,
                'success': success,
                'details': details
            }
            self.access_log.append(log_entry)
            
            # Keep only last 10000 entries
            if len(self.access_log) > 10000:
                self.access_log = self.access_log[-10000:]
                
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security monitoring summary."""
        if not self.access_log:
            return {"message": "No access events logged"}
            
        total_events = len(self.access_log)
        failed_events = sum(1 for entry in self.access_log if not entry['success'])
        
        # Recent events
        recent_events = self.access_log[-10:] if self.access_log else []
        
        return {
            'total_events': total_events,
            'failed_events': failed_events,
            'failure_rate': failed_events / total_events if total_events > 0 else 0,
            'recent_events': recent_events,
            'active_rate_limits': len(self.rate_limits),
            'report_time': __import__('datetime').datetime.now().isoformat()
        }


class DataProtector:
    """Protect sensitive data and configuration."""
    
    def __init__(self):
        self.sensitive_keys = {
            'api_key', 'password', 'secret', 'token', 'private_key',
            'credential', 'auth', 'database_url'
        }
        
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive information in data."""
        masked = {}
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_keys):
                if isinstance(value, str) and len(value) > 4:
                    masked[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    masked[key] = '***'
            else:
                masked[key] = value
                
        return masked
        
    def calculate_checksum(self, data: str) -> str:
        """Calculate SHA-256 checksum for data integrity."""
        return hashlib.sha256(data.encode()).hexdigest()
        
    def verify_integrity(self, data: str, expected_checksum: str) -> bool:
        """Verify data integrity using checksum."""
        actual_checksum = self.calculate_checksum(data)
        return secrets.compare_digest(actual_checksum, expected_checksum)
        
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_hex(length)


class SecurityMonitor:
    """Monitor system for security threats."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.threat_indicators = []
        self.blocked_operations = 0
        
    def detect_anomalous_actions(self, actions: List[float], baseline_mean: float = 0.0, baseline_std: float = 1.0) -> bool:
        """Detect anomalous action patterns that might indicate attacks."""
        try:
            if not actions:
                return False
                
            # Calculate action statistics
            action_mean = sum(actions) / len(actions)
            action_variance = sum((x - action_mean) ** 2 for x in actions) / len(actions)
            action_std = action_variance ** 0.5
            
            # Check for statistical anomalies
            if abs(action_mean - baseline_mean) > self.policy.max_action_deviation:
                self._log_threat("Anomalous action mean detected", {
                    'action_mean': action_mean,
                    'baseline_mean': baseline_mean,
                    'deviation': abs(action_mean - baseline_mean)
                })
                return True
                
            # Check for extreme values
            for action in actions:
                if abs(action) > 10.0:  # Extreme value threshold
                    self._log_threat("Extreme action value detected", {
                        'action_value': action,
                        'threshold': 10.0
                    })
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return True  # Err on the side of caution
            
    def detect_replay_attack(self, current_actions: List[float], recent_actions: List[List[float]]) -> bool:
        """Detect potential replay attacks."""
        try:
            if not recent_actions or not current_actions:
                return False
                
            # Check for exact matches with recent actions
            for past_actions in recent_actions:
                if len(past_actions) == len(current_actions):
                    if all(abs(a - b) < 1e-6 for a, b in zip(current_actions, past_actions)):
                        self._log_threat("Potential replay attack detected", {
                            'current_actions': current_actions,
                            'matching_past_actions': past_actions
                        })
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Replay attack detection failed: {e}")
            return False
            
    def _log_threat(self, threat_type: str, details: Dict[str, Any]):
        """Log security threat."""
        threat = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'type': threat_type,
            'details': details,
            'severity': 'high'
        }
        self.threat_indicators.append(threat)
        logger.warning(f"Security threat detected: {threat_type}")
        
        # Keep only last 1000 threats
        if len(self.threat_indicators) > 1000:
            self.threat_indicators = self.threat_indicators[-1000:]
            
    def get_threat_summary(self) -> Dict[str, Any]:
        """Get threat monitoring summary."""
        return {
            'total_threats': len(self.threat_indicators),
            'blocked_operations': self.blocked_operations,
            'recent_threats': self.threat_indicators[-5:],
            'threat_types': list(set(t['type'] for t in self.threat_indicators)),
            'report_time': __import__('datetime').datetime.now().isoformat()
        }


# Enhanced security hardening with validation integration
class EnhancedSecurityHardening:
    """Enhanced security hardening with comprehensive validation."""
    
    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy()
        self.input_sanitizer = InputSanitizer(self.policy)
        self.access_controller = AccessController(self.policy)
        self.data_protector = DataProtector()
        self.security_monitor = SecurityMonitor(self.policy)
        
        # Integration with security validation
        self.validation_integrated = False
        self._attempt_validation_integration()
    
    def _attempt_validation_integration(self):
        """Attempt to integrate with comprehensive security validation."""
        try:
            from .security_validation import get_security_validation_suite
            self.validation_suite = get_security_validation_suite()
            self.validation_integrated = True
            logger.info("Security hardening integrated with validation suite")
        except ImportError:
            logger.info("Security validation suite not available for hardening integration")
            self.validation_suite = None
        except Exception as e:
            logger.warning(f"Failed to integrate with validation suite: {e}")
            self.validation_suite = None
    
    def get_hardening_status(self) -> Dict[str, Any]:
        """Get comprehensive hardening status."""
        status = {
            "policy": {
                "max_action_deviation": self.policy.max_action_deviation,
                "max_consecutive_errors": self.policy.max_consecutive_errors,
                "input_validation_strict": self.policy.input_validation_strict,
                "rate_limiting_enabled": self.policy.enable_rate_limiting,
                "max_requests_per_minute": self.policy.max_requests_per_minute
            },
            "security_summary": self.access_controller.get_security_summary(),
            "threat_summary": self.security_monitor.get_threat_summary(),
            "validation_integrated": self.validation_integrated,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        # Add validation suite status if available
        if self.validation_integrated and self.validation_suite:
            try:
                validation_status = {
                    "security_score": self.validation_suite.metrics.security_score,
                    "total_findings": self.validation_suite.metrics.total_findings,
                    "critical_findings": self.validation_suite.metrics.critical_findings,
                    "last_scan": self.validation_suite.metrics.last_scan_time.isoformat() if self.validation_suite.metrics.last_scan_time else None
                }
                status["validation_status"] = validation_status
            except Exception as e:
                logger.error(f"Error getting validation status: {e}")
        
        return status
    
    def apply_security_hardening(self, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply security hardening to configuration."""
        hardened_config = target_config.copy()
        
        # Sanitize configuration
        hardened_config = self.input_sanitizer.sanitize_config(hardened_config)
        
        # Apply additional hardening measures
        hardening_applied = []
        
        # Enforce secure defaults
        if "session_timeout" not in hardened_config:
            hardened_config["session_timeout"] = 3600  # 1 hour default
            hardening_applied.append("Added default session timeout")
        
        if "encryption_enabled" not in hardened_config:
            hardened_config["encryption_enabled"] = True
            hardening_applied.append("Enabled encryption by default")
        
        if "audit_logging" not in hardened_config:
            hardened_config["audit_logging"] = True
            hardening_applied.append("Enabled audit logging")
        
        # Remove potentially dangerous configurations
        dangerous_keys = ["debug_mode", "disable_auth", "allow_all_origins", "insecure_mode"]
        for key in dangerous_keys:
            if key in hardened_config:
                del hardened_config[key]
                hardening_applied.append(f"Removed dangerous configuration: {key}")
        
        # Log hardening actions
        if hardening_applied:
            self.access_controller.log_access(
                "config_hardening", 
                "system", 
                True, 
                f"Applied {len(hardening_applied)} hardening measures"
            )
        
        return {
            "hardened_config": hardened_config,
            "hardening_applied": hardening_applied,
            "original_keys": list(target_config.keys()),
            "hardened_keys": list(hardened_config.keys())
        }
    
    def validate_security_posture(self) -> Dict[str, Any]:
        """Validate current security posture."""
        validation_results = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "hardening_status": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        # Check policy configuration
        if self.policy.max_action_deviation > 5.0:
            validation_results["issues"].append({
                "severity": "medium",
                "description": f"High action deviation threshold: {self.policy.max_action_deviation}",
                "recommendation": "Consider lowering max_action_deviation for better security"
            })
        
        if self.policy.max_requests_per_minute > 10000:
            validation_results["issues"].append({
                "severity": "low", 
                "description": f"High rate limit: {self.policy.max_requests_per_minute}/min",
                "recommendation": "Consider lowering rate limits to prevent abuse"
            })
        
        if not self.policy.input_validation_strict:
            validation_results["issues"].append({
                "severity": "high",
                "description": "Strict input validation is disabled",
                "recommendation": "Enable strict input validation for better security"
            })
        
        # Check recent security events
        security_summary = self.access_controller.get_security_summary()
        if isinstance(security_summary, dict) and security_summary.get("failure_rate", 0) > 0.1:
            validation_results["issues"].append({
                "severity": "high",
                "description": f"High security failure rate: {security_summary['failure_rate']:.1%}",
                "recommendation": "Investigate recent security failures"
            })
        
        # Check threat indicators
        threat_summary = self.security_monitor.get_threat_summary()
        if isinstance(threat_summary, dict) and threat_summary.get("total_threats", 0) > 10:
            validation_results["issues"].append({
                "severity": "critical",
                "description": f"Multiple security threats detected: {threat_summary['total_threats']}",
                "recommendation": "Immediate investigation of security threats required"
            })
        
        # Set overall status
        critical_issues = sum(1 for issue in validation_results["issues"] if issue["severity"] == "critical")
        high_issues = sum(1 for issue in validation_results["issues"] if issue["severity"] == "high")
        
        if critical_issues > 0:
            validation_results["hardening_status"] = "critical"
        elif high_issues > 0:
            validation_results["hardening_status"] = "warning"
        
        # Add validation suite results if available
        if self.validation_integrated and self.validation_suite:
            try:
                benchmarks = self.validation_suite.get_security_benchmarks()
                validation_results["security_benchmarks"] = benchmarks
            except Exception as e:
                logger.error(f"Error getting security benchmarks: {e}")
        
        return validation_results
    
    def get_real_time_security_metrics(self) -> Dict[str, Any]:
        """Get real-time security metrics for monitoring."""
        metrics = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "access_control": {
                "total_events": len(self.access_controller.access_log),
                "recent_failures": sum(1 for event in self.access_controller.access_log[-100:] 
                                     if not event.get("success", True)),
                "active_rate_limits": len(self.access_controller.rate_limits)
            },
            "threat_monitoring": {
                "total_threats": len(self.security_monitor.threat_indicators),
                "recent_threats": len([t for t in self.security_monitor.threat_indicators 
                                     if (datetime.now() - 
                                         datetime.fromisoformat(t.get("timestamp", "1970-01-01T00:00:00"))).days < 1]),
                "blocked_operations": self.security_monitor.blocked_operations
            }
        }
        
        # Add validation suite metrics if available
        if self.validation_integrated and self.validation_suite:
            try:
                suite_metrics = self.validation_suite.metrics
                metrics["validation_suite"] = {
                    "security_score": suite_metrics.security_score,
                    "total_scans": suite_metrics.total_scans,
                    "critical_findings": suite_metrics.critical_findings,
                    "high_findings": suite_metrics.high_findings,
                    "last_scan": suite_metrics.last_scan_time.isoformat() if suite_metrics.last_scan_time else None
                }
            except Exception as e:
                logger.error(f"Error getting validation suite metrics: {e}")
        
        return metrics


def run_security_hardening_check() -> Dict[str, Any]:
    """Run comprehensive security hardening check."""
    hardening = EnhancedSecurityHardening()
    
    results = {
        "hardening_status": hardening.get_hardening_status(),
        "security_posture": hardening.validate_security_posture(),
        "real_time_metrics": hardening.get_real_time_security_metrics(),
        "recommendations": []
    }
    
    # Generate recommendations based on findings
    security_posture = results["security_posture"]
    if security_posture["hardening_status"] != "healthy":
        results["recommendations"].append(
            "Review and address security hardening issues identified in the security posture check"
        )
    
    threat_count = results["real_time_metrics"]["threat_monitoring"]["total_threats"]
    if threat_count > 0:
        results["recommendations"].append(
            f"Investigate {threat_count} security threats detected by the monitoring system"
        )
    
    if not results["hardening_status"]["validation_integrated"]:
        results["recommendations"].append(
            "Consider integrating comprehensive security validation suite for enhanced security monitoring"
        )
    
    return results


# Global security components with enhanced integration
security_policy = SecurityPolicy()
input_sanitizer = InputSanitizer(security_policy)
access_controller = AccessController(security_policy)
data_protector = DataProtector()
security_monitor = SecurityMonitor(security_policy)

# Enhanced security hardening instance
enhanced_security_hardening = EnhancedSecurityHardening(security_policy)