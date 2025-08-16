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


# Global security components
security_policy = SecurityPolicy()
input_sanitizer = InputSanitizer(security_policy)
access_controller = AccessController(security_policy)
data_protector = DataProtector()
security_monitor = SecurityMonitor(security_policy)