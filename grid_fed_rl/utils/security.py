"""Security utilities and vulnerability scanning for grid systems."""

import re
import ast
import os
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    severity: str  # critical, high, medium, low
    issue_type: str
    description: str
    location: Optional[str] = None
    recommendation: str = ""


class InputValidator:
    """Comprehensive input validation for security."""
    
    @staticmethod
    def validate_numeric_input(value: Any, min_val: float = None, max_val: float = None, 
                             allow_nan: bool = False, allow_inf: bool = False) -> Tuple[bool, str]:
        """Validate numeric input with security constraints."""
        
        # Type check
        if not isinstance(value, (int, float, np.number)):
            if isinstance(value, (list, tuple, np.ndarray)):
                try:
                    value = np.array(value, dtype=float)
                    if value.size == 0:
                        return False, "Empty array not allowed"
                except (ValueError, TypeError):
                    return False, "Invalid numeric data in array"
            else:
                return False, f"Invalid numeric type: {type(value)}"
        
        # Convert to numpy for consistent checking
        if not isinstance(value, np.ndarray):
            value = np.array([value])
        
        # Check for NaN
        if not allow_nan and np.any(np.isnan(value)):
            return False, "NaN values not allowed"
        
        # Check for infinity
        if not allow_inf and np.any(np.isinf(value)):
            return False, "Infinite values not allowed"
        
        # Range validation
        if min_val is not None and np.any(value < min_val):
            return False, f"Value(s) below minimum: {min_val}"
        
        if max_val is not None and np.any(value > max_val):
            return False, f"Value(s) above maximum: {max_val}"
        
        # Check for extremely large values that could cause overflow
        if np.any(np.abs(value) > 1e15):
            return False, "Values too large, potential overflow risk"
        
        return True, "Valid"
    
    @staticmethod
    def validate_array_shape(array: np.ndarray, expected_shape: tuple, 
                           max_elements: int = 1000000) -> Tuple[bool, str]:
        """Validate array shape and size for security."""
        
        if not isinstance(array, np.ndarray):
            return False, "Input is not a numpy array"
        
        # Check shape
        if expected_shape is not None and array.shape != expected_shape:
            return False, f"Shape mismatch: expected {expected_shape}, got {array.shape}"
        
        # Check total elements to prevent memory exhaustion
        total_elements = array.size
        if total_elements > max_elements:
            return False, f"Array too large: {total_elements} elements > {max_elements}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_string_input(text: str, max_length: int = 1000, 
                            allowed_chars: Optional[str] = None) -> Tuple[bool, str]:
        """Validate string input for security."""
        
        if not isinstance(text, str):
            return False, "Input is not a string"
        
        # Length check
        if len(text) > max_length:
            return False, f"String too long: {len(text)} > {max_length}"
        
        # Character validation
        if allowed_chars is not None:
            pattern = f"^[{re.escape(allowed_chars)}]*$"
            if not re.match(pattern, text):
                return False, "String contains invalid characters"
        
        # Check for potential injection patterns
        suspicious_patterns = [
            r'<script.*?>', r'javascript:', r'vbscript:', r'onload=', r'onerror=',
            r'eval\(', r'exec\(', r'import\s+', r'__.*__', r'\.\./', r'[;&|`]'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Suspicious pattern detected: {pattern}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_file_path(path: str, allowed_extensions: List[str] = None,
                          base_directory: str = None) -> Tuple[bool, str]:
        """Validate file path for security."""
        
        if not isinstance(path, str):
            return False, "Path is not a string"
        
        # Check for path traversal attempts
        if '..' in path or '~' in path:
            return False, "Path traversal attempt detected"
        
        # Check for absolute paths if base directory is specified
        if base_directory and os.path.isabs(path):
            return False, "Absolute paths not allowed"
        
        # Resolve path and check if it's within allowed directory
        if base_directory:
            try:
                full_path = os.path.abspath(os.path.join(base_directory, path))
                base_path = os.path.abspath(base_directory)
                
                if not full_path.startswith(base_path):
                    return False, "Path outside allowed directory"
            except (OSError, ValueError):
                return False, "Invalid path"
        
        # Check file extension
        if allowed_extensions:
            _, ext = os.path.splitext(path.lower())
            if ext not in [e.lower() for e in allowed_extensions]:
                return False, f"Invalid file extension: {ext}"
        
        return True, "Valid"


class CodeSecurityScanner:
    """Security scanner for code analysis."""
    
    def __init__(self):
        self.issues = []
        
        # Patterns for potential security issues
        self.security_patterns = {
            'critical': [
                (r'eval\s*\(', 'Use of eval() can execute arbitrary code'),
                (r'exec\s*\(', 'Use of exec() can execute arbitrary code'),
                (r'__import__\s*\(', 'Dynamic imports can be dangerous'),
                (r'subprocess\.call\s*\(', 'Subprocess calls need input validation'),
                (r'os\.system\s*\(', 'OS system calls are dangerous'),
            ],
            'high': [
                (r'pickle\.loads?\s*\(', 'Pickle deserialization can execute code'),
                (r'yaml\.load\s*\(', 'YAML load without safe_load is dangerous'),
                (r'input\s*\(', 'Raw input() can be dangerous in Python 2'),
                (r'open\s*\([^,]*[\'"]w', 'File write operations need validation'),
            ],
            'medium': [
                (r'random\.seed\s*\(', 'Random seed should use secure source'),
                (r'hashlib\.md5\s*\(', 'MD5 is cryptographically broken'),
                (r'hashlib\.sha1\s*\(', 'SHA1 is deprecated for security'),
                (r'ssl\..*PROTOCOL_TLS', 'Use specific TLS version'),
            ],
            'low': [
                (r'print\s*\(.*password', 'Potential password in print statement'),
                (r'logging\..*password', 'Potential password in logs'),
                (r'# TODO.*security', 'Security-related TODO item'),
                (r'# FIXME.*security', 'Security-related FIXME item'),
            ]
        }
    
    def scan_code(self, code: str, filename: str = "<string>") -> List[SecurityIssue]:
        """Scan code for security issues."""
        issues = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for severity, patterns in self.security_patterns.items():
                for pattern, description in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity=severity,
                            issue_type='code_analysis',
                            description=description,
                            location=f"{filename}:{line_num}",
                            recommendation=self._get_recommendation(pattern)
                        ))
        
        return issues
    
    def scan_file(self, filepath: str) -> List[SecurityIssue]:
        """Scan a file for security issues."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.scan_code(code, filepath)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to scan file {filepath}: {e}")
            return []
    
    def _get_recommendation(self, pattern: str) -> str:
        """Get security recommendation for a pattern."""
        recommendations = {
            r'eval\s*\(': 'Use ast.literal_eval() for safe evaluation',
            r'exec\s*\(': 'Avoid exec(), use specific function calls',
            r'pickle\.loads?': 'Use json or implement custom serialization',
            r'yaml\.load': 'Use yaml.safe_load() instead',
            r'hashlib\.md5': 'Use SHA-256 or better',
            r'random\.seed': 'Use secrets module for cryptographic purposes',
        }
        
        for pat, rec in recommendations.items():
            if re.search(pat, pattern):
                return rec
        
        return 'Review code for security implications'


class NetworkSecurityAnalyzer:
    """Analyze network configurations for security issues."""
    
    def analyze_network_config(self, network_config: Dict[str, Any]) -> List[SecurityIssue]:
        """Analyze network configuration for security issues."""
        issues = []
        
        # Check for reasonable parameter ranges
        base_voltage = network_config.get('base_voltage', 12.47)
        if base_voltage > 1000:  # kV
            issues.append(SecurityIssue(
                severity='medium',
                issue_type='configuration',
                description='Extremely high base voltage could indicate data corruption',
                recommendation='Verify voltage units and values'
            ))
        
        # Check load parameters
        loads = network_config.get('loads', [])
        for i, load in enumerate(loads):
            if isinstance(load, dict):
                power = load.get('power', 0)
                if power > 1e9:  # 1 GW
                    issues.append(SecurityIssue(
                        severity='high',
                        issue_type='configuration',
                        description=f'Load {i} has unrealistic power value: {power}',
                        location=f'loads[{i}].power',
                        recommendation='Verify load power units and values'
                    ))
        
        # Check for suspicious parameter values
        if 'custom_params' in network_config:
            custom = network_config['custom_params']
            if isinstance(custom, dict):
                for key, value in custom.items():
                    if isinstance(key, str) and ('secret' in key.lower() or 'password' in key.lower()):
                        issues.append(SecurityIssue(
                            severity='critical',
                            issue_type='data_exposure',
                            description=f'Potential secret in configuration: {key}',
                            recommendation='Remove secrets from configuration files'
                        ))
        
        return issues


class SecurityAuditor:
    """Main security auditing system."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.code_scanner = CodeSecurityScanner()
        self.network_analyzer = NetworkSecurityAnalyzer()
        
        self.audit_results = []
    
    def audit_system(self, system_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comprehensive security audit."""
        
        audit_report = {
            'timestamp': __import__('time').time(),
            'issues': {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            },
            'summary': {
                'total_issues': 0,
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0
            },
            'recommendations': []
        }
        
        all_issues = []
        
        # 1. Network configuration analysis
        if system_config and 'network' in system_config:
            network_issues = self.network_analyzer.analyze_network_config(
                system_config['network']
            )
            all_issues.extend(network_issues)
        
        # 2. Input validation tests
        test_inputs = [
            ('numeric_overflow', 1e20, 'Extremely large numeric value'),
            ('array_size', np.zeros(1000000), 'Very large array'),
            ('string_length', 'x' * 10000, 'Very long string'),
            ('suspicious_string', '<script>alert("xss")</script>', 'XSS attempt'),
        ]
        
        for test_name, test_value, description in test_inputs:
            try:
                if isinstance(test_value, (int, float)):
                    valid, msg = self.input_validator.validate_numeric_input(
                        test_value, max_val=1e10
                    )
                elif isinstance(test_value, np.ndarray):
                    valid, msg = self.input_validator.validate_array_shape(
                        test_value, None, max_elements=100000
                    )
                elif isinstance(test_value, str):
                    valid, msg = self.input_validator.validate_string_input(
                        test_value, max_length=1000
                    )
                
                if not valid:
                    logger.info(f"Input validation correctly rejected {test_name}: {msg}")
                else:
                    all_issues.append(SecurityIssue(
                        severity='medium',
                        issue_type='input_validation',
                        description=f'Input validation failed to catch: {description}',
                        recommendation='Strengthen input validation rules'
                    ))
                    
            except Exception as e:
                all_issues.append(SecurityIssue(
                    severity='high',
                    issue_type='exception_handling',
                    description=f'Unhandled exception in validation: {e}',
                    recommendation='Add proper exception handling'
                ))
        
        # 3. Categorize issues
        for issue in all_issues:
            audit_report['issues'][issue.severity].append({
                'type': issue.issue_type,
                'description': issue.description,
                'location': issue.location,
                'recommendation': issue.recommendation
            })
        
        # 4. Generate summary
        for severity in ['critical', 'high', 'medium', 'low']:
            count = len(audit_report['issues'][severity])
            audit_report['summary'][f'{severity}_count'] = count
            audit_report['summary']['total_issues'] += count
        
        # 5. Generate recommendations
        if audit_report['summary']['critical_count'] > 0:
            audit_report['recommendations'].append(
                'CRITICAL: Address all critical security issues immediately'
            )
        
        if audit_report['summary']['high_count'] > 0:
            audit_report['recommendations'].append(
                'HIGH: Review and fix high-severity security issues'
            )
        
        if audit_report['summary']['total_issues'] == 0:
            audit_report['recommendations'].append(
                'Good: No security issues detected in current audit scope'
            )
        
        return audit_report


# Global security auditor
global_security_auditor = SecurityAuditor()