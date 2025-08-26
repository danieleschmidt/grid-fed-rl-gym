"""Comprehensive security validation system for Grid-Fed-RL-Gym project.

This module provides comprehensive security scanning, validation, and monitoring
capabilities that integrate with the existing security infrastructure.
"""

import ast
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
import traceback
import importlib.util
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path
import logging
import threading
from collections import defaultdict, Counter

# Import existing security components
from .security import (
    SecurityAuditor, SecurityIssue, InputValidator, CodeSecurityScanner,
    NetworkSecurityAnalyzer, AdvancedEncryption, SecureCommunicationManager,
    SecurityViolationError, SecurityContext, SecurityRole, EncryptionLevel
)
from .security_hardening import (
    SecurityPolicy, InputSanitizer, AccessController, DataProtector, 
    SecurityMonitor, SecurityLevel
)
from .health_monitoring import HealthMonitor, HealthStatus, HealthMetric

# Optional imports with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecuritySeverity(Enum):
    """Security issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityCategory(Enum):
    """Security issue categories."""
    INPUT_VALIDATION = "input_validation"
    DEPENDENCY_SECURITY = "dependency_security"
    CODE_INJECTION = "code_injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ENCRYPTION = "data_encryption"
    NETWORK_SECURITY = "network_security"
    DATA_EXPOSURE = "data_exposure"
    CONFIGURATION = "configuration"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ComplianceStandard(Enum):
    """Security compliance standards."""
    NIST_CSF = "nist_cybersecurity_framework"
    NERC_CIP = "nerc_critical_infrastructure_protection"
    ISO_27001 = "iso_27001"
    OWASP_TOP10 = "owasp_top_10"
    CUSTOM_GRID = "custom_grid_security"


@dataclass
class SecurityFinding:
    """Detailed security finding with remediation guidance."""
    severity: SecuritySeverity
    category: SecurityCategory
    title: str
    description: str
    location: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    compliance_mapping: Dict[ComplianceStandard, List[str]] = field(default_factory=dict)
    cvss_score: Optional[float] = None
    confidence: float = 1.0  # 0.0 to 1.0
    first_detected: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary for serialization."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "remediation": self.remediation,
            "references": self.references,
            "compliance_mapping": {k.value: v for k, v in self.compliance_mapping.items()},
            "cvss_score": self.cvss_score,
            "confidence": self.confidence,
            "first_detected": self.first_detected.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }


@dataclass
class SecurityMetrics:
    """Security performance metrics."""
    total_scans: int = 0
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    resolved_findings: int = 0
    false_positives: int = 0
    scan_duration_seconds: float = 0.0
    last_scan_time: Optional[datetime] = None
    average_scan_time: float = 0.0
    findings_by_category: Dict[str, int] = field(default_factory=dict)
    security_score: float = 100.0  # 0-100 scale


class DependencyScanner:
    """Security scanner for Python dependencies."""
    
    def __init__(self):
        self.known_vulnerabilities = {}
        self.vulnerability_db_updated = None
        self.load_vulnerability_database()
    
    def load_vulnerability_database(self):
        """Load known vulnerability database."""
        # In a production system, this would load from a real CVE database
        # For demonstration, using a simplified vulnerability database
        self.known_vulnerabilities = {
            "requests": {
                "2.25.0": [
                    {
                        "cve_id": "CVE-2021-33503",
                        "severity": SecuritySeverity.MEDIUM,
                        "description": "Catastrophic backtracking in URL parsing",
                        "fixed_version": "2.25.1"
                    }
                ]
            },
            "pillow": {
                "8.0.0": [
                    {
                        "cve_id": "CVE-2021-25290",
                        "severity": SecuritySeverity.HIGH,
                        "description": "Buffer overflow in TiffDecode.c",
                        "fixed_version": "8.1.1"
                    }
                ]
            },
            "numpy": {
                "1.19.0": [
                    {
                        "cve_id": "CVE-2021-33430",
                        "severity": SecuritySeverity.MEDIUM,
                        "description": "Buffer overflow in numpy.distutils",
                        "fixed_version": "1.21.0"
                    }
                ]
            }
        }
        self.vulnerability_db_updated = datetime.now()
    
    def scan_requirements_file(self, requirements_path: str) -> List[SecurityFinding]:
        """Scan requirements.txt for vulnerable dependencies."""
        findings = []
        
        if not os.path.exists(requirements_path):
            return findings
        
        try:
            with open(requirements_path, 'r') as f:
                requirements = f.readlines()
            
            for line_num, line in enumerate(requirements, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    package_info = self._parse_requirement(line)
                    if package_info:
                        package_findings = self._check_package_vulnerabilities(
                            package_info, requirements_path, line_num, line
                        )
                        findings.extend(package_findings)
        
        except Exception as e:
            logger.error(f"Error scanning requirements file {requirements_path}: {e}")
            findings.append(SecurityFinding(
                severity=SecuritySeverity.MEDIUM,
                category=SecurityCategory.DEPENDENCY_SECURITY,
                title="Requirements File Scan Error",
                description=f"Failed to scan requirements file: {e}",
                file_path=requirements_path,
                remediation="Ensure requirements file is properly formatted and accessible"
            ))
        
        return findings
    
    def _parse_requirement(self, requirement_line: str) -> Optional[Dict[str, str]]:
        """Parse a requirement line to extract package name and version."""
        # Simple parsing - in production, would use proper requirement parser
        requirement_line = requirement_line.split('#')[0].strip()  # Remove comments
        
        # Handle various version specifiers
        patterns = [
            r'^([a-zA-Z0-9\-_\.]+)(==|>=|<=|>|<|~=|!=)([0-9\.]+.*?)$',
            r'^([a-zA-Z0-9\-_\.]+)$'  # No version specified
        ]
        
        for pattern in patterns:
            match = re.match(pattern, requirement_line)
            if match:
                if len(match.groups()) >= 3:
                    return {
                        "name": match.group(1).lower(),
                        "operator": match.group(2),
                        "version": match.group(3)
                    }
                else:
                    return {
                        "name": match.group(1).lower(),
                        "operator": None,
                        "version": None
                    }
        
        return None
    
    def _check_package_vulnerabilities(
        self, 
        package_info: Dict[str, str], 
        file_path: str, 
        line_number: int, 
        line_content: str
    ) -> List[SecurityFinding]:
        """Check a specific package for known vulnerabilities."""
        findings = []
        package_name = package_info["name"]
        package_version = package_info.get("version")
        
        if package_name in self.known_vulnerabilities:
            package_vulns = self.known_vulnerabilities[package_name]
            
            if package_version and package_version in package_vulns:
                for vuln in package_vulns[package_version]:
                    finding = SecurityFinding(
                        severity=vuln["severity"],
                        category=SecurityCategory.DEPENDENCY_SECURITY,
                        title=f"Vulnerable Dependency: {package_name}",
                        description=f"Package {package_name} version {package_version} has known vulnerability: {vuln['description']}",
                        file_path=file_path,
                        line_number=line_number,
                        code_snippet=line_content,
                        remediation=f"Update {package_name} to version {vuln['fixed_version']} or later",
                        references=[f"CVE: {vuln['cve_id']}"],
                        compliance_mapping={
                            ComplianceStandard.NIST_CSF: ["ID.RA-1", "PR.IP-12"],
                            ComplianceStandard.OWASP_TOP10: ["A06:2021 – Vulnerable Components"]
                        }
                    )
                    findings.append(finding)
        
        return findings


class NetworkSecurityValidator:
    """Validate network security configurations and protocols."""
    
    def __init__(self):
        self.secure_protocols = {"TLS", "HTTPS", "SSH", "SFTP"}
        self.insecure_protocols = {"HTTP", "FTP", "TELNET", "SNMP"}
        self.secure_ports = {443, 22, 993, 995, 587}
        self.insecure_ports = {21, 23, 25, 53, 80, 110, 143, 161, 389}
    
    def validate_network_config(self, config: Dict[str, Any]) -> List[SecurityFinding]:
        """Validate network configuration for security issues."""
        findings = []
        
        # Check for insecure protocols
        if "protocols" in config:
            protocols = config["protocols"]
            if isinstance(protocols, list):
                for protocol in protocols:
                    if isinstance(protocol, str) and protocol.upper() in self.insecure_protocols:
                        findings.append(SecurityFinding(
                            severity=SecuritySeverity.HIGH,
                            category=SecurityCategory.NETWORK_SECURITY,
                            title="Insecure Network Protocol",
                            description=f"Use of insecure protocol: {protocol}",
                            remediation=f"Replace {protocol} with secure alternative",
                            compliance_mapping={
                                ComplianceStandard.NIST_CSF: ["PR.DS-2", "PR.AC-5"],
                                ComplianceStandard.NERC_CIP: ["CIP-007-6 R1"]
                            }
                        ))
        
        # Check for insecure ports
        if "ports" in config:
            ports = config["ports"]
            if isinstance(ports, list):
                for port in ports:
                    if isinstance(port, int) and port in self.insecure_ports:
                        findings.append(SecurityFinding(
                            severity=SecuritySeverity.MEDIUM,
                            category=SecurityCategory.NETWORK_SECURITY,
                            title="Insecure Network Port",
                            description=f"Use of potentially insecure port: {port}",
                            remediation="Use secure ports and protocols for network communication",
                            compliance_mapping={
                                ComplianceStandard.NERC_CIP: ["CIP-005-5 R1"]
                            }
                        ))
        
        # Check for plaintext credentials
        for key, value in config.items():
            if isinstance(key, str) and any(term in key.lower() for term in ["password", "secret", "key", "token"]):
                if isinstance(value, str) and not self._appears_encrypted(value):
                    findings.append(SecurityFinding(
                        severity=SecuritySeverity.CRITICAL,
                        category=SecurityCategory.DATA_EXPOSURE,
                        title="Plaintext Credentials in Network Config",
                        description=f"Plaintext credential found in configuration: {key}",
                        remediation="Encrypt or secure credentials using proper secret management",
                        compliance_mapping={
                            ComplianceStandard.NIST_CSF: ["PR.AC-1", "PR.DS-1"],
                            ComplianceStandard.ISO_27001: ["A.9.4.3", "A.10.1.2"]
                        }
                    ))
        
        return findings
    
    def _appears_encrypted(self, value: str) -> bool:
        """Heuristic to determine if a value appears to be encrypted."""
        # Simple heuristics - in production, would be more sophisticated
        if len(value) < 8:
            return False
        
        # Check for common encryption patterns
        if any(pattern in value for pattern in ["$2b$", "sha256:", "aes:", "-----BEGIN", "-----END"]):
            return True
        
        # Check for high entropy (likely encrypted/encoded)
        unique_chars = len(set(value))
        entropy_ratio = unique_chars / len(value)
        
        return entropy_ratio > 0.7 and len(value) > 16


class AuthenticationValidator:
    """Validate authentication and authorization implementations."""
    
    def __init__(self):
        self.weak_password_patterns = [
            r'^.{0,7}$',  # Too short
            r'^[a-z]+$',  # Only lowercase
            r'^[A-Z]+$',  # Only uppercase
            r'^[0-9]+$',  # Only numbers
            r'password|123456|qwerty|admin|root|guest',  # Common weak passwords
        ]
        
        self.insecure_auth_methods = {
            "basic_auth": "HTTP Basic Authentication transmits credentials in plaintext",
            "md5_hash": "MD5 is cryptographically broken",
            "sha1_hash": "SHA1 is deprecated for security purposes"
        }
    
    def validate_authentication_config(self, config: Dict[str, Any]) -> List[SecurityFinding]:
        """Validate authentication configuration."""
        findings = []
        
        # Check for weak authentication methods
        if "auth_method" in config:
            auth_method = str(config["auth_method"]).lower()
            if auth_method in self.insecure_auth_methods:
                findings.append(SecurityFinding(
                    severity=SecuritySeverity.HIGH,
                    category=SecurityCategory.AUTHENTICATION,
                    title="Insecure Authentication Method",
                    description=f"Insecure authentication method: {auth_method}. {self.insecure_auth_methods[auth_method]}",
                    remediation="Use secure authentication methods like OAuth 2.0, JWT with strong cryptography, or certificate-based authentication",
                    compliance_mapping={
                        ComplianceStandard.NIST_CSF: ["PR.AC-1", "PR.AC-7"],
                        ComplianceStandard.OWASP_TOP10: ["A07:2021 – Identification and Authentication Failures"]
                    }
                ))
        
        # Check session configuration
        if "session_config" in config:
            session_config = config["session_config"]
            if isinstance(session_config, dict):
                # Check session timeout
                timeout = session_config.get("timeout_minutes", 0)
                if timeout > 480:  # 8 hours
                    findings.append(SecurityFinding(
                        severity=SecuritySeverity.MEDIUM,
                        category=SecurityCategory.AUTHENTICATION,
                        title="Excessive Session Timeout",
                        description=f"Session timeout is too long: {timeout} minutes",
                        remediation="Set session timeout to a reasonable value (e.g., 30-120 minutes)",
                        compliance_mapping={
                            ComplianceStandard.NIST_CSF: ["PR.AC-12"]
                        }
                    ))
                
                # Check if secure flag is set for cookies
                if not session_config.get("secure_cookies", False):
                    findings.append(SecurityFinding(
                        severity=SecuritySeverity.MEDIUM,
                        category=SecurityCategory.AUTHENTICATION,
                        title="Insecure Session Cookies",
                        description="Session cookies are not configured with secure flag",
                        remediation="Enable secure flag for session cookies to prevent transmission over insecure connections",
                        compliance_mapping={
                            ComplianceStandard.OWASP_TOP10: ["A05:2021 – Security Misconfiguration"]
                        }
                    ))
        
        # Check for default credentials
        if "users" in config and isinstance(config["users"], list):
            for user in config["users"]:
                if isinstance(user, dict):
                    username = user.get("username", "").lower()
                    password = user.get("password", "")
                    
                    if username in ["admin", "root", "administrator", "user", "guest"]:
                        findings.append(SecurityFinding(
                            severity=SecuritySeverity.HIGH,
                            category=SecurityCategory.AUTHENTICATION,
                            title="Default Username",
                            description=f"Default or common username detected: {username}",
                            remediation="Use unique, non-default usernames for all accounts",
                            compliance_mapping={
                                ComplianceStandard.NIST_CSF: ["PR.AC-1"],
                                ComplianceStandard.NERC_CIP: ["CIP-004-6 R4"]
                            }
                        ))
                    
                    # Check password strength
                    if password:
                        weakness = self._check_password_strength(password)
                        if weakness:
                            findings.append(SecurityFinding(
                                severity=SecuritySeverity.HIGH,
                                category=SecurityCategory.AUTHENTICATION,
                                title="Weak Password Policy",
                                description=f"Weak password detected: {weakness}",
                                remediation="Implement strong password policy with minimum length, complexity requirements, and regular rotation",
                                compliance_mapping={
                                    ComplianceStandard.NIST_CSF: ["PR.AC-1"],
                                    ComplianceStandard.ISO_27001: ["A.9.4.3"]
                                }
                            ))
        
        return findings
    
    def _check_password_strength(self, password: str) -> Optional[str]:
        """Check password strength and return weakness description."""
        for pattern in self.weak_password_patterns:
            if re.search(pattern, password, re.IGNORECASE):
                if pattern == r'^.{0,7}$':
                    return "Password is too short (less than 8 characters)"
                elif pattern == r'^[a-z]+$':
                    return "Password contains only lowercase letters"
                elif pattern == r'^[A-Z]+$':
                    return "Password contains only uppercase letters"
                elif pattern == r'^[0-9]+$':
                    return "Password contains only numbers"
                else:
                    return "Password is a common weak password"
        
        return None


class CodeInjectionScanner:
    """Scan for code injection vulnerabilities."""
    
    def __init__(self):
        self.injection_patterns = {
            SecuritySeverity.CRITICAL: [
                (r'eval\s*\(', "Use of eval() function can execute arbitrary code"),
                (r'exec\s*\(', "Use of exec() function can execute arbitrary code"),
                (r'__import__\s*\(.*input', "Dynamic import with user input is dangerous"),
                (r'compile\s*\(.*input', "Dynamic compilation with user input is dangerous"),
            ],
            SecuritySeverity.HIGH: [
                (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True', "Shell execution with user input is dangerous"),
                (r'os\.system\s*\(', "os.system() can execute arbitrary commands"),
                (r'pickle\.loads?\s*\(', "Pickle deserialization can execute arbitrary code"),
                (r'yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader', "YAML load with unsafe loader"),
            ],
            SecuritySeverity.MEDIUM: [
                (r'input\s*\([^)]*\)', "Raw input() function usage should be validated"),
                (r'open\s*\([^)]*[\'"]w[\'"]', "File write operations should validate paths"),
                (r'format\s*\([^)]*\{.*\}', "String formatting with user input can be dangerous"),
            ]
        }
        
        self.sql_injection_patterns = [
            (r'execute\s*\([^)]*%s[^)]*\)', "Potential SQL injection with string formatting"),
            (r'query\s*\([^)]*\+[^)]*\)', "SQL query concatenation can lead to injection"),
            (r'cursor\.execute\s*\([^)]*format', "String formatting in SQL queries"),
        ]
    
    def scan_file(self, file_path: str) -> List[SecurityFinding]:
        """Scan a Python file for injection vulnerabilities."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Scan for general injection patterns
            for severity, patterns in self.injection_patterns.items():
                for pattern, description in patterns:
                    findings.extend(self._find_pattern_in_file(
                        pattern, description, lines, file_path, severity, SecurityCategory.CODE_INJECTION
                    ))
            
            # Scan for SQL injection patterns
            for pattern, description in self.sql_injection_patterns:
                findings.extend(self._find_pattern_in_file(
                    pattern, description, lines, file_path, SecuritySeverity.HIGH, SecurityCategory.CODE_INJECTION
                ))
            
            # Parse AST for more sophisticated analysis
            findings.extend(self._analyze_ast(content, file_path))
            
        except Exception as e:
            logger.error(f"Error scanning file {file_path}: {e}")
            findings.append(SecurityFinding(
                severity=SecuritySeverity.MEDIUM,
                category=SecurityCategory.CODE_INJECTION,
                title="File Scan Error",
                description=f"Failed to scan file for injection vulnerabilities: {e}",
                file_path=file_path,
                remediation="Ensure file is valid Python code and accessible"
            ))
        
        return findings
    
    def _find_pattern_in_file(
        self, 
        pattern: str, 
        description: str, 
        lines: List[str], 
        file_path: str,
        severity: SecuritySeverity,
        category: SecurityCategory
    ) -> List[SecurityFinding]:
        """Find pattern matches in file lines."""
        findings = []
        
        for line_num, line in enumerate(lines, 1):
            if re.search(pattern, line, re.IGNORECASE):
                findings.append(SecurityFinding(
                    severity=severity,
                    category=category,
                    title="Potential Code Injection Vulnerability",
                    description=description,
                    file_path=file_path,
                    line_number=line_num,
                    code_snippet=line.strip(),
                    remediation=self._get_remediation_for_pattern(pattern),
                    compliance_mapping={
                        ComplianceStandard.OWASP_TOP10: ["A03:2021 – Injection"],
                        ComplianceStandard.NIST_CSF: ["PR.DS-2"]
                    }
                ))
        
        return findings
    
    def _analyze_ast(self, content: str, file_path: str) -> List[SecurityFinding]:
        """Analyze AST for injection vulnerabilities."""
        findings = []
        
        try:
            tree = ast.parse(content)
            
            class InjectionVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.findings = []
                
                def visit_Call(self, node):
                    # Check for dangerous function calls
                    if hasattr(node.func, 'id'):
                        func_name = node.func.id
                        if func_name in ['eval', 'exec']:
                            self.findings.append(SecurityFinding(
                                severity=SecuritySeverity.CRITICAL,
                                category=SecurityCategory.CODE_INJECTION,
                                title=f"Dangerous Function Call: {func_name}",
                                description=f"Use of {func_name}() can execute arbitrary code",
                                file_path=file_path,
                                line_number=node.lineno,
                                remediation=f"Avoid using {func_name}() or implement strict input validation",
                                compliance_mapping={
                                    ComplianceStandard.OWASP_TOP10: ["A03:2021 – Injection"]
                                }
                            ))
                    
                    self.generic_visit(node)
            
            visitor = InjectionVisitor()
            visitor.visit(tree)
            findings.extend(visitor.findings)
            
        except SyntaxError as e:
            findings.append(SecurityFinding(
                severity=SecuritySeverity.LOW,
                category=SecurityCategory.CODE_INJECTION,
                title="Syntax Error in AST Analysis",
                description=f"Syntax error prevented AST analysis: {e}",
                file_path=file_path,
                line_number=e.lineno if e.lineno else 0,
                remediation="Fix syntax errors in the code"
            ))
        except Exception as e:
            logger.error(f"AST analysis failed for {file_path}: {e}")
        
        return findings
    
    def _get_remediation_for_pattern(self, pattern: str) -> str:
        """Get specific remediation advice for a pattern."""
        remediation_map = {
            r'eval\s*\(': "Use ast.literal_eval() for safe evaluation of literals, or implement proper input validation",
            r'exec\s*\(': "Avoid exec() entirely, or implement strict sandboxing and input validation",
            r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True': "Use shell=False and pass arguments as a list",
            r'pickle\.loads?\s*\(': "Use json or implement custom serialization for untrusted data",
            r'os\.system\s*\(': "Use subprocess with proper argument handling instead of os.system()"
        }
        
        for pat, remedy in remediation_map.items():
            if re.search(pat, pattern):
                return remedy
        
        return "Implement proper input validation and sanitization"


class DataEncryptionValidator:
    """Validate data encryption implementations and configurations."""
    
    def __init__(self):
        self.weak_algorithms = {
            "md5": "MD5 is cryptographically broken",
            "sha1": "SHA1 is deprecated for security purposes",
            "des": "DES has insufficient key length",
            "3des": "3DES is deprecated",
            "rc4": "RC4 has known vulnerabilities"
        }
        
        self.strong_algorithms = {
            "aes", "sha256", "sha512", "rsa", "ecdsa", "chacha20"
        }
        
        self.minimum_key_lengths = {
            "rsa": 2048,
            "dsa": 2048,
            "ecdsa": 256,
            "aes": 128
        }
    
    def validate_encryption_config(self, config: Dict[str, Any]) -> List[SecurityFinding]:
        """Validate encryption configuration."""
        findings = []
        
        # Check for weak encryption algorithms
        if "encryption_algorithm" in config:
            algorithm = str(config["encryption_algorithm"]).lower()
            if algorithm in self.weak_algorithms:
                findings.append(SecurityFinding(
                    severity=SecuritySeverity.HIGH,
                    category=SecurityCategory.DATA_ENCRYPTION,
                    title="Weak Encryption Algorithm",
                    description=f"Weak encryption algorithm: {algorithm}. {self.weak_algorithms[algorithm]}",
                    remediation="Use strong encryption algorithms like AES-256, RSA-2048, or equivalent",
                    compliance_mapping={
                        ComplianceStandard.NIST_CSF: ["PR.DS-1"],
                        ComplianceStandard.ISO_27001: ["A.10.1.1"]
                    }
                ))
        
        # Check key lengths
        if "key_length" in config and "encryption_algorithm" in config:
            key_length = config["key_length"]
            algorithm = str(config["encryption_algorithm"]).lower()
            
            if algorithm in self.minimum_key_lengths:
                min_length = self.minimum_key_lengths[algorithm]
                if isinstance(key_length, int) and key_length < min_length:
                    findings.append(SecurityFinding(
                        severity=SecuritySeverity.HIGH,
                        category=SecurityCategory.DATA_ENCRYPTION,
                        title="Insufficient Key Length",
                        description=f"Key length {key_length} is insufficient for {algorithm}. Minimum: {min_length}",
                        remediation=f"Use key length of at least {min_length} bits for {algorithm}",
                        compliance_mapping={
                            ComplianceStandard.NIST_CSF: ["PR.DS-1"]
                        }
                    ))
        
        # Check for hardcoded keys or secrets
        for key, value in config.items():
            if isinstance(key, str) and any(term in key.lower() for term in ["key", "secret", "password", "token"]):
                if isinstance(value, str) and len(value) > 0:
                    if self._appears_hardcoded(key, value):
                        findings.append(SecurityFinding(
                            severity=SecuritySeverity.CRITICAL,
                            category=SecurityCategory.DATA_ENCRYPTION,
                            title="Hardcoded Encryption Key",
                            description=f"Hardcoded encryption key/secret found: {key}",
                            remediation="Use secure key management systems and environment variables for secrets",
                            compliance_mapping={
                                ComplianceStandard.NIST_CSF: ["PR.AC-1", "PR.DS-1"],
                                ComplianceStandard.OWASP_TOP10: ["A02:2021 – Cryptographic Failures"]
                            }
                        ))
        
        # Check for insecure random number generation
        if "random_source" in config:
            random_source = str(config["random_source"]).lower()
            if "random.random" in random_source or "math.random" in random_source:
                findings.append(SecurityFinding(
                    severity=SecuritySeverity.HIGH,
                    category=SecurityCategory.DATA_ENCRYPTION,
                    title="Insecure Random Number Generation",
                    description="Using non-cryptographic random number generator for security purposes",
                    remediation="Use cryptographically secure random number generators (secrets module)",
                    compliance_mapping={
                        ComplianceStandard.OWASP_TOP10: ["A02:2021 – Cryptographic Failures"]
                    }
                ))
        
        return findings
    
    def _appears_hardcoded(self, key_name: str, value: str) -> bool:
        """Check if a key appears to be hardcoded."""
        # Simple heuristics for hardcoded keys
        if len(value) < 8:
            return True
        
        # Check if it's obviously a placeholder or example
        if any(term in value.lower() for term in ["example", "test", "sample", "changeme", "password"]):
            return True
        
        # Check if it's in a development/test configuration
        if any(term in key_name.lower() for term in ["test", "dev", "example", "sample"]):
            return False
        
        # If it's a short, simple string, it's likely hardcoded
        if len(value) < 32 and value.isalnum():
            return True
        
        return False


class SecurityValidationSuite:
    """Comprehensive security validation suite."""
    
    def __init__(self, base_directory: str = None):
        self.base_directory = base_directory or os.getcwd()
        self.metrics = SecurityMetrics()
        self.findings_history: List[SecurityFinding] = []
        self.scan_lock = threading.Lock()
        
        # Initialize component scanners
        self.dependency_scanner = DependencyScanner()
        self.network_validator = NetworkSecurityValidator()
        self.auth_validator = AuthenticationValidator()
        self.injection_scanner = CodeInjectionScanner()
        self.encryption_validator = DataEncryptionValidator()
        
        # Integrate with existing security components
        self.input_validator = InputValidator()
        self.code_scanner = CodeSecurityScanner()
        self.network_analyzer = NetworkSecurityAnalyzer()
        self.security_auditor = SecurityAuditor()
        
        logger.info(f"Security validation suite initialized for directory: {self.base_directory}")
    
    def run_comprehensive_scan(
        self, 
        scan_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        start_time = time.time()
        
        with self.scan_lock:
            logger.info("Starting comprehensive security scan...")
            
            scan_config = scan_config or {}
            findings = []
            
            try:
                # 1. Dependency Security Scan
                if scan_config.get("scan_dependencies", True):
                    logger.info("Running dependency security scan...")
                    dep_findings = self._scan_dependencies()
                    findings.extend(dep_findings)
                
                # 2. Code Injection Scan
                if scan_config.get("scan_code_injection", True):
                    logger.info("Running code injection scan...")
                    injection_findings = self._scan_code_injection()
                    findings.extend(injection_findings)
                
                # 3. Input Validation Scan
                if scan_config.get("scan_input_validation", True):
                    logger.info("Running input validation scan...")
                    input_findings = self._scan_input_validation()
                    findings.extend(input_findings)
                
                # 4. Authentication & Authorization Scan
                if scan_config.get("scan_auth", True):
                    logger.info("Running authentication scan...")
                    auth_findings = self._scan_authentication()
                    findings.extend(auth_findings)
                
                # 5. Data Encryption Scan
                if scan_config.get("scan_encryption", True):
                    logger.info("Running encryption validation scan...")
                    encryption_findings = self._scan_encryption()
                    findings.extend(encryption_findings)
                
                # 6. Network Security Scan
                if scan_config.get("scan_network", True):
                    logger.info("Running network security scan...")
                    network_findings = self._scan_network_security()
                    findings.extend(network_findings)
                
                # 7. Configuration Security Scan
                if scan_config.get("scan_configuration", True):
                    logger.info("Running configuration security scan...")
                    config_findings = self._scan_configuration_security()
                    findings.extend(config_findings)
                
                # Update metrics and history
                self._update_scan_metrics(findings, start_time)
                self.findings_history.extend(findings)
                
                # Generate comprehensive report
                report = self._generate_security_report(findings, start_time)
                
                logger.info(f"Security scan completed in {time.time() - start_time:.2f} seconds")
                return report
                
            except Exception as e:
                logger.error(f"Error during security scan: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "scan_duration": time.time() - start_time
                }
    
    def _scan_dependencies(self) -> List[SecurityFinding]:
        """Scan for dependency security issues."""
        findings = []
        
        # Scan requirements files
        requirements_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "dev-requirements.txt"
        ]
        
        for req_file in requirements_files:
            req_path = os.path.join(self.base_directory, req_file)
            if os.path.exists(req_path):
                findings.extend(self.dependency_scanner.scan_requirements_file(req_path))
        
        return findings
    
    def _scan_code_injection(self) -> List[SecurityFinding]:
        """Scan for code injection vulnerabilities."""
        findings = []
        
        # Find Python files to scan
        python_files = []
        for root, dirs, files in os.walk(self.base_directory):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', 'node_modules'}]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        # Scan each Python file
        for py_file in python_files:
            try:
                file_findings = self.injection_scanner.scan_file(py_file)
                findings.extend(file_findings)
            except Exception as e:
                logger.error(f"Error scanning {py_file} for injections: {e}")
        
        return findings
    
    def _scan_input_validation(self) -> List[SecurityFinding]:
        """Scan for input validation issues."""
        findings = []
        
        # Test various input validation scenarios
        test_cases = [
            ("large_number", 1e20, "Extremely large numeric input"),
            ("negative_overflow", -1e20, "Extremely negative numeric input"),
            ("malicious_string", "<script>alert('xss')</script>", "XSS payload"),
            ("sql_injection", "'; DROP TABLE users; --", "SQL injection payload"),
            ("path_traversal", "../../../etc/passwd", "Path traversal payload"),
            ("command_injection", "; rm -rf /", "Command injection payload"),
        ]
        
        for test_name, test_input, description in test_cases:
            try:
                if isinstance(test_input, (int, float)):
                    valid, msg = self.input_validator.validate_numeric_input(
                        test_input, max_val=1e10
                    )
                elif isinstance(test_input, str):
                    valid, msg = self.input_validator.validate_string_input(
                        test_input, max_length=1000
                    )
                
                if valid:
                    # Input validation failed to catch malicious input
                    findings.append(SecurityFinding(
                        severity=SecuritySeverity.HIGH,
                        category=SecurityCategory.INPUT_VALIDATION,
                        title="Input Validation Bypass",
                        description=f"Input validation failed to reject {description}",
                        remediation="Strengthen input validation rules and add specific checks for malicious patterns"
                    ))
            except Exception as e:
                # Unhandled exception during validation
                findings.append(SecurityFinding(
                    severity=SecuritySeverity.MEDIUM,
                    category=SecurityCategory.INPUT_VALIDATION,
                    title="Input Validation Exception",
                    description=f"Unhandled exception during input validation: {e}",
                    remediation="Add proper exception handling to input validation"
                ))
        
        return findings
    
    def _scan_authentication(self) -> List[SecurityFinding]:
        """Scan for authentication and authorization issues."""
        findings = []
        
        # Look for authentication configuration files
        auth_config_files = [
            "auth_config.json",
            "security_config.json",
            "config/auth.json",
            "settings.py",
            "config.py"
        ]
        
        for config_file in auth_config_files:
            config_path = os.path.join(self.base_directory, config_file)
            if os.path.exists(config_path):
                try:
                    config_data = self._load_config_file(config_path)
                    if config_data:
                        findings.extend(self.auth_validator.validate_authentication_config(config_data))
                except Exception as e:
                    logger.error(f"Error scanning auth config {config_path}: {e}")
        
        return findings
    
    def _scan_encryption(self) -> List[SecurityFinding]:
        """Scan for encryption implementation issues."""
        findings = []
        
        # Look for encryption configuration
        crypto_config_files = [
            "crypto_config.json",
            "encryption_config.json",
            "security_config.json"
        ]
        
        for config_file in crypto_config_files:
            config_path = os.path.join(self.base_directory, config_file)
            if os.path.exists(config_path):
                try:
                    config_data = self._load_config_file(config_path)
                    if config_data:
                        findings.extend(self.encryption_validator.validate_encryption_config(config_data))
                except Exception as e:
                    logger.error(f"Error scanning crypto config {config_path}: {e}")
        
        return findings
    
    def _scan_network_security(self) -> List[SecurityFinding]:
        """Scan for network security issues."""
        findings = []
        
        # Look for network configuration files
        network_config_files = [
            "network_config.json",
            "server_config.json",
            "api_config.json"
        ]
        
        for config_file in network_config_files:
            config_path = os.path.join(self.base_directory, config_file)
            if os.path.exists(config_path):
                try:
                    config_data = self._load_config_file(config_path)
                    if config_data:
                        findings.extend(self.network_validator.validate_network_config(config_data))
                except Exception as e:
                    logger.error(f"Error scanning network config {config_path}: {e}")
        
        return findings
    
    def _scan_configuration_security(self) -> List[SecurityFinding]:
        """Scan for configuration security issues."""
        findings = []
        
        # Look for configuration files with potential secrets
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.ini", "*.cfg", "*.conf"]
        
        for pattern in config_patterns:
            for config_file in Path(self.base_directory).rglob(pattern):
                # Skip common non-sensitive config files
                if any(skip in str(config_file) for skip in ['.git', '__pycache__', 'venv', 'env']):
                    continue
                
                try:
                    findings.extend(self._scan_config_file_for_secrets(str(config_file)))
                except Exception as e:
                    logger.error(f"Error scanning config file {config_file}: {e}")
        
        return findings
    
    def _scan_config_file_for_secrets(self, file_path: str) -> List[SecurityFinding]:
        """Scan a configuration file for hardcoded secrets."""
        findings = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Patterns for potential secrets
            secret_patterns = [
                (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^"\'\\s]+)["\']?', "Password"),
                (r'(?i)(api[_-]?key)\s*[=:]\s*["\']?([^"\'\\s]+)["\']?', "API Key"),
                (r'(?i)(secret[_-]?key)\s*[=:]\s*["\']?([^"\'\\s]+)["\']?', "Secret Key"),
                (r'(?i)(access[_-]?token)\s*[=:]\s*["\']?([^"\'\\s]+)["\']?', "Access Token"),
                (r'(?i)(private[_-]?key)\s*[=:]\s*["\']?([^"\'\\s]+)["\']?', "Private Key"),
            ]
            
            for line_num, line in enumerate(lines, 1):
                for pattern, secret_type in secret_patterns:
                    match = re.search(pattern, line)
                    if match:
                        secret_value = match.group(2) if len(match.groups()) > 1 else match.group(1)
                        
                        # Skip obvious placeholders
                        if any(placeholder in secret_value.lower() for placeholder in 
                               ["placeholder", "example", "changeme", "your_", "xxx", "***"]):
                            continue
                        
                        findings.append(SecurityFinding(
                            severity=SecuritySeverity.CRITICAL,
                            category=SecurityCategory.DATA_EXPOSURE,
                            title=f"Hardcoded {secret_type} in Configuration",
                            description=f"Hardcoded {secret_type.lower()} found in configuration file",
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=line.strip(),
                            remediation=f"Move {secret_type.lower()} to environment variables or secure secret management",
                            compliance_mapping={
                                ComplianceStandard.NIST_CSF: ["PR.AC-1", "PR.DS-1"],
                                ComplianceStandard.OWASP_TOP10: ["A02:2021 – Cryptographic Failures"]
                            }
                        ))
        
        except Exception as e:
            logger.error(f"Error scanning config file {file_path}: {e}")
        
        return findings
    
    def _load_config_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load configuration file data."""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    return json.load(f)
                elif file_path.endswith(('.yaml', '.yml')):
                    # Would use yaml.safe_load if available
                    return {}
                elif file_path.endswith('.py'):
                    # Extract configuration from Python files
                    content = f.read()
                    # Simple extraction - in production would be more sophisticated
                    config_vars = {}
                    for line in content.split('\n'):
                        if '=' in line and not line.strip().startswith('#'):
                            try:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('\'"')
                                config_vars[key] = value
                            except:
                                continue
                    return config_vars
        except Exception as e:
            logger.error(f"Error loading config file {file_path}: {e}")
            return None
    
    def _update_scan_metrics(self, findings: List[SecurityFinding], start_time: float):
        """Update scan metrics based on findings."""
        self.metrics.total_scans += 1
        self.metrics.total_findings = len(findings)
        self.metrics.scan_duration_seconds = time.time() - start_time
        self.metrics.last_scan_time = datetime.now()
        
        # Update average scan time
        if self.metrics.average_scan_time == 0:
            self.metrics.average_scan_time = self.metrics.scan_duration_seconds
        else:
            self.metrics.average_scan_time = (
                (self.metrics.average_scan_time * (self.metrics.total_scans - 1) + 
                 self.metrics.scan_duration_seconds) / self.metrics.total_scans
            )
        
        # Count findings by severity
        severity_counts = Counter(finding.severity for finding in findings)
        self.metrics.critical_findings = severity_counts[SecuritySeverity.CRITICAL]
        self.metrics.high_findings = severity_counts[SecuritySeverity.HIGH]
        self.metrics.medium_findings = severity_counts[SecuritySeverity.MEDIUM]
        self.metrics.low_findings = severity_counts[SecuritySeverity.LOW]
        
        # Count findings by category
        category_counts = Counter(finding.category.value for finding in findings)
        self.metrics.findings_by_category = dict(category_counts)
        
        # Calculate security score (0-100)
        total_weighted_findings = (
            severity_counts[SecuritySeverity.CRITICAL] * 4 +
            severity_counts[SecuritySeverity.HIGH] * 3 +
            severity_counts[SecuritySeverity.MEDIUM] * 2 +
            severity_counts[SecuritySeverity.LOW] * 1
        )
        
        # Security score decreases based on weighted findings
        self.metrics.security_score = max(0, 100 - min(100, total_weighted_findings * 2))
    
    def _generate_security_report(self, findings: List[SecurityFinding], start_time: float) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        # Categorize findings
        findings_by_severity = defaultdict(list)
        findings_by_category = defaultdict(list)
        
        for finding in findings:
            findings_by_severity[finding.severity.value].append(finding.to_dict())
            findings_by_category[finding.category.value].append(finding.to_dict())
        
        # Generate compliance summary
        compliance_summary = self._generate_compliance_summary(findings)
        
        # Generate remediation priorities
        remediation_priorities = self._generate_remediation_priorities(findings)
        
        return {
            "scan_summary": {
                "status": "completed",
                "scan_duration_seconds": time.time() - start_time,
                "scan_timestamp": datetime.now().isoformat(),
                "total_findings": len(findings),
                "security_score": self.metrics.security_score,
                "base_directory": self.base_directory
            },
            "findings_summary": {
                "by_severity": {
                    "critical": len(findings_by_severity["critical"]),
                    "high": len(findings_by_severity["high"]),
                    "medium": len(findings_by_severity["medium"]),
                    "low": len(findings_by_severity["low"]),
                    "info": len(findings_by_severity["info"])
                },
                "by_category": {k: len(v) for k, v in findings_by_category.items()}
            },
            "detailed_findings": {
                "critical": findings_by_severity["critical"],
                "high": findings_by_severity["high"],
                "medium": findings_by_severity["medium"],
                "low": findings_by_severity["low"],
                "info": findings_by_severity["info"]
            },
            "compliance_summary": compliance_summary,
            "remediation_priorities": remediation_priorities,
            "metrics": {
                "total_scans": self.metrics.total_scans,
                "average_scan_time": self.metrics.average_scan_time,
                "findings_trend": self._get_findings_trend(),
                "security_score_history": self._get_security_score_history()
            },
            "recommendations": self._generate_high_level_recommendations(findings)
        }
    
    def _generate_compliance_summary(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Generate compliance mapping summary."""
        compliance_violations = defaultdict(list)
        
        for finding in findings:
            for standard, controls in finding.compliance_mapping.items():
                for control in controls:
                    compliance_violations[standard.value].append({
                        "control": control,
                        "severity": finding.severity.value,
                        "title": finding.title
                    })
        
        return {
            "standards_affected": list(compliance_violations.keys()),
            "violations_by_standard": dict(compliance_violations),
            "compliance_score": self._calculate_compliance_score(compliance_violations)
        }
    
    def _calculate_compliance_score(self, violations: Dict[str, List]) -> Dict[str, float]:
        """Calculate compliance scores for different standards."""
        scores = {}
        
        for standard, violation_list in violations.items():
            if not violation_list:
                scores[standard] = 100.0
                continue
            
            # Weight violations by severity
            total_weight = 0
            for violation in violation_list:
                if violation["severity"] == "critical":
                    total_weight += 4
                elif violation["severity"] == "high":
                    total_weight += 3
                elif violation["severity"] == "medium":
                    total_weight += 2
                elif violation["severity"] == "low":
                    total_weight += 1
            
            # Calculate score (0-100)
            scores[standard] = max(0, 100 - min(100, total_weight * 2))
        
        return scores
    
    def _generate_remediation_priorities(self, findings: List[SecurityFinding]) -> List[Dict[str, Any]]:
        """Generate prioritized remediation recommendations."""
        priorities = []
        
        # Group similar findings
        finding_groups = defaultdict(list)
        for finding in findings:
            key = (finding.severity, finding.category, finding.title)
            finding_groups[key].append(finding)
        
        # Sort by severity and count
        sorted_groups = sorted(
            finding_groups.items(),
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x[0][0].value, 0),
                len(x[1])
            ),
            reverse=True
        )
        
        for (severity, category, title), group_findings in sorted_groups:
            priorities.append({
                "severity": severity.value,
                "category": category.value,
                "title": title,
                "count": len(group_findings),
                "description": group_findings[0].description,
                "remediation": group_findings[0].remediation,
                "affected_files": list(set(f.file_path for f in group_findings if f.file_path)),
                "estimated_effort": self._estimate_remediation_effort(severity, len(group_findings))
            })
        
        return priorities[:20]  # Top 20 priorities
    
    def _estimate_remediation_effort(self, severity: SecuritySeverity, count: int) -> str:
        """Estimate remediation effort."""
        base_effort = {
            SecuritySeverity.CRITICAL: 4,
            SecuritySeverity.HIGH: 3,
            SecuritySeverity.MEDIUM: 2,
            SecuritySeverity.LOW: 1
        }.get(severity, 1)
        
        total_effort = base_effort * count
        
        if total_effort <= 2:
            return "Low (1-2 hours)"
        elif total_effort <= 8:
            return "Medium (4-8 hours)"
        elif total_effort <= 16:
            return "High (1-2 days)"
        else:
            return "Very High (2+ days)"
    
    def _get_findings_trend(self) -> List[Dict[str, Any]]:
        """Get historical findings trend."""
        # In a real implementation, this would track findings over time
        return [
            {
                "date": datetime.now().isoformat(),
                "total_findings": self.metrics.total_findings,
                "critical": self.metrics.critical_findings,
                "high": self.metrics.high_findings
            }
        ]
    
    def _get_security_score_history(self) -> List[Dict[str, Any]]:
        """Get security score history."""
        # In a real implementation, this would track scores over time
        return [
            {
                "date": datetime.now().isoformat(),
                "security_score": self.metrics.security_score
            }
        ]
    
    def _generate_high_level_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate high-level security recommendations."""
        recommendations = []
        
        # Count findings by category
        category_counts = Counter(finding.category for finding in findings)
        
        if category_counts[SecurityCategory.DEPENDENCY_SECURITY] > 0:
            recommendations.append(
                "Implement automated dependency scanning and update vulnerable packages regularly"
            )
        
        if category_counts[SecurityCategory.CODE_INJECTION] > 0:
            recommendations.append(
                "Review and strengthen input validation to prevent injection attacks"
            )
        
        if category_counts[SecurityCategory.DATA_EXPOSURE] > 0:
            recommendations.append(
                "Implement proper secret management and remove hardcoded credentials"
            )
        
        if category_counts[SecurityCategory.AUTHENTICATION] > 0:
            recommendations.append(
                "Strengthen authentication mechanisms and implement proper access controls"
            )
        
        if category_counts[SecurityCategory.DATA_ENCRYPTION] > 0:
            recommendations.append(
                "Upgrade encryption algorithms and implement proper key management"
            )
        
        if not recommendations:
            recommendations.append(
                "Maintain current security posture and continue regular security assessments"
            )
        
        return recommendations
    
    def integrate_with_health_monitoring(self, health_monitor: HealthMonitor) -> None:
        """Integrate security validation with health monitoring system."""
        
        # Add security-specific health metrics
        security_metrics = {
            "security_score": HealthMetric(
                name="Security Score",
                value=self.metrics.security_score,
                threshold_warning=80.0,
                threshold_critical=60.0,
                status=HealthStatus.HEALTHY,
                last_updated=datetime.now(),
                unit="score"
            ),
            "critical_findings": HealthMetric(
                name="Critical Security Findings",
                value=float(self.metrics.critical_findings),
                threshold_warning=1.0,
                threshold_critical=5.0,
                status=HealthStatus.HEALTHY,
                last_updated=datetime.now(),
                unit="findings"
            ),
            "high_findings": HealthMetric(
                name="High Security Findings",
                value=float(self.metrics.high_findings),
                threshold_warning=3.0,
                threshold_critical=10.0,
                status=HealthStatus.HEALTHY,
                last_updated=datetime.now(),
                unit="findings"
            )
        }
        
        # Add security metrics to health monitor
        health_monitor.metrics.update(security_metrics)
        
        # Add security health check
        def security_health_check():
            """Security-specific health check."""
            try:
                # Quick security validation
                if self.metrics.critical_findings > 0:
                    return False
                if self.metrics.security_score < 70:
                    return False
                return True
            except Exception:
                return False
        
        from .health_monitoring import SystemCheck
        security_check = SystemCheck(
            name="Security Validation",
            check_function=security_health_check,
            frequency_seconds=300.0  # Check every 5 minutes
        )
        
        health_monitor.checks.append(security_check)
        logger.info("Security validation integrated with health monitoring system")
    
    def get_security_benchmarks(self) -> Dict[str, Any]:
        """Get security performance benchmarks and metrics."""
        return {
            "scan_performance": {
                "average_scan_time": self.metrics.average_scan_time,
                "total_scans": self.metrics.total_scans,
                "last_scan_duration": self.metrics.scan_duration_seconds,
                "throughput_findings_per_second": (
                    self.metrics.total_findings / self.metrics.scan_duration_seconds
                    if self.metrics.scan_duration_seconds > 0 else 0
                )
            },
            "security_metrics": {
                "current_security_score": self.metrics.security_score,
                "total_findings": self.metrics.total_findings,
                "findings_by_severity": {
                    "critical": self.metrics.critical_findings,
                    "high": self.metrics.high_findings,
                    "medium": self.metrics.medium_findings,
                    "low": self.metrics.low_findings
                },
                "findings_by_category": self.metrics.findings_by_category,
                "resolved_findings": self.metrics.resolved_findings,
                "false_positive_rate": (
                    self.metrics.false_positives / max(1, self.metrics.total_findings)
                )
            },
            "trend_analysis": {
                "findings_trend": self._get_findings_trend(),
                "security_score_trend": self._get_security_score_history()
            },
            "benchmark_targets": {
                "target_security_score": 95.0,
                "max_critical_findings": 0,
                "max_high_findings": 2,
                "target_scan_time": 30.0,  # seconds
                "target_false_positive_rate": 0.05  # 5%
            }
        }


# Global security validation suite instance (lazy initialization to avoid circular imports)
_security_validation_suite = None

def get_security_validation_suite() -> SecurityValidationSuite:
    """Get the global security validation suite instance."""
    global _security_validation_suite
    if _security_validation_suite is None:
        _security_validation_suite = SecurityValidationSuite()
    return _security_validation_suite

# For backward compatibility
security_validation_suite = get_security_validation_suite()