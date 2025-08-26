"""Enhanced security utilities with advanced encryption for grid systems."""

import re
import ast
import os
import time
import hashlib
import hmac
import base64
import secrets
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from collections import defaultdict
# Optional cryptography imports with fallback
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.x509.oid import NameOID
    from cryptography import x509
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    
    # Mock classes for basic functionality
    class Fernet:
        def __init__(self, key): self.key = key
        def encrypt(self, data): return base64.b64encode(data)
        def decrypt(self, data): return base64.b64decode(data)
        @staticmethod
        def generate_key(): return base64.b64encode(secrets.token_bytes(32))

try:
    from cryptography.x509.verification import PolicyBuilder, StoreBuilder
    CRYPTOGRAPHY_X509_VERIFICATION_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_X509_VERIFICATION_AVAILABLE = False
    
    # Mock classes for basic functionality
    class PolicyBuilder:
        def __init__(self): pass
    class StoreBuilder:
        def __init__(self): pass

logger = logging.getLogger(__name__)


class EncryptionLevel(Enum):
    """Encryption strength levels."""
    BASIC = "basic"  # Fernet (AES 128)
    STANDARD = "standard"  # AES 256
    HIGH = "high"  # RSA 2048 + AES 256
    MAXIMUM = "maximum"  # RSA 4096 + AES 256 + HMAC


class SecurityRole(Enum):
    """Security roles for access control."""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    CLIENT = "client"
    AUDITOR = "auditor"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    role: SecurityRole
    permissions: List[str]
    session_token: str
    expiry_time: datetime
    client_cert_fingerprint: Optional[str] = None
    additional_claims: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        return datetime.now() > self.expiry_time
    
    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


@dataclass
class EncryptedMessage:
    """Encrypted message container."""
    encrypted_data: bytes
    encryption_level: EncryptionLevel
    sender_id: str
    recipient_id: str
    timestamp: datetime
    signature: Optional[bytes] = None
    key_fingerprint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "encrypted_data": base64.b64encode(self.encrypted_data).decode('utf-8'),
            "encryption_level": self.encryption_level.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp.isoformat(),
            "signature": base64.b64encode(self.signature).decode('utf-8') if self.signature else None,
            "key_fingerprint": self.key_fingerprint
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedMessage':
        return cls(
            encrypted_data=base64.b64decode(data["encrypted_data"]),
            encryption_level=EncryptionLevel(data["encryption_level"]),
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            signature=base64.b64decode(data["signature"]) if data["signature"] else None,
            key_fingerprint=data.get("key_fingerprint")
        )


class AdvancedEncryption:
    """Advanced encryption system for federated communication."""
    
    def __init__(self, encryption_level: EncryptionLevel = EncryptionLevel.HIGH):
        self.encryption_level = encryption_level
        self.symmetric_keys: Dict[str, bytes] = {}
        self.asymmetric_keys: Dict[str, Tuple[Any, Any]] = {}  # (private_key, public_key)
        self.certificates: Dict[str, x509.Certificate] = {}
        self.trusted_ca_certs: List[x509.Certificate] = []
        
        # Key rotation
        self.key_rotation_interval = timedelta(hours=24)
        self.last_key_rotation: Dict[str, datetime] = {}
        
        # Generate master keys
        self._initialize_master_keys()
        
        logger.info(f"Advanced encryption initialized with level: {encryption_level.value}")
    
    def _initialize_master_keys(self):
        """Initialize master encryption keys."""
        # Generate symmetric master key
        if self.encryption_level in [EncryptionLevel.BASIC, EncryptionLevel.STANDARD]:
            self.master_key = Fernet.generate_key()
            self.fernet = Fernet(self.master_key)
        
        # Generate asymmetric keys for higher security levels
        if self.encryption_level in [EncryptionLevel.HIGH, EncryptionLevel.MAXIMUM]:
            key_size = 4096 if self.encryption_level == EncryptionLevel.MAXIMUM else 2048
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            public_key = private_key.public_key()
            
            self.master_private_key = private_key
            self.master_public_key = public_key
            
            # Store in keys dictionary
            self.asymmetric_keys["master"] = (private_key, public_key)
    
    def generate_client_keypair(self, client_id: str) -> Tuple[bytes, bytes]:
        """Generate RSA keypair for a client."""
        key_size = 4096 if self.encryption_level == EncryptionLevel.MAXIMUM else 2048
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        public_key = private_key.public_key()
        
        # Store keys
        self.asymmetric_keys[client_id] = (private_key, public_key)
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        logger.info(f"Generated keypair for client: {client_id}")
        return private_pem, public_pem
    
    def generate_symmetric_key(self, client_id: str) -> bytes:
        """Generate symmetric key for a client."""
        key = secrets.token_bytes(32)  # 256-bit key
        self.symmetric_keys[client_id] = key
        self.last_key_rotation[client_id] = datetime.now()
        
        logger.debug(f"Generated symmetric key for client: {client_id}")
        return key
    
    def encrypt_message(
        self, 
        data: Union[str, bytes, Dict],
        recipient_id: str,
        sender_id: str = "master"
    ) -> EncryptedMessage:
        """Encrypt a message for secure transmission."""
        # Serialize data if needed
        if isinstance(data, dict):
            plaintext = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            plaintext = data.encode('utf-8')
        else:
            plaintext = data
        
        timestamp = datetime.now()
        
        if self.encryption_level == EncryptionLevel.BASIC:
            encrypted_data = self.fernet.encrypt(plaintext)
            signature = None
            key_fingerprint = None
            
        elif self.encryption_level == EncryptionLevel.STANDARD:
            # AES-256-GCM encryption
            key = self.symmetric_keys.get(recipient_id)
            if not key:
                key = self.generate_symmetric_key(recipient_id)
            
            # Generate random IV
            iv = secrets.token_bytes(12)  # 96-bit IV for GCM
            
            # Encrypt with AES-GCM
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            # Combine IV + tag + ciphertext
            encrypted_data = iv + encryptor.tag + ciphertext
            signature = None
            key_fingerprint = hashlib.sha256(key).hexdigest()[:16]
            
        elif self.encryption_level in [EncryptionLevel.HIGH, EncryptionLevel.MAXIMUM]:
            # Hybrid encryption: RSA + AES
            
            # Generate session key
            session_key = secrets.token_bytes(32)
            
            # Encrypt data with AES
            iv = secrets.token_bytes(12)
            cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv))
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            
            # Encrypt session key with RSA
            recipient_public_key = self.asymmetric_keys.get(recipient_id, (None, None))[1]
            if not recipient_public_key:
                recipient_public_key = self.master_public_key
            
            encrypted_session_key = recipient_public_key.encrypt(
                session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted session key + IV + tag + ciphertext
            encrypted_data = encrypted_session_key + iv + encryptor.tag + ciphertext
            
            # Digital signature for maximum security
            signature = None
            if self.encryption_level == EncryptionLevel.MAXIMUM:
                sender_private_key = self.asymmetric_keys.get(sender_id, (None, None))[0]
                if sender_private_key:
                    signature = sender_private_key.sign(
                        plaintext,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
            
            key_fingerprint = hashlib.sha256(session_key).hexdigest()[:16]
        
        return EncryptedMessage(
            encrypted_data=encrypted_data,
            encryption_level=self.encryption_level,
            sender_id=sender_id,
            recipient_id=recipient_id,
            timestamp=timestamp,
            signature=signature,
            key_fingerprint=key_fingerprint
        )
    
    def decrypt_message(
        self, 
        encrypted_message: EncryptedMessage,
        recipient_id: str = "master"
    ) -> Union[str, bytes, Dict]:
        """Decrypt a received message."""
        
        encrypted_data = encrypted_message.encrypted_data
        
        if encrypted_message.encryption_level == EncryptionLevel.BASIC:
            plaintext = self.fernet.decrypt(encrypted_data)
            
        elif encrypted_message.encryption_level == EncryptionLevel.STANDARD:
            # AES-256-GCM decryption
            key = self.symmetric_keys.get(encrypted_message.sender_id)
            if not key:
                raise SecurityViolationError(f"No symmetric key for sender: {encrypted_message.sender_id}")
            
            # Extract components
            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            # Decrypt
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
        elif encrypted_message.encryption_level in [EncryptionLevel.HIGH, EncryptionLevel.MAXIMUM]:
            # Hybrid decryption: RSA + AES
            
            # Determine key size for RSA
            key_size = 512 if encrypted_message.encryption_level == EncryptionLevel.MAXIMUM else 256
            
            # Extract components
            encrypted_session_key = encrypted_data[:key_size]
            iv = encrypted_data[key_size:key_size+12]
            tag = encrypted_data[key_size+12:key_size+28]
            ciphertext = encrypted_data[key_size+28:]
            
            # Decrypt session key with RSA
            recipient_private_key = self.asymmetric_keys.get(recipient_id, (None, None))[0]
            if not recipient_private_key:
                recipient_private_key = self.master_private_key
            
            session_key = recipient_private_key.decrypt(
                encrypted_session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt data with AES
            cipher = Cipher(algorithms.AES(session_key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Verify signature if present
            if encrypted_message.signature and encrypted_message.encryption_level == EncryptionLevel.MAXIMUM:
                sender_public_key = self.asymmetric_keys.get(encrypted_message.sender_id, (None, None))[1]
                if sender_public_key:
                    try:
                        sender_public_key.verify(
                            encrypted_message.signature,
                            plaintext,
                            padding.PSS(
                                mgf=padding.MGF1(hashes.SHA256()),
                                salt_length=padding.PSS.MAX_LENGTH
                            ),
                            hashes.SHA256()
                        )
                        logger.debug(f"Signature verified for sender: {encrypted_message.sender_id}")
                    except Exception as e:
                        raise SecurityViolationError(f"Signature verification failed: {e}")
        
        # Try to decode as JSON, otherwise return as string
        try:
            return json.loads(plaintext.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                return plaintext.decode('utf-8')
            except UnicodeDecodeError:
                return plaintext
    
    def should_rotate_key(self, client_id: str) -> bool:
        """Check if a client's key should be rotated."""
        last_rotation = self.last_key_rotation.get(client_id)
        if not last_rotation:
            return True
        
        return datetime.now() - last_rotation > self.key_rotation_interval
    
    def rotate_client_key(self, client_id: str):
        """Rotate a client's symmetric key."""
        if client_id in self.symmetric_keys:
            old_key = self.symmetric_keys[client_id]
            new_key = self.generate_symmetric_key(client_id)
            
            logger.info(f"Rotated symmetric key for client: {client_id}")
            return new_key
        
        return None
    
    def get_public_key_pem(self, client_id: str) -> Optional[bytes]:
        """Get public key in PEM format for a client."""
        if client_id in self.asymmetric_keys:
            _, public_key = self.asymmetric_keys[client_id]
            return public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        return None
    
    def load_public_key_pem(self, client_id: str, public_key_pem: bytes):
        """Load a public key from PEM format."""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            
            # Update or create entry
            if client_id in self.asymmetric_keys:
                private_key, _ = self.asymmetric_keys[client_id]
                self.asymmetric_keys[client_id] = (private_key, public_key)
            else:
                self.asymmetric_keys[client_id] = (None, public_key)
            
            logger.info(f"Loaded public key for client: {client_id}")
        except Exception as e:
            raise SecurityViolationError(f"Failed to load public key: {e}")


class SecureKeyExchange:
    """Secure key exchange protocol for federated clients."""
    
    def __init__(self, encryption_system: AdvancedEncryption):
        self.encryption = encryption_system
        self.pending_exchanges: Dict[str, Dict] = {}
        self.completed_exchanges: Dict[str, datetime] = {}
        
    def initiate_key_exchange(self, client_id: str) -> Dict[str, Any]:
        """Initiate key exchange with a client."""
        # Generate challenge
        challenge = secrets.token_bytes(32)
        timestamp = datetime.now()
        
        # Store pending exchange
        self.pending_exchanges[client_id] = {
            "challenge": challenge,
            "timestamp": timestamp,
            "status": "pending"
        }
        
        # Create exchange initiation message
        exchange_data = {
            "type": "key_exchange_init",
            "challenge": base64.b64encode(challenge).decode('utf-8'),
            "timestamp": timestamp.isoformat(),
            "server_public_key": base64.b64encode(
                self.encryption.get_public_key_pem("master")
            ).decode('utf-8')
        }
        
        logger.info(f"Initiated key exchange with client: {client_id}")
        return exchange_data
    
    def process_key_exchange_response(
        self, 
        client_id: str, 
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process client's key exchange response."""
        
        if client_id not in self.pending_exchanges:
            raise SecurityViolationError(f"No pending key exchange for client: {client_id}")
        
        pending = self.pending_exchanges[client_id]
        
        # Verify challenge response
        expected_challenge = pending["challenge"]
        received_challenge = base64.b64decode(response_data["challenge_response"])
        
        if not hmac.compare_digest(expected_challenge, received_challenge):
            raise SecurityViolationError("Key exchange challenge verification failed")
        
        # Load client public key
        client_public_key_pem = base64.b64decode(response_data["client_public_key"])
        self.encryption.load_public_key_pem(client_id, client_public_key_pem)
        
        # Generate symmetric key for this client
        symmetric_key = self.encryption.generate_symmetric_key(client_id)
        
        # Mark exchange as completed
        self.completed_exchanges[client_id] = datetime.now()
        del self.pending_exchanges[client_id]
        
        # Create completion message with encrypted symmetric key
        completion_data = {
            "type": "key_exchange_complete",
            "status": "success",
            "encrypted_symmetric_key": base64.b64encode(
                self.encryption.asymmetric_keys[client_id][1].encrypt(
                    symmetric_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            ).decode('utf-8')
        }
        
        logger.info(f"Completed key exchange with client: {client_id}")
        return completion_data


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
                (r'(?<!pattern.*?)eval\s*\(', 'Use of eval() can execute arbitrary code'),
                (r'(?<!pattern.*?)exec\s*\(', 'Use of exec() can execute arbitrary code'),
                (r'__import__\s*\(', 'Dynamic imports can be dangerous'),
                (r'subprocess\.call\s*\(', 'Subprocess calls need input validation'),
                (r'os\.system\s*\(', 'OS system calls are dangerous'),
            ],
            'high': [
                (r'pickle\.loads?\s*\(', 'Pickle deserialization can execute code'),
                (r'yaml\.load\s*\(', 'YAML load without safe_load is dangerous'),
                (r'(?<!validate_)input\s*\(', 'Raw input() can be dangerous in Python 2'),
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
            r'(?<!pattern.*?)eval\s*\(': 'Use ast.literal_eval() for safe evaluation',
            r'(?<!pattern.*?)exec\s*\(': 'Avoid exec(), use specific function calls',
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
            'timestamp': time.time(),
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


class SessionManager:
    """Manages secure sessions for authenticated users."""
    
    def __init__(self, session_timeout: timedelta = timedelta(hours=8)):
        self.session_timeout = session_timeout
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.session_lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(
        self, 
        user_id: str, 
        role: SecurityRole,
        permissions: List[str],
        client_cert_fingerprint: Optional[str] = None
    ) -> SecurityContext:
        """Create a new secure session."""
        
        session_token = secrets.token_urlsafe(32)
        expiry_time = datetime.now() + self.session_timeout
        
        context = SecurityContext(
            user_id=user_id,
            role=role,
            permissions=permissions,
            session_token=session_token,
            expiry_time=expiry_time,
            client_cert_fingerprint=client_cert_fingerprint
        )
        
        with self.session_lock:
            self.active_sessions[session_token] = context
        
        logger.info(f"Created session for user: {user_id}")
        return context
    
    def validate_session(self, session_token: str) -> Optional[SecurityContext]:
        """Validate and return security context for a session."""
        
        with self.session_lock:
            context = self.active_sessions.get(session_token)
        
        if not context:
            return None
        
        if context.is_expired():
            self.revoke_session(session_token)
            return None
        
        return context
    
    def revoke_session(self, session_token: str):
        """Revoke a session."""
        with self.session_lock:
            if session_token in self.active_sessions:
                user_id = self.active_sessions[session_token].user_id
                del self.active_sessions[session_token]
                logger.info(f"Revoked session for user: {user_id}")
    
    def _cleanup_expired_sessions(self):
        """Background thread to cleanup expired sessions."""
        while True:
            try:
                current_time = datetime.now()
                expired_tokens = []
                
                with self.session_lock:
                    for token, context in self.active_sessions.items():
                        if context.expiry_time <= current_time:
                            expired_tokens.append(token)
                
                for token in expired_tokens:
                    self.revoke_session(token)
                
                # Sleep for 5 minutes before next cleanup
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                time.sleep(60)  # Shorter sleep on error


class SecureCommunicationManager:
    """Manages secure communication for federated grid control."""
    
    def __init__(
        self, 
        encryption_level: EncryptionLevel = EncryptionLevel.HIGH,
        enable_certificate_validation: bool = True
    ):
        self.encryption = AdvancedEncryption(encryption_level)
        self.key_exchange = SecureKeyExchange(self.encryption)
        self.session_manager = SessionManager()
        self.enable_certificate_validation = enable_certificate_validation
        
        # Message queues for secure communication
        self.outbound_queue: Dict[str, List[EncryptedMessage]] = defaultdict(list)
        self.inbound_queue: Dict[str, List[EncryptedMessage]] = defaultdict(list)
        
        # Communication statistics
        self.message_stats = {
            "sent": 0,
            "received": 0,
            "encryption_errors": 0,
            "signature_failures": 0
        }
        
        logger.info("Secure communication manager initialized")
    
    def register_client(self, client_id: str) -> Dict[str, Any]:
        """Register a new client for secure communication."""
        
        # Generate keypair for client
        private_key_pem, public_key_pem = self.encryption.generate_client_keypair(client_id)
        
        # Initiate key exchange
        exchange_data = self.key_exchange.initiate_key_exchange(client_id)
        
        registration_data = {
            "client_id": client_id,
            "client_private_key": base64.b64encode(private_key_pem).decode('utf-8'),
            "client_public_key": base64.b64encode(public_key_pem).decode('utf-8'),
            "key_exchange": exchange_data,
            "encryption_level": self.encryption.encryption_level.value
        }
        
        logger.info(f"Registered client for secure communication: {client_id}")
        return registration_data
    
    def send_secure_message(
        self, 
        recipient_id: str, 
        message_data: Union[str, Dict, bytes],
        sender_id: str = "server"
    ) -> str:
        """Send a secure message to a client."""
        
        try:
            # Encrypt message
            encrypted_msg = self.encryption.encrypt_message(
                message_data, recipient_id, sender_id
            )
            
            # Add to outbound queue
            self.outbound_queue[recipient_id].append(encrypted_msg)
            self.message_stats["sent"] += 1
            
            # Generate message ID for tracking
            message_id = hashlib.sha256(
                f"{encrypted_msg.sender_id}_{encrypted_msg.recipient_id}_{encrypted_msg.timestamp}".encode()
            ).hexdigest()[:16]
            
            logger.debug(f"Queued secure message {message_id} for {recipient_id}")
            return message_id
            
        except Exception as e:
            self.message_stats["encryption_errors"] += 1
            logger.error(f"Failed to send secure message to {recipient_id}: {e}")
            raise
    
    def receive_secure_message(
        self, 
        encrypted_message_data: Dict[str, Any],
        recipient_id: str = "server"
    ) -> Union[str, Dict, bytes]:
        """Receive and decrypt a secure message."""
        
        try:
            # Reconstruct encrypted message
            encrypted_msg = EncryptedMessage.from_dict(encrypted_message_data)
            
            # Decrypt message
            decrypted_data = self.encryption.decrypt_message(encrypted_msg, recipient_id)
            
            # Add to inbound queue for audit
            self.inbound_queue[encrypted_msg.sender_id].append(encrypted_msg)
            self.message_stats["received"] += 1
            
            logger.debug(f"Received and decrypted message from {encrypted_msg.sender_id}")
            return decrypted_data
            
        except Exception as e:
            if "signature" in str(e).lower():
                self.message_stats["signature_failures"] += 1
            else:
                self.message_stats["encryption_errors"] += 1
            
            logger.error(f"Failed to receive secure message: {e}")
            raise
    
    def get_pending_messages(self, client_id: str) -> List[Dict[str, Any]]:
        """Get pending messages for a client."""
        messages = self.outbound_queue.get(client_id, [])
        
        # Convert to dictionaries and clear queue
        message_dicts = [msg.to_dict() for msg in messages]
        self.outbound_queue[client_id] = []
        
        return message_dicts
    
    def rotate_client_keys(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Rotate encryption keys for a client."""
        if self.encryption.should_rotate_key(client_id):
            new_key = self.encryption.rotate_client_key(client_id)
            
            if new_key:
                # Send new key to client securely
                key_update_msg = {
                    "type": "key_rotation",
                    "new_key_fingerprint": hashlib.sha256(new_key).hexdigest()[:16],
                    "timestamp": datetime.now().isoformat()
                }
                
                self.send_secure_message(client_id, key_update_msg, "server")
                
                return {
                    "client_id": client_id,
                    "rotated": True,
                    "new_key_fingerprint": key_update_msg["new_key_fingerprint"]
                }
        
        return None
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            **self.message_stats,
            "active_clients": len(self.encryption.symmetric_keys),
            "pending_key_exchanges": len(self.key_exchange.pending_exchanges),
            "completed_key_exchanges": len(self.key_exchange.completed_exchanges),
            "outbound_queue_size": sum(len(msgs) for msgs in self.outbound_queue.values()),
            "inbound_queue_size": sum(len(msgs) for msgs in self.inbound_queue.values()),
            "encryption_level": self.encryption.encryption_level.value
        }


class SecurityViolationError(Exception):
    """Security policy violation detected."""
    def __init__(self, message: str, violation_type: str = "general"):
        super().__init__(message)
        self.violation_type = violation_type


# Enhanced security auditor with encryption capabilities
class EnhancedSecurityAuditor(SecurityAuditor):
    """Enhanced security auditor with encryption and communication security."""
    
    def __init__(self):
        super().__init__()
        self.communication_manager = None
        
    def set_communication_manager(self, comm_manager: SecureCommunicationManager):
        """Set communication manager for security audits."""
        self.communication_manager = comm_manager
    
    def audit_communication_security(self) -> Dict[str, Any]:
        """Audit communication security."""
        if not self.communication_manager:
            return {"error": "No communication manager configured"}
        
        stats = self.communication_manager.get_communication_stats()
        
        # Security assessment
        security_score = 100
        issues = []
        recommendations = []
        
        # Check encryption level
        if stats["encryption_level"] == EncryptionLevel.BASIC.value:
            security_score -= 30
            issues.append("Basic encryption level provides minimal security")
            recommendations.append("Upgrade to HIGH or MAXIMUM encryption level")
        elif stats["encryption_level"] == EncryptionLevel.STANDARD.value:
            security_score -= 10
            issues.append("Standard encryption lacks digital signatures")
            recommendations.append("Consider upgrading to HIGH encryption level")
        
        # Check error rates
        total_messages = stats["sent"] + stats["received"]
        if total_messages > 0:
            error_rate = (stats["encryption_errors"] + stats["signature_failures"]) / total_messages
            if error_rate > 0.05:  # 5% error rate
                security_score -= 20
                issues.append(f"High encryption error rate: {error_rate:.1%}")
                recommendations.append("Investigate encryption errors and key management")
        
        # Check key rotation
        if stats["active_clients"] > 0 and stats["completed_key_exchanges"] == 0:
            security_score -= 15
            issues.append("No key exchanges completed")
            recommendations.append("Ensure proper key exchange with all clients")
        
        return {
            "communication_stats": stats,
            "security_score": max(0, security_score),
            "issues": issues,
            "recommendations": recommendations,
            "audit_timestamp": datetime.now().isoformat()
        }


# Integration with comprehensive security validation
def integrate_security_validation():
    """Integrate with comprehensive security validation system."""
    try:
        from .security_validation import get_security_validation_suite
        from .health_monitoring import system_health
        security_validation_suite = get_security_validation_suite()
        
        # Integrate security validation with health monitoring
        security_validation_suite.integrate_with_health_monitoring(system_health)
        
        # Set up enhanced security auditor with communication manager
        global_security_auditor.set_communication_manager(global_communication_manager)
        
        logger.info("Security validation integration completed successfully")
        return True
    except ImportError as e:
        logger.warning(f"Security validation integration not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to integrate security validation: {e}")
        return False


# Enhanced security audit with validation suite integration
def run_comprehensive_security_audit(include_validation_suite: bool = True) -> Dict[str, Any]:
    """Run comprehensive security audit including validation suite."""
    audit_results = {}
    
    # Run traditional security audit
    traditional_audit = global_security_auditor.audit_system()
    audit_results["traditional_audit"] = traditional_audit
    
    # Run communication security audit
    comm_audit = global_security_auditor.audit_communication_security()
    audit_results["communication_audit"] = comm_audit
    
    # Run comprehensive validation suite if available
    if include_validation_suite:
        try:
            from .security_validation import get_security_validation_suite
            security_validation_suite = get_security_validation_suite()
            validation_results = security_validation_suite.run_comprehensive_scan()
            audit_results["comprehensive_validation"] = validation_results
            
            # Get security benchmarks
            benchmarks = security_validation_suite.get_security_benchmarks()
            audit_results["security_benchmarks"] = benchmarks
            
        except ImportError:
            logger.warning("Comprehensive security validation suite not available")
            audit_results["comprehensive_validation"] = {
                "status": "not_available",
                "message": "Security validation suite not imported"
            }
        except Exception as e:
            logger.error(f"Error running comprehensive validation: {e}")
            audit_results["comprehensive_validation"] = {
                "status": "error",
                "error": str(e)
            }
    
    # Generate combined security score
    combined_score = _calculate_combined_security_score(audit_results)
    audit_results["combined_security_score"] = combined_score
    
    return audit_results


def _calculate_combined_security_score(audit_results: Dict[str, Any]) -> float:
    """Calculate combined security score from all audits."""
    scores = []
    weights = []
    
    # Traditional audit score (weight: 0.3)
    if "traditional_audit" in audit_results:
        trad_audit = audit_results["traditional_audit"]
        if "summary" in trad_audit and "total_issues" in trad_audit["summary"]:
            total_issues = trad_audit["summary"]["total_issues"]
            trad_score = max(0, 100 - (total_issues * 5))  # 5 points per issue
            scores.append(trad_score)
            weights.append(0.3)
    
    # Communication audit score (weight: 0.2)
    if "communication_audit" in audit_results:
        comm_audit = audit_results["communication_audit"]
        if "security_score" in comm_audit:
            scores.append(comm_audit["security_score"])
            weights.append(0.2)
    
    # Comprehensive validation score (weight: 0.5)
    if "comprehensive_validation" in audit_results:
        comp_val = audit_results["comprehensive_validation"]
        if isinstance(comp_val, dict) and "scan_summary" in comp_val:
            if "security_score" in comp_val["scan_summary"]:
                scores.append(comp_val["scan_summary"]["security_score"])
                weights.append(0.5)
    
    # Calculate weighted average
    if scores and weights:
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    return 0.0


# Global instances
global_security_auditor = EnhancedSecurityAuditor()
global_communication_manager = SecureCommunicationManager()

# Note: Security validation integration is performed lazily to avoid circular imports
# Call integrate_security_validation() manually when needed