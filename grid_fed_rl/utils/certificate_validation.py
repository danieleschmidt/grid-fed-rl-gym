"""Certificate validation and secure authentication system for grid operations."""

import os
import logging
import hashlib
import secrets
import base64
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
import threading

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID, AuthorityInformationAccessOID
from cryptography.x509.verification import PolicyBuilder, StoreBuilder
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

from .exceptions import SecurityViolationError, NonRetryableError
from .distributed_tracing import trace_federated_operation

logger = logging.getLogger(__name__)


class CertificateType(Enum):
    """Types of certificates in the grid system."""
    ROOT_CA = "root_ca"
    INTERMEDIATE_CA = "intermediate_ca"
    SERVER = "server"
    CLIENT = "client"
    DEVICE = "device"  # For grid devices like sensors, controllers
    OPERATOR = "operator"  # For human operators


class ValidationResult(Enum):
    """Certificate validation results."""
    VALID = "valid"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    REVOKED = "revoked"
    UNTRUSTED = "untrusted"
    INVALID_SIGNATURE = "invalid_signature"
    INVALID_PURPOSE = "invalid_purpose"
    MISSING_EXTENSIONS = "missing_extensions"
    ERROR = "error"


@dataclass
class CertificateInfo:
    """Information about a certificate."""
    certificate: x509.Certificate
    cert_type: CertificateType
    subject_name: str
    issuer_name: str
    serial_number: str
    not_before: datetime
    not_after: datetime
    fingerprint: str
    key_usage: List[str] = field(default_factory=list)
    extended_key_usage: List[str] = field(default_factory=list)
    san_dns_names: List[str] = field(default_factory=list)
    san_ip_addresses: List[str] = field(default_factory=list)
    
    @classmethod
    def from_certificate(cls, cert: x509.Certificate, cert_type: CertificateType) -> 'CertificateInfo':
        """Create CertificateInfo from an x509 certificate."""
        
        # Extract subject and issuer names
        subject_name = cert.subject.rfc4514_string()
        issuer_name = cert.issuer.rfc4514_string()
        
        # Generate fingerprint
        fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
        
        # Extract key usage
        key_usage = []
        try:
            ku_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.KEY_USAGE)
            ku = ku_ext.value
            if ku.digital_signature:
                key_usage.append("digital_signature")
            if ku.content_commitment:
                key_usage.append("content_commitment")
            if ku.key_encipherment:
                key_usage.append("key_encipherment")
            if ku.data_encipherment:
                key_usage.append("data_encipherment")
            if ku.key_agreement:
                key_usage.append("key_agreement")
            if ku.key_cert_sign:
                key_usage.append("key_cert_sign")
            if ku.crl_sign:
                key_usage.append("crl_sign")
        except x509.ExtensionNotFound:
            pass
        
        # Extract extended key usage
        extended_key_usage = []
        try:
            eku_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.EXTENDED_KEY_USAGE)
            eku = eku_ext.value
            for usage in eku:
                if usage == ExtendedKeyUsageOID.SERVER_AUTH:
                    extended_key_usage.append("server_auth")
                elif usage == ExtendedKeyUsageOID.CLIENT_AUTH:
                    extended_key_usage.append("client_auth")
                elif usage == ExtendedKeyUsageOID.CODE_SIGNING:
                    extended_key_usage.append("code_signing")
                elif usage == ExtendedKeyUsageOID.TIME_STAMPING:
                    extended_key_usage.append("time_stamping")
        except x509.ExtensionNotFound:
            pass
        
        # Extract Subject Alternative Names
        san_dns_names = []
        san_ip_addresses = []
        try:
            san_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            san = san_ext.value
            san_dns_names = [name.value for name in san.get_values_for_type(x509.DNSName)]
            san_ip_addresses = [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]
        except x509.ExtensionNotFound:
            pass
        
        return cls(
            certificate=cert,
            cert_type=cert_type,
            subject_name=subject_name,
            issuer_name=issuer_name,
            serial_number=str(cert.serial_number),
            not_before=cert.not_valid_before.replace(tzinfo=timezone.utc),
            not_after=cert.not_valid_after.replace(tzinfo=timezone.utc),
            fingerprint=fingerprint,
            key_usage=key_usage,
            extended_key_usage=extended_key_usage,
            san_dns_names=san_dns_names,
            san_ip_addresses=san_ip_addresses
        )
    
    def is_expired(self, check_time: Optional[datetime] = None) -> bool:
        """Check if certificate is expired."""
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        return check_time > self.not_after
    
    def is_not_yet_valid(self, check_time: Optional[datetime] = None) -> bool:
        """Check if certificate is not yet valid."""
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        return check_time < self.not_before
    
    def days_until_expiry(self, check_time: Optional[datetime] = None) -> int:
        """Get days until certificate expires."""
        if check_time is None:
            check_time = datetime.now(timezone.utc)
        delta = self.not_after - check_time
        return max(0, delta.days)


@dataclass
class ValidationReport:
    """Certificate validation report."""
    certificate_info: CertificateInfo
    result: ValidationResult
    trust_chain: List[CertificateInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    policy_violations: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.result == ValidationResult.VALID
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "subject_name": self.certificate_info.subject_name,
            "issuer_name": self.certificate_info.issuer_name,
            "serial_number": self.certificate_info.serial_number,
            "fingerprint": self.certificate_info.fingerprint,
            "cert_type": self.certificate_info.cert_type.value,
            "result": self.result.value,
            "not_before": self.certificate_info.not_before.isoformat(),
            "not_after": self.certificate_info.not_after.isoformat(),
            "days_until_expiry": self.certificate_info.days_until_expiry(),
            "key_usage": self.certificate_info.key_usage,
            "extended_key_usage": self.certificate_info.extended_key_usage,
            "trust_chain_length": len(self.trust_chain),
            "errors": self.errors,
            "warnings": self.warnings,
            "policy_violations": self.policy_violations,
            "validation_time": self.validation_time.isoformat()
        }


class CertificateRevocationList:
    """Certificate Revocation List (CRL) management."""
    
    def __init__(self):
        self.revoked_certificates: Set[str] = set()  # Serial numbers
        self.revoked_fingerprints: Set[str] = set()
        self.crl_update_time: Optional[datetime] = None
        self.lock = threading.Lock()
    
    def add_revoked_certificate(self, serial_number: str, fingerprint: str):
        """Add a certificate to the revocation list."""
        with self.lock:
            self.revoked_certificates.add(serial_number)
            self.revoked_fingerprints.add(fingerprint)
            self.crl_update_time = datetime.now(timezone.utc)
        
        logger.warning(f"Certificate revoked - Serial: {serial_number}, Fingerprint: {fingerprint[:16]}...")
    
    def is_revoked(self, cert_info: CertificateInfo) -> bool:
        """Check if a certificate is revoked."""
        with self.lock:
            return (cert_info.serial_number in self.revoked_certificates or 
                    cert_info.fingerprint in self.revoked_fingerprints)
    
    def load_from_file(self, crl_path: str):
        """Load CRL from file (placeholder - would integrate with actual CRL format)."""
        try:
            if os.path.exists(crl_path):
                with open(crl_path, 'r') as f:
                    crl_data = json.load(f)
                
                with self.lock:
                    self.revoked_certificates.update(crl_data.get('serial_numbers', []))
                    self.revoked_fingerprints.update(crl_data.get('fingerprints', []))
                    self.crl_update_time = datetime.now(timezone.utc)
                
                logger.info(f"Loaded CRL with {len(self.revoked_certificates)} revoked certificates")
        except Exception as e:
            logger.error(f"Failed to load CRL from {crl_path}: {e}")
    
    def save_to_file(self, crl_path: str):
        """Save CRL to file."""
        try:
            with self.lock:
                crl_data = {
                    'serial_numbers': list(self.revoked_certificates),
                    'fingerprints': list(self.revoked_fingerprints),
                    'update_time': self.crl_update_time.isoformat() if self.crl_update_time else None
                }
            
            os.makedirs(os.path.dirname(crl_path), exist_ok=True)
            with open(crl_path, 'w') as f:
                json.dump(crl_data, f, indent=2)
                
            logger.info(f"Saved CRL to {crl_path}")
        except Exception as e:
            logger.error(f"Failed to save CRL to {crl_path}: {e}")


class CertificateAuthority:
    """Certificate Authority for grid system."""
    
    def __init__(
        self, 
        ca_cert_path: str,
        ca_key_path: str,
        ca_key_password: Optional[str] = None
    ):
        self.ca_cert_path = ca_cert_path
        self.ca_key_path = ca_key_path
        self.ca_key_password = ca_key_password
        
        # Load CA certificate and private key
        self.ca_cert = self._load_ca_certificate()
        self.ca_private_key = self._load_ca_private_key()
        
        # Certificate database
        self.issued_certificates: Dict[str, CertificateInfo] = {}
        self.certificate_counter = 1
        
        logger.info(f"Certificate Authority initialized - CA: {self.ca_cert.subject.rfc4514_string()}")
    
    def _load_ca_certificate(self) -> x509.Certificate:
        """Load CA certificate from file."""
        try:
            with open(self.ca_cert_path, 'rb') as f:
                return x509.load_pem_x509_certificate(f.read())
        except Exception as e:
            raise SecurityViolationError(f"Failed to load CA certificate: {e}")
    
    def _load_ca_private_key(self):
        """Load CA private key from file."""
        try:
            with open(self.ca_key_path, 'rb') as f:
                key_data = f.read()
            
            password = self.ca_key_password.encode() if self.ca_key_password else None
            return serialization.load_pem_private_key(key_data, password=password)
        except Exception as e:
            raise SecurityViolationError(f"Failed to load CA private key: {e}")
    
    def issue_certificate(
        self,
        subject_name: str,
        cert_type: CertificateType,
        public_key,
        validity_days: int = 365,
        san_dns_names: Optional[List[str]] = None,
        san_ip_addresses: Optional[List[str]] = None
    ) -> CertificateInfo:
        """Issue a new certificate."""
        
        # Create subject name
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Grid-Fed-RL"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, cert_type.value.title()),
        ])
        
        # Generate serial number
        serial_number = self.certificate_counter
        self.certificate_counter += 1
        
        # Set validity period
        not_before = datetime.now(timezone.utc)
        not_after = not_before + timedelta(days=validity_days)
        
        # Build certificate
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(self.ca_cert.subject)
        builder = builder.public_key(public_key)
        builder = builder.serial_number(serial_number)
        builder = builder.not_valid_before(not_before)
        builder = builder.not_valid_after(not_after)
        
        # Add extensions based on certificate type
        builder = self._add_extensions(builder, cert_type, san_dns_names, san_ip_addresses)
        
        # Sign certificate
        certificate = builder.sign(self.ca_private_key, hashes.SHA256())
        
        # Create certificate info
        cert_info = CertificateInfo.from_certificate(certificate, cert_type)
        
        # Store in database
        self.issued_certificates[cert_info.fingerprint] = cert_info
        
        logger.info(f"Issued {cert_type.value} certificate for {subject_name}")
        return cert_info
    
    def _add_extensions(
        self, 
        builder: x509.CertificateBuilder, 
        cert_type: CertificateType,
        san_dns_names: Optional[List[str]] = None,
        san_ip_addresses: Optional[List[str]] = None
    ) -> x509.CertificateBuilder:
        """Add appropriate extensions based on certificate type."""
        
        # Basic constraints
        if cert_type in [CertificateType.ROOT_CA, CertificateType.INTERMEDIATE_CA]:
            builder = builder.add_extension(
                x509.BasicConstraints(ca=True, path_length=None),
                critical=True
            )
        else:
            builder = builder.add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True
            )
        
        # Key usage
        if cert_type in [CertificateType.ROOT_CA, CertificateType.INTERMEDIATE_CA]:
            key_usage = x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False
            )
        else:
            key_usage = x509.KeyUsage(
                digital_signature=True,
                content_commitment=True,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False
            )
        
        builder = builder.add_extension(key_usage, critical=True)
        
        # Extended key usage
        extended_usages = []
        if cert_type == CertificateType.SERVER:
            extended_usages.append(ExtendedKeyUsageOID.SERVER_AUTH)
        elif cert_type in [CertificateType.CLIENT, CertificateType.OPERATOR]:
            extended_usages.append(ExtendedKeyUsageOID.CLIENT_AUTH)
        elif cert_type == CertificateType.DEVICE:
            extended_usages.extend([
                ExtendedKeyUsageOID.CLIENT_AUTH,
                ExtendedKeyUsageOID.SERVER_AUTH
            ])
        
        if extended_usages:
            builder = builder.add_extension(
                x509.ExtendedKeyUsage(extended_usages),
                critical=True
            )
        
        # Subject Alternative Names
        san_values = []
        if san_dns_names:
            san_values.extend([x509.DNSName(name) for name in san_dns_names])
        if san_ip_addresses:
            import ipaddress
            for ip_str in san_ip_addresses:
                try:
                    ip = ipaddress.ip_address(ip_str)
                    san_values.append(x509.IPAddress(ip))
                except ValueError:
                    logger.warning(f"Invalid IP address in SAN: {ip_str}")
        
        if san_values:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_values),
                critical=False
            )
        
        # Subject Key Identifier
        builder = builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(builder._public_key),
            critical=False
        )
        
        # Authority Key Identifier
        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(self.ca_cert.public_key()),
            critical=False
        )
        
        return builder
    
    def get_issued_certificate(self, fingerprint: str) -> Optional[CertificateInfo]:
        """Get an issued certificate by fingerprint."""
        return self.issued_certificates.get(fingerprint)
    
    def list_issued_certificates(self, cert_type: Optional[CertificateType] = None) -> List[CertificateInfo]:
        """List all issued certificates, optionally filtered by type."""
        certs = list(self.issued_certificates.values())
        
        if cert_type:
            certs = [cert for cert in certs if cert.cert_type == cert_type]
        
        return certs


class CertificateValidator:
    """Advanced certificate validator for grid security."""
    
    def __init__(
        self,
        trusted_ca_certs: List[x509.Certificate],
        crl: Optional[CertificateRevocationList] = None,
        require_extended_key_usage: bool = True
    ):
        self.trusted_ca_certs = trusted_ca_certs
        self.crl = crl or CertificateRevocationList()
        self.require_extended_key_usage = require_extended_key_usage
        
        # Build trust store
        self.trust_store = self._build_trust_store()
        
        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "expired_certificates": 0,
            "revoked_certificates": 0
        }
        
        logger.info(f"Certificate validator initialized with {len(trusted_ca_certs)} trusted CAs")
    
    def _build_trust_store(self):
        """Build certificate trust store."""
        store_builder = StoreBuilder()
        
        for ca_cert in self.trusted_ca_certs:
            store_builder = store_builder.add_certs([ca_cert])
        
        return store_builder.build()
    
    @trace_federated_operation("validate_certificate", "security")
    def validate_certificate(
        self,
        cert: x509.Certificate,
        cert_type: CertificateType,
        purpose: Optional[str] = None,
        hostname: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> ValidationReport:
        """Validate a certificate comprehensively."""
        
        self.validation_stats["total_validations"] += 1
        
        cert_info = CertificateInfo.from_certificate(cert, cert_type)
        report = ValidationReport(certificate_info=cert_info, result=ValidationResult.VALID)
        
        try:
            # 1. Time validity check
            current_time = datetime.now(timezone.utc)
            
            if cert_info.is_not_yet_valid(current_time):
                report.result = ValidationResult.NOT_YET_VALID
                report.errors.append(f"Certificate not yet valid until {cert_info.not_before}")
                return report
            
            if cert_info.is_expired(current_time):
                report.result = ValidationResult.EXPIRED
                report.errors.append(f"Certificate expired on {cert_info.not_after}")
                self.validation_stats["expired_certificates"] += 1
                return report
            
            # Add expiry warning if certificate expires soon
            days_until_expiry = cert_info.days_until_expiry(current_time)
            if days_until_expiry <= 30:
                report.warnings.append(f"Certificate expires in {days_until_expiry} days")
            
            # 2. Revocation check
            if self.crl.is_revoked(cert_info):
                report.result = ValidationResult.REVOKED
                report.errors.append("Certificate has been revoked")
                self.validation_stats["revoked_certificates"] += 1
                return report
            
            # 3. Trust chain validation
            try:
                trust_chain = self._validate_trust_chain(cert)
                report.trust_chain = [CertificateInfo.from_certificate(c, CertificateType.INTERMEDIATE_CA) 
                                    for c in trust_chain]
            except Exception as e:
                report.result = ValidationResult.UNTRUSTED
                report.errors.append(f"Trust chain validation failed: {e}")
                return report
            
            # 4. Signature validation
            try:
                self._validate_signature(cert)
            except Exception as e:
                report.result = ValidationResult.INVALID_SIGNATURE
                report.errors.append(f"Signature validation failed: {e}")
                return report
            
            # 5. Purpose validation
            if purpose:
                purpose_valid, purpose_errors = self._validate_purpose(cert_info, purpose)
                if not purpose_valid:
                    report.result = ValidationResult.INVALID_PURPOSE
                    report.errors.extend(purpose_errors)
                    return report
            
            # 6. Extensions validation
            ext_valid, ext_errors, ext_warnings = self._validate_extensions(cert_info, cert_type)
            if not ext_valid:
                report.result = ValidationResult.MISSING_EXTENSIONS
                report.errors.extend(ext_errors)
                return report
            report.warnings.extend(ext_warnings)
            
            # 7. Hostname/IP validation for server certificates
            if cert_type == CertificateType.SERVER:
                if hostname and not self._validate_hostname(cert_info, hostname):
                    report.warnings.append(f"Certificate not valid for hostname: {hostname}")
                
                if ip_address and not self._validate_ip_address(cert_info, ip_address):
                    report.warnings.append(f"Certificate not valid for IP address: {ip_address}")
            
            # 8. Policy compliance check
            policy_violations = self._check_policy_compliance(cert_info, cert_type)
            report.policy_violations = policy_violations
            if policy_violations:
                report.warnings.extend([f"Policy violation: {v}" for v in policy_violations])
            
            self.validation_stats["successful_validations"] += 1
            
        except Exception as e:
            report.result = ValidationResult.ERROR
            report.errors.append(f"Validation error: {e}")
            self.validation_stats["failed_validations"] += 1
            logger.error(f"Certificate validation error: {e}")
        
        return report
    
    def _validate_trust_chain(self, cert: x509.Certificate) -> List[x509.Certificate]:
        """Validate certificate trust chain."""
        # Use cryptography library's chain validation
        policy = PolicyBuilder().store(self.trust_store).build()
        
        try:
            # This is a simplified approach - in production you'd want more sophisticated chain building
            chain = policy.build_chain([cert])
            return list(chain)
        except Exception as e:
            raise SecurityViolationError(f"Trust chain validation failed: {e}")
    
    def _validate_signature(self, cert: x509.Certificate):
        """Validate certificate signature."""
        # Find issuer certificate
        issuer_cert = None
        for ca_cert in self.trusted_ca_certs:
            if cert.issuer == ca_cert.subject:
                issuer_cert = ca_cert
                break
        
        if not issuer_cert:
            raise SecurityViolationError("Issuer certificate not found in trust store")
        
        # Verify signature
        try:
            issuer_public_key = issuer_cert.public_key()
            issuer_public_key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm
            )
        except Exception as e:
            raise SecurityViolationError(f"Signature verification failed: {e}")
    
    def _validate_purpose(self, cert_info: CertificateInfo, purpose: str) -> Tuple[bool, List[str]]:
        """Validate certificate purpose against extended key usage."""
        errors = []
        
        purpose_mappings = {
            "server_auth": "server_auth",
            "client_auth": "client_auth",
            "code_signing": "code_signing",
            "time_stamping": "time_stamping"
        }
        
        required_eku = purpose_mappings.get(purpose)
        if not required_eku:
            errors.append(f"Unknown purpose: {purpose}")
            return False, errors
        
        if self.require_extended_key_usage and required_eku not in cert_info.extended_key_usage:
            errors.append(f"Certificate does not have required extended key usage: {required_eku}")
            return False, errors
        
        return True, errors
    
    def _validate_extensions(
        self, 
        cert_info: CertificateInfo, 
        cert_type: CertificateType
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate certificate extensions."""
        errors = []
        warnings = []
        
        # Key usage validation
        required_key_usages = {
            CertificateType.ROOT_CA: ["digital_signature", "key_cert_sign", "crl_sign"],
            CertificateType.INTERMEDIATE_CA: ["digital_signature", "key_cert_sign", "crl_sign"],
            CertificateType.SERVER: ["digital_signature", "key_encipherment"],
            CertificateType.CLIENT: ["digital_signature", "key_encipherment"],
            CertificateType.DEVICE: ["digital_signature", "key_encipherment"],
            CertificateType.OPERATOR: ["digital_signature", "key_encipherment"]
        }
        
        required_usages = required_key_usages.get(cert_type, [])
        missing_usages = [usage for usage in required_usages if usage not in cert_info.key_usage]
        
        if missing_usages:
            errors.append(f"Missing required key usages: {', '.join(missing_usages)}")
        
        # Extended key usage validation
        required_extended_usages = {
            CertificateType.SERVER: ["server_auth"],
            CertificateType.CLIENT: ["client_auth"],
            CertificateType.OPERATOR: ["client_auth"],
            CertificateType.DEVICE: ["client_auth"]
        }
        
        required_ext_usages = required_extended_usages.get(cert_type, [])
        missing_ext_usages = [usage for usage in required_ext_usages 
                             if usage not in cert_info.extended_key_usage]
        
        if self.require_extended_key_usage and missing_ext_usages:
            errors.append(f"Missing required extended key usages: {', '.join(missing_ext_usages)}")
        
        return len(errors) == 0, errors, warnings
    
    def _validate_hostname(self, cert_info: CertificateInfo, hostname: str) -> bool:
        """Validate certificate against hostname."""
        # Check CN
        if hostname.lower() in cert_info.subject_name.lower():
            return True
        
        # Check SAN DNS names
        for san_name in cert_info.san_dns_names:
            if hostname.lower() == san_name.lower():
                return True
            # Simple wildcard support
            if san_name.startswith("*.") and hostname.lower().endswith(san_name[1:].lower()):
                return True
        
        return False
    
    def _validate_ip_address(self, cert_info: CertificateInfo, ip_address: str) -> bool:
        """Validate certificate against IP address."""
        return ip_address in cert_info.san_ip_addresses
    
    def _check_policy_compliance(
        self, 
        cert_info: CertificateInfo, 
        cert_type: CertificateType
    ) -> List[str]:
        """Check certificate against security policies."""
        violations = []
        
        # Key size policy (example)
        try:
            public_key = cert_info.certificate.public_key()
            if hasattr(public_key, 'key_size'):
                if public_key.key_size < 2048:
                    violations.append(f"Key size {public_key.key_size} below minimum 2048 bits")
        except Exception:
            pass
        
        # Validity period policy
        validity_period = cert_info.not_after - cert_info.not_before
        max_validity_days = {
            CertificateType.ROOT_CA: 3652,  # 10 years
            CertificateType.INTERMEDIATE_CA: 1826,  # 5 years
            CertificateType.SERVER: 825,  # ~2 years
            CertificateType.CLIENT: 365,  # 1 year
            CertificateType.DEVICE: 730,  # 2 years
            CertificateType.OPERATOR: 365  # 1 year
        }
        
        max_days = max_validity_days.get(cert_type, 365)
        if validity_period.days > max_days:
            violations.append(f"Validity period {validity_period.days} days exceeds maximum {max_days} days")
        
        return violations
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        
        stats = self.validation_stats.copy()
        stats["success_rate"] = (
            self.validation_stats["successful_validations"] / total 
            if total > 0 else 0.0
        )
        
        return stats
    
    def validate_certificate_chain(
        self, 
        cert_chain: List[x509.Certificate],
        cert_types: List[CertificateType]
    ) -> List[ValidationReport]:
        """Validate a complete certificate chain."""
        reports = []
        
        if len(cert_chain) != len(cert_types):
            raise ValueError("Certificate chain and types lists must have same length")
        
        for cert, cert_type in zip(cert_chain, cert_types):
            report = self.validate_certificate(cert, cert_type)
            reports.append(report)
            
            # Stop if any certificate in chain is invalid
            if not report.is_valid():
                logger.warning(f"Certificate chain validation failed at {cert_type.value} certificate")
                break
        
        return reports


class SecureAuthenticator:
    """Secure authentication system using certificates."""
    
    def __init__(
        self,
        certificate_validator: CertificateValidator,
        session_timeout: timedelta = timedelta(hours=8)
    ):
        self.validator = certificate_validator
        self.session_timeout = session_timeout
        self.authenticated_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Secure authenticator initialized")
    
    def authenticate_certificate(
        self, 
        cert_pem: bytes,
        cert_type: CertificateType,
        client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Authenticate using client certificate."""
        
        try:
            # Parse certificate
            cert = x509.load_pem_x509_certificate(cert_pem)
            
            # Validate certificate
            report = self.validator.validate_certificate(cert, cert_type, "client_auth")
            
            if not report.is_valid():
                raise SecurityViolationError(f"Certificate validation failed: {report.errors}")
            
            # Extract client information
            subject_name = report.certificate_info.subject_name
            fingerprint = report.certificate_info.fingerprint
            
            if client_id is None:
                # Extract client ID from certificate CN
                try:
                    cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
                    client_id = cn
                except (IndexError, AttributeError):
                    raise SecurityViolationError("Cannot determine client ID from certificate")
            
            # Create session
            session_token = secrets.token_urlsafe(32)
            session_data = {
                "client_id": client_id,
                "subject_name": subject_name,
                "cert_fingerprint": fingerprint,
                "cert_type": cert_type.value,
                "authenticated_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + self.session_timeout,
                "validation_report": report.to_dict()
            }
            
            with self.session_lock:
                self.authenticated_sessions[session_token] = session_data
            
            logger.info(f"Successfully authenticated client: {client_id}")
            
            return {
                "authenticated": True,
                "session_token": session_token,
                "client_id": client_id,
                "expires_at": session_data["expires_at"].isoformat(),
                "warnings": report.warnings,
                "cert_expires_in_days": report.certificate_info.days_until_expiry()
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {
                "authenticated": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate an existing session."""
        with self.session_lock:
            session_data = self.authenticated_sessions.get(session_token)
        
        if not session_data:
            return None
        
        # Check expiration
        if datetime.now(timezone.utc) > session_data["expires_at"]:
            self.revoke_session(session_token)
            return None
        
        return session_data
    
    def revoke_session(self, session_token: str):
        """Revoke a session."""
        with self.session_lock:
            if session_token in self.authenticated_sessions:
                client_id = self.authenticated_sessions[session_token]["client_id"]
                del self.authenticated_sessions[session_token]
                logger.info(f"Revoked session for client: {client_id}")
    
    def _cleanup_expired_sessions(self):
        """Background cleanup of expired sessions."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                expired_tokens = []
                
                with self.session_lock:
                    for token, session_data in self.authenticated_sessions.items():
                        if current_time > session_data["expires_at"]:
                            expired_tokens.append(token)
                
                for token in expired_tokens:
                    self.revoke_session(token)
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                time.sleep(60)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions."""
        with self.session_lock:
            return [
                {
                    "client_id": data["client_id"],
                    "authenticated_at": data["authenticated_at"].isoformat(),
                    "expires_at": data["expires_at"].isoformat(),
                    "cert_type": data["cert_type"],
                    "cert_fingerprint": data["cert_fingerprint"][:16] + "..."
                }
                for data in self.authenticated_sessions.values()
            ]
    
    def get_authentication_statistics(self) -> Dict[str, Any]:
        """Get authentication statistics."""
        with self.session_lock:
            active_sessions = len(self.authenticated_sessions)
            
            # Group by certificate type
            by_cert_type = {}
            for session_data in self.authenticated_sessions.values():
                cert_type = session_data["cert_type"]
                by_cert_type[cert_type] = by_cert_type.get(cert_type, 0) + 1
        
        return {
            "active_sessions": active_sessions,
            "sessions_by_cert_type": by_cert_type,
            "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
            "validator_stats": self.validator.get_validation_statistics()
        }