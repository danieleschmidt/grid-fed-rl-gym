"""Compliance framework for global data protection and regulatory requirements."""

import time
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore/Malaysia)
    PIPL = "pipl"          # Personal Information Protection Law (China)
    LGPD = "lgpd"          # Lei Geral de Proteção de Dados (Brazil)
    POPIA = "popia"        # Protection of Personal Information Act (South Africa)
    PIPEDA = "pipeda"      # Personal Information Protection and Electronic Documents Act (Canada)
    # Additional power grid specific regulations
    NERC_CIP = "nerc_cip"  # NERC Critical Infrastructure Protection (North America)
    FERC = "ferc"          # Federal Energy Regulatory Commission (US)
    IEEE_2030 = "ieee_2030" # IEEE Smart Grid Standards (International)
    IEC_61850 = "iec_61850" # IEC Power System Communication (International)


class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL = "personal"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    PSEUDONYMIZED = "pseudonymized"
    ANONYMOUS = "anonymous"
    PUBLIC = "public"


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    GRID_SIMULATION = "grid_simulation"
    PERFORMANCE_MONITORING = "performance_monitoring"
    RESEARCH_DEVELOPMENT = "research_development"
    COMPLIANCE_REPORTING = "compliance_reporting"
    SYSTEM_OPTIMIZATION = "system_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    TRAINING_MODELS = "training_models"


@dataclass
class DataSubject:
    """Information about data subject."""
    id: str
    category: str  # user, operator, system, etc.
    jurisdiction: str
    consent_given: bool = False
    consent_timestamp: Optional[datetime] = None
    retention_period_days: int = 2555  # Default 7 years


@dataclass
class DataRecord:
    """Individual data record with compliance metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_category: DataCategory = DataCategory.TECHNICAL
    subject_id: Optional[str] = None
    purpose: ProcessingPurpose = ProcessingPurpose.GRID_SIMULATION
    created_timestamp: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    retention_until: Optional[datetime] = None
    encrypted: bool = False
    pseudonymized: bool = False
    jurisdiction: str = "US"
    legal_basis: str = "legitimate_interest"
    
    def __post_init__(self):
        if self.retention_until is None:
            # Default retention based on jurisdiction
            if self.jurisdiction in ["EU", "UK"]:
                self.retention_until = self.created_timestamp + timedelta(days=2555)  # 7 years
            elif self.jurisdiction == "CN":
                self.retention_until = self.created_timestamp + timedelta(days=1095)  # 3 years
            else:
                self.retention_until = self.created_timestamp + timedelta(days=2555)  # 7 years


class ComplianceManager:
    """Manage compliance with data protection regulations."""
    
    def __init__(self, framework: ComplianceFramework = ComplianceFramework.GDPR):
        self.framework = framework
        self.data_records: Dict[str, DataRecord] = {}
        self.data_subjects: Dict[str, DataSubject] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.consent_records: Dict[str, Dict] = {}
        self.deletion_requests: List[Dict[str, Any]] = []
        
        # Framework-specific configurations
        self.framework_config = self._get_framework_config(framework)
        
    def _get_framework_config(self, framework: ComplianceFramework) -> Dict[str, Any]:
        """Get configuration for specific compliance framework."""
        configs = {
            ComplianceFramework.GDPR: {
                "requires_consent": True,
                "right_to_erasure": True,
                "right_to_portability": True,
                "right_to_rectification": True,
                "data_protection_officer_required": True,
                "breach_notification_hours": 72,
                "default_retention_days": 2555,  # 7 years
                "requires_impact_assessment": True,
                "lawful_bases": ["consent", "contract", "legal_obligation", "vital_interests", "public_task", "legitimate_interests"]
            },
            ComplianceFramework.CCPA: {
                "requires_consent": False,
                "right_to_erasure": True,
                "right_to_portability": True,
                "right_to_rectification": False,
                "data_protection_officer_required": False,
                "breach_notification_hours": None,
                "default_retention_days": 2555,  # 7 years
                "requires_impact_assessment": False,
                "lawful_bases": ["business_purpose", "commercial_purpose"]
            },
            ComplianceFramework.PDPA: {
                "requires_consent": True,
                "right_to_erasure": True,
                "right_to_portability": True,
                "right_to_rectification": True,
                "data_protection_officer_required": True,
                "breach_notification_hours": 72,
                "default_retention_days": 1826,  # 5 years
                "requires_impact_assessment": True,
                "lawful_bases": ["consent", "contract", "legal_obligation", "legitimate_interests"]
            },
            ComplianceFramework.PIPL: {
                "requires_consent": True,
                "right_to_erasure": True,
                "right_to_portability": True,
                "right_to_rectification": True,
                "data_protection_officer_required": True,
                "breach_notification_hours": 72,
                "default_retention_days": 1095,  # 3 years
                "requires_impact_assessment": True,
                "lawful_bases": ["consent", "contract", "legal_obligation", "public_interest"]
            }
        }
        
        return configs.get(framework, configs[ComplianceFramework.GDPR])
        
    def register_data_subject(self, subject: DataSubject) -> None:
        """Register a data subject."""
        self.data_subjects[subject.id] = subject
        self._log_audit_event("data_subject_registered", {"subject_id": subject.id})
        
    def record_data_processing(self, record: DataRecord) -> str:
        """Record data processing activity."""
        # Validate compliance requirements
        if not self._validate_processing_legality(record):
            raise ValueError(f"Data processing not legally compliant: {record.id}")
            
        self.data_records[record.id] = record
        self._log_audit_event("data_processing_recorded", {
            "record_id": record.id,
            "category": record.data_category.value,
            "purpose": record.purpose.value
        })
        
        return record.id
        
    def _validate_processing_legality(self, record: DataRecord) -> bool:
        """Validate that data processing is legally compliant."""
        # Check if consent is required and obtained
        if (self.framework_config["requires_consent"] and 
            record.data_category == DataCategory.PERSONAL and
            record.subject_id):
            
            if record.subject_id not in self.data_subjects:
                return False
                
            subject = self.data_subjects[record.subject_id]
            if not subject.consent_given:
                return False
                
        # Check retention period
        if record.retention_until and record.retention_until < datetime.now():
            return False
            
        return True
        
    def process_erasure_request(self, subject_id: str, reason: str = "data_subject_request") -> bool:
        """Process right to erasure request."""
        if not self.framework_config["right_to_erasure"]:
            return False
            
        # Find all records for subject
        records_to_delete = [
            record_id for record_id, record in self.data_records.items()
            if record.subject_id == subject_id
        ]
        
        # Check if erasure is allowed (some data may need to be retained for legal reasons)
        deletable_records = []
        for record_id in records_to_delete:
            record = self.data_records[record_id]
            if self._can_delete_record(record):
                deletable_records.append(record_id)
                
        # Delete records
        for record_id in deletable_records:
            del self.data_records[record_id]
            
        self._log_audit_event("erasure_request_processed", {
            "subject_id": subject_id,
            "reason": reason,
            "records_deleted": len(deletable_records),
            "records_retained": len(records_to_delete) - len(deletable_records)
        })
        
        return len(deletable_records) > 0
        
    def _can_delete_record(self, record: DataRecord) -> bool:
        """Check if record can be deleted considering legal obligations."""
        # Some records must be retained for legal/regulatory reasons
        if record.purpose in [ProcessingPurpose.COMPLIANCE_REPORTING]:
            return False
            
        # Check if within legal retention period for certain data
        if record.data_category == DataCategory.FINANCIAL:
            # Financial records often have longer retention requirements
            min_retention = datetime.now() - timedelta(days=2555)  # 7 years
            if record.created_timestamp > min_retention:
                return False
                
        return True
        
    def export_data_for_subject(self, subject_id: str) -> Dict[str, Any]:
        """Export all data for a subject (data portability)."""
        if not self.framework_config["right_to_portability"]:
            return {}
            
        subject_records = {
            record_id: {
                "id": record.id,
                "category": record.data_category.value,
                "purpose": record.purpose.value,
                "created": record.created_timestamp.isoformat(),
                "last_accessed": record.last_accessed.isoformat(),
                "encrypted": record.encrypted,
                "pseudonymized": record.pseudonymized
            }
            for record_id, record in self.data_records.items()
            if record.subject_id == subject_id
        }
        
        self._log_audit_event("data_export_requested", {
            "subject_id": subject_id,
            "records_exported": len(subject_records)
        })
        
        return {
            "subject_id": subject_id,
            "export_timestamp": datetime.now().isoformat(),
            "records": subject_records,
            "framework": self.framework.value
        }
        
    def cleanup_expired_data(self) -> int:
        """Clean up expired data records."""
        current_time = datetime.now()
        expired_records = []
        
        for record_id, record in self.data_records.items():
            if record.retention_until and record.retention_until < current_time:
                if self._can_delete_record(record):
                    expired_records.append(record_id)
                    
        # Delete expired records
        for record_id in expired_records:
            del self.data_records[record_id]
            
        self._log_audit_event("expired_data_cleanup", {
            "records_deleted": len(expired_records)
        })
        
        return len(expired_records)
        
    def conduct_privacy_impact_assessment(self, processing_purpose: ProcessingPurpose, 
                                        data_categories: List[DataCategory]) -> Dict[str, Any]:
        """Conduct privacy impact assessment."""
        if not self.framework_config["requires_impact_assessment"]:
            return {"required": False}
            
        # Calculate risk score
        risk_score = 0
        risk_factors = []
        
        # High-risk data categories
        if DataCategory.PERSONAL in data_categories:
            risk_score += 3
            risk_factors.append("Personal data processing")
            
        if DataCategory.FINANCIAL in data_categories:
            risk_score += 2
            risk_factors.append("Financial data processing")
            
        # High-risk purposes
        if processing_purpose in [ProcessingPurpose.RESEARCH_DEVELOPMENT, ProcessingPurpose.TRAINING_MODELS]:
            risk_score += 2
            risk_factors.append("Research/ML training purpose")
            
        # Determine risk level
        if risk_score >= 5:
            risk_level = "HIGH"
        elif risk_score >= 3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        assessment = {
            "required": True,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "recommendations": self._get_risk_mitigation_recommendations(risk_level),
            "conducted_date": datetime.now().isoformat(),
            "framework": self.framework.value
        }
        
        self._log_audit_event("privacy_impact_assessment", assessment)
        
        return assessment
        
    def _get_risk_mitigation_recommendations(self, risk_level: str) -> List[str]:
        """Get risk mitigation recommendations."""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Implement data minimization principles",
                "Use strong encryption for data at rest and in transit",
                "Implement pseudonymization where possible",
                "Conduct regular compliance audits",
                "Implement access controls and monitoring",
                "Consider appointing a Data Protection Officer"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Implement appropriate technical and organizational measures",
                "Use encryption for sensitive data",
                "Implement access logging and monitoring",
                "Regular review of data retention periods"
            ])
        else:  # LOW
            recommendations.extend([
                "Implement basic security measures",
                "Regular review of processing activities",
                "Maintain audit trail"
            ])
            
        return recommendations
        
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log audit event."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "framework": self.framework.value,
            "id": str(uuid.uuid4())
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep only last 10000 entries to prevent memory issues
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
            
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status."""
        current_time = datetime.now()
        
        # Count records by category
        category_counts = {}
        for record in self.data_records.values():
            category = record.data_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
        # Check for expired data
        expired_count = sum(
            1 for record in self.data_records.values()
            if record.retention_until and record.retention_until < current_time
        )
        
        # Check consent compliance
        consent_issues = 0
        if self.framework_config["requires_consent"]:
            for record in self.data_records.values():
                if (record.data_category == DataCategory.PERSONAL and 
                    record.subject_id and
                    record.subject_id in self.data_subjects):
                    subject = self.data_subjects[record.subject_id]
                    if not subject.consent_given:
                        consent_issues += 1
                        
        return {
            "framework": self.framework.value,
            "total_records": len(self.data_records),
            "total_subjects": len(self.data_subjects),
            "records_by_category": category_counts,
            "expired_records": expired_count,
            "consent_issues": consent_issues,
            "audit_events": len(self.audit_log),
            "last_cleanup": max([e["timestamp"] for e in self.audit_log if e["event_type"] == "expired_data_cleanup"], default="Never"),
            "compliance_score": self._calculate_compliance_score()
        }
        
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        score = 100.0
        
        # Deduct for expired data
        current_time = datetime.now()
        expired_count = sum(
            1 for record in self.data_records.values()
            if record.retention_until and record.retention_until < current_time
        )
        
        if len(self.data_records) > 0:
            expired_ratio = expired_count / len(self.data_records)
            score -= expired_ratio * 30  # Up to 30 points deduction
            
        # Deduct for consent issues
        consent_issues = 0
        if self.framework_config["requires_consent"]:
            for record in self.data_records.values():
                if (record.data_category == DataCategory.PERSONAL and 
                    record.subject_id and
                    record.subject_id in self.data_subjects):
                    subject = self.data_subjects[record.subject_id]
                    if not subject.consent_given:
                        consent_issues += 1
                        
        if len(self.data_records) > 0:
            consent_ratio = consent_issues / len(self.data_records)
            score -= consent_ratio * 40  # Up to 40 points deduction
            
        return max(0.0, score)


# Global compliance manager
global_compliance = ComplianceManager()


def set_compliance_framework(framework: str) -> None:
    """Set global compliance framework."""
    global global_compliance
    try:
        framework_enum = ComplianceFramework(framework.lower())
        global_compliance = ComplianceManager(framework_enum)
        logger.info(f"Compliance framework set to {framework}")
    except ValueError:
        logger.error(f"Invalid compliance framework: {framework}")


def record_data_activity(data_category: str, purpose: str, subject_id: Optional[str] = None) -> str:
    """Convenience function to record data activity."""
    try:
        category_enum = DataCategory(data_category.lower())
        purpose_enum = ProcessingPurpose(purpose.lower())
        
        record = DataRecord(
            data_category=category_enum,
            purpose=purpose_enum,
            subject_id=subject_id
        )
        
        return global_compliance.record_data_processing(record)
    except ValueError as e:
        logger.error(f"Invalid data activity parameters: {e}")
        return ""