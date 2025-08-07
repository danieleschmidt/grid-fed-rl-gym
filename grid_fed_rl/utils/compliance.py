"""Compliance and regulatory utilities for global deployment."""

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


@dataclass
class ComplianceRecord:
    """Record for compliance audit trail."""
    timestamp: datetime
    regulation: str
    requirement: str
    status: str  # compliant, non_compliant, not_applicable
    evidence: Optional[str] = None
    remediation_required: bool = False
    remediation_deadline: Optional[datetime] = None


@dataclass
class DataProcessingRecord:
    """Record for data processing activities (GDPR compliance)."""
    data_type: str
    processing_purpose: str
    legal_basis: str
    data_subject_consent: bool
    retention_period: timedelta
    data_minimization: bool
    encryption_applied: bool
    access_controls: List[str]


class GDPRCompliance:
    """GDPR (General Data Protection Regulation) compliance utilities."""
    
    def __init__(self):
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        
    def record_data_processing(
        self,
        data_type: str,
        processing_purpose: str,
        legal_basis: str,
        consent_given: bool = False,
        retention_days: int = 30,
        encrypted: bool = True,
        access_controls: List[str] = None
    ) -> str:
        """Record data processing activity for GDPR compliance."""
        
        if access_controls is None:
            access_controls = ["authenticated_users"]
        
        record = DataProcessingRecord(
            data_type=data_type,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_subject_consent=consent_given,
            retention_period=timedelta(days=retention_days),
            data_minimization=self._check_data_minimization(data_type, processing_purpose),
            encryption_applied=encrypted,
            access_controls=access_controls
        )
        
        self.processing_records.append(record)
        
        record_id = hashlib.md5(
            f"{data_type}_{processing_purpose}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        logger.info(f"Data processing recorded: {record_id}")
        return record_id
    
    def record_consent(
        self,
        data_subject_id: str,
        processing_purposes: List[str],
        consent_given: bool,
        consent_method: str = "explicit"
    ) -> None:
        """Record consent for data processing."""
        
        self.consent_records[data_subject_id] = {
            "purposes": processing_purposes,
            "consent_given": consent_given,
            "consent_method": consent_method,
            "timestamp": datetime.now(),
            "withdrawn": False,
            "withdrawal_timestamp": None
        }
        
        logger.info(f"Consent recorded for subject: {data_subject_id}")
    
    def withdraw_consent(self, data_subject_id: str) -> bool:
        """Withdraw consent for data subject."""
        
        if data_subject_id in self.consent_records:
            self.consent_records[data_subject_id]["withdrawn"] = True
            self.consent_records[data_subject_id]["withdrawal_timestamp"] = datetime.now()
            logger.info(f"Consent withdrawn for subject: {data_subject_id}")
            return True
        
        return False
    
    def check_retention_compliance(self) -> List[Dict[str, Any]]:
        """Check data retention compliance."""
        
        violations = []
        current_time = datetime.now()
        
        for i, record in enumerate(self.processing_records):
            # Calculate when data should be deleted
            deletion_due = current_time - record.retention_period
            
            if current_time > deletion_due:
                violations.append({
                    "record_index": i,
                    "data_type": record.data_type,
                    "processing_purpose": record.processing_purpose,
                    "retention_exceeded_by": current_time - deletion_due,
                    "action_required": "delete_data"
                })
        
        return violations
    
    def _check_data_minimization(self, data_type: str, processing_purpose: str) -> bool:
        """Check if data collection follows minimization principle."""
        
        # Simplified data minimization check
        sensitive_data_types = [
            "personal_identification", "location_data", "biometric_data",
            "financial_data", "health_data", "communication_content"
        ]
        
        if data_type in sensitive_data_types:
            # For sensitive data, only specific purposes justify collection
            justified_purposes = [
                "legal_compliance", "vital_interests", "public_task",
                "contract_performance", "legitimate_interests"
            ]
            return processing_purpose in justified_purposes
        
        return True
    
    def generate_privacy_notice(self, language: str = "en") -> str:
        """Generate privacy notice based on processing activities."""
        
        notices = {
            "en": {
                "title": "Privacy Notice - Grid Federated RL System",
                "intro": "This notice describes how we process your data:",
                "purposes": "Data Processing Purposes:",
                "rights": "Your Rights: You have the right to access, rectify, delete, and port your data.",
                "contact": "Contact our Data Protection Officer for questions."
            },
            "de": {
                "title": "Datenschutzerklärung - Grid Federated RL System", 
                "intro": "Diese Erklärung beschreibt, wie wir Ihre Daten verarbeiten:",
                "purposes": "Zwecke der Datenverarbeitung:",
                "rights": "Ihre Rechte: Sie haben das Recht auf Zugang, Berichtigung, Löschung und Übertragung Ihrer Daten.",
                "contact": "Kontaktieren Sie unseren Datenschutzbeauftragten bei Fragen."
            }
        }
        
        template = notices.get(language, notices["en"])
        
        notice = f"{template['title']}\n\n"
        notice += f"{template['intro']}\n\n"
        notice += f"{template['purposes']}\n"
        
        for record in self.processing_records:
            notice += f"- {record.processing_purpose}: {record.data_type}\n"
        
        notice += f"\n{template['rights']}\n\n"
        notice += f"{template['contact']}\n"
        
        return notice


class CCPACompliance:
    """CCPA (California Consumer Privacy Act) compliance utilities."""
    
    def __init__(self):
        self.data_categories = [
            "identifiers", "commercial_information", "internet_activity",
            "geolocation", "biometric_information", "professional_information"
        ]
        self.consumer_requests: List[Dict[str, Any]] = []
        
    def handle_consumer_request(
        self,
        request_type: str,  # access, delete, opt_out
        consumer_id: str,
        data_categories: List[str] = None
    ) -> str:
        """Handle CCPA consumer privacy requests."""
        
        if data_categories is None:
            data_categories = []
        
        request = {
            "request_id": hashlib.md5(f"{consumer_id}_{request_type}_{datetime.now()}".encode()).hexdigest()[:16],
            "type": request_type,
            "consumer_id": consumer_id,
            "categories": data_categories,
            "timestamp": datetime.now(),
            "status": "received",
            "response_due": datetime.now() + timedelta(days=45)  # CCPA requirement
        }
        
        self.consumer_requests.append(request)
        logger.info(f"CCPA request received: {request['request_id']}")
        
        return request["request_id"]
    
    def process_access_request(self, request_id: str) -> Dict[str, Any]:
        """Process consumer data access request."""
        
        # Find request
        request = next((r for r in self.consumer_requests if r["request_id"] == request_id), None)
        
        if not request:
            return {"error": "Request not found"}
        
        if request["type"] != "access":
            return {"error": "Not an access request"}
        
        # Simulate data collection
        consumer_data = {
            "categories_collected": self.data_categories,
            "sources": ["direct_interaction", "automated_collection"],
            "business_purposes": ["operational", "analytics", "compliance"],
            "third_party_sharing": False,
            "retention_period": "As specified in privacy policy"
        }
        
        # Update request status
        request["status"] = "completed"
        request["completion_date"] = datetime.now()
        
        return {
            "request_id": request_id,
            "consumer_data": consumer_data,
            "completion_date": datetime.now()
        }


class PIIPLCompliance:
    """PIPL (Personal Information Protection Law) compliance for China."""
    
    def __init__(self):
        self.cross_border_transfers: List[Dict[str, Any]] = []
        
    def record_cross_border_transfer(
        self,
        destination_country: str,
        data_type: str,
        transfer_purpose: str,
        adequacy_decision: bool = False,
        standard_contractual_clauses: bool = False
    ) -> str:
        """Record cross-border data transfer."""
        
        transfer_id = hashlib.md5(
            f"{destination_country}_{data_type}_{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        transfer = {
            "transfer_id": transfer_id,
            "destination_country": destination_country,
            "data_type": data_type,
            "purpose": transfer_purpose,
            "timestamp": datetime.now(),
            "adequacy_decision": adequacy_decision,
            "contractual_clauses": standard_contractual_clauses,
            "approved": adequacy_decision or standard_contractual_clauses
        }
        
        self.cross_border_transfers.append(transfer)
        logger.info(f"Cross-border transfer recorded: {transfer_id}")
        
        return transfer_id
    
    def check_transfer_compliance(self) -> List[Dict[str, Any]]:
        """Check compliance of cross-border transfers."""
        
        non_compliant = []
        
        for transfer in self.cross_border_transfers:
            if not transfer["approved"]:
                non_compliant.append({
                    "transfer_id": transfer["transfer_id"],
                    "destination": transfer["destination_country"],
                    "issue": "No adequate legal basis for transfer",
                    "recommendation": "Obtain adequacy decision or implement SCCs"
                })
        
        return non_compliant


class ComplianceManager:
    """Comprehensive compliance management system."""
    
    def __init__(self):
        self.gdpr = GDPRCompliance()
        self.ccpa = CCPACompliance()
        self.pipl = PIIPLCompliance()
        
        self.compliance_records: List[ComplianceRecord] = []
        self.regulatory_frameworks = {
            "GDPR": ["EU", "EEA"],
            "CCPA": ["California", "US"],
            "PIPL": ["China"],
            "PDPA": ["Singapore"],
            "LGPD": ["Brazil"],
            "PIPEDA": ["Canada"]
        }
        
    def determine_applicable_regulations(self, regions: List[str]) -> List[str]:
        """Determine which regulations apply based on deployment regions."""
        
        applicable = []
        
        for regulation, jurisdictions in self.regulatory_frameworks.items():
            for region in regions:
                if any(jurisdiction.lower() in region.lower() for jurisdiction in jurisdictions):
                    if regulation not in applicable:
                        applicable.append(regulation)
        
        return applicable
    
    def create_compliance_record(
        self,
        regulation: str,
        requirement: str,
        status: str,
        evidence: str = None,
        remediation_needed: bool = False,
        remediation_days: int = 30
    ) -> str:
        """Create a compliance audit record."""
        
        record = ComplianceRecord(
            timestamp=datetime.now(),
            regulation=regulation,
            requirement=requirement,
            status=status,
            evidence=evidence,
            remediation_required=remediation_needed,
            remediation_deadline=datetime.now() + timedelta(days=remediation_days) if remediation_needed else None
        )
        
        self.compliance_records.append(record)
        
        record_id = hashlib.md5(
            f"{regulation}_{requirement}_{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        logger.info(f"Compliance record created: {record_id}")
        return record_id
    
    def run_compliance_audit(self, regions: List[str]) -> Dict[str, Any]:
        """Run comprehensive compliance audit for specified regions."""
        
        applicable_regulations = self.determine_applicable_regulations(regions)
        
        audit_results = {
            "timestamp": datetime.now(),
            "regions": regions,
            "applicable_regulations": applicable_regulations,
            "compliance_status": {},
            "violations": [],
            "recommendations": []
        }
        
        # Check GDPR compliance
        if "GDPR" in applicable_regulations:
            gdpr_violations = self.gdpr.check_retention_compliance()
            audit_results["compliance_status"]["GDPR"] = {
                "retention_violations": len(gdpr_violations),
                "processing_records": len(self.gdpr.processing_records),
                "consent_records": len(self.gdpr.consent_records)
            }
            
            if gdpr_violations:
                audit_results["violations"].extend([
                    f"GDPR retention violation: {v['data_type']}" for v in gdpr_violations
                ])
                audit_results["recommendations"].append(
                    "Implement automated data retention and deletion processes"
                )
        
        # Check CCPA compliance
        if "CCPA" in applicable_regulations:
            pending_requests = [r for r in self.ccpa.consumer_requests if r["status"] == "received"]
            overdue_requests = [
                r for r in pending_requests 
                if datetime.now() > r["response_due"]
            ]
            
            audit_results["compliance_status"]["CCPA"] = {
                "pending_requests": len(pending_requests),
                "overdue_requests": len(overdue_requests)
            }
            
            if overdue_requests:
                audit_results["violations"].extend([
                    f"CCPA overdue request: {r['request_id']}" for r in overdue_requests
                ])
                audit_results["recommendations"].append(
                    "Process consumer privacy requests within 45 days"
                )
        
        # Check PIPL compliance
        if "PIPL" in applicable_regulations:
            transfer_violations = self.pipl.check_transfer_compliance()
            audit_results["compliance_status"]["PIPL"] = {
                "cross_border_transfers": len(self.pipl.cross_border_transfers),
                "transfer_violations": len(transfer_violations)
            }
            
            if transfer_violations:
                audit_results["violations"].extend([
                    f"PIPL transfer violation: {v['destination']}" for v in transfer_violations
                ])
                audit_results["recommendations"].append(
                    "Ensure legal basis for all cross-border data transfers"
                )
        
        # Overall compliance score
        total_violations = len(audit_results["violations"])
        total_checks = len(applicable_regulations) * 3  # Rough estimate
        
        audit_results["compliance_score"] = max(0, (total_checks - total_violations) / total_checks * 100)
        
        return audit_results
    
    def generate_compliance_report(
        self,
        regions: List[str],
        output_format: str = "json"
    ) -> str:
        """Generate comprehensive compliance report."""
        
        audit_results = self.run_compliance_audit(regions)
        
        if output_format == "json":
            return json.dumps(audit_results, default=str, indent=2)
        
        elif output_format == "text":
            report = f"Compliance Report - {datetime.now().strftime('%Y-%m-%d')}\n"
            report += "=" * 50 + "\n\n"
            
            report += f"Regions: {', '.join(regions)}\n"
            report += f"Applicable Regulations: {', '.join(audit_results['applicable_regulations'])}\n"
            report += f"Compliance Score: {audit_results['compliance_score']:.1f}%\n\n"
            
            if audit_results["violations"]:
                report += "Violations:\n"
                for violation in audit_results["violations"]:
                    report += f"- {violation}\n"
                report += "\n"
            
            if audit_results["recommendations"]:
                report += "Recommendations:\n"
                for rec in audit_results["recommendations"]:
                    report += f"- {rec}\n"
            
            return report
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


# Global compliance manager
global_compliance_manager = ComplianceManager()