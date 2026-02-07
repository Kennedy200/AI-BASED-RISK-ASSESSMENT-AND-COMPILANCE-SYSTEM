"""
KYC (Know Your Customer) verification logic.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from app.core.compliance.aml_rules import AlertSeverity


@dataclass
class KYCAlert:
    """KYC Alert data class."""
    check_id: str
    check_name: str
    severity: AlertSeverity
    description: str
    alert_type: str = "kyc"
    metadata: Optional[Dict] = None


class KYCChecker:
    """
    KYC verification and screening.
    """
    
    # Document expiry thresholds
    ID_EXPIRY_WARNING_DAYS = 30
    ID_EXPIRY_CRITICAL_DAYS = 7
    
    # Verification thresholds
    MAX_VERIFICATION_AGE_DAYS = 365  # Re-verify annually
    
    def __init__(self, sanctions_list: Optional[List[Dict]] = None):
        """
        Initialize KYC checker.
        
        Args:
            sanctions_list: Optional sanctions list for screening
        """
        self.sanctions_list = sanctions_list or []
    
    def verify_customer(
        self,
        customer: Dict[str, Any],
        documents: Optional[List[Dict]] = None
    ) -> List[KYCAlert]:
        """
        Perform full KYC verification on a customer.
        
        Args:
            customer: Customer data
            documents: Customer documents
            
        Returns:
            List of KYC alerts
        """
        alerts = []
        
        # Document checks
        if documents:
            alerts.extend(self.check_document_expiry(documents))
            alerts.extend(self.check_document_validity(documents))
        
        # Verification status
        alerts.extend(self.check_verification_status(customer))
        
        # Sanctions screening
        alerts.extend(self.screen_sanctions(customer))
        
        # PEP screening
        alerts.extend(self.screen_pep(customer))
        
        # Risk assessment
        alerts.extend(self.assess_customer_risk(customer))
        
        return alerts
    
    def check_document_expiry(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[KYCAlert]:
        """
        Check if documents are expired or expiring soon.
        """
        alerts = []
        now = datetime.utcnow()
        
        for doc in documents:
            doc_type = doc.get('type', 'ID')
            expiry_date = doc.get('expiry_date')
            
            if not expiry_date:
                continue
            
            if isinstance(expiry_date, str):
                try:
                    expiry_date = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
                except ValueError:
                    continue
            
            days_until_expiry = (expiry_date - now).days
            
            if days_until_expiry < 0:
                alerts.append(KYCAlert(
                    check_id="KYC-001",
                    check_name="Document Expired",
                    severity=AlertSeverity.CRITICAL,
                    description=f"{doc_type} document expired {abs(days_until_expiry)} days ago",
                    alert_type="kyc_document_expired",
                    metadata={'document_type': doc_type, 'expiry_date': expiry_date.isoformat()}
                ))
            elif days_until_expiry <= self.ID_EXPIRY_CRITICAL_DAYS:
                alerts.append(KYCAlert(
                    check_id="KYC-002",
                    check_name="Document Expiring Soon",
                    severity=AlertSeverity.HIGH,
                    description=f"{doc_type} document expires in {days_until_expiry} days",
                    alert_type="kyc_document_expiring",
                    metadata={'document_type': doc_type, 'days_remaining': days_until_expiry}
                ))
            elif days_until_expiry <= self.ID_EXPIRY_WARNING_DAYS:
                alerts.append(KYCAlert(
                    check_id="KYC-003",
                    check_name="Document Expiry Warning",
                    severity=AlertSeverity.MEDIUM,
                    description=f"{doc_type} document expires in {days_until_expiry} days",
                    alert_type="kyc_document_warning",
                    metadata={'document_type': doc_type, 'days_remaining': days_until_expiry}
                ))
        
        return alerts
    
    def check_document_validity(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[KYCAlert]:
        """
        Check document validity and authenticity markers.
        """
        alerts = []
        
        for doc in documents:
            # Check if document was verified
            if not doc.get('verified', False):
                alerts.append(KYCAlert(
                    check_id="KYC-004",
                    check_name="Document Not Verified",
                    severity=AlertSeverity.HIGH,
                    description=f"{doc.get('type', 'ID')} document has not been verified",
                    alert_type="kyc_document_unverified",
                    metadata={'document_type': doc.get('type')}
                ))
            
            # Check verification age
            verified_at = doc.get('verified_at')
            if verified_at:
                if isinstance(verified_at, str):
                    verified_at = datetime.fromisoformat(verified_at.replace('Z', '+00:00'))
                
                age_days = (datetime.utcnow() - verified_at).days
                if age_days > self.MAX_VERIFICATION_AGE_DAYS:
                    alerts.append(KYCAlert(
                        check_id="KYC-005",
                        check_name="Document Verification Stale",
                        severity=AlertSeverity.MEDIUM,
                        description=f"{doc.get('type', 'ID')} verification is {age_days} days old",
                        alert_type="kyc_verification_stale",
                        metadata={'document_type': doc.get('type'), 'age_days': age_days}
                    ))
        
        return alerts
    
    def check_verification_status(
        self,
        customer: Dict[str, Any]
    ) -> List[KYCAlert]:
        """
        Check overall customer verification status.
        """
        alerts = []
        
        verification_status = customer.get('verification_status', 'unverified')
        
        if verification_status == 'unverified':
            alerts.append(KYCAlert(
                check_id="KYC-006",
                check_name="Customer Unverified",
                severity=AlertSeverity.CRITICAL,
                description="Customer identity is not verified",
                alert_type="kyc_unverified"
            ))
        elif verification_status == 'pending':
            alerts.append(KYCAlert(
                check_id="KYC-007",
                check_name="Verification Pending",
                severity=AlertSeverity.MEDIUM,
                description="Customer verification is pending review",
                alert_type="kyc_pending"
            ))
        elif verification_status == 'rejected':
            alerts.append(KYCAlert(
                check_id="KYC-008",
                check_name="Verification Rejected",
                severity=AlertSeverity.CRITICAL,
                description="Customer verification was rejected",
                alert_type="kyc_rejected"
            ))
        
        # Check if enhanced due diligence is required
        if customer.get('requires_edd', False):
            alerts.append(KYCAlert(
                check_id="KYC-009",
                check_name="Enhanced Due Diligence Required",
                severity=AlertSeverity.HIGH,
                description="Customer requires enhanced due diligence",
                alert_type="kyc_edd_required"
            ))
        
        return alerts
    
    def screen_sanctions(
        self,
        customer: Dict[str, Any]
    ) -> List[KYCAlert]:
        """
        Screen customer against sanctions lists.
        """
        alerts = []
        
        if not self.sanctions_list:
            return alerts
        
        customer_name = customer.get('name', '').lower()
        
        for entry in self.sanctions_list:
            entry_name = entry.get('name', '').lower()
            
            # Simple name matching (in production, use fuzzy matching)
            if customer_name == entry_name:
                alerts.append(KYCAlert(
                    check_id="KYC-010",
                    check_name="Sanctions List Match",
                    severity=AlertSeverity.CRITICAL,
                    description=f"Customer matches sanctions list entry: {entry.get('name')}",
                    alert_type="kyc_sanctions_match",
                    metadata={
                        'list_source': entry.get('list_source'),
                        'list_program': entry.get('list_program')
                    }
                ))
        
        return alerts
    
    def screen_pep(
        self,
        customer: Dict[str, Any]
    ) -> List[KYCAlert]:
        """
        Screen customer against PEP (Politically Exposed Persons) lists.
        """
        alerts = []
        
        is_pep = customer.get('is_pep', False)
        pep_status = customer.get('pep_status')
        
        if is_pep:
            if pep_status == 'active':
                alerts.append(KYCAlert(
                    check_id="KYC-011",
                    check_name="Active PEP",
                    severity=AlertSeverity.HIGH,
                    description="Customer is an active Politically Exposed Person",
                    alert_type="kyc_active_pep",
                    metadata={'pep_role': customer.get('pep_role')}
                ))
            elif pep_status == 'former':
                alerts.append(KYCAlert(
                    check_id="KYC-012",
                    check_name="Former PEP",
                    severity=AlertSeverity.MEDIUM,
                    description="Customer is a former Politically Exposed Person",
                    alert_type="kyc_former_pep",
                    metadata={'pep_role': customer.get('pep_role')}
                ))
        
        # Check for family/associate connections
        if customer.get('pep_family_connection'):
            alerts.append(KYCAlert(
                check_id="KYC-013",
                check_name="PEP Family Connection",
                severity=AlertSeverity.MEDIUM,
                description="Customer has family connection to a PEP",
                alert_type="kyc_pep_family"
            ))
        
        return alerts
    
    def assess_customer_risk(
        self,
        customer: Dict[str, Any]
    ) -> List[KYCAlert]:
        """
        Assess customer risk profile.
        """
        alerts = []
        
        risk_score = customer.get('risk_score', 0)
        risk_rating = customer.get('risk_rating', 'low')
        
        if risk_rating == 'high' or risk_score > 80:
            alerts.append(KYCAlert(
                check_id="KYC-014",
                check_name="High Risk Customer",
                severity=AlertSeverity.HIGH,
                description=f"Customer has high risk rating (score: {risk_score})",
                alert_type="kyc_high_risk",
                metadata={'risk_score': risk_score, 'risk_rating': risk_rating}
            ))
        
        # Check for adverse media
        if customer.get('adverse_media_flag', False):
            alerts.append(KYCAlert(
                check_id="KYC-015",
                check_name="Adverse Media Found",
                severity=AlertSeverity.HIGH,
                description="Adverse media found for customer",
                alert_type="kyc_adverse_media"
            ))
        
        # Check for complex ownership structure
        if customer.get('complex_ownership_structure', False):
            alerts.append(KYCAlert(
                check_id="KYC-016",
                check_name="Complex Ownership Structure",
                severity=AlertSeverity.MEDIUM,
                description="Customer has complex ownership structure",
                alert_type="kyc_complex_ownership"
            ))
        
        return alerts


# Singleton
kyc_checker = KYCChecker()
