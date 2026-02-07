"""
Composite risk scoring combining ML and compliance factors.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from app.core.compliance.aml_rules import AMLAlert, AlertSeverity


@dataclass
class RiskScore:
    """Composite risk score result."""
    overall_score: int  # 0-100
    ml_score: int  # 0-100
    compliance_score: int  # 0-100
    risk_level: str  # low, medium, high, critical
    factors: List[Dict[str, Any]]
    recommendations: List[str]


class RiskScorer:
    """
    Calculate composite risk scores combining ML predictions and compliance alerts.
    """
    
    # Risk score weights
    ML_WEIGHT = 0.6
    COMPLIANCE_WEIGHT = 0.4
    
    # Severity scores for compliance alerts
    SEVERITY_SCORES = {
        AlertSeverity.LOW: 10,
        AlertSeverity.MEDIUM: 30,
        AlertSeverity.HIGH: 60,
        AlertSeverity.CRITICAL: 100
    }
    
    # Alert type multipliers (some alerts are more significant)
    ALERT_MULTIPLIERS = {
        'aml_structuring': 1.5,
        'aml_ctr_threshold': 1.2,
        'aml_high_risk_geography': 1.3,
        'kyc_sanctions_match': 2.0,
        'kyc_document_expired': 1.5,
        'kyc_unverified': 1.5,
    }
    
    def calculate_risk(
        self,
        ml_probability: float,
        compliance_alerts: List[AMLAlert],
        transaction_amount: Optional[float] = None,
        customer_risk_rating: Optional[str] = None
    ) -> RiskScore:
        """
        Calculate composite risk score.
        
        Args:
            ml_probability: ML model fraud probability (0-1)
            compliance_alerts: List of compliance alerts
            transaction_amount: Optional transaction amount for additional context
            customer_risk_rating: Optional customer risk rating
            
        Returns:
            RiskScore with overall score and breakdown
        """
        # ML score (0-100)
        ml_score = int(ml_probability * 100)
        
        # Compliance score (0-100)
        compliance_score = self._calculate_compliance_score(compliance_alerts)
        
        # Apply customer risk rating adjustment
        customer_multiplier = self._get_customer_risk_multiplier(customer_risk_rating)
        
        # Calculate weighted composite score
        overall_score = int(
            (ml_score * self.ML_WEIGHT + compliance_score * self.COMPLIANCE_WEIGHT) *
            customer_multiplier
        )
        
        # Cap at 100
        overall_score = min(100, overall_score)
        
        # Determine risk level
        risk_level = self._get_risk_level(overall_score)
        
        # Build factors list
        factors = self._build_risk_factors(ml_score, compliance_alerts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score, ml_score, compliance_alerts, transaction_amount
        )
        
        return RiskScore(
            overall_score=overall_score,
            ml_score=ml_score,
            compliance_score=compliance_score,
            risk_level=risk_level,
            factors=factors,
            recommendations=recommendations
        )
    
    def _calculate_compliance_score(self, alerts: List[AMLAlert]) -> int:
        """Calculate compliance risk score from alerts."""
        if not alerts:
            return 0
        
        total_score = 0
        
        for alert in alerts:
            base_score = self.SEVERITY_SCORES.get(alert.severity, 10)
            multiplier = self.ALERT_MULTIPLIERS.get(alert.alert_type, 1.0)
            total_score += base_score * multiplier
        
        # Cap at 100
        return min(100, int(total_score))
    
    def _get_customer_risk_multiplier(self, risk_rating: Optional[str]) -> float:
        """Get risk multiplier based on customer risk rating."""
        multipliers = {
            'low': 0.9,
            'medium': 1.0,
            'high': 1.2,
            'critical': 1.5
        }
        return multipliers.get(risk_rating, 1.0)
    
    def _get_risk_level(self, score: int) -> str:
        """Convert score to risk level."""
        if score < 30:
            return 'low'
        elif score < 50:
            return 'medium'
        elif score < 75:
            return 'high'
        else:
            return 'critical'
    
    def _build_risk_factors(
        self,
        ml_score: int,
        alerts: List[AMLAlert]
    ) -> List[Dict[str, Any]]:
        """Build list of risk factors."""
        factors = []
        
        # ML factor
        factors.append({
            'type': 'ml_prediction',
            'name': 'ML Fraud Probability',
            'score': ml_score,
            'weight': self.ML_WEIGHT,
            'contribution': ml_score * self.ML_WEIGHT
        })
        
        # Compliance factors
        for alert in alerts:
            base_score = self.SEVERITY_SCORES.get(alert.severity, 10)
            multiplier = self.ALERT_MULTIPLIERS.get(alert.alert_type, 1.0)
            score = int(base_score * multiplier)
            
            factors.append({
                'type': 'compliance',
                'name': alert.rule_name,
                'alert_type': alert.alert_type,
                'severity': alert.severity.value,
                'score': score,
                'weight': self.COMPLIANCE_WEIGHT,
                'description': alert.description
            })
        
        return factors
    
    def _generate_recommendations(
        self,
        overall_score: int,
        ml_score: int,
        alerts: List[AMLAlert],
        transaction_amount: Optional[float]
    ) -> List[str]:
        """Generate action recommendations based on risk."""
        recommendations = []
        
        # Score-based recommendations
        if overall_score >= 75:
            recommendations.append("IMMEDIATE: Block transaction and escalate to compliance team")
            recommendations.append("Initiate enhanced due diligence review")
        elif overall_score >= 50:
            recommendations.append("REVIEW: Hold transaction for manual review")
            recommendations.append("Verify customer identity and transaction purpose")
        elif overall_score >= 30:
            recommendations.append("MONITOR: Increase monitoring frequency")
            recommendations.append("Consider requesting additional documentation")
        
        # ML-specific recommendations
        if ml_score >= 80:
            recommendations.append("ML model indicates high fraud probability - verify transaction details")
        elif ml_score >= 50:
            recommendations.append("ML model shows elevated risk - review transaction patterns")
        
        # Alert-specific recommendations
        alert_types = {alert.alert_type for alert in alerts}
        
        if 'aml_structuring' in alert_types:
            recommendations.append("Potential structuring detected - review transaction history")
        
        if 'aml_ctr_threshold' in alert_types:
            recommendations.append("CTR threshold exceeded - file regulatory report")
        
        if 'kyc_sanctions_match' in alert_types:
            recommendations.append("SANCTIONS MATCH - Freeze account and contact compliance immediately")
        
        if 'kyc_document_expired' in alert_types:
            recommendations.append("Request updated identification documents")
        
        # Amount-based recommendations
        if transaction_amount and transaction_amount > 50000:
            recommendations.append("Large transaction amount - verify source of funds")
        
        if not recommendations:
            recommendations.append("No specific action required - standard processing")
        
        return recommendations
    
    def compare_transactions(
        self,
        transactions: List[Dict[str, Any]]
    ) -> List[RiskScore]:
        """
        Calculate risk scores for multiple transactions.
        
        Args:
            transactions: List of transaction data dicts with 'ml_probability' and 'alerts'
            
        Returns:
            List of RiskScore objects
        """
        results = []
        
        for txn in transactions:
            score = self.calculate_risk(
                ml_probability=txn.get('ml_probability', 0),
                compliance_alerts=txn.get('alerts', []),
                transaction_amount=txn.get('amount'),
                customer_risk_rating=txn.get('customer_risk_rating')
            )
            results.append(score)
        
        return results


# Singleton
risk_scorer = RiskScorer()
