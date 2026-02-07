"""
AML (Anti-Money Laundering) Rule Engine.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AMLAlert:
    """AML Alert data class."""
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    description: str
    triggered_value: Any
    threshold: Any
    alert_type: str = "aml"
    metadata: Optional[Dict] = None


class AMLRuleEngine:
    """
    Anti-Money Laundering rule engine.
    Implements regulatory checks for suspicious activity.
    """
    
    # Regulatory thresholds
    CTR_THRESHOLD = 10000  # Currency Transaction Report (US) - $10,000
    STRUCTURING_WINDOW_HOURS = 24
    STRUCTURING_COUNT = 3  # Number of transactions
    STRUCTURING_THRESHOLD_PCT = 0.9  # 90% of CTR threshold
    VELOCITY_THRESHOLD = 5  # Transactions per hour
    HIGH_RISK_COUNTRIES = {'NG', 'RU', 'IR', 'KP', 'SY', 'AF', 'BY', 'MM', 'VE', 'CD'}
    
    # Business hours (for unusual hours check)
    BUSINESS_HOURS_START = 6   # 6 AM
    BUSINESS_HOURS_END = 23    # 11 PM
    
    def __init__(self):
        """Initialize AML rule engine."""
        self.rules = [
            self.check_ctr_threshold,
            self.check_structuring,
            self.check_velocity,
            self.check_high_risk_geography,
            self.check_unusual_hours,
            self.check_new_account_large_txn,
            self.check_rapid_succession,
            self.check_round_amount,
        ]
    
    def evaluate(
        self,
        transaction: Dict[str, Any],
        customer_history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> List[AMLAlert]:
        """
        Evaluate transaction against all AML rules.
        
        Args:
            transaction: Current transaction data
            customer_history: Previous transactions (for velocity/structuring)
            customer_info: Customer information (account age, etc.)
            
        Returns:
            List of triggered alerts (empty if compliant)
        """
        alerts = []
        
        for rule in self.rules:
            try:
                alert = rule(transaction, customer_history, customer_info)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                # Log error but continue with other rules
                print(f"Error in rule {rule.__name__}: {e}")
        
        return alerts
    
    def check_ctr_threshold(
        self,
        txn: Dict[str, Any],
        history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> Optional[AMLAlert]:
        """
        Rule AML-001: Transactions > $10,000 require reporting.
        Currency Transaction Report (CTR) threshold.
        """
        amount = txn.get('amount', 0)
        if amount > self.CTR_THRESHOLD:
            return AMLAlert(
                rule_id="AML-001",
                rule_name="CTR Threshold Exceeded",
                severity=AlertSeverity.HIGH,
                description=f"Transaction amount ${amount:,.2f} exceeds ${self.CTR_THRESHOLD:,} CTR threshold",
                triggered_value=amount,
                threshold=self.CTR_THRESHOLD,
                alert_type="aml_ctr_threshold"
            )
        return None
    
    def check_structuring(
        self,
        txn: Dict[str, Any],
        history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> Optional[AMLAlert]:
        """
        Rule AML-002: Potential structuring/smurfing.
        Multiple transactions just below CTR threshold to evade reporting.
        """
        if not history:
            return None
        
        amount = txn.get('amount', 0)
        structuring_min = self.CTR_THRESHOLD * self.STRUCTURING_THRESHOLD_PCT
        
        # Check if amount is just below threshold (90-100%)
        if amount < self.CTR_THRESHOLD and amount > structuring_min:
            # Count similar transactions in window
            txn_time = txn.get('timestamp', datetime.utcnow())
            if isinstance(txn_time, (int, float)):
                txn_time = datetime.fromtimestamp(txn_time)
            
            window_start = txn_time - timedelta(hours=self.STRUCTURING_WINDOW_HOURS)
            
            similar_count = 1  # Include current transaction
            for h in history:
                h_amount = h.get('amount', 0)
                h_time = h.get('timestamp')
                if isinstance(h_time, (int, float)):
                    h_time = datetime.fromtimestamp(h_time)
                
                if (structuring_min < h_amount < self.CTR_THRESHOLD and
                    h_time > window_start):
                    similar_count += 1
            
            if similar_count >= self.STRUCTURING_COUNT:
                return AMLAlert(
                    rule_id="AML-002",
                    rule_name="Potential Structuring",
                    severity=AlertSeverity.CRITICAL,
                    description=f"{similar_count} transactions between ${structuring_min:,.0f}-${self.CTR_THRESHOLD:,.0f} in {self.STRUCTURING_WINDOW_HOURS}h",
                    triggered_value=similar_count,
                    threshold=self.STRUCTURING_COUNT,
                    alert_type="aml_structuring"
                )
        return None
    
    def check_velocity(
        self,
        txn: Dict[str, Any],
        history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> Optional[AMLAlert]:
        """
        Rule AML-003: Unusual transaction frequency (velocity).
        """
        if not history:
            return None
        
        txn_time = txn.get('timestamp', datetime.utcnow())
        if isinstance(txn_time, (int, float)):
            txn_time = datetime.fromtimestamp(txn_time)
        
        # Count transactions in last hour
        one_hour_ago = txn_time - timedelta(hours=1)
        recent_count = 1  # Include current
        
        for h in history:
            h_time = h.get('timestamp')
            if isinstance(h_time, (int, float)):
                h_time = datetime.fromtimestamp(h_time)
            if h_time > one_hour_ago:
                recent_count += 1
        
        if recent_count > self.VELOCITY_THRESHOLD:
            return AMLAlert(
                rule_id="AML-003",
                rule_name="Velocity Check Failed",
                severity=AlertSeverity.MEDIUM,
                description=f"{recent_count} transactions in last hour",
                triggered_value=recent_count,
                threshold=self.VELOCITY_THRESHOLD,
                alert_type="aml_velocity"
            )
        return None
    
    def check_high_risk_geography(
        self,
        txn: Dict[str, Any],
        history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> Optional[AMLAlert]:
        """
        Rule AML-004: Transactions involving high-risk jurisdictions.
        """
        country = txn.get('country', '').upper()
        if not country and customer_info:
            country = customer_info.get('country', '').upper()
        
        if country in self.HIGH_RISK_COUNTRIES:
            return AMLAlert(
                rule_id="AML-004",
                rule_name="High-Risk Geography",
                severity=AlertSeverity.HIGH,
                description=f"Transaction involves high-risk jurisdiction: {country}",
                triggered_value=country,
                threshold=None,
                alert_type="aml_high_risk_geography"
            )
        return None
    
    def check_unusual_hours(
        self,
        txn: Dict[str, Any],
        history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> Optional[AMLAlert]:
        """
        Rule AML-005: Transactions outside normal business hours.
        """
        hour = txn.get('hour_of_day')
        if hour is None:
            # Try to extract from timestamp
            timestamp = txn.get('timestamp')
            if timestamp:
                if isinstance(timestamp, (int, float)):
                    hour = datetime.fromtimestamp(timestamp).hour
                else:
                    hour = timestamp.hour if hasattr(timestamp, 'hour') else None
        
        if hour is not None and (hour < self.BUSINESS_HOURS_START or hour > self.BUSINESS_HOURS_END):
            return AMLAlert(
                rule_id="AML-005",
                rule_name="Unusual Hours",
                severity=AlertSeverity.LOW,
                description=f"Transaction at {hour}:00 (outside business hours {self.BUSINESS_HOURS_START}:00-{self.BUSINESS_HOURS_END}:00)",
                triggered_value=hour,
                threshold=f"{self.BUSINESS_HOURS_START}-{self.BUSINESS_HOURS_END}",
                alert_type="aml_unusual_hours"
            )
        return None
    
    def check_new_account_large_txn(
        self,
        txn: Dict[str, Any],
        history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> Optional[AMLAlert]:
        """
        Rule AML-006: Large transaction from recently opened account.
        """
        if not customer_info:
            return None
        
        account_age_days = customer_info.get('account_age_days')
        if account_age_days is None:
            return None
        
        amount = txn.get('amount', 0)
        threshold_amount = 1000
        
        if account_age_days < 7 and amount > threshold_amount:
            return AMLAlert(
                rule_id="AML-006",
                rule_name="New Account Large Transaction",
                severity=AlertSeverity.MEDIUM,
                description=f"${amount:,.2f} transaction from {account_age_days}-day-old account",
                triggered_value=amount,
                threshold=threshold_amount,
                alert_type="aml_new_account_large_txn",
                metadata={'account_age_days': account_age_days}
            )
        return None
    
    def check_rapid_succession(
        self,
        txn: Dict[str, Any],
        history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> Optional[AMLAlert]:
        """
        Rule AML-007: Multiple transactions in rapid succession.
        """
        if not history or len(history) < 2:
            return None
        
        txn_time = txn.get('timestamp')
        if isinstance(txn_time, (int, float)):
            txn_time = datetime.fromtimestamp(txn_time)
        
        if not txn_time:
            return None
        
        # Check for transactions within 1 minute
        one_minute_ago = txn_time - timedelta(minutes=1)
        rapid_count = 1
        
        for h in history[-5:]:  # Check last 5 transactions
            h_time = h.get('timestamp')
            if isinstance(h_time, (int, float)):
                h_time = datetime.fromtimestamp(h_time)
            if h_time and h_time > one_minute_ago:
                rapid_count += 1
        
        if rapid_count >= 3:
            return AMLAlert(
                rule_id="AML-007",
                rule_name="Rapid Succession Transactions",
                severity=AlertSeverity.MEDIUM,
                description=f"{rapid_count} transactions within 1 minute",
                triggered_value=rapid_count,
                threshold=3,
                alert_type="aml_rapid_succession"
            )
        return None
    
    def check_round_amount(
        self,
        txn: Dict[str, Any],
        history: Optional[List[Dict]] = None,
        customer_info: Optional[Dict] = None
    ) -> Optional[AMLAlert]:
        """
        Rule AML-008: Round amount transactions (potential structuring indicator).
        """
        amount = txn.get('amount', 0)
        
        # Check if amount is round (ends in .00, 000, 0000)
        if amount >= 1000 and amount == int(amount):
            # Check if it's a round number (divisible by 100 or 1000)
            if amount % 1000 == 0 or amount % 500 == 0:
                return AMLAlert(
                    rule_id="AML-008",
                    rule_name="Round Amount Transaction",
                    severity=AlertSeverity.LOW,
                    description=f"Transaction amount ${amount:,.2f} is a round number",
                    triggered_value=amount,
                    threshold=None,
                    alert_type="aml_round_amount"
                )
        return None


# Singleton
aml_engine = AMLRuleEngine()
