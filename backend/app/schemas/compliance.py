"""
Compliance schemas.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


# Alert schemas
class ComplianceAlertBase(BaseModel):
    """Base compliance alert schema."""
    alert_type: str
    severity: str
    title: str
    description: str


class ComplianceAlertCreate(ComplianceAlertBase):
    """Alert creation schema."""
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    triggered_value: Optional[float] = None
    threshold: Optional[float] = None
    transaction_id: Optional[int] = None
    metadata: Optional[Dict] = None


class ComplianceAlertResponse(ComplianceAlertBase):
    """Alert response schema."""
    id: int
    alert_id: str
    status: str
    rule_id: Optional[str]
    rule_name: Optional[str]
    triggered_value: Optional[float]
    threshold: Optional[float]
    transaction_id: Optional[int]
    metadata: Optional[Dict]
    assigned_to: Optional[int]
    resolved_by: Optional[int]
    resolved_at: Optional[datetime]
    resolution_notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AlertAssignmentRequest(BaseModel):
    """Alert assignment request."""
    assigned_to: int


class AlertResolutionRequest(BaseModel):
    """Alert resolution request."""
    status: str  # resolved, false_positive, escalated
    resolution_notes: str


# Alert list/filter
class AlertListRequest(BaseModel):
    """Alert list filter request."""
    status: Optional[str] = None
    severity: Optional[str] = None
    alert_type: Optional[str] = None
    assigned_to: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 50
    offset: int = 0


class AlertListResponse(BaseModel):
    """Alert list response."""
    total: int
    alerts: List[ComplianceAlertResponse]
    summary: Dict[str, int]  # Count by status, severity


# Rule schemas
class ComplianceRuleBase(BaseModel):
    """Base compliance rule schema."""
    name: str
    description: str
    rule_type: str  # aml, kyc
    severity: str


class ComplianceRuleCreate(ComplianceRuleBase):
    """Rule creation schema."""
    rule_id: str
    threshold_value: Optional[float] = None
    threshold_currency: Optional[str] = None
    rule_config: Dict = {}


class ComplianceRuleUpdate(BaseModel):
    """Rule update schema."""
    name: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    is_active: Optional[bool] = None
    threshold_value: Optional[float] = None
    rule_config: Optional[Dict] = None


class ComplianceRuleResponse(ComplianceRuleBase):
    """Rule response schema."""
    id: int
    rule_id: str
    threshold_value: Optional[float]
    threshold_currency: Optional[str]
    rule_config: Dict
    is_active: bool
    trigger_count: int
    false_positive_count: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Compliance check
class ComplianceCheckRequest(BaseModel):
    """Compliance check request."""
    transaction: Dict[str, Any]
    customer_history: Optional[List[Dict]] = None
    customer_info: Optional[Dict] = None


class ComplianceAlertResult(BaseModel):
    """Compliance alert result."""
    rule_id: str
    rule_name: str
    severity: str
    description: str
    alert_type: str
    triggered_value: Optional[Any]
    threshold: Optional[Any]


class ComplianceCheckResponse(BaseModel):
    """Compliance check response."""
    is_compliant: bool
    alerts: List[ComplianceAlertResult]
    risk_score: int
    recommendations: List[str]


# Risk scoring
class RiskScoreRequest(BaseModel):
    """Risk score request."""
    ml_probability: float
    compliance_alerts: List[ComplianceAlertResult]
    transaction_amount: Optional[float] = None
    customer_risk_rating: Optional[str] = None


class RiskFactor(BaseModel):
    """Risk factor breakdown."""
    type: str
    name: str
    score: int
    weight: float
    contribution: Optional[float] = None
    description: Optional[str] = None


class RiskScoreResponse(BaseModel):
    """Risk score response."""
    overall_score: int
    ml_score: int
    compliance_score: int
    risk_level: str
    factors: List[RiskFactor]
    recommendations: List[str]


# Dashboard
class ComplianceDashboardResponse(BaseModel):
    """Compliance dashboard data."""
    open_alerts: int
    alerts_by_severity: Dict[str, int]
    alerts_by_type: Dict[str, int]
    recent_alerts: List[ComplianceAlertResponse]
    resolution_stats: Dict[str, Any]
