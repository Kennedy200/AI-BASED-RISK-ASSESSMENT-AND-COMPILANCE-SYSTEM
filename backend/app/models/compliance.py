"""
Compliance Alert and Rule models for AML/KYC.
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AlertSeverity(str, PyEnum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, PyEnum):
    """Alert review status."""
    OPEN = "open"
    UNDER_INVESTIGATION = "under_investigation"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"


class AlertType(str, PyEnum):
    """Types of compliance alerts."""
    AML_CTR_THRESHOLD = "aml_ctr_threshold"
    AML_STRUCTURING = "aml_structuring"
    AML_VELOCITY = "aml_velocity"
    AML_HIGH_RISK_GEOGRAPHY = "aml_high_risk_geography"
    AML_UNUSUAL_HOURS = "aml_unusual_hours"
    AML_NEW_ACCOUNT_LARGE_TXN = "aml_new_account_large_txn"
    KYC_VERIFICATION_FAILED = "kyc_verification_failed"
    KYC_DOCUMENT_EXPIRED = "kyc_document_expired"
    KYC_SANCTIONS_MATCH = "kyc_sanctions_match"


class ComplianceAlert(Base):
    """Compliance alert for AML/KYC violations."""
    __tablename__ = "compliance_alerts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    alert_id: Mapped[str] = mapped_column(
        String(100), unique=True, index=True, nullable=False
    )
    
    # Alert details
    alert_type: Mapped[AlertType] = mapped_column(Enum(AlertType), nullable=False)
    severity: Mapped[AlertSeverity] = mapped_column(Enum(AlertSeverity), nullable=False)
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus), default=AlertStatus.OPEN
    )
    
    # Description
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Rule that triggered
    rule_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    rule_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Triggered values
    triggered_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Related transaction
    transaction_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("transactions.id", ondelete="SET NULL"), nullable=True
    )
    
    # Additional data
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Assignment
    assigned_to: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    
    # Resolution
    resolved_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    transaction: Mapped[Optional["Transaction"]] = relationship(
        "Transaction", back_populates="compliance_alerts"
    )


class ComplianceRule(Base):
    """Compliance rule definition."""
    __tablename__ = "compliance_rules"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    rule_id: Mapped[str] = mapped_column(
        String(50), unique=True, index=True, nullable=False
    )
    
    # Rule details
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    rule_type: Mapped[str] = mapped_column(String(50), nullable=False)  # aml, kyc
    
    # Configuration
    severity: Mapped[AlertSeverity] = mapped_column(Enum(AlertSeverity), nullable=False)
    threshold_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold_currency: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    
    # Rule logic (stored as JSON for flexibility)
    rule_config: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Statistics
    trigger_count: Mapped[int] = mapped_column(Integer, default=0)
    false_positive_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    created_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )


class SanctionsList(Base):
    """Sanctions list entries for KYC screening."""
    __tablename__ = "sanctions_list"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Entity information
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # individual, entity, vessel
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    aliases: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Comma-separated
    
    # Identifiers
    date_of_birth: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    place_of_birth: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    nationality: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Address
    addresses: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # List source
    list_source: Mapped[str] = mapped_column(String(100), nullable=False)  # OFAC, UN, EU, etc.
    list_program: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
