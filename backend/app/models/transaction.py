"""
Transaction and Analysis Result models.
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, List, Dict, Any

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class TransactionStatus(str, PyEnum):
    """Transaction review status."""
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class RiskLevel(str, PyEnum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Transaction(Base):
    """Transaction model for storing transaction data and analysis."""
    __tablename__ = "transactions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(
        String(100), unique=True, index=True, nullable=False
    )
    
    # Transaction data (encrypted sensitive fields in production)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Original data (JSON for flexibility)
    raw_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Analysis results
    fraud_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 0-100
    risk_level: Mapped[Optional[RiskLevel]] = mapped_column(Enum(RiskLevel), nullable=True)
    model_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Status
    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus), default=TransactionStatus.PENDING
    )
    
    # Review
    reviewed_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    review_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Batch processing
    batch_id: Mapped[Optional[str]] = mapped_column(String(100), index=True, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    analyst: Mapped[Optional["User"]] = relationship("User", back_populates="transactions")
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        "AnalysisResult", back_populates="transaction", lazy="dynamic"
    )
    compliance_alerts: Mapped[List["ComplianceAlert"]] = relationship(
        "ComplianceAlert", back_populates="transaction", lazy="dynamic"
    )


class AnalysisResult(Base):
    """Detailed analysis result for a transaction."""
    __tablename__ = "analysis_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("transactions.id", ondelete="CASCADE"), nullable=False
    )
    
    # Model information
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), default="1.0")
    
    # Prediction
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 or 1
    threshold_used: Mapped[float] = mapped_column(Float, default=0.5)
    
    # Explainability (SHAP values)
    feature_importance: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    top_features: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    
    # Processing time
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    transaction: Mapped["Transaction"] = relationship("Transaction", back_populates="analysis_results")


class BatchJob(Base):
    """Batch processing job for multiple transactions."""
    __tablename__ = "batch_jobs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    job_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    
    # Job details
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    total_records: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_records: Mapped[int] = mapped_column(Integer, default=0)
    failed_records: Mapped[int] = mapped_column(Integer, default=0)
    
    # Status
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending, processing, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Results summary
    high_risk_count: Mapped[int] = mapped_column(Integer, default=0)
    medium_risk_count: Mapped[int] = mapped_column(Integer, default=0)
    low_risk_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # User
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    
    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
