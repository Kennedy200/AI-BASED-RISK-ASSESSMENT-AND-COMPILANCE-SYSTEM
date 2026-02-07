"""
Audit logging models for compliance and security.
"""
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, Any

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class AuditAction(str, PyEnum):
    """Audit action types."""
    # Authentication
    LOGIN = "login"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    TOKEN_REFRESH = "token_refresh"
    
    # User Management
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_LOCKED = "user_locked"
    USER_UNLOCKED = "user_unlocked"
    
    # Transaction Analysis
    TRANSACTION_ANALYZED = "transaction_analyzed"
    TRANSACTION_REVIEWED = "transaction_reviewed"
    TRANSACTION_APPROVED = "transaction_approved"
    TRANSACTION_REJECTED = "transaction_rejected"
    BATCH_UPLOADED = "batch_uploaded"
    BATCH_PROCESSED = "batch_processed"
    
    # Compliance
    ALERT_CREATED = "alert_created"
    ALERT_ASSIGNED = "alert_assigned"
    ALERT_RESOLVED = "alert_resolved"
    ALERT_ESCALATED = "alert_escalated"
    
    # File Operations
    FILE_UPLOADED = "file_uploaded"
    FILE_DOWNLOADED = "file_downloaded"
    FILE_DELETED = "file_deleted"
    
    # System
    SETTINGS_CHANGED = "settings_changed"
    RULE_CREATED = "rule_created"
    RULE_UPDATED = "rule_updated"
    RULE_DELETED = "rule_deleted"
    MODEL_TRAINED = "model_trained"


class AuditLog(Base):
    """Audit log entry for all system actions."""
    __tablename__ = "audit_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Action details
    action: Mapped[AuditAction] = mapped_column(Enum(AuditAction), nullable=False, index=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # user, transaction, alert
    resource_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Actor
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    username: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Request details
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)  # IPv6 compatible
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_method: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    request_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Changes (for update operations)
    previous_values: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    new_values: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Result
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Additional context
    metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")
    
    def __repr__(self) -> str:
        return f"<AuditLog {self.action} by {self.username} at {self.created_at}>"


class SecurityEvent(Base):
    """Security events for threat detection."""
    __tablename__ = "security_events"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Event details
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # low, medium, high, critical
    
    # Description
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Actor
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    username: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Request details
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    
    # Event data
    event_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Status
    is_resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_by: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def resolve(self, user_id: int, notes: Optional[str] = None):
        """Mark security event as resolved."""
        self.is_resolved = True
        self.resolved_by = user_id
        self.resolved_at = datetime.utcnow()
        self.resolution_notes = notes
