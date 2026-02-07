"""
Audit logging for security and compliance.
"""
from datetime import datetime
from typing import Optional, Dict, Any
import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog, AuditAction, SecurityEvent


class AuditLogger:
    """Audit logger for system actions."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def log(
        self,
        action: AuditAction,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_method: Optional[str] = None,
        request_path: Optional[str] = None,
        previous_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Create an audit log entry.
        
        Args:
            action: The action performed
            user_id: User ID who performed the action
            username: Username who performed the action
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            ip_address: Client IP address
            user_agent: Client user agent
            request_method: HTTP method
            request_path: Request path
            previous_values: Previous values (for updates)
            new_values: New values (for updates)
            success: Whether action succeeded
            error_message: Error message if failed
            metadata: Additional metadata
            
        Returns:
            Created audit log entry
        """
        log_entry = AuditLog(
            action=action,
            user_id=user_id,
            username=username,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_method=request_method,
            request_path=request_path,
            previous_values=previous_values,
            new_values=new_values,
            success=success,
            error_message=error_message,
            metadata=metadata,
            created_at=datetime.utcnow()
        )
        
        self.db.add(log_entry)
        await self.db.flush()
        
        return log_entry
    
    async def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            severity: low, medium, high, critical
            description: Event description
            user_id: Related user ID
            username: Related username
            ip_address: Source IP
            event_data: Additional event data
            
        Returns:
            Created security event
        """
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            event_data=event_data,
            created_at=datetime.utcnow()
        )
        
        self.db.add(event)
        await self.db.flush()
        
        return event
    
    async def log_login(
        self,
        user_id: int,
        username: str,
        ip_address: str,
        success: bool,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """Log a login attempt."""
        action = AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED
        
        return await self.log(
            action=action,
            user_id=user_id if success else None,
            username=username,
            resource_type="user",
            resource_id=str(user_id) if user_id else None,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message
        )
    
    async def log_transaction_action(
        self,
        action: AuditAction,
        user_id: int,
        username: str,
        transaction_id: str,
        ip_address: Optional[str] = None,
        previous_values: Optional[Dict] = None,
        new_values: Optional[Dict] = None
    ) -> AuditLog:
        """Log a transaction-related action."""
        return await self.log(
            action=action,
            user_id=user_id,
            username=username,
            resource_type="transaction",
            resource_id=transaction_id,
            ip_address=ip_address,
            previous_values=previous_values,
            new_values=new_values
        )
    
    async def log_file_upload(
        self,
        user_id: int,
        username: str,
        filename: str,
        file_size: int,
        ip_address: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """Log a file upload."""
        return await self.log(
            action=AuditAction.FILE_UPLOADED,
            user_id=user_id,
            username=username,
            resource_type="file",
            resource_id=filename,
            ip_address=ip_address,
            success=success,
            error_message=error_message,
            metadata={
                "filename": filename,
                "file_size": file_size
            }
        )


def get_audit_logger(db: AsyncSession) -> AuditLogger:
    """Get audit logger instance."""
    return AuditLogger(db)
