"""
Role and permission definitions for RBAC.
"""
from enum import Enum
from typing import List, Set


class Permission(str, Enum):
    """System permissions."""
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_MANAGE_ROLES = "user:manage_roles"
    
    # Transaction analysis
    TRANSACTION_ANALYZE = "transaction:analyze"
    TRANSACTION_READ = "transaction:read"
    TRANSACTION_REVIEW = "transaction:review"
    TRANSACTION_APPROVE = "transaction:approve"
    TRANSACTION_REJECT = "transaction:reject"
    
    # Batch processing
    BATCH_UPLOAD = "batch:upload"
    BATCH_READ = "batch:read"
    BATCH_PROCESS = "batch:process"
    
    # Compliance
    COMPLIANCE_ALERT_READ = "compliance:alert_read"
    COMPLIANCE_ALERT_ASSIGN = "compliance:alert_assign"
    COMPLIANCE_ALERT_RESOLVE = "compliance:alert_resolve"
    COMPLIANCE_RULE_MANAGE = "compliance:rule_manage"
    
    # Model management
    MODEL_READ = "model:read"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    
    # Audit
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"
    
    # System
    SYSTEM_SETTINGS = "system:settings"
    SYSTEM_MAINTENANCE = "system:maintenance"


# Role to permission mappings
ROLE_PERMISSIONS: dict[str, List[Permission]] = {
    "super_admin": list(Permission),  # All permissions
    
    "admin": [
        Permission.USER_CREATE,
        Permission.USER_READ,
        Permission.USER_UPDATE,
        Permission.USER_MANAGE_ROLES,
        Permission.TRANSACTION_ANALYZE,
        Permission.TRANSACTION_READ,
        Permission.TRANSACTION_REVIEW,
        Permission.TRANSACTION_APPROVE,
        Permission.TRANSACTION_REJECT,
        Permission.BATCH_UPLOAD,
        Permission.BATCH_READ,
        Permission.BATCH_PROCESS,
        Permission.COMPLIANCE_ALERT_READ,
        Permission.COMPLIANCE_ALERT_ASSIGN,
        Permission.COMPLIANCE_ALERT_RESOLVE,
        Permission.COMPLIANCE_RULE_MANAGE,
        Permission.MODEL_READ,
        Permission.MODEL_TRAIN,
        Permission.MODEL_DEPLOY,
        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,
        Permission.SYSTEM_SETTINGS,
    ],
    
    "analyst": [
        Permission.TRANSACTION_ANALYZE,
        Permission.TRANSACTION_READ,
        Permission.TRANSACTION_REVIEW,
        Permission.TRANSACTION_APPROVE,
        Permission.TRANSACTION_REJECT,
        Permission.BATCH_UPLOAD,
        Permission.BATCH_READ,
        Permission.COMPLIANCE_ALERT_READ,
        Permission.COMPLIANCE_ALERT_RESOLVE,
        Permission.MODEL_READ,
    ],
    
    "auditor": [
        Permission.TRANSACTION_READ,
        Permission.BATCH_READ,
        Permission.COMPLIANCE_ALERT_READ,
        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,
    ],
    
    "api_client": [
        Permission.TRANSACTION_ANALYZE,
        Permission.TRANSACTION_READ,
        Permission.BATCH_UPLOAD,
        Permission.BATCH_READ,
    ],
}


def get_role_permissions(role_name: str) -> Set[Permission]:
    """
    Get permissions for a role.
    
    Args:
        role_name: Name of the role
        
    Returns:
        Set of permissions
    """
    return set(ROLE_PERMISSIONS.get(role_name, []))


def has_permission(user_roles: List[str], permission: Permission) -> bool:
    """
    Check if any of the user's roles has a permission.
    
    Args:
        user_roles: List of role names
        permission: Permission to check
        
    Returns:
        True if user has permission
    """
    for role in user_roles:
        if role == "super_admin":
            return True
        if permission in ROLE_PERMISSIONS.get(role, []):
            return True
    return False


def get_all_permissions(user_roles: List[str]) -> Set[Permission]:
    """
    Get all permissions for a list of roles.
    
    Args:
        user_roles: List of role names
        
    Returns:
        Set of all permissions
    """
    permissions = set()
    for role in user_roles:
        permissions.update(get_role_permissions(role))
    return permissions
