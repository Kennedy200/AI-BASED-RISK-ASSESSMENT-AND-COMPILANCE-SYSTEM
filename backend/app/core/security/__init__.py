"""Security modules."""
from app.core.security.passwords import hash_password, verify_password, check_password_strength
from app.core.security.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_token,
    TokenData
)
from app.core.security.encryption import SecureFileManager, get_secure_file_manager
from app.core.security.permissions import Permission, has_permission, get_role_permissions
from app.core.security.audit import AuditLogger, get_audit_logger

__all__ = [
    # Passwords
    "hash_password",
    "verify_password",
    "check_password_strength",
    # Auth
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "verify_token",
    "TokenData",
    # Encryption
    "SecureFileManager",
    "get_secure_file_manager",
    # Permissions
    "Permission",
    "has_permission",
    "get_role_permissions",
    # Audit
    "AuditLogger",
    "get_audit_logger",
]
