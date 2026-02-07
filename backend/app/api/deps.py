"""
API dependencies for authentication and database.
"""
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.core.security.auth import verify_token
from app.models.user import User

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Authorization credentials
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = verify_token(credentials.credentials)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    from sqlalchemy import select
    result = await db.execute(
        select(User).where(User.id == token_data.user_id)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    if user.is_locked():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is locked"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current user from token
        
    Returns:
        User object
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


def require_permissions(*permissions: str):
    """
    Dependency factory for permission checking.
    
    Args:
        *permissions: Required permissions
        
    Returns:
        Dependency function
    """
    async def permission_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        user_permissions = set()
        for role in current_user.roles:
            user_permissions.update(role.permissions.split(",") if role.permissions else [])
        
        # Check if user has any of the required permissions
        if not any(perm in user_permissions for perm in permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return current_user
    
    return permission_checker


def require_role(role_name: str):
    """
    Dependency factory for role checking.
    
    Args:
        role_name: Required role name
        
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:
        if not any(role.name == role_name for role in current_user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {role_name}"
            )
        return current_user
    
    return role_checker


class RoleChecker:
    """Role-based access checker."""
    
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles
    
    async def __call__(self, user: User = Depends(get_current_user)) -> User:
        if user.is_superuser:
            return user
        
        user_roles = [role.name for role in user.roles]
        if not any(role in self.allowed_roles for role in user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted"
            )
        return user


# Common role checkers
require_admin = RoleChecker(["super_admin", "admin"])
require_analyst = RoleChecker(["super_admin", "admin", "analyst"])
require_auditor = RoleChecker(["super_admin", "admin", "auditor"])
