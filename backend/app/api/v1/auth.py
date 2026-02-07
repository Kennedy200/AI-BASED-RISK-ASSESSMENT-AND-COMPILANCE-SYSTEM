"""
Authentication API endpoints.
"""
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.core.security import (
    verify_password, hash_password, create_access_token,
    create_refresh_token, verify_token, check_password_strength
)
from app.models.user import User, Role, UserStatus, RefreshToken
from app.schemas.auth import (
    LoginRequest, LoginResponse, Token, UserCreate, UserResponse,
    UserUpdate, UserPasswordUpdate, PasswordResetRequest
)
from app.core.security.audit import get_audit_logger

router = APIRouter()
security = HTTPBearer()


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Authenticate user and return JWT tokens.
    """
    # Find user by username or email
    result = await db.execute(
        select(User).where(
            (User.username == login_data.username) | (User.email == login_data.username)
        )
    )
    user = result.scalar_one_or_none()
    
    audit_logger = get_audit_logger(db)
    client_ip = request.client.host if request.client else None
    
    # Check if user exists
    if not user:
        await audit_logger.log_login(
            user_id=0,
            username=login_data.username,
            ip_address=client_ip,
            success=False,
            error_message="User not found"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Check if account is locked
    if user.is_locked():
        await audit_logger.log_login(
            user_id=user.id,
            username=user.username,
            ip_address=client_ip,
            success=False,
            error_message="Account locked"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is locked. Please try again later."
        )
    
    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        # Increment failed attempts
        user.failed_login_attempts += 1
        
        # Lock account after 5 failed attempts
        if user.failed_login_attempts >= 5:
            user.locked_until = datetime.utcnow() + timedelta(minutes=30)
        
        await db.flush()
        
        await audit_logger.log_login(
            user_id=user.id,
            username=user.username,
            ip_address=client_ip,
            success=False,
            error_message="Invalid password"
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Check MFA if enabled
    if user.mfa_enabled:
        if not login_data.mfa_code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="MFA code required"
            )
        # TODO: Verify MFA code
    
    # Reset failed attempts
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login = datetime.utcnow()
    
    # Get user roles
    role_names = [role.name for role in user.roles]
    
    # Generate tokens
    access_token = create_access_token(
        user_id=user.id,
        username=user.username,
        roles=role_names
    )
    refresh_token = create_refresh_token(user.id, user.username)
    
    # Store refresh token
    refresh_token_record = RefreshToken(
        token=refresh_token,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(refresh_token_record)
    
    await db.flush()
    
    # Log successful login
    await audit_logger.log_login(
        user_id=user.id,
        username=user.username,
        ip_address=client_ip,
        success=True
    )
    
    return LoginResponse(
        user=UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            phone=user.phone,
            status=user.status.value,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            mfa_enabled=user.mfa_enabled,
            failed_login_attempts=user.failed_login_attempts,
            last_login=user.last_login,
            created_at=user.created_at,
            updated_at=user.updated_at,
            roles=role_names
        ),
        token=Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800  # 30 minutes
        )
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Refresh access token using refresh token.
    """
    # Verify refresh token
    token_data = verify_token(refresh_token, token_type="refresh")
    
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Check if token is in database and not revoked
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token == refresh_token,
            RefreshToken.revoked_at.is_(None)
        )
    )
    token_record = result.scalar_one_or_none()
    
    if not token_record or not token_record.is_valid():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Get user
    result = await db.execute(
        select(User).where(User.id == token_data.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Generate new tokens
    role_names = [role.name for role in user.roles]
    
    new_access_token = create_access_token(
        user_id=user.id,
        username=user.username,
        roles=role_names
    )
    new_refresh_token = create_refresh_token(user.id, user.username)
    
    # Revoke old refresh token
    token_record.revoked_at = datetime.utcnow()
    
    # Store new refresh token
    new_token_record = RefreshToken(
        token=new_refresh_token,
        user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=7)
    )
    db.add(new_token_record)
    
    await db.flush()
    
    return Token(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=1800
    )


@router.post("/logout")
async def logout(
    refresh_token: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Logout user and revoke refresh token.
    """
    # Revoke refresh token
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token == refresh_token)
    )
    token_record = result.scalar_one_or_none()
    
    if token_record:
        token_record.revoked_at = datetime.utcnow()
        await db.flush()
    
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get current user information.
    """
    role_names = [role.name for role in current_user.roles]
    
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        phone=current_user.phone,
        status=current_user.status.value,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        mfa_enabled=current_user.mfa_enabled,
        failed_login_attempts=current_user.failed_login_attempts,
        last_login=current_user.last_login,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        roles=role_names
    )


@router.put("/me", response_model=UserResponse)
async def update_me(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Update current user information.
    """
    # Update fields
    for field, value in user_update.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    
    await db.flush()
    
    role_names = [role.name for role in current_user.roles]
    
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        phone=current_user.phone,
        status=current_user.status.value,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        mfa_enabled=current_user.mfa_enabled,
        failed_login_attempts=current_user.failed_login_attempts,
        last_login=current_user.last_login,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        roles=role_names
    )


@router.post("/change-password")
async def change_password(
    password_data: UserPasswordUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Change user password.
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Check password strength
    strength = check_password_strength(password_data.new_password)
    if strength["strength"] == "weak":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Password too weak: {', '.join(strength['feedback'])}"
        )
    
    # Update password
    current_user.hashed_password = hash_password(password_data.new_password)
    await db.flush()
    
    return {"message": "Password changed successfully"}


@router.post("/password-reset-request")
async def request_password_reset(
    reset_request: PasswordResetRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Request password reset (sends email with reset link).
    """
    # Find user
    result = await db.execute(
        select(User).where(User.email == reset_request.email)
    )
    user = result.scalar_one_or_none()
    
    # Always return success to prevent email enumeration
    if not user:
        return {"message": "If the email exists, a reset link has been sent"}
    
    # TODO: Generate reset token and send email
    
    return {"message": "If the email exists, a reset link has been sent"}
