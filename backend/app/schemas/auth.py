"""
Authentication and user schemas.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# Token schemas
class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    """Token payload."""
    sub: Optional[int] = None
    username: Optional[str] = None
    type: str = "access"


# User schemas
class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8)
    role_names: Optional[List[str]] = []


class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


class UserPasswordUpdate(BaseModel):
    """Password update schema."""
    current_password: str
    new_password: str = Field(..., min_length=8)


class UserInDB(UserBase):
    """User as stored in database."""
    id: int
    status: str
    is_active: bool
    is_superuser: bool
    mfa_enabled: bool
    failed_login_attempts: int
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserResponse(UserInDB):
    """User response schema."""
    roles: List[str] = []


# Login schemas
class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str
    mfa_code: Optional[str] = None


class LoginResponse(BaseModel):
    """Login response."""
    user: UserResponse
    token: Token


# MFA schemas
class MFASetupResponse(BaseModel):
    """MFA setup response."""
    secret: str
    qr_code_uri: str
    backup_codes: List[str]


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""
    code: str


# Password reset schemas
class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8)


# Role schemas
class RoleBase(BaseModel):
    """Base role schema."""
    name: str
    description: Optional[str] = None


class RoleCreate(RoleBase):
    """Role creation schema."""
    permissions: List[str] = []


class RoleUpdate(BaseModel):
    """Role update schema."""
    description: Optional[str] = None
    permissions: Optional[List[str]] = None


class RoleResponse(RoleBase):
    """Role response schema."""
    id: int
    permissions: List[str] = []
    created_at: datetime
    
    class Config:
        from_attributes = True
