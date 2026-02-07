"""
JWT token creation and validation.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from jose import JWTError, jwt
from pydantic import BaseModel

from app.config import settings


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[int] = None
    username: Optional[str] = None
    roles: list = []
    permissions: list = []
    token_type: str = "access"


def create_access_token(
    user_id: int,
    username: str,
    roles: list = None,
    permissions: list = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User ID
        username: Username
        roles: List of user roles
        permissions: List of user permissions
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "sub": str(user_id),
        "username": username,
        "roles": roles or [],
        "permissions": permissions or [],
        "type": "access",
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": f"{user_id}_{datetime.utcnow().timestamp()}"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(user_id: int, username: str) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        user_id: User ID
        username: Username
        
    Returns:
        JWT refresh token string
    """
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode = {
        "sub": str(user_id),
        "username": username,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": f"refresh_{user_id}_{datetime.utcnow().timestamp()}"
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None


def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
    """
    Verify a JWT token and return token data.
    
    Args:
        token: JWT token string
        token_type: Expected token type (access or refresh)
        
    Returns:
        TokenData if valid, None otherwise
    """
    payload = decode_token(token)
    
    if payload is None:
        return None
    
    # Check token type
    if payload.get("type") != token_type:
        return None
    
    # Check expiration
    exp = payload.get("exp")
    if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
        return None
    
    user_id = payload.get("sub")
    if user_id is None:
        return None
    
    return TokenData(
        user_id=int(user_id),
        username=payload.get("username"),
        roles=payload.get("roles", []),
        permissions=payload.get("permissions", []),
        token_type=payload.get("type", "access")
    )


def get_token_expiry(token: str) -> Optional[datetime]:
    """
    Get token expiration time.
    
    Args:
        token: JWT token string
        
    Returns:
        Expiration datetime or None
    """
    payload = decode_token(token)
    if payload and "exp" in payload:
        return datetime.fromtimestamp(payload["exp"])
    return None
