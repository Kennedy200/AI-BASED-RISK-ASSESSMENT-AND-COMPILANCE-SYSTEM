"""
Authentication tests.
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password, create_access_token
from app.models.user import User, UserStatus


@pytest.fixture
async def test_user(db: AsyncSession):
    """Create a test user."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=hash_password("testpassword123"),
        status=UserStatus.ACTIVE,
        is_active=True
    )
    db.add(user)
    await db.flush()
    return user


def test_password_hashing():
    """Test password hashing and verification."""
    password = "testpassword123"
    hashed = hash_password(password)
    
    assert verify_password(password, hashed)
    assert not verify_password("wrongpassword", hashed)


def test_token_creation():
    """Test JWT token creation."""
    token = create_access_token(
        user_id=1,
        username="testuser",
        roles=["analyst"]
    )
    
    assert token is not None
    assert isinstance(token, str)


@pytest.mark.asyncio
async def test_login(client: AsyncClient, test_user):
    """Test login endpoint."""
    response = await client.post("/api/v1/auth/login", json={
        "username": "testuser",
        "password": "testpassword123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
    assert "access_token" in data["token"]


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient):
    """Test login with invalid credentials."""
    response = await client.post("/api/v1/auth/login", json={
        "username": "nonexistent",
        "password": "wrongpassword"
    })
    
    assert response.status_code == 401
