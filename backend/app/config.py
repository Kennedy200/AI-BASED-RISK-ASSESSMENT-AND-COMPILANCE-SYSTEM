"""
Application configuration using Pydantic Settings.
All sensitive values are loaded from environment variables.
"""
from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "AI Fraud Detection & Compliance API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # File Encryption
    FILE_ENCRYPTION_KEY: str = Field(..., env="FILE_ENCRYPTION_KEY")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_detection",
        env="DATABASE_URL"
    )
    DATABASE_URL_SYNC: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/fraud_detection",
        env="DATABASE_URL_SYNC"
    )
    
    # Redis (for caching/sessions)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    # ML Model Paths
    MODEL_PATH: str = Field(default="../models", env="MODEL_PATH")
    DATA_PATH: str = Field(default="../data", env="DATA_PATH")
    
    # Default ML Model
    DEFAULT_MODEL: str = Field(default="xgboost", env="DEFAULT_MODEL")
    
    # Audit Logging
    AUDIT_LOG_RETENTION_DAYS: int = 365
    
    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        """Ensure secret key is sufficiently long."""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        return v
    
    @validator("FILE_ENCRYPTION_KEY")
    def validate_encryption_key(cls, v):
        """Ensure encryption key is sufficiently long."""
        if len(v) < 32:
            raise ValueError("FILE_ENCRYPTION_KEY must be at least 32 characters")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()
