"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.v1 import api_router
from app.db.base import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting up...")
    
    # Initialize database
    await init_db()
    print("Database initialized")
    
    # Load ML models
    try:
        from app.core.ml import get_predictor
        predictor = get_predictor()
        if predictor.is_ready():
            print(f"ML models loaded: {list(predictor.models.keys())}")
        else:
            print("WARNING: ML models not found. Run training first.")
    except Exception as e:
        print(f"WARNING: Could not load ML models: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down...")
    await close_db()
    print("Database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-Based Risk Assessment and Compliance Monitoring System for Fraud Detection",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.core.ml import get_predictor
    
    predictor = get_predictor()
    
    return {
        "status": "healthy",
        "database": "connected",
        "ml_models": {
            "loaded": predictor.is_ready(),
            "available": list(predictor.models.keys()) if predictor.is_ready() else []
        }
    }


@app.get("/api/v1")
async def api_info():
    """API information endpoint."""
    return {
        "version": "v1",
        "endpoints": {
            "auth": "/api/v1/auth",
            "analysis": "/api/v1/analysis",
            "compliance": "/api/v1/compliance",
            "upload": "/api/v1/upload"
        }
    }
