"""API v1 routes."""
from fastapi import APIRouter

from app.api.v1 import auth, analysis, compliance, upload

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(compliance.router, prefix="/compliance", tags=["Compliance"])
api_router.include_router(upload.router, prefix="/upload", tags=["Upload"])
