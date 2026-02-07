"""
Analysis and prediction schemas.
"""
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# Single transaction analysis
class TransactionFeatures(BaseModel):
    """Transaction features for analysis."""
    Time: float = Field(..., description="Seconds since first transaction")
    Amount: float = Field(..., description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class SingleAnalysisRequest(BaseModel):
    """Single transaction analysis request."""
    features: TransactionFeatures
    model_name: Optional[str] = None
    include_explanation: bool = True
    include_compliance_check: bool = True


class FeatureContribution(BaseModel):
    """Feature contribution to prediction."""
    feature: str
    contribution: float
    direction: str  # increases_risk or decreases_risk
    value: float


class ExplanationResult(BaseModel):
    """Model explanation result."""
    top_features: List[FeatureContribution]
    base_value: float
    prediction_probability: float
    method: str


class PredictionResult(BaseModel):
    """Single prediction result."""
    fraud_probability: float
    risk_score: int  # 0-100
    risk_level: str  # Low, Medium, High
    prediction: int  # 0 or 1
    model_used: str
    model_confidence: float
    threshold_used: float
    processing_time_ms: int


class SingleAnalysisResponse(BaseModel):
    """Single analysis response."""
    prediction: PredictionResult
    explanation: Optional[ExplanationResult] = None
    compliance_alerts: Optional[List[Dict]] = None
    composite_risk: Optional[Dict] = None


# Batch analysis
class BatchAnalysisRequest(BaseModel):
    """Batch analysis request."""
    file_id: str
    model_name: Optional[str] = None
    callback_url: Optional[str] = None


class BatchJobStatus(BaseModel):
    """Batch job status."""
    job_id: str
    status: str  # pending, processing, completed, failed
    total_records: int
    processed_records: int
    failed_records: int
    progress_percentage: float
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class BatchAnalysisResult(BaseModel):
    """Batch analysis result item."""
    row_index: int
    prediction: PredictionResult
    risk_level: str


class BatchAnalysisResponse(BaseModel):
    """Batch analysis response."""
    job: BatchJobStatus
    results: Optional[List[BatchAnalysisResult]] = None
    summary: Optional[Dict[str, Any]] = None


# Model comparison
class ModelInfo(BaseModel):
    """Model information."""
    name: str
    loaded: bool
    metrics: Optional[Dict] = None


class ModelComparisonRequest(BaseModel):
    """Model comparison request."""
    features: TransactionFeatures


class ModelComparisonResponse(BaseModel):
    """Model comparison response."""
    predictions: Dict[str, PredictionResult]
    ensemble_prediction: Optional[PredictionResult] = None


# Model management
class ModelListResponse(BaseModel):
    """List of available models."""
    available_models: List[str]
    default_model: str
    feature_count: int
    models: Dict[str, ModelInfo]


class ModelMetricsResponse(BaseModel):
    """Model metrics response."""
    model_name: str
    metrics: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
