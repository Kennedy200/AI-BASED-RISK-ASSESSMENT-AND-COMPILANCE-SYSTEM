"""
Analysis API endpoints for fraud detection.
"""
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user, require_analyst
from app.core.ml import get_predictor, get_explainer
from app.core.compliance import aml_engine, risk_scorer
from app.core.data import transaction_validator
from app.models.user import User
from app.schemas.analysis import (
    SingleAnalysisRequest, SingleAnalysisResponse, PredictionResult,
    ExplanationResult, FeatureContribution, ModelComparisonResponse,
    ModelListResponse, ModelMetricsResponse
)

router = APIRouter()


@router.post("/single", response_model=SingleAnalysisResponse)
async def analyze_single(
    request: SingleAnalysisRequest,
    current_user: User = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Analyze a single transaction for fraud risk.
    """
    predictor = get_predictor()
    
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded. Please train models first."
        )
    
    # Convert features to dict
    features = request.features.dict()
    
    # Validate input
    is_valid, errors = transaction_validator.validate_single_transaction(features)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {', '.join(errors)}"
        )
    
    # Get prediction
    prediction_result = predictor.predict(
        features=features,
        model_name=request.model_name
    )
    
    # Build prediction response
    prediction = PredictionResult(**prediction_result)
    
    # Get explanation if requested
    explanation = None
    if request.include_explanation:
        explainer = get_explainer(predictor)
        exp_result = explainer.explain(features, request.model_name)
        
        if "error" not in exp_result:
            explanation = ExplanationResult(
                top_features=[
                    FeatureContribution(**f) for f in exp_result.get("top_features", [])
                ],
                base_value=exp_result.get("base_value", 0),
                prediction_probability=exp_result.get("prediction_probability", 0),
                method=exp_result.get("method", "unknown")
            )
    
    # Compliance check if requested
    compliance_alerts = None
    composite_risk = None
    
    if request.include_compliance_check:
        # Build transaction data for compliance
        txn_data = {
            "amount": features.get("Amount"),
            "timestamp": features.get("Time"),
            "hour_of_day": (features.get("Time", 0) / 3600) % 24 if features.get("Time") else None,
        }
        
        alerts = aml_engine.evaluate(txn_data)
        
        compliance_alerts = [
            {
                "rule_id": alert.rule_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "description": alert.description,
                "alert_type": alert.alert_type
            }
            for alert in alerts
        ]
        
        # Calculate composite risk
        risk_result = risk_scorer.calculate_risk(
            ml_probability=prediction.fraud_probability,
            compliance_alerts=alerts,
            transaction_amount=features.get("Amount")
        )
        
        composite_risk = {
            "overall_score": risk_result.overall_score,
            "ml_score": risk_result.ml_score,
            "compliance_score": risk_result.compliance_score,
            "risk_level": risk_result.risk_level,
            "recommendations": risk_result.recommendations
        }
    
    return SingleAnalysisResponse(
        prediction=prediction,
        explanation=explanation,
        compliance_alerts=compliance_alerts,
        composite_risk=composite_risk
    )


@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    request: SingleAnalysisRequest,
    current_user: User = Depends(require_analyst)
) -> Any:
    """
    Get predictions from all models for comparison.
    """
    predictor = get_predictor()
    
    if not predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not loaded"
        )
    
    features = request.features.dict()
    
    # Get comparison from all models
    comparisons = predictor.compare_models(features)
    
    # Build response
    predictions = {}
    for name, result in comparisons.items():
        if name != "ensemble":
            predictions[name] = PredictionResult(**result)
    
    ensemble = comparisons.get("ensemble", {})
    ensemble_prediction = None
    if ensemble:
        ensemble_prediction = PredictionResult(
            fraud_probability=ensemble.get("fraud_probability", 0),
            risk_score=ensemble.get("risk_score", 0),
            risk_level=ensemble.get("risk_level", "Low"),
            prediction=ensemble.get("prediction", 0),
            model_used="ensemble",
            model_confidence=0,
            threshold_used=0.5,
            processing_time_ms=0
        )
    
    return ModelComparisonResponse(
        predictions=predictions,
        ensemble_prediction=ensemble_prediction
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    List available ML models and their info.
    """
    predictor = get_predictor()
    
    info = predictor.get_model_info()
    
    return ModelListResponse(
        available_models=info["available_models"],
        default_model=info["default_model"],
        feature_count=info["feature_count"],
        models=info["models"]
    )


@router.get("/models/{model_name}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_name: str,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get metrics for a specific model.
    """
    predictor = get_predictor()
    
    if model_name not in predictor.models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_name} not found"
        )
    
    metrics = predictor.metrics.get(model_name, {})
    feature_importance = predictor.get_feature_importance(model_name)
    
    return ModelMetricsResponse(
        model_name=model_name,
        metrics=metrics,
        feature_importance=feature_importance
    )


@router.get("/models/{model_name}/importance")
async def get_feature_importance(
    model_name: str,
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get feature importance for a model.
    """
    predictor = get_predictor()
    
    importance = predictor.get_feature_importance(model_name)
    
    return {
        "model_name": model_name or predictor.default_model,
        "feature_importance": importance
    }
