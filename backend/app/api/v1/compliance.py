"""
Compliance API endpoints.
"""
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user, require_analyst, require_admin
from app.core.compliance import aml_engine, kyc_checker, risk_scorer
from app.models.user import User
from app.models.compliance import ComplianceAlert, ComplianceRule, AlertStatus, AlertSeverity
from app.schemas.compliance import (
    ComplianceCheckRequest, ComplianceCheckResponse,
    ComplianceAlertResponse, AlertListResponse, AlertListRequest,
    AlertAssignmentRequest, AlertResolutionRequest,
    ComplianceRuleResponse, RiskScoreRequest, RiskScoreResponse,
    ComplianceDashboardResponse
)

router = APIRouter()


@router.post("/check", response_model=ComplianceCheckResponse)
async def check_compliance(
    request: ComplianceCheckRequest,
    current_user: User = Depends(require_analyst)
) -> Any:
    """
    Check transaction against compliance rules.
    """
    # Run AML checks
    alerts = aml_engine.evaluate(
        transaction=request.transaction,
        customer_history=request.customer_history,
        customer_info=request.customer_info
    )
    
    # Convert to response format
    alert_results = [
        {
            "rule_id": alert.rule_id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "description": alert.description,
            "alert_type": alert.alert_type,
            "triggered_value": alert.triggered_value,
            "threshold": alert.threshold
        }
        for alert in alerts
    ]
    
    # Calculate risk score
    compliance_score = risk_scorer._calculate_compliance_score(alerts)
    
    # Generate recommendations
    recommendations = []
    for alert in alerts:
        if alert.severity == AlertSeverity.CRITICAL:
            recommendations.append(f"CRITICAL: {alert.description}")
        elif alert.severity == AlertSeverity.HIGH:
            recommendations.append(f"HIGH: {alert.description}")
    
    if not recommendations:
        recommendations.append("No compliance issues detected")
    
    return ComplianceCheckResponse(
        is_compliant=len(alerts) == 0,
        alerts=alert_results,
        risk_score=compliance_score,
        recommendations=recommendations
    )


@router.post("/risk-score", response_model=RiskScoreResponse)
async def calculate_risk_score(
    request: RiskScoreRequest,
    current_user: User = Depends(require_analyst)
) -> Any:
    """
    Calculate composite risk score.
    """
    # Convert alert dicts to objects for risk scorer
    from app.core.compliance.aml_rules import AMLAlert, AlertSeverity
    
    alerts = []
    for alert_data in request.compliance_alerts:
        alerts.append(AMLAlert(
            rule_id=alert_data.get("rule_id", ""),
            rule_name=alert_data.get("rule_name", ""),
            severity=AlertSeverity(alert_data.get("severity", "low")),
            description=alert_data.get("description", ""),
            triggered_value=alert_data.get("triggered_value"),
            threshold=alert_data.get("threshold"),
            alert_type=alert_data.get("alert_type", "unknown")
        ))
    
    # Calculate risk
    result = risk_scorer.calculate_risk(
        ml_probability=request.ml_probability,
        compliance_alerts=alerts,
        transaction_amount=request.transaction_amount,
        customer_risk_rating=request.customer_risk_rating
    )
    
    return RiskScoreResponse(
        overall_score=result.overall_score,
        ml_score=result.ml_score,
        compliance_score=result.compliance_score,
        risk_level=result.risk_level,
        factors=result.factors,
        recommendations=result.recommendations
    )


# Alert management endpoints
@router.get("/alerts", response_model=AlertListResponse)
async def list_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    assigned_to: Optional[int] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    List compliance alerts with filters.
    """
    # Build query
    query = select(ComplianceAlert)
    
    if status:
        query = query.where(ComplianceAlert.status == status)
    if severity:
        query = query.where(ComplianceAlert.severity == severity)
    if alert_type:
        query = query.where(ComplianceAlert.alert_type == alert_type)
    if assigned_to:
        query = query.where(ComplianceAlert.assigned_to == assigned_to)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get alerts
    query = query.order_by(ComplianceAlert.created_at.desc())
    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    alerts = result.scalars().all()
    
    # Get summary stats
    status_counts = {}
    severity_counts = {}
    
    for alert in alerts:
        status_counts[alert.status.value] = status_counts.get(alert.status.value, 0) + 1
        severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
    
    return AlertListResponse(
        total=total,
        alerts=alerts,
        summary={
            "by_status": status_counts,
            "by_severity": severity_counts
        }
    )


@router.get("/alerts/{alert_id}", response_model=ComplianceAlertResponse)
async def get_alert(
    alert_id: str,
    current_user: User = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get a specific alert.
    """
    result = await db.execute(
        select(ComplianceAlert).where(ComplianceAlert.alert_id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    return alert


@router.post("/alerts/{alert_id}/assign")
async def assign_alert(
    alert_id: str,
    assignment: AlertAssignmentRequest,
    current_user: User = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Assign an alert to a user.
    """
    result = await db.execute(
        select(ComplianceAlert).where(ComplianceAlert.alert_id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    alert.assigned_to = assignment.assigned_to
    alert.status = AlertStatus.UNDER_INVESTIGATION
    
    await db.flush()
    
    return {"message": "Alert assigned successfully"}


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution: AlertResolutionRequest,
    current_user: User = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Resolve an alert.
    """
    result = await db.execute(
        select(ComplianceAlert).where(ComplianceAlert.alert_id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    from datetime import datetime
    
    alert.status = AlertStatus(resolution.status)
    alert.resolved_by = current_user.id
    alert.resolved_at = datetime.utcnow()
    alert.resolution_notes = resolution.resolution_notes
    
    await db.flush()
    
    return {"message": "Alert resolved successfully"}


# Rules management
@router.get("/rules", response_model=List[ComplianceRuleResponse])
async def list_rules(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    List compliance rules (admin only).
    """
    result = await db.execute(
        select(ComplianceRule).order_by(ComplianceRule.rule_id)
    )
    rules = result.scalars().all()
    
    return rules


@router.get("/dashboard", response_model=ComplianceDashboardResponse)
async def get_dashboard(
    current_user: User = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get compliance dashboard data.
    """
    # Count open alerts
    result = await db.execute(
        select(func.count()).where(ComplianceAlert.status == AlertStatus.OPEN)
    )
    open_alerts = result.scalar()
    
    # Count by severity
    severity_counts = {}
    for severity in AlertSeverity:
        result = await db.execute(
            select(func.count()).where(
                ComplianceAlert.severity == severity,
                ComplianceAlert.status == AlertStatus.OPEN
            )
        )
        severity_counts[severity.value] = result.scalar()
    
    # Count by type
    result = await db.execute(
        select(
            ComplianceAlert.alert_type,
            func.count()
        ).where(
            ComplianceAlert.status == AlertStatus.OPEN
        ).group_by(ComplianceAlert.alert_type)
    )
    type_counts = {row[0]: row[1] for row in result.all()}
    
    # Recent alerts
    result = await db.execute(
        select(ComplianceAlert).order_by(
            ComplianceAlert.created_at.desc()
        ).limit(10)
    )
    recent_alerts = result.scalars().all()
    
    # Resolution stats
    from datetime import datetime, timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    
    result = await db.execute(
        select(func.count()).where(
            ComplianceAlert.resolved_at >= week_ago
        )
    )
    resolved_this_week = result.scalar()
    
    return ComplianceDashboardResponse(
        open_alerts=open_alerts,
        alerts_by_severity=severity_counts,
        alerts_by_type=type_counts,
        recent_alerts=recent_alerts,
        resolution_stats={
            "resolved_this_week": resolved_this_week,
            "average_resolution_time_hours": 24  # Placeholder
        }
    )
