"""
File upload API endpoints.
"""
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, require_analyst
from app.core.security import get_secure_file_manager
from app.core.data.validators import transaction_validator
from app.schemas.upload import (
    FileUploadResponse, FilePreviewResponse, BatchUploadRequest, BatchUploadResponse
)

router = APIRouter()


@router.post("/file", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Upload a file for batch processing.
    Supports CSV and Excel files.
    """
    # Validate file extension
    if not transaction_validator.validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Supported: CSV, XLSX, XLS"
        )
    
    # Sanitize filename
    filename = transaction_validator.sanitize_filename(file.filename)
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Validate file size (max 100MB)
    max_size = 100 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: 100MB"
        )
    
    # Encrypt and store file
    secure_manager = get_secure_file_manager()
    file_id = secure_manager.encrypt_store(content)
    
    # Log upload
    from app.core.security.audit import get_audit_logger
    audit_logger = get_audit_logger(db)
    await audit_logger.log_file_upload(
        user_id=current_user.id,
        username=current_user.username,
        filename=filename,
        file_size=file_size
    )
    
    return FileUploadResponse(
        file_id=file_id,
        filename=filename,
        file_size=file_size,
        content_type=file.content_type or "application/octet-stream",
        uploaded_at=secure_manager.temp_dir.stat().st_mtime
    )


@router.get("/file/{file_id}/preview", response_model=FilePreviewResponse)
async def preview_file(
    file_id: str,
    max_rows: int = 100,
    current_user = Depends(require_analyst)
) -> Any:
    """
    Preview uploaded file contents.
    """
    import pandas as pd
    
    secure_manager = get_secure_file_manager()
    
    if not secure_manager.exists(file_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or expired"
        )
    
    try:
        # Decrypt and load file
        content = secure_manager.decrypt_retrieve(file_id)
        
        # Try to read as CSV first, then Excel
        try:
            df = pd.read_csv(pd.io.common.BytesIO(content), nrows=max_rows * 2)
        except Exception:
            try:
                df = pd.read_excel(pd.io.common.BytesIO(content), nrows=max_rows * 2)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not parse file: {str(e)}"
                )
        
        # Validate columns
        is_valid, errors, warnings = transaction_validator.validate_dataframe(df)
        
        # Detect column mapping
        detected_columns = {}
        required_cols = transaction_validator.REQUIRED_COLUMNS
        
        for req_col in required_cols:
            if req_col in df.columns:
                detected_columns[req_col] = req_col
            else:
                # Try to find similar column
                for col in df.columns:
                    if req_col.lower() in col.lower() or col.lower() in req_col.lower():
                        detected_columns[req_col] = col
                        break
        
        # Get sample data
        preview_df = df.head(min(max_rows, len(df)))
        sample_data = preview_df.to_dict('records')
        
        return FilePreviewResponse(
            file_id=file_id,
            filename="uploaded_file",  # We don't store the original filename securely
            total_rows=len(df),
            preview_rows=len(preview_df),
            columns=list(df.columns),
            detected_columns=detected_columns,
            sample_data=sample_data,
            validation_errors=errors,
            validation_warnings=warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@router.post("/file/{file_id}/process", response_model=BatchUploadResponse)
async def process_file(
    file_id: str,
    request: BatchUploadRequest,
    current_user = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Process uploaded file for batch analysis.
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    secure_manager = get_secure_file_manager()
    
    if not secure_manager.exists(file_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or expired"
        )
    
    try:
        # Decrypt and load file
        content = secure_manager.decrypt_retrieve(file_id)
        
        # Read file
        try:
            df = pd.read_csv(pd.io.common.BytesIO(content))
        except Exception:
            df = pd.read_excel(pd.io.common.BytesIO(content))
        
        # Apply column mapping
        column_mapping = {m.source_column: m.target_column for m in request.column_mapping}
        df = df.rename(columns=column_mapping)
        
        # Validate
        is_valid, errors, warnings = transaction_validator.validate_dataframe(df)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation failed: {errors}"
            )
        
        # Create batch job
        job_id = str(uuid.uuid4())
        
        from app.models.transaction import BatchJob
        
        batch_job = BatchJob(
            job_id=job_id,
            filename=request.file_id,  # Store file_id as reference
            total_records=len(df),
            status="pending",
            user_id=current_user.id
        )
        db.add(batch_job)
        await db.flush()
        
        # TODO: If async processing requested, queue background task
        # For now, process synchronously
        
        from app.core.ml import get_predictor
        predictor = get_predictor()
        
        if not predictor.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML models not loaded"
            )
        
        # Update status
        batch_job.status = "processing"
        batch_job.started_at = datetime.utcnow()
        
        # Process records
        results = []
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        failed = 0
        
        for idx, row in df.iterrows():
            try:
                features = row.to_dict()
                prediction = predictor.predict(features, request.model_name)
                
                risk_level = prediction['risk_level']
                if risk_level == 'High':
                    high_risk += 1
                elif risk_level == 'Medium':
                    medium_risk += 1
                else:
                    low_risk += 1
                
                results.append({
                    'row_index': idx,
                    'prediction': prediction
                })
                
            except Exception as e:
                failed += 1
                results.append({
                    'row_index': idx,
                    'error': str(e)
                })
        
        # Update job
        batch_job.status = "completed"
        batch_job.completed_at = datetime.utcnow()
        batch_job.processed_records = len(df) - failed
        batch_job.failed_records = failed
        batch_job.high_risk_count = high_risk
        batch_job.medium_risk_count = medium_risk
        batch_job.low_risk_count = low_risk
        
        await db.flush()
        
        # Clean up secure file
        secure_manager.secure_delete(file_id)
        
        return BatchUploadResponse(
            job_id=job_id,
            file_id=file_id,
            status="completed",
            message=f"Processed {len(df)} records",
            total_records=len(df),
            estimated_completion=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/jobs/{job_id}/status")
async def get_job_status(
    job_id: str,
    current_user = Depends(require_analyst),
    db: AsyncSession = Depends(get_db)
) -> Any:
    """
    Get batch job status.
    """
    from app.models.transaction import BatchJob
    
    result = await db.execute(
        select(BatchJob).where(BatchJob.job_id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "total_records": job.total_records,
        "processed_records": job.processed_records,
        "failed_records": job.failed_records,
        "high_risk_count": job.high_risk_count,
        "medium_risk_count": job.medium_risk_count,
        "low_risk_count": job.low_risk_count,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }
