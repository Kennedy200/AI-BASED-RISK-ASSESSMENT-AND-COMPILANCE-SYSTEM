"""
File upload schemas.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class FileUploadResponse(BaseModel):
    """File upload response."""
    file_id: str
    filename: str
    file_size: int
    content_type: str
    uploaded_at: datetime
    message: str = "File uploaded successfully"


class ColumnMapping(BaseModel):
    """Column mapping for uploaded files."""
    source_column: str
    target_column: str
    data_type: Optional[str] = "auto"


class FilePreviewRequest(BaseModel):
    """File preview request."""
    file_id: str
    column_mapping: Optional[List[ColumnMapping]] = None
    max_rows: int = Field(default=100, le=1000)


class FilePreviewResponse(BaseModel):
    """File preview response."""
    file_id: str
    filename: str
    total_rows: int
    preview_rows: int
    columns: List[str]
    detected_columns: Dict[str, str]  # source -> detected target
    sample_data: List[Dict[str, Any]]
    validation_errors: List[str]
    validation_warnings: List[str]


class FileValidationResult(BaseModel):
    """File validation result."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    total_rows: int
    valid_rows: int
    invalid_rows: int


class BatchUploadRequest(BaseModel):
    """Batch upload processing request."""
    file_id: str
    column_mapping: List[ColumnMapping]
    model_name: Optional[str] = None
    process_async: bool = True


class BatchUploadResponse(BaseModel):
    """Batch upload processing response."""
    job_id: str
    file_id: str
    status: str
    message: str
    total_records: int
    estimated_completion: Optional[str] = None


class UploadProgress(BaseModel):
    """Upload progress status."""
    file_id: str
    status: str  # uploaded, validating, processing, completed, failed
    progress_percentage: float
    records_processed: int
    records_total: int
    current_step: str
    error_message: Optional[str] = None
