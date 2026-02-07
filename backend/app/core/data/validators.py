"""
Input validation for transaction data.
"""
from typing import Dict, Any, List, Optional, Tuple
import re

import pandas as pd
import numpy as np


class TransactionValidator:
    """Validator for transaction data."""
    
    # Expected columns for creditcard.csv format
    REQUIRED_COLUMNS = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    OPTIONAL_COLUMNS = ['Class']
    
    # Validation rules
    AMOUNT_MIN = 0.0
    AMOUNT_MAX = 1000000.0  # $1M max
    TIME_MIN = 0
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_dataframe(
        self, 
        df: pd.DataFrame,
        require_class: bool = False
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a transaction DataFrame.
        
        Args:
            df: DataFrame to validate
            require_class: Whether Class column is required
            
        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Check for required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
        
        # Check for Class column if required
        if require_class and 'Class' not in df.columns:
            self.errors.append("Class column required for training data")
        
        if self.errors:
            return False, self.errors, self.warnings
        
        # Validate data types and ranges
        self._validate_amount(df)
        self._validate_time(df)
        self._validate_v_features(df)
        
        if require_class:
            self._validate_class(df)
        
        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            cols_with_nulls = null_counts[null_counts > 0].to_dict()
            self.warnings.append(f"Columns with missing values: {cols_with_nulls}")
        
        # Check for duplicates
        if df.duplicated().any():
            dup_count = df.duplicated().sum()
            self.warnings.append(f"Found {dup_count} duplicate rows")
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_amount(self, df: pd.DataFrame):
        """Validate Amount column."""
        if 'Amount' not in df.columns:
            return
        
        # Check for negative amounts
        negative = (df['Amount'] < 0).sum()
        if negative > 0:
            self.errors.append(f"Found {negative} negative amounts")
        
        # Check for zero amounts
        zero = (df['Amount'] == 0).sum()
        if zero > 0:
            self.warnings.append(f"Found {zero} zero-amount transactions")
        
        # Check for extremely large amounts
        large = (df['Amount'] > self.AMOUNT_MAX).sum()
        if large > 0:
            self.warnings.append(f"Found {large} transactions over ${self.AMOUNT_MAX:,.0f}")
    
    def _validate_time(self, df: pd.DataFrame):
        """Validate Time column."""
        if 'Time' not in df.columns:
            return
        
        # Check for negative time
        negative = (df['Time'] < 0).sum()
        if negative > 0:
            self.errors.append(f"Found {negative} negative time values")
    
    def _validate_v_features(self, df: pd.DataFrame):
        """Validate V1-V28 PCA features."""
        for i in range(1, 29):
            col = f'V{i}'
            if col not in df.columns:
                continue
            
            # Check for non-numeric values
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.errors.append(f"Column {col} is not numeric")
                continue
            
            # Check for extreme outliers (PCA features should be normalized)
            outliers = (np.abs(df[col]) > 10).sum()
            if outliers > len(df) * 0.01:  # More than 1% are extreme
                self.warnings.append(f"Column {col} has {outliers} extreme values (|x| > 10)")
    
    def _validate_class(self, df: pd.DataFrame):
        """Validate Class column."""
        if 'Class' not in df.columns:
            return
        
        # Check for valid values
        invalid = (~df['Class'].isin([0, 1])).sum()
        if invalid > 0:
            self.errors.append(f"Found {invalid} invalid Class values (must be 0 or 1)")
    
    def validate_single_transaction(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single transaction dict.
        
        Args:
            data: Transaction data dictionary
            
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        # Check required fields
        for col in self.REQUIRED_COLUMNS:
            if col not in data:
                errors.append(f"Missing required field: {col}")
        
        if errors:
            return False, errors
        
        # Validate Amount
        amount = data.get('Amount')
        if amount is not None:
            if not isinstance(amount, (int, float)):
                errors.append("Amount must be numeric")
            elif amount < 0:
                errors.append("Amount cannot be negative")
            elif amount > self.AMOUNT_MAX:
                errors.append(f"Amount exceeds maximum (${self.AMOUNT_MAX:,.0f})")
        
        # Validate Time
        time_val = data.get('Time')
        if time_val is not None:
            if not isinstance(time_val, (int, float)):
                errors.append("Time must be numeric")
            elif time_val < 0:
                errors.append("Time cannot be negative")
        
        # Validate V features
        for i in range(1, 29):
            col = f'V{i}'
            val = data.get(col)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(f"{col} must be numeric")
        
        return len(errors) == 0, errors
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to prevent path traversal.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path components
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Remove null bytes
        filename = filename.replace('\x00', '')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
        
        return filename
    
    def validate_file_extension(self, filename: str, allowed: List[str] = None) -> bool:
        """
        Validate file extension.
        
        Args:
            filename: Filename to check
            allowed: List of allowed extensions (default: csv, xlsx, xls)
            
        Returns:
            True if valid
        """
        if allowed is None:
            allowed = ['.csv', '.xlsx', '.xls']
        
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        return f'.{ext}' in allowed or ext in [a.lstrip('.') for a in allowed]


# Singleton
transaction_validator = TransactionValidator()
