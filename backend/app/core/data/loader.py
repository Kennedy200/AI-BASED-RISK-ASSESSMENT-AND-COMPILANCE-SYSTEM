"""
Data loading and preprocessing for the creditcard.csv dataset.
"""
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.config import settings


class CreditCardDataLoader:
    """
    Loads and preprocesses creditcard.csv for training and inference.
    """
    
    DATA_PATH = Path(settings.DATA_PATH) / "creditcard.csv"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Optional custom data path
        """
        if data_path:
            self.DATA_PATH = Path(data_path)
        self._df: Optional[pd.DataFrame] = None
        self._feature_names: Optional[List[str]] = None
    
    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load raw dataset with validation.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            DataFrame with creditcard data
            
        Raises:
            FileNotFoundError: If dataset not found
            ValueError: If dataset schema is invalid
        """
        if self._df is not None and not force_reload:
            return self._df.copy()
        
        if not self.DATA_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.DATA_PATH}. "
                "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )
        
        df = pd.read_csv(self.DATA_PATH)
        
        # Validate schema
        expected_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Validate target
        if not df['Class'].isin([0, 1]).all():
            raise ValueError("Invalid values in Class column")
        
        fraud_count = df['Class'].sum()
        if fraud_count == 0:
            raise ValueError("No fraud cases found (should be 492)")
        
        print(f"Loaded {len(df)} transactions ({fraud_count} fraud, {len(df) - fraud_count} normal)")
        print(f"Fraud rate: {fraud_count / len(df) * 100:.3f}%")
        
        self._df = df
        return df.copy()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from raw data.
        
        Args:
            df: Raw dataframe
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Time-based features
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / (3600 * 24)).astype(int)
        
        # Amount features
        df['Amount_log'] = np.log1p(df['Amount'])
        
        # Amount binning (using quantiles)
        try:
            df['Amount_bin'] = pd.qcut(
                df['Amount'], 
                q=5, 
                labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                duplicates='drop'
            )
        except ValueError:
            # If too many duplicates, use fewer bins
            df['Amount_bin'] = pd.qcut(
                df['Amount'], 
                q=3, 
                labels=['low', 'medium', 'high'],
                duplicates='drop'
            )
        
        # Velocity features (time difference from previous transaction)
        df['Time_diff'] = df['Time'].diff().fillna(0)
        
        # Interaction features (top correlated features based on domain knowledge)
        df['V1_V2_interaction'] = df['V1'] * df['V2']
        df['V3_V4_interaction'] = df['V3'] * df['V4']
        
        # Amount interactions
        df['Amount_V1_interaction'] = df['Amount'] * df['V1']
        df['Amount_V3_interaction'] = df['Amount'] * df['V3']
        
        return df
    
    def prepare_ml(
        self, 
        df: Optional[pd.DataFrame] = None,
        include_engineered: bool = True,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features and target for machine learning.
        
        Args:
            df: DataFrame to use (loads if None)
            include_engineered: Whether to include engineered features
            test_size: Test set proportion (default from class)
            random_state: Random seed (default from class)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if df is None:
            df = self.load()
        
        if include_engineered:
            df = self.engineer_features(df)
            # Exclude target, original time (we have Hour/Day), and categorical bin
            feature_cols = [c for c in df.columns if c not in ['Class', 'Time', 'Amount_bin']]
        else:
            # Original features only (for comparison)
            feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        
        X = df[feature_cols]
        y = df['Class']
        
        self._feature_names = feature_cols
        
        # Stratified split to preserve fraud ratio
        return train_test_split(
            X, y,
            test_size=test_size or self.TEST_SIZE,
            stratify=y,
            random_state=random_state or self.RANDOM_STATE
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names used for ML."""
        if self._feature_names is None:
            # Default feature names
            return [f'V{i}' for i in range(1, 29)] + [
                'Time', 'Amount', 'Hour', 'Day', 'Amount_log',
                'Time_diff', 'V1_V2_interaction', 'V3_V4_interaction',
                'Amount_V1_interaction', 'Amount_V3_interaction'
            ]
        return self._feature_names
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Human-readable feature descriptions."""
        return {
            'Time': 'Seconds since first transaction',
            'V1-V28': 'PCA-transformed confidential features',
            'Amount': 'Transaction amount (USD)',
            'Amount_log': 'Log-transformed amount',
            'Hour': 'Hour of day (0-23)',
            'Day': 'Day number',
            'Time_diff': 'Time since previous transaction',
            'V1_V2_interaction': 'Interaction between V1 and V2',
            'V3_V4_interaction': 'Interaction between V3 and V4',
            'Amount_V1_interaction': 'Interaction between Amount and V1',
            'Amount_V3_interaction': 'Interaction between Amount and V3',
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        df = self.load()
        
        fraud_df = df[df['Class'] == 1]
        normal_df = df[df['Class'] == 0]
        
        return {
            "total_transactions": len(df),
            "fraud_count": len(fraud_df),
            "normal_count": len(normal_df),
            "fraud_rate": len(fraud_df) / len(df),
            "fraud_percentage": len(fraud_df) / len(df) * 100,
            "avg_fraud_amount": fraud_df['Amount'].mean(),
            "avg_normal_amount": normal_df['Amount'].mean(),
            "max_fraud_amount": fraud_df['Amount'].max(),
            "max_normal_amount": normal_df['Amount'].max(),
            "time_span_hours": (df['Time'].max() - df['Time'].min()) / 3600,
            "time_span_days": (df['Time'].max() - df['Time'].min()) / (3600 * 24),
        }
    
    def get_sample(self, n: int = 1000, fraud_ratio: Optional[float] = None) -> pd.DataFrame:
        """
        Get a stratified sample of the dataset.
        
        Args:
            n: Sample size
            fraud_ratio: Override fraud ratio (default uses dataset ratio)
            
        Returns:
            Sample DataFrame
        """
        df = self.load()
        
        if fraud_ratio is None:
            # Use stratified sample
            return df.groupby('Class', group_keys=False).apply(
                lambda x: x.sample(min(len(x), n // 2), random_state=self.RANDOM_STATE)
            ).sample(frac=1, random_state=self.RANDOM_STATE).reset_index(drop=True)
        else:
            # Custom fraud ratio
            fraud_n = int(n * fraud_ratio)
            normal_n = n - fraud_n
            
            fraud_sample = df[df['Class'] == 1].sample(
                min(fraud_n, len(df[df['Class'] == 1])),
                random_state=self.RANDOM_STATE
            )
            normal_sample = df[df['Class'] == 0].sample(
                min(normal_n, len(df[df['Class'] == 0])),
                random_state=self.RANDOM_STATE
            )
            
            return pd.concat([fraud_sample, normal_sample]).sample(
                frac=1, random_state=self.RANDOM_STATE
            ).reset_index(drop=True)


# Singleton instance
data_loader = CreditCardDataLoader()
