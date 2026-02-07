"""
ML Inference engine for fraud detection.
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.core.data.loader import CreditCardDataLoader


class FraudPredictor:
    """
    Production inference engine for fraud detection.
    Loads trained models and provides unified prediction interface.
    """
    
    MODEL_PATH = Path(settings.MODEL_PATH)
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Custom model path
        """
        if model_path:
            self.MODEL_PATH = Path(model_path)
        
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.default_model = settings.DEFAULT_MODEL
        self.feature_names: List[str] = []
        self.metadata: Dict[str, Any] = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models from disk."""
        model_files = {
            'logistic_regression': 'fraud_logistic_regression_v1.pkl',
            'random_forest': 'fraud_random_forest_v1.pkl',
            'xgboost': 'fraud_xgboost_v1.pkl'
        }
        
        for name, filename in model_files.items():
            path = self.MODEL_PATH / filename
            if path.exists():
                try:
                    self.models[name] = joblib.load(path)
                    print(f"Loaded model: {name}")
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
        
        # Load metrics
        metrics_path = self.MODEL_PATH / 'metrics.json'
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    self.metrics = json.load(f)
            except Exception as e:
                print(f"Failed to load metrics: {e}")
        
        # Load metadata
        metadata_path = self.MODEL_PATH / 'training_metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('features', [])
            except Exception as e:
                print(f"Failed to load metadata: {e}")
        
        if not self.models:
            print("WARNING: No models loaded. Please train models first.")
    
    def predict(
        self,
        features: Union[pd.DataFrame, Dict[str, Any]],
        model_name: str = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Predict fraud risk for transaction(s).
        
        Args:
            features: Input features (dict for single, DataFrame for batch)
            model_name: Specific model to use, or None for default
            threshold: Classification threshold
            
        Returns:
            Dictionary with prediction results
        """
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            raise ValueError(
                f"Model {model_name} not found. "
                f"Available: {list(self.models.keys())}"
            )
        
        model = self.models[model_name]
        
        # Convert dict to DataFrame if needed
        is_single = isinstance(features, dict)
        if is_single:
            features = pd.DataFrame([features])
        
        # Ensure correct column order
        if self.feature_names:
            # Only use features the model was trained on
            available_features = [f for f in self.feature_names if f in features.columns]
            features = features[available_features]
        elif hasattr(model, 'feature_names_in_'):
            features = features[model.feature_names_in_]
        
        # Measure prediction time
        start_time = time.time()
        
        # Predict
        probability = model.predict_proba(features)[:, 1]
        prediction = (probability > threshold).astype(int)
        
        processing_time = int((time.time() - start_time) * 1000)  # ms
        
        # Convert to risk score (0-100)
        risk_score = (probability * 100).astype(int)
        
        # Risk level categorization
        risk_level = np.where(
            risk_score < 30, 'Low',
            np.where(risk_score < 70, 'Medium', 'High')
        )
        
        result = {
            'fraud_probability': float(probability[0]) if is_single else probability.tolist(),
            'risk_score': int(risk_score[0]) if is_single else risk_score.tolist(),
            'risk_level': risk_level[0] if is_single else risk_level.tolist(),
            'prediction': int(prediction[0]) if is_single else prediction.tolist(),
            'model_used': model_name,
            'model_confidence': self._get_model_confidence(model_name),
            'threshold_used': threshold,
            'processing_time_ms': processing_time
        }
        
        return result
    
    def predict_batch(
        self,
        df: pd.DataFrame,
        model_name: str = None,
        include_features: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Predict for batch of transactions.
        
        Args:
            df: DataFrame with transaction features
            model_name: Model to use
            include_features: Whether to include input features in output
            
        Returns:
            List of prediction results
        """
        results = []
        
        for idx, row in df.iterrows():
            result = self.predict(row.to_dict(), model_name)
            result['row_index'] = idx
            
            if include_features:
                result['features'] = row.to_dict()
            
            results.append(result)
        
        return results
    
    def compare_models(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get predictions from all models for comparison.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary with predictions from each model
        """
        comparisons = {}
        for name in self.models.keys():
            comparisons[name] = self.predict(features, name)
        
        # Add ensemble prediction (average)
        probabilities = [comp['fraud_probability'] for comp in comparisons.values()]
        avg_prob = sum(probabilities) / len(probabilities)
        
        comparisons['ensemble'] = {
            'fraud_probability': avg_prob,
            'risk_score': int(avg_prob * 100),
            'risk_level': 'High' if avg_prob > 0.7 else 'Medium' if avg_prob > 0.3 else 'Low',
            'prediction': 1 if avg_prob > 0.5 else 0,
            'model_used': 'ensemble',
            'individual_predictions': {name: comp['prediction'] for name, comp in comparisons.items()}
        }
        
        return comparisons
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """
        Get feature importance for model explanation.
        
        Args:
            model_name: Model to get importance from
            
        Returns:
            Dictionary of feature names to importance scores
        """
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            names = self.feature_names
        elif hasattr(model, 'coef_'):
            # Linear models (Logistic Regression)
            importances = np.abs(model.coef_[0])
            names = self.feature_names
        elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            # Pipeline (Logistic Regression with scaler)
            classifier = model.named_steps['classifier']
            importances = np.abs(classifier.coef_[0])
            names = self.feature_names
        else:
            return {}
        
        # Map to feature names
        if len(names) != len(importances):
            names = [f'feature_{i}' for i in range(len(importances))]
        
        # Return top 10 features
        return dict(sorted(
            zip(names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'available_models': list(self.models.keys()),
            'default_model': self.default_model,
            'feature_count': len(self.feature_names),
            'models': {}
        }
        
        for name in self.models.keys():
            model_info = {
                'loaded': True,
                'metrics': self.metrics.get(name, {})
            }
            
            # Get model-specific info
            model = self.models[name]
            if hasattr(model, 'n_features_in_'):
                model_info['n_features'] = model.n_features_in_
            
            info['models'][name] = model_info
        
        return info
    
    def _get_model_confidence(self, model_name: str) -> float:
        """Get model's F1 score as confidence metric."""
        if model_name in self.metrics:
            return self.metrics[model_name].get('f1_score', 0.0)
        return 0.0
    
    def is_ready(self) -> bool:
        """Check if predictor is ready (models loaded)."""
        return len(self.models) > 0


# Singleton instance
_predictor: Optional[FraudPredictor] = None


def get_predictor() -> FraudPredictor:
    """Get or create predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = FraudPredictor()
    return _predictor
