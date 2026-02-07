"""
SHAP-based explainability for fraud predictions.
"""
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# SHAP is optional - handle import error gracefully
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available. Explainability features disabled.")


class ModelExplainer:
    """
    SHAP-based explainability for fraud predictions.
    """
    
    def __init__(self, predictor):
        """
        Initialize explainer.
        
        Args:
            predictor: FraudPredictor instance
        """
        self.predictor = predictor
        self.explainers: Dict[str, Any] = {}
        
        if SHAP_AVAILABLE:
            self._init_explainers()
    
    def _init_explainers(self):
        """Initialize SHAP explainers for each model."""
        if not SHAP_AVAILABLE:
            return
        
        for name, model in self.predictor.models.items():
            try:
                if name == 'logistic_regression':
                    # For linear models, we need a background dataset
                    # Use zeros as a simple background (model is linear)
                    self.explainers[name] = None  # Will use coefficient-based explanation
                else:
                    # Tree explainer for tree-based models
                    self.explainers[name] = shap.TreeExplainer(model)
            except Exception as e:
                print(f"Failed to initialize explainer for {name}: {e}")
                self.explainers[name] = None
    
    def explain(
        self,
        features: Dict[str, Any],
        model_name: str = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Generate explanation for prediction.
        
        Args:
            features: Input features
            model_name: Model to explain
            top_n: Number of top features to return
            
        Returns:
            Dictionary with explanation
        """
        if model_name is None:
            model_name = self.predictor.default_model
        
        if model_name not in self.predictor.models:
            return {'error': f'Model {model_name} not found'}
        
        model = self.predictor.models[model_name]
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Ensure correct column order
        if self.predictor.feature_names:
            available = [f for f in self.predictor.feature_names if f in X.columns]
            X = X[available]
        elif hasattr(model, 'feature_names_in_'):
            X = X[model.feature_names_in_]
        
        # Generate explanation based on model type
        if model_name == 'logistic_regression' or 'logistic' in model_name:
            return self._explain_linear(model, X, top_n)
        else:
            return self._explain_tree(model, X, model_name, top_n)
    
    def _explain_linear(
        self,
        model,
        X: pd.DataFrame,
        top_n: int
    ) -> Dict[str, Any]:
        """Explain linear model using coefficients."""
        # Get coefficients
        if hasattr(model, 'named_steps'):
            # Pipeline
            classifier = model.named_steps['classifier']
            scaler = model.named_steps.get('scaler')
            coef = classifier.coef_[0]
            
            # Scale features if scaler exists
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values
        else:
            coef = model.coef_[0]
            X_scaled = X.values
        
        # Calculate contribution (coefficient * feature value)
        contributions = coef * X_scaled[0]
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Sort by absolute contribution
        feature_contribs = list(zip(feature_names, contributions, X.values[0]))
        feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Build result
        top_features = []
        for name, contrib, value in feature_contribs[:top_n]:
            top_features.append({
                'feature': name,
                'contribution': float(contrib),
                'direction': 'increases_risk' if contrib > 0 else 'decreases_risk',
                'value': float(value),
                'coefficient': float(coef[feature_names.index(name)])
            })
        
        # Calculate base value (intercept)
        if hasattr(model, 'named_steps'):
            intercept = model.named_steps['classifier'].intercept_[0]
        else:
            intercept = model.intercept_[0]
        
        # Convert log-odds to probability
        total_contrib = sum(contributions)
        prediction_prob = 1 / (1 + np.exp(-(intercept + total_contrib)))
        
        return {
            'top_features': top_features,
            'base_value': float(intercept),
            'prediction_log_odds': float(intercept + total_contrib),
            'prediction_probability': float(prediction_prob),
            'method': 'coefficient',
            'model': 'logistic_regression'
        }
    
    def _explain_tree(
        self,
        model,
        X: pd.DataFrame,
        model_name: str,
        top_n: int
    ) -> Dict[str, Any]:
        """Explain tree-based model using SHAP."""
        if not SHAP_AVAILABLE or self.explainers.get(model_name) is None:
            # Fallback to feature importance
            return self._explain_with_importance(model, X, top_n)
        
        try:
            explainer = self.explainers[model_name]
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # For binary classification, shap_values is list [class_0, class_1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Fraud class
            
            # Get feature names
            feature_names = X.columns.tolist()
            
            # Build contributions
            values = X.values[0]
            feature_contribs = list(zip(feature_names, shap_values[0], values))
            feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = []
            for name, contrib, value in feature_contribs[:top_n]:
                top_features.append({
                    'feature': name,
                    'contribution': float(contrib),
                    'direction': 'increases_risk' if contrib > 0 else 'decreases_risk',
                    'value': float(value)
                })
            
            # Get expected value
            base_value = float(explainer.expected_value)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])
            
            prediction = base_value + sum(shap_values[0])
            
            return {
                'top_features': top_features,
                'base_value': base_value,
                'prediction': float(prediction),
                'method': 'shap',
                'model': model_name
            }
            
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._explain_with_importance(model, X, top_n)
    
    def _explain_with_importance(
        self,
        model,
        X: pd.DataFrame,
        top_n: int
    ) -> Dict[str, Any]:
        """Fallback explanation using feature importance."""
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return {'error': 'Model does not support explanation'}
        
        feature_names = X.columns.tolist()
        
        # Sort by importance
        feature_importance = list(zip(feature_names, importances, X.values[0]))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        top_features = []
        for name, importance, value in feature_importance[:top_n]:
            top_features.append({
                'feature': name,
                'importance': float(importance),
                'value': float(value),
                'note': 'Feature importance (not contribution)'
            })
        
        return {
            'top_features': top_features,
            'method': 'feature_importance',
            'note': 'SHAP not available, using feature importance as fallback'
        }
    
    def explain_batch(
        self,
        df: pd.DataFrame,
        model_name: str = None,
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of transactions.
        
        Args:
            df: DataFrame with features
            model_name: Model to use
            top_n: Number of top features per explanation
            
        Returns:
            List of explanations
        """
        explanations = []
        for idx, row in df.iterrows():
            exp = self.explain(row.to_dict(), model_name, top_n)
            exp['row_index'] = idx
            explanations.append(exp)
        
        return explanations


# Singleton
_explainer: Optional[ModelExplainer] = None


def get_explainer(predictor=None) -> ModelExplainer:
    """Get or create explainer singleton."""
    global _explainer
    if _explainer is None:
        if predictor is None:
            from app.core.ml.predictor import get_predictor
            predictor = get_predictor()
        _explainer = ModelExplainer(predictor)
    return _explainer
