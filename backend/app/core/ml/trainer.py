"""
ML Model training pipeline for fraud detection.
Trains three models: Logistic Regression, Random Forest, XGBoost.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score
)

import xgboost as xgb

from app.config import settings
from app.core.data.loader import CreditCardDataLoader


class ModelTrainer:
    """Train and evaluate fraud detection models."""
    
    def __init__(self, data_loader: CreditCardDataLoader = None, model_path: str = None):
        """
        Initialize model trainer.
        
        Args:
            data_loader: Data loader instance
            model_path: Path to save models
        """
        self.data_loader = data_loader or CreditCardDataLoader()
        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict] = {}
        self.X_test: pd.DataFrame = None
        self.y_test: pd.Series = None
        self.feature_names: list = []
    
    def train_all(self, save: bool = True) -> Tuple[Dict, Dict]:
        """
        Train all three models and save artifacts.
        
        Args:
            save: Whether to save models to disk
            
        Returns:
            (models dict, metrics dict)
        """
        # Load and prepare data
        print("Loading and preparing data...")
        df = self.data_loader.load()
        X_train, X_test, y_train, y_test = self.data_loader.prepare_ml(df)
        
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = list(X_train.columns)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(self.feature_names)}")
        print(f"Fraud rate in training: {y_train.mean():.4f}")
        
        # Train Model 1: Logistic Regression
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        self.models['logistic_regression'] = self._train_logistic_regression(X_train, y_train)
        
        # Train Model 2: Random Forest
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        self.models['random_forest'] = self._train_random_forest(X_train, y_train)
        
        # Train Model 3: XGBoost
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        self.models['xgboost'] = self._train_xgboost(X_train, y_train)
        
        # Evaluate all models
        print("\n" + "="*50)
        print("Evaluating models...")
        print("="*50)
        self._evaluate_all()
        
        # Save models and metrics
        if save:
            self._save_artifacts()
        
        return self.models, self.metrics
    
    def _train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Train Logistic Regression model.
        Baseline model: Interpretable, fast, linear decision boundary.
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                class_weight='balanced',  # Handle imbalance
                max_iter=1000,
                random_state=42,
                solver='lbfgs',
                n_jobs=-1
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Print feature importance (coefficients)
        coef = pipeline.named_steps['classifier'].coef_[0]
        top_features = sorted(
            zip(X_train.columns, coef),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        print("Top 5 features by coefficient magnitude:")
        for name, value in top_features:
            print(f"  {name}: {value:.4f}")
        
        return pipeline
    
    def _train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train Random Forest model.
        Tree ensemble: Handles non-linearity, provides feature importance.
        """
        model = RandomForestClassifier(
            n_estimators=100,           # Number of trees
            max_depth=10,               # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',  # Handle imbalance per tree
            random_state=42,
            n_jobs=-1                   # Use all cores
        )
        
        model.fit(X_train, y_train)
        
        # Print feature importance
        importances = model.feature_importances_
        top_features = sorted(
            zip(X_train.columns, importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print("Top 5 features by importance:")
        for name, value in top_features:
            print(f"  {name}: {value:.4f}")
        
        return model
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Train XGBoost model.
        Gradient boosting: State-of-the-art for tabular data.
        """
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # Critical for imbalanced data
            eval_metric='aucpr',  # Area under precision-recall curve
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        
        # Print feature importance
        importances = model.feature_importances_
        top_features = sorted(
            zip(X_train.columns, importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print("Top 5 features by importance:")
        for name, value in top_features:
            print(f"  {name}: {value:.4f}")
        
        return model
    
    def _evaluate_all(self):
        """Calculate comprehensive metrics for all models."""
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            self.metrics[name] = {
                'accuracy': float(accuracy_score(self.y_test, y_pred)),
                'precision': float(precision_score(self.y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(self.y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(self.y_test, y_pred, zero_division=0)),
                'roc_auc': float(roc_auc_score(self.y_test, y_prob)),
                'average_precision': float(average_precision_score(self.y_test, y_prob)),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'classification_report': classification_report(
                    self.y_test, y_pred, output_dict=True, zero_division=0
                )
            }
            
            # Find optimal threshold (maximize F1)
            precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_prob)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            if optimal_idx < len(thresholds):
                self.metrics[name]['optimal_threshold'] = float(thresholds[optimal_idx])
                self.metrics[name]['optimal_f1'] = float(f1_scores[optimal_idx])
            
            # Print summary
            cm = self.metrics[name]['confusion_matrix']
            print(f"  Accuracy: {self.metrics[name]['accuracy']:.4f}")
            print(f"  Precision: {self.metrics[name]['precision']:.4f}")
            print(f"  Recall: {self.metrics[name]['recall']:.4f}")
            print(f"  F1 Score: {self.metrics[name]['f1_score']:.4f}")
            print(f"  ROC AUC: {self.metrics[name]['roc_auc']:.4f}")
            print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    
    def _save_artifacts(self):
        """Save models, metrics, and metadata."""
        print(f"\nSaving artifacts to {self.model_path}...")
        
        # Save models
        for name, model in self.models.items():
            filename = f"fraud_{name}_v1.pkl"
            filepath = self.model_path / filename
            joblib.dump(model, filepath)
            print(f"  Saved {filename}")
        
        # Save metrics
        metrics_path = self.model_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"  Saved metrics.json")
        
        # Save metadata
        metadata = {
            'trained_at': datetime.utcnow().isoformat(),
            'dataset': 'creditcard.csv',
            'n_samples': len(self.y_test) * 5,  # Approximate
            'test_size': len(self.y_test),
            'fraud_rate': float(self.y_test.mean()),
            'features': self.feature_names,
            'n_features': len(self.feature_names),
            'models': list(self.models.keys()),
            'best_model_by_f1': max(self.metrics, key=lambda x: self.metrics[x]['f1_score']),
            'best_model_by_auc': max(self.metrics, key=lambda x: self.metrics[x]['roc_auc'])
        }
        
        metadata_path = self.model_path / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved training_metadata.json")
        
        print("\n" + "="*50)
        print("Training complete!")
        print(f"Best model by F1: {metadata['best_model_by_f1']}")
        print(f"Best model by AUC: {metadata['best_model_by_auc']}")
        print("="*50)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get model comparison as DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
        
        comparison = []
        for name, metrics in self.metrics.items():
            comparison.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics['roc_auc'],
                'Avg Precision': metrics['average_precision']
            })
        
        return pd.DataFrame(comparison)


# CLI usage
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all()
    
    # Print comparison
    print("\nModel Comparison:")
    print(trainer.get_model_comparison().to_string(index=False))
