I'll create a comprehensive `AGENT.md` file that serves as the master instruction document for Kimi (or any AI coding assistant) working on this project. This will include the complete file tree, training specifications, and detailed implementation guidance.

---

## AGENT.md

```markdown
# AGENT INSTRUCTION DOCUMENT
## AI-Based Risk Assessment and Compliance Monitoring System for Fraud Detection

**Version:** 1.0  
**Last Updated:** 2024-01-15  
**Primary Dataset:** creditcard.csv (Kaggle Credit Card Fraud Detection)  
**Stack:** React + TypeScript + Tailwind + FastAPI + Python + PostgreSQL

---

## 1. PROJECT OVERVIEW

### 1.1 What We're Building
A production-ready financial fraud detection platform that combines:
- **Machine Learning:** Three-model ensemble (Logistic Regression, Random Forest, XGBoost) for fraud risk scoring
- **Compliance Engine:** AML (Anti-Money Laundering) and KYC (Know Your Customer) rule checking
- **Secure File Processing:** Excel/CSV upload with AES-256 encryption and intelligent column mapping
- **Role-Based Access Control:** Multi-level permissions (Super Admin, Admin, Analyst, Auditor, API Client)
- **Explainable AI:** SHAP values for model interpretability

### 1.2 Core Value Proposition
Financial institutions upload transaction data → AI assesses fraud risk → Compliance violations flagged → Human analysts review high-risk cases → Audit trail maintained for regulators

### 1.3 Academic Context
This is a final year project demonstrating:
- AI/ML application in financial GRC (Governance, Risk, Compliance)
- Modern full-stack development practices
- Security-first architecture for sensitive data
- Real-world software engineering workflows

---

## 2. FILE TREE STRUCTURE

```
fraud-detection-grc/
├── AGENT.md                          # This file - master instructions
├── README.md                         # Project overview for GitHub
├── docker-compose.yml                # Full stack orchestration
├── .env.example                      # Environment variable template
├── .gitignore                        # Standard Python + Node ignores
│
├── backend/                          # FastAPI Application
│   ├── Dockerfile                    # Python 3.11 slim, non-root user
│   ├── requirements.txt              # Python dependencies
│   ├── pytest.ini                    # Test configuration
│   │
│   ├── app/                          # Main application package
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI entry point, middleware
│   │   ├── config.py                 # Pydantic settings, env vars
│   │   ├── dependencies.py           # FastAPI dependencies (DB, auth)
│   │   │
│   │   ├── api/                      # API route modules
│   │   │   ├── __init__.py
│   │   │   ├── router.py             # Main API router aggregator
│   │   │   │
│   │   │   ├── v1/                   # API Version 1
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py           # JWT login, MFA, password reset
│   │   │   │   ├── users.py          # User CRUD, role assignment
│   │   │   │   ├── analysis.py       # Single/batch prediction endpoints
│   │   │   │   ├── transactions.py   # Transaction review, approval
│   │   │   │   ├── compliance.py     # AML/KYC checks, alerts
│   │   │   │   ├── models.py         # Model info, comparison metrics
│   │   │   │   ├── admin.py          # System admin, audit logs
│   │   │   │   └── upload.py         # Secure file upload handling
│   │   │   │
│   │   │   └── deps.py               # Common dependencies (get_db, get_current_user)
│   │   │
│   │   ├── core/                     # Core business logic
│   │   │   ├── __init__.py
│   │   │   ├── security/             # Security modules
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py           # JWT token creation/validation
│   │   │   │   ├── passwords.py      # Bcrypt hashing
│   │   │   │   ├── encryption.py     # AES-256 file encryption
│   │   │   │   ├── permissions.py    # Role/permission definitions
│   │   │   │   └── audit.py          # Audit logging
│   │   │   │
│   │   │   ├── ml/                   # Machine learning
│   │   │   │   ├── __init__.py
│   │   │   │   ├── models/           # Saved model files (.pkl, .joblib)
│   │   │   │   │   ├── fraud_lr_v1.pkl
│   │   │   │   │   ├── fraud_rf_v1.pkl
│   │   │   │   │   ├── fraud_xgb_v1.pkl
│   │   │   │   │   └── preprocessor.pkl
│   │   │   │   │
│   │   │   │   ├── trainer.py        # Model training pipeline
│   │   │   │   ├── predictor.py      # Inference engine
│   │   │   │   ├── preprocessor.py   # Feature engineering
│   │   │   │   ├── explainer.py      # SHAP explainability
│   │   │   │   └── evaluator.py      # Model comparison metrics
│   │   │   │
│   │   │   ├── compliance/           # Compliance engine
│   │   │   │   ├── __init__.py
│   │   │   │   ├── aml_rules.py      # AML rule definitions
│   │   │   │   ├── kyc_checks.py     # KYC verification logic
│   │   │   │   ├── alert_manager.py  # Alert generation/assignment
│   │   │   │   └── risk_scorer.py    # Composite risk calculation
│   │   │   │
│   │   │   └── data/                 # Data processing
│   │   │       ├── __init__.py
│   │   │       ├── loader.py         # Dataset loading (Kaggle)
│   │   │       ├── synthetic.py      # Synthetic data generation
│   │   │       └── validators.py     # Input validation
│   │   │
│   │   ├── db/                       # Database
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # SQLAlchemy base, session
│   │   │   ├── init_db.py            # Database initialization
│   │   │   └── migrations/           # Alembic migrations
│   │   │       ├── env.py
│   │   │       ├── versions/         # Migration files
│   │   │       └── script.py.mako
│   │   │
│   │   ├── models/                   # SQLAlchemy models
│   │   │   ├── __init__.py
│   │   │   ├── user.py               # User, Role, Permission models
│   │   │   ├── transaction.py        # Transaction, AnalysisResult
│   │   │   ├── compliance.py         # Alert, ComplianceRule
│   │   │   └── audit.py              # AuditLog model
│   │   │
│   │   └── schemas/                  # Pydantic schemas
│   │       ├── __init__.py
│   │       ├── auth.py               # Login, Token, User schemas
│   │       ├── transaction.py        # Transaction input/output
│   │       ├── analysis.py           # Prediction results
│   │       ├── compliance.py         # Alert, Rule schemas
│   │       └── upload.py             # File upload schemas
│   │
│   ├── tests/                        # Test suite
│   │   ├── __init__.py
│   │   ├── conftest.py               # Pytest fixtures
│   │   ├── test_auth.py
│   │   ├── test_analysis.py
│   │   ├── test_compliance.py
│   │   └── test_security.py
│   │
│   └── scripts/                      # Utility scripts
│       ├── train_models.py           # Train and save ML models
│       ├── generate_data.py          # Create synthetic test data
│       └── seed_db.py                # Seed database with initial data
│
├── frontend/                         # React Application
│   ├── Dockerfile                    # Node 20 Alpine, nginx serve
│   ├── package.json                  # Dependencies + scripts
│   ├── tsconfig.json                 # TypeScript config
│   ├── vite.config.ts                # Vite build configuration
│   ├── tailwind.config.js            # Tailwind + shadcn theme
│   └── components.json               # shadcn/ui configuration
│
│   ├── public/                       # Static assets
│   │   ├── favicon.ico
│   │   └── logo.svg
│
│   └── src/
│       ├── main.tsx                  # React entry point
│       ├── App.tsx                   # Root component, routing
│       ├── index.css                 # Global styles + Tailwind
│       ├── vite-env.d.ts             # Vite type declarations
│
│       ├── api/                      # API client layer
│       │   ├── client.ts             # Axios instance, interceptors
│       │   ├── auth.ts               # Auth API calls
│       │   ├── analysis.ts           # Analysis API calls
│       │   ├── transactions.ts       # Transaction API calls
│       │   ├── compliance.ts         # Compliance API calls
│       │   └── upload.ts             # File upload (multipart/form-data)
│
│       ├── components/               # React components
│       │   ├── ui/                   # shadcn/ui components (auto-generated)
│       │   │   ├── button.tsx
│       │   │   ├── input.tsx
│       │   │   ├── table.tsx
│       │   │   ├── card.tsx
│       │   │   ├── dialog.tsx
│       │   │   ├── select.tsx
│       │   │   ├── tabs.tsx
│       │   │   ├── toast.tsx
│       │   │   └── ... (add as needed)
│       │   │
│       │   ├── auth/                 # Authentication components
│       │   │   ├── LoginForm.tsx
│       │   │   ├── MFASetup.tsx
│       │   │   ├── PasswordReset.tsx
│       │   │   └── ProtectedRoute.tsx
│       │   │
│       │   ├── upload/               # File upload components
│       │   │   ├── SmartUpload.tsx   # Main upload component
│       │   │   ├── ExcelParser.ts    # Client-side Excel conversion
│       │   │   ├── ColumnMapper.tsx  # Column mapping interface
│       │   │   ├── FilePreview.tsx   # Data preview table
│       │   │   └── ProgressTracker.tsx
│       │   │
│       │   ├── analysis/             # Risk analysis components
│       │   │   ├── RiskGauge.tsx     # Visual risk score (0-100)
│       │   │   ├── ExplanationPanel.tsx # SHAP explanations
│       │   │   ├── ModelComparison.tsx # Metrics comparison
│       │   │   ├── ConfusionMatrix.tsx
│       │   │   └── ROCCurve.tsx
│       │   │
│       │   ├── compliance/           # Compliance components
│       │   │   ├── AlertTable.tsx
│       │   │   ├── AlertDetail.tsx
│       │   │   ├── ComplianceDashboard.tsx
│       │   │   └── RuleEditor.tsx
│       │   │
│       │   ├── transactions/         # Transaction management
│       │   │   ├── TransactionTable.tsx
│       │   │   ├── TransactionDetail.tsx
│       │   │   ├── ApprovalWorkflow.tsx
│       │   │   └── BatchResults.tsx
│       │   │
│       │   └── layout/               # App shell components
│       │       ├── Navigation.tsx    # Sidebar navigation
│       │       ├── Header.tsx        # Top bar with user menu
│       │       ├── Footer.tsx
│       │       └── Layout.tsx        # Main layout wrapper
│
│       ├── context/                  # React context providers
│       │   ├── AuthContext.tsx       # Authentication state
│       │   └── ThemeContext.tsx      # Dark/light mode
│
│       ├── hooks/                    # Custom React hooks
│       │   ├── useAuth.ts            # Auth context consumer
│       │   ├── usePermission.ts      # Permission checking
│       │   ├── useAnalysis.ts        # Analysis operations
│       │   ├── useUpload.ts          # File upload logic
│       │   └── useToast.ts           # Notification system
│
│       ├── lib/                      # Utility libraries
│       │   ├── utils.ts              # General utilities (cn function)
│       │   ├── validators.ts         # Zod schemas
│       │   ├── formatters.ts         # Number/date formatting
│       │   └── constants.ts          # App constants
│
│       ├── pages/                    # Route pages (lazy loaded)
│       │   ├── Login.tsx
│       │   ├── Register.tsx
│       │   ├── Dashboard.tsx         # Executive summary view
│       │   ├── Upload.tsx            # File upload page
│       │   ├── Analysis.tsx          # Single transaction analysis
│       │   ├── BatchResults.tsx      # Batch processing results
│       │   ├── ModelComparison.tsx   # Model metrics page
│       │   ├── Compliance.tsx        # Alerts and monitoring
│       │   ├── Transactions.tsx      # Transaction list
│       │   ├── AuditLogs.tsx         # Audit trail (admin)
│       │   ├── UserManagement.tsx    # Admin user management
│       │   ├── Settings.tsx          # System settings
│       │   └── NotFound.tsx
│
│       ├── stores/                   # Zustand state stores
│       │   ├── authStore.ts
│       │   ├── analysisStore.ts
│       │   └── uploadStore.ts
│
│       └── types/                    # TypeScript type definitions
│           ├── auth.ts
│           ├── transaction.ts
│           ├── analysis.ts
│           ├── compliance.ts
│           └── api.ts
│
├── data/                             # Data directory (gitignored)
│   ├── creditcard.csv                # PRIMARY: Kaggle dataset (150MB)
│   ├── synthetic/                    # Generated synthetic data
│   └── samples/                      # Sample files for testing
│
├── models/                           # Trained ML models (gitignored)
│   ├── fraud_lr_v1.pkl
│   ├── fraud_rf_v1.pkl
│   ├── fraud_xgb_v1.pkl
│   ├── preprocessor.pkl
│   └── training_metadata.json
│
├── notebooks/                        # Jupyter notebooks (exploration)
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_explainability.ipynb
│
├── docs/                             # Documentation
│   ├── architecture.md
│   ├── api_spec.md
│   ├── security.md
│   └── deployment.md
│
└── scripts/                          # Utility scripts
    ├── setup.sh                      # Initial setup
    ├── train.sh                      # Train models
    ├── test.sh                       # Run test suite
    └── deploy.sh                     # Deployment script
```

---

## 3. TRAINING DATA SPECIFICATION

### 3.1 Primary Dataset: creditcard.csv

**Source:** Kaggle Credit Card Fraud Detection  
**Location:** `data/creditcard.csv`  
**Size:** 284,807 rows × 31 columns (150MB)  
**Fraud Distribution:** 0.172% (492 frauds) - Highly imbalanced

**Column Schema:**
| Column | Type | Description | Usage |
|--------|------|-------------|-------|
| `Time` | float | Seconds elapsed from first transaction | Feature engineering (hour, velocity) |
| `V1` - `V28` | float | PCA-transformed features (anonymized) | Primary model inputs |
| `Amount` | float | Transaction amount in USD | Feature + risk factor |
| `Class` | int | 1 = Fraud, 0 = Normal | Target variable |

**Data Characteristics:**
- All features are numeric (no categorical preprocessing needed)
- `Amount` varies widely (requires log transformation)
- Extreme class imbalance (use SMOTE or class weights)
- No missing values
- Features V1-V28 are already normalized (PCA output)

### 3.2 Data Loading Specification

```python
# backend/app/core/data/loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class CreditCardDataLoader:
    """
    Loads and preprocesses creditcard.csv for training and inference.
    """
    
    DATA_PATH = Path("data/creditcard.csv")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    def load(self) -> pd.DataFrame:
        """Load raw dataset with validation."""
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
        assert df['Class'].isin([0, 1]).all(), "Invalid values in Class column"
        assert df['Class'].sum() > 0, "No fraud cases found (should be 492)"
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from raw data.
        """
        df = df.copy()
        
        # Time-based features
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / (3600 * 24)).astype(int)
        
        # Amount features
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_bin'] = pd.qcut(df['Amount'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Velocity features (simulated - would need customer ID for real)
        # For this dataset, we use time-window aggregates
        df['Time_diff'] = df['Time'].diff().fillna(0)
        
        # Interaction features
        df['V1_V2_interaction'] = df['V1'] * df['V2']
        
        return df
    
    def prepare_ml(self, df: pd.DataFrame, include_engineered: bool = True) -> tuple:
        """
        Prepare features and target for machine learning.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if include_engineered:
            df = self.engineer_features(df)
            feature_cols = [c for c in df.columns if c not in ['Class', 'Time', 'Amount_bin']]
        else:
            # Original features only (for comparison)
            feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        
        X = df[feature_cols]
        y = df['Class']
        
        # Stratified split to preserve fraud ratio
        return train_test_split(
            X, y, 
            test_size=self.TEST_SIZE, 
            stratify=y, 
            random_state=self.RANDOM_STATE
        )
    
    def get_feature_names(self) -> dict:
        """Human-readable feature descriptions."""
        return {
            'Time': 'Seconds since first transaction',
            'V1-V28': 'PCA-transformed confidential features',
            'Amount': 'Transaction amount (USD)',
            'Amount_log': 'Log-transformed amount',
            'Hour': 'Hour of day (0-23)',
            'Day': 'Day number',
        }
```

---

## 4. MACHINE LEARNING SPECIFICATION

### 4.1 Model Architecture: Three-Model Ensemble

We train three distinct models to demonstrate different approaches to fraud detection:

#### Model 1: Logistic Regression (Baseline)
```python
# backend/app/core/ml/trainer.py - Model 1
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_logistic_regression(X_train, y_train):
    """
    Baseline model: Interpretable, fast, linear decision boundary.
    Good for: Understanding feature directions, regulatory explainability
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            class_weight='balanced',  # Handle imbalance
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline
```

**Hyperparameters:**
- `class_weight='balanced'` (automatically adjusts for 0.172% fraud rate)
- `max_iter=1000` (ensure convergence)
- `solver='lbfgs'` (default, good for small datasets)

**Expected Performance:**
- Accuracy: ~95%
- Precision: ~10% (many false positives due to imbalance)
- Recall: ~90% (catches most frauds)
- F1: ~0.18
- AUC-ROC: ~0.95

---

#### Model 2: Random Forest (Tree Ensemble)
```python
# backend/app/core/ml/trainer.py - Model 2
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    """
    Tree ensemble: Handles non-linearity, provides feature importance.
    Good for: Production use, robust predictions, feature insights
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
    return model
```

**Hyperparameters:**
- `n_estimators=100` (standard, good performance/speed tradeoff)
- `max_depth=10` (limit depth to prevent overfitting on PCA features)
- `class_weight='balanced_subsample'` (more aggressive balancing)
- `n_jobs=-1` (parallel training)

**Expected Performance:**
- Accuracy: ~99.9%
- Precision: ~80%
- Recall: ~75%
- F1: ~0.77
- AUC-ROC: ~0.98

---

#### Model 3: XGBoost (Gradient Boosting)
```python
# backend/app/core/ml/trainer.py - Model 3
import xgboost as xgb

def train_xgboost(X_train, y_train):
    """
    Gradient boosting: State-of-the-art for tabular data.
    Good for: Maximum accuracy, handling complex interactions
    """
    # Calculate scale_pos_weight for imbalanced data
    # scale_pos_weight = number of negative / number of positive
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
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
        eval_set=[(X_train, y_train)],  # Monitor training
        verbose=False
    )
    return model
```

**Hyperparameters:**
- `scale_pos_weight=~579` (284315 / 492) - Critical for imbalance
- `max_depth=6` (prevent overfitting)
- `learning_rate=0.1` (standard)
- `eval_metric='aucpr'` (Precision-Recall AUC better for imbalanced data)

**Expected Performance:**
- Accuracy: ~99.95%
- Precision: ~90%
- Recall: ~85%
- F1: ~0.87
- AUC-ROC: ~0.99

---

### 4.2 Training Pipeline

```python
# backend/app/core/ml/trainer.py - Main training script
import joblib
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve
)

class ModelTrainer:
    def __init__(self, data_loader: CreditCardDataLoader):
        self.data_loader = data_loader
        self.models = {}
        self.metrics = {}
        
    def train_all(self, save_path: str = "models/"):
        """Train all three models and save artifacts."""
        
        # Load and prepare data
        df = self.data_loader.load()
        X_train, X_test, y_train, y_test = self.data_loader.prepare_ml(df)
        
        # Store test set for evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Train Model 1: Logistic Regression
        print("Training Logistic Regression...")
        self.models['logistic_regression'] = train_logistic_regression(X_train, y_train)
        
        # Train Model 2: Random Forest
        print("Training Random Forest...")
        self.models['random_forest'] = train_random_forest(X_train, y_train)
        
        # Train Model 3: XGBoost
        print("Training XGBoost...")
        self.models['xgboost'] = train_xgboost(X_train, y_train)
        
        # Evaluate all models
        self._evaluate_all()
        
        # Save models and metrics
        self._save_artifacts(save_path)
        
        return self.models, self.metrics
    
    def _evaluate_all(self):
        """Calculate comprehensive metrics for all models."""
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            self.metrics[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_prob),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            # Find optimal threshold (maximize F1)
            precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_prob)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            optimal_idx = np.argmax(f1_scores)
            self.metrics[name]['optimal_threshold'] = thresholds[optimal_idx]
            self.metrics[name]['optimal_f1'] = f1_scores[optimal_idx]
    
    def _save_artifacts(self, save_path: str):
        """Save models, metrics, and metadata."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{save_path}/fraud_{name}_v1.pkl")
        
        # Save metrics
        with open(f"{save_path}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save metadata
        metadata = {
            'trained_at': datetime.utcnow().isoformat(),
            'dataset': 'creditcard.csv',
            'n_samples': len(self.y_test) * 5,  # Approximate
            'fraud_rate': float(self.y_test.mean()),
            'features': list(self.X_test.columns),
            'models': list(self.models.keys())
        }
        with open(f"{save_path}/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {save_path}")
        print(f"Best model by F1: {max(self.metrics, key=lambda x: self.metrics[x]['f1_score'])}")

# CLI usage
if __name__ == "__main__":
    from backend.app.core.data.loader import CreditCardDataLoader
    
    loader = CreditCardDataLoader()
    trainer = ModelTrainer(loader)
    trainer.train_all()
```

---

### 4.3 Inference Engine

```python
# backend/app/core/ml/predictor.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union

class FraudPredictor:
    """
    Production inference engine for fraud detection.
    Loads trained models and provides unified prediction interface.
    """
    
    MODEL_PATH = Path("models/")
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.default_model = 'xgboost'
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
                self.models[name] = joblib.load(path)
                print(f"Loaded model: {name}")
        
        # Load metrics
        metrics_path = self.MODEL_PATH / 'metrics.json'
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                self.metrics = json.load(f)
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray, Dict], 
                model_name: str = None) -> Dict:
        """
        Predict fraud risk for transaction(s).
        
        Args:
            features: Input features (dict for single, DataFrame for batch)
            model_name: Specific model to use, or None for default
            
        Returns:
            Dictionary with risk score, probability, explanation
        """
        if model_name is None:
            model_name = self.default_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        # Ensure correct column order
        if hasattr(model, 'feature_names_in_'):
            features = features[model.feature_names_in_]
        
        # Predict
        probability = model.predict_proba(features)[:, 1]
        prediction = (probability > 0.5).astype(int)
        
        # Convert to risk score (0-100)
        risk_score = (probability * 100).astype(int)
        
        # Risk level categorization
        risk_level = np.where(
            risk_score < 30, 'Low',
            np.where(risk_score < 70, 'Medium', 'High')
        )
        
        return {
            'fraud_probability': float(probability[0]) if len(probability) == 1 else probability.tolist(),
            'risk_score': int(risk_score[0]) if len(risk_score) == 1 else risk_score.tolist(),
            'risk_level': risk_level[0] if len(risk_level) == 1 else risk_level.tolist(),
            'prediction': int(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
            'model_used': model_name,
            'model_confidence': self._get_model_confidence(model_name)
        }
    
    def predict_batch(self, df: pd.DataFrame, model_name: str = None) -> List[Dict]:
        """Predict for batch of transactions."""
        results = []
        for _, row in df.iterrows():
            result = self.predict(row.to_dict(), model_name)
            result['transaction_id'] = row.get('transaction_id', _)
            results.append(result)
        return results
    
    def compare_models(self, features: Dict) -> Dict:
        """Get predictions from all models for comparison."""
        comparisons = {}
        for name in self.models.keys():
            comparisons[name] = self.predict(features, name)
        return comparisons
    
    def _get_model_confidence(self, model_name: str) -> float:
        """Get model's F1 score as confidence metric."""
        if model_name in self.metrics:
            return self.metrics[model_name].get('f1_score', 0.0)
        return 0.0
    
    def get_feature_importance(self, model_name: str = None) -> Dict:
        """Get feature importance for model explanation."""
        if model_name is None:
            model_name = self.default_model
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_[0])
        else:
            return {}
        
        # Map to feature names
        if hasattr(model, 'feature_names_in_'):
            names = model.feature_names_in_
        else:
            names = [f'feature_{i}' for i in range(len(importances))]
        
        return dict(sorted(
            zip(names, importances),
            key=lambda x: x[1],
            reverse=True
        )[:10])  # Top 10 features

# Singleton instance for FastAPI
predictor = FraudPredictor()
```

---

### 4.4 Explainability (SHAP)

```python
# backend/app/core/ml/explainer.py
import shap
import numpy as np
import pandas as pd
from typing import Dict, List

class ModelExplainer:
    """
    SHAP-based explainability for fraud predictions.
    """
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.explainers = {}
        self._init_explainers()
    
    def _init_explainers(self):
        """Initialize SHAP explainers for each model."""
        for name, model in self.predictor.models.items():
            if name == 'logistic_regression':
                # Linear explainer for linear models
                self.explainers[name] = shap.LinearExplainer(
                    model, 
                    shap.sample(self.predictor.X_test, 100)
                )
            else:
                # Tree explainer for tree-based models
                self.explainers[name] = shap.TreeExplainer(model)
    
    def explain(self, features: Dict, model_name: str = None) -> Dict:
        """
        Generate SHAP explanation for prediction.
        
        Returns:
            Top 3 features driving the prediction with values
        """
        if model_name is None:
            model_name = self.predictor.default_model
        
        # Convert to array
        X = pd.DataFrame([features])
        if hasattr(self.predictor.models[model_name], 'feature_names_in_'):
            X = X[self.predictor.models[model_name].feature_names_in_]
        
        # Calculate SHAP values
        explainer = self.explainers[model_name]
        shap_values = explainer.shap_values(X)
        
        # For binary classification, shap_values is list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Fraud class
        
        # Get top 3 features
        feature_names = X.columns
        shap_dict = dict(zip(feature_names, shap_values[0]))
        top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        return {
            'top_features': [
                {
                    'feature': name,
                    'contribution': float(value),
                    'direction': 'increases_risk' if value > 0 else 'decreases_risk',
                    'value': float(X[name].values[0])
                }
                for name, value in top_features
            ],
            'base_value': float(explainer.expected_value),
            'prediction': float(shap_values[0].sum() + explainer.expected_value)
        }
```

---

## 5. COMPLIANCE ENGINE SPECIFICATION

### 5.1 AML Rule Engine

```python
# backend/app/core/compliance/aml_rules.py
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AMLAlert:
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    description: str
    triggered_value: float
    threshold: float

class AMLRuleEngine:
    """
    Anti-Money Laundering rule engine.
    Implements regulatory checks for suspicious activity.
    """
    
    # Regulatory thresholds
    CTR_THRESHOLD = 10000  # Currency Transaction Report (US)
    STRUCTURING_WINDOW = 24  # hours
    STRUCTURING_COUNT = 3  # transactions
    HIGH_RISK_COUNTRIES = {'NG', 'RU', 'IR', 'KP', 'SY'}  # ISO codes
    
    def __init__(self):
        self.rules = [
            self.check_ctr_threshold,
            self.check_structuring,
            self.check_velocity,
            self.check_high_risk_geography,
            self.check_unusual_hours,
            self.check_new_account_large_txn
        ]
    
    def evaluate(self, transaction: Dict, customer_history: List[Dict] = None) -> List[AMLAlert]:
        """
        Evaluate transaction against all AML rules.
        
        Args:
            transaction: Current transaction data
            customer_history: Previous transactions (for velocity/structuring)
            
        Returns:
            List of triggered alerts (empty if compliant)
        """
        alerts = []
        
        for rule in self.rules:
            alert = rule(transaction, customer_history)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def check_ctr_threshold(self, txn: Dict, history: List[Dict] = None) -> AMLAlert:
        """Rule: Transactions > $10,000 require reporting."""
        amount = txn.get('amount', 0)
        if amount > self.CTR_THRESHOLD:
            return AMLAlert(
                rule_id="AML-001",
                rule_name="CTR Threshold Exceeded",
                severity=AlertSeverity.HIGH,
                description=f"Transaction amount ${amount:,.2f} exceeds ${self.CTR_THRESHOLD:,} CTR threshold",
                triggered_value=amount,
                threshold=self.CTR_THRESHOLD
            )
        return None
    
    def check_structuring(self, txn: Dict, history: List[Dict] = None) -> AMLAlert:
        """
        Rule: Multiple transactions just below threshold to evade reporting.
        """
        if not history:
            return None
        
        amount = txn.get('amount', 0)
        # Check if amount is just below threshold (90-100%)
        if amount < self.CTR_THRESHOLD and amount > (self.CTR_THRESHOLD * 0.9):
            # Count similar transactions in window
            recent = [
                h for h in history 
                if h.get('amount', 0) > (self.CTR_THRESHOLD * 0.9)
                and h.get('amount', 0) < self.CTR_THRESHOLD
            ]
            
            if len(recent) >= self.STRUCTURING_COUNT:
                return AMLAlert(
                    rule_id="AML-002",
                    rule_name="Potential Structuring",
                    severity=AlertSeverity.CRITICAL,
                    description=f"{len(recent)+1} transactions just below CTR threshold in {self.STRUCTURING_WINDOW}h",
                    triggered_value=len(recent)+1,
                    threshold=self.STRUCTURING_COUNT
                )
        return None
    
    def check_velocity(self, txn: Dict, history: List[Dict] = None) -> AMLAlert:
        """Rule: Unusual transaction frequency."""
        if not history:
            return None
        
        # Count transactions in last hour
        recent_count = len([h for h in history if self._hours_ago(h, txn) < 1])
        
        if recent_count > 5:
            return AMLAlert(
                rule_id="AML-003",
                rule_name="Velocity Check Failed",
                severity=AlertSeverity.MEDIUM,
                description=f"{recent_count} transactions in last hour",
                triggered_value=recent_count,
                threshold=5
            )
        return None
    
    def check_high_risk_geography(self, txn: Dict, history: List[Dict] = None) -> AMLAlert:
        """Rule: Transactions involving high-risk jurisdictions."""
        country = txn.get('country', '').upper()
        if country in self.HIGH_RISK_COUNTRIES:
            return AMLAlert(
                rule_id="AML-004",
                rule_name="High-Risk Geography",
                severity=AlertSeverity.HIGH,
                description=f"Transaction involves high-risk jurisdiction: {country}",
                triggered_value=country,
                threshold=None
            )
        return None
    
    def check_unusual_hours(self, txn: Dict, history: List[Dict] = None) -> AMLAlert:
        """Rule: Transactions outside normal business hours."""
        hour = txn.get('hour_of_day', 12)
        if hour < 6 or hour > 23:
            return AMLAlert(
                rule_id="AML-005",
                rule_name="Unusual Hours",
                severity=AlertSeverity.LOW,
                description=f"Transaction at {hour}:00 (outside 6AM-11PM)",
                triggered_value=hour,
                threshold="6-23"
            )
        return None
    
    def check_new_account_large_txn(self, txn: Dict, history: List[Dict] = None) -> AMLAlert:
        """Rule: Large transaction from recently opened account."""
        account_age = txn.get('account_age_days', 365)
        amount = txn.get('amount', 0)
        
        if account_age < 7 and amount > 1000:
            return AMLAlert(
                rule_id="AML-006",
                rule_name="New Account Large Transaction",
                severity=AlertSeverity.MEDIUM,
                description=f"${amount:,.2f} transaction from {account_age}-day-old account",
                triggered_value=amount,
                threshold=1000
            )
        return None
    
    def _hours_ago(self, past_txn: Dict, current_txn: Dict) -> float:
        """Calculate hours between transactions."""
        past_time = past_txn.get('timestamp', 0)
        current_time = current_txn.get('timestamp', 0)
        return (current_time - past_time) / 3600

# Singleton
aml_engine = AMLRuleEngine()
```

---

## 6. SECURITY SPECIFICATION

### 6.1 File Encryption

```python
# backend/app/core/security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import tempfile
import shutil
from pathlib import Path

class SecureFileManager:
    """
    AES-256 encryption for uploaded files.
    Guarantees secure storage and deletion.
    """
    
    def __init__(self):
        self.master_key = os.getenv('FILE_ENCRYPTION_KEY')
        if not self.master_key:
            raise ValueError("FILE_ENCRYPTION_KEY environment variable required")
        
        # Create temp directory in RAM if possible (/dev/shm on Linux)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="fraud_secure_", dir="/dev/shm"))
        
        # Initialize cipher
        self._init_cipher()
    
    def _init_cipher(self):
        """Initialize Fernet cipher with derived key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        self.cipher = Fernet(key)
    
    def encrypt_store(self, file_bytes: bytes) -> str:
        """
        Encrypt and store file. Returns secure file ID.
        """
        file_id = base64.urlsafe_b64encode(os.urandom(32)).decode()
        encrypted = self.cipher.encrypt(file_bytes)
        
        secure_path = self.temp_dir / f"{file_id}.enc"
        secure_path.write_bytes(encrypted)
        os.chmod(secure_path, 0o600)  # Owner read/write only
        
        return file_id
    
    def decrypt_retrieve(self, file_id: str) -> bytes:
        """Decrypt and return file content."""
        secure_path = self.temp_dir / f"{file_id}.enc"
        if not secure_path.exists():
            raise FileNotFoundError("Secure file expired or invalid")
        
        encrypted = secure_path.read_bytes()
        return self.cipher.decrypt(encrypted)
    
    def secure_delete(self, file_id: str):
        """Cryptographic erasure - overwrite then delete."""
        secure_path = self.temp_dir / f"{file_id}.enc"
        if not secure_path.exists():
            return
        
        # Overwrite with random data
        size = secure_path.stat().st_size
        with open(secure_path, 'wb') as f:
            f.write(os.urandom(size))
        
        # Sync to disk, then delete
        os.fsync(f.fileno())
        secure_path.unlink()
    
    def cleanup(self):
        """Emergency cleanup of all files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

secure_manager = SecureFileManager()
```

---

## 7. API ENDPOINTS

### 7.1 Core Endpoints

| Method | Endpoint | Handler | Auth | Description |
|--------|----------|---------|------|-------------|
| POST | `/api/v1/auth/login` | `auth.login` | No | JWT token issuance |
| POST | `/api/v1/auth/refresh` | `auth.refresh` | Refresh | New access token |
| POST | `/api/v1/analysis/single` | `analysis.analyze_single` | Yes | Single transaction risk score |
| POST | `/api/v1/analysis/batch` | `analysis.analyze_batch` | Yes | CSV batch processing |
| GET | `/api/v1/analysis/status/{job_id}` | `analysis.get_status` | Yes | Batch job status |
| POST | `/api/v1/upload` | `upload.upload_file` | Yes | Secure file upload |
| GET | `/api/v1/models` | `models.list_models` | Yes | Available models & metrics |
| POST | `/api/v1/compliance/check` | `compliance.check_transaction` | Yes | AML/KYC verification |
| GET | `/api/v1/admin/audit-logs` | `admin.get_audit_logs` | Admin | System audit trail |

---

## 8. DEVELOPMENT COMMANDS

```bash
# Setup
cp .env.example .env
docker-compose up -d db redis
cd backend && pip install -r requirements.txt
cd frontend && npm install

# Train models
cd backend
python -m app.scripts.train_models

# Run development
docker-compose up  # Full stack
# OR separately:
cd backend && uvicorn app.main:app --reload
cd frontend && npm run dev

# Testing
cd backend && pytest
cd frontend && npm test
```

---

## 9. KEY IMPLEMENTATION NOTES FOR AI ASSISTANT

### When Writing Code:

1. **Always use type hints** - Python 3.11+ style
2. **Pydantic models for all inputs/outputs** - FastAPI dependency
3. **Async/await for I/O operations** - Database, file system
4. **Never commit secrets** - Use environment variables
5. **Handle imbalanced data** - Use `class_weight` or SMOTE
6. **Log security events** - All auth attempts, file operations
7. **Validate file types** - Magic numbers, not just extensions
8. **Sanitize filenames** - Prevent path traversal
9. **Use transactions** - Database consistency
10. **Implement timeouts** - ML inference, external calls

### When Generating Components:

- **Backend**: FastAPI router → Service layer → Repository pattern
- **Frontend**: Page → Components → Hooks → API client
- **ML**: Data loader → Preprocessor → Model → Evaluator → Explainer
- **Security**: Validate → Sanitize → Encrypt → Audit → Cleanup

### File Naming Conventions:

- Backend: `snake_case.py`, classes `PascalCase`, functions `snake_case`
- Frontend: `PascalCase.tsx` for components, `camelCase.ts` for utilities
- Models: `fraud_{algorithm}_v{version}.pkl`
- Database: Table names `snake_case`, columns `snake_case`

---

## 10. VALIDATION CHECKLIST

Before considering a feature complete:

- [ ] Type hints throughout
- [ ] Pydantic validation schemas
- [ ] Error handling with appropriate HTTP status codes
- [ ] Unit tests with >80% coverage
- [ ] Security review (no hardcoded secrets, input validation)
- [ ] API documentation (FastAPI auto-generated)
- [ ] Frontend TypeScript interfaces match API
- [ ] Accessibility (ARIA labels, keyboard navigation)
- [ ] Responsive design (mobile, tablet, desktop)
- [ ] Audit logging for security events

---

**END OF AGENT INSTRUCTIONS**

This document is the single source of truth. Update it as the project evolves.
```

---

This `AGENT.md` provides comprehensive instructions for any AI assistant (including Kimi) to understand the project architecture, implement features correctly, and maintain consistency across the codebase. It covers the complete file tree, ML training specifications with your `creditcard.csv` dataset, security requirements, and development workflows.