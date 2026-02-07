# Fraud Detection Backend

FastAPI-based backend for AI-Based Risk Assessment and Compliance Monitoring System.

## Features

- **Authentication & Authorization**: JWT-based auth with RBAC
- **ML Models**: 3-model ensemble (Logistic Regression, Random Forest, XGBoost)
- **Compliance Engine**: AML/KYC rule checking
- **Secure File Upload**: AES-256 encryption
- **Audit Logging**: Complete audit trail

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
cp ../.env.example .env
# Edit .env with your settings
```

### 3. Start Database

```bash
cd ..
docker-compose up -d db redis
```

### 4. Initialize Database

```bash
cd backend
python -c "import asyncio; from app.db.base import init_db; asyncio.run(init_db())"
python scripts/seed_db.py
```

### 5. Train Models

```bash
python scripts/train_models.py
```

### 6. Run Server

```bash
uvicorn app.main:app --reload
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

```bash
pytest
```

## Project Structure

```
backend/
├── app/
│   ├── api/           # API routes
│   ├── core/          # Core business logic
│   │   ├── security/  # Auth, encryption
│   │   ├── ml/        # ML models
│   │   ├── compliance/# AML/KYC
│   │   └── data/      # Data loading
│   ├── db/            # Database
│   ├── models/        # SQLAlchemy models
│   └── schemas/       # Pydantic schemas
├── scripts/           # Utility scripts
└── tests/             # Test suite
```
