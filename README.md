# MLOps Pipeline Documentation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐         │
│    │   Data  │────▶│  Train   │────▶│ Evaluate │────▶│ Register │         │
│    │  (DVC)  │     │  Model   │     │  Model   │     │ (MLflow) │         │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘         │
│         │                                                        │           │
│         ▼                                                        ▼           │
│    ┌──────────┐                                           ┌──────────┐       │
│    │  DVC     │                                           │  Deploy  │       │
│    │ Version  │                                           │ K8s/HPA  │       │
│    └──────────┘                                           └──────────┘       │
│                                                                     │          │
│         ┌───────────────────────────────────────────────────────────┘          │
│         ▼                                                                     │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐                             │
│    │  Drift  │────▶│ Retrain  │────▶│  A/B     │                             │
│    │ Detect  │     │ (Champion│     │  Testing │                             │
│    │ PSI/KS  │     │Challenger)│     │          │                             │
│    └──────────┘     └──────────┘     └──────────┘                             │
│         │                │                                                     │
│         ▼                ▼                                                     │
│    ┌──────────┐     ┌──────────┐                                               │
│    │Dashboard │◀────│ Monitoring│                                              │
│    │Streamlit │     │ Logs/OTEL │                                              │
│    └──────────┘     └──────────┘                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Versioning (DVC)
- Track datasets like Git tracks code
- Reproducible pipelines
- Remote storage support

```bash
dvc init
dvc add data/train.csv
dvc push
```

### 2. Drift Detection
- **Data Drift**: Input distribution changes (PSI, KS Test, KL Divergence)
- **Concept Drift**: Model relationship changes (accuracy degradation)

Run: `python detect.py`

### 3. Model Training
- Scikit-learn based training
- MLflow experiment tracking

Run: `python train.py`

### 4. Model Registry (MLflow)
- Version control for models
- Stage management (Staging → Production)
- Model comparison

Run: `python registry.py`

### 5. Auto-Retraining (Champion vs Challenger)
- Automatic retraining on drift detection
- Performance comparison before deployment

Run: `python retrain.py`

### 6. Monitoring Dashboard
- Streamlit-based real-time monitoring
- Accuracy, drift status, predictions

Run: `streamlit run dashboard.py`

### 7. API with Security
- FastAPI-based prediction service
- JWT authentication
- A/B testing support

### 8. Kubernetes Deployment
- Health checks (liveness, readiness)
- Horizontal Pod Autoscaling
- Rolling updates

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init

# Train model
python train.py

# Detect drift
python detect.py

# Retrain if needed
python retrain.py

# Start dashboard
streamlit run dashboard.py

# Start API
uvicorn app:app --reload
```

## CI/CD Pipeline

GitHub Actions automatically:
1. Runs tests and linting
2. Trains model on code changes
3. Detects drift
4. Deploys to staging
5. Promotes to production

## Environment Variables

```env
MLFLOW_TRACKING_URI=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
SLACK_WEBHOOK=...
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/predict` | POST | Make prediction |
| `/predict_ab` | POST | A/B test prediction |
| `/metrics` | GET | Model metrics |
# end-to-end-ml
