# MLOps Pipeline Documentation
Data (DVC) ───▶ DVC Versioning
   │
   ▼
Train Model
   │
   ▼
Evaluate Model
   │
   ▼
Register Model (MLflow)
   │
   ▼
Deploy (K8s / HPA)
   │
   ▼
Monitoring (Logs / OTEL)
   │
   ▼
Dashboard (Streamlit)
   │
   ▼
Drift Detection (PSI / KS)
   │
   ▼
Retrain (Champion / Challenger)
   │
   ▼
A/B Testing
   │
   ▼
Register Model (MLflow)  ───▶ (loop continues)



