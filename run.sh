#!/bin/bash

set -e

echo "========================================="
echo "      MLOps Pipeline Runner"
echo "========================================="

cd "$(dirname "$0")"

source ~/venv/bin/activate

echo ""
echo ">>> 1: Training model..."
python train.py

echo ""
echo ">>>  2: Running drift detection..."
python detect.py

DRIFT_STATUS=$(grep -o "data_drift=True\|concept_drift=True" drift_status.txt 2>/dev/null || echo "")

if [ -n "$DRIFT_STATUS" ]; then
    echo ""
    echo ">>> Drift detected - retraining..."
    python retrain.py
else
    echo ""
    echo ">>> No drift detected"
fi

echo ""
echo ">>> 3: Registering model with MLflow..."
python registry.py 2>/dev/null || echo "MLflow skipped"

echo ""
echo "========================================="
echo " Pipeline completed!"
echo "========================================="
