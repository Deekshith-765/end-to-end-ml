#!/bin/bash

echo "Starting MLOps Pipeline..."

echo " Training  model..."
python3 train.py

echo "  Running pipeline..."
python3 main.py

echo " Pipeline finished!"
