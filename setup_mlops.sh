#!/bin/bash

# MLOps Setup Script
# Installs MLflow and TensorBoard, then launches dashboards

echo "================================"
echo "MLOps Tools Setup"
echo "================================"

# Install dependencies
echo "Installing MLflow and TensorBoard..."
pip install mlflow tensorboard

# Create directories
mkdir -p mlruns
mkdir -p logs/tensorboard

echo ""
echo "✅ MLOps tools installed!"
echo ""
echo "================================"
echo "Quick Start Commands"
echo "================================"
echo ""
echo "1. Train models with MLflow tracking:"
echo "   python train_with_mlflow.py"
echo ""
echo "2. View MLflow dashboard:"
echo "   mlflow ui"
echo "   Then open: http://localhost:5000"
echo ""
echo "3. View TensorBoard (if LSTM trained):"
echo "   tensorboard --logdir=logs/tensorboard"
echo "   Then open: http://localhost:6006"
echo ""
echo "================================"
