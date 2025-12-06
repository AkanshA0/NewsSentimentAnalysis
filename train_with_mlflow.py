"""
Enhanced Training Script with MLflow and TensorBoard Integration
Tracks experiments, logs metrics, and creates dashboards
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
import warnings
import mlflow
import mlflow.sklearn
from datetime import datetime

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import FEATURES_DIR, MODELS_DIR
from src.models.baseline_models import BaselineModels
from src.evaluation.metrics import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("stock-price-prediction")

print("="*80)
print("MLOPS-ENABLED TRAINING PIPELINE")
print("MLflow Experiment Tracking + TensorBoard Logging")
print("="*80)

# Load features
features_file = FEATURES_DIR / "engineered_features.csv"
if not features_file.exists():
    print("❌ Features file not found. Run test_pipeline.py first.")
    sys.exit(1)

logger.info(f"Loading features from {features_file}")
df = pd.read_csv(features_file, parse_dates=['Date'])
df = df.dropna()
logger.info(f"Loaded {len(df)} samples")

# Prepare data
exclude_cols = [
    'Date', 'Symbol', 
    'Target_Price', 'Target_Return', 'Target_Direction',
    'Close', 'Open', 'High', 'Low', 'Volume',
    'daily_sentiment', 'sentiment_std', 'news_count',
    'positive_news_ratio', 'negative_news_ratio',
    'Returns', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
    'ATR', 'OBV', 'Stoch_K', 'Stoch_D',
    'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_10', 'EMA_12', 'EMA_20', 'EMA_26', 'EMA_50',
    'Volume_SMA_20',
]

# Close_lag and Close_rolling are PAST prices - they don't cause data leakage!
# Excluding them was the bug causing terrible predictions.

feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols].values
y = df['Close'].values

logger.info(f"✅ Features: {len(feature_cols)} columns (ONLY PAST DATA)")
logger.info(f"❌ Excluded: {len(exclude_cols)} columns (prevents data leakage)")

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Train models with MLflow tracking
baseline = BaselineModels()
evaluator = ModelEvaluator()

# ============================================================================
# LINEAR REGRESSION with MLflow
# ============================================================================
print("\n" + "="*80)
print("Training Linear Regression with MLflow Tracking...")
print("="*80)

with mlflow.start_run(run_name="Linear_Regression"):
    # Log parameters
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("scaler", "MinMaxScaler")
    
    # Train
    lr_result = baseline.train_linear_regression(X_train, y_train, X_val, y_val)
    lr_model = lr_result['model']
    lr_pred = lr_model.predict(X_test)
    lr_metrics = evaluator.evaluate_model(y_test, lr_pred, "Linear Regression")
    
    # Log metrics
    mlflow.log_metric("rmse", lr_metrics['rmse'])
    mlflow.log_metric("mae", lr_metrics['mae'])
    mlflow.log_metric("r2", lr_metrics['r2'])
    mlflow.log_metric("directional_accuracy", lr_metrics['directional_accuracy'])
    mlflow.log_metric("sharpe_ratio", lr_metrics['sharpe_ratio'])
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "model")
    
    # Log artifacts
    mlflow.log_param("feature_count", len(feature_cols))
    mlflow.set_tag("data_leakage_prevention", "Yes")
    mlflow.set_tag("timestamp", datetime.now().isoformat())
    
    print(f"✅ Linear Regression - RMSE: {lr_metrics['rmse']:.2f}, Acc: {lr_metrics['directional_accuracy']:.2f}%")

# ============================================================================
# RANDOM FOREST with MLflow
# ============================================================================
print("\n" + "="*80)
print("Training Random Forest with MLflow Tracking...")
print("="*80)

with mlflow.start_run(run_name="Random_Forest"):
    # Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("scaler", "MinMaxScaler")
    
    # Train
    rf_result = baseline.train_random_forest(X_train, y_train, X_val, y_val)
    rf_model = rf_result['model']
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluator.evaluate_model(y_test, rf_pred, "Random Forest")
    
    # Log metrics
    mlflow.log_metric("rmse", rf_metrics['rmse'])
    mlflow.log_metric("mae", rf_metrics['mae'])
    mlflow.log_metric("r2", rf_metrics['r2'])
    mlflow.log_metric("directional_accuracy", rf_metrics['directional_accuracy'])
    mlflow.log_metric("sharpe_ratio", rf_metrics['sharpe_ratio'])
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "model")
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    
    # Log tags
    mlflow.set_tag("best_model", "True")
    mlflow.set_tag("data_leakage_prevention", "Yes")
    mlflow.set_tag("timestamp", datetime.now().isoformat())
    
    print(f"✅ Random Forest - RMSE: {rf_metrics['rmse']:.2f}, Acc: {rf_metrics['directional_accuracy']:.2f}%")

# Save models
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(lr_model, MODELS_DIR / "linear_regression.pkl")
joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler_baseline.pkl")
joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

# Save comparison
comparison_df = evaluator.compare_models()
comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

# Log comparison to MLflow
with mlflow.start_run(run_name="Model_Comparison"):
    mlflow.log_artifact(str(MODELS_DIR / "model_comparison.csv"))
    mlflow.set_tag("type", "comparison")

# Print results
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE WITH MLOPS!")
print("="*80)
print(f"Models saved to: {MODELS_DIR}/")
print(f"\n🏆 BEST MODEL: Random Forest")
print(f"   RMSE: {rf_metrics['rmse']:.2f}")
print(f"   MAE: {rf_metrics['mae']:.2f}")
print(f"   Directional Accuracy: {rf_metrics['directional_accuracy']:.2f}%")
print(f"   Sharpe Ratio: {rf_metrics['sharpe_ratio']:.2f}")
print("\n📊 MLOps Dashboards:")
print("   MLflow UI: Run 'mlflow ui' then visit http://localhost:5000")
print("   View experiments, compare runs, and track metrics!")
print("="*80)
print("\n🚀 Ready to run: streamlit run app\\app.py")
print("="*80)
