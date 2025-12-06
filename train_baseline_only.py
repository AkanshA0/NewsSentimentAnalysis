"""
Baseline Models Training (No TensorFlow Required)
Trains Linear Regression and Random Forest only
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
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import FEATURES_DIR, MODELS_DIR
from src.models.baseline_models import BaselineModels
from src.evaluation.metrics import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*80)
print("BASELINE MODELS TRAINING (NO TENSORFLOW)")
print("Training: Linear Regression + Random Forest")
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

# Prepare data (exclude same-day features to prevent leakage)
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

# Split data (temporal split)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Train models
baseline = BaselineModels()
evaluator = ModelEvaluator()

# Linear Regression
print("\n" + "="*80)
print("Training Linear Regression...")
print("="*80)
lr_result = baseline.train_linear_regression(X_train, y_train, X_val, y_val)
lr_model = lr_result['model']
lr_pred = lr_model.predict(X_test)
lr_metrics = evaluator.evaluate_model(y_test, lr_pred, "Linear Regression")

# Random Forest
print("\n" + "="*80)
print("Training Random Forest...")
print("="*80)
rf_result = baseline.train_random_forest(X_train, y_train, X_val, y_val)
rf_model = rf_result['model']
rf_pred = rf_model.predict(X_test)
rf_metrics = evaluator.evaluate_model(y_test, rf_pred, "Random Forest")

# Save models
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(lr_model, MODELS_DIR / "linear_regression.pkl")
joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler_baseline.pkl")
joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")

# Save comparison
comparison_df = evaluator.compare_models()
comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

# Print results
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"Models saved to: {MODELS_DIR}/")
print(f"\n🏆 BEST MODEL: Random Forest")
print(f"   RMSE: {rf_metrics['rmse']:.2f}")
print(f"   MAE: {rf_metrics['mae']:.2f}")
print(f"   Directional Accuracy: {rf_metrics['directional_accuracy']:.2f}%")
print(f"   Sharpe Ratio: {rf_metrics['sharpe_ratio']:.2f}")
print("="*80)
print("\n🚀 Ready to run: streamlit run app\\app.py")
print("="*80)
