"""
Diagnostic Script - Check Model Predictions
"""

import pandas as pd
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.config import FEATURES_DIR, MODELS_DIR

print("="*80)
print("MODEL PREDICTION DIAGNOSTIC")
print("="*80)

# Load data
df = pd.read_csv(FEATURES_DIR / "engineered_features.csv", parse_dates=['Date'])
print(f"\n📅 Data Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"📊 Total Rows: {len(df)}")

# Check GOOGL
googl_data = df[df['Symbol'] == 'GOOGL'].sort_values('Date')
print(f"\n🔍 GOOGL Data:")
print(f"   Last Date: {googl_data['Date'].iloc[-1].date()}")
print(f"   Last Price: ${googl_data['Close'].iloc[-1]:.2f}")
print(f"   Price Range: ${googl_data['Close'].min():.2f} - ${googl_data['Close'].max():.2f}")

# Load model
model = joblib.load(MODELS_DIR / "random_forest.pkl")
scaler = joblib.load(MODELS_DIR / "scaler_baseline.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

print(f"\n🤖 Model Info:")
print(f"   Type: {type(model).__name__}")
print(f"   Features: {len(feature_cols)}")

# Check sentiment features
sentiment_features = [f for f in feature_cols if 'sentiment' in f.lower()]
print(f"   Sentiment Features: {len(sentiment_features)}")
print(f"   Examples: {sentiment_features[:5]}")

# Make prediction
latest_row = googl_data[feature_cols].tail(1).values
latest_scaled = scaler.transform(latest_row)
prediction = model.predict(latest_scaled)[0]

actual_last = googl_data['Close'].iloc[-1]
print(f"\n🎯 Prediction:")
print(f"   Input (last price): ${actual_last:.2f}")
print(f"   Predicted (next day): ${prediction:.2f}")
print(f"   Difference: ${prediction - actual_last:.2f} ({((prediction - actual_last)/actual_last)*100:.2f}%)")

# Check if sentiment is being used
print(f"\n💭 Sentiment Check:")
for feat in sentiment_features[:3]:
    if feat in googl_data.columns:
        val = googl_data[feat].iloc[-1]
        print(f"   {feat}: {val:.4f}")

# Feature importance
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 10 Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

# Check for issues
days_old = (pd.Timestamp.now() - googl_data['Date'].max()).days
if days_old > 30:
    print(f"⚠️  Data is {days_old} days old - predictions may not match current prices!")
    print(f"   Solution: Run 'python test_pipeline.py' to fetch latest data")

if len(sentiment_features) == 0:
    print("⚠️  No sentiment features found in model!")
else:
    print(f"✅ Model uses {len(sentiment_features)} sentiment features")

if abs(prediction - actual_last) / actual_last > 0.1:
    print(f"⚠️  Large prediction difference (>{10}%)")
    print(f"   This is normal for volatile stocks or old data")
else:
    print(f"✅ Prediction difference is reasonable (<10%)")

print("="*80)
