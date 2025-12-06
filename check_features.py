"""
Quick check - what features are actually in the saved model?
"""
import joblib
from pathlib import Path

feature_cols = joblib.load('models/feature_cols.pkl')
print(f"Total features: {len(feature_cols)}")
print(f"\nFeature list:")
for i, feat in enumerate(feature_cols, 1):
    print(f"{i}. {feat}")

# Check for Close_lag
close_features = [f for f in feature_cols if 'Close' in f]
print(f"\n\nClose-related features: {len(close_features)}")
print(close_features)
