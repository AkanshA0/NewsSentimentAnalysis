"""
Quick script to check what features are being used for training
"""

import pandas as pd
from pathlib import Path

# Load features
features_file = Path("data/features/engineered_features.csv")
df = pd.read_csv(features_file, parse_dates=['Date'])

print("="*80)
print("FEATURE ANALYSIS")
print("="*80)

# Define exclusions
exclude_cols = [
    'Date', 'Symbol', 
    'Target_Price', 'Target_Return', 'Target_Direction',
    'Close', 'Open', 'High', 'Low', 'Volume',
    'daily_sentiment', 'sentiment_std', 'news_count',
    'positive_news_ratio', 'negative_news_ratio',
]

close_lag_cols = [col for col in df.columns if 'Close_lag' in col or 'Close_rolling' in col]
exclude_cols.extend(close_lag_cols)

feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\nTotal columns: {len(df.columns)}")
print(f"Excluded columns: {len(exclude_cols)}")
print(f"Feature columns: {len(feature_cols)}")

print("\n" + "="*80)
print("FEATURES BEING USED:")
print("="*80)
for i, col in enumerate(feature_cols, 1):
    print(f"{i:2d}. {col}")

print("\n" + "="*80)
print("EXCLUDED COLUMNS:")
print("="*80)
for i, col in enumerate(exclude_cols, 1):
    print(f"{i:2d}. {col}")

# Check for potential leakage
print("\n" + "="*80)
print("POTENTIAL LEAKAGE CHECK:")
print("="*80)

suspicious = []
for col in feature_cols:
    # Check if any feature might contain same-day information
    if any(x in col.lower() for x in ['rsi', 'macd', 'bb_', 'atr', 'obv', 'stoch', 'sma', 'ema']):
        suspicious.append(col)

if suspicious:
    print(f"\n⚠️  WARNING: {len(suspicious)} technical indicators found!")
    print("These are calculated from same-day OHLCV and may cause leakage:")
    for col in suspicious[:10]:
        print(f"  - {col}")
    print("\nThese should be LAGGED or calculated from PAST data only!")
else:
    print("✅ No obvious leakage detected")

print("="*80)
