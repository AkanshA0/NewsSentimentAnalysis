"""
Generate Demo-Ready Sentiment Data
Creates realistic sentiment timeline for presentation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

print("="*80)
print("GENERATING DEMO-READY SENTIMENT DATA")
print("Creating realistic sentiment timeline for presentation")
print("="*80)

# Load engineered features
features_file = Path("data/features/engineered_features.csv")
df = pd.read_csv(features_file, parse_dates=['Date'])

print(f"\n📊 Loaded {len(df)} data points")

# Generate realistic sentiment data
np.random.seed(42)  # Reproducible

timeline_data = []

for _, row in df.iterrows():
    # Generate realistic sentiment based on stock performance
    # Positive correlation with returns
    base_sentiment = row.get('Returns_lag_1', 0) * 2  # Slight correlation
    
    # Add realistic noise and variation
    noise = np.random.normal(0, 0.15)
    sentiment = base_sentiment + noise
    
    # Clip to realistic range
    sentiment = np.clip(sentiment, -0.3, 0.3)
    
    # Generate news count (1-10 articles per day)
    news_count = np.random.poisson(3) + 1  # Average 3-4 articles
    
    # Calculate ratios based on sentiment
    if sentiment > 0.05:
        positive_ratio = 0.6 + np.random.uniform(0, 0.3)
        negative_ratio = 0.1 + np.random.uniform(0, 0.2)
    elif sentiment < -0.05:
        positive_ratio = 0.1 + np.random.uniform(0, 0.2)
        negative_ratio = 0.6 + np.random.uniform(0, 0.3)
    else:
        positive_ratio = 0.4 + np.random.uniform(0, 0.2)
        negative_ratio = 0.3 + np.random.uniform(0, 0.2)
    
    # Ensure ratios sum sensibly
    positive_ratio = min(positive_ratio, 0.9)
    negative_ratio = min(negative_ratio, 1 - positive_ratio)
    
    # Sentiment std dev
    sentiment_std = abs(sentiment) * 0.3 + np.random.uniform(0.05, 0.15)
    
    timeline_data.append({
        'Date': row['Date'],
        'Symbol': row['Symbol'],
        'daily_sentiment': round(sentiment, 4),
        'sentiment_std': round(sentiment_std, 4),
        'news_count': int(news_count),
        'positive_news_ratio': round(positive_ratio, 3),
        'negative_news_ratio': round(negative_ratio, 3),
    })

# Create DataFrame
timeline_df = pd.DataFrame(timeline_data)

# Save
output_file = Path("data/features/sentiment_timeline.csv")
timeline_df.to_csv(output_file, index=False)

print(f"\n✅ Generated {len(timeline_df)} sentiment data points")
print(f"📁 Saved to: {output_file}")

# Show statistics
print("\n" + "="*80)
print("SENTIMENT STATISTICS")
print("="*80)

for symbol in ['AAPL', 'GOOGL', 'TSLA', 'NVDA']:
    symbol_data = timeline_df[timeline_df['Symbol'] == symbol]
    print(f"\n{symbol}:")
    print(f"  Average Sentiment: {symbol_data['daily_sentiment'].mean():.4f}")
    print(f"  Min Sentiment: {symbol_data['daily_sentiment'].min():.4f}")
    print(f"  Max Sentiment: {symbol_data['daily_sentiment'].max():.4f}")
    print(f"  Positive Days: {(symbol_data['daily_sentiment'] > 0).sum()} ({(symbol_data['daily_sentiment'] > 0).mean()*100:.1f}%)")
    print(f"  Negative Days: {(symbol_data['daily_sentiment'] < 0).sum()} ({(symbol_data['daily_sentiment'] < 0).mean()*100:.1f}%)")
    print(f"  Total News Articles: {symbol_data['news_count'].sum()}")

print("\n" + "="*80)
print("SAMPLE DATA")
print("="*80)
print(timeline_df.head(10))

print("\n" + "="*80)
print("✅ DEMO DATA READY!")
print("="*80)
print("\nYour app will now show:")
print("  • Sentiment timeline chart with realistic data")
print("  • Positive/negative sentiment distribution")
print("  • Correlation with stock price movements")
print("\nThis is DEMO DATA for presentation purposes.")
print("For production, collect real news data.")
print("="*80)
