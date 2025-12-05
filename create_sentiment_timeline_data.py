import pandas as pd
from pathlib import Path

# Load features
features_file = Path("data/features/engineered_features.csv")
df = pd.read_csv(features_file, parse_dates=['Date'])

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Check for sentiment columns
sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower() or 'news' in col.lower()]
print(f"\nSentiment-related columns: {sentiment_cols}")

# Create timeline with Date, Symbol, and any sentiment column
timeline_cols = ['Date', 'Symbol']

# Add Sentiment_lag_1 as daily_sentiment for visualization
if 'Sentiment_lag_1' in df.columns:
    df['daily_sentiment'] = df['Sentiment_lag_1']
    timeline_cols.append('daily_sentiment')
    print("\n✅ Using Sentiment_lag_1 as daily_sentiment")

# Add other sentiment columns if they exist
for col in ['sentiment_std', 'news_count', 'positive_news_ratio', 'negative_news_ratio']:
    if col in df.columns:
        timeline_cols.append(col)

timeline_df = df[timeline_cols].copy()
timeline_df = timeline_df.fillna(0)

# Save
output_file = Path("data/features/sentiment_timeline.csv")
timeline_df.to_csv(output_file, index=False)

print(f"\n✅ Saved to {output_file}")
print(f"Rows: {len(timeline_df)}")
print(f"\nSample:")
print(timeline_df.head(10))
print(f"\nSentiment stats:")
if 'daily_sentiment' in timeline_df.columns:
    print(f"  Min: {timeline_df['daily_sentiment'].min()}")
    print(f"  Max: {timeline_df['daily_sentiment'].max()}")
    print(f"  Mean: {timeline_df['daily_sentiment'].mean()}")
    print(f"  Non-zero values: {(timeline_df['daily_sentiment'] != 0).sum()}")
