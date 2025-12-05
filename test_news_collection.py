"""
Test news collection to see what's actually being fetched
"""

from src.data_collection.news_collector import NewsCollector
from src.preprocessing.feature_engineer import FeatureEngineer

print("="*80)
print("TESTING NEWS COLLECTION")
print("="*80)

# Test for AAPL
symbol = "AAPL"
print(f"\nTesting news collection for {symbol}...")

collector = NewsCollector(symbols=[symbol])
news_df = collector.collect_all_news(max_articles_per_source=10)

print(f"\n✅ Collected {len(news_df)} articles")

if len(news_df) > 0:
    print("\nSample articles:")
    for idx, row in news_df.head(5).iterrows():
        print(f"\n{idx+1}. {row['title']}")
        print(f"   Source: {row['source']}")
        print(f"   Date: {row['published_date']}")
    
    # Test sentiment analysis
    print("\n" + "="*80)
    print("TESTING SENTIMENT ANALYSIS")
    print("="*80)
    
    engineer = FeatureEngineer(use_finbert=False)
    news_with_sentiment = engineer.add_sentiment_features(news_df)
    
    print(f"\nSentiment scores:")
    for idx, row in news_with_sentiment.head(5).iterrows():
        print(f"\n{idx+1}. {row['title'][:60]}...")
        print(f"   Sentiment: {row['sentiment_score']:.3f}")
    
    avg_sentiment = news_with_sentiment['sentiment_score'].mean()
    print(f"\n📊 Average sentiment: {avg_sentiment:.3f}")
    
else:
    print("\n❌ No articles collected!")
    print("\nPossible issues:")
    print("1. Yahoo Finance: 404 errors (known issue)")
    print("2. Google News: May not have recent articles")
    print("3. Finviz: Might be blocking requests")
    print("\nThis is why sentiment is showing as 0 in the app.")

print("\n" + "="*80)
