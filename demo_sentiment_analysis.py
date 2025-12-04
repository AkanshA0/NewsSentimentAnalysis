"""
Enhanced News Collection and Sentiment Analysis Demo
Collects more news and shows actual sentiment scores
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_collection.news_collector import NewsCollector
from src.preprocessing.feature_engineer import FeatureEngineer
from src.utils.config import STOCK_SYMBOLS, RAW_DATA_DIR

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_enhanced_news():
    """Collect more news articles."""
    logger.info("="*80)
    logger.info("ENHANCED NEWS COLLECTION")
    logger.info("="*80)
    
    # Initialize collector
    collector = NewsCollector(symbols=STOCK_SYMBOLS)
    
    # Collect more articles (50 per source = 150 per stock)
    logger.info("Collecting 50 articles per source (may take a few minutes)...")
    news_df = collector.collect_all_news(max_articles_per_source=50)
    
    # Print summary
    print("\n" + "="*80)
    print("NEWS COLLECTION SUMMARY")
    print("="*80)
    summary = collector.get_news_summary()
    print(summary.to_string(index=False))
    print("="*80)
    
    # Save news
    collector.save_news(RAW_DATA_DIR)
    
    # Show sample articles
    print("\n" + "="*80)
    print("SAMPLE NEWS ARTICLES")
    print("="*80)
    for symbol in STOCK_SYMBOLS:
        symbol_news = news_df[news_df['symbol'] == symbol].head(3)
        print(f"\n{symbol} - Recent Articles:")
        for idx, row in symbol_news.iterrows():
            print(f"  • {row['title'][:80]}...")
            print(f"    Source: {row['source']}, Date: {row['published_date']}")
    print("="*80)
    
    return news_df


def analyze_sentiment(news_df):
    """Analyze sentiment using FinBERT and TextBlob."""
    logger.info("\n" + "="*80)
    logger.info("SENTIMENT ANALYSIS WITH FINBERT")
    logger.info("="*80)
    
    # Initialize feature engineer with FinBERT
    logger.info("Loading FinBERT model (this may take a moment)...")
    engineer = FeatureEngineer(use_finbert=True)
    
    # Analyze sentiment for first 20 articles (for demo)
    sample_news = news_df.head(20).copy()
    logger.info(f"Analyzing sentiment for {len(sample_news)} sample articles...")
    
    news_with_sentiment = engineer.add_sentiment_features(sample_news)
    
    # Display results
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*80)
    
    for idx, row in news_with_sentiment.iterrows():
        print(f"\nArticle: {row['title'][:70]}...")
        print(f"Symbol: {row['symbol']}")
        print(f"Sentiment Scores:")
        print(f"  • TextBlob:  {row['sentiment_textblob']:+.3f}")
        print(f"  • FinBERT:   {row['sentiment_finbert']:+.3f}")
        print(f"  • Combined:  {row['sentiment_score']:+.3f}")
        
        # Interpret sentiment
        score = row['sentiment_score']
        if score > 0.1:
            interpretation = "📈 POSITIVE (Bullish)"
        elif score < -0.1:
            interpretation = "📉 NEGATIVE (Bearish)"
        else:
            interpretation = "➡️ NEUTRAL"
        print(f"  → {interpretation}")
    
    print("\n" + "="*80)
    print("SENTIMENT STATISTICS")
    print("="*80)
    print(f"Average Sentiment: {news_with_sentiment['sentiment_score'].mean():+.3f}")
    print(f"Sentiment Std Dev: {news_with_sentiment['sentiment_score'].std():.3f}")
    print(f"Most Positive:     {news_with_sentiment['sentiment_score'].max():+.3f}")
    print(f"Most Negative:     {news_with_sentiment['sentiment_score'].min():+.3f}")
    
    # Sentiment distribution
    positive = (news_with_sentiment['sentiment_score'] > 0.1).sum()
    negative = (news_with_sentiment['sentiment_score'] < -0.1).sum()
    neutral = len(news_with_sentiment) - positive - negative
    
    print(f"\nSentiment Distribution:")
    print(f"  Positive: {positive} ({positive/len(news_with_sentiment)*100:.1f}%)")
    print(f"  Neutral:  {neutral} ({neutral/len(news_with_sentiment)*100:.1f}%)")
    print(f"  Negative: {negative} ({negative/len(news_with_sentiment)*100:.1f}%)")
    print("="*80)
    
    return news_with_sentiment


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("ENHANCED NEWS COLLECTION & SENTIMENT ANALYSIS")
    print("="*80)
    print("Configuration:")
    print(f"  • Stocks: {', '.join(STOCK_SYMBOLS)}")
    print(f"  • News Period: 1 year (practical for scraping)")
    print(f"  • Articles per source: 50 (150 total per stock)")
    print(f"  • Sentiment Models: FinBERT + TextBlob")
    print("="*80)
    
    # Step 1: Collect news
    news_df = collect_enhanced_news()
    
    # Step 2: Analyze sentiment
    if len(news_df) > 0:
        news_with_sentiment = analyze_sentiment(news_df)
        
        # Save results
        output_file = RAW_DATA_DIR / "news_with_sentiment_sample.csv"
        news_with_sentiment.to_csv(output_file, index=False)
        logger.info(f"\n💾 Saved sentiment analysis results to {output_file}")
    else:
        logger.warning("No news collected, skipping sentiment analysis")
    
    print("\n✅ Enhanced news collection and sentiment analysis complete!")


if __name__ == "__main__":
    main()
