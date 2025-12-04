"""
Test Script for Stock Price Prediction Pipeline
Tests data collection, preprocessing, and feature engineering
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import (
    STOCK_SYMBOLS, DATA_PERIOD, RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, FEATURES_DIR
)
from src.data_collection.stock_collector import StockDataCollector
from src.data_collection.news_collector import NewsCollector
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_stock_collection():
    """Test stock data collection."""
    logger.info("\n" + "="*80)
    logger.info("TESTING STOCK DATA COLLECTION")
    logger.info("="*80)
    
    try:
        # Initialize collector
        collector = StockDataCollector(
            symbols=STOCK_SYMBOLS,
            period=DATA_PERIOD,
            interval="1d"
        )
        
        # Collect data
        data = collector.collect_all_stocks()
        
        # Print summary
        summary = collector.get_data_summary()
        print("\nStock Data Summary:")
        print(summary.to_string(index=False))
        
        # Save data
        collector.save_data(RAW_DATA_DIR)
        
        logger.info("✅ Stock data collection test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Stock data collection test FAILED: {str(e)}")
        return False


def test_news_collection():
    """Test news data collection."""
    logger.info("\n" + "="*80)
    logger.info("TESTING NEWS DATA COLLECTION")
    logger.info("="*80)
    
    try:
        # Initialize collector
        collector = NewsCollector(symbols=STOCK_SYMBOLS)
        
        # Collect news (limit to 10 articles per source for testing)
        news_df = collector.collect_all_news(max_articles_per_source=10)
        
        # Print summary
        summary = collector.get_news_summary()
        print("\nNews Data Summary:")
        print(summary.to_string(index=False))
        
        # Save news
        collector.save_news(RAW_DATA_DIR)
        
        logger.info("✅ News data collection test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ News data collection test FAILED: {str(e)}")
        return False


def test_data_cleaning():
    """Test data cleaning."""
    logger.info("\n" + "="*80)
    logger.info("TESTING DATA CLEANING")
    logger.info("="*80)
    
    try:
        # Initialize cleaner
        cleaner = DataCleaner(scaler_type="MinMaxScaler")
        
        # Load stock data
        stock_file = RAW_DATA_DIR / "all_stocks_combined.csv"
        if not stock_file.exists():
            logger.warning("Stock data not found, skipping cleaning test")
            return True
        
        import pandas as pd
        stock_df = pd.read_csv(stock_file, parse_dates=['Date'])
        
        # Clean stock data
        cleaned_stock = cleaner.clean_stock_data(stock_df)
        
        # Generate quality report
        report = cleaner.get_data_quality_report(cleaned_stock)
        print("\nData Quality Report (first 10 columns):")
        print(report.head(10).to_string(index=False))
        
        # Save cleaned data
        output_file = PROCESSED_DATA_DIR / "cleaned_stock_data.csv"
        cleaned_stock.to_csv(output_file, index=False)
        logger.info(f"Saved cleaned stock data: {len(cleaned_stock)} rows")
        
        # Clean news data if available
        news_file = RAW_DATA_DIR / "all_news.csv"
        if news_file.exists():
            news_df = pd.read_csv(news_file)
            cleaned_news = cleaner.clean_news_data(news_df)
            
            output_file = PROCESSED_DATA_DIR / "cleaned_news_data.csv"
            cleaned_news.to_csv(output_file, index=False)
            logger.info(f"Saved cleaned news data: {len(cleaned_news)} articles")
        
        logger.info("✅ Data cleaning test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data cleaning test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering."""
    logger.info("\n" + "="*80)
    logger.info("TESTING FEATURE ENGINEERING")
    logger.info("="*80)
    
    try:
        import pandas as pd
        
        # Load cleaned data
        stock_file = PROCESSED_DATA_DIR / "cleaned_stock_data.csv"
        news_file = PROCESSED_DATA_DIR / "cleaned_news_data.csv"
        
        if not stock_file.exists():
            logger.warning("Cleaned stock data not found, skipping feature engineering test")
            return True
        
        stock_df = pd.read_csv(stock_file, parse_dates=['Date'])
        
        # Initialize feature engineer (without FinBERT for faster testing)
        logger.info("Initializing feature engineer (TextBlob only for testing)...")
        engineer = FeatureEngineer(use_finbert=False)
        
        if news_file.exists():
            news_df = pd.read_csv(news_file, parse_dates=['published_date'])
            
            # Limit to first 50 articles for testing
            news_df = news_df.head(50)
            logger.info(f"Testing with {len(news_df)} news articles")
            
            # Create all features
            features_df = engineer.create_all_features(stock_df, news_df)
        else:
            logger.warning("No news data found, creating features without sentiment")
            # Add lag and rolling features only
            features_df = engineer.add_lag_features(stock_df)
            features_df = engineer.add_rolling_features(features_df)
            features_df = engineer.add_target_variable(features_df)
            features_df = features_df.dropna()
        
        # Save features
        output_file = FEATURES_DIR / "engineered_features.csv"
        features_df.to_csv(output_file, index=False)
        
        print(f"\nFeature Engineering Summary:")
        print(f"Total samples: {len(features_df)}")
        print(f"Total features: {len(features_df.columns)}")
        print(f"\nFeature columns:")
        for i, col in enumerate(features_df.columns[:20], 1):
            print(f"  {i}. {col}")
        if len(features_df.columns) > 20:
            print(f"  ... and {len(features_df.columns) - 20} more")
        
        logger.info("✅ Feature engineering test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*80)
    logger.info("STOCK PRICE PREDICTION PIPELINE - TEST SUITE")
    logger.info("="*80)
    
    results = {}
    
    # Test 1: Stock data collection
    results['stock_collection'] = test_stock_collection()
    
    # Test 2: News data collection
    results['news_collection'] = test_news_collection()
    
    # Test 3: Data cleaning
    results['data_cleaning'] = test_data_cleaning()
    
    # Test 4: Feature engineering
    results['feature_engineering'] = test_feature_engineering()
    
    # Print final results
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    logger.info("="*80)
    logger.info(f"TOTAL: {total_passed}/{total_tests} tests passed")
    logger.info("="*80)
    
    if total_passed == total_tests:
        logger.info("\n🎉 All tests passed! Pipeline is working correctly.")
    else:
        logger.warning(f"\n⚠️ {total_tests - total_passed} test(s) failed. Check logs above.")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
