"""
Feature Engineering Module
Creates advanced features for stock price prediction including sentiment analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Setup logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PyTorch/FinBERT, but don't fail if DLL error
FINBERT_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    FINBERT_AVAILABLE = True
    logger.info("✅ FinBERT available")
except (OSError, ImportError) as e:
    logger.warning(f"⚠️  FinBERT not available (PyTorch DLL error): {str(e)[:100]}")
    logger.info("📊 Using TextBlob-only sentiment analysis")


class FeatureEngineer:
    """
    Creates features for stock price prediction.
    
    Features:
    - Sentiment analysis (FinBERT for financial news, TextBlob for backup)
    - Technical indicators (already calculated in stock_collector)
    - Lag features
    - Rolling statistics
    - News aggregation features
    """
    
    def __init__(self, use_finbert: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            use_finbert: Whether to use FinBERT (requires more memory)
        """
        self.use_finbert = use_finbert and FINBERT_AVAILABLE
        
        # Initialize sentiment analyzers
        logger.info("Initializing sentiment analyzers...")
        
        # FinBERT (deep learning, specifically trained for financial text)
        if self.use_finbert:
            try:
                logger.info("Loading FinBERT model (this may take a moment)...")
                self.finbert = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("✅ FinBERT initialized")
            except Exception as e:
                logger.warning(f"Could not load FinBERT: {str(e)}")
                logger.warning("Falling back to TextBlob only")
                self.use_finbert = False
        else:
            logger.info("📊 Using TextBlob-only (FinBERT disabled)")
        
        logger.info("✅ TextBlob initialized (always available)")
    
    def analyze_sentiment_textblob(self, text: str) -> float:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def analyze_sentiment_finbert(self, text: str) -> float:
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1)
        """
        if not self.use_finbert:
            return 0.0
        
        try:
            # Truncate text to max length
            text = text[:512]
            
            result = self.finbert(text)[0]
            label = result['label'].lower()
            score = result['score']
            
            # Convert to -1 to 1 scale
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:  # neutral
                return 0.0
        except:
            return 0.0
    
    def add_sentiment_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment scores to news data using FinBERT and TextBlob.
        
        Args:
            news_df: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment scores
        """
        logger.info(f"Analyzing sentiment for {len(news_df)} articles...")
        
        df = news_df.copy()
        
        # Initialize sentiment columns
        df['sentiment_textblob'] = 0.0
        df['sentiment_finbert'] = 0.0
        
        # Analyze each article
        for idx, row in df.iterrows():
            text = row['text']
            
            # TextBlob (fast, general-purpose)
            df.at[idx, 'sentiment_textblob'] = self.analyze_sentiment_textblob(text)
            
            # FinBERT (accurate for financial news)
            if self.use_finbert:
                df.at[idx, 'sentiment_finbert'] = self.analyze_sentiment_finbert(text)
            
            # Progress logging
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} articles")
        
        # Create ensemble sentiment score (average of both methods)
        if self.use_finbert:
            df['sentiment_score'] = (
                df['sentiment_textblob'] + 
                df['sentiment_finbert']
            ) / 2
        else:
            # If FinBERT failed to load, use only TextBlob
            df['sentiment_score'] = df['sentiment_textblob']
        
        logger.info(f"✅ Sentiment analysis complete")
        logger.info(f"   Average sentiment: {df['sentiment_score'].mean():.3f}")
        logger.info(f"   Sentiment std: {df['sentiment_score'].std():.3f}")
        
        return df
    
    def aggregate_daily_sentiment(
        self, 
        news_df: pd.DataFrame,
        stock_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment by day and merge with stock data.
        
        Args:
            news_df: DataFrame with news and sentiment scores
            stock_df: DataFrame with stock data
            
        Returns:
            DataFrame with daily sentiment features
        """
        logger.info("Aggregating daily sentiment...")
        
        # Ensure date columns are datetime and handle missing columns
        if 'published_date' in news_df.columns:
            news_df = news_df.copy()
            news_df['published_date'] = pd.to_datetime(news_df['published_date'], errors='coerce')
            news_df['date'] = news_df['published_date'].dt.date
        
        if 'Date' in stock_df.columns:
            stock_df = stock_df.copy()
            stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
            stock_df['date'] = stock_df['Date'].dt.date
        
        result_data = []
        
        for symbol in stock_df['Symbol'].unique():
            symbol_stock = stock_df[stock_df['Symbol'] == symbol].copy()
            symbol_news = news_df[news_df['symbol'] == symbol].copy()
            
            for idx, row in symbol_stock.iterrows():
                date = row['date']
                
                # Get news for this date and previous 3 days (news can affect future prices)
                date_range = pd.date_range(end=date, periods=3).date
                relevant_news = symbol_news[symbol_news['date'].isin(date_range)]
                
                # Calculate sentiment features
                if len(relevant_news) > 0:
                    row['daily_sentiment'] = relevant_news['sentiment_score'].mean()
                    row['sentiment_std'] = relevant_news['sentiment_score'].std()
                    row['news_count'] = len(relevant_news)
                    row['positive_news_ratio'] = (relevant_news['sentiment_score'] > 0.1).sum() / len(relevant_news)
                    row['negative_news_ratio'] = (relevant_news['sentiment_score'] < -0.1).sum() / len(relevant_news)
                else:
                    row['daily_sentiment'] = 0.0
                    row['sentiment_std'] = 0.0
                    row['news_count'] = 0
                    row['positive_news_ratio'] = 0.0
                    row['negative_news_ratio'] = 0.0
                
                result_data.append(row)
        
        result_df = pd.DataFrame(result_data)
        result_df = result_df.drop('date', axis=1)
        
        logger.info(f"✅ Daily sentiment aggregated: {len(result_df)} rows")
        
        return result_df
    
    def add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 3, 5, 7]) -> pd.DataFrame:
        """
        Add lag features for time series prediction.
        
        Args:
            df: DataFrame with stock data
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        logger.info(f"Adding lag features: {lags}")
        
        df = df.copy()
        
        for symbol in df['Symbol'].unique():
            mask = df['Symbol'] == symbol
            
            for lag in lags:
                # Price lags
                df.loc[mask, f'Close_lag_{lag}'] = df.loc[mask, 'Close'].shift(lag)
                df.loc[mask, f'Returns_lag_{lag}'] = df.loc[mask, 'Returns'].shift(lag)
                
                # Sentiment lags
                if 'daily_sentiment' in df.columns:
                    df.loc[mask, f'Sentiment_lag_{lag}'] = df.loc[mask, 'daily_sentiment'].shift(lag)
        
        logger.info(f"✅ Lag features added")
        
        return df
    
    def add_rolling_features(
        self, 
        df: pd.DataFrame, 
        windows: List[int] = [3, 7, 14, 30]
    ) -> pd.DataFrame:
        """
        Add rolling window features.
        
        Args:
            df: DataFrame with stock data
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        logger.info(f"Adding rolling features: {windows}")
        
        df = df.copy()
        
        for symbol in df['Symbol'].unique():
            mask = df['Symbol'] == symbol
            
            for window in windows:
                # Rolling price statistics
                df.loc[mask, f'Close_rolling_mean_{window}'] = df.loc[mask, 'Close'].rolling(window).mean()
                df.loc[mask, f'Close_rolling_std_{window}'] = df.loc[mask, 'Close'].rolling(window).std()
                df.loc[mask, f'Volume_rolling_mean_{window}'] = df.loc[mask, 'Volume'].rolling(window).mean()
                
                # Rolling sentiment statistics
                if 'daily_sentiment' in df.columns:
                    df.loc[mask, f'Sentiment_rolling_mean_{window}'] = df.loc[mask, 'daily_sentiment'].rolling(window).mean()
                    df.loc[mask, f'Sentiment_rolling_std_{window}'] = df.loc[mask, 'daily_sentiment'].rolling(window).std()
        
        logger.info(f"✅ Rolling features added")
        
        return df
    
    def add_target_variable(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Add target variable (future price).
        
        Args:
            df: DataFrame with stock data
            horizon: Number of days ahead to predict
            
        Returns:
            DataFrame with target variable
        """
        logger.info(f"Adding target variable (horizon={horizon} days)")
        
        df = df.copy()
        
        for symbol in df['Symbol'].unique():
            mask = df['Symbol'] == symbol
            df.loc[mask, 'Target_Price'] = df.loc[mask, 'Close'].shift(-horizon)
            df.loc[mask, 'Target_Return'] = df.loc[mask, 'Returns'].shift(-horizon)
            df.loc[mask, 'Target_Direction'] = (df.loc[mask, 'Target_Return'] > 0).astype(int)
        
        logger.info(f"✅ Target variable added")
        
        return df
    
    def create_all_features(
        self,
        stock_df: pd.DataFrame,
        news_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create all features in one pipeline.
        
        Args:
            stock_df: DataFrame with stock data
            news_df: DataFrame with news data
            
        Returns:
            DataFrame with all features
        """
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info("="*80)
        
        # Step 1: Sentiment analysis
        news_with_sentiment = self.add_sentiment_features(news_df)
        
        # Step 2: Aggregate daily sentiment
        stock_with_sentiment = self.aggregate_daily_sentiment(news_with_sentiment, stock_df)
        
        # Step 3: Add lag features
        df = self.add_lag_features(stock_with_sentiment)
        
        # Step 4: Add rolling features
        df = self.add_rolling_features(df)
        
        # Step 5: Add target variable
        df = self.add_target_variable(df, horizon=1)
        
        # Drop rows with NaN (from lag/rolling features)
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)
        
        logger.info(f"Removed {removed} rows with NaN values from feature engineering")
        logger.info(f"Final dataset: {len(df)} rows with {len(df.columns)} features")
        logger.info("="*80)
        
        return df


def main():
    """
    Main function to demonstrate usage.
    """
    from src.utils.config import PROCESSED_DATA_DIR, FEATURES_DIR
    
    # Initialize feature engineer
    engineer = FeatureEngineer(use_finbert=True)
    
    # Load cleaned data
    stock_file = PROCESSED_DATA_DIR / "cleaned_stock_data.csv"
    news_file = PROCESSED_DATA_DIR / "cleaned_news_data.csv"
    
    if stock_file.exists() and news_file.exists():
        stock_df = pd.read_csv(stock_file)
        news_df = pd.read_csv(news_file)
        
        # Create all features
        features_df = engineer.create_all_features(stock_df, news_df)
        
        # Save features
        output_file = FEATURES_DIR / "engineered_features.csv"
        features_df.to_csv(output_file, index=False)
        logger.info(f"💾 Saved features to {output_file}")
        
        # Print feature summary
        print("\n" + "="*80)
        print("FEATURE SUMMARY")
        print("="*80)
        print(f"Total features: {len(features_df.columns)}")
        print(f"Total samples: {len(features_df)}")
        print("\nFeature columns:")
        for col in features_df.columns:
            print(f"  - {col}")
        print("="*80 + "\n")
    else:
        logger.error("Cleaned data files not found. Run data cleaning first.")


if __name__ == "__main__":
    main()
