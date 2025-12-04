"""
Data Cleaning and Preprocessing Module
Handles missing values, outliers, and data quality issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Cleans and preprocesses stock and news data.
    
    Features:
    - Missing value imputation
    - Outlier detection and handling
    - Data normalization
    - Temporal alignment of news and price data
    """
    
    def __init__(self, scaler_type: str = "MinMaxScaler"):
        """
        Initialize the data cleaner.
        
        Args:
            scaler_type: Type of scaler ('MinMaxScaler' or 'StandardScaler')
        """
        self.scaler_type = scaler_type
        self.scalers = {}
        
        logger.info(f"Initialized DataCleaner with {scaler_type}")
    
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock price data.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning stock data for {df['Symbol'].unique()}")
        
        df = df.copy()
        
        # Convert Date column to datetime FIRST
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Sort by date
        df = df.sort_values(['Symbol', 'Date'])
        
        # Handle missing values
        logger.info("Handling missing values...")
        
        # Forward fill for price data (use previous day's price)
        price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in price_cols:
            if col in df.columns:
                df[col] = df.groupby('Symbol')[col].fillna(method='ffill')
        
        # Fill remaining NaNs with backward fill
        df = df.fillna(method='bfill')
        
        # Handle outliers in volume (cap at 99th percentile)
        if 'Volume' in df.columns:
            for symbol in df['Symbol'].unique():
                mask = df['Symbol'] == symbol
                q99 = df.loc[mask, 'Volume'].quantile(0.99)
                df.loc[mask & (df['Volume'] > q99), 'Volume'] = q99
        
        # Remove any remaining rows with NaN values
        initial_len = len(df)
        df = df.dropna()
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} rows with remaining NaN values")
        
        logger.info(f"✅ Stock data cleaned: {len(df)} rows")
        
        return df
    
    def clean_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean news data.
        
        Args:
            df: DataFrame with news data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning news data...")
        
        df = df.copy()
        
        # Convert published_date to datetime
        if 'published_date' in df.columns:
            df['published_date'] = pd.to_datetime(df['published_date'])
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=['title'], keep='first')
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate articles")
        
        # Remove articles with missing text
        df = df.dropna(subset=['text', 'title'])
        
        # Clean text (remove extra whitespace, special characters)
        df['text'] = df['text'].str.strip()
        df['title'] = df['title'].str.strip()
        
        # Remove very short articles (less than 10 characters)
        df = df[df['text'].str.len() >= 10]
        
        logger.info(f"✅ News data cleaned: {len(df)} articles")
        
        return df
    
    def align_news_with_stock_data(
        self, 
        stock_df: pd.DataFrame, 
        news_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align news data with stock trading days.
        
        Args:
            stock_df: DataFrame with stock data
            news_df: DataFrame with news data
            
        Returns:
            DataFrame with aligned data
        """
        logger.info("Aligning news data with stock trading days...")
        
        aligned_data = []
        
        for symbol in stock_df['Symbol'].unique():
            # Get stock data for this symbol
            symbol_stock = stock_df[stock_df['Symbol'] == symbol].copy()
            symbol_news = news_df[news_df['symbol'] == symbol].copy()
            
            # For each trading day, aggregate news from that day
            for idx, row in symbol_stock.iterrows():
                date = row['Date'].date()
                
                # Get news for this date
                day_news = symbol_news[
                    symbol_news['published_date'].dt.date == date
                ]
                
                # Add news count
                row['news_count'] = len(day_news)
                
                aligned_data.append(row)
        
        result_df = pd.DataFrame(aligned_data)
        
        logger.info(f"✅ Data aligned: {len(result_df)} rows")
        
        return result_df
    
    def normalize_features(
        self, 
        df: pd.DataFrame, 
        feature_cols: list,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: DataFrame with features
            feature_cols: List of columns to normalize
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            DataFrame with normalized features
        """
        logger.info(f"Normalizing {len(feature_cols)} features...")
        
        df = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            # Create scaler if fitting
            if fit:
                if self.scaler_type == "MinMaxScaler":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                
                # Fit and transform
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                # Use existing scaler
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]])
                else:
                    logger.warning(f"No scaler found for {col}, skipping normalization")
        
        logger.info(f"✅ Features normalized")
        
        return df
    
    def save_scalers(self, output_dir: Path):
        """
        Save fitted scalers.
        
        Args:
            output_dir: Directory to save scalers
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scaler_path = output_dir / "scalers.joblib"
        joblib.dump(self.scalers, scaler_path)
        
        logger.info(f"💾 Saved {len(self.scalers)} scalers to {scaler_path}")
    
    def load_scalers(self, scaler_path: Path):
        """
        Load fitted scalers.
        
        Args:
            scaler_path: Path to scalers file
        """
        self.scalers = joblib.load(scaler_path)
        logger.info(f"✅ Loaded {len(self.scalers)} scalers from {scaler_path}")
    
    def get_data_quality_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with quality metrics
        """
        report = []
        
        for col in df.columns:
            metrics = {
                'Column': col,
                'Type': str(df[col].dtype),
                'Missing_Count': df[col].isnull().sum(),
                'Missing_Percent': (df[col].isnull().sum() / len(df)) * 100,
                'Unique_Values': df[col].nunique(),
                'Sample_Value': str(df[col].iloc[0]) if len(df) > 0 else None
            }
            
            # Add numerical statistics if applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                metrics['Mean'] = df[col].mean()
                metrics['Std'] = df[col].std()
                metrics['Min'] = df[col].min()
                metrics['Max'] = df[col].max()
            
            report.append(metrics)
        
        return pd.DataFrame(report)


def main():
    """
    Main function to demonstrate usage.
    """
    from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    # Initialize cleaner
    cleaner = DataCleaner(scaler_type="MinMaxScaler")
    
    # Load stock data
    stock_file = RAW_DATA_DIR / "all_stocks_combined.csv"
    if stock_file.exists():
        stock_df = pd.read_csv(stock_file)
        
        # Clean stock data
        cleaned_stock = cleaner.clean_stock_data(stock_df)
        
        # Generate quality report
        print("\n" + "="*80)
        print("DATA QUALITY REPORT - STOCK DATA")
        print("="*80)
        report = cleaner.get_data_quality_report(cleaned_stock)
        print(report.to_string(index=False))
        print("="*80 + "\n")
        
        # Save cleaned data
        output_file = PROCESSED_DATA_DIR / "cleaned_stock_data.csv"
        cleaned_stock.to_csv(output_file, index=False)
        logger.info(f"💾 Saved cleaned stock data to {output_file}")
    
    # Load news data
    news_file = RAW_DATA_DIR / "all_news.csv"
    if news_file.exists():
        news_df = pd.read_csv(news_file)
        
        # Clean news data
        cleaned_news = cleaner.clean_news_data(news_df)
        
        # Save cleaned data
        output_file = PROCESSED_DATA_DIR / "cleaned_news_data.csv"
        cleaned_news.to_csv(output_file, index=False)
        logger.info(f"💾 Saved cleaned news data to {output_file}")
    
    print(f"\n✅ Data cleaning complete!")


if __name__ == "__main__":
    main()
