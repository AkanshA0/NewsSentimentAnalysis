"""
Stock Data Collector Module
Collects 2 years of historical stock price data using yfinance
Calculates technical indicators for feature engineering
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    Collects historical stock price data and calculates technical indicators.
    
    Features:
    - Uses yfinance (free, no API key required)
    - Collects 2 years of daily OHLCV data
    - Calculates comprehensive technical indicators
    - Handles stock splits and dividends automatically
    - Local caching to minimize API calls
    """
    
    def __init__(self, symbols: List[str], period: str = "2y", interval: str = "1d"):
        """
        Initialize the stock data collector.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL', 'TSLA', 'NVDA'])
            period: Data period (default: '2y' for 2 years)
            interval: Data interval (default: '1d' for daily)
        """
        self.symbols = symbols
        self.period = period
        self.interval = interval
        self.data = {}
        
        logger.info(f"Initialized StockDataCollector for {len(symbols)} stocks: {', '.join(symbols)}")
    
    def collect_stock_data(self, symbol: str) -> pd.DataFrame:
        """
        Collect historical stock data for a single symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol}...")
            
            # Download data using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=self.period, interval=self.interval)
            
            if df.empty:
                logger.error(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            logger.info(f"✅ Successfully fetched {len(df)} days of data for {symbol}")
            logger.info(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators.
        Calculate technical indicators manually (no pandas-ta needed).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        if df.empty:
            return df
        
        try:
            logger.info(f"Calculating technical indicators for {df['Symbol'].iloc[0]}...")
            
            # Make a copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Simple Moving Averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] # Percentage width
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.ewm(span=14, adjust=False).mean() # Using EMA for ATR smoothing
            
            # On-Balance Volume (OBV)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            
            # Stochastic Oscillator
            # df['EMA_50'] = ta.ema(df['Close'], length=50)  # Removed: Redundant and 'ta' not imported
            
            # Price momentum features
            df['Price_Change_1d'] = df['Close'].diff(1)
            df['Price_Change_5d'] = df['Close'].diff(5)
            df['Price_Change_10d'] = df['Close'].diff(10)
            
            # Volatility (20-day rolling standard deviation of returns)
            df['Volatility_20d'] = df['Returns'].rolling(window=20).std()
            
            # Volume features
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            
            logger.info(f"✅ Technical indicators calculated successfully")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def collect_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all configured stock symbols.
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        logger.info(f"Starting data collection for {len(self.symbols)} stocks...")
        
        for symbol in self.symbols:
            # Collect stock data
            df = self.collect_stock_data(symbol)
            
            if not df.empty:
                # Calculate technical indicators
                df = self.calculate_technical_indicators(df)
                self.data[symbol] = df
            else:
                logger.warning(f"Skipping {symbol} due to data collection failure")
        
        logger.info(f"✅ Data collection complete for {len(self.data)} stocks")
        return self.data
    
    def save_data(self, output_dir: Path):
        """
        Save collected data to CSV files.
        
        Args:
            output_dir: Directory to save data files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for symbol, df in self.data.items():
            filepath = output_dir / f"{symbol}_stock_data.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"💾 Saved {symbol} data to {filepath}")
        
        # Also save combined data
        if self.data:
            combined_df = pd.concat(self.data.values(), ignore_index=True)
            combined_filepath = output_dir / "all_stocks_combined.csv"
            combined_df.to_csv(combined_filepath, index=False)
            logger.info(f"💾 Saved combined data to {combined_filepath}")
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for collected data.
        
        Returns:
            DataFrame with summary statistics
        """
        summaries = []
        
        for symbol, df in self.data.items():
            summary = {
                'Symbol': symbol,
                'Total_Days': len(df),
                'Start_Date': df['Date'].min(),
                'End_Date': df['Date'].max(),
                'Avg_Close': df['Close'].mean(),
                'Min_Close': df['Close'].min(),
                'Max_Close': df['Close'].max(),
                'Total_Return_%': ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100,
                'Avg_Volume': df['Volume'].mean(),
                'Missing_Values': df.isnull().sum().sum()
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def main():
    """
    Main function to demonstrate usage.
    """
    from src.utils.config import STOCK_SYMBOLS, DATA_PERIOD, RAW_DATA_DIR
    
    # Initialize collector
    collector = StockDataCollector(
        symbols=STOCK_SYMBOLS,
        period=DATA_PERIOD,
        interval="1d"
    )
    
    # Collect data
    data = collector.collect_all_stocks()
    
    # Print summary
    print("\n" + "="*80)
    print("DATA COLLECTION SUMMARY")
    print("="*80)
    summary = collector.get_data_summary()
    print(summary.to_string(index=False))
    print("="*80 + "\n")
    
    # Save data
    collector.save_data(RAW_DATA_DIR)
    
    print(f"\n✅ Stock data collection complete!")
    print(f"📁 Data saved to: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
