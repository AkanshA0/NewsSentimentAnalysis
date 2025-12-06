"""
Train a simple ML model on stock data to predict next-day returns.

Requirements:
    pip install yfinance scikit-learn pandas numpy

Usage:
    python train_stock_model.py
"""

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def download_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download historical stock data using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT").
        period: Time period for history (e.g., "6mo", "1y", "2y", "5y").

    Returns:
        DataFrame with OHLCV and other columns.
    """
    print(f"Downloading data for {ticker} ({period})...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        raise ValueError("No data downloaded. Check the ticker or period.")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple technical features and create a prediction target.

    Target:
        Next-day return (percentage change of Close price).

    Features:
        - Daily returns
        - 5, 10, 20-day moving averages of Close
        - 5, 10-day moving averages of Volume
        - Rolling volatility (10-day std of returns)

    Args:
        df: Raw price DataFrame from yfinance.

    Returns:
        DataFrame with feature columns and "target" column.
    """
    data = df.copy()

    # Daily return
    data["return_1d"] = data["Close"].pct_change()

    # Moving averages of price
    data["ma_5"] = data["Close"].rolling(window=5).mean()
    data["ma_10"] = data["Close"].rolling(window=10).mean()
    data["ma_20"] = data["Close"].rolling(window=20).mean()

    # Moving averages of volume
    data["vol_ma_5"] = data["Volume"].rolling(window=5).mean()
    data["vol_ma_10"] = data["Volume"].rolling(window=10).mean()

    # Volatility (std of returns)
    data["volatility_10"] = data["return_1d"].rolling(window=10).std()

    # Target: next-day return
    data["target"] = data["return_1d"].shift(-1)

    # Drop rows with NaNs (from rolling and shifting)
    data = data.dropna()

    return data
