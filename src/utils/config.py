"""
Configuration file for Stock Price Prediction with News Sentiment Analysis
This file contains all configuration parameters for the project.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, 
                 MODELS_DIR, LOGS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STOCK CONFIGURATION
# ============================================================================
STOCK_SYMBOLS = ["AAPL", "GOOGL", "TSLA", "NVDA", "GME"]

# Stock display names
STOCK_NAMES = {
    'AAPL': 'Apple Inc',
    'GOOGL': 'Alphabet Inc (Google)',
    'TSLA': 'Tesla Inc',
    'NVDA': 'NVIDIA Corp',
    'GME': 'GameStop Corp'  # Highly volatile, lots of news
}

DATA_PERIOD = "2y"  # 2 years of historical stock data
NEWS_PERIOD = "1y"  # 1 year of news data (more practical for scraping)
DATA_INTERVAL = "1d"  # Daily data

# ============================================================================
# DATA COLLECTION CONFIGURATION
# ============================================================================

# Stock Data
STOCK_DATA_SOURCE = "yfinance"

# News Data Sources
NEWS_SOURCES = {
    "yahoo_finance": {
        "enabled": True,
        "scraping": True,
        "base_url": "https://finance.yahoo.com/quote/{symbol}/news"
    },
    "google_news": {
        "enabled": True,
        "rss_feeds": True,
        "base_url": "https://news.google.com/rss/search?q={symbol}+stock"
    },
    "finviz": {
        "enabled": True,
        "scraping": True,
        "base_url": "https://finviz.com/quote.ashx?t={symbol}"
    }
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Technical Indicators
TECHNICAL_INDICATORS = [
    "RSI",           # Relative Strength Index
    "MACD",          # Moving Average Convergence Divergence
    "BB",            # Bollinger Bands
    "ATR",           # Average True Range
    "OBV",           # On-Balance Volume
    "STOCH",         # Stochastic Oscillator
    "SMA_10",        # Simple Moving Average (10 days)
    "SMA_20",
    "SMA_50",
    "EMA_10",        # Exponential Moving Average (10 days)
    "EMA_20",
    "EMA_50"
]

# Sentiment Analysis Models (Free, optimized for financial news)
SENTIMENT_MODELS = {
    "finbert": {
        "enabled": True,
        "model_name": "ProsusAI/finbert",
        "source": "huggingface",
        "description": "Specifically trained on financial text"
    },
    "textblob": {
        "enabled": True,
        "source": "textblob",
        "description": "General-purpose sentiment analysis (backup)"
    }
}

# Sequence Length for Time Series
SEQUENCE_LENGTH = 60  # Use 60 days of historical data for prediction

# Feature Scaling
SCALER_TYPE = "MinMaxScaler"  # or "StandardScaler"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Train/Validation/Test Split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Model Architectures
MODELS = {
    "baseline_arima": {
        "enabled": True,
        "order": (5, 1, 0)
    },
    "baseline_linear": {
        "enabled": True
    },
    "price_lstm": {
        "enabled": True,
        "units": [128, 64, 32],
        "dropout": 0.2,
        "batch_norm": True,
        "activation": "tanh"
    },
    "sentiment_lstm": {
        "enabled": True,
        "units": [64, 32],
        "dropout": 0.2,
        "batch_norm": True,
        "activation": "tanh"
    },
    "multi_input_lstm": {
        "enabled": True,
        "price_branch_units": [128, 64],
        "sentiment_branch_units": [64, 32],
        "merged_units": [64, 32],
        "dropout": 0.2,
        "batch_norm": True,
        "activation": "tanh"
    },
    "ensemble": {
        "enabled": True,
        "weights": {
            "baseline": 0.1,
            "price_lstm": 0.3,
            "sentiment_lstm": 0.3,
            "multi_input": 0.3
        }
    }
}

# Training Configuration
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "huber",  # Robust to outliers
    "early_stopping_patience": 15,
    "reduce_lr_patience": 7,
    "reduce_lr_factor": 0.5
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVALUATION_METRICS = [
    "RMSE",                    # Root Mean Squared Error
    "MAE",                     # Mean Absolute Error
    "MAPE",                    # Mean Absolute Percentage Error
    "R2",                      # R-squared
    "MSE",                     # Mean Squared Error
    "Directional_Accuracy",    # Percentage of correct up/down predictions
    "Sharpe_Ratio",           # Risk-adjusted return
    "Max_Drawdown"            # Maximum loss from peak
]

# ============================================================================
# MLFLOW CONFIGURATION
# ============================================================================
MLFLOW_CONFIG = {
    "tracking_uri": "file:///./mlruns",  # Local tracking
    "experiment_name": "stock_price_prediction",
    "artifact_location": "./mlruns",
    "log_models": True,
    "log_artifacts": True
}

# ============================================================================
# AIRFLOW CONFIGURATION
# ============================================================================
AIRFLOW_CONFIG = {
    "dag_schedule": {
        "data_collection": "0 2 * * *",      # Daily at 2 AM
        "model_retraining": "0 3 * * 0"      # Weekly on Sunday at 3 AM
    }
}

# ============================================================================
# STREAMLIT APP CONFIGURATION
# ============================================================================
STREAMLIT_CONFIG = {
    "page_title": "Stock Price Prediction with News Sentiment",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#0e1117",
        "secondaryBackgroundColor": "#262730",
        "textColor": "#fafafa"
    }
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
VISUALIZATION_CONFIG = {
    "figure_size": (12, 6),
    "dpi": 100,
    "style": "seaborn-v0_8-darkgrid",
    "color_palette": "husl",
    "save_format": "png"
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "stock_prediction.log"
}

# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================
DEPLOYMENT_CONFIG = {
    "platform": "huggingface",  # or "local", "colab"
    "model_format": "tensorflow",  # or "pytorch", "onnx"
    "api_rate_limit": 100,  # requests per minute
    "cache_predictions": True,
    "cache_ttl": 3600  # 1 hour in seconds
}

print(f"Configuration loaded successfully!")
print(f"Analyzing stocks: {', '.join(STOCK_SYMBOLS)}")
print(f"Data period: {DATA_PERIOD}")
print(f"Sequence length: {SEQUENCE_LENGTH} days")
