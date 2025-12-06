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
