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


def train_test_split_time_series(
    data: pd.DataFrame,
    test_size: float = 0.2,
):
    """
    Split time series data into train and test sets without shuffling.

    Args:
        data: DataFrame with features and "target" column.
        test_size: Fraction of samples to be used for testing (0 < test_size < 1).

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = data.drop(columns=["target"])
    y = data["target"]

    n_samples = len(data)
    n_test = int(n_samples * test_size)

    X_train = X.iloc[:-n_test]
    X_test = X.iloc[-n_test:]
    y_train = y.iloc[:-n_test]
    y_test = y.iloc[-n_test:]

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
):
    """
    Scale features with StandardScaler (fit on train, apply to test).

    Args:
        X_train: Training features.
        X_test: Test features.

    Returns:
        X_train_scaled, X_test_scaled, fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_model(
    X_train,
    y_train,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on stock features.

    Args:
        X_train: Training feature matrix.
        y_train: Training targets.
        n_estimators: Number of trees.
        max_depth: Max depth of trees (None = unlimited).

    Returns:
        Trained RandomForestRegressor model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, feature_names: list[str]):
    """
    Evaluate the model and print metrics and feature importances.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.
        feature_names: Names of features used in training.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Evaluation on Test Set ===")
    print(f"RMSE: {rmse:.6f}")
    print(f"R^2 : {r2:.4f}")

    # Feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\n=== Feature Importances ===")
    print(importance_df.to_string(index=False))


def main():
    # ---- Config ----
    ticker = "AAPL"
    period = "2y"
    test_size = 0.2

    # ---- Pipeline ----
    df_raw = download_stock_data(ticker, period=period)
    data = add_features(df_raw)

    print(f"Total samples after feature engineering: {len(data)}")

    X_train, X_test, y_train, y_test = train_test_split_time_series(
        data, test_size=test_size
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    feature_names = list(X_train.columns)

    model = train_model(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test, feature_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
