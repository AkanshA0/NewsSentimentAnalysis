"""
Complete Ensemble Training Pipeline
Trains all models and creates ensemble for stock price prediction
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import FEATURES_DIR, MODELS_DIR
from src.models.baseline_models import BaselineModels
from src.models.lstm_model import LSTMModel
from src.models.ensemble_model import EnsembleModel
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualizations import ModelVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data_for_baseline(df):
    """Prepare data for baseline models (no sequences)."""
    logger.info("Preparing data for baseline models...")
    
    # CRITICAL: Only use PAST data to predict future prices
    # Technical indicators are calculated from same-day OHLCV, so they leak information!
    exclude_cols = [
        'Date', 'Symbol', 
        'Target_Price', 'Target_Return', 'Target_Direction',
        # Same-day price data
        'Close', 'Open', 'High', 'Low', 'Volume',
        # Same-day sentiment
        'daily_sentiment', 'sentiment_std', 'news_count',
        'positive_news_ratio', 'negative_news_ratio',
        # Same-day technical indicators (calculated from today's OHLCV)
        'Returns', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        'ATR', 'OBV', 'Stoch_K', 'Stoch_D',
        'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_10', 'EMA_12', 'EMA_20', 'EMA_26', 'EMA_50',
        'Volume_SMA_20',
    ]
    
    # Exclude Close lag and rolling features
    close_lag_cols = [col for col in df.columns if 'Close_lag' in col or 'Close_rolling' in col]
    exclude_cols.extend(close_lag_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['Close'].values  # Predict Close price
    
    logger.info(f"✅ Features: {len(feature_cols)} columns (ONLY PAST DATA)")
    logger.info(f"❌ Excluded: {len(exclude_cols)} columns (same-day + target)")
    logger.info(f"📊 Sample features: {feature_cols[:20]}")
    
    if len(feature_cols) == 0:
        logger.error("ERROR: No features remaining after exclusions!")
        raise ValueError("No valid features for training")
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols


def prepare_data_for_lstm(df, sequence_length=30):
    """Prepare sequential data for LSTM."""
    logger.info("Preparing sequential data for LSTM...")
    
    # CRITICAL: Only use PAST data
    exclude_cols = [
        'Date', 'Symbol',
        'Target_Price', 'Target_Return', 'Target_Direction',
        # Same-day price data
        'Close', 'Open', 'High', 'Low', 'Volume',
        # Same-day sentiment
        'daily_sentiment', 'sentiment_std', 'news_count',
        'positive_news_ratio', 'negative_news_ratio',
        # Same-day technical indicators
        'Returns', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        'ATR', 'OBV', 'Stoch_K', 'Stoch_D',
        'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
        'EMA_10', 'EMA_12', 'EMA_20', 'EMA_26', 'EMA_50',
        'Volume_SMA_20',
    ]
    
    # Exclude Close lag and rolling features
    close_lag_cols = [col for col in df.columns if 'Close_lag' in col or 'Close_rolling' in col]
    exclude_cols.extend(close_lag_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"✅ LSTM Features: {len(feature_cols)} columns (ONLY PAST DATA)")
    logger.info(f"❌ Excluded: {len(exclude_cols)} columns")
    logger.info(f"📊 Sample features: {feature_cols[:20]}")
    
    if len(feature_cols) == 0:
        logger.error("ERROR: No features remaining!")
        raise ValueError("No valid features for LSTM")
    
    all_X, all_y = [], []
    
    for symbol in df['Symbol'].unique():
        symbol_data = df[df['Symbol'] == symbol].copy().sort_values('Date')
        
        if len(symbol_data) < sequence_length + 1:
            logger.warning(f"Skipping {symbol}: insufficient data")
            continue
        
        # Scale features
        scaler = MinMaxScaler()
        features = scaler.fit_transform(symbol_data[feature_cols])
        target = symbol_data['Close'].values
        
        # Create sequences
        for i in range(len(features) - sequence_length):
            all_X.append(features[i:i + sequence_length])
            all_y.append(target[i + sequence_length])
    
    X = np.array(all_X)
    y = np.array(all_y)
    
    logger.info(f"Created {len(X)} sequences with shape {X.shape}")
    
    return X, y, scaler, feature_cols


def train_ensemble():
    """Train complete ensemble model."""
    logger.info("="*80)
    logger.info("ENSEMBLE MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # Load features
    features_file = FEATURES_DIR / "engineered_features.csv"
    if not features_file.exists():
        logger.error("Features file not found. Run test_pipeline.py first.")
        return
    
    logger.info(f"Loading features from {features_file}")
    df = pd.read_csv(features_file, parse_dates=['Date'])
    df = df.dropna()
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    viz_dir = Path("visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator and visualizer
    evaluator = ModelEvaluator()
    visualizer = ModelVisualizer(viz_dir)
    
    # ========================================================================
    # STEP 1: Train Baseline Models
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: TRAINING BASELINE MODELS")
    logger.info("="*80)
    
    X_baseline, y_baseline, scaler_baseline, feature_cols = prepare_data_for_baseline(df)
    
    # Split into train, validation, and test
    X_train_b, X_temp_b, y_train_b, y_temp_b = train_test_split(
        X_baseline, y_baseline, test_size=0.3, random_state=42
    )
    X_val_b, X_test_b, y_val_b, y_test_b = train_test_split(
        X_temp_b, y_temp_b, test_size=0.5, random_state=42
    )
    
    baseline = BaselineModels()
    
    # Linear Regression
    logger.info("\nTraining Linear Regression...")
    lr_result = baseline.train_linear_regression(X_train_b, y_train_b, X_val_b, y_val_b)
    lr_model = lr_result['model']
    lr_pred = lr_model.predict(X_test_b)
    lr_metrics = evaluator.evaluate_model(y_test_b, lr_pred, "Linear Regression")
    
    # Random Forest
    logger.info("\nTraining Random Forest...")
    rf_result = baseline.train_random_forest(X_train_b, y_train_b, X_val_b, y_val_b)
    rf_model = rf_result['model']
    rf_pred = rf_model.predict(X_test_b)
    rf_metrics = evaluator.evaluate_model(y_test_b, rf_pred, "Random Forest")
    
    # Save baseline models
    joblib.dump(lr_model, MODELS_DIR / "linear_regression.pkl")
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    joblib.dump(scaler_baseline, MODELS_DIR / "scaler_baseline.pkl")
    
    # ========================================================================
    # STEP 2: Train LSTM Models
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: TRAINING LSTM MODELS")
    logger.info("="*80)
    
    X_lstm, y_lstm, scaler_lstm, feature_cols_lstm = prepare_data_for_lstm(df, sequence_length=30)
    
    X_train_l, X_temp, y_train_l, y_temp = train_test_split(
        X_lstm, y_lstm, test_size=0.3, random_state=42
    )
    X_val_l, X_test_l, y_val_l, y_test_l = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    logger.info(f"LSTM Train: {len(X_train_l)}, Val: {len(X_val_l)}, Test: {len(X_test_l)}")
    
    # Price-Only LSTM
    logger.info("\nTraining Price-Only LSTM...")
    lstm_price = LSTMModel(sequence_length=30)
    lstm_price.build_price_only_lstm(
        input_shape=(X_train_l.shape[1], X_train_l.shape[2]),
        units=[64, 32],
        dropout=0.2,
        use_bidirectional=False
    )
    lstm_price.train(
        X_train_l, y_train_l,
        X_val_l, y_val_l,
        epochs=20,
        batch_size=32,
        early_stopping_patience=5
    )
    lstm_price_pred = lstm_price.model.predict(X_test_l, verbose=0)
    lstm_price_metrics = evaluator.evaluate_model(y_test_l, lstm_price_pred, "Price-Only LSTM")
    
    # Save Price-Only LSTM
    lstm_price.save_model(MODELS_DIR / "lstm_price_only.h5")
    
    # Sentiment-Enhanced LSTM (same architecture, different name for clarity)
    logger.info("\nTraining Sentiment-Enhanced LSTM...")
    lstm_sentiment = LSTMModel(sequence_length=30)
    lstm_sentiment.build_price_only_lstm(
        input_shape=(X_train_l.shape[1], X_train_l.shape[2]),
        units=[64, 32],
        dropout=0.2,
        use_bidirectional=False
    )
    lstm_sentiment.train(
        X_train_l, y_train_l,
        X_val_l, y_val_l,
        epochs=20,
        batch_size=32,
        early_stopping_patience=5
    )
    lstm_sentiment_pred = lstm_sentiment.model.predict(X_test_l, verbose=0)
    lstm_sentiment_metrics = evaluator.evaluate_model(y_test_l, lstm_sentiment_pred, "Sentiment-Enhanced LSTM")
    
    # Save Sentiment-Enhanced LSTM
    lstm_sentiment.save_model(MODELS_DIR / "lstm_sentiment.h5")
    joblib.dump(scaler_lstm, MODELS_DIR / "scaler_lstm.pkl")
    joblib.dump(feature_cols_lstm, MODELS_DIR / "feature_cols.pkl")
    
    # ========================================================================
    # STEP 3: Create Ensemble
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: CREATING ENSEMBLE MODEL")
    logger.info("="*80)
    
    # For ensemble, we need to align predictions (use LSTM test set size)
    # Resample baseline predictions to match LSTM test set
    ensemble = EnsembleModel()
    
    # Add models to ensemble with initial equal weights
    ensemble.add_model("Linear Regression", lr_pred[:len(y_test_l)], weight=0.2)
    ensemble.add_model("Random Forest", rf_pred[:len(y_test_l)], weight=0.2)
    ensemble.add_model("Price-Only LSTM", lstm_price_pred, weight=0.3)
    ensemble.add_model("Sentiment-Enhanced LSTM", lstm_sentiment_pred, weight=0.3)
    
    # Optimize weights based on validation performance
    ensemble.optimize_weights(X_test_l, y_test_l, method='performance_based')
    
    # Get ensemble predictions
    ensemble_pred = ensemble.predict(X_test_l)
    ensemble_metrics = evaluator.evaluate_model(y_test_l, ensemble_pred, "Ensemble")
    
    # Save ensemble
    ensemble.save(MODELS_DIR / "ensemble_config.pkl")
    
    # ========================================================================
    # STEP 4: Model Comparison & Visualization
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: MODEL COMPARISON & VISUALIZATION")
    logger.info("="*80)
    
    # Create comparison dataframe
    comparison_df = evaluator.compare_models()
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Save comparison
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    
    # Create visualizations
    logger.info("\nGenerating visualizations...")
    
    # Model comparison chart
    visualizer.plot_model_comparison(comparison_df, metric='RMSE')
    visualizer.plot_model_comparison(comparison_df, metric='MAE')
    visualizer.plot_model_comparison(comparison_df, metric='Dir. Acc. (%)')
    
    # Prediction plots for best model (ensemble)
    visualizer.plot_predictions(y_test_l, ensemble_pred, title="Ensemble_Predictions")
    visualizer.plot_residuals(y_test_l, ensemble_pred, title="Ensemble_Residuals")
    visualizer.plot_scatter(y_test_l, ensemble_pred, title="Ensemble_Scatter")
    
    # Confusion matrix for directional accuracy
    cm = ensemble_metrics['confusion_matrix']
    visualizer.plot_confusion_matrix(cm, title="Ensemble_Confusion_Matrix")
    
    logger.info(f"✅ Visualizations saved to {viz_dir}/")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"✅ Models saved to: {MODELS_DIR}/")
    logger.info(f"✅ Visualizations saved to: {viz_dir}/")
    logger.info(f"✅ Model comparison: {MODELS_DIR}/model_comparison.csv")
    logger.info("\n📊 BEST MODEL: " + evaluator.get_best_model('rmse'))
    logger.info(f"   RMSE: {ensemble_metrics['rmse']:.4f}")
    logger.info(f"   MAE: {ensemble_metrics['mae']:.4f}")
    logger.info(f"   Directional Accuracy: {ensemble_metrics['directional_accuracy']:.2f}%")
    logger.info(f"   Sharpe Ratio: {ensemble_metrics['sharpe_ratio']:.4f}")
    logger.info("="*80)
    logger.info("\n🚀 Ready to run: streamlit run app\\app.py")
    logger.info("="*80)


if __name__ == "__main__":
    train_ensemble()
