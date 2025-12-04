"""
Baseline Models Module
Implements simple baseline models for comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModels:
    """
    Baseline models for stock price prediction.
    
    Models:
    - Linear Regression
    - Random Forest
    - ARIMA
    """
    
    def __init__(self):
        """Initialize baseline models."""
        self.models = {}
        self.predictions = {}
        
        logger.info("Initialized BaselineModels")
    
    def train_linear_regression(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """
        Train Linear Regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Linear Regression...")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred)
        }
        
        logger.info(f"✅ Linear Regression - Val RMSE: {metrics['val_rmse']:.4f}, Val MAE: {metrics['val_mae']:.4f}")
        
        self.models['linear_regression'] = model
        
        return {
            'model': model,
            'metrics': metrics,
            'train_predictions': train_pred,
            'val_predictions': val_pred
        }
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_estimators: int = 100
    ) -> Dict:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_estimators: Number of trees
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'val_r2': r2_score(y_val, val_pred)
        }
        
        logger.info(f"✅ Random Forest - Val RMSE: {metrics['val_rmse']:.4f}, Val MAE: {metrics['val_mae']:.4f}")
        
        self.models['random_forest'] = model
        
        return {
            'model': model,
            'metrics': metrics,
            'train_predictions': train_pred,
            'val_predictions': val_pred,
            'feature_importance': model.feature_importances_
        }
    
    def train_arima(
        self,
        train_series: pd.Series,
        val_series: pd.Series,
        order: Tuple[int, int, int] = (5, 1, 0)
    ) -> Dict:
        """
        Train ARIMA model.
        
        Args:
            train_series: Training time series
            val_series: Validation time series
            order: ARIMA order (p, d, q)
            
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training ARIMA{order}...")
        
        try:
            # Fit ARIMA model
            model = ARIMA(train_series, order=order)
            fitted_model = model.fit()
            
            # In-sample predictions
            train_pred = fitted_model.fittedvalues
            
            # Out-of-sample forecast
            forecast_steps = len(val_series)
            val_pred = fitted_model.forecast(steps=forecast_steps)
            
            # Metrics
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(train_series[len(train_series)-len(train_pred):], train_pred)),
                'train_mae': mean_absolute_error(train_series[len(train_series)-len(train_pred):], train_pred),
                'val_rmse': np.sqrt(mean_squared_error(val_series, val_pred)),
                'val_mae': mean_absolute_error(val_series, val_pred)
            }
            
            logger.info(f"✅ ARIMA - Val RMSE: {metrics['val_rmse']:.4f}, Val MAE: {metrics['val_mae']:.4f}")
            
            self.models['arima'] = fitted_model
            
            return {
                'model': fitted_model,
                'metrics': metrics,
                'train_predictions': train_pred,
                'val_predictions': val_pred
            }
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {str(e)}")
            return None
    
    def save_models(self, output_dir: Path):
        """
        Save trained models.
        
        Args:
            output_dir: Directory to save models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = output_dir / f"baseline_{name}.joblib"
            joblib.dump(model, filepath)
            logger.info(f"💾 Saved {name} to {filepath}")
    
    def load_models(self, model_dir: Path):
        """
        Load trained models.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_dir = Path(model_dir)
        
        for model_file in model_dir.glob("baseline_*.joblib"):
            name = model_file.stem.replace("baseline_", "")
            self.models[name] = joblib.load(model_file)
            logger.info(f"✅ Loaded {name} from {model_file}")


def main():
    """
    Main function to demonstrate usage.
    """
    from src.utils.config import FEATURES_DIR, MODELS_DIR
    
    # Load features
    features_file = FEATURES_DIR / "engineered_features.csv"
    
    if features_file.exists():
        df = pd.read_csv(features_file)
        
        # Prepare data (simple example)
        feature_cols = [col for col in df.columns if col not in 
                       ['Symbol', 'Date', 'Target_Price', 'Target_Return', 'Target_Direction']]
        
        X = df[feature_cols].values
        y = df['Target_Price'].values
        
        # Simple train/val split (70/30)
        split_idx = int(len(X) * 0.7)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Remove NaN
        mask_train = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        mask_val = ~np.isnan(X_val).any(axis=1) & ~np.isnan(y_val)
        
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        X_val, y_val = X_val[mask_val], y_val[mask_val]
        
        # Initialize baseline models
        baseline = BaselineModels()
        
        # Train models
        lr_results = baseline.train_linear_regression(X_train, y_train, X_val, y_val)
        rf_results = baseline.train_random_forest(X_train, y_train, X_val, y_val)
        
        # Save models
        baseline.save_models(MODELS_DIR)
        
        print("\n" + "="*80)
        print("BASELINE MODELS SUMMARY")
        print("="*80)
        print(f"Linear Regression - Val RMSE: {lr_results['metrics']['val_rmse']:.4f}")
        print(f"Random Forest     - Val RMSE: {rf_results['metrics']['val_rmse']:.4f}")
        print("="*80 + "\n")
        
    else:
        logger.error("Features file not found. Run feature engineering first.")


if __name__ == "__main__":
    main()
