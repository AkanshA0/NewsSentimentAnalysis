"""
Ensemble Model Module
Combines multiple models for robust predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model combining multiple prediction models.
    
    Combines:
    - Baseline models (Linear Regression, Random Forest)
    - LSTM models (Price-Only, Sentiment-Enhanced, Multi-Input)
    
    Uses weighted averaging based on validation performance.
    """
    
    def __init__(self):
        """Initialize ensemble model."""
        self.models = {}
        self.weights = {}
        self.predictions = {}
        
        logger.info("Initialized EnsembleModel")
    
    def add_model(self, name: str, model: any, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Trained model
            weight: Weight for this model in ensemble
        """
        self.models[name] = model
        self.weights[name] = weight
        
        logger.info(f"Added model '{name}' with weight {weight}")
    
    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = 'performance_based'
    ):
        """
        Optimize ensemble weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            method: Weight optimization method
        """
        logger.info(f"Optimizing ensemble weights using {method}...")
        
        if method == 'performance_based':
            # Calculate performance for each model
            performances = {}
            
            for name, model in self.models.items():
                try:
                    # Get predictions
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_val)
                    else:
                        pred = model  # Already predictions
                    
                    # Calculate RMSE (lower is better)
                    rmse = np.sqrt(mean_squared_error(y_val, pred))
                    performances[name] = 1 / rmse  # Inverse for weight
                    
                except Exception as e:
                    logger.warning(f"Could not evaluate {name}: {str(e)}")
                    performances[name] = 0.0
            
            # Normalize to sum to 1
            total = sum(performances.values())
            if total > 0:
                self.weights = {name: perf / total for name, perf in performances.items()}
            
            logger.info("Optimized weights:")
            for name, weight in self.weights.items():
                logger.info(f"  {name}: {weight:.4f}")
        
        elif method == 'equal':
            # Equal weights
            n_models = len(self.models)
            self.weights = {name: 1.0 / n_models for name in self.models.keys()}
            logger.info(f"Using equal weights: {1.0/n_models:.4f} each")
    
    def predict(self, X: np.ndarray, return_individual: bool = False) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            return_individual: Whether to return individual model predictions
            
        Returns:
            Ensemble predictions (and optionally individual predictions)
        """
        individual_preds = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                    if len(pred.shape) > 1:
                        pred = pred.flatten()
                else:
                    pred = model  # Already predictions
                
                individual_preds[name] = pred
                
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {str(e)}")
                individual_preds[name] = np.zeros(len(X))
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for name, pred in individual_preds.items():
            weight = self.weights.get(name, 0.0)
            ensemble_pred += weight * pred
        
        self.predictions = individual_preds
        
        if return_individual:
            return ensemble_pred, individual_preds
        
        return ensemble_pred
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate ensemble on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating ensemble model...")
        
        # Get predictions
        y_pred, individual_preds = self.predict(X_test, return_individual=True)
        
        # Ensemble metrics
        metrics = {
            'ensemble': {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
        }
        
        # Directional accuracy
        y_test_direction = np.diff(y_test) > 0
        y_pred_direction = np.diff(y_pred) > 0
        metrics['ensemble']['directional_accuracy'] = np.mean(y_test_direction == y_pred_direction) * 100
        
        # Individual model metrics
        for name, pred in individual_preds.items():
            metrics[name] = {
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'mae': mean_absolute_error(y_test, pred),
                'r2': r2_score(y_test, pred)
            }
        
        logger.info(f"✅ Ensemble RMSE: {metrics['ensemble']['rmse']:.4f}, MAE: {metrics['ensemble']['mae']:.4f}")
        logger.info(f"   Directional Accuracy: {metrics['ensemble']['directional_accuracy']:.2f}%")
        
        return metrics
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from models that support it.
        
        Returns:
            Dictionary of feature importances
        """
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances[name] = np.abs(model.coef_)
        
        return importances
    
    def save(self, filepath: Path):
        """
        Save ensemble model.
        
        Args:
            filepath: Path to save ensemble
        """
        ensemble_data = {
            'weights': self.weights,
            'model_names': list(self.models.keys())
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"💾 Saved ensemble configuration to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load ensemble configuration.
        
        Args:
            filepath: Path to ensemble file
        """
        ensemble_data = joblib.load(filepath)
        self.weights = ensemble_data['weights']
        
        logger.info(f"Loaded ensemble configuration from {filepath}")


def main():
    """
    Main function to demonstrate usage.
    """
    logger.info("Ensemble Model module - use in training pipeline")


if __name__ == "__main__":
    main()
