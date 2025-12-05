"""
Evaluation Metrics Module
Comprehensive metrics for stock price prediction models
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
        logger.info("Initialized ModelEvaluator")
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        
        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['max_error'] = np.max(np.abs(errors))
        
        return metrics
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate directional prediction accuracy."""
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        metrics = {
            'directional_accuracy': accuracy_score(true_direction, pred_direction) * 100,
            'up_precision': precision_score(true_direction, pred_direction, zero_division=0) * 100,
            'up_recall': recall_score(true_direction, pred_direction, zero_division=0) * 100,
            'f1_score': f1_score(true_direction, pred_direction, zero_division=0) * 100
        }
        
        cm = confusion_matrix(true_direction, pred_direction)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def calculate_financial_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate financial performance metrics."""
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        if np.std(pred_returns) > 0:
            sharpe_ratio = np.mean(pred_returns) / np.std(pred_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        cumulative = np.cumprod(1 + pred_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        metrics = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'avg_return_pct': np.mean(true_returns) * 100,
            'volatility_pct': np.std(true_returns) * 100
        }
        
        return metrics
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict:
        """Comprehensive model evaluation."""
        logger.info(f"Evaluating {model_name}...")
        
        regression_metrics = self.calculate_regression_metrics(y_true, y_pred)
        directional_metrics = self.calculate_directional_accuracy(y_true, y_pred)
        financial_metrics = self.calculate_financial_metrics(y_true, y_pred)
        
        all_metrics = {
            'model_name': model_name,
            **regression_metrics,
            **directional_metrics,
            **financial_metrics
        }
        
        self.metrics[model_name] = all_metrics
        
        logger.info(f"  RMSE: {regression_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {regression_metrics['mae']:.4f}")
        logger.info(f"  R²: {regression_metrics['r2']:.4f}")
        logger.info(f"  Directional Accuracy: {directional_metrics['directional_accuracy']:.2f}%")
        
        return all_metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models."""
        if not self.metrics:
            logger.warning("No models evaluated yet")
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in self.metrics.items():
            row = {
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE (%)': metrics['mape'],
                'Dir. Acc. (%)': metrics['directional_accuracy'],
                'Sharpe Ratio': metrics['sharpe_ratio']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('RMSE')
        
        return df
    
    def get_best_model(self, metric: str = 'rmse') -> str:
        """Get the best performing model."""
        if not self.metrics:
            return None
        
        if metric in ['rmse', 'mae', 'mse', 'mape', 'max_drawdown_pct']:
            best_model = min(self.metrics.items(), key=lambda x: x[1][metric])
        else:
            best_model = max(self.metrics.items(), key=lambda x: x[1][metric])
        
        return best_model[0]


def main():
    """Demo usage."""
    logger.info("Model Evaluator module - use in training pipeline")


if __name__ == "__main__":
    main()
