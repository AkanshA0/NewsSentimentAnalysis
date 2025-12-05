"""
Visualization Module
Creates comprehensive visualizations for model evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
import logging

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Creates visualizations for model evaluation."""
    
    def __init__(self, output_dir: Path):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ModelVisualizer, saving to {output_dir}")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs Actual", dates: Optional[pd.DatetimeIndex] = None) -> Path:
        """Plot predicted vs actual values over time."""
        fig, ax = plt.subplots(figsize=(15, 6))
        
        x = dates if dates is not None else np.arange(len(y_true))
        
        ax.plot(x, y_true, label='Actual', linewidth=2, alpha=0.8)
        ax.plot(x, y_pred, label='Predicted', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {filepath}")
        return filepath
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residual Plot") -> Path:
        """Plot residuals (errors) over time."""
        residuals = y_true - y_pred.flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.scatter(range(len(residuals)), residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Residual')
        ax1.set_title('Residuals Over Time')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residual Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {filepath}")
        return filepath
    
    def plot_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predicted vs Actual Scatter") -> Path:
        """Scatter plot of predicted vs actual."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred.flatten(), alpha=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {filepath}")
        return filepath
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str = "Directional Prediction Confusion Matrix") -> Path:
        """Plot confusion matrix for directional predictions."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Direction')
        ax.set_ylabel('Actual Direction')
        ax.set_title(title)
        ax.set_xticklabels(['Down', 'Up'])
        ax.set_yticklabels(['Down', 'Up'])
        
        filepath = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {filepath}")
        return filepath
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, metric: str = 'RMSE') -> Path:
        """Bar plot comparing models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'Model Comparison - {metric}')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        filepath = self.output_dir / f"model_comparison_{metric.lower().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').replace('%', 'pct')}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {filepath}")
        return filepath
    
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray, title: str = "Feature Importance", top_n: int = 20) -> Path:
        """Plot feature importance."""
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        filepath = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved plot: {filepath}")
        return filepath


def main():
    """Demo usage."""
    logger.info("Model Visualizer module - use in evaluation pipeline")


if __name__ == "__main__":
    main()
