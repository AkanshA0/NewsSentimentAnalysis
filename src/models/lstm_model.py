"""
LSTM Models Module
Implements LSTM-based models for stock price prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel:
    """
    LSTM-based models for stock price prediction.
    
    Variants:
    - Price-Only LSTM: Uses only historical prices and technical indicators
    - Sentiment-Enhanced LSTM: Adds news sentiment features
    - Multi-Input LSTM: Separate branches for price and sentiment
    """
    
    def __init__(self, sequence_length: int = 60):
        """
        Initialize LSTM model builder.
        
        Args:
            sequence_length: Number of time steps to look back
        """
        self.sequence_length = sequence_length
        self.model = None
        self.history = None
        
        logger.info(f"Initialized LSTMModel with sequence_length={sequence_length}")
    
    def create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            data: Feature array
            target: Target array
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (sequences, targets)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_price_only_lstm(
        self,
        input_shape: Tuple[int, int],
        units: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        use_bidirectional: bool = True
    ) -> keras.Model:
        """
        Build Price-Only LSTM model.
        
        Args:
            input_shape: Shape of input (sequence_length, n_features)
            units: List of LSTM units for each layer
            dropout: Dropout rate
            use_bidirectional: Whether to use bidirectional LSTM
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building Price-Only LSTM model...")
        
        model = models.Sequential(name="Price_Only_LSTM")
        
        # First LSTM layer
        if use_bidirectional:
            model.add(layers.Bidirectional(
                layers.LSTM(units[0], return_sequences=True),
                input_shape=input_shape
            ))
        else:
            model.add(layers.LSTM(units[0], return_sequences=True, input_shape=input_shape))
        
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
        
        # Additional LSTM layers
        for i, unit in enumerate(units[1:]):
            return_seq = i < len(units) - 2  # Return sequences except for last layer
            
            if use_bidirectional:
                model.add(layers.Bidirectional(layers.LSTM(unit, return_sequences=return_seq)))
            else:
                model.add(layers.LSTM(unit, return_sequences=return_seq))
            
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))  # Output layer
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Price-Only LSTM built with {model.count_params():,} parameters")
        
        self.model = model
        return model
    
    def build_multi_input_lstm(
        self,
        price_input_shape: Tuple[int, int],
        sentiment_input_shape: Tuple[int, int],
        price_units: List[int] = [128, 64],
        sentiment_units: List[int] = [64, 32],
        merged_units: List[int] = [64, 32],
        dropout: float = 0.2
    ) -> keras.Model:
        """
        Build Multi-Input LSTM with separate branches for price and sentiment.
        
        Args:
            price_input_shape: Shape of price input
            sentiment_input_shape: Shape of sentiment input
            price_units: LSTM units for price branch
            sentiment_units: LSTM units for sentiment branch
            merged_units: Dense units after merging
            dropout: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building Multi-Input LSTM model...")
        
        # Price branch
        price_input = layers.Input(shape=price_input_shape, name='price_input')
        price_lstm = layers.Bidirectional(layers.LSTM(price_units[0], return_sequences=True))(price_input)
        price_lstm = layers.BatchNormalization()(price_lstm)
        price_lstm = layers.Dropout(dropout)(price_lstm)
        price_lstm = layers.Bidirectional(layers.LSTM(price_units[1]))(price_lstm)
        price_lstm = layers.BatchNormalization()(price_lstm)
        price_lstm = layers.Dropout(dropout)(price_lstm)
        
        # Sentiment branch
        sentiment_input = layers.Input(shape=sentiment_input_shape, name='sentiment_input')
        sentiment_lstm = layers.Bidirectional(layers.LSTM(sentiment_units[0], return_sequences=True))(sentiment_input)
        sentiment_lstm = layers.BatchNormalization()(sentiment_lstm)
        sentiment_lstm = layers.Dropout(dropout)(sentiment_lstm)
        sentiment_lstm = layers.Bidirectional(layers.LSTM(sentiment_units[1]))(sentiment_lstm)
        sentiment_lstm = layers.BatchNormalization()(sentiment_lstm)
        sentiment_lstm = layers.Dropout(dropout)(sentiment_lstm)
        
        # Merge branches
        merged = layers.concatenate([price_lstm, sentiment_lstm])
        
        # Dense layers
        for units in merged_units:
            merged = layers.Dense(units, activation='relu')(merged)
            merged = layers.Dropout(dropout)(merged)
        
        # Output
        output = layers.Dense(1)(merged)
        
        # Create model
        model = models.Model(
            inputs=[price_input, sentiment_input],
            outputs=output,
            name="Multi_Input_LSTM"
        )
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Multi-Input LSTM built with {model.count_params():,} parameters")
        
        self.model = model
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 15,
        reduce_lr_patience: int = 7,
        tensorboard_log_dir: Optional[Path] = None
    ) -> keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            tensorboard_log_dir: Directory for TensorBoard logs
            
        Returns:
            Training history
        """
        logger.info(f"Training model for up to {epochs} epochs...")
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Add TensorBoard callback if log directory provided
        if tensorboard_log_dir:
            tensorboard_log_dir = Path(tensorboard_log_dir)
            tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=str(tensorboard_log_dir),
                    histogram_freq=1
                )
            )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history = history
        
        logger.info("✅ Training complete")
        
        return history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        }
        
        # Directional accuracy
        y_test_direction = np.diff(y_test) > 0
        y_pred_direction = np.diff(y_pred.flatten()) > 0
        metrics['directional_accuracy'] = np.mean(y_test_direction == y_pred_direction) * 100
        
        logger.info(f"✅ Test RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        logger.info(f"   Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        return metrics
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 Saved training history plot to {save_path}")
        
        plt.close()
    
    def save_model(self, filepath: Path):
        """
        Save trained model.
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        logger.info(f"💾 Saved model to {filepath}")
    
    def load_model(self, filepath: Path):
        """
        Load trained model.
        
        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"✅ Loaded model from {filepath}")


def main():
    """
    Main function to demonstrate usage.
    """
    logger.info("LSTM Model module - use in training pipeline")


if __name__ == "__main__":
    main()
