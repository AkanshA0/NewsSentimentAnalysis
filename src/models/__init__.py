"""
Models package initialization
"""

from .baseline_models import BaselineModels

# Try to import TensorFlow-dependent models
try:
    from .lstm_model import LSTMModel
    from .ensemble_model import EnsembleModel
    __all__ = ['BaselineModels', 'LSTMModel', 'EnsembleModel']
except (ImportError, OSError):
    # TensorFlow not available (DLL error)
    __all__ = ['BaselineModels']
