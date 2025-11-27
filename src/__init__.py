"""Vehicle Detection MLOps Package"""
__version__ = "1.0.0"
__author__ = "Your Name"

from .config import config
from .inference import VehicleDetector

# Lazy imports for training components (not needed in production)
def __getattr__(name):
    if name == 'ModelTrainer':
        from .train import ModelTrainer
        return ModelTrainer
    elif name == 'DataPreparator':
        from .data_preparation import DataPreparator
        return DataPreparator
    elif name == 'ModelEvaluator':
        from .evaluate import ModelEvaluator
        return ModelEvaluator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'config',
    'VehicleDetector',
    'ModelTrainer',
    'DataPreparator',
    'ModelEvaluator',
]