"""Vehicle Detection MLOps Package"""
__version__ = "1.0.0"
__author__ = "Your Name"

from .config import config
from .inference import VehicleDetector
from .train import ModelTrainer
from .data_preparation import DataPreparator
from .evaluate import ModelEvaluator

__all__ = [
    'config',
    'VehicleDetector',
    'ModelTrainer',
    'DataPreparator',
    'ModelEvaluator',
]