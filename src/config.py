import os
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    """Data configuration"""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    train_split: float = 0.7
    val_split: float = 0.1
    test_split: float = 0.2
    num_classes: int = 8
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [
                'auto', 'bus', 'car', 'lcv', 
                'motorcycle', 'multiaxle', 'tractor', 'truck'
            ]

@dataclass
class ModelConfig:
    """Model training configuration"""
    model_name: str = "yolov8n.pt"
    epochs: int = 30
    batch_size: int = 16
    img_size: int = 640
    device: str = "cuda"
    project_name: str = "vehicle-detection"
    experiment_name: str = "yolov8n-8classes"
    learning_rate: float = 0.01
    workers: int = 8
    
@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    project: str = "vehicle-detection-mlops"
    entity: str = None  # Set your wandb username
    log_model: bool = True
    
@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    model_path: str = "models/best.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    
class Config:
    """Main configuration class"""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.wandb = WandbConfig()
        self.deployment = DeploymentConfig()
        
        # Set paths
        self.project_root = Path(__file__).parent.parent
        self.data_root = self.project_root / "data"
        self.models_root = self.project_root / "models"
        
        # Create directories
        self.data_root.mkdir(exist_ok=True, parents=True)
        self.models_root.mkdir(exist_ok=True, parents=True)
        
    def get_wandb_api_key(self) -> str:
        """Get WandB API key from environment"""
        return os.getenv("WANDB_API_KEY", "")
    
    def get_hf_token(self) -> str:
        """Get Hugging Face token from environment"""
        return os.getenv("HF_TOKEN", "")

# Global config instance
config = Config()