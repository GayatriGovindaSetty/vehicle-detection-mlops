import os
import torch
from pathlib import Path
from ultralytics import YOLO
import wandb
import logging
from datetime import datetime
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train YOLO model with WandB logging"""
    
    def __init__(self, data_yaml_path: str, model_name: str = "yolov8n.pt"):
        self.data_yaml_path = data_yaml_path
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize WandB
        self.init_wandb()
        
    def init_wandb(self):
        """Initialize Weights & Biases"""
        wandb_api_key = config.get_wandb_api_key()
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
            wandb.login()
            
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                name=f"{config.model.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": self.model_name,
                    "epochs": config.model.epochs,
                    "batch_size": config.model.batch_size,
                    "img_size": config.model.img_size,
                    "device": self.device
                }
            )
            logger.info("WandB initialized")
        else:
            logger.warning("WandB API key not found. Training without WandB logging.")
    
    def train(self, epochs: int = None, batch_size: int = None, 
              img_size: int = None, save_dir: str = None):
        """Train the model"""
        
        # Use config values if not specified
        epochs = epochs or config.model.epochs
        batch_size = batch_size or config.model.batch_size
        img_size = img_size or config.model.img_size
        save_dir = save_dir or str(config.models_root)
        
        logger.info(f"Starting training with {epochs} epochs...")
        
        # Load model
        model = YOLO(self.model_name)
        
        # Train
        results = model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            project=save_dir,
            name=config.model.experiment_name,
            verbose=True,
            patience=10,
            save=True,
            plots=True,
            workers=config.model.workers,
            lr0=config.model.learning_rate,
        )
        
        logger.info("Training completed!")
        
        # Get best model path
        best_model_path = Path(save_dir) / config.model.experiment_name / "weights" / "best.pt"
        
        # Save to models directory
        final_model_path = config.models_root / "best.pt"
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            logger.info(f"Best model saved to {final_model_path}")
        
        # Log to WandB
        if wandb.run is not None:
            wandb.log({
                "final_mAP": results.results_dict.get('metrics/mAP50(B)', 0),
                "final_precision": results.results_dict.get('metrics/precision(B)', 0),
                "final_recall": results.results_dict.get('metrics/recall(B)', 0),
            })
            
            # Save model to WandB
            if config.wandb.log_model:
                artifact = wandb.Artifact('vehicle-detection-model', type='model')
                artifact.add_file(str(final_model_path))
                wandb.log_artifact(artifact)
        
        return results, final_model_path
    
    def validate(self, model_path: str = None):
        """Validate the trained model"""
        model_path = model_path or str(config.models_root / "best.pt")
        logger.info(f"Validating model: {model_path}")
        
        model = YOLO(model_path)
        results = model.val(data=self.data_yaml_path)
        
        logger.info("Validation completed!")
        logger.info(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer(
        data_yaml_path="data/processed/data.yaml",
        model_name="yolov8n.pt"
    )
    
    try:
        results, model_path = trainer.train()
        trainer.validate(model_path)
    finally:
        trainer.cleanup()
