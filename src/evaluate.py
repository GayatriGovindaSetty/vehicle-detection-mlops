import json
import logging
from pathlib import Path
from ultralytics import YOLO
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate trained YOLO model"""
    
    def __init__(self, model_path: str = None, data_yaml: str = None):
        self.model_path = model_path or str(config.models_root / "best.pt")
        self.data_yaml = data_yaml or "data/processed/data.yaml"
        
    def evaluate(self):
        """Run evaluation on test set"""
        logger.info(f"Evaluating model: {self.model_path}")
        
        model = YOLO(self.model_path)
        results = model.val(data=self.data_yaml, split='test')
        
        # Extract metrics
        metrics = {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }
        
        # Log results
        logger.info(f"Results: {json.dumps(metrics, indent=2)}")
        
        # Save to file
        output_path = Path("logs/evaluation_results.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate()
