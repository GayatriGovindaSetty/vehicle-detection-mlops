import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from ultralytics import YOLO
import logging
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleDetector:
    """Vehicle detection inference class"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = None, 
                 iou_threshold: float = None):
        self.model_path = model_path or config.deployment.model_path
        self.conf_threshold = conf_threshold or config.deployment.confidence_threshold
        self.iou_threshold = iou_threshold or config.deployment.iou_threshold
        
        logger.info(f"Loading model from {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Get class names from the model itself (handles both custom and COCO models)
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
            logger.info(f"Model loaded with {len(self.class_names)} classes: {self.class_names}")
        else:
            # Fallback to config class names
            self.class_names = config.data.class_names
            logger.warning(f"Using config class names: {self.class_names}")
        
    def detect(self, image: Union[str, np.ndarray], return_annotated: bool = True):
        """
        Detect vehicles in an image
        
        Args:
            image: Path to image or numpy array
            return_annotated: Whether to return annotated image
            
        Returns:
            results, annotated_image (if return_annotated=True)
        """
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=config.deployment.max_detections,
            verbose=False
        )
        
        result = results[0]
        logger.info(f"Detection completed: {len(result.boxes)} objects detected")
        
        if return_annotated:
            annotated_image = self.annotate_image(image, result)
            return result, annotated_image
        
        return result
    
    def annotate_image(self, image: Union[str, np.ndarray], result) -> np.ndarray:
        """
        Annotate image with detection results
        
        Args:
            image: Input image
            result: Detection result
            
        Returns:
            Annotated image
        """
        # Read image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            # Gradio provides RGB images, no conversion needed
            if not isinstance(img, np.ndarray):
                img = np.array(img)
        
        # Draw bounding boxes
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.cpu().numpy())
            class_id = int(box.cls.cpu().numpy())
            
            # Safely get class name
            if class_id < len(self.class_names):
                label = self.class_names[class_id]
            else:
                label = f"class_{class_id}"
                logger.warning(f"Unknown class_id: {class_id}, using {label}")
            
            # Use different colors for different classes
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0),  # Orange
                (128, 0, 255),  # Purple
            ]
            color = colors[class_id % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            label_text = f'{label}: {conf:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                img, (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), color, -1
            )
            
            cv2.putText(
                img, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return img
    
    def detect_batch(self, images: List[Union[str, np.ndarray]], 
                    batch_size: int = 8) -> List[Tuple]:
        """
        Detect vehicles in batch of images
        
        Args:
            images: List of image paths or numpy arrays
            batch_size: Batch size for inference
            
        Returns:
            List of (result, annotated_image) tuples
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self.model(
                batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=config.deployment.max_detections
            )
            
            for j, result in enumerate(batch_results):
                annotated = self.annotate_image(batch[j], result)
                results.append((result, annotated))
        
        return results
    
    def get_detection_stats(self, result) -> dict:
        """
        Get detection statistics
        
        Args:
            result: Detection result
            
        Returns:
            Dictionary with detection statistics
        """
        boxes = result.boxes
        
        stats = {
            'total_detections': len(boxes),
            'class_counts': {},
            'average_confidence': 0.0,
            'detections': []
        }
        
        if len(boxes) == 0:
            return stats
        
        # Calculate statistics
        confidences = []
        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            class_id = int(box.cls.cpu().numpy())
            label = self.class_names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            confidences.append(conf)
            
            if label not in stats['class_counts']:
                stats['class_counts'][label] = 0
            stats['class_counts'][label] += 1
            
            stats['detections'].append({
                'class': label,
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
        
        stats['average_confidence'] = np.mean(confidences)
        
        return stats

if __name__ == "__main__":
    # Example usage
    detector = VehicleDetector(model_path="models/best.pt")
    
    # Single image detection
    result, annotated = detector.detect("test_image.jpg")
    stats = detector.get_detection_stats(result)
    
    print(f"Total detections: {stats['total_detections']}")
    print(f"Class counts: {stats['class_counts']}")
    print(f"Average confidence: {stats['average_confidence']:.2f}")
