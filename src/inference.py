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
        self.class_names = config.data.class_names
        
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
            max_det=config.deployment.max_detections
        )
        
        if return_annotated:
            annotated_image = self.annotate_image(image, results[0])
            return results[0], annotated_image
        
        return results[0]
    
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
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Assume BGR, convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf.cpu().numpy())
            class_id = int(box.cls.cpu().numpy())
            label = self.class_names[class_id]
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label with background
            label_text = f'{label}: {conf:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            cv2.rectangle(
                img, (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), (255, 0, 0), -1
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
