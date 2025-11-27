import cv2
import numpy as np
from typing import Tuple

def resize_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

def format_detection_text(stats: dict) -> str:
    """
    Format detection statistics for display
    
    Args:
        stats: Detection statistics dictionary
        
    Returns:
        Formatted text
    """
    text = f"Total Detections: {stats['total_detections']}\n\n"
    
    if stats['class_counts']:
        text += "Detected Classes:\n"
        for cls, count in stats['class_counts'].items():
            text += f"  • {cls}: {count}\n"
    
    if stats['average_confidence'] > 0:
        text += f"\nAverage Confidence: {stats['average_confidence']:.2%}"
    
    return text

def create_color_map(num_classes: int = 8) -> dict:
    """
    Create color map for different classes
    
    Args:
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class names to colors
    """
    colors = [
        (255, 0, 0),    # Red - auto
        (0, 255, 0),    # Green - bus
        (0, 0, 255),    # Blue - car
        (255, 255, 0),  # Yellow - lcv
        (255, 0, 255),  # Magenta - motorcycle
        (0, 255, 255),  # Cyan - multiaxle
        (128, 0, 128),  # Purple - tractor
        (255, 128, 0),  # Orange - truck
    ]
    
    class_names = ['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck']
    
    return {name: colors[i % len(colors)] for i, name in enumerate(class_names)}

def validate_image(image: np.ndarray) -> Tuple[bool, str]:
    """
    Validate input image
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "No image provided"
    
    if len(image.shape) != 3:
        return False, "Image must be RGB"
    
    if image.shape[2] != 3:
        return False, "Image must have 3 channels"
    
    return True, ""