import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import VehicleDetector
from src.config import config

class TestVehicleDetector:
    """Test vehicle detector"""
    
    def test_detector_initialization(self):
        """Test detector initialization with default parameters"""
        detector = VehicleDetector()
        
        assert detector.conf_threshold == config.deployment.confidence_threshold
        assert detector.iou_threshold == config.deployment.iou_threshold
        assert len(detector.class_names) == 8
    
    def test_detector_custom_thresholds(self):
        """Test detector with custom thresholds"""
        detector = VehicleDetector(conf_threshold=0.5, iou_threshold=0.4)
        
        assert detector.conf_threshold == 0.5
        assert detector.iou_threshold == 0.4
    
    def test_class_names(self):
        """Test class names are correct"""
        detector = VehicleDetector()
        expected_classes = ['auto', 'bus', 'car', 'lcv', 
                          'motorcycle', 'multiaxle', 'tractor', 'truck']
        
        assert detector.class_names == expected_classes
    
    def test_stats_structure_empty(self):
        """Test stats structure with no detections"""
        detector = VehicleDetector()
        
        # Mock empty result
        class MockBoxes:
            def __len__(self):
                return 0
        
        class MockResult:
            boxes = MockBoxes()
        
        stats = detector.get_detection_stats(MockResult())
        
        assert stats['total_detections'] == 0
        assert stats['class_counts'] == {}
        assert stats['average_confidence'] == 0.0
        assert stats['detections'] == []

class TestInferenceUtils:
    """Test inference utility functions"""
    
    def test_threshold_boundaries(self):
        """Test threshold boundary conditions"""
        # Minimum threshold
        detector1 = VehicleDetector(conf_threshold=0.0)
        assert detector1.conf_threshold == 0.0
        
        # Maximum threshold
        detector2 = VehicleDetector(conf_threshold=1.0)
        assert detector2.conf_threshold == 1.0
        
        # Typical threshold
        detector3 = VehicleDetector(conf_threshold=0.25)
        assert detector3.conf_threshold == 0.25

if __name__ == "__main__":
    pytest.main([__file__, "-v"])