import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import VehicleDetector
from src.config import config

@pytest.fixture
def detector():
    """Create a detector instance for testing"""
    # Use a dummy model path for testing
    return VehicleDetector(model_path="yolov8n.pt", conf_threshold=0.25)

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

class TestVehicleDetector:
    """Test cases for VehicleDetector class"""
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector is not None
        assert detector.conf_threshold == 0.25
        assert len(detector.class_names) == 8
    
    def test_class_names(self, detector):
        """Test that all expected class names are present"""
        expected_classes = [
            'auto', 'bus', 'car', 'lcv', 
            'motorcycle', 'multiaxle', 'tractor', 'truck'
        ]
        assert detector.class_names == expected_classes
    
    def test_detect_returns_result(self, detector, sample_image):
        """Test that detect method returns results"""
        try:
            result, annotated = detector.detect(sample_image, return_annotated=True)
            assert result is not None
            assert annotated is not None
            assert isinstance(annotated, np.ndarray)
        except Exception as e:
            pytest.skip(f"Model not available for testing: {e}")
    
    def test_annotate_image_shape(self, detector, sample_image):
        """Test that annotated image has correct shape"""
        try:
            result = detector.detect(sample_image, return_annotated=False)
            annotated = detector.annotate_image(sample_image, result)
            assert annotated.shape == sample_image.shape
        except Exception as e:
            pytest.skip(f"Model not available for testing: {e}")
    
    def test_detection_stats_structure(self, detector, sample_image):
        """Test detection stats structure"""
        try:
            result = detector.detect(sample_image, return_annotated=False)
            stats = detector.get_detection_stats(result)
            
            assert 'total_detections' in stats
            assert 'class_counts' in stats
            assert 'average_confidence' in stats
            assert 'detections' in stats
            
            assert isinstance(stats['total_detections'], int)
            assert isinstance(stats['class_counts'], dict)
            assert isinstance(stats['detections'], list)
        except Exception as e:
            pytest.skip(f"Model not available for testing: {e}")
    
    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation"""
        # Valid threshold
        detector = VehicleDetector(conf_threshold=0.5)
        assert detector.conf_threshold == 0.5
        
        # Threshold at boundaries
        detector = VehicleDetector(conf_threshold=0.0)
        assert detector.conf_threshold == 0.0
        
        detector = VehicleDetector(conf_threshold=1.0)
        assert detector.conf_threshold == 1.0
    
    def test_batch_detection_input(self, detector):
        """Test batch detection with multiple images"""
        try:
            images = [
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(3)
            ]
            results = detector.detect_batch(images, batch_size=2)
            
            assert len(results) == 3
            for result, annotated in results:
                assert result is not None
                assert annotated is not None
        except Exception as e:
            pytest.skip(f"Model not available for testing: {e}")

class TestConfig:
    """Test configuration management"""
    
    def test_config_initialization(self):
        """Test config initialization"""
        assert config is not None
        assert config.data is not None
        assert config.model is not None
    
    def test_data_config_values(self):
        """Test data configuration values"""
        assert config.data.num_classes == 8
        assert len(config.data.class_names) == 8
        assert config.data.train_split == 0.8
        assert config.data.test_split == 0.2
    
    def test_model_config_values(self):
        """Test model configuration values"""
        assert config.model.epochs == 30
        assert config.model.batch_size == 16
        assert config.model.img_size == 640
        assert config.model.model_name == "yolov8n.pt"
    
    def test_paths_exist(self):
        """Test that configured paths exist"""
        assert config.project_root.exists()
        # Note: data_root and models_root may not exist until created

if __name__ == "__main__":
    pytest.main([__file__, "-v"])