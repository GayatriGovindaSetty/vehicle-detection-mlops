import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data_preparation import DataPreparator

class TestDataConfiguration:
    """Test data configuration"""
    
    def test_class_names_count(self):
        """Test number of classes"""
        assert config.data.num_classes == 8
        assert len(config.data.class_names) == 8
    
    def test_class_names_content(self):
        """Test class names"""
        expected = ['auto', 'bus', 'car', 'lcv', 
                   'motorcycle', 'multiaxle', 'tractor', 'truck']
        assert config.data.class_names == expected
    
    def test_data_splits(self):
        """Test data split configuration"""
        assert config.data.train_split == 0.7
        assert config.data.val_split == 0.1
        assert config.data.test_split == 0.2
        assert config.data.train_split + config.data.val_split + config.data.test_split == 1.0
    
    def test_data_paths(self):
        """Test data path configuration"""
        assert config.data.raw_data_path == "data/raw"
        assert config.data.processed_data_path == "data/processed"

class TestDataPreparator:
    """Test data preparation"""
    
    def test_preparator_initialization(self):
        """Test DataPreparator initialization"""
        preparator = DataPreparator(
            source_dir="data/raw",
            destination_root="data/processed",
            class_names=config.data.class_names
        )
        
        assert preparator.source_dir == Path("data/raw")
        assert preparator.destination_root == Path("data/processed")
        assert len(preparator.class_names) == 8

if __name__ == "__main__":
    pytest.main([__file__, "-v"])