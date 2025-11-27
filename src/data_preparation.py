import os
import shutil
from pathlib import Path
from typing import Tuple, List
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparator:
    """Prepare dataset for YOLO training"""
    
    def __init__(self, source_dir: str, destination_root: str, class_names: List[str]):
        """
        Initialize DataPreparator
        
        Args:
            source_dir: Path to source dataset (with train/images and train/labels)
            destination_root: Path where processed data will be saved
            class_names: List of class names
        """
        self.source_dir = Path(source_dir)
        self.destination_root = Path(destination_root)
        self.class_names = class_names
        
    def create_directory_structure(self):
        """Create the dataset directory structure"""
        logger.info("Creating directory structure...")
        
        dirs = [
            self.destination_root / 'train' / 'images',
            self.destination_root / 'train' / 'labels',
            self.destination_root / 'val' / 'images',
            self.destination_root / 'val' / 'labels',
            self.destination_root / 'test' / 'images',
            self.destination_root / 'test' / 'labels',
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def split_dataset(self, test_size: float = 0.2, val_size: float = 0.1, 
                     random_state: int = 42) -> Tuple[List, List, List]:
        """Split dataset into train, val, and test sets"""
        logger.info("Splitting dataset...")
        
        # Get all image files from source
        source_images = self.source_dir / 'train' / 'images'
        images = [f.name for f in source_images.glob('*.jpg')]
        labels = [f.replace('.jpg', '.txt') for f in images]
        
        # First split: train+val and test
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            images, labels, test_size=test_size, random_state=random_state
        )
        
        # Second split: train and val
        val_ratio = val_size / (1 - test_size)
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images, train_val_labels, test_size=val_ratio, random_state=random_state
        )
        
        logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
    
    def copy_files(self, images: List[str], labels: List[str], split: str):
        """Copy files to destination"""
        logger.info(f"Copying {split} files...")
        
        source_images = self.source_dir / 'train' / 'images'
        source_labels = self.source_dir / 'train' / 'labels'
        
        for img, lbl in tqdm(zip(images, labels), total=len(images), desc=f"Copying {split}"):
            # Copy image
            src_img = source_images / img
            dst_img = self.destination_root / split / 'images' / img
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_lbl = source_labels / lbl
            dst_lbl = self.destination_root / split / 'labels' / lbl
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
    
    def create_yaml_config(self):
        """Create YOLO data configuration file"""
        logger.info("Creating YAML configuration...")
        
        data_yaml = {
            'path': str(self.destination_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = self.destination_root / 'data.yaml'
        with open(yaml_path, 'w') as file:
            yaml.dump(data_yaml, file, default_flow_style=False)
        
        logger.info(f"YAML config saved to {yaml_path}")
        return yaml_path
    
    def prepare(self, test_size: float = 0.2, val_size: float = 0.1):
        """Complete data preparation pipeline"""
        logger.info("Starting data preparation pipeline...")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Split dataset
        train_data, val_data, test_data = self.split_dataset(test_size, val_size)
        
        # Copy files
        self.copy_files(*train_data, 'train')
        self.copy_files(*val_data, 'val')
        self.copy_files(*test_data, 'test')
        
        # Create YAML config
        yaml_path = self.create_yaml_config()
        
        logger.info("Data preparation completed!")
        return yaml_path

if __name__ == "__main__":
    from src.config import config
    
    # Example usage
    preparator = DataPreparator(
        source_dir="data/raw",
        destination_root="data/processed",
        class_names=config.data.class_names
    )
    
    preparator.prepare()
