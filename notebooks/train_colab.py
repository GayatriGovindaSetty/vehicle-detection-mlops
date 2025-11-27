"""
Google Colab Training Script for Vehicle Detection
Upload this to Colab and run with GPU enabled

Instructions:
1. Upload to Google Colab
2. Runtime > Change runtime type > GPU (T4 or better)
3. Mount Google Drive with your dataset
4. Run all cells
5. Download trained model
"""

# ==============================================================================
# CELL 1: Install Dependencies
# ==============================================================================
print("📦 Installing dependencies...")
!pip install -q ultralytics wandb

# ==============================================================================
# CELL 2: Mount Google Drive
# ==============================================================================
print("📁 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Your dataset should be in Google Drive at:
# /content/drive/MyDrive/vehicle-detection-dataset/

# ==============================================================================
# CELL 3: Import Libraries
# ==============================================================================
print("📚 Importing libraries...")
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from ultralytics import YOLO
import torch
import wandb
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# CELL 4: Configuration
# ==============================================================================
print("⚙️ Setting up configuration...")

# CHANGE THESE PATHS TO YOUR GOOGLE DRIVE LOCATION
DATASET_PATH = '/content/drive/MyDrive/vehicle-detection-dataset'
OUTPUT_PATH = '/content/vehicle-detection-output'

# Training parameters
EPOCHS = 30
BATCH_SIZE = 16
IMG_SIZE = 640
MODEL_NAME = 'yolov8n.pt'

# Class names
CLASS_NAMES = ['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck']

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# ==============================================================================
# CELL 5: Prepare Dataset
# ==============================================================================
print("🔄 Preparing dataset...")

# Create working directory
os.makedirs(f'{OUTPUT_PATH}/dataset/train/images', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/dataset/train/labels', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/dataset/val/images', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/dataset/val/labels', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/dataset/test/images', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/dataset/test/labels', exist_ok=True)

# Get source paths
source_images = f'{DATASET_PATH}/train/images'
source_labels = f'{DATASET_PATH}/train/labels'

# Get all images
images = [f for f in os.listdir(source_images) if f.endswith('.jpg')]
labels = [f.replace('.jpg', '.txt') for f in images]

print(f"Found {len(images)} images")

# Split dataset: 70% train, 10% val, 20% test
train_val_imgs, test_imgs, train_val_lbls, test_lbls = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    train_val_imgs, train_val_lbls, test_size=0.125, random_state=42
)

print(f"Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

# Copy files
def copy_files(imgs, lbls, split):
    print(f"Copying {split} files...")
    for img, lbl in tqdm(zip(imgs, lbls), total=len(imgs)):
        # Copy image
        src = os.path.join(source_images, img)
        dst = f'{OUTPUT_PATH}/dataset/{split}/images/{img}'
        if os.path.exists(src):
            shutil.copy2(src, dst)
        
        # Copy label
        src = os.path.join(source_labels, lbl)
        dst = f'{OUTPUT_PATH}/dataset/{split}/labels/{lbl}'
        if os.path.exists(src):
            shutil.copy2(src, dst)

copy_files(train_imgs, train_lbls, 'train')
copy_files(val_imgs, val_lbls, 'val')
copy_files(test_imgs, test_lbls, 'test')

# ==============================================================================
# CELL 6: Create YAML Configuration
# ==============================================================================
print("📝 Creating YAML configuration...")

data_yaml = {
    'path': f'{OUTPUT_PATH}/dataset',
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': len(CLASS_NAMES),
    'names': CLASS_NAMES
}

yaml_path = f'{OUTPUT_PATH}/dataset/data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"YAML saved to: {yaml_path}")

# ==============================================================================
# CELL 7: Initialize WandB (Optional)
# ==============================================================================
print("📊 Initializing WandB...")

# OPTIONAL: Login to WandB for experiment tracking
# Uncomment and add your API key
# wandb.login(key='YOUR_WANDB_API_KEY')
# wandb.init(project='vehicle-detection-colab', name=f'run-{EPOCHS}epochs')

print("⚠️ WandB login skipped. Uncomment code above to enable tracking.")

# ==============================================================================
# CELL 8: Train Model
# ==============================================================================
print("🚀 Starting training...")

# Load model
model = YOLO(MODEL_NAME)

# Train
results = model.train(
    data=yaml_path,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=device,
    project=OUTPUT_PATH,
    name='vehicle-detection',
    verbose=True,
    patience=10,
    save=True,
    plots=True,
    workers=2,  # Lower for Colab
    lr0=0.01,
)

print("✅ Training completed!")

# ==============================================================================
# CELL 9: Validate Model
# ==============================================================================
print("🔍 Validating model...")

# Run validation
metrics = model.val()

print("\n📊 Validation Results:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# ==============================================================================
# CELL 10: Test on Sample Images
# ==============================================================================
print("🖼️ Testing on sample images...")

import matplotlib.pyplot as plt
import cv2
import random

# Get test images
test_image_dir = f'{OUTPUT_PATH}/dataset/test/images'
test_images = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
sample_images = random.sample(test_images, min(3, len(test_images)))

# Load best model
best_model = YOLO(f'{OUTPUT_PATH}/vehicle-detection/weights/best.pt')

fig, axes = plt.subplots(len(sample_images), 1, figsize=(12, 4*len(sample_images)))
if len(sample_images) == 1:
    axes = [axes]

for i, img_name in enumerate(sample_images):
    img_path = os.path.join(test_image_dir, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = best_model(img_path)
    
    # Plot
    axes[i].imshow(img_rgb)
    axes[i].set_title(f'Image: {img_name}')
    axes[i].axis('off')
    
    # Draw detections
    for pred in results[0].boxes:
        x1, y1, x2, y2 = pred.xyxy[0].cpu().numpy()
        conf = float(pred.conf.cpu().numpy())
        class_id = int(pred.cls.cpu().numpy())
        label = CLASS_NAMES[class_id]
        
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, color='red', linewidth=2)
        axes[i].add_patch(rect)
        axes[i].text(x1, y1-5, f'{label}: {conf:.2f}', 
                    color='white', fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/sample_detections.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Sample detections saved to: {OUTPUT_PATH}/sample_detections.png")

# ==============================================================================
# CELL 11: Save Model to Google Drive
# ==============================================================================
print("💾 Saving model to Google Drive...")

# Copy best model to Google Drive
drive_model_path = '/content/drive/MyDrive/vehicle-detection-model'
os.makedirs(drive_model_path, exist_ok=True)

shutil.copy2(
    f'{OUTPUT_PATH}/vehicle-detection/weights/best.pt',
    f'{drive_model_path}/best.pt'
)

shutil.copy2(
    f'{OUTPUT_PATH}/vehicle-detection/weights/last.pt',
    f'{drive_model_path}/last.pt'
)

print(f"✅ Model saved to Google Drive: {drive_model_path}")

# ==============================================================================
# CELL 12: Download Model (Alternative)
# ==============================================================================
print("⬇️ Preparing model for download...")

from google.colab import files

# Download best model
print("Downloading best.pt...")
files.download(f'{OUTPUT_PATH}/vehicle-detection/weights/best.pt')

print("✅ Model downloaded! Upload to your GitHub repository models/ folder")

# ==============================================================================
# CELL 13: Summary
# ==============================================================================
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print(f"\n📊 Final Metrics:")
print(f"   mAP50: {metrics.box.map50:.4f}")
print(f"   mAP50-95: {metrics.box.map:.4f}")
print(f"   Precision: {metrics.box.mp:.4f}")
print(f"   Recall: {metrics.box.mr:.4f}")
print(f"\n📁 Model Locations:")
print(f"   Best model: {OUTPUT_PATH}/vehicle-detection/weights/best.pt")
print(f"   Google Drive: {drive_model_path}/best.pt")
print(f"\n📝 Next Steps:")
print("   1. Download best.pt from Google Drive")
print("   2. Upload to GitHub repository: models/best.pt")
print("   3. Push to main branch to trigger deployment")
print("   4. Your app will deploy automatically to Hugging Face!")
print("="*60)