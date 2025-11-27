# 📖 Complete Setup Guide

This guide will walk you through setting up the entire MLOps pipeline from scratch.

## 📋 Table of Contents

1. [Prerequisites Setup](#1-prerequisites-setup)
2. [Project Setup](#2-project-setup)
3. [Data Preparation](#3-data-preparation)
4. [Model Training](#4-model-training)
5. [Local Testing](#5-local-testing)
6. [CI/CD Configuration](#6-cicd-configuration)
7. [Deployment](#7-deployment)
8. [Monitoring](#8-monitoring)

---

## 1. Prerequisites Setup

### 1.1 Install Required Software

**Python 3.10+**
```bash
# Check Python version
python --version

# If needed, install Python 3.10
# Windows: Download from python.org
# Mac: brew install python@3.10
# Linux: sudo apt install python3.10
```

**Git**
```bash
# Check Git installation
git --version

# Install if needed
# Windows: Download from git-scm.com
# Mac: brew install git
# Linux: sudo apt install git
```

**Docker**
```bash
# Check Docker installation
docker --version

# Install from docker.com
```

### 1.2 Create Required Accounts

**GitHub Account**
- Sign up at https://github.com
- Create a new repository: `vehicle-detection-mlops`
- Set repository to public or private

**Kaggle Account**
1. Sign up at https://www.kaggle.com
2. Go to Account Settings
3. Create API Token (downloads kaggle.json)
4. Place in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\YourUser\.kaggle\kaggle.json` (Windows)

**Weights & Biases Account**
1. Sign up at https://wandb.ai
2. Go to Settings > API Keys
3. Copy your API key
4. Save for later use

**Hugging Face Account**
1. Sign up at https://huggingface.co
2. Go to Settings > Access Tokens
3. Create token with write permissions
4. Save token for later use

---

## 2. Project Setup

### 2.1 Clone and Initialize

```bash
# Clone the repository
git clone https://github.com/yourusername/vehicle-detection-mlops.git
cd vehicle-detection-mlops

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2.2 Configure Environment Variables

Create a `.env` file:

```bash
# .env file
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_hugging_face_token_here
HF_USERNAME=your_hugging_face_username
```

**Important**: Add `.env` to `.gitignore`

```bash
echo ".env" >> .gitignore
```

### 2.3 Create Directory Structure

```bash
# Create necessary directories
mkdir -p data/raw/train/images data/raw/train/labels
mkdir -p data/processed
mkdir -p models
mkdir -p tests
mkdir -p logs
```

---

## 3. Data Preparation

### 3.1 Download Dataset from Kaggle

**Method 1: Using Kaggle CLI**

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (replace with actual dataset name)
kaggle datasets download -d [dataset-id]/vehicle-detection-8-classes-object-detection

# Extract
unzip vehicle-detection-8-classes-object-detection.zip -d data/raw/

# Clean up
rm vehicle-detection-8-classes-object-detection.zip
```

**Method 2: Manual Download**

1. Go to Kaggle dataset page
2. Click "Download"
3. Extract to `data/raw/`

### 3.2 Verify Data Structure

```bash
# Check data structure
ls data/raw/train/images/  # Should show .jpg files
ls data/raw/train/labels/  # Should show .txt files

# Count files
echo "Images: $(ls data/raw/train/images/*.jpg | wc -l)"
echo "Labels: $(ls data/raw/train/labels/*.txt | wc -l)"
```

### 3.3 Prepare Dataset

```bash
# Run data preparation script
python -m src.data_preparation

# Verify processed data
ls data/processed/train/images/
ls data/processed/val/images/
ls data/processed/test/images/
```

---

## 4. Model Training

### 4.1 Choose Training Environment

You have three options for training:

#### Option A: Local Training (Requires GPU)

**Check GPU availability:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Train locally:**
```bash
# Set environment variable
export WANDB_API_KEY=your_key_here

# Run training
python -m src.train
```

#### Option B: Kaggle Notebook

1. **Create New Notebook**
   - Go to Kaggle > Code > New Notebook
   - Enable GPU accelerator (Settings > Accelerator > GPU T4 x2)

2. **Upload Code**
   ```python
   # In Kaggle notebook
   # Upload your source files or clone from GitHub
   
   !git clone https://github.com/yourusername/vehicle-detection-mlops.git
   %cd vehicle-detection-mlops
   
   # Install dependencies
   !pip install -r requirements.txt
   ```

3. **Run Training**
   ```python
   # Link to Kaggle dataset
   # Add dataset in "Input" section
   
   from src.data_preparation import DataPreparator
   from src.train import ModelTrainer
   
   # Prepare data
   preparator = DataPreparator(
       source_images='/kaggle/input/vehicle-detection/train/images',
       source_labels='/kaggle/input/vehicle-detection/train/labels',
       destination_root='/kaggle/working/dataset',
       class_names=['auto', 'bus', 'car', 'lcv', 'motorcycle', 'multiaxle', 'tractor', 'truck']
   )
   yaml_path = preparator.prepare()
   
   # Train model
   trainer = ModelTrainer(data_yaml_path=yaml_path)
   results, model_path = trainer.train()
   ```

4. **Download Trained Model**
   - After training completes
   - Download `models/best.pt`
   - Place in your local `models/` directory

#### Option C: Google Colab

1. **Open Colab**
   - Go to https://colab.research.google.com
   - Runtime > Change runtime type > GPU

2. **Clone Repository**
   ```python
   !git clone https://github.com/yourusername/vehicle-detection-mlops.git
   %cd vehicle-detection-mlops
   !pip install -r requirements.txt
   ```

3. **Mount Google Drive (Optional)**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Upload Dataset and Train**
   ```python
   # Upload dataset or link from Kaggle
   # Train as shown in Kaggle example
   ```

### 4.2 Monitor Training

**WandB Dashboard:**
```bash
# Training logs to: https://wandb.ai/your-entity/vehicle-detection-mlops
```

**Terminal Output:**
```bash
# Watch for:
# - Epoch progress
# - Loss decreasing
# - mAP increasing
# - Training time per epoch
```

### 4.3 Evaluate Model

```bash
# After training
python -m src.evaluate

# Check metrics
cat logs/evaluation_results.json
```

---

## 5. Local Testing

### 5.1 Test Model Inference

```bash
# Test single image
python -c "
from src.inference import VehicleDetector
detector = VehicleDetector('models/best.pt')
result, img = detector.detect('test_image.jpg')
print(detector.get_detection_stats(result))
"
```

### 5.2 Run Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### 5.3 Test Gradio App

```bash
# Run app
python app/app.py

# Open browser to http://localhost:7860
# Upload test images
# Verify detections work
```

### 5.4 Test Docker Container

```bash
# Build image
docker build -t vehicle-detection:test .

# Run container
docker run -p 7860:7860 vehicle-detection:test

# Test in browser
# Stop container: Ctrl+C
```

---

## 6. CI/CD Configuration

### 6.1 Configure GitHub Secrets

1. Go to your GitHub repository
2. Settings > Secrets and variables > Actions
3. Add the following secrets:

```
Name: WANDB_API_KEY
Value: your_wandb_api_key

Name: HF_TOKEN
Value: your_hugging_face_token

Name: HF_USERNAME
Value: your_hugging_face_username
```

### 6.2 Enable GitHub Actions

1. Go to Actions tab
2. Enable Actions if disabled
3. Workflows will trigger automatically on push

### 6.3 Test CI Pipeline

```bash
# Create feature branch
git checkout -b test-ci

# Make small change
echo "# Test" >> README.md

# Commit and push
git add .
git commit -m "Test CI pipeline"
git push origin test-ci

# Check Actions tab for pipeline status
```

### 6.4 Merge to Main

```bash
# After CI passes
git checkout main
git merge test-ci
git push origin main

# This triggers deployment pipeline
```

---

## 7. Deployment

### 7.1 Automatic Deployment to Hugging Face

**After pushing to main:**

1. GitHub Actions runs deployment workflow
2. Builds Docker image
3. Pushes to Hugging Face Space
4. Space builds and deploys

**Monitor deployment:**
- GitHub Actions tab shows progress
- Hugging Face Space logs show build status

### 7.2 Manual Deployment (Alternative)

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login
# Enter your token when prompted

# Create space
huggingface-cli repo create vehicle-detection --type space --space_sdk docker

# Clone space
git clone https://huggingface.co/spaces/yourusername/vehicle-detection hf-space

# Copy files
cd hf-space
cp -r ../app ../src ../models ../Dockerfile ../requirements.txt ../README.md .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### 7.3 Verify Deployment

1. Go to `https://huggingface.co/spaces/yourusername/vehicle-detection`
2. Wait for build to complete (5-10 minutes)
3. Test the interface
4. Upload test images
5. Verify detections work

### 7.4 Update Deployment

```bash
# Make changes locally
# Commit and push to main
git add .
git commit -m "Update model"
git push origin main

# GitHub Actions automatically deploys
# Or manually push to HF Space
```

---

## 8. Monitoring

### 8.1 Monitor Application

**Hugging Face Metrics:**
- Space logs: Check build/runtime logs
- Usage: Monitor API calls

**Application Logs:**
```bash
# View Gradio logs in Space
# Settings > Logs
```

### 8.2 Monitor Model Performance

**WandB Dashboard:**
- Training metrics
- Validation metrics
- Model comparisons

**Custom Monitoring:**
```python
# Add to your inference code
import logging

logging.basicConfig(filename='inference.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Log detections
logger.info(f"Detected {num_vehicles} vehicles")
```

### 8.3 Set Up Alerts

**GitHub Actions:**
- Email notifications on workflow failure
- Settings > Notifications

**WandB:**
- Set up alerts for metric thresholds
- Project > Alerts

---

## 🎉 Deployment Complete!

Your MLOps pipeline is now fully set up and deployed!

### Quick Access Links:

- **GitHub Repository**: https://github.com/yourusername/vehicle-detection-mlops
- **Hugging Face Space**: https://huggingface.co/spaces/yourusername/vehicle-detection
- **WandB Project**: https://wandb.ai/your-entity/vehicle-detection-mlops

### Next Steps:

1. ✅ Test the deployed application
2. ✅ Monitor performance metrics
3. ✅ Iterate and improve model
4. ✅ Add new features
5. ✅ Scale infrastructure if needed

---

## 🆘 Need Help?

- **Issues**: Open issue on GitHub
- **Discussions**: GitHub Discussions tab
- **Documentation**: Check README.md
- **Community**: Join Ultralytics Discord

Happy detecting! 🚗🤖