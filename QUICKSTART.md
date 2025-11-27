# 🚀 Quick Start Guide (Google Colab)

Complete setup in 5 steps!

## Step 1: Setup Repository (2 min)

```bash
# Clone repository
git clone https://github.com/yourusername/vehicle-detection-mlops.git
cd vehicle-detection-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Dataset (3 min)

### Upload to Google Drive

1. Download your dataset from source
2. Extract and upload to Google Drive
3. Structure should be:
```
MyDrive/vehicle-detection-dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
```

## Step 3: Train on Google Colab (30-45 min)

### 3.1 Open Colab
1. Go to https://colab.research.google.com
2. File → Upload notebook
3. Upload `notebooks/train_colab.py` (or copy content)

### 3.2 Configure Runtime
1. Runtime → Change runtime type
2. Hardware accelerator → **GPU** (T4 recommended)
3. Save

### 3.3 Update Paths in Notebook
Find this section and update:
```python
# CHANGE THESE PATHS
DATASET_PATH = '/content/drive/MyDrive/vehicle-detection-dataset'
```

### 3.4 Run Training
1. Click "Run all" or run cells sequentially
2. When prompted, mount Google Drive
3. Training will take 30-45 minutes
4. Monitor progress in output

### 3.5 Download Model
Two options:
- **Option A**: Model auto-saves to Google Drive
- **Option B**: Download from cell output (best.pt)

## Step 4: Add Model to Repository (1 min)

```bash
# Copy downloaded model to your repo
cp ~/Downloads/best.pt models/best.pt

# Commit
git add models/best.pt
git commit -m "Add trained model"
```

## Step 5: Deploy (Automatic!)

### 5.1 Setup GitHub Secrets
1. Go to GitHub repository → Settings → Secrets
2. Add these secrets:
   - `HF_TOKEN`: Your Hugging Face token
   - `HF_USERNAME`: Your Hugging Face username
   - (Optional) `WANDB_API_KEY`: For experiment tracking

### 5.2 Push to Deploy
```bash
git push origin main
```

That's it! GitHub Actions will:
1. ✅ Run tests
2. ✅ Build Docker image
3. ✅ Deploy to Hugging Face Spaces

## 🎉 Done!

Your app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/vehicle-detection
```

## 📊 Training Tips

### For Better Results:
- Use **GPU T4 or better** in Colab
- Increase **epochs to 50-100** for production
- Try **larger models** (yolov8s.pt, yolov8m.pt)
- Add **data augmentation**

### Monitor Training:
```python
# In Colab, enable WandB (optional)
wandb.login(key='YOUR_KEY')
```

## 🆘 Troubleshooting

### Colab Disconnects
- Colab free tier has time limits
- Save checkpoints frequently
- Use `resume=True` to continue training

### Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 8  # Instead of 16
```

### Model Too Large for GitHub
```bash
# Use Git LFS
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

Or host model separately:
- Google Drive
- Hugging Face Model Hub
- AWS S3

## 📝 Next Steps

1. **Test locally**: `python app/app.py`
2. **Run tests**: `pytest tests/`
3. **Improve model**: Adjust hyperparameters
4. **Monitor deployment**: Check Hugging Face logs
5. **Share your app**: Send link to users!

---

**Time Breakdown:**
- Setup: 2 min
- Dataset prep: 3 min  
- Training: 30-45 min
- Deploy: 1 min + 5 min auto-deploy

**Total: ~40-50 minutes** ⚡