# GPU Server Setup Guide

## Quick Setup on Remote GPU Server

### Step 1: Clone the Repository
```bash
git clone git@github.com:srtanveer/Thesis-NLP.git
cd Thesis-NLP/Roberta\ BT/
```

### Step 2: Run GPU Installation Script
```bash
chmod +x install_gpu.sh
./install_gpu.sh
```

This script will:
- Detect your CUDA version automatically
- Install PyTorch with appropriate CUDA support
- Install all other dependencies
- Verify GPU availability

---

## Manual Installation (If Script Fails)

### Option 1: For CUDA 11.8
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Option 2: For CUDA 12.1
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Option 3: CPU Only
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Verify Installation

### Check CUDA Availability
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count())"
```

### Check GPU Info
```bash
nvidia-smi
```

---

## Run the Script

### Activate Environment and Run
```bash
source .venv/bin/activate
python 50-bt-roberta.py
```

### Or Run Directly
```bash
.venv/bin/python 50-bt-roberta.py
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Make sure you've activated the virtual environment:
```bash
source .venv/bin/activate
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in the script. Edit `50-bt-roberta.py`:
```python
per_device_train_batch_size=16,  # Change from 32 to 16 or 8
per_device_eval_batch_size=16,   # Change from 32 to 16 or 8
```

### Issue: PyTorch not detecting GPU
**Solution:** Reinstall PyTorch with correct CUDA version:
```bash
# Check your CUDA version first
nvcc --version

# Then install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "googletrans" errors during augmentation
**Solution:** The script will automatically fall back to the original dataset if back-translation fails. Check your internet connection.

---

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended for GPU Training
- CUDA-capable GPU (8GB+ VRAM)
- CUDA 11.8 or 12.1
- 16GB+ RAM
- 20GB disk space

---

## Expected Runtime

- **Data Loading & Preprocessing:** ~2-5 minutes
- **Back-Translation (50% of dataset):** ~30-60 minutes (depends on API rate limits)
- **Training (5 epochs):** 
  - With GPU: ~15-30 minutes
  - Without GPU: ~2-4 hours

---

## Output Files

After successful execution, you'll find these timestamped CSV files:
1. `predictions_YYYYMMDD_HHMMSS.csv`
2. `confusion_matrix_YYYYMMDD_HHMMSS.csv`
3. `classification_report_YYYYMMDD_HHMMSS.csv`
4. `training_summary_YYYYMMDD_HHMMSS.csv`
5. `augmented_data.csv`
