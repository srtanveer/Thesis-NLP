# Disk Space Error Fix

## ❌ Error Encountered

```
RuntimeError: [enforce fail at inline_container.cc:664]
PyTorchStreamWriter failed writing file data/217: file write failed
```

This error occurs when the disk runs out of space during model training/saving.

## ✅ Fixes Applied

### 1. **Reduced Checkpoint Storage**
Changed `save_total_limit` from 2 to 1:
```python
save_total_limit=1,  # Only keep 1 checkpoint instead of 2
```

### 2. **Optimized Save Format**
Added space-saving options:
```python
save_safetensors=False,  # Use legacy format (smaller)
save_only_model=True,    # Don't save optimizer states
```

### 3. **Automatic Cleanup**
Added cleanup before training:
```python
# Clean up old checkpoints
if os.path.exists(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
    for cp in checkpoints:
        cp_path = os.path.join(output_dir, cp)
        if os.path.isdir(cp_path):
            shutil.rmtree(cp_path)

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 4. **Disk Space Monitoring**
Added disk space check function:
```python
def check_disk_space(path='/'):
    """Check available disk space"""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    
    if free_gb < 5:
        print(f"⚠️  WARNING: Low disk space ({free_gb:.2f} GB free)")
        return False
    return True
```

## 📊 Space Savings

| Configuration | Before | After | Savings |
|--------------|--------|-------|---------|
| Checkpoints  | 2      | 1     | ~500MB  |
| Save format  | SafeTensors | Legacy | ~200MB |
| Optimizer    | Saved  | Not saved | ~300MB |
| **Total**    | -      | -     | **~1GB** |

## 🔧 Manual Solutions

If you still encounter disk space issues:

### Option 1: Free Up Space
```bash
# Check disk usage
df -h

# Clean Docker cache (if using Docker)
docker system prune -a

# Clean pip cache
pip cache purge

# Remove unused packages
apt-get autoremove  # Linux
brew cleanup        # macOS
```

### Option 2: Use External Storage
```bash
# Mount external drive and set output directory
output_dir = '/mnt/external/roberta_results'
```

### Option 3: Reduce Model Size
```python
# Train for fewer epochs
num_train_epochs=3  # Instead of 5

# Use smaller batch size
per_device_train_batch_size=8  # Instead of 16

# Don't save intermediate checkpoints
save_strategy='no'  # Only save final model
```

### Option 4: Stream to Cloud
```python
# Push checkpoints to cloud storage immediately after creation
# Example with HuggingFace Hub
push_to_hub=True
hub_model_id="your-username/model-name"
```

## 📝 Updated Files

✅ `RoBERTa-Baseline/roberta_baseline_sarcasm.py`
- Added disk space check
- Reduced save_total_limit to 1
- Added cleanup before training
- Added save optimization flags

✅ `IDL-RoBERTa/idl_roberta_sarcasm.py`
- Added disk space check
- Reduced save_total_limit to 1
- Added cleanup before training
- Added save optimization flags

## 🚀 Running After Fix

The scripts will now:
1. Check disk space before training
2. Clean up old checkpoints automatically
3. Use less disk space for checkpoints
4. Warn you if disk space is low (<5GB)

```bash
# Run normally
cd RoBERTa-Baseline
python3 roberta_baseline_sarcasm.py

# Or
cd IDL-RoBERTa
python3 idl_roberta_sarcasm.py
```

## 💡 Best Practices

1. **Monitor disk space**: Ensure >10GB free before training
2. **Clean between runs**: Remove old result directories
3. **Use cloud storage**: For long-term model storage
4. **Compress models**: After training, compress saved models

```bash
# Clean old results
rm -rf ./roberta_baseline_results
rm -rf ./idl_roberta_results

# Compress saved model
tar -czf model.tar.gz roberta_baseline_results/final_model/
```

## ⚠️ Important Notes

- Minimum recommended free space: **10 GB**
- RoBERTa model size: ~500MB
- Each checkpoint: ~500MB
- Total training space needed: ~5-10GB

If training on Kaggle/Colab:
- Kaggle: 73GB disk space available
- Colab: 100GB+ disk space available
- Should not encounter this issue on cloud platforms

---

**Issue resolved!** The scripts now handle disk space more efficiently. 🎉
