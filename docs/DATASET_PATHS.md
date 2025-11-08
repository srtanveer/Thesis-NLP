# Dataset Paths Configuration Guide

## ✅ Current Status: FIXED

All dataset paths have been updated to work with the new directory structure.

## 📁 Dataset Location

The datasets are now centrally located in:
```
Server/datasets/
├── Sarcasm_Headlines_Dataset_v2.json
└── Sarcasm_Headlines_Dataset.json
```

## 🔧 Path Configuration by File

### 1. IDL-RoBERTa (Enhanced Model)

#### `IDL-RoBERTa/idl_roberta_sarcasm.py`
**Status**: ✅ FIXED - Updated with relative path
```python
possible_paths = [
    '/kaggle/input/sarcasm-headlines-dataset/Sarcasm_Headlines_Dataset_v2.json',  # Kaggle
    '/kaggle/input/sarcasm-headlines-dataset-v2/Sarcasm_Headlines_Dataset_v2.json',  # Kaggle alt
    '../datasets/Sarcasm_Headlines_Dataset_v2.json',  # NEW: Relative to IDL-RoBERTa folder
    './Sarcasm_Headlines_Dataset_v2.json',  # Same directory
    'Sarcasm_Headlines_Dataset_v2.json'  # Current directory
]
```

#### `IDL-RoBERTa/idl_roberta_sarcasm.ipynb`
**Status**: ✅ OK - Uses Kaggle path (correct for Kaggle/Colab)
```python
dataset_path = '/kaggle/input/sarcasm-headlines-dataset/Sarcasm_Headlines_Dataset_v2.json'
```

---

### 2. RoBERTa Baseline

#### `RoBERTa-Baseline/roberta_baseline_sarcasm.py`
**Status**: ✅ FIXED - Updated with relative path
```python
possible_paths = [
    '/kaggle/input/sarcasm-headlines-dataset/Sarcasm_Headlines_Dataset_v2.json',  # Kaggle
    '/kaggle/input/sarcasm-headlines-dataset-v2/Sarcasm_Headlines_Dataset_v2.json',  # Kaggle alt
    '../datasets/Sarcasm_Headlines_Dataset_v2.json',  # NEW: Relative to RoBERTa-Baseline folder
    './Sarcasm_Headlines_Dataset_v2.json',  # Same directory
    'Sarcasm_Headlines_Dataset_v2.json'  # Current directory
]
```

#### `RoBERTa-Baseline/roberta_baseline_sarcasm.ipynb`
**Status**: ✅ OK - Uses Kaggle path (correct for Kaggle/Colab)
```python
dataset_path = '/kaggle/input/sarcasm-headlines-dataset/Sarcasm_Headlines_Dataset_v2.json'
```

---

### 3. RoBERTa Back-Translation

#### `RoBERTa-BackTranslation/roberta_backtranslation_sarcasm.py`
**Status**: ✅ FIXED - Added multi-path detection
```python
possible_paths = [
    '../datasets/Sarcasm_Headlines_Dataset_v2.json',  # NEW: Relative path
    'Datasets/Sarcasm_Headlines_Dataset_v2.json',  # Old path (backward compatible)
    './Sarcasm_Headlines_Dataset_v2.json',  # Same directory
    'Sarcasm_Headlines_Dataset_v2.json'  # Current directory
]
```

---

## 🚀 Usage Instructions

### For Local GPU Server (from model directories):

```bash
# From IDL-RoBERTa directory
cd IDL-RoBERTa
python3 idl_roberta_sarcasm.py
# ✅ Will find dataset at: ../datasets/Sarcasm_Headlines_Dataset_v2.json

# From RoBERTa-Baseline directory
cd RoBERTa-Baseline
python3 roberta_baseline_sarcasm.py
# ✅ Will find dataset at: ../datasets/Sarcasm_Headlines_Dataset_v2.json

# From RoBERTa-BackTranslation directory
cd RoBERTa-BackTranslation
python3 roberta_backtranslation_sarcasm.py
# ✅ Will find dataset at: ../datasets/Sarcasm_Headlines_Dataset_v2.json
```

### For Kaggle/Colab (using notebooks):

1. Upload the `.ipynb` file to Kaggle/Colab
2. Add dataset as input: `/kaggle/input/sarcasm-headlines-dataset/`
3. Run all cells
4. ✅ Will automatically use: `/kaggle/input/sarcasm-headlines-dataset/Sarcasm_Headlines_Dataset_v2.json`

### Alternative: Copy dataset to script directory

If you prefer, you can copy the dataset file:
```bash
# Copy to IDL-RoBERTa directory
cp datasets/Sarcasm_Headlines_Dataset_v2.json IDL-RoBERTa/

# Copy to RoBERTa-Baseline directory
cp datasets/Sarcasm_Headlines_Dataset_v2.json RoBERTa-Baseline/

# Copy to RoBERTa-BackTranslation directory
cp datasets/Sarcasm_Headlines_Dataset_v2.json RoBERTa-BackTranslation/
```

Then the scripts will find it in the same directory.

---

## 🔍 How Path Detection Works

All Python scripts now use **smart path detection**:

1. **Try Kaggle paths first** (for cloud environments)
2. **Try relative path** `../datasets/` (for local organized structure)
3. **Try current directory** `./` (if dataset copied locally)
4. **Try filename only** (Python path resolution)
5. **Raise clear error** if not found with list of tried paths

This ensures maximum compatibility across:
- ✅ Kaggle platform
- ✅ Google Colab
- ✅ Local GPU servers
- ✅ Development machines
- ✅ Different working directories

---

## 📊 Verification

To verify all paths are working:

```bash
cd /Users/idontbyte/Desktop/THESIS/Implementation/Server

# Check dataset exists
ls -lh datasets/Sarcasm_Headlines_Dataset_v2.json

# Test each script (dry run - will show if dataset loads)
cd IDL-RoBERTa && python3 -c "import os; print('Dataset found!' if os.path.exists('../datasets/Sarcasm_Headlines_Dataset_v2.json') else 'Dataset NOT found')"
cd ../RoBERTa-Baseline && python3 -c "import os; print('Dataset found!' if os.path.exists('../datasets/Sarcasm_Headlines_Dataset_v2.json') else 'Dataset NOT found')"
cd ../RoBERTa-BackTranslation && python3 -c "import os; print('Dataset found!' if os.path.exists('../datasets/Sarcasm_Headlines_Dataset_v2.json') else 'Dataset NOT found')"
```

---

## ✅ Summary

**All dataset paths are now correctly configured!**

- ✅ Python scripts: Multi-path detection including `../datasets/`
- ✅ Notebooks: Kaggle paths for cloud platforms
- ✅ Backward compatible: Still checks old paths
- ✅ Clear error messages: Shows all tried paths if not found

**No manual path editing needed** - the scripts will automatically find the dataset in the new structure! 🎉
