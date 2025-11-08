# Sarcasm Detection Research - Main Project

This repository contains three different approaches to sarcasm detection for thesis research.

## 📁 Directory Structure

```
Server/
├── IDL-RoBERTa/                      # Enhanced RoBERTa with incongruity features
│   ├── idl_roberta_sarcasm.ipynb     # Notebook version
│   ├── idl_roberta_sarcasm.py        # Script version
│   ├── requirements.txt              # Dependencies
│   ├── setup_environment.sh          # Setup script
│   └── README.md                     # Detailed documentation
│
├── RoBERTa-Baseline/                 # Standard RoBERTa baseline
│   ├── roberta_baseline_sarcasm.ipynb # Notebook version
│   ├── roberta_baseline_sarcasm.py    # Script version
│   ├── requirements.txt               # Dependencies
│   └── README.md                      # Detailed documentation
│
├── RoBERTa-BackTranslation/          # RoBERTa with data augmentation
│   ├── roberta_backtranslation_sarcasm.py # Training script
│   ├── preprocessed_data.csv          # Augmented dataset
│   ├── requirements.txt               # Dependencies
│   ├── install.sh                     # CPU setup
│   ├── install_gpu.sh                 # GPU setup
│   └── README.md                      # Detailed documentation
│
├── datasets/                          # Centralized datasets
│   ├── Sarcasm_Headlines_Dataset_v2.json
│   ├── Sarcasm_Headlines_Dataset.json
│   └── ...
│
└── docs/                              # Documentation
    ├── IDL-RoBERTa-README.md
    ├── RoBERTa-BackTranslation-README.md
    ├── GPU_SETUP.md
    └── QUICK_START.sh
```

## 🎯 Research Objectives

Compare three approaches to sarcasm detection:

### 1. **RoBERTa Baseline** (Standard)
- Pure RoBERTa-base model
- No additional features
- Establishes performance baseline

### 2. **IDL-RoBERTa** (Enhanced)
- Commonsense incongruity detection
- Token-level attention mechanism
- VADER sentiment integration
- Custom architecture

### 3. **RoBERTa-BackTranslation** (Augmented)
- Data augmentation via back-translation
- Standard RoBERTa architecture
- Enhanced training data diversity

## 🚀 Quick Start

### Choose Your Approach:

**For Baseline Comparison:**
```bash
cd RoBERTa-Baseline
python3 roberta_baseline_sarcasm.py
```

**For Enhanced Model:**
```bash
cd IDL-RoBERTa
python3 idl_roberta_sarcasm.py
```

**For Augmented Data:**
```bash
cd RoBERTa-BackTranslation
python3 roberta_backtranslation_sarcasm.py
```

### Using Notebooks (Kaggle/Colab):
Each directory contains `.ipynb` files ready for cloud platforms.

## 📊 Dataset

**Sarcasm Headlines Dataset v2**
- **Source**: News headlines
- **Sarcastic**: The Onion
- **Non-sarcastic**: HuffPost  
- **Total Samples**: ~28,620
- **Format**: JSON (line-delimited)
- **Location**: `datasets/Sarcasm_Headlines_Dataset_v2.json`

## 🔧 Requirements

All approaches require:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Accelerate 0.26.0+
- GPU recommended (but works on CPU)

See individual `requirements.txt` in each directory.

## 📈 Expected Results

Each model produces:
- **Metrics**: Accuracy, F1, Precision, Recall
- **Reports**: Classification reports
- **Predictions**: CSV with true/predicted labels
- **Checkpoints**: Saved model weights

## 🎓 Thesis Context

This implementation compares different architectural and data augmentation strategies for sarcasm detection, measuring:
1. Impact of incongruity features (IDL-RoBERTa vs Baseline)
2. Effect of data augmentation (BackTranslation vs Baseline)
3. Trade-offs between complexity and performance

## 📝 Citation

If you use this code for research, please cite appropriately.

## 🤝 Contributing

This is thesis research code. For questions or collaboration, please contact the repository owner.

## 📄 License

See repository license file.

---

**Author**: Thesis Research Project  
**Date**: November 2025  
**Framework**: PyTorch + Hugging Face Transformers
