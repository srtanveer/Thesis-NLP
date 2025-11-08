# RoBERTa Baseline for Sarcasm Detection

This directory contains the **baseline RoBERTa model** for sarcasm detection without any incongruity features.

## Purpose
Compare standard RoBERTa performance with the enhanced IDL-RoBERTa model to measure the improvement gained from incongruity features.

## Files
- `roberta_baseline_sarcasm.ipynb` - Jupyter notebook for Kaggle/Colab
- `roberta_baseline_sarcasm.py` - Python script for GPU servers
- `requirements.txt` - Package dependencies

## Quick Start

### For Kaggle/Colab:
1. Upload `roberta_baseline_sarcasm.ipynb`
2. Upload dataset to `/kaggle/input/sarcasm-headlines-dataset/`
3. Run all cells

### For GPU Server:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python3 roberta_baseline_sarcasm.py
```

## Model Details
- **Architecture**: RoBERTa-base (125M parameters)
- **Task**: Binary classification (sarcastic vs non-sarcastic)
- **No custom features**: Standard sequence classification
- **Training**: 5 epochs, batch size 16, learning rate 2e-5

## Dataset
- **Name**: Sarcasm Headlines Dataset v2
- **Source**: News headlines from The Onion (sarcastic) and HuffPost (non-sarcastic)
- **Samples**: ~28,620 headlines
- **Split**: 80% train, 20% test

## Expected Output
- Classification report (precision, recall, F1)
- Predictions CSV file
- Results summary text file
- Sample predictions with true/predicted labels

## Comparison
Use results from this baseline to compare with:
- IDL-RoBERTa (enhanced with incongruity features)
- RoBERTa-BackTranslation (data augmentation approach)
