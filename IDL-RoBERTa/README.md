# IDL-RoBERTa: Incongruity Detection Enhanced RoBERTa

This directory contains the **enhanced IDL-RoBERTa model** with incongruity features for improved sarcasm detection.

## Purpose
Advanced sarcasm detection using commonsense incongruity features, token-level attention, and sentiment analysis.

## Files
- `idl_roberta_sarcasm.ipynb` - Jupyter notebook for Kaggle/Colab
- `idl_roberta_sarcasm.py` - Python script for GPU servers
- `requirements.txt` - Package dependencies
- `setup_environment.sh` - Environment setup script

## Quick Start

### For Kaggle/Colab:
1. Upload `idl_roberta_sarcasm.ipynb`
2. Upload dataset to `/kaggle/input/sarcasm-headlines-dataset/`
3. Run all cells (auto-installs dependencies)

### For GPU Server:
```bash
# Option 1: Use setup script
bash setup_environment.sh

# Option 2: Manual installation
pip install -r requirements.txt
python3 idl_roberta_sarcasm.py
```

## Model Features
- **Base Model**: RoBERTa-base
- **Enhanced Features**:
  - Commonsense incongruity detection
  - Token-level attention mechanism
  - Sentiment analysis (VADER)
  - Custom IDLRobertaForSarcasm architecture
- **Parameters**: ~125M (base) + custom layers

## Key Innovations
1. **Incongruity Features**: Detects contradictions using commonsense knowledge
2. **Token Attention**: Focuses on important tokens for sarcasm
3. **Sentiment Integration**: Combines sentiment polarity with context
4. **Multi-level Processing**: Sentence and token-level feature extraction

## Dataset
- **Name**: Sarcasm Headlines Dataset v2
- **Samples**: ~28,620 headlines
- **Split**: 80% train, 20% test
- **Classes**: Sarcastic (The Onion) vs Non-sarcastic (HuffPost)

## Training Configuration
- Epochs: 5
- Batch size: 16
- Learning rate: 2e-5
- Optimizer: AdamW with warmup
- Evaluation: F1 score optimization

## Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Accelerate >= 0.26.0
- NLTK (VADER sentiment)
- scikit-learn, pandas, numpy

## Expected Output
- Training metrics and loss curves
- Evaluation metrics (accuracy, F1, precision, recall)
- Classification report
- Predictions with confidence scores
- Model checkpoints

## Research Context
This model is part of a thesis research comparing different approaches to sarcasm detection:
- **Baseline**: Standard RoBERTa
- **IDL-RoBERTa**: This enhanced version
- **RoBERTa-BT**: Back-translation augmentation
