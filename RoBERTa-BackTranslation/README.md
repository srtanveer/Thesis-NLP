# RoBERTa with Back-Translation Data Augmentation

This directory contains the **RoBERTa model trained with back-translation augmented data** for sarcasm detection.

## Purpose
Improve sarcasm detection through data augmentation using back-translation techniques.

## Files
- `roberta_backtranslation_sarcasm.py` - Training script with augmented data
- `preprocessed_data.csv` - Back-translated dataset
- `requirements.txt` - Package dependencies
- `install.sh` - CPU installation script
- `install_gpu.sh` - GPU installation script

## Quick Start

### GPU Server:
```bash
# Install dependencies
bash install_gpu.sh

# Run training
python3 roberta_backtranslation_sarcasm.py
```

### CPU (for testing):
```bash
# Install dependencies
bash install.sh

# Run training (will be slower)
python3 roberta_backtranslation_sarcasm.py
```

## Approach
- **Data Augmentation**: Back-translation to increase dataset diversity
- **Model**: Standard RoBERTa-base
- **Training Data**: Original + back-translated samples
- **Goal**: Improve generalization through data variety

## Dataset
- **Original**: Sarcasm detection corpus
- **Augmented**: Back-translated variations
- **Format**: CSV with preprocessed samples
- **Classes**: Binary (sarcastic vs non-sarcastic)

## Training Configuration
Configured for 50% back-translation augmentation to balance original and synthetic data.

## Expected Output
- Model trained on augmented dataset
- Evaluation metrics
- Comparison with non-augmented baseline

## Research Context
This approach tests whether data augmentation via back-translation improves sarcasm detection compared to:
- Standard RoBERTa baseline
- IDL-RoBERTa with incongruity features
