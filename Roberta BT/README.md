# RoBERTa with Back-Translation - Sarcasm Detection

## Overview
This script trains a RoBERTa model for sarcasm detection using back-translation data augmentation. Optimized for local GPU execution.

## Key Features

### 1. **GPU Optimization**
- Automatic GPU detection and utilization
- FP16 mixed precision training (when GPU available)
- Increased batch size (32) for better GPU utilization
- Parallel data loading with 4 workers
- Efficient memory management

### 2. **Enhanced Data Processing**
- Loads Sarcasm Headlines Dataset v2
- Samples 50% of data for training
- Back-translation augmentation (English → French → English)
- Automatic duplicate removal
- Increased tokenizer max_length to 128 (from 32) for better context

### 3. **Comprehensive Output Files**
All outputs are saved with timestamps to avoid overwriting:

1. **predictions_YYYYMMDD_HHMMSS.csv**
   - Tweet text
   - True labels and predicted labels
   - True class and predicted class names
   - Correct/Incorrect flag

2. **confusion_matrix_YYYYMMDD_HHMMSS.csv**
   - 2x2 confusion matrix
   - Actual vs Predicted breakdown

3. **classification_report_YYYYMMDD_HHMMSS.csv**
   - Precision, Recall, F1-Score for each class
   - Support (number of samples)
   - Macro and weighted averages

4. **training_summary_YYYYMMDD_HHMMSS.csv**
   - All evaluation metrics (Accuracy, F1, Precision, Recall)
   - Training time
   - Dataset statistics

5. **augmented_data.csv**
   - Complete augmented dataset used for training

## Requirements
```bash
pip install pandas numpy torch transformers scikit-learn nltk googletrans==3.1.0a0 tqdm
```

## Usage
```bash
python 50-bt-roberta.py
```

## Training Configuration
- **Model**: RoBERTa-base
- **Epochs**: 5
- **Batch Size**: 32 (train and eval)
- **Learning Rate**: 2e-5
- **Warmup Steps**: 500
- **Max Sequence Length**: 128
- **Weight Decay**: 0.01
- **Optimizer**: AdamW (default)

## Expected Output
The script provides a 10-step progress display:
1. Import libraries and NLTK data
2. Define back-translation function
3. Load and preprocess dataset
4. Sample 50% of dataset
5. Augment with back-translation
6. Prepare train/eval datasets
7. Set up training configuration
8. Initialize model and trainer
9. Train the model
10. Evaluate and generate results

## GPU vs CPU
- **GPU**: Uses FP16 mixed precision for faster training
- **CPU**: Falls back to FP32, slower but still functional

## Notes
- Back-translation uses Google Translate API (may be rate-limited)
- Training time depends on hardware and augmentation success
- All intermediate files are saved for inspection
