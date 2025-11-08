# IDL-RoBERTa for Sarcasm Detection

Enhanced model with Commonsense Incongruity Features for detecting sarcasm in news headlines.

## Quick Start

### Option 1: Automatic Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup_environment.sh

# Run setup
./setup_environment.sh

# Run the training script
python3 robarta_sarcasm_headlines.py
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install --upgrade pip
pip install accelerate>=0.26.0
pip install transformers>=4.30.0
pip install torch pandas numpy scikit-learn nltk tqdm

# Run the script
python3 robarta_sarcasm_headlines.py
```

### Option 3: Using requirements.txt

```bash
pip install -r requirements.txt
python3 robarta_sarcasm_headlines.py
```

## Important Notes

### Critical Dependency
The script requires **accelerate>=0.26.0**. If you see an error like:
```
ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`
```

Run this command:
```bash
pip install --upgrade accelerate>=0.26.0
```

### Dataset
The script expects `Sarcasm_Headlines_Dataset_v2.json` in one of these locations:
- `/kaggle/input/sarcasm-headlines-dataset/`
- Current directory
- Same directory as the script

### GPU Support
- The script automatically detects GPU availability
- Training on GPU is significantly faster
- CPU training is supported but slower

## Files

- `robarta_sarcasm_headlines.py` - Main training script
- `robarta.ipynb` - Jupyter notebook version (Kaggle-ready)
- `requirements.txt` - Python dependencies
- `setup_environment.sh` - Automated setup script
- `README.md` - This file

## Output

The script creates:
- `idl_roberta_results/` - Training outputs and checkpoints
- `final_results.txt` - Evaluation metrics report
- `predictions.csv` - Model predictions on test set

## Dataset Format

The script expects a JSON lines file with this format:
```json
{"is_sarcastic": 1, "headline": "...", "article_link": "..."}
{"is_sarcastic": 0, "headline": "...", "article_link": "..."}
```

## Model Architecture

- Base: RoBERTa-base
- Enhanced with token-level incongruity features
- Commonsense dictionary built from training data
- Sentiment-based feature extraction using VADER

## Training Configuration

- Epochs: 5
- Batch size: 16 (GPU) or 8 (CPU)
- Learning rate: 2e-5
- Train/Test split: 80/20
- Metric: F1 Score

## Troubleshooting

### Import Errors
If you see import errors, run:
```bash
pip install --upgrade accelerate transformers torch
```

### CUDA Out of Memory
Reduce batch size in the script (line ~420):
```python
batch_size = 8  # or even 4
```

### Dataset Not Found
Ensure the dataset file is in the correct location or update the path in line ~90.

## Support

For issues, check:
1. All dependencies are installed: `pip list | grep -E "accelerate|transformers|torch"`
2. GPU is detected: Check script output for "GPU detected"
3. Dataset path is correct: Check the "Found dataset at:" message

## License

MIT License - See repository for details.
