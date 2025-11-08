#!/usr/bin/env python3
"""
RoBERTa Baseline for Sarcasm Detection
Standard RoBERTa model without incongruity features
"""

# Install/upgrade required packages
import subprocess
import sys

def install_requirements():
    """Install required packages if not available"""
    packages = [
        'accelerate>=0.26.0',
        'transformers>=4.30.0',
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'tqdm'
    ]
    
    print("=" * 70)
    print("📦 CHECKING AND INSTALLING REQUIRED PACKAGES")
    print("=" * 70)
    
    for package in packages:
        try:
            print(f"Installing/upgrading {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", package])
            print(f"✅ {package} ready")
        except Exception as e:
            print(f"⚠️  Warning installing {package}: {e}")
    
    print("✅ All packages checked\n")

# Run installation
try:
    install_requirements()
except Exception as e:
    print(f"⚠️  Package installation had issues: {e}")
    print("Continuing anyway...\n")

# Suppress warnings
import os
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check disk space
def check_disk_space(path='/'):
    """Check available disk space"""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_percent = (stat.used / stat.total) * 100
        print(f"\n💾 Disk Space:")
        print(f"   Total: {total_gb:.2f} GB")
        print(f"   Free: {free_gb:.2f} GB")
        print(f"   Used: {used_percent:.1f}%")
        
        if free_gb < 5:
            print(f"\n⚠️  WARNING: Low disk space ({free_gb:.2f} GB free)")
            print(f"   Training may fail. Recommended: >10 GB free")
            return False
        return True
    except Exception as e:
        print(f"Could not check disk space: {e}")
        return True

check_disk_space()

# =============================================================================
# 1. LOAD SARCASM HEADLINES DATASET
# =============================================================================
print("\n" + "="*70)
print("📂 LOADING SARCASM HEADLINES DATASET")
print("="*70)

# Try multiple possible dataset paths
possible_paths = [
    '/kaggle/input/sarcasm-headlines-dataset/Sarcasm_Headlines_Dataset_v2.json',
    '/kaggle/input/sarcasm-headlines-dataset-v2/Sarcasm_Headlines_Dataset_v2.json',
    '../datasets/Sarcasm_Headlines_Dataset_v2.json',
    './Sarcasm_Headlines_Dataset_v2.json',
    'Sarcasm_Headlines_Dataset_v2.json'
]

dataset_path = None
for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        print(f"✅ Found dataset at: {path}")
        break

if dataset_path is None:
    raise FileNotFoundError(
        "Could not find Sarcasm_Headlines_Dataset_v2.json. "
        "Please ensure the dataset is uploaded or the path is correct."
    )

# Load JSON dataset
data_list = []
try:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

if len(data_list) == 0:
    raise ValueError("No data loaded from the dataset file!")

# Convert to DataFrame
df = pd.DataFrame(data_list)

# Ensure required columns exist
if 'headline' not in df.columns or 'is_sarcastic' not in df.columns:
    raise ValueError(f"Dataset missing required columns. Found: {df.columns.tolist()}")

# Clean the data
df = df.dropna(subset=['headline', 'is_sarcastic'])
df['headline'] = df['headline'].astype(str).str.strip()
df = df[df['headline'].str.len() > 0]
df = df.reset_index(drop=True)

print(f"✅ Loaded {len(df)} samples")
print(f"📊 Dataset columns: {df.columns.tolist()}")
print(f"\n📊 Class distribution:")
print(df['is_sarcastic'].value_counts())
print(f"\n📝 Sample headlines:")
print(df[['headline', 'is_sarcastic']].head(10))

# =============================================================================
# 2. TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "="*70)
print("🔀 SPLITTING DATASET")
print("="*70)

# Split into train (80%) and test (20%)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['is_sarcastic']
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"✅ Train samples: {len(train_df)}")
print(f"✅ Test samples: {len(test_df)}")
print(f"Train distribution: {train_df['is_sarcastic'].value_counts().to_dict()}")
print(f"Test distribution: {test_df['is_sarcastic'].value_counts().to_dict()}")

# =============================================================================
# 3. DATASET CLASS
# =============================================================================
class SarcasmDataset(Dataset):
    """Simple dataset for RoBERTa baseline"""
    
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['headline']
        label = item['is_sarcastic']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# 4. PREPARE DATASETS
# =============================================================================
print("\n" + "="*70)
print("📊 PREPARING DATASETS")
print("="*70)

train_data = train_df.to_dict('records')
eval_data = test_df.to_dict('records')

# Load tokenizer
print("📥 Loading RoBERTa tokenizer...")
try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=True)
except Exception as e:
    print(f"Warning: Could not load fast tokenizer, falling back to slow tokenizer: {e}")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=False)

print("✅ Tokenizer loaded")

# Create datasets
train_dataset = SarcasmDataset(train_data, tokenizer, max_len=128)
eval_dataset = SarcasmDataset(eval_data, tokenizer, max_len=128)

print(f"✅ Train dataset: {len(train_dataset)} samples")
print(f"✅ Eval dataset: {len(eval_dataset)} samples")

# =============================================================================
# 5. METRICS FUNCTION
# =============================================================================
def compute_metrics(pred):
    """Compute accuracy and F1 score"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    
    return {
        'accuracy': acc,
        'f1': f1
    }

# =============================================================================
# 6. TRAINING ARGUMENTS
# =============================================================================
print("\n" + "="*70)
print("⚙️  CONFIGURING TRAINING")
print("="*70)

# Determine output directory (Kaggle or local)
if os.path.exists('/kaggle/working'):
    output_dir = '/kaggle/working/roberta_baseline_results'
else:
    output_dir = './roberta_baseline_results'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

# Adjust batch size based on GPU availability
batch_size = 16 if torch.cuda.is_available() else 8

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=os.path.join(output_dir, 'logs'),
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    learning_rate=2e-5,
    report_to='none',
    fp16=torch.cuda.is_available(),
    save_total_limit=1,  # Reduced from 2 to save disk space
    dataloader_num_workers=0,
    disable_tqdm=False,
    logging_first_step=True,
    seed=42,
    save_safetensors=False,  # Use legacy format to save space
    save_only_model=True,  # Don't save optimizer states
)

print(f"✅ Output directory: {output_dir}")
print(f"✅ Number of epochs: {training_args.num_train_epochs}")
print(f"✅ Batch size: {training_args.per_device_train_batch_size}")
print(f"✅ Learning rate: {training_args.learning_rate}")
print(f"✅ FP16 training: {training_args.fp16}")

# =============================================================================
# 7. LOAD MODEL
# =============================================================================
print("\n" + "="*70)
print("🚀 LOADING ROBERTA MODEL")
print("="*70)

print("📥 Loading RoBERTa-base for sequence classification...")
try:
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=2,
        problem_type="single_label_classification"
    )
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"✅ Model moved to {device}")
print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# =============================================================================
# 8. INITIALIZE TRAINER
# =============================================================================
print("\n🔧 Initializing Trainer...")

try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    print("✅ Trainer initialized successfully")
except Exception as e:
    print(f"Error initializing trainer: {e}")
    raise

# =============================================================================
# 10. TRAIN THE MODEL
# =============================================================================
print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)

# Clean up cache before training to free disk space
print("\n🧹 Cleaning cache to free disk space...")
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Clear old checkpoints if they exist
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
        for cp in checkpoints:
            cp_path = os.path.join(output_dir, cp)
            if os.path.isdir(cp_path):
                shutil.rmtree(cp_path)
                print(f"   Removed old checkpoint: {cp}")
    print("✅ Cleanup complete")
except Exception as e:
    print(f"⚠️  Cleanup warning: {e}")

# Check disk space again
check_disk_space()

try:
    train_result = trainer.train()
    print("\n✅ Training completed successfully")
    
    # Save final model
    trainer.save_model(os.path.join(output_dir, 'final_model'))
    print(f"✅ Final model saved to {os.path.join(output_dir, 'final_model')}")
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("💾 Saving current state...")
    trainer.save_model(os.path.join(output_dir, 'interrupted_model'))
    raise
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    raise

# =============================================================================
# 10. EVALUATE ON TEST SET
# =============================================================================
print("\n" + "="*70)
print("📊 EVALUATING ON TEST SET")
print("="*70)

try:
    results = trainer.evaluate()
    
    print("\n📊 Evaluation Metrics:")
    for key, value in results.items():
        print(f"   {key}: {value:.4f}")
    
    # Get predictions
    print("\n📊 Generating predictions...")
    predictions = trainer.predict(eval_dataset)
    preds = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids
    
    print("\n" + "="*70)
    print("📊 CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(
        true_labels,
        preds,
        target_names=['Non-Sarcastic', 'Sarcastic'],
        digits=4
    ))
    
except Exception as e:
    print(f"\n❌ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    results = {}
    preds = np.array([])
    true_labels = np.array([])

# =============================================================================
# 11. SAVE RESULTS
# =============================================================================
try:
    results_file = os.path.join(output_dir, 'roberta_baseline_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ROBERTA BASELINE - SARCASM DETECTION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: RoBERTa-base (Standard)\n")
        f.write(f"Dataset: Sarcasm Headlines Dataset v2\n")
        f.write(f"Dataset path: {dataset_path}\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Train samples: {len(train_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {batch_size}\n\n")
        
        if results:
            f.write("Evaluation Metrics:\n")
            for key, value in results.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        if len(preds) > 0 and len(true_labels) > 0:
            f.write("\n" + "="*70 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n")
            f.write(classification_report(
                true_labels,
                preds,
                target_names=['Non-Sarcastic', 'Sarcastic'],
                digits=4
            ))
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # Save predictions to CSV
    if len(preds) > 0 and len(true_labels) > 0:
        predictions_df = pd.DataFrame({
            'headline': test_df['headline'].values,
            'true_label': true_labels,
            'predicted_label': preds,
            'correct': (true_labels == preds).astype(int)
        })
        predictions_csv = os.path.join(output_dir, 'roberta_baseline_predictions.csv')
        predictions_df.to_csv(predictions_csv, index=False)
        print(f"📄 Predictions saved to: {predictions_csv}")
        
        # Calculate accuracy
        accuracy = (true_labels == preds).mean()
        print(f"\n✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show sample predictions
        print("\n" + "="*70)
        print("📝 SAMPLE PREDICTIONS")
        print("="*70)
        
        sample_indices = np.random.choice(len(test_df), min(15, len(test_df)), replace=False)
        for idx in sample_indices:
            headline = test_df.iloc[idx]['headline']
            true_label = true_labels[idx]
            pred_label = preds[idx]
            
            emoji = "✅" if true_label == pred_label else "❌"
            true_str = "Sarcastic" if true_label == 1 else "Non-Sarcastic"
            pred_str = "Sarcastic" if pred_label == 1 else "Non-Sarcastic"
            
            print(f"\n{emoji} {headline[:90]}...")
            print(f"   True: {true_str:15} | Predicted: {pred_str}")
    
except Exception as e:
    print(f"\n⚠️  Warning: Could not save results: {e}")

print("\n" + "="*70)
print("🎉 BASELINE EXPERIMENT COMPLETED!")
print("="*70)
print("\n📌 Summary:")
print(f"   Model: RoBERTa-base (Standard)")
print(f"   No incongruity features")
print(f"   Simple architecture for comparison")
print(f"\n💡 Compare with IDL-RoBERTa results to measure improvement!")
print("="*70)
