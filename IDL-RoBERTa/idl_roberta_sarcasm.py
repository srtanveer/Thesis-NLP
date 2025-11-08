#!/usr/bin/env python3
"""
IDL-RoBERTa for Sarcasm Detection on Sarcasm Headlines Dataset
Enhanced Model with Commonsense Incongruity Features
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
        'nltk',
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

# Suppress TensorFlow and other warnings
import os
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

# Disable symlink warnings on Windows
if sys.platform == "win32":
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.file_download")

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaModel, 
    Trainer, 
    TrainingArguments,
    RobertaConfig,
    RobertaPreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set random seeds for reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Download NLTK resources
print("📥 Downloading NLTK resources...")
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Could not download punkt: {e}")
    
try:
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Info: punkt_tab not available (optional): {e}")
    
try:
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    print(f"Warning: Could not download vader_lexicon: {e}")

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

# Load JSON dataset (handle different formats)
data_list = []
try:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
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
df = df[df['headline'].str.len() > 0]  # Remove empty headlines
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

print(f"✅ Train samples: {len(train_df)}")
print(f"✅ Test samples: {len(test_df)}")
print(f"Train distribution: {train_df['is_sarcastic'].value_counts().to_dict()}")
print(f"Test distribution: {test_df['is_sarcastic'].value_counts().to_dict()}")

# Reset indices
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# =============================================================================
# 3. BUILD ENHANCED COMMONSENSE DICTIONARY
# =============================================================================
print("\n" + "="*70)
print("🔨 BUILDING ENHANCED COMMONSENSE DICTIONARY")
print("="*70)

sia = SentimentIntensityAnalyzer()

# Collect sentiment scores per word and label
word_sentiments = defaultdict(lambda: {'non_sarc': [], 'sarc': []})

for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Analyzing training data"):
    text = row['headline']
    label = row['is_sarcastic']
    
    # Get sentence-level sentiment
    sent_score = sia.polarity_scores(text)['compound']
    
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    for token in tokens:
        if token.isalpha() and len(token) > 2:  # Filter meaningful words
            if label == 0:
                word_sentiments[token]['non_sarc'].append(sent_score)
            else:
                word_sentiments[token]['sarc'].append(sent_score)

# Build incongruity dictionary with stronger signals
commonsense_dict = {}
min_count = 10  # Increased for reliability
threshold = 0.15  # Higher threshold for clearer signals

for word, sents in word_sentiments.items():
    non_sarc = sents['non_sarc']
    sarc = sents['sarc']
    
    if len(non_sarc) >= min_count and len(sarc) >= min_count:
        expected_sent = np.mean(non_sarc)
        sarcastic_sent = np.mean(sarc)
        incongruity = abs(expected_sent - sarcastic_sent)
        
        if incongruity > threshold:
            commonsense_dict[word] = {
                'expected': expected_sent,
                'sarcastic': sarcastic_sent,
                'incongruity': incongruity,
                'flip': 1 if (expected_sent > 0 and sarcastic_sent < 0) or 
                             (expected_sent < 0 and sarcastic_sent > 0) else 0
            }

print(f"✅ Built dictionary with {len(commonsense_dict)} incongruent words")

# Show some examples
if len(commonsense_dict) > 0:
    print("\n📝 Sample incongruent words:")
    sample_words = list(commonsense_dict.items())[:10]
    for word, info in sample_words:
        print(f"  '{word}': incongruity={info['incongruity']:.3f}, flip={info['flip']}")

# =============================================================================
# 4. ENHANCED INCONGRUITY FEATURE EXTRACTION
# =============================================================================
def extract_incongruity_features(text, max_len=32):
    """Extract token-level AND sentence-level incongruity features"""
    tokens = nltk.word_tokenize(text.lower())
    
    # Token-level features
    token_features = []
    incongruity_scores = []
    flip_count = 0
    
    for token in tokens[:max_len]:
        if token in commonsense_dict:
            info = commonsense_dict[token]
            incongruity = info['incongruity']
            flip = info['flip']
            expected = info['expected']
            sarcastic = info['sarcastic']
            
            incongruity_scores.append(incongruity)
            flip_count += flip
            
            # Multi-dimensional features per token
            token_features.append([
                incongruity,           # Magnitude of incongruity
                flip,                  # Sentiment flip indicator
                expected,              # Expected sentiment
                sarcastic,             # Sarcastic sentiment
                expected - sarcastic   # Signed difference
            ])
        else:
            token_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Pad or truncate
    while len(token_features) < max_len:
        token_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
    token_features = token_features[:max_len]
    
    # Sentence-level aggregated features
    if incongruity_scores:
        mean_incong = np.mean(incongruity_scores)
        max_incong = np.max(incongruity_scores)
        std_incong = np.std(incongruity_scores)
        flip_ratio = flip_count / len(tokens)
    else:
        mean_incong = max_incong = std_incong = flip_ratio = 0.0
    
    sentence_features = [mean_incong, max_incong, std_incong, flip_ratio]
    
    return (torch.tensor(token_features, dtype=torch.float), 
            torch.tensor(sentence_features, dtype=torch.float))

# =============================================================================
# 5. ENHANCED IDL-ROBERTA MODEL
# =============================================================================
class IDLRobertaForSarcasm(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        
        self.token_feature_dim = 5  # 5 features per token
        self.sentence_feature_dim = 4  # 4 sentence-level features
        
        # Token-level attention with incongruity
        self.token_attention = nn.Sequential(
            nn.Linear(config.hidden_size + self.token_feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Incongruity feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(self.sentence_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Final classifier combining everything
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask, token_features, sentence_features, 
                labels=None):
        # Get RoBERTa embeddings
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch, seq_len, hidden]
        
        # Combine RoBERTa embeddings with token-level incongruity features
        combined = torch.cat([sequence_output, token_features], dim=-1)
        
        # Compute attention scores using incongruity-aware features
        attention_scores = self.token_attention(combined).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Weighted context vector
        context = torch.bmm(attention_weights.unsqueeze(1), sequence_output).squeeze(1)
        context = self.dropout(context)
        
        # Process sentence-level incongruity features
        sentence_repr = self.feature_processor(sentence_features)
        
        # Combine context and incongruity features
        final_repr = torch.cat([context, sentence_repr], dim=-1)
        
        # Classification
        logits = self.classifier(final_repr)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return {'loss': loss, 'logits': logits} if loss is not None else logits

# =============================================================================
# 6. CUSTOM DATASET
# =============================================================================
class SarcasmDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['headline']  # Changed from 'Tweet' to 'headline'
        label = item['is_sarcastic']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract incongruity features
        token_feats, sentence_feats = extract_incongruity_features(text, self.max_len)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_features': token_feats,
            'sentence_features': sentence_feats,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# 7. PREPARE DATASETS
# =============================================================================
print("\n" + "="*70)
print("📊 PREPARING DATASETS")
print("="*70)

train_data = train_df.to_dict('records')
eval_data = test_df.to_dict('records')

# Load tokenizer with error handling
print("📥 Loading RoBERTa tokenizer...")
try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=True)
except Exception as e:
    print(f"Warning: Could not load fast tokenizer, falling back to slow tokenizer: {e}")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_fast=False)

print("✅ Tokenizer loaded")

train_dataset = SarcasmDataset(train_data, tokenizer)
eval_dataset = SarcasmDataset(eval_data, tokenizer)

print(f"✅ Train dataset: {len(train_dataset)} samples")
print(f"✅ Eval dataset: {len(eval_dataset)} samples")

# =============================================================================
# 8. COMPUTE METRICS
# =============================================================================
def compute_metrics(pred):
    labels = pred.label_ids
    if isinstance(labels, tuple):
        labels = labels[0]
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    
    return {'accuracy': acc, 'f1': f1}

# =============================================================================
# 9. TRAINING ARGUMENTS
# =============================================================================
print("\n" + "="*70)
print("⚙️  CONFIGURING TRAINING")
print("="*70)

# Determine output directory (Kaggle or local)
if os.path.exists('/kaggle/working'):
    output_dir = '/kaggle/working/idl_roberta_results'
else:
    output_dir = './idl_roberta_results'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

# Check for GPU availability
device_name = "CPU"
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"🎮 GPU detected: {device_name}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("⚠️  No GPU detected, training will use CPU (slow)")

# Adjust batch size based on available memory
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
    fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA is available
    gradient_accumulation_steps=2,  # Effective batch size: 32
    save_total_limit=1,  # Reduced from 2 to save disk space
    dataloader_num_workers=0,  # Prevent multiprocessing issues
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
print(f"✅ Device: {device_name}")

# =============================================================================
# 10. INITIALIZE MODEL AND TRAINER
# =============================================================================
print("\n" + "="*70)
print("🚀 INITIALIZING IDL-ROBERTA MODEL")
print("="*70)

# Load model with error handling
print("📥 Loading RoBERTa base model...")
try:
    model = IDLRobertaForSarcasm.from_pretrained('roberta-base')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"✅ Model moved to {device}")
print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Initialize trainer with error handling
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
# 11. TRAIN THE MODEL
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

# Train with error handling and checkpointing
try:
    train_result = trainer.train()
    print("✅ Training completed successfully")
    
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
# 12. EVALUATE AND GENERATE CLASSIFICATION REPORT
# =============================================================================
print("\n" + "="*70)
print("📊 FINAL EVALUATION ON TEST SET")
print("="*70)

try:
    results = trainer.evaluate()
    
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    # Get predictions for detailed analysis
    print("\n📊 Generating predictions...")
    predictions = trainer.predict(eval_dataset)
    preds = predictions.predictions.argmax(-1)
    
    # Handle different label formats
    if isinstance(predictions.label_ids, tuple):
        true_labels = predictions.label_ids[0]
    else:
        true_labels = predictions.label_ids
    
    print("\n" + "="*70)
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(true_labels, preds, 
                              target_names=['Non-Sarcastic', 'Sarcastic'],
                              digits=4))
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"💾 Best model saved to: {training_args.output_dir}")
    
except Exception as e:
    print(f"\n❌ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    # Continue to save results even if evaluation fails
    results = {}
    preds = np.array([])
    true_labels = np.array([])

# =============================================================================
# 13. SAVE RESULTS
# =============================================================================
try:
    results_file = os.path.join(output_dir, 'final_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("FINAL EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
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
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n")
            f.write(classification_report(true_labels, preds, 
                                         target_names=['Non-Sarcastic', 'Sarcastic'],
                                         digits=4))
    
    print(f"\n📄 Results saved to: {results_file}")
    
    # Also save predictions as CSV
    if len(preds) > 0 and len(true_labels) > 0:
        predictions_df = pd.DataFrame({
            'headline': test_df['headline'].values,
            'true_label': true_labels,
            'predicted_label': preds,
            'correct': (true_labels == preds).astype(int)
        })
        predictions_csv = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_csv, index=False)
        print(f"📄 Predictions saved to: {predictions_csv}")
        
        # Show some example predictions
        print("\n" + "="*70)
        print("📝 SAMPLE PREDICTIONS")
        print("="*70)
        sample_indices = np.random.choice(len(test_df), min(10, len(test_df)), replace=False)
        for idx in sample_indices:
            headline = test_df.iloc[idx]['headline']
            true_label = true_labels[idx]
            pred_label = preds[idx]
            
            emoji = "✅" if true_label == pred_label else "❌"
            true_str = "Sarcastic" if true_label == 1 else "Non-Sarcastic"
            pred_str = "Sarcastic" if pred_label == 1 else "Non-Sarcastic"
            
            print(f"\n{emoji} {headline[:80]}...")
            print(f"   True: {true_str:15} | Predicted: {pred_str}")
    
except Exception as e:
    print(f"\n⚠️  Warning: Could not save results: {e}")

print("\n" + "="*70)
print("🎉 SCRIPT COMPLETED")
print("="*70)
