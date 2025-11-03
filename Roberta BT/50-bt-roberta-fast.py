# RoBERTa WITHOUT Back-Translation for Sarcasm Detection
# Optimized for Local GPU Execution - FAST VERSION

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
import time
import os
from datetime import datetime

print("="*60)
print("RoBERTa - Sarcasm Detection (NO Back-Translation)")
print("="*60)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print("="*60)

print("\n[1/7] Importing libraries and downloading NLTK data")
print("✓ Libraries imported successfully\n")

# Load and preprocess dataset
print("[2/7] Loading and preprocessing dataset")
path = 'Datasets/Sarcasm_Headlines_Dataset_v2.json'

df = pd.read_json(path, lines=True)
df = df[['headline', 'is_sarcastic']]
df['Tweet'] = df['headline']
df['Sarcasm (yes/no)'] = df['is_sarcastic'].map({1: 'yes', 0: 'no'})
df = df[['Tweet', 'Sarcasm (yes/no)', 'is_sarcastic']]
df = df.dropna()
df = df[df['Tweet'].str.strip() != '']
df = df.drop_duplicates(subset=['Tweet'])

print(f"✓ Dataset loaded: {len(df)} samples")
print(f"  - Sarcastic: {(df['is_sarcastic']==1).sum()}")
print(f"  - Non-sarcastic: {(df['is_sarcastic']==0).sum()}\n")

# Sample 50% of the dataset
print("[3/7] Sampling 50% of the dataset")
df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
print(f"✓ Sampled dataset: {len(df)} samples")
print(f"  - Sarcastic: {(df['is_sarcastic']==1).sum()}")
print(f"  - Non-sarcastic: {(df['is_sarcastic']==0).sum()}\n")

# Use the dataset without augmentation
print("[4/7] Skipping augmentation for faster training")
df_augmented = df.copy()
print(f"✓ Using original dataset without augmentation")
print(f"  - Total samples: {len(df_augmented)}\n")

df_augmented.to_csv('dataset_no_augmentation.csv', index=False)

# Define custom SarcasmDataset class
class SarcasmDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['Tweet']
        label = item['is_sarcastic']
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Prepare train and eval datasets
print("[5/7] Preparing train and eval datasets")
train_df, eval_df = train_test_split(df_augmented, test_size=0.2, random_state=42)
train_data = train_df.to_dict('records')
eval_data = eval_df.to_dict('records')

print(f"✓ Train samples: {len(train_data)}")
print(f"✓ Eval samples: {len(eval_data)}\n")

# Initialize tokenizer and datasets
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
train_dataset = SarcasmDataset(train_data, tokenizer)
eval_dataset = SarcasmDataset(eval_data, tokenizer)

# Define enhanced compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Set up TrainingArguments - optimized for local GPU
print("[6/7] Setting up training configuration")
training_args = TrainingArguments(
    output_dir='results',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    learning_rate=2e-5,
    report_to='none',
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    gradient_accumulation_steps=1,
    save_total_limit=2,
)
print("✓ Training configuration set\n")

# Initialize model and trainer
print("[7/7] Initializing model and trainer")
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)
print("✓ Model and trainer initialized\n")

# Train the model
print("[8/7] Training the model")
print("="*60)
start_time = time.time()
trainer.train()
training_time = time.time() - start_time
print("="*60)
print(f"✓ Training completed in {training_time/60:.2f} minutes\n")

# Evaluate the model
print("[9/7] Evaluating the model and generating results")
results = trainer.evaluate()

print("\n" + "="*60)
print("EVALUATION RESULTS (WITHOUT BACK-TRANSLATION)")
print("="*60)
print(f"Accuracy:  {results['eval_accuracy']:.4f}")
print(f"F1 Score:  {results['eval_f1']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall:    {results['eval_recall']:.4f}")
print("="*60)

# Generate predictions for the test set and save to CSV
print("\nGenerating detailed predictions...")
predictions = trainer.predict(eval_dataset)
pred_labels = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids

# Create detailed results DataFrame
eval_df_copy = eval_df.copy().reset_index(drop=True)
eval_df_copy['True_Label'] = true_labels
eval_df_copy['Predicted_Label'] = pred_labels
eval_df_copy['Correct'] = (true_labels == pred_labels)
eval_df_copy['True_Class'] = eval_df_copy['True_Label'].map({0: 'Non-sarcastic', 1: 'Sarcastic'})
eval_df_copy['Predicted_Class'] = eval_df_copy['Predicted_Label'].map({0: 'Non-sarcastic', 1: 'Sarcastic'})

# Save predictions to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
predictions_file = f'predictions_no_bt_{timestamp}.csv'
eval_df_copy.to_csv(predictions_file, index=False)
print(f"✓ Detailed predictions saved to: {predictions_file}")

# Generate and save confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
cm_df = pd.DataFrame(cm, 
                     index=['Actual Non-sarcastic', 'Actual Sarcastic'],
                     columns=['Predicted Non-sarcastic', 'Predicted Sarcastic'])
cm_file = f'confusion_matrix_no_bt_{timestamp}.csv'
cm_df.to_csv(cm_file)
print(f"✓ Confusion matrix saved to: {cm_file}")

# Generate and save classification report
report = classification_report(true_labels, pred_labels, 
                               target_names=['Non-sarcastic', 'Sarcastic'],
                               output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_file = f'classification_report_no_bt_{timestamp}.csv'
report_df.to_csv(report_file)
print(f"✓ Classification report saved to: {report_file}")

# Save summary metrics
summary = {
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Training Time (min)', 
               'Total Samples', 'Train Samples', 'Test Samples', 
               'Sarcastic Samples', 'Non-sarcastic Samples', 'Augmentation'],
    'Value': [
        f"{results['eval_accuracy']:.4f}",
        f"{results['eval_f1']:.4f}",
        f"{results['eval_precision']:.4f}",
        f"{results['eval_recall']:.4f}",
        f"{training_time/60:.2f}",
        len(df_augmented),
        len(train_data),
        len(eval_data),
        (df_augmented['is_sarcastic']==1).sum(),
        (df_augmented['is_sarcastic']==0).sum(),
        'None (No Back-Translation)'
    ]
}
summary_df = pd.DataFrame(summary)
summary_file = f'training_summary_no_bt_{timestamp}.csv'
summary_df.to_csv(summary_file, index=False)
print(f"✓ Training summary saved to: {summary_file}")

print("\n" + "="*60)
print("COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nOutput files generated:")
print(f"  1. {predictions_file}")
print(f"  2. {cm_file}")
print(f"  3. {report_file}")
print(f"  4. {summary_file}")
print(f"  5. dataset_no_augmentation.csv")
print("="*60)
