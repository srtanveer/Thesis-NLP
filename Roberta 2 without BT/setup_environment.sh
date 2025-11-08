#!/bin/bash
# Setup script for IDL-RoBERTa Sarcasm Detection
# Run this before executing the main script

echo "========================================================================"
echo "🔧 SETTING UP ENVIRONMENT FOR IDL-ROBERTA"
echo "========================================================================"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install accelerate first (critical dependency)
echo "📦 Installing accelerate>=0.26.0..."
pip install --upgrade "accelerate>=0.26.0"

# Install transformers
echo "📦 Installing transformers..."
pip install --upgrade "transformers>=4.30.0"

# Install PyTorch (use appropriate version for your system)
echo "📦 Installing PyTorch..."
pip install torch torchvision torchaudio

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install pandas numpy scikit-learn nltk tqdm

echo ""
echo "========================================================================"
echo "✅ ENVIRONMENT SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "You can now run: python3 robarta_sarcasm_headlines.py"
echo ""
