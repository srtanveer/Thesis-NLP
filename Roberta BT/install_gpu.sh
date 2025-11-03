#!/bin/bash

# Installation script for GPU servers
# RoBERTa with Back-Translation - Sarcasm Detection

echo "=========================================="
echo "RoBERTa Back-Translation Setup (GPU)"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Check for CUDA availability
echo "Checking for CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c2-)
    echo "✓ CUDA $CUDA_VERSION detected"
    echo ""
    
    # Determine PyTorch CUDA version
    if [[ "$CUDA_VERSION" == 11.* ]]; then
        TORCH_CUDA="cu118"
    elif [[ "$CUDA_VERSION" == 12.* ]]; then
        TORCH_CUDA="cu121"
    else
        TORCH_CUDA="cu118"  # Default to CUDA 11.8
    fi
    
    echo "Installing PyTorch with CUDA support ($TORCH_CUDA)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$TORCH_CUDA
else
    echo "⚠ CUDA not detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo ""

# Install remaining dependencies
echo "Installing remaining dependencies from requirements.txt..."
pip install transformers>=4.30.0
pip install tokenizers>=0.13.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install nltk>=3.8.0
pip install googletrans==3.1.0a0
pip install tqdm>=4.65.0
pip install huggingface-hub>=0.16.0
pip install safetensors>=0.3.0
echo ""

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
echo ""

echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "To run the code:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run the script:"
echo "     python 50-bt-roberta.py"
echo ""
echo "Or run directly:"
echo "  .venv/bin/python 50-bt-roberta.py"
echo ""
echo "=========================================="
