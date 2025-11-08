#!/bin/bash
# Quick commands for your GPU server

echo "==================================================="
echo "COPY AND RUN THESE COMMANDS ON YOUR GPU SERVER:"
echo "==================================================="
echo ""
echo "# 1. Clone the repository (if not already done)"
echo "git clone git@github.com:srtanveer/Thesis-NLP.git"
echo "cd Thesis-NLP"
echo 'cd "Roberta BT"'
echo ""
echo "# 2. Run the GPU installation script"
echo "chmod +x install_gpu.sh"
echo "./install_gpu.sh"
echo ""
echo "# 3. Run the training script"
echo "source .venv/bin/activate"
echo "python 50-bt-roberta.py"
echo ""
echo "==================================================="
echo "OR USE THIS ONE-LINER AFTER CLONING:"
echo "==================================================="
echo ""
echo 'cd "Thesis-NLP/Roberta BT" && chmod +x install_gpu.sh && ./install_gpu.sh && source .venv/bin/activate && python 50-bt-roberta.py'
echo ""
echo "==================================================="
