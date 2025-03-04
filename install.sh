#!/bin/bash
which python

# Install prerequirements
pip install -r requirements.txt

# Install PyTorch with suitable cuda version
# Needed for Grounded DINO
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Create models folder if doesn't exist
mkdir -p models

cd models
########################
# Setup grounded sam 2 #
########################

# git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2

cd checkpoints
bash download_ckpts.sh

cd ../gdino_checkpoints
bash download_ckpts.sh

# Export cuda path if needed
export CUDA_HOME=/usr/local/cuda

echo "CUDA_HOME: $CUDA_HOME"

cd ..

# Install Segment Anything 2
pip install -e .

# Install Grounded DINO
pip install --no-build-isolation -e grounding_dino