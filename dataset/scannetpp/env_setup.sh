#!/bin/bash

# env_setup.sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Installing Python packages..."

# Install basic packages
pip install munch tqdm pyyaml numpy imageio lz4 opencv-python Pillow scipy open3d torchmetrics

# Install torch and torchvision with CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
  --find-links https://download.pytorch.org/whl/torch_stable.html

echo "Installation completed successfully!"
