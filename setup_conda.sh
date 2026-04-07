#!/bin/bash
# setup_conda.sh - Automated setup for FlashRAG via Conda

echo "Start the Conda environment setup for FlashRAG"

# 1. Create and activate the Conda environment
conda create -n flashrag_env python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate flashrag_env

# 2. Install FAISS-GPU natively (to prevent C++/CUDA conflicts)
echo "📦 Installing FAISS-GPU"
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y

# 3. Install the core dependencies
echo "📦 Installing the core dependencies"
pip install -e .[core]
pip install vllm>=0.4.1 sentence-transformers termcolor

# 4. Block WebUI dependencies to fix serialization bugs
echo "📦 Release notes for Gradio and Pydantic"
pip install gradio==5.9.1 pydantic==2.10.6 "huggingface-hub<1.0"

# 5. Inject CUDA paths for the environment
echo "⚙️ Configuring CUDA paths for the Conda environment"
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo "✅ Setup completed successfully! Use ‘conda activate flashrag_env’ to get started."