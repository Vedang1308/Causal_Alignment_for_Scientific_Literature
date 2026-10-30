#!/usr/bin/env bash
# ==============================================================================
# setup_sol.sh - Environment Setup Script for ASU SOL Supercomputer
# Target Hardware: NVIDIA A100 GPU nodes
# Project: Context-Conditioned Causal Alignment (CCST)
# ==============================================================================

set -e

echo "=========================================================="
echo " Starting CCST Environment Setup on ASU SOL Supercomputer"
echo "=========================================================="

# 1. Check for NVIDIA GPU
echo "[1/5] Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not detected. If running on a login node, remember to request a GPU node (e.g. srun/interactive with A100)."
fi

# 2. Check / Load Modules on SOL
echo "[2/5] Checking module environment..."
if command -v module &> /dev/null; then
    echo "Loading recommended SOL modules..."
    module purge || true
    module load mamba/latest || module load anaconda/py3 || true
    module load cuda/12.1 || module load cuda/11.8 || true
fi

# 3. Create or activate conda/mamba environment
ENV_NAME="ccst_env"
echo "[3/5] Setting up Conda environment: $ENV_NAME..."

CONDA_CMD="conda"
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
fi

# Initialize conda if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    
    if conda env list | grep -q "$ENV_NAME"; then
        echo "Conda environment '$ENV_NAME' already exists. Activating..."
        conda activate "$ENV_NAME"
    else
        echo "Creating new conda environment '$ENV_NAME'..."
        if [ -f "environment_sol.yml" ]; then
            $CONDA_CMD env create -f environment_sol.yml -y
            conda activate "$ENV_NAME"
        else
            $CONDA_CMD create -n "$ENV_NAME" python=3.10 -y
            conda activate "$ENV_NAME"
            pip install -r requirements.txt
        fi
    fi
else
    echo "Conda not found. Falling back to Python virtualenv..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# 4. Verify PyTorch and CUDA availability in Python
echo "[4/5] Verifying PyTorch and CUDA acceleration..."
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Device Count:', torch.cuda.device_count())
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
"

# 5. Run Phase 1 Verification Tests
echo "[5/5] Running Phase 1 Foundation Verification..."
python phase1_foundation/run_phase1_verification.py

echo "=========================================================="
echo " Phase 1 Environment & Schemas Setup Successfully on SOL!"
echo "=========================================================="

