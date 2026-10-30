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

# 1. Resolve Storage Quota: Redirect Conda/Pip from /home to /scratch
USER_NAME=$(whoami)
SCRATCH_DIR="/scratch/${USER_NAME}"

if [ -d "$SCRATCH_DIR" ]; then
    echo "[Storage] Using high-capacity scratch space at: $SCRATCH_DIR"
    mkdir -p "${SCRATCH_DIR}/.conda/pkgs"
    mkdir -p "${SCRATCH_DIR}/.conda/envs"
    mkdir -p "${SCRATCH_DIR}/tmp"
    mkdir -p "${SCRATCH_DIR}/.cache/pip"

    export TMPDIR="${SCRATCH_DIR}/tmp"
    export PIP_CACHE_DIR="${SCRATCH_DIR}/.cache/pip"
    
    # Configure Conda to use scratch space for packages and environments
    if command -v conda &> /dev/null || command -v mamba &> /dev/null; then
        conda config --add pkgs_dirs "${SCRATCH_DIR}/.conda/pkgs" 2>/dev/null || true
        conda config --add envs_dirs "${SCRATCH_DIR}/.conda/envs" 2>/dev/null || true
    fi
    ENV_DIR="${SCRATCH_DIR}/.conda/envs/ccst_env"
else
    echo "[Storage] Scratch directory ($SCRATCH_DIR) not found, using default home directory."
    ENV_DIR="ccst_env"
fi

# 2. Check for NVIDIA GPU
echo "[1/5] Checking GPU status..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not detected. Request a GPU node using: interactive -p general -q public -t 4:00:00 --gres=gpu:a100:1"
fi

# 3. Check / Load Modules on SOL
echo "[2/5] Checking module environment..."
if command -v module &> /dev/null; then
    echo "Loading recommended SOL modules..."
    module purge 2>/dev/null || true
    module load mamba/latest 2>/dev/null || module load mamba 2>/dev/null || true
fi

# 4. Create or activate conda/mamba environment on scratch
ENV_NAME="ccst_env"
echo "[3/5] Setting up Conda environment: $ENV_NAME..."

CONDA_CMD="conda"
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
fi

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    
    # Check if env already exists
    if conda env list | grep -q "ccst_env"; then
        echo "Conda environment 'ccst_env' already exists. Activating..."
        conda activate ccst_env || conda activate "$ENV_DIR"
    else
        echo "Creating new conda environment 'ccst_env' in $ENV_DIR..."
        if [ -d "$SCRATCH_DIR" ]; then
            $CONDA_CMD env create -p "$ENV_DIR" -f environment_sol.yml -y
            conda activate "$ENV_DIR"
        else
            $CONDA_CMD env create -n "$ENV_NAME" -f environment_sol.yml -y
            conda activate "$ENV_NAME"
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

# 5. Verify PyTorch and CUDA availability in Python
echo "[4/5] Verifying PyTorch and CUDA acceleration on A100..."
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Device Count:', torch.cuda.device_count())
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
else:
    print('WARNING: CUDA is not available inside PyTorch.')
"

# 6. Run Phase 1 Verification Tests
echo "[5/5] Running Phase 1 Foundation Verification..."
python phase1_foundation/run_phase1_verification.py

echo "=========================================================="
echo " Phase 1 Environment & Schemas Setup Successfully on SOL!"
echo "=========================================================="
