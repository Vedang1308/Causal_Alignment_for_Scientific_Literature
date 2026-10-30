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
    
    # Configure Conda to prefer scratch space
    if command -v conda &> /dev/null || command -v mamba &> /dev/null; then
        conda config --add pkgs_dirs "${SCRATCH_DIR}/.conda/pkgs" 2>/dev/null || true
        conda config --add envs_dirs "${SCRATCH_DIR}/.conda/envs" 2>/dev/null || true
    fi
    ENV_DIR="${SCRATCH_DIR}/.conda/envs/ccst_env"
else
    echo "[Storage] Scratch directory not found, using home directory."
    ENV_DIR="${HOME}/.conda/envs/ccst_env"
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
echo "[3/5] Setting up Conda environment at: $ENV_DIR..."

CONDA_CMD="conda"
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
fi

# Verify if environment is complete (must have bin/python and bin/pip)
if [ -f "${ENV_DIR}/bin/python" ] && [ -f "${ENV_DIR}/bin/pip" ]; then
    echo "Existing complete conda environment found at: $ENV_DIR"
else
    echo "No complete environment found at $ENV_DIR. Creating fresh installation..."
    # Remove any corrupted partial env directory if present
    rm -rf "$ENV_DIR" 2>/dev/null || true
    rm -rf "${HOME}/.conda/envs/ccst_env" 2>/dev/null || true
    
    $CONDA_CMD env create -p "$ENV_DIR" -f environment_sol.yml -y
fi

# Set explicit python and pip paths to avoid subshell activation glitches
ENV_PYTHON="${ENV_DIR}/bin/python"
ENV_PIP="${ENV_DIR}/bin/pip"

if [ ! -f "$ENV_PYTHON" ]; then
    echo "ERROR: Python binary not found at $ENV_PYTHON"
    exit 1
fi

# Ensure pip packages from requirements are installed
echo "Ensuring all required pip packages are installed..."
$ENV_PIP install --no-cache-dir \
    pydantic>=2.5.0 \
    sentence-transformers>=2.2.2 \
    transformers>=4.35.0 \
    accelerate>=0.25.0 \
    rank-bm25>=0.2.2 \
    faiss-cpu>=1.7.4 \
    krippendorff>=0.6.0 \
    requests>=2.31.0 \
    tqdm>=4.66.0 \
    pytest>=7.4.0 2>/dev/null || true

# 5. Verify PyTorch and CUDA availability on NVIDIA A100
echo "[4/5] Verifying PyTorch and CUDA acceleration on A100 using: $ENV_PYTHON..."
$ENV_PYTHON -c "
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
$ENV_PYTHON phase1_foundation/run_phase1_verification.py

echo "=========================================================="
echo " Phase 1 Environment & Schemas Setup Successfully on SOL!"
echo " To activate your environment anytime in your shell, run:"
echo "     conda activate ${ENV_DIR}"
echo "=========================================================="
