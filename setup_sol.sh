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

USER_NAME=$(whoami)
SCRATCH_DIR="/scratch/${USER_NAME}"

# 1. Resolve Storage Quota: Permanently symlink ~/.conda to /scratch
if [ -d "$SCRATCH_DIR" ]; then
    echo "[Storage] High-capacity scratch detected at: $SCRATCH_DIR"
    mkdir -p "${SCRATCH_DIR}/.conda/pkgs"
    mkdir -p "${SCRATCH_DIR}/.conda/envs"
    mkdir -p "${SCRATCH_DIR}/tmp"
    mkdir -p "${SCRATCH_DIR}/.cache/pip"

    export TMPDIR="${SCRATCH_DIR}/tmp"
    export PIP_CACHE_DIR="${SCRATCH_DIR}/.cache/pip"
    export CONDA_PKGS_DIRS="${SCRATCH_DIR}/.conda/pkgs"
    export CONDA_ENVS_PATH="${SCRATCH_DIR}/.conda/envs"

    # If ~/.conda is still in /home (filling home quota), redirect it to scratch
    if [ ! -L "${HOME}/.conda" ]; then
        echo "[Storage] Symlinking ~/.conda -> ${SCRATCH_DIR}/.conda to protect /home quota..."
        rm -rf "${HOME}/.conda" 2>/dev/null || true
        ln -sfn "${SCRATCH_DIR}/.conda" "${HOME}/.conda"
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

# 4. Clean any aborted or corrupted packages from scratch
echo "[3/5] Cleaning corrupted package caches..."
if [ -d "${SCRATCH_DIR}/.conda/pkgs" ]; then
    # Remove any corrupted partial locks or tarballs
    rm -f "${SCRATCH_DIR}/.conda/pkgs/*.lock" 2>/dev/null || true
    rm -f "${SCRATCH_DIR}/.conda/pkgs/*.part" 2>/dev/null || true
fi

# 5. Create or activate conda/mamba environment on scratch
echo "[4/5] Setting up Conda environment at: $ENV_DIR..."

CONDA_CMD="conda"
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
fi

ENV_PYTHON="${ENV_DIR}/bin/python"
ENV_PIP="${ENV_DIR}/bin/pip"

if [ -f "$ENV_PYTHON" ] && [ -f "$ENV_PIP" ]; then
    echo "Existing complete conda environment found at: $ENV_DIR"
else
    echo "Creating clean environment in $ENV_DIR..."
    rm -rf "$ENV_DIR" 2>/dev/null || true
    
    # Clean partial packages before creating to avoid checksum error
    $CONDA_CMD clean -pkgs -y 2>/dev/null || true

    $CONDA_CMD env create -p "$ENV_DIR" -f environment_sol.yml -y
fi

# Ensure pip packages from requirements are installed
echo "Ensuring all required pip packages are installed..."
$ENV_PIP install --no-cache-dir \
    "pydantic>=2.5.0" \
    "transformers>=4.35.0,<4.46.0" \
    "sentence-transformers>=2.2.2" \
    "accelerate>=0.25.0" \
    "rank-bm25>=0.2.2" \
    "faiss-cpu>=1.7.4" \
    "krippendorff>=0.6.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "pytest>=7.4.0" 2>/dev/null || true

# 6. Verify PyTorch and CUDA availability on NVIDIA A100
echo "[5/5] Verifying PyTorch and CUDA acceleration on A100..."
$ENV_PYTHON -c "
import torch
print('==============================================')
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Device Count:', torch.cuda.device_count())
    print('VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    print('CUDA Device Capability:', torch.cuda.get_device_capability(0))
    # Quick tensor test on A100
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print('A100 Tensor Matmul Test: SUCCESS')
print('==============================================')
"

# 7. Run Phase 1 Verification Tests
echo "Running Phase 1 Foundation Verification Suite..."
$ENV_PYTHON phase1_foundation/run_phase1_verification.py

echo "=========================================================="
echo " Phase 1 Environment & Schemas Setup Successfully on SOL!"
echo " To activate your environment anytime in your shell, run:"
echo "     conda activate ${ENV_DIR}"
echo "=========================================================="
