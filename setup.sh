#!/usr/bin/env bash
#
# Setup script for playerSTOVE / nextPlayer.
#
# Modes:
#   ./setup.sh           - Full setup (training + GUI, needs CUDA)
#   ./setup.sh --gui     - GUI-only setup (CPU JAX, no CUDA required)
#
# What it does:
#   1. Creates a fresh conda environment with Python 3.11
#   2. Installs pip dependencies (full or GUI-only)
#   3. Installs the package in editable mode (pip install -e .)
#   4. Fixes LD_LIBRARY_PATH for pip-installed CUDA libs (full mode)
#   5. Creates the checkpoints directory for training
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="playerSTOVE"
PYTHON_VERSION="3.11"

GUI_ONLY=false
if [[ "${1:-}" == "--gui" ]]; then
    GUI_ONLY=true
fi

echo "============================================"
echo "  playerSTOVE setup"
if $GUI_ONLY; then
    echo "  Mode: GUI only (CPU JAX)"
else
    echo "  Mode: Full (training + GUI, CUDA)"
fi
echo "============================================"
echo

# ── 1. Conda environment ──────────────────────────────────────────
if conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Removing..."
    conda deactivate 2>/dev/null || true
    conda remove -n "$ENV_NAME" --all -y
fi

echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
echo

# Activate (works in both bash and zsh)
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── 2. Pip dependencies ──────────────────────────────────────────
if $GUI_ONLY; then
    echo "Installing GUI-only dependencies..."
    pip install -r "${REPO_DIR}/requirements_environment.txt"
else
    echo "Installing full dependencies (with CUDA)..."
    pip install -r "${REPO_DIR}/requirements.txt"
fi
echo

# ── 3. Editable install ──────────────────────────────────────────
echo "Installing package in editable mode..."
pip install -e "${REPO_DIR}"
echo

# ── 4. CUDA LD_LIBRARY_PATH fix (full mode only) ─────────────────
if ! $GUI_ONLY; then
    echo "Setting up CUDA library paths for pip-installed nvidia packages..."
    ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
    mkdir -p "$ACTIVATE_DIR"
    cat > "${ACTIVATE_DIR}/cuda_ld_path.sh" << 'CUDAEOF'
_NVIDIA_LIB=$(python -c "import nvidia; import pathlib; print(':'.join(str(p) for p in pathlib.Path(nvidia.__path__[0]).glob('*/lib')))" 2>/dev/null)
if [ -n "$_NVIDIA_LIB" ]; then
    export LD_LIBRARY_PATH="$_NVIDIA_LIB:$LD_LIBRARY_PATH"
fi
unset _NVIDIA_LIB
CUDAEOF
    echo "  Created ${ACTIVATE_DIR}/cuda_ld_path.sh"
    echo "  (Takes effect on next 'conda activate ${ENV_NAME}')"
    echo
fi

# ── 5. Create checkpoint + image-dump directories ────────────────
CKPT_DIR="${REPO_DIR}/checkpoints"
IMG_DIR="${REPO_DIR}/images"
mkdir -p "$CKPT_DIR"
mkdir -p "$IMG_DIR"
echo "Checkpoint directory: ${CKPT_DIR}"
echo "Images directory:     ${IMG_DIR}"
echo

# ── Done ──────────────────────────────────────────────────────────
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate:  conda activate ${ENV_NAME}"
if $GUI_ONLY; then
    echo "  Play:      python -m nextPlayer.gui.play"
else
    echo "  Play:      python -m nextPlayer.gui.play"
    echo "  Train:     python modSTOVE_pretraining.py"
fi
echo "============================================"
