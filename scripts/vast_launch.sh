#!/bin/bash
# ============================================================================
# SoftPaw UST — Vast.ai Launch Script
#
# One-command setup + training on a fresh Vast.ai GPU instance.
#
# Usage:
#   1. Rent a GPU instance on vast.ai (A100/A6000/4090, 24GB+ VRAM)
#   2. SSH into the instance
#   3. Clone your repo:
#        git clone <your-repo-url> softpaw && cd softpaw/training
#   4. Run this script:
#        bash scripts/vast_launch.sh
#
# Options:
#   bash scripts/vast_launch.sh --quick        # ~30 min validation run
#   bash scripts/vast_launch.sh --medium       # ~8-12 hr run
#   bash scripts/vast_launch.sh --multi-gpu    # Use all GPUs on the instance
#   bash scripts/vast_launch.sh --resume checkpoints/step_10000.pt
#   bash scripts/vast_launch.sh --batch_size 64
#   bash scripts/vast_launch.sh --skip-data    # Skip dataset download
#
# Estimated times (RTX 4090):
#   --quick:   ~30 min   (50K pages, 5 epochs, ~$0.15)
#   --medium:  ~8-12 hrs (500K pages, 10 epochs, ~$3)
#   (default): ~150+ hrs (4M pages, 40 epochs, ~$40+)
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TRAINING_DIR"

# Parse flags
MULTI_GPU=false
SKIP_DATA=false
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-gpu)   MULTI_GPU=true; shift ;;
        --skip-data)   SKIP_DATA=true; shift ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================"
echo "SoftPaw UST — Vast.ai Training Launch"
echo "============================================"
echo "Working directory: $TRAINING_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. System check
# ---------------------------------------------------------------------------
echo "[1/4] System check..."

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "  GPUs: $GPU_COUNT x $GPU_NAME ($GPU_MEM each)"

python3 -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Installing Python dependencies..."

pip install -q -r requirements.txt 2>&1 | tail -3
echo "  Done."

# Quick import test
python3 -c "
import torch, numpy, scipy, yaml, tqdm, wandb
from model import SoftPawUST
from config import SoftPawConfig
print('  All imports OK.')
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ---------------------------------------------------------------------------
# 3. Download datasets
# ---------------------------------------------------------------------------
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "[3/4] Downloading datasets..."
    bash scripts/download_data.sh ./datasets
else
    echo ""
    echo "[3/4] Skipping dataset download (--skip-data flag)"
fi

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Starting training..."
echo ""

# Determine launch command
if [ "$MULTI_GPU" = true ] && [ "$GPU_COUNT" -gt 1 ]; then
    echo "  Multi-GPU training with $GPU_COUNT GPUs (DDP)"
    LAUNCH_CMD="torchrun --nproc_per_node=$GPU_COUNT train.py"
else
    echo "  Single-GPU training"
    LAUNCH_CMD="python3 train.py"
fi

# Build full command
FULL_CMD="$LAUNCH_CMD --no_wandb $EXTRA_ARGS"
echo "  Command: $FULL_CMD"
echo ""
echo "============================================"
echo "  Training starting now..."
echo "============================================"
echo ""

# Run training
exec $FULL_CMD
