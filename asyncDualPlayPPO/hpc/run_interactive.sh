#!/bin/bash

# Get the project root (assuming this script is in asyncDualPlayPPO/hpc/)
# This makes it robust regardless of where you call it from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIF_IMAGE="$PROJECT_ROOT/isaac-lab.sif"

# Check if srun is needed (simple check if we have a GPU visible)
if ! command -v nvidia-smi &> /dev/null; then
    echo "----------------------------------------------------------------"
    echo "[WARNING] nvidia-smi not found. You might not be on a GPU node."
    echo "To get a GPU node, run this first:"
    echo "srun --partition=gpushort --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty /bin/bash"
    echo "----------------------------------------------------------------"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found at expected path!"
    echo "Expected: $SIF_IMAGE"
    exit 1
fi

echo "[INFO] Running interactive training..."
echo "[INFO] Project Root: $PROJECT_ROOT"
echo "[INFO] Image: $SIF_IMAGE"

# Run the container with the command
apptainer exec --nv \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project \
    --bind "$PROJECT_ROOT/runs":/workspace/isaaclab/logs \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train.py \
    --num_envs 2048 \
    --max_iterations 5000 \
    --exp_name "hpc_ppo_low_interactive" \
    --save_interval 500 \
    --headless
