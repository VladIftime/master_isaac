#!/bin/bash

# Interactive session launcher (no SLURM job).
# Get a GPU node first:
#   srun --partition=gpushort --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty /bin/bash
# Then run this script from the asyncDualPlayPPO/hpc/ directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIF_IMAGE="$PROJECT_ROOT/isaac-lab.sif"

# ── I/O config ──────────────────────────────────────────────────────────────
# Use local scratch for logs to avoid NFS write stalls.
# In an srun session $TMPDIR is usually set; fall back to /tmp.
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_interactive_$$"
mkdir -p "$LOCAL_RUNS"
# ────────────────────────────────────────────────────────────────────────────

# ── Env scaling ──────────────────────────────────────────────────────────────
# Small interactive profile — fast startup, same 131k buffer target.
#   num_envs=512 → nsteps = 131072 / 512 = 256
NUM_ENVS=512
NSTEPS=256
# ────────────────────────────────────────────────────────────────────────────

if ! command -v nvidia-smi &>/dev/null; then
    echo "----------------------------------------------------------------"
    echo "[WARNING] nvidia-smi not found. You might not be on a GPU node."
    echo "Request one with:"
    echo "  srun --partition=gpushort --gres=gpu:a100:1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty /bin/bash"
    echo "----------------------------------------------------------------"
    read -p "Continue anyway? (y/n) " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image not found at: $SIF_IMAGE"
    exit 1
fi

echo "[INFO] Running interactive training..."
echo "[INFO] Project root:  $PROJECT_ROOT"
echo "[INFO] Local runs:    $LOCAL_RUNS"
echo "[INFO] Buffer size:   ${NUM_ENVS} envs × ${NSTEPS} steps = $(( NUM_ENVS * NSTEPS )) transitions"

apptainer exec --nv \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$LOCAL_RUNS":"$PROJECT_ROOT/runs" \
    --bind "$LOCAL_RUNS":/workspace/isaaclab/logs \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train.py \
    --num_envs "$NUM_ENVS" \
    --nsteps   "$NSTEPS" \
    --max_iterations 5000 \
    --exp_name "hpc_bco_interactive" \
    --save_interval 500 \
    --headless

# Offer to sync back after interactive run finishes
echo ""
read -p "Sync results to $PROJECT_ROOT/runs? (y/n) " -n 1 -r; echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/"
    echo "[INFO] Sync complete."
fi
