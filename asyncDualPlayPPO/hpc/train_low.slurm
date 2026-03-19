#!/bin/bash
#SBATCH --job-name=asp_low
#SBATCH --time=01:00:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=v100:1       # V100  32 GB — cheapest/fastest queue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j-low.out

# ── What this script is for ──────────────────────────────────────────────────
# Quick smoke-test / env-count sweep.  Runs for 1 hour on a V100 (32 GB).
# Swap --gpus-per-node to a100:1 (40 GB) or l40s:1 (48 GB) if V100 is busy.
#
# Env-count guide (headless, single-GPU):
#   V100  32 GB  →  512 envs   (safe)  |  1024 envs  (tight, watch OOM)
#   A100  40 GB  →  1024 envs  (safe)  |  2048 envs  (tight)
#   L40S  48 GB  →  2048 envs  (safe)
#   RTX Pro 6000 96 GB → 4096+ envs
#
# nsteps must be ≥ bob_timesteps (200).  512 gives a comfortable margin and
# keeps storage at 512 × num_envs × obs_dim — well within 32 GB.
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

NUM_ENVS=512
NSTEPS=512

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] GPU:           V100 32 GB"
echo "[INFO] Envs × nsteps: ${NUM_ENVS} × ${NSTEPS} = $(( NUM_ENVS * NSTEPS )) buffer slots"
echo "[INFO] Local runs:    $LOCAL_RUNS"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found!"
    echo "[INFO]  Pull it with: apptainer pull isaac-lab.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0"
    exit 1
fi

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
    --num_envs      "$NUM_ENVS" \
    --nsteps        "$NSTEPS" \
    --max_iterations 200 \
    --save_interval  25 \
    --exp_name      "hpc_low_${NUM_ENVS}env" \
    --arm_config    default \
    --headless \
    2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

# ── Sync results back to NFS ─────────────────────────────────────────────────
echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/"
echo "[INFO] Sync complete."
