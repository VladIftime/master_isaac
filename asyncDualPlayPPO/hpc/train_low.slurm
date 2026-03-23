#!/bin/bash
#SBATCH --job-name=asp_smoke
#SBATCH --time=01:00:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1       # A100  40 GB — V100 lacks sm_80 support for this container
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j-smoke.out

# ── What this script is for ──────────────────────────────────────────────────
# Fast smoke-test to verify the ABC fix is working.  20 min on a V100.
#
# What to check in the output:
#   1. ABC loss should NOT be flat at -1.016 — expect variation (-3 to -5 range)
#   2. Bob's Rew should show non-zero values
#   3. No "[Alice Update] Skipped" messages (both agents always update now)
#   4. Bob SR should start climbing within ~50 iterations
#
# nsteps = 256 (min safe value above bob_timesteps=200, saves memory+time)
# num_envs = 64 (small enough for fast iterations, large enough for signal)
# max_iterations = 100 (enough to see if ABC loss moves)
# save_interval = 50 (only 2 checkpoints to minimize I/O)
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

NUM_ENVS=64
NSTEPS=256

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] GPU:           V100 32 GB"
echo "[INFO] Envs × nsteps: ${NUM_ENVS} × ${NSTEPS} = $(( NUM_ENVS * NSTEPS )) buffer slots"
echo "[INFO] Local runs:    $LOCAL_RUNS"
echo "[INFO] Mode:          SMOKE TEST — verifying ABC fix"

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
    --max_iterations 100 \
    --save_interval  50 \
    --exp_name      "smoke_abc_fix" \
    --arm_config    default \
    --headless \
    2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

# ── Sync results back to NFS ─────────────────────────────────────────────────
echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/"
echo "[INFO] Sync complete."

# ── Quick summary ─────────────────────────────────────────────────────────────
echo ""
echo "=== SMOKE TEST CHECKLIST ==="
echo "Grep the output for these:"
echo "  grep 'ABC'        → loss should vary, NOT stuck at -1.016"
echo "  grep 'BobSuccess' → should appear within first 100 iters"
echo "  grep 'Skipped'    → should be ZERO matches (Alice always updates)"
echo "==========================="
