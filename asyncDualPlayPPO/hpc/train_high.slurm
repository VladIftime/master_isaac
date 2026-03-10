#!/bin/bash
#SBATCH --job-name=adp_high
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --output=slurm-%j-high.out

# ── I/O config ──────────────────────────────────────────────────────────────
# Write checkpoints and TensorBoard logs to the node's fast local scratch.
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"
# ────────────────────────────────────────────────────────────────────────────

# ── Env scaling ──────────────────────────────────────────────────────────────
# Buffer budget: keep total transitions ≈ 131 072.
#   num_envs=4096 → nsteps = 131072 / 4096 = 32
NUM_ENVS=4096
NSTEPS=32
# ────────────────────────────────────────────────────────────────────────────

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] Project root:  $PROJECT_ROOT"
echo "[INFO] Local runs:    $LOCAL_RUNS"
echo "[INFO] Buffer size:   ${NUM_ENVS} envs × ${NSTEPS} steps = $(( NUM_ENVS * NSTEPS )) transitions"

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
    --num_envs "$NUM_ENVS" \
    --nsteps   "$NSTEPS" \
    --max_iterations 100000 \
    --exp_name "hpc_high" \
    --max_alice_bob_ratio 5 \
    --save_interval 2000 \
    --headless \
    | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
               -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

# ── Sync results back to NFS ─────────────────────────────────────────────────
echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/"
echo "[INFO] Sync complete."
