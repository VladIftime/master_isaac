#!/bin/bash
#SBATCH --job-name=adp_low
#SBATCH --time=01:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j-low.out

# ── I/O config ──────────────────────────────────────────────────────────────
# Write checkpoints and TensorBoard logs to the node's fast local scratch,
# not to the NFS home directory.  Large .pt files on NFS cause 30-60s stalls
# every few updates.  At the end of the job, rsync the results back home.
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"
# ────────────────────────────────────────────────────────────────────────────

# ── Env scaling ──────────────────────────────────────────────────────────────
# Buffer budget: keep total transitions ≈ 131 072 (fits on an A100 with room).
#   num_envs=128 → nsteps = 131072 / 128 = 128
# num_envs is passed on the command line; nsteps is set via --nsteps override.
NUM_ENVS=128    
NSTEPS=128
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
    --max_iterations 5000 \
    --exp_name "hpc_low" \
    --max_alice_bob_ratio 5 \
    --save_interval 500 \
    --headless \
    | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
               -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

# ── Sync results back to NFS ─────────────────────────────────────────────────
echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/"
echo "[INFO] Sync complete."
