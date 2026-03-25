#!/bin/bash
#SBATCH --job-name=asp_high
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM, 1536 GB node RAM / 8 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output=slurm-%j-high.out

LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

NUM_ENVS=4096
NSTEPS=256

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] Envs × nsteps: ${NUM_ENVS} × ${NSTEPS}"
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
    --num_envs       "$NUM_ENVS" \
    --nsteps         "$NSTEPS" \
    --max_iterations 1000 \
    --save_interval  50 \
    --exp_name       "hpc_high_${NUM_ENVS}env" \
    --arm_config     default \
    --headless \
    2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/"
echo "[INFO] Sync complete."
