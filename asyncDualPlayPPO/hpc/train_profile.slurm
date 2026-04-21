#!/bin/bash
#SBATCH --job-name=asp_profile
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=slurm-%j-profile.out

set -o pipefail

NUM_ENVS=2048
MAX_ITERATIONS=3
NUM_OBJECTS=1
EXP_NAME="profile_${NUM_ENVS}env_${MAX_ITERATIONS}iter"

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:   $SLURM_JOB_ID"
echo "[INFO] Node:     $SLURMD_NODENAME"
echo "[INFO] Envs:     $NUM_ENVS"
echo "[INFO] Iters:    $MAX_ITERATIONS"
echo "[INFO] Exp:      $EXP_NAME"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found in $(pwd)"
    exit 1
fi

# Purge stale bytecode so the container always recompiles from the current .py sources
find "$PROJECT_ROOT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "[INFO] Cleared __pycache__ directories"

apptainer exec --nv \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train.py \
    --num_envs       "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --num_objects    "$NUM_OBJECTS" \
    --exp_name       "$EXP_NAME" \
    --arm_config     rotated \
    --headless \
    --profile \
    2>&1 | grep --line-buffered -v \
        -e "\[Lula\] Joint .* is specified as a mimic joint" \
        -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
