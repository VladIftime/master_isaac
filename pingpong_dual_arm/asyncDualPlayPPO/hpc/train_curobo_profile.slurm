#!/bin/bash
#SBATCH --job-name=asp_curobo_prof
#SBATCH --time=00:45:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=slurm-%j-curobo-profile.out

set -o pipefail

NUM_ENVS=2048
MAX_ITERATIONS=3
NUM_OBJECTS=1
EXP_NAME="profile_curobo_${NUM_ENVS}env_${MAX_ITERATIONS}iter"

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:   $SLURM_JOB_ID"
echo "[INFO] Node:     $SLURMD_NODENAME"
echo "[INFO] Envs:     $NUM_ENVS"
echo "[INFO] Iters:    $MAX_ITERATIONS"
echo "[INFO] Exp:      $EXP_NAME"
echo "[INFO] Note: cuRobo CUDA graph warm-up (~30 s) precedes training; not counted in profiler timings."

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found in $(pwd)"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/curobo_overlay.img" ]; then
    echo "[ERROR] curobo_overlay.img not found in $PROJECT_ROOT"
    echo "[INFO]  Run: apptainer overlay create --size 8192 curobo_overlay.img"
    exit 1
fi

# Purge stale bytecode so the container always recompiles from current .py sources
find "$PROJECT_ROOT" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "[INFO] Cleared __pycache__ directories"

apptainer exec --nv --overlay "$PROJECT_ROOT/curobo_overlay.img":ro \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train_curobo.py \
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
