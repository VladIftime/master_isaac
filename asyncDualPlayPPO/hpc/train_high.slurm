#!/bin/bash
#SBATCH --job-name=dense_high
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=slurm-%j-high_alice_dense_4096.out

# Load Apptainer
# Apptainer is pre-installed on this system, no module load needed
# module load local_apptainer || module load Apptainer

# Config
SIF_IMAGE="isaac-lab.sif" # Ensure this image matches your Isaac Lab version
PROJECT_ROOT=$(pwd)

# Command to run (training)
CMD="/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train.py"

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"
echo "[INFO] Root: $PROJECT_ROOT"

# Check for image
if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found!"
    echo "[INFO] Please pull the image: apptainer pull isaac-lab.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0"
    exit 1
fi

# Run Container
# We utilize the --bind flags to mount:
# 1. The project directory to /workspace/isaaclab/user_project
# 2. The logs directory to persist data
# 3. .cache to speed up shader compilation
# 4. .isaac_cache to provide writable directories for Isaac Sim internal data
apptainer exec --nv \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project \
    --bind "$PROJECT_ROOT/runs":/workspace/isaaclab/logs \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    $CMD \
    --num_envs 2048 \
    --max_iterations 100000 \
    --exp_name "hpc_ppo_high" \
    --save_interval 2000 \
    --headless | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
