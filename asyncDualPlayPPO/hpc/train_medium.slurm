#!/bin/bash
#SBATCH --job-name=dense_med
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --output=slurm-%j-dense_med_4096.out


# Load Apptainer
# Apptainer is pre-installed on this system, no module load needed
# module load local_apptainer || module load Apptainer

# Config
SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found!"
    exit 1
fi

# Medium profile: 4096 environments, 12 hours
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
    --max_iterations 50000 \
    --exp_name "hpc_ppo_med" \
    --save_interval 1000 \
    --headless | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
