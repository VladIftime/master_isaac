#!/bin/bash
#SBATCH --job-name=adp_a100_scale
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-high.out

# ── I/O config ──────────────────────────────────────────────────────────────
# Write checkpoints and TensorBoard logs to the node's fast local scratch.
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"
# ────────────────────────────────────────────────────────────────────────────

# ── Env scaling for A100 (40GB/80GB) ────────────────────────────────────────
# We want to saturate the A100's compute while maintaining good PPO update frequency.
# 8,192 environments is the sweet spot. 
# 1024 steps ensures multiple full Alice/Bob episodes are completed before updating.
# Buffer size: 8,192 * 1024 = 8,388,608 transitions per update.
NUM_ENVS=8192
NSTEPS=1024
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

# To reach the OpenAI paper scale of ~1.5 Billion steps:
# 1,500,000,000 / 8,388,608 ≈ 180 iterations. 
# We set max_iterations to 250 (approx. 2.1 Billion steps) to ensure it keeps 
# training if the A100 crunches through it faster than expected within 24 hours.

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
    --max_iterations 250 \
    --exp_name "hpc_a100_run_1.5B_steps"