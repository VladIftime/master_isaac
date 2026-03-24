#!/bin/bash
#SBATCH --job-name=asp_high
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # RTX Pro 6000  96 GB — most VRAM on Habrok
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=slurm-%j-high.out

# ── What this script is for ──────────────────────────────────────────────────
# Full-scale training run (24 h).  GPU VRAM is not the bottleneck — IsaacSim
# physics uses ~15 MB CPU RAM per env, so node RAM is the real limit.
# Habrok node specs (from wiki):
#
#   GPU           VRAM   Node RAM   CPUs   Max safe envs
#   ─────────────────────────────────────────────────────
#   rtx_pro_6000   96 GB   192 GB    128     2048  (4096 OOMs node RAM)
#   l40s           48 GB   512 GB     56     4096+ (plenty of RAM)
#   a100           40 GB   512 GB     64     4096+ (plenty of RAM)
#   v100           32 GB   128 GB     36      512
#
# For 4096 envs use l40s or a100 — they have 512 GB RAM.
# Current: rtx_pro_6000 with 2048 envs (~40 GB RAM usage, safe on 192 GB node).
#
# nsteps is the PPO buffer capacity per env (set in ppo_continuous.yaml).
# Actual rollout lengths: Alice=100 steps, Bob=200 steps.
# nsteps=256 gives 28% headroom over Bob's 200-step limit.
# nsteps=2048 pre-allocates ~50 GB RAM for 4096 envs → CPU OOM.
#
# Step budget: each iteration ≈ (100 + 200) × 4096 = 1.23 M env-steps.
# 1000 iterations ≈ 1.23 B steps, close to the OpenAI paper's ~1.5 B.
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

NUM_ENVS=2048   # 4096 → ~60 GB physics RAM alone (OOM). Paper uses 1856. 2048 ≈ paper scale.
NSTEPS=256

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] GPU:           RTX Pro 6000 96 GB"
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
    --num_envs       "$NUM_ENVS" \
    --nsteps         "$NSTEPS" \
    --max_iterations 1000 \
    --save_interval  50 \
    --exp_name       "hpc_high_${NUM_ENVS}env" \
    --arm_config     default \
    --headless \
    2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

# ── Sync results back to NFS ─────────────────────────────────────────────────
echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/"
echo "[INFO] Sync complete."
