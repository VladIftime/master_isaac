#!/bin/bash
#SBATCH --job-name=throw_sac_d
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-throw-sac-direct.out
#SBATCH --signal=B:USR1@120

WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (Time Limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NUM_ENVS=4096
MAX_ITERATIONS=100000
SEED=42
PLAYING_ARM="right"
EXP_NAME="hpc_throw_sac_direct_${NUM_ENVS}env"

PROJECT_ROOT=$(pwd)
SIF_IMAGE="$(dirname "$PROJECT_ROOT")/asyncDualPlayPPO/isaac-lab.sif"
LOG_DIR="$PROJECT_ROOT/logs/sac/throwing_primitive"

LOCAL_LOGS="${TMPDIR:-/tmp}/throw_sac_direct_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_LOGS"
mkdir -p "$PROJECT_ROOT/.cache/warp"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/data"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/cache"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/logs"

CONTAINER_PROJECT="/workspace/isaaclab/user_project/throwing_enviroment"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP + RESUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════

cleanup() {
    echo "[INFO] Syncing results from $LOCAL_LOGS -> $LOG_DIR ..."
    mkdir -p "$LOG_DIR"
    rsync -a --no-perms "$LOCAL_LOGS/" "$LOG_DIR/" 2>/dev/null
    echo "[INFO] Sync complete."

    # Fallback: create latest_checkpoint.zip from most recent agent checkpoint
    # (only if signal-saved latest_checkpoint.zip doesn't already exist anywhere)
    EXISTING_LATEST=$(find "$LOG_DIR" -name "latest_checkpoint.zip" -type f 2>/dev/null | head -1)
    if [ -z "$EXISTING_LATEST" ]; then
        LATEST_AGENT_ZIP=$(find "$LOG_DIR" -name "agent_*_steps.zip" -type f 2>/dev/null | sort -V | tail -1)
        if [ -n "$LATEST_AGENT_ZIP" ]; then
            RUN_DIR=$(dirname "$(dirname "$LATEST_AGENT_ZIP")")
            cp "$LATEST_AGENT_ZIP" "$RUN_DIR/latest_checkpoint.zip"
            echo "[CKPT] Fallback: created latest_checkpoint.zip from $(basename "$LATEST_AGENT_ZIP")"
        fi
    fi

    if [ "$WILL_RESUBMIT" = "true" ]; then
        SBATCH_OUT=$(sbatch --export=ALL,RESUME_CHAIN=1 "$PROJECT_ROOT/hpc/train_sac_direct.slurm" 2>&1)
        NEXT_JOB=$(echo "$SBATCH_OUT" | awk '/Submitted batch job/{print $4}')
        if [ -n "$NEXT_JOB" ]; then
            echo "[INFO] Training incomplete — chained next job: $NEXT_JOB"
        else
            echo "[ERROR] Resubmission failed. sbatch output: $SBATCH_OUT"
        fi
    else
        echo "[INFO] Training complete or stop requested — no resubmission."
        touch "$LOG_DIR/$EXP_NAME.done"
    fi
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-RESUME
# ═══════════════════════════════════════════════════════════════════════════════

RESUME_ARGS=""
if [ "$RESUME_CHAIN" = "1" ] && [ -d "$LOG_DIR" ]; then
    LATEST_CKPT_HOST=$(find "$LOG_DIR" -name "latest_checkpoint.zip" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST_CKPT_HOST" ]; then
        LATEST_CKPT=$(echo "$LATEST_CKPT_HOST" | sed "s|^$PROJECT_ROOT|$CONTAINER_PROJECT|")
        echo "[INFO] Resuming from checkpoint: $LATEST_CKPT"
        RESUME_ARGS="--checkpoint $LATEST_CKPT"
    fi
    rsync -a --no-perms "$LOG_DIR/" "$LOCAL_LOGS/"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# JOB INFO
# ═══════════════════════════════════════════════════════════════════════════════

echo "==================================================================="
echo "  SAC Throw Primitive — DirectRLEnv (fast)"
echo "==================================================================="
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Node           : $SLURMD_NODENAME"
echo "  GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Num envs       : $NUM_ENVS"
echo "  Max iterations : $MAX_ITERATIONS"
echo "  Seed           : $SEED"
echo "  Arm            : $PLAYING_ARM"
echo "  Experiment     : $EXP_NAME"
echo "  Project root   : $PROJECT_ROOT"
echo "  Container      : $SIF_IMAGE"
echo "  Local scratch  : $LOCAL_LOGS"
echo "  Resume         : ${RESUME_ARGS:-none}"
echo "==================================================================="

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found!"
    echo "[INFO]  Pull it with: apptainer pull isaac-lab.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0"
    exit 1
fi

apptainer exec --nv \
    --pwd "$CONTAINER_PROJECT" \
    --env PYTHONPATH="$CONTAINER_PROJECT:$CONTAINER_PROJECT/source/Throwing" \
    --env WARP_CACHE_PATH=/root/.cache/warp \
    --bind "$PROJECT_ROOT":"$CONTAINER_PROJECT" \
    --bind "$LOCAL_LOGS":"$CONTAINER_PROJECT/logs/sac/throwing_primitive" \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p "$CONTAINER_PROJECT/scripts/train_sac.py" \
        --headless \
        --num_envs       "$NUM_ENVS" \
        --max_iterations "$MAX_ITERATIONS" \
        --seed           "$SEED" \
        --playing_arm_side "$PLAYING_ARM" \
        $RESUME_ARGS \
    2>&1 | tee "$LOCAL_LOGS/train_${SLURM_JOB_ID}.log"

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Training exited with code $TRAIN_EXIT; will resubmit."
    WILL_RESUBMIT=true
fi

echo "[INFO] Training finished. Exit code: $TRAIN_EXIT"
