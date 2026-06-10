#!/bin/bash
#SBATCH --job-name=throw_sac
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-throw-sac.out
#SBATCH --signal=B:USR1@120

WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (Time Limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these for your run
# ═══════════════════════════════════════════════════════════════════════════════

NUM_ENVS=512
MAX_ITERATIONS=100000
SEED=42
PLAYING_ARM="right"
EXP_NAME="hpc_throw_sac_${NUM_ENVS}env"

# Paths
PROJECT_ROOT="/home/vlad/IsaacLab/vlad/master_isaac/throwing_enviroment"
VENV_PATH="/home/vlad/env_isaaclab"
LOG_DIR="$PROJECT_ROOT/logs/skrl/throwing_primitive"

# Local scratch for faster I/O (synced to NFS on exit)
LOCAL_RUNS="${TMPDIR:-/tmp}/throw_sac_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP + RESUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════

cleanup() {
    echo "[INFO] Syncing results from $LOCAL_RUNS → $LOG_DIR ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$LOG_DIR/" 2>/dev/null
    echo "[INFO] Sync complete."

    if [ "$WILL_RESUBMIT" = "true" ]; then
        SBATCH_OUT=$(sbatch "$PROJECT_ROOT/hpc/train_sac.slurm" 2>&1)
        NEXT_JOB=$(echo "$SBATCH_OUT" | awk '/Submitted batch job/{print $4}')
        if [ -n "$NEXT_JOB" ]; then
            echo "[INFO] Training incomplete — chained next job: $NEXT_JOB"
        else
            echo "[ERROR] Resubmission failed. sbatch output: $SBATCH_OUT"
        fi
    else
        echo "[INFO] Training complete or stop requested — no resubmission."
    fi
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-RESUME: find latest checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

RESUME_ARGS=""
if [ -d "$LOG_DIR" ]; then
    LATEST_DIR=$(ls -td "$LOG_DIR"/*_sac_torch 2>/dev/null | head -1)
    if [ -n "$LATEST_DIR" ] && [ -d "$LATEST_DIR" ]; then
        LATEST_CKPT=$(ls "$LATEST_DIR"/checkpoints/agent_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -z "$LATEST_CKPT" ]; then
            LATEST_CKPT=$(ls "$LATEST_DIR"/agent_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
        fi
        if [ -n "$LATEST_CKPT" ]; then
            echo "[INFO] Resuming from checkpoint: $LATEST_CKPT"
            RESUME_ARGS="--checkpoint $LATEST_CKPT"
            rsync -a --no-perms "$LATEST_DIR/" "$LOCAL_RUNS/$(basename $LATEST_DIR)/"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT JOB INFO
# ═══════════════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════"
echo "  SAC Throw Primitive — HPC Training"
echo "═══════════════════════════════════════════════════════════════"
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Node           : $SLURMD_NODENAME"
echo "  GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Num envs       : $NUM_ENVS"
echo "  Max iterations : $MAX_ITERATIONS"
echo "  Seed           : $SEED"
echo "  Arm            : $PLAYING_ARM"
echo "  Experiment     : $EXP_NAME"
echo "  Local scratch  : $LOCAL_RUNS"
echo "  Resume         : ${RESUME_ARGS:-none}"
echo "═══════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATE ENVIRONMENT + RUN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

python scripts/train_sac.py \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --seed "$SEED" \
    --playing_arm_side "$PLAYING_ARM" \
    $RESUME_ARGS \
    2>&1 | tee "$LOCAL_RUNS/train_${SLURM_JOB_ID}.log"

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Training exited with code $TRAIN_EXIT; will resubmit."
    WILL_RESUBMIT=true
fi

echo "[INFO] Training finished. Exit code: $TRAIN_EXIT"
