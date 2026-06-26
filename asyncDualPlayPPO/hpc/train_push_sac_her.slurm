#!/bin/bash
#SBATCH --job-name=push_sac_her
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-push-sac-her.out
#SBATCH --signal=B:USR1@120

WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (Time Limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NUM_ENVS=528
MAX_ITERATIONS=3000
SEED=42
EXP_NAME="hpc_push_sac_her_${NUM_ENVS}env"
MAX_RESUBMITS=50

PROJECT_ROOT=$(pwd)
SIF_IMAGE="${PROJECT_ROOT}/isaac-lab.sif"
CUROBO_OVERLAY="${PROJECT_ROOT}/curobo_overlay.img"

LOCAL_LOGS="${TMPDIR:-/tmp}/push_sac_her_${SLURM_JOB_ID}"
LOCAL_RUNS="${TMPDIR:-/tmp}/push_sac_runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_LOGS"
mkdir -p "$LOCAL_RUNS"
mkdir -p "$PROJECT_ROOT/.cache"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/data"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/cache"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/logs"

CONTAINER_PROJECT="/workspace/isaaclab/user_project/asyncDualPlayPPO"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP + RESUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════

cleanup() {
    echo "[INFO] Syncing results to NFS..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/sac/" 2>/dev/null
    rsync -a --no-perms "$LOCAL_LOGS/" "$PROJECT_ROOT/runs/sac/" 2>/dev/null
    echo "[INFO] Sync complete."

    # Fallback: create latest_checkpoint.zip from the most recent agent checkpoint
    # Only if latest_checkpoint.zip does NOT already exist (never overwrite).
    # LatestCheckpointCallback and SIGTERM handler already save with replay buffer.
    LATEST_AGENT_ZIP=$(find "$PROJECT_ROOT/runs/sac" \( -name "agent_*_steps.zip" -o -name "agent_final.zip" \) -type f 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_AGENT_ZIP" ]; then
        RUN_DIR="$(dirname "$LATEST_AGENT_ZIP")"
        if [ ! -f "$RUN_DIR/latest_checkpoint.zip" ]; then
            cp "$LATEST_AGENT_ZIP" "$RUN_DIR/latest_checkpoint.zip"
            echo "[CKPT] Fallback: created latest_checkpoint.zip from $(basename "$LATEST_AGENT_ZIP")"
        fi
    fi

    if [ "$WILL_RESUBMIT" = "true" ]; then
        CHAIN_COUNT=${CHAIN_COUNT:-0}
        CHAIN_COUNT=$((CHAIN_COUNT + 1))
        if [ "$CHAIN_COUNT" -ge "$MAX_RESUBMITS" ]; then
            echo "[INFO] Reached CHAIN_COUNT=$CHAIN_COUNT >= MAX_RESUBMITS=$MAX_RESUBMITS"
        else
            echo "[INFO] Chain count: $CHAIN_COUNT / $MAX_RESUBMITS"
            sbatch --export=ALL,RESUME_CHAIN=1,CHAIN_COUNT="$CHAIN_COUNT" \
                "$PROJECT_ROOT/hpc/train_push_sac_her.slurm"
        fi
    else
        echo "[INFO] Training complete — no resubmission."
        touch "$PROJECT_ROOT/runs/sac/$EXP_NAME.done"
    fi
}
trap cleanup EXIT

trap 'echo "[INFO] Caught SIGTERM; forwarding to child..."; kill -TERM $APPTAINER_PID 2>/dev/null; wait $APPTAINER_PID 2>/dev/null' TERM

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-RESUME
# ═══════════════════════════════════════════════════════════════════════════════

RESUME_ARGS=""
if [ "$RESUME_CHAIN" = "1" ] && [ -d "$PROJECT_ROOT/runs/sac" ]; then
    # Sync previous runs from NFS to local scratch so the container bind mount
    # has access to prior logs and checkpoints for resume.
    rsync -a --no-perms "$PROJECT_ROOT/runs/sac/" "$LOCAL_RUNS/" 2>/dev/null
    LATEST_CKPT=$(find "$PROJECT_ROOT/runs/sac" -name "latest_checkpoint.zip" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$LATEST_CKPT" ] && [ -s "$LATEST_CKPT" ]; then
        CKPT_PATH="$CONTAINER_PROJECT/runs/sac/${LATEST_CKPT#$PROJECT_ROOT/runs/sac/}"
        echo "[INFO] Resuming from checkpoint: $CKPT_PATH"
        RESUME_ARGS="--checkpoint $CKPT_PATH"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# JOB INFO
# ═══════════════════════════════════════════════════════════════════════════════

echo "==================================================================="
echo "  SAC + HER Push Primitive — DirectRLEnv"
echo "==================================================================="
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Num envs       : $NUM_ENVS"
echo "  Max iterations : $MAX_ITERATIONS"
echo "  Seed           : $SEED"
echo "  Experiment     : $EXP_NAME"
echo "  Resume         : ${RESUME_ARGS:-none}"
echo "==================================================================="

# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found!"
    exit 1
fi

apptainer exec --nv \
    --pwd "$CONTAINER_PROJECT" \
    --overlay "$CUROBO_OVERLAY":ro \
    --bind "$PROJECT_ROOT":"$CONTAINER_PROJECT" \
    --bind "$LOCAL_RUNS":"$CONTAINER_PROJECT/runs/sac" \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p "$CONTAINER_PROJECT/train_push_sac_her.py" \
        --headless \
        --num_envs       "$NUM_ENVS" \
        --max_iterations "$MAX_ITERATIONS" \
        --seed           "$SEED" \
        --exp_name       "$EXP_NAME" \
        $RESUME_ARGS \
    2>&1 | grep --line-buffered -v \
        -e "\[Lula\] Joint .* is specified as a mimic joint" \
        -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined" &
APPTAINER_PID=$!
wait $APPTAINER_PID
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    if [ $TRAIN_EXIT -eq 137 ] || [ $TRAIN_EXIT -eq 143 ]; then
        echo "[INFO] Training preempted (exit=$TRAIN_EXIT); will resubmit."
        WILL_RESUBMIT=true
    else
        echo "[ERROR] Training crashed with code $TRAIN_EXIT"
    fi
fi

echo "[INFO] Training finished. Exit code: $TRAIN_EXIT"
