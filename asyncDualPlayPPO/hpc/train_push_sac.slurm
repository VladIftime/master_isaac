#!/bin/bash
#SBATCH --job-name=push_sac
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-push-sac.out
#SBATCH --signal=B:USR1@120
WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (Time Limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

NUM_ENVS=2048
MAX_ITERATIONS=100000
SAVE_INTERVAL=10
EXP_NAME="hpc_push_sac_abs_${NUM_ENVS}env"
EXTRA_ARGS=""
# Uncomment for relative observation + action mode:
# EXP_NAME="hpc_push_sac_rel_${NUM_ENVS}env"
# EXTRA_ARGS="--rel-obs --rel-act"

SIF_IMAGE="isaac-lab.sif"
CURBO_OVERLAY="curobo_overlay.img"
PROJECT_ROOT=$(pwd)
NFS_RUN_DIR="$PROJECT_ROOT/runs/$EXP_NAME"
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

cleanup() {
    echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/" 2>/dev/null
    echo "[INFO] Sync complete."

    if [ "$WILL_RESUBMIT" = "true" ]; then
        SBATCH_OUT=$(sbatch --export=ALL,RESUME_CHAIN=1 "$PROJECT_ROOT/asyncDualPlayPPO/hpc/train_push_sac.slurm" 2>&1)
        NEXT_JOB=$(echo "$SBATCH_OUT" | awk '/Submitted batch job/{print $4}')
        if [ -n "$NEXT_JOB" ]; then
            echo "[INFO] Training incomplete — chained next job: $NEXT_JOB"
        else
            echo "[ERROR] Resubmission failed. sbatch output: $SBATCH_OUT"
        fi
    else
        echo "[INFO] Training complete or stop requested — no resubmission."
        touch "$PROJECT_ROOT/runs/$EXP_NAME/.done"
    fi
}
trap cleanup EXIT

RESUME_ARGS=""
if [ "$RESUME_CHAIN" = "1" ] && [ -d "$NFS_RUN_DIR/agent" ]; then
    LATEST_AGENT=$(ls "$NFS_RUN_DIR/agent"/agent_*.pt 2>/dev/null | grep -v best | grep -v final | grep -v emergency \
        | sed 's/.*agent_//' | sed 's/\.pt//' | sort -n | tail -1)
    if [ -n "$LATEST_AGENT" ]; then
        AGENT_FILE="agent_${LATEST_AGENT}.pt"
        if [ -f "$NFS_RUN_DIR/agent/$AGENT_FILE" ]; then
            echo "[INFO] Resuming from checkpoint $AGENT_FILE"
            rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/"
            RESUME_ARGS="--chkpt runs/$EXP_NAME/agent/$AGENT_FILE"
        fi
    fi
fi

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] Envs:          $NUM_ENVS"
echo "[INFO] Experiment:    $EXP_NAME"
echo "[INFO] Extra args:    $EXTRA_ARGS"
echo "[INFO] Local runs:    $LOCAL_RUNS"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found!"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$CURBO_OVERLAY" ]; then
    echo "[ERROR] $CURBO_OVERLAY not found in $PROJECT_ROOT"
    exit 1
fi

apptainer exec --nv --env WARP_CACHE_ROOT=/tmp/warp_cache --pwd /workspace/isaaclab/user_project/asyncDualPlayPPO \
    --overlay "$PROJECT_ROOT/$CURBO_OVERLAY":ro \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$LOCAL_RUNS":/workspace/isaaclab/user_project/asyncDualPlayPPO/runs \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train_push_sac.py \
    --num_envs       "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --save_interval  "$SAVE_INTERVAL" \
    --exp_name       "$EXP_NAME" \
    --headless \
    $EXTRA_ARGS \
    $RESUME_ARGS \
    2>&1 | grep --line-buffered -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
TRAIN_EXIT=$?

echo "[INFO] Training exit code: $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Training command failed; will resubmit."
    WILL_RESUBMIT=true
fi
