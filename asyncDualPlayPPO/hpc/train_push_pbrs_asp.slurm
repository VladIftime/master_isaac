#!/bin/bash
#SBATCH --job-name=pbrs_asp
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-pbrs_asp.out
#SBATCH --signal=B:USR1@120
WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (Time Limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

NUM_ENVS=528
MAX_ITERATIONS=3000
SAVE_INTERVAL=100
ALICE_PUSHES=5
BOB_PUSHES=10
MAX_GOALS=2
EXP_NAME="hpc_pbrs_asp_${NUM_ENVS}env"

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
        SBATCH_OUT=$(sbatch "$PROJECT_ROOT/hpc/train_push_pbrs_asp.slurm" 2>&1)
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

RESUME_ARGS=""
ITER_NUM=""
if [ -f "$NFS_RUN_DIR/alice/latest_checkpoint.pt" ] && [ -f "$NFS_RUN_DIR/bob/latest_checkpoint.pt" ]; then
    ITER_NUM=$(cat "$NFS_RUN_DIR/latest_iter.txt" 2>/dev/null)
    if [ -n "$ITER_NUM" ]; then
        echo "[INFO] Resuming from iteration $ITER_NUM"
        echo "[INFO]   Alice: $NFS_RUN_DIR/alice/latest_checkpoint.pt"
        echo "[INFO]   Bob:   $NFS_RUN_DIR/bob/latest_checkpoint.pt"
        rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/"
        RESUME_ARGS="--chkpt_alice runs/$EXP_NAME/alice/latest_checkpoint.pt --chkpt_bob runs/$EXP_NAME/bob/latest_checkpoint.pt --resume_iteration $ITER_NUM"
    fi
fi

GLOBAL_ITER_START=${ITER_NUM:-0}
echo "[INFO] Global iteration start: $GLOBAL_ITER_START"
echo "[INFO] Max iterations target:  $MAX_ITERATIONS"
echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] Envs:          $NUM_ENVS"
echo "[INFO] Model:         PBRS Model C (ASP + PBRS)"
echo "[INFO] Alice pushes:  $ALICE_PUSHES"
echo "[INFO] Bob pushes:    $BOB_PUSHES"
echo "[INFO] Max goals/ep:  $MAX_GOALS"
echo "[INFO] Local runs:    $LOCAL_RUNS"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found!"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/$CURBO_OVERLAY" ]; then
    echo "[ERROR] $CURBO_OVERLAY not found in $PROJECT_ROOT"
    exit 1
fi

apptainer exec --nv --pwd /workspace/isaaclab/user_project/asyncDualPlayPPO \
    --overlay "$PROJECT_ROOT/$CURBO_OVERLAY":ro \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$LOCAL_RUNS":/workspace/isaaclab/user_project/asyncDualPlayPPO/runs \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train_push_pbrs_asp.py \
    --num_envs            "$NUM_ENVS" \
    --max_iterations      "$MAX_ITERATIONS" \
    --save_interval       "$SAVE_INTERVAL" \
    --alice_pushes        "$ALICE_PUSHES" \
    --bob_pushes          "$BOB_PUSHES" \
    --max_goals_per_episode "$MAX_GOALS" \
    --no_abc \
    --rel-obs \
    --exp_name            "$EXP_NAME" \
    --headless \
    $RESUME_ARGS \
    2>&1 | grep --line-buffered -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
TRAIN_EXIT=$?

LATEST_AFTER=$(cat "$NFS_RUN_DIR/latest_iter.txt" 2>/dev/null)
echo "[INFO] Global iteration end: ${LATEST_AFTER:-unknown}"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Training command failed; will resubmit."
    WILL_RESUBMIT=true
fi
