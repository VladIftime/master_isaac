#!/bin/bash
#SBATCH --job-name=asp_safe
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-push_asp_safe.out
#SBATCH --signal=B:USR1@120
WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (Time Limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

NUM_ENVS=1024
MAX_ITERATIONS=100000
SAVE_INTERVAL=100
ALICE_PUSHES=5
BOB_PUSHES=10
MAX_GOALS=2
EXP_NAME="hpc_push_asp_safe_${NUM_ENVS}env"

SIF_IMAGE="isaac-lab.sif"
CURBO_OVERLAY="curobo_overlay.img"
PROJECT_ROOT=$(pwd)
NFS_RUN_DIR="$PROJECT_ROOT/runs/$EXP_NAME"
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

# --- Ensure sync + resubmit on ANY exit ---
cleanup() {
    echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/" 2>/dev/null
    echo "[INFO] Sync complete."

    if [ "$WILL_RESUBMIT" = "true" ]; then
        SBATCH_OUT=$(sbatch "$PROJECT_ROOT/asyncDualPlayPPO/hpc/train_push_asp_new_prim.slurm" 2>&1)
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

# --- Auto-resume ---
RESUME_ARGS=""
ITER_NUM=""
if [ -d "$NFS_RUN_DIR/bob" ] && [ -d "$NFS_RUN_DIR/alice" ]; then
    LATEST_BOB=$(ls "$NFS_RUN_DIR/bob"/model_*.pt 2>/dev/null | grep -v best | grep -v final \
        | sed 's/.*model_//' | sed 's/\.pt//' | sort -n | tail -1)
    if [ -n "$LATEST_BOB" ]; then
        ITER_NUM="$LATEST_BOB"
        BOB_FILE="model_${ITER_NUM}.pt"
        ALICE_FILE="model_${ITER_NUM}.pt"
        if [ -f "$NFS_RUN_DIR/bob/$BOB_FILE" ] && [ -f "$NFS_RUN_DIR/alice/$ALICE_FILE" ]; then
            echo "[INFO] Resuming from iteration $ITER_NUM"
            echo "[INFO]   Alice: $NFS_RUN_DIR/alice/$ALICE_FILE"
            echo "[INFO]   Bob:   $NFS_RUN_DIR/bob/$BOB_FILE"
            rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/"
            RESUME_ARGS="--chkpt_alice runs/$EXP_NAME/alice/$ALICE_FILE --chkpt_bob runs/$EXP_NAME/bob/$BOB_FILE --resume_iteration $ITER_NUM"
        fi
    fi
fi

GLOBAL_ITER_START=${ITER_NUM:-0}
echo "[INFO] === P54 ONLY (safe descend + slide-through, no penalty) ==="
echo "[INFO] Global iteration start: $GLOBAL_ITER_START"
echo "[INFO] Max iterations target:  $MAX_ITERATIONS"
echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Envs:          $NUM_ENVS"
echo "[INFO] Alice pushes:  $ALICE_PUSHES"
echo "[INFO] Bob pushes:    $BOB_PUSHES"

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
    /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train_push_asp.py \
    --num_envs            "$NUM_ENVS" \
    --max_iterations      "$MAX_ITERATIONS" \
    --save_interval       "$SAVE_INTERVAL" \
    --alice_pushes        "$ALICE_PUSHES" \
    --bob_pushes          "$BOB_PUSHES" \
    --max_goals_per_episode "$MAX_GOALS" \
    --no_abc \
    --use_safe_descend \
    --exp_name            "$EXP_NAME" \
    --headless \
    $RESUME_ARGS \
    2>&1 | grep --line-buffered -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
TRAIN_EXIT=$?

LATEST_AFTER=$(ls "$NFS_RUN_DIR/bob"/model_*.pt 2>/dev/null | grep -v best | grep -v final \
    | sed 's/.*model_//' | sed 's/\.pt//' | sort -n | tail -1)
echo "[INFO] Global iteration end: ${LATEST_AFTER:-unknown}"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Training command failed; will resubmit."
    WILL_RESUBMIT=true
fi
