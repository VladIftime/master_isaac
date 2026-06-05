#!/bin/bash
#SBATCH --job-name=asp_resume
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM, 1536 GB node RAM / 8 GPUs = 192 GB/slot
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-resume.out
#SBATCH --signal=B:USR1@120

WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (Time Limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

NUM_ENVS=256
NSTEPS=256
NUM_OBJECTS=1
EXP_NAME="hpc_small_${NUM_ENVS}env_${NUM_OBJECTS}obj"

# --- SPECIFIC CHECKPOINT TO RESUME FROM ---
# Based on our investigation, the highest valid saved checkpoint is 3800.
# The job slurm-28696521-small.out crashed mid-epoch at 4450 before saving.
RESUME_ITER=3800
# We add 100,000 to the current iteration to train for another 100,000 epochs
MAX_ITERATIONS=$((RESUME_ITER + 100000))
SAVE_INTERVAL=10

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

# IMPORTANT: If your files are still in /scratch/, use restore_training_logs.sh first
# or change this NFS_RUN_DIR to point to /scratch/s3426394/runs/$EXP_NAME
NFS_RUN_DIR="$PROJECT_ROOT/runs/$EXP_NAME"

LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

# --- Ensure sync + resubmit on ANY exit (time limit, OOM, signal) ---
cleanup() {
    echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/" 2>/dev/null
    echo "[INFO] Sync complete."

    if [ "$WILL_RESUBMIT" = "true" ]; then
        SBATCH_OUT=$(sbatch "$PROJECT_ROOT/hpc/train_small_resume.slurm" 2>&1)
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

# --- Resume logic for SPECIFIC iteration ---
BOB_FILE="model_${RESUME_ITER}.pt"
ALICE_FILE="model_${RESUME_ITER}.pt"

if [ -f "$NFS_RUN_DIR/bob/$BOB_FILE" ] && [ -f "$NFS_RUN_DIR/alice/$ALICE_FILE" ]; then
    echo "[INFO] Resuming EXPLICITLY from iteration $RESUME_ITER"
    echo "[INFO]   Alice: $NFS_RUN_DIR/alice/$ALICE_FILE"
    echo "[INFO]   Bob:   $NFS_RUN_DIR/bob/$BOB_FILE"

    # Copy existing run data to local scratch so container sees it
    rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/"

    RESUME_ARGS="--chkpt_alice runs/$EXP_NAME/alice/$ALICE_FILE --chkpt_bob runs/$EXP_NAME/bob/$BOB_FILE --resume_iteration $RESUME_ITER"
else
    echo "[ERROR] Specific checkpoints not found at $NFS_RUN_DIR for iter $RESUME_ITER"
    echo "[ERROR] Either the iteration is wrong or you need to restore files from /scratch/"
    exit 1
fi

GLOBAL_ITER_START=$RESUME_ITER
echo "[INFO] Global iteration start: $GLOBAL_ITER_START"
echo "[INFO] Max iterations target:  $MAX_ITERATIONS"

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] Envs × nsteps: ${NUM_ENVS} × ${NSTEPS}"
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
    --max_iterations "$MAX_ITERATIONS" \
    --save_interval  "$SAVE_INTERVAL" \
    --exp_name       "$EXP_NAME" \
    --arm_config     rotated \
    --num_objects    "$NUM_OBJECTS" \
    --headless \
    $RESUME_ARGS \
    2>&1 | grep --line-buffered -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                                  -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
TRAIN_EXIT=$?

# Log final checkpoint iteration so the next job knows where we ended
LATEST_AFTER=$(ls "$NFS_RUN_DIR/bob"/model_*.pt 2>/dev/null | grep -v best | grep -v final \
    | sed 's/.*model_//' | sed 's/\.pt//' | sort -n | tail -1)
echo "[INFO] Global iteration end: ${LATEST_AFTER:-unknown}"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "[ERROR] Training command failed; will resubmit."
    WILL_RESUBMIT=true
fi
