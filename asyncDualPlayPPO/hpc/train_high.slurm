#!/bin/bash
#SBATCH --job-name=asp_high
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM, 1536 GB node RAM / 8 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output=slurm-%j-high.out
#SBATCH --signal=B:TERM@120

NUM_ENVS=2048
NSTEPS=256
MAX_ITERATIONS=1000
SAVE_INTERVAL=5
EXP_NAME="hpc_high_${NUM_ENVS}env"

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)
NFS_RUN_DIR="$PROJECT_ROOT/runs/$EXP_NAME"
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

# --- Ensure sync + resubmit on ANY exit (time limit, OOM, signal) ---
cleanup() {
    echo "[INFO] Syncing results from $LOCAL_RUNS → $PROJECT_ROOT/runs ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/" 2>/dev/null
    echo "[INFO] Sync complete."

    if [ ! -f "$NFS_RUN_DIR/bob/model_final.pt" ]; then
        NEXT_JOB=$(sbatch "$PROJECT_ROOT/hpc/train_high.slurm" 2>/dev/null | awk '{print $4}')
        echo "[INFO] Training incomplete — chained next job: $NEXT_JOB"
    else
        echo "[INFO] Training complete — no resubmission needed."
    fi
}
trap cleanup EXIT

# --- Auto-resume: find latest checkpoint on NFS ---
RESUME_ARGS=""
if [ -d "$NFS_RUN_DIR/bob" ] && [ -d "$NFS_RUN_DIR/alice" ]; then
    # Find highest iteration number checkpoint (sort numerically, not by mtime)
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

            # Copy existing run data to local scratch so container sees it
            rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/"

            RESUME_ARGS="--chkpt_alice runs/$EXP_NAME/alice/$ALICE_FILE --chkpt_bob runs/$EXP_NAME/bob/$BOB_FILE --resume_iteration $ITER_NUM"
        fi
    fi
fi

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
    --arm_config     default \
    --headless \
    $RESUME_ARGS \
    2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
