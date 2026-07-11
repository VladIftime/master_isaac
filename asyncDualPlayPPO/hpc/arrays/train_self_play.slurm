#!/bin/bash
# ── Array template: self-play PPO (train_e / train_g / train_d) ────────────────
# Params line format (whitespace-separated):
#   SCRIPT SEED NUM_ENVS MAX_ITERS SAVE_INTERVAL EXP_NAME [EXTRA_FLAGS...]
# EXTRA typically: --alice_pushes 5 --bob_pushes 10 --max_goals_per_episode 2
#                  --char_length 0.07 --dpose_threshold 0.055 --no_abc [--no_ge]
# Launcher sets --array / --mem / --time and exports:
#   PARAMS JOB_MEM JOB_TIME TEMPLATE MAX_RESUBMITS
#SBATCH --job-name=sp_train
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=23:00:00
#SBATCH --output=slurm-%A_%a-sp_train.out
#SBATCH --signal=B:USR1@120

set -o pipefail
PROJECT_ROOT=$(pwd)
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/$USER/final_results_thesis}"
TEMPLATE="${TEMPLATE:-$PROJECT_ROOT/hpc/arrays/train_self_play.slurm}"
JOB_MEM="${JOB_MEM:-24G}"
JOB_TIME="${JOB_TIME:-23:00:00}"
MAX_RESUBMITS="${MAX_RESUBMITS:-10}"
CHAIN_COUNT="${CHAIN_COUNT:-0}"

if [ -z "$PARAMS" ] || [ ! -f "$PARAMS" ]; then
    echo "[ERROR] PARAMS file not set or missing: '$PARAMS'"; exit 1
fi
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS")
if [ -z "$LINE" ]; then echo "[ERROR] No line $SLURM_ARRAY_TASK_ID in $PARAMS"; exit 1; fi
read -r SCRIPT SEED NUM_ENVS MAX_ITERS SAVE_INTERVAL EXP_NAME EXTRA <<< "$LINE"

WILL_RESUBMIT=false
trap 'echo "[INFO] SIGUSR1 (time limit); forwarding to container + will resubmit."; kill -USR1 $APPTAINER_PID 2>/dev/null; WILL_RESUBMIT=true' USR1

SIF_IMAGE="isaac-lab.sif"
CURBO_OVERLAY="curobo_overlay.img"
NFS_RUN_DIR="$RESULTS_ROOT/$EXP_NAME"
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS" "$RESULTS_ROOT"

cleanup() {
    echo "[INFO] Syncing $LOCAL_RUNS -> $RESULTS_ROOT ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$RESULTS_ROOT/" 2>/dev/null
    if [ "$WILL_RESUBMIT" = "true" ]; then
        if [ "$CHAIN_COUNT" -ge "$MAX_RESUBMITS" ]; then
            echo "[INFO] CHAIN_COUNT=$CHAIN_COUNT >= MAX_RESUBMITS=$MAX_RESUBMITS — stop."
        else
            sbatch --array="${SLURM_ARRAY_TASK_ID}-${SLURM_ARRAY_TASK_ID}" \
                   --mem="$JOB_MEM" --time="$JOB_TIME" \
                   --export=ALL,PARAMS="$PARAMS",JOB_MEM="$JOB_MEM",JOB_TIME="$JOB_TIME",TEMPLATE="$TEMPLATE",MAX_RESUBMITS="$MAX_RESUBMITS",CHAIN_COUNT=$((CHAIN_COUNT + 1)),RESUME_CHAIN=1,RESULTS_ROOT="$RESULTS_ROOT" \
                   "$TEMPLATE" && echo "[INFO] Chained resubmit (chain $((CHAIN_COUNT + 1))/$MAX_RESUBMITS)."
        fi
    else
        echo "[INFO] Complete or stopped — no resubmission."
        touch "$NFS_RUN_DIR/.done" 2>/dev/null
    fi
}
trap cleanup EXIT

# ── Resume detection (alice + bob latest_checkpoint) ──────────────────────────
RESUME_ARGS=""
ITER_NUM=""
if [ -f "$NFS_RUN_DIR/alice/latest_checkpoint.pt" ] && [ -f "$NFS_RUN_DIR/bob/latest_checkpoint.pt" ]; then
    ITER_NUM=$(cat "$NFS_RUN_DIR/latest_iter.txt" 2>/dev/null)
    if [ -n "$ITER_NUM" ]; then
        echo "[INFO] Resuming $EXP_NAME from iter $ITER_NUM (alice+bob)"
        rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/" 2>/dev/null
        RESUME_ARGS="--chkpt_alice runs/$EXP_NAME/alice/latest_checkpoint.pt --chkpt_bob runs/$EXP_NAME/bob/latest_checkpoint.pt --resume_iteration $ITER_NUM"
    fi
fi

echo "==================================================================="
echo "  Self-play training | array task $SLURM_ARRAY_TASK_ID"
echo "  Script    : $SCRIPT"
echo "  Exp name  : $EXP_NAME"
echo "  Seed      : $SEED   Envs: $NUM_ENVS   MaxIters: $MAX_ITERS"
echo "  Extra     : $EXTRA"
echo "  Results   : $NFS_RUN_DIR"
echo "  Resume    : ${RESUME_ARGS:-none}"
echo "==================================================================="

if [ ! -f "$SIF_IMAGE" ]; then echo "[ERROR] $SIF_IMAGE not found in $PROJECT_ROOT"; exit 1; fi
if [ ! -f "$PROJECT_ROOT/$CURBO_OVERLAY" ]; then echo "[ERROR] $CURBO_OVERLAY not found"; exit 1; fi

apptainer exec --nv --pwd /workspace/isaaclab/user_project/asyncDualPlayPPO \
    --overlay "$PROJECT_ROOT/$CURBO_OVERLAY":ro \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$LOCAL_RUNS":/workspace/isaaclab/user_project/asyncDualPlayPPO/runs \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p "/workspace/isaaclab/user_project/asyncDualPlayPPO/$SCRIPT" \
    --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" --save_interval "$SAVE_INTERVAL" \
    --seed "$SEED" --exp_name "$EXP_NAME" --headless $EXTRA $RESUME_ARGS \
    2>&1 | grep --line-buffered -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined" &
APPTAINER_PID=$!
wait $APPTAINER_PID
TRAIN_EXIT=$?
echo "[INFO] Exit code: $TRAIN_EXIT"
if [ $TRAIN_EXIT -ne 0 ] && { [ $TRAIN_EXIT -eq 137 ] || [ $TRAIN_EXIT -eq 143 ]; }; then
    echo "[INFO] Preempted; will resubmit."; WILL_RESUBMIT=true
fi
