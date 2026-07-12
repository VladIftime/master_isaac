#!/bin/bash
# ── Array template: gym-pusht controlled testbed (train_*_gym_*) ──────────────
# Params line format (whitespace-separated):
#   SCRIPT SEED NUM_ENVS PUSH_NSTEPS MAX_ITERS SAVE_INTERVAL EXP_NAME [EXTRA_FLAGS...]
# Launcher sets --array / --mem / --time and exports:
#   PARAMS JOB_MEM JOB_TIME TEMPLATE MAX_RESUBMITS
#SBATCH --job-name=gym_train
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=slurm-%A_%a-gym_train.out
#SBATCH --signal=B:USR1@120
#SBATCH --requeue                        # preempt/qos cluster: let SLURM auto-requeue on preemption

set -o pipefail
PROJECT_ROOT=$(pwd)
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/$USER/final_results_thesis}"
TEMPLATE="${TEMPLATE:-$PROJECT_ROOT/hpc/arrays/train_gym.slurm}"
JOB_MEM="${JOB_MEM:-16G}"
JOB_TIME="${JOB_TIME:-23:00:00}"
MAX_RESUBMITS="${MAX_RESUBMITS:-10}"
CHAIN_COUNT="${CHAIN_COUNT:-0}"
source "$PROJECT_ROOT/hpc/arrays/_campaign.sh"

if [ -z "$PARAMS" ] || [ ! -f "$PARAMS" ]; then
    echo "[ERROR] PARAMS file not set or missing: '$PARAMS'"; exit 1
fi
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS")
if [ -z "$LINE" ]; then echo "[ERROR] No line $SLURM_ARRAY_TASK_ID in $PARAMS"; exit 1; fi
read -r SCRIPT SEED NUM_ENVS PUSH_NSTEPS MAX_ITERS SAVE_INTERVAL EXP_NAME EXTRA <<< "$LINE"

WILL_RESUBMIT=false
trap 'echo "[INFO] SIGUSR1; forwarding + will resubmit."; kill -USR1 $APPTAINER_PID 2>/dev/null; WILL_RESUBMIT=true' USR1

SIF_IMAGE="isaac-lab.sif"
GYM_OVERLAY="gym_overlay.img"
NFS_RUN_DIR="$RESULTS_ROOT/$EXP_NAME"
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS" "$RESULTS_ROOT"

cleanup() {
    kill "${FLUSHER_PID:-}" 2>/dev/null
    echo "[INFO] Syncing $LOCAL_RUNS -> $RESULTS_ROOT ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$RESULTS_ROOT/" 2>/dev/null
    campaign_clear_running
    if [ "$WILL_RESUBMIT" = "true" ] && ! campaign_aborted; then
        if [ "$CHAIN_COUNT" -ge "$MAX_RESUBMITS" ]; then
            echo "[INFO] CHAIN_COUNT=$CHAIN_COUNT >= MAX_RESUBMITS=$MAX_RESUBMITS — stop."
        else
            SB_OUT=$(sbatch --array="${SLURM_ARRAY_TASK_ID}-${SLURM_ARRAY_TASK_ID}" \
                   --mem="$JOB_MEM" --time="$JOB_TIME" \
                   --export=ALL,PARAMS="$PARAMS",JOB_MEM="$JOB_MEM",JOB_TIME="$JOB_TIME",TEMPLATE="$TEMPLATE",MAX_RESUBMITS="$MAX_RESUBMITS",CHAIN_COUNT=$((CHAIN_COUNT + 1)),RESUME_CHAIN=1,RESULTS_ROOT="$RESULTS_ROOT",CAMPAIGN_DIR="${CAMPAIGN_DIR:-}",FAIL_LOG="${FAIL_LOG:-}",ABORT_ON_FAIL="${ABORT_ON_FAIL:-1}" \
                   "$TEMPLATE" 2>&1)
            echo "$SB_OUT"
            NEWID=$(echo "$SB_OUT" | awk '/Submitted batch job/{print $4}')
            [ -n "$NEWID" ] && [ -n "${CAMPAIGN_DIR:-}" ] && touch "$CAMPAIGN_DIR/jobids/$NEWID" 2>/dev/null
            echo "[INFO] Chained resubmit (chain $((CHAIN_COUNT + 1))/$MAX_RESUBMITS)."
        fi
    elif [ "$WILL_RESUBMIT" = "true" ]; then
        echo "[INFO] Campaign aborted — not resubmitting $EXP_NAME."
    elif [ "${JOB_DONE:-false}" = "true" ]; then
        echo "[INFO] Training complete — marking .done."
        touch "$NFS_RUN_DIR/.done" 2>/dev/null
    else
        echo "[INFO] No resubmission (crash or abort)."
    fi
}
trap cleanup EXIT
campaign_start

# ── Requeue cap (preempt/qos REQUEUE cluster) ────────────────────────────────
# SLURM_RESTART_COUNT increments on every requeue (preemption included). Past the
# cap we assume a node faulting on every requeue (indistinguishable from a real
# preemption by exit code) and stop THIS task from looping — WITHOUT braking the
# campaign (a legitimately-preempted low-QoS job can hit high counts). Exiting 0
# with no .done leaves the config idle, so the next submit_all.sh reconcile
# resubmits it (self-heal). USR1 time-limit resubmits spawn a fresh jobid, so this
# counts preemptions PER 23h segment, not cumulatively.
RESTART_CAP="${RESTART_CAP:-25}"
echo "[INFO] SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0} (cap ${RESTART_CAP})"
if [ "${SLURM_RESTART_COUNT:-0}" -ge "$RESTART_CAP" ]; then
    echo "[WARN] restart count ${SLURM_RESTART_COUNT} >= ${RESTART_CAP} for $EXP_NAME —"\
         " disabling further requeue and exiting (campaign NOT braked; reconcile self-heals)."
    echo "[REQUEUE-CAP] $(date '+%F %T') $EXP_NAME jobid=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT}" \
        >> "${FAIL_LOG:-$HOME/FAIL_$(date +%y%m%d).log}" 2>/dev/null
    scontrol update JobId="$SLURM_JOB_ID" Requeue=0 2>/dev/null
    exit 0
fi

RESUME_ARGS=""
ITER_NUM=""
# latest_iter.txt is ADVISORY (may be torn on a partial flush); resume whenever the
# checkpoint(s) exist and let Python derive the authoritative iteration from the
# embedded checkpoint value. Handles both single-agent (agent/) gym A/B and ASP
# (bob/ + alice/) gym C.
if [ -f "$NFS_RUN_DIR/bob/latest_checkpoint.pt" ] && [ -f "$NFS_RUN_DIR/alice/latest_checkpoint.pt" ]; then
    ITER_NUM=$(cat "$NFS_RUN_DIR/latest_iter.txt" 2>/dev/null)
    echo "[INFO] Resuming $EXP_NAME (ASP; latest_iter hint=${ITER_NUM:-<unreadable>})"
    rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/" 2>/dev/null
    RESUME_ARGS="--chkpt_alice runs/$EXP_NAME/alice/latest_checkpoint.pt --chkpt_bob runs/$EXP_NAME/bob/latest_checkpoint.pt --resume_iteration ${ITER_NUM:-0}"
elif [ -f "$NFS_RUN_DIR/agent/latest_checkpoint.pt" ]; then
    ITER_NUM=$(cat "$NFS_RUN_DIR/agent/latest_iter.txt" 2>/dev/null)
    echo "[INFO] Resuming $EXP_NAME (single-agent; latest_iter hint=${ITER_NUM:-<unreadable>})"
    rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/" 2>/dev/null
    RESUME_ARGS="--chkpt runs/$EXP_NAME/agent/latest_checkpoint.pt --resume_iteration ${ITER_NUM:-0}"
fi

echo "==================================================================="
echo "  gym-pusht training | array task $SLURM_ARRAY_TASK_ID"
echo "  Script: $SCRIPT  Exp: $EXP_NAME"
echo "  Seed: $SEED  Envs: $NUM_ENVS  push_nsteps: $PUSH_NSTEPS  MaxIters: $MAX_ITERS"
echo "  Extra: $EXTRA   Results: $NFS_RUN_DIR   Resume: ${RESUME_ARGS:-none}"
echo "==================================================================="

if [ ! -f "$SIF_IMAGE" ]; then echo "[ERROR] $SIF_IMAGE not found"; exit 1; fi

apptainer exec --pwd /workspace/isaaclab/user_project/asyncDualPlayPPO \
    --overlay "$PROJECT_ROOT/$GYM_OVERLAY":ro \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$LOCAL_RUNS":/workspace/isaaclab/user_project/asyncDualPlayPPO/runs \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p "/workspace/isaaclab/user_project/asyncDualPlayPPO/$SCRIPT" \
    --num_envs "$NUM_ENVS" --push_nsteps "$PUSH_NSTEPS" --max_iterations "$MAX_ITERS" \
    --save_interval "$SAVE_INTERVAL" --seed "$SEED" --exp_name "$EXP_NAME" $EXTRA $RESUME_ARGS &
APPTAINER_PID=$!
# Periodic durable flush: node-local $LOCAL_RUNS -> scratch $RESULTS_ROOT every
# FLUSH_SECS, so a preemption/OOM that cuts cleanup() short still leaves a recent
# checkpoint on scratch. Atomic per-file saves (ppo.save tmp->os.replace; rsync
# temp-then-rename) guarantee a mid-flush copy is never truncated.
FLUSH_SECS="${FLUSH_SECS:-600}"
( while true; do sleep "$FLUSH_SECS"; rsync -a --no-perms "$LOCAL_RUNS/" "$RESULTS_ROOT/" 2>/dev/null; done ) &
FLUSHER_PID=$!
wait $APPTAINER_PID
TRAIN_EXIT=$?
kill "$FLUSHER_PID" 2>/dev/null
echo "[INFO] Exit code: $TRAIN_EXIT"
if [ "$WILL_RESUBMIT" = "true" ]; then
    echo "[INFO] Time-limit path — resubmitting."
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    JOB_DONE=true
elif [ "$TRAIN_EXIT" -eq 137 ] || [ "$TRAIN_EXIT" -eq 143 ]; then
    # This partition is PreemptMode=REQUEUE (preempt/qos): a preemption exits 137/143
    # and SLURM AUTO-REQUEUES this same job (our resume logic continues from
    # latest_checkpoint). So do NOTHING here: self-resubmitting would create a
    # duplicate job racing on the same checkpoint (corruption), and braking would
    # kill the whole campaign on an unrelated preemption. Genuine VRAM/CUDA OOM
    # surfaces as a Python RuntimeError (non-137 exit) and is caught by the brake
    # below; a host-RAM OOM (rare, mem is right-sized) is left to SLURM's requeue.
    echo "[INFO] Exit $TRAIN_EXIT — leaving to SLURM --requeue (preemption); no self-action."
else
    echo "[ERROR] Crash (exit $TRAIN_EXIT) — tripping campaign brake."
    campaign_trip_abort "$TRAIN_EXIT"
fi
