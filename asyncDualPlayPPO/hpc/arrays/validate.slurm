#!/bin/bash
# ── Array template: validation (run manually when checkpoints are ready) ──────
# Params line format (whitespace-separated):
#   VALIDATOR EXP_NAME [EXTRA_FLAGS...]
#   VALIDATOR = validate_push.py (A/B/baseline)  or  validate_push_asp.py (E/G)
# EXTRA for validate_push.py     : --rel-obs --rel-act
# EXTRA for validate_push_asp.py : --dpose-obs --char-length 0.07 --dpose-threshold 0.055
# Protocol is fixed to SUMMARY.md: 30 scenes, best-of-20, max 30 pushes.
# Launcher sets --array / --mem / --time and exports: PARAMS
#SBATCH --job-name=validate
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%A_%a-validate.out

set -o pipefail
PROJECT_ROOT=$(pwd)
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/$USER/final_results_thesis}"
CONTAINER_PROJECT="/workspace/isaaclab/user_project/asyncDualPlayPPO"
NUM_TESTS=30
MAX_PUSHES=30
MAX_TRIES=20

if [ -z "$PARAMS" ] || [ ! -f "$PARAMS" ]; then
    echo "[ERROR] PARAMS file not set or missing: '$PARAMS'"; exit 1
fi
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS")
if [ -z "$LINE" ]; then echo "[ERROR] No line $SLURM_ARRAY_TASK_ID in $PARAMS"; exit 1; fi
read -r VALIDATOR EXP_NAME EXTRA <<< "$LINE"

SIF_IMAGE="isaac-lab.sif"
CURBO_OVERLAY="curobo_overlay.img"
NFS_RUN_DIR="$RESULTS_ROOT/$EXP_NAME"

if [ ! -d "$NFS_RUN_DIR" ]; then echo "[ERROR] run dir not found: $NFS_RUN_DIR"; exit 1; fi

# Select checkpoint + CSV tag by validator family.  Prefer model_best.pt; if
# absent and ALLOW_LATEST=1, fall back to latest_checkpoint.pt and write results
# into a separate validation_latest/ subfolder (distinct filenames, and excluded
# from collate_seeds which globs only one level).  Validated on the host first so
# a missing checkpoint fails fast before the slow container boot.
ALLOW_LATEST="${ALLOW_LATEST:-0}"
case "$VALIDATOR" in
    *asp*)  SUB="bob";   FLAG="--chkpt_bob"; TAG="asp" ;;
    *)      SUB="agent"; FLAG="--chkpt";     TAG="push" ;;
esac
BEST_HOST="$NFS_RUN_DIR/$SUB/model_best.pt"
LATEST_HOST="$NFS_RUN_DIR/$SUB/latest_checkpoint.pt"
if [ -f "$BEST_HOST" ]; then
    CKPT_REL="runs/$EXP_NAME/$SUB/model_best.pt"
    OUT_HOST="$NFS_RUN_DIR"
    OUT_CONT="$CONTAINER_PROJECT/runs/$EXP_NAME"
    MODE="best"
elif [ "$ALLOW_LATEST" = "1" ] && [ -f "$LATEST_HOST" ]; then
    CKPT_REL="runs/$EXP_NAME/$SUB/latest_checkpoint.pt"
    OUT_HOST="$NFS_RUN_DIR/validation_latest"
    OUT_CONT="$CONTAINER_PROJECT/runs/$EXP_NAME/validation_latest"
    MODE="latest"
    echo "[WARN] model_best.pt absent — validating latest_checkpoint.pt (results in validation_latest/)."
else
    echo "[ERROR] no model_best.pt$([ "$ALLOW_LATEST" = "1" ] && echo ' or latest_checkpoint.pt') for $EXP_NAME"
    exit 1
fi
mkdir -p "$OUT_HOST" 2>/dev/null
CKPT_ARGS="$FLAG $CKPT_REL"
# Disc-scene evaluations get an _sdisc suffix so a T-block model's cross-eval on
# disc scenes never overwrites its own T-block validation CSV in the same run dir.
SCENE_SUFFIX=""
case " $EXTRA " in *" --scene-set disc "*) SCENE_SUFFIX="_sdisc" ;; esac
CSV_OUT="$OUT_CONT/validation_results_${TAG}${SCENE_SUFFIX}.csv"

# Ensure the nested bind mount-point exists on the host (runs/ under the
# project bind); RESULTS_ROOT is mounted over it inside the container.
mkdir -p "$PROJECT_ROOT/runs"

echo "==================================================================="
echo "  Validation | task $SLURM_ARRAY_TASK_ID | $VALIDATOR"
echo "  Exp: $EXP_NAME   Tag: $TAG   Checkpoint mode: $MODE"
echo "  Protocol: num_tests=$NUM_TESTS max_pushes=$MAX_PUSHES max_tries=$MAX_TRIES"
echo "  CSV -> $OUT_HOST/validation_results_${TAG}${SCENE_SUFFIX}.csv"
echo "  Extra: $EXTRA"
echo "==================================================================="

if [ ! -f "$SIF_IMAGE" ]; then echo "[ERROR] $SIF_IMAGE not found"; exit 1; fi

# Bind RESULTS_ROOT directly as runs/ so checkpoints resolve and the CSV lands
# straight back on /scratch (validation is short — no local-scratch staging).
apptainer exec --nv --pwd "$CONTAINER_PROJECT" \
    --overlay "$PROJECT_ROOT/$CURBO_OVERLAY":ro \
    --bind "$PROJECT_ROOT":"$CONTAINER_PROJECT" \
    --bind "$RESULTS_ROOT":"$CONTAINER_PROJECT/runs" \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p "$CONTAINER_PROJECT/tests/$VALIDATOR" \
    $CKPT_ARGS --num_tests "$NUM_TESTS" --max_pushes "$MAX_PUSHES" --max_tries "$MAX_TRIES" \
    --csv "$CSV_OUT" --headless $EXTRA \
    2>&1 | grep --line-buffered -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
echo "[INFO] Validation exit: $?"
