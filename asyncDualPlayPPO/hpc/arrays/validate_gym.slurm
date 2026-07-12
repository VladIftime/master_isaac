#!/bin/bash
# ── Array template: gym-pusht validation (validate_pusht_gym.py) ─────────────
# Params line format:  VALIDATOR EXP_NAME [EXTRA_FLAGS...]
#   (VALIDATOR is always validate_pusht_gym.py; kept for launcher uniformity.)
# ASP vs single-agent is auto-detected from the run dir (bob/ vs agent/).
# Prefers model_best.pt; with ALLOW_LATEST=1 falls back to latest_checkpoint.pt
# (results in validation_latest/). Launcher exports PARAMS / RESULTS_ROOT / ALLOW_LATEST.
#SBATCH --job-name=gymvalidate
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%A_%a-gymvalidate.out

set -o pipefail
PROJECT_ROOT=$(pwd)
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/$USER/final_results_thesis}"
CONTAINER_PROJECT="/workspace/isaaclab/user_project/asyncDualPlayPPO"
ALLOW_LATEST="${ALLOW_LATEST:-0}"
NUM_TESTS=30
MAX_PUSHES=30
MAX_TRIES=20

if [ -z "$PARAMS" ] || [ ! -f "$PARAMS" ]; then echo "[ERROR] PARAMS missing: '$PARAMS'"; exit 1; fi
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAMS")
if [ -z "$LINE" ]; then echo "[ERROR] No line $SLURM_ARRAY_TASK_ID in $PARAMS"; exit 1; fi
read -r VALIDATOR EXP_NAME EXTRA <<< "$LINE"

SIF_IMAGE="isaac-lab.sif"
CURBO_OVERLAY="curobo_overlay.img"
NFS_RUN_DIR="$RESULTS_ROOT/$EXP_NAME"
[ -d "$NFS_RUN_DIR" ] || { echo "[ERROR] run dir not found: $NFS_RUN_DIR"; exit 1; }

# Auto-detect ASP (bob/) vs single-agent (agent/).
if [ -d "$NFS_RUN_DIR/bob" ]; then SUB="bob"; FLAG="--chkpt-bob"; else SUB="agent"; FLAG="--chkpt"; fi
BEST_HOST="$NFS_RUN_DIR/$SUB/model_best.pt"
LATEST_HOST="$NFS_RUN_DIR/$SUB/latest_checkpoint.pt"
if [ -f "$BEST_HOST" ]; then
    CKPT_REL="runs/$EXP_NAME/$SUB/model_best.pt"; OUT_HOST="$NFS_RUN_DIR"
    OUT_CONT="$CONTAINER_PROJECT/runs/$EXP_NAME"; MODE="best"
elif [ "$ALLOW_LATEST" = "1" ] && [ -f "$LATEST_HOST" ]; then
    CKPT_REL="runs/$EXP_NAME/$SUB/latest_checkpoint.pt"; OUT_HOST="$NFS_RUN_DIR/validation_latest"
    OUT_CONT="$CONTAINER_PROJECT/runs/$EXP_NAME/validation_latest"; MODE="latest"
    echo "[WARN] model_best.pt absent — validating latest_checkpoint.pt (validation_latest/)."
else
    echo "[ERROR] no model_best.pt$([ "$ALLOW_LATEST" = "1" ] && echo ' or latest_checkpoint.pt') for $EXP_NAME"; exit 1
fi
mkdir -p "$OUT_HOST" "$PROJECT_ROOT/runs" 2>/dev/null
CSV_OUT="$OUT_CONT/validation_results_gym.csv"

echo "==================================================================="
echo "  gym validation | task $SLURM_ARRAY_TASK_ID | $EXP_NAME | mode=$MODE ($SUB)"
echo "  CSV -> $OUT_HOST/validation_results_gym.csv"
echo "==================================================================="
[ -f "$SIF_IMAGE" ] || { echo "[ERROR] $SIF_IMAGE not found"; exit 1; }

apptainer exec --nv --pwd "$CONTAINER_PROJECT" \
    --overlay "$PROJECT_ROOT/$CURBO_OVERLAY":ro \
    --bind "$PROJECT_ROOT":"$CONTAINER_PROJECT" \
    --bind "$RESULTS_ROOT":"$CONTAINER_PROJECT/runs" \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p "$CONTAINER_PROJECT/tests/validate_pusht_gym.py" \
    $FLAG "$CKPT_REL" --num-tests "$NUM_TESTS" --max-pushes "$MAX_PUSHES" --max-tries "$MAX_TRIES" \
    --csv "$CSV_OUT" $EXTRA
echo "[INFO] gym validation exit: $?"
