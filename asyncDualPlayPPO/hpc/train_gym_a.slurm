#!/bin/bash
#SBATCH --job-name=gym_a
#SBATCH --time=12:00:00
#SBATCH --partition=regular            # CPU partition (gym is CPU-bound). Adjust to your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32             # gym is CPU-bound: num_envs = this. Edit to node capacity.
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm-%j-gym_a.out
#SBATCH --signal=B:USR1@120
# ---------------------------------------------------------------------------
# gym-pusht PBRS Model A (single-agent, no curriculum) on Habrok CPU node.
# Uses isaac-lab.sif (container Python has torch) + gym_overlay.img (gym deps).
# gym-pusht (modified arena_width source) is bound + editable-installed in the
# overlay.  No GPU / no --nv: PPO runs on CPU, env workers on CPU (AsyncVectorEnv).
# One-time setup: see hpc/README_gym.md
# ---------------------------------------------------------------------------
WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (time limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

PUSH_NSTEPS=30
MAX_ITERATIONS=12000
SAVE_INTERVAL=200
SEED=42
EXP_NAME="hpc_gym_a"
NUM_ENVS=${SLURM_CPUS_PER_TASK:-32}     # one AsyncVectorEnv worker per core

SIF_IMAGE="isaac-lab.sif"
GYM_OVERLAY="gym_overlay.img"
PROJECT_ROOT=$(pwd)                                   # .../asyncDualPlayPPO  (run sbatch from here)
GYMPUSHT_DIR="$PROJECT_ROOT/../gym-pusht"             # sibling repo (modified arena_width)
NFS_RUN_DIR="$PROJECT_ROOT/runs/$EXP_NAME"
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"
CROOT=/workspace/isaaclab/user_project

cleanup() {
    echo "[INFO] Syncing $LOCAL_RUNS -> $PROJECT_ROOT/runs ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/" 2>/dev/null
    if [ "$WILL_RESUBMIT" = "true" ]; then
        SBATCH_OUT=$(sbatch --export=ALL,RESUME_CHAIN=1 "$PROJECT_ROOT/hpc/train_gym_a.slurm" 2>&1)
        echo "[INFO] Training incomplete — chained: $SBATCH_OUT"
    else
        echo "[INFO] Training complete or stop requested."
        touch "$PROJECT_ROOT/runs/$EXP_NAME/.done"
    fi
}
trap cleanup EXIT

RESUME_ARGS=""
if [ "$RESUME_CHAIN" = "1" ] && [ -f "$NFS_RUN_DIR/agent/latest_checkpoint.pt" ]; then
    ITER_NUM=$(cat "$NFS_RUN_DIR/agent/latest_iter.txt" 2>/dev/null)
    if [ -n "$ITER_NUM" ]; then
        echo "[INFO] Resuming from iteration $ITER_NUM"
        rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/"
        RESUME_ARGS="--chkpt runs/$EXP_NAME/agent/latest_checkpoint.pt --resume_iteration $ITER_NUM"
    fi
fi

echo "[INFO] Job $SLURM_JOB_ID on $SLURMD_NODENAME | envs=$NUM_ENVS push_nsteps=$PUSH_NSTEPS | model=gym-A"
[ -f "$SIF_IMAGE" ]    || { echo "[ERROR] $SIF_IMAGE not found in $PROJECT_ROOT"; exit 1; }
[ -f "$GYM_OVERLAY" ]  || { echo "[ERROR] $GYM_OVERLAY not found — build it once (hpc/README_gym.md)"; exit 1; }
[ -d "$GYMPUSHT_DIR" ] || { echo "[ERROR] gym-pusht not found at $GYMPUSHT_DIR"; exit 1; }

apptainer exec \
    --overlay "$PROJECT_ROOT/$GYM_OVERLAY":ro \
    --bind "$PROJECT_ROOT":"$CROOT/asyncDualPlayPPO" \
    --bind "$GYMPUSHT_DIR":"$CROOT/gym-pusht" \
    --bind "$LOCAL_RUNS":"$CROOT/asyncDualPlayPPO/runs" \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --pwd "$CROOT/asyncDualPlayPPO" \
    --env PYTHONPATH="$CROOT:$CROOT/gym-pusht" \
    --env SDL_VIDEODRIVER=dummy \
    --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1 \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p \
      "$CROOT/asyncDualPlayPPO/train_a_gym_pbrs_simple.py" \
      --num_envs "$NUM_ENVS" --push_nsteps "$PUSH_NSTEPS" --device cpu \
      --max_iterations "$MAX_ITERATIONS" --save_interval "$SAVE_INTERVAL" \
      --seed "$SEED" --exp_name "$EXP_NAME" \
      $RESUME_ARGS \
      2>&1 | grep --line-buffered -v -e "pkg_resources is deprecated" -e "Hello from the pygame community"
TRAIN_EXIT=$?
[ $TRAIN_EXIT -ne 0 ] && { echo "[ERROR] training command failed; will resubmit."; WILL_RESUBMIT=true; }
