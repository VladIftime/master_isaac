#!/bin/bash
#SBATCH --job-name=gym_c
#SBATCH --time=12:00:00
#SBATCH --partition=regular            # CPU partition. Adjust to your cluster.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # C is SINGLE-PROCESS (sync ASP): it does NOT scale with
                                       # cores. A few cores suffice (env loop + torch update).
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm-%j-gym_c.out
#SBATCH --signal=B:USR1@120
# ---------------------------------------------------------------------------
# gym-pusht PBRS Model C (ASP, Alice+Bob) on Habrok CPU.
#
# NOTE: GymPushASPEnv is synchronous/single-process (the Alice<->Bob cross-phase
# delayed reward forces central orchestration), so it does NOT parallelise on
# CPU cores — throughput is ~constant (~22 push/s).  Reaching ~10M pushes takes
# ~days of wall-clock, spread over many auto-resubmit chains.  (The ASP evidence
# at scale comes from the Isaac runs; this gym-C run is a controlled supplement.)
# One-time setup: build gym_overlay.img — see "gym-pusht controlled testbed" in
# implementations.md.
# ---------------------------------------------------------------------------
WILL_RESUBMIT=false
trap 'echo "[INFO] Caught SIGUSR1 (time limit); will resubmit."; WILL_RESUBMIT=true' USR1
set -o pipefail

NUM_ENVS=32                            # PPO batch = num_envs x (alice+bob pushes); fixed (sync).
ALICE_PUSHES=5
BOB_PUSHES=10
MAX_GOALS=2
MAX_ITERATIONS=30000
SAVE_INTERVAL=100
SEED=42
EXP_NAME="hpc_gym_c"

SIF_IMAGE="isaac-lab.sif"
GYM_OVERLAY="gym_overlay.img"
PROJECT_ROOT=$(pwd)
GYMPUSHT_DIR="$PROJECT_ROOT/../gym-pusht"
NFS_RUN_DIR="$PROJECT_ROOT/runs/$EXP_NAME"
LOCAL_RUNS="${TMPDIR:-/tmp}/runs_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"
CROOT=/workspace/isaaclab/user_project

cleanup() {
    echo "[INFO] Syncing $LOCAL_RUNS -> $PROJECT_ROOT/runs ..."
    rsync -a --no-perms "$LOCAL_RUNS/" "$PROJECT_ROOT/runs/" 2>/dev/null
    if [ "$WILL_RESUBMIT" = "true" ]; then
        SBATCH_OUT=$(sbatch --export=ALL,RESUME_CHAIN=1 "$PROJECT_ROOT/hpc/train_gym_c.slurm" 2>&1)
        echo "[INFO] Training incomplete — chained: $SBATCH_OUT"
    else
        echo "[INFO] Training complete or stop requested."
        touch "$PROJECT_ROOT/runs/$EXP_NAME/.done"
    fi
}
trap cleanup EXIT

RESUME_ARGS=""
if [ "$RESUME_CHAIN" = "1" ] && [ -f "$NFS_RUN_DIR/bob/latest_checkpoint.pt" ]; then
    ITER_NUM=$(cat "$NFS_RUN_DIR/latest_iter.txt" 2>/dev/null)
    if [ -n "$ITER_NUM" ]; then
        echo "[INFO] Resuming from iteration $ITER_NUM"
        rsync -a --no-perms "$NFS_RUN_DIR/" "$LOCAL_RUNS/$EXP_NAME/"
        RESUME_ARGS="--chkpt_alice runs/$EXP_NAME/alice/latest_checkpoint.pt \
                     --chkpt_bob runs/$EXP_NAME/bob/latest_checkpoint.pt \
                     --resume_iteration $ITER_NUM"
    fi
fi

echo "[INFO] Job $SLURM_JOB_ID on $SLURMD_NODENAME | envs=$NUM_ENVS alice=$ALICE_PUSHES bob=$BOB_PUSHES | model=gym-C (ASP, single-process)"
[ -f "$SIF_IMAGE" ]    || { echo "[ERROR] $SIF_IMAGE not found in $PROJECT_ROOT"; exit 1; }
[ -f "$GYM_OVERLAY" ]  || { echo "[ERROR] $GYM_OVERLAY not found — build it once (see implementations.md)"; exit 1; }
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
      "$CROOT/asyncDualPlayPPO/train_c_gym_pbrs_asp.py" \
      --num_envs "$NUM_ENVS" --alice_pushes "$ALICE_PUSHES" --bob_pushes "$BOB_PUSHES" \
      --max_goals_per_episode "$MAX_GOALS" --device cpu \
      --max_iterations "$MAX_ITERATIONS" --save_interval "$SAVE_INTERVAL" \
      --seed "$SEED" --exp_name "$EXP_NAME" \
      $RESUME_ARGS \
      2>&1 | grep --line-buffered -v -e "pkg_resources is deprecated" -e "Hello from the pygame community"
TRAIN_EXIT=$?
[ $TRAIN_EXIT -ne 0 ] && { echo "[ERROR] training command failed; will resubmit."; WILL_RESUBMIT=true; }
