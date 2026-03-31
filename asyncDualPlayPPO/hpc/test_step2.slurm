#!/bin/bash
#SBATCH --job-name=test_abc_step2
#SBATCH --time=00:30:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%j-test_step2.out

# ── Step 2: Pure BC Overfit ───────────────────────────────────────────────────
# Isolates the ABC (Asymmetric Behavioral Cloning) loss.
# "God Mode" trajectory injected: constant bin=8 for X, bin=5 for all others.
# abc_coef=1.0, value_loss_coef=0.0 (gradient flows only through BC loss).
#
# PASS criterion:  BC loss → large negative; Bob's greedy actions → hardcoded bins.
# FAIL signal:     BC loss stays near 0 or gradient vanishes.
#
# Uses only 4 envs — this is a lightweight gradient-flow test (< 10 min).
# ─────────────────────────────────────────────────────────────────────────────

NUM_ENVS=4
NUM_ITER=100

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] Test step:     2 — Pure BC overfit"
echo "[INFO] Envs:          $NUM_ENVS"
echo "[INFO] Iterations:    $NUM_ITER"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found at $(pwd)/$SIF_IMAGE"
    echo "[INFO]  Pull it with: apptainer pull isaac-lab.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0"
    exit 1
fi

apptainer exec --nv \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$SIF_IMAGE" \
    /workspace/isaaclab/isaaclab.sh -p \
        /workspace/isaaclab/user_project/asyncDualPlayPPO/tests/test_abc.py \
    --step          2 \
    --num_envs      "$NUM_ENVS" \
    --num_iterations "$NUM_ITER" \
    --headless \
    2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

# ── Pass/Fail summary ─────────────────────────────────────────────────────────
echo ""
echo "=== STEP 2 CHECKLIST ==="
echo "  grep 'STEP 2 PASSED'  → must appear"
echo "  grep 'BC loss'        → should trend strongly negative (< -3.0)"
echo "  grep 'bin'            → Bob greedy actions should converge to [8,5,5,5,5,5]"
echo "  grep 'FAILED'         → should be zero matches"
echo "========================="
