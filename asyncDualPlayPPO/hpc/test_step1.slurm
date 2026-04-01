#!/bin/bash
#SBATCH --job-name=test_abc_step1
#SBATCH --time=05:30:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1       # A100 required (sm_80+ for IsaacLab container)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-test_step1.out

# ── Step 1: Pure PPO Overfit ──────────────────────────────────────────────────
# Isolates sparse reward + PPO math.
# DummyAliceWrapper: goal locked at [0.15, 0.5, 0.05] (no real Alice needed).
# abc_coef=0.0, ent_coef=0.0 (no BC, no entropy bonus).
#
# PASS criterion:  Bob success rate → 1.0 within 100–200 updates.
# FAIL signal:     SR stuck at 0.0 → RMPflow action mapping or env reset broken.
#
# Recommended: 64 envs × 200 iterations ≈ 20–40 min on A100.
# ─────────────────────────────────────────────────────────────────────────────

NUM_ENVS=64
NUM_ITER=200

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] Test step:     1 — Pure PPO overfit"
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
    --step          1 \
    --num_envs      "$NUM_ENVS" \
    --num_iterations "$NUM_ITER" \
    --headless \
    2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

# ── Pass/Fail summary ─────────────────────────────────────────────────────────
echo ""
echo "=== STEP 1 CHECKLIST ==="
echo "  grep 'STEP 1 PASSED'  → must appear"
echo "  grep 'SR'             → success rate should reach ~1.0 by iter 200"
echo "  grep 'FAILED'         → should be zero matches"
echo "========================="
