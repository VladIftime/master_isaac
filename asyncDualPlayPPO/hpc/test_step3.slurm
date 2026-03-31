#!/bin/bash
#SBATCH --job-name=test_abc_step3
#SBATCH --time=02:00:00
#SBATCH --partition=gpulong
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-test_step3.out

# ── Step 3: Full ASP Integration ──────────────────────────────────────────────
# Exercises the complete Asymmetric Self-Play loop end-to-end:
#   Alice (real, no DummyWrapper) proposes goals → validate → Bob pursues them.
# abc_coef=0.5, ent_coef=0.01, pos_threshold=0.20 (20 cm success radius, easy mode).
#
# PASS criterion:
#   - Alice valid-goal rate  > 10 %  (she learns to move objects enough)
#   - Bob success rate       >  5 %  (he can reach easy 20 cm targets)
# FAIL signal:  Either SR stays at 0.0 — full pipeline broken somewhere.
#
# 64 envs × 200 iterations ≈ 30–60 min on A100 (full sim, both agents).
# If this passes, scale up to 1024 envs + shrink pos_threshold to 0.04 m.
# ─────────────────────────────────────────────────────────────────────────────

NUM_ENVS=64
NUM_ITER=200

SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:        $SLURM_JOB_ID"
echo "[INFO] Node:          $SLURMD_NODENAME"
echo "[INFO] Test step:     3 — Full ASP integration (easy mode)"
echo "[INFO] Envs:          $NUM_ENVS"
echo "[INFO] Iterations:    $NUM_ITER"
echo "[INFO] Success radius: 20 cm (easy mode)"

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
    --step          3 \
    --num_envs      "$NUM_ENVS" \
    --num_iterations "$NUM_ITER" \
    --headless \
    2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                   -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

# ── Pass/Fail summary ─────────────────────────────────────────────────────────
echo ""
echo "=== STEP 3 CHECKLIST ==="
echo "  grep 'STEP 3 PASSED'       → must appear"
echo "  grep 'Alice valid'         → should be > 10%"
echo "  grep 'Bob SR'              → should be > 5%"
echo "  grep 'Alice Reward'        → reward signal should vary (not all zeros)"
echo "  grep 'Valid Goal'          → some Alice completions should pass"
echo "  grep 'FAILED'              → should be zero matches"
echo "========================="
echo ""
echo "Next step if PASSED:"
echo "  → Scale to 1024 envs"
echo "  → Reduce pos_threshold from 0.20 → 0.04 m"
echo "  → Reduce rot_threshold from 0.25 → 0.10 rad"
