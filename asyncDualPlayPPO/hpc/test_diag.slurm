#!/bin/bash
#SBATCH --job-name=test_abc_diag
#SBATCH --time=02:00:00
#SBATCH --partition=gpulong
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-%j-test_diag.out

# ── Diagnostic test suite ─────────────────────────────────────────────────────
#
# Runs four targeted checks to pinpoint why Step 1 (Pure PPO) never gets SR > 0.
# Each check isolates one layer of the stack.  Run after Step 1 fails.
#
# Test A  —  Step 1 wide threshold  (--step 1 --pos_threshold 0.20)
#   Q: Can PPO learn at ALL if the goal is easy to hit?
#   PASS: SR climbs from 0%.   → Problem was exploration (4cm too tight, need ABC).
#   FAIL: SR stays 0% even at 20cm → Deeper bug: obs encoding, reward, or RMPFlow.
#
# Test B  —  Step 4 goal distance  (--step 4)
#   Q: Does goal_distance() return ~0 when the object IS at the goal?
#   PASS: [DistCheck] lines show pos < 0.01, marked ✓
#   FAIL: pos ≈ 2 m  → double env_origins subtraction in goal_distance()
#         pos ≈ large → world/local frame mismatch in stored goal_states
#
# Test C  —  Step 5 reward pipeline  (--step 5)
#   Q: Does sparse reward fire when the object IS physically at the goal?
#   PASS: SR > 90% (DummyBobWrapper teleports target to goal, reward must fire)
#   FAIL: SR = 0%  → success threshold broken or reward function never fires
#
# Test D  —  Step 6 movement detection  (--step 6)
#   Q: Does validate_goal() correctly detect that Alice moved the object?
#   PASS: [MoveCheck] measured_dist ≈ 0.30 m, marked ✓, "Valid Goal" lines present
#   FAIL: measured_dist ≈ 0   → initial_states stale or in wrong frame
#         measured_dist ≈ 2 m → env_origins subtraction bug
#
# Interpretation guide:
#   A=PASS, B/C/D=PASS  →  Pure exploration problem; ABC should fix it.
#   A=FAIL, C=PASS       →  Reward fires but PPO can't learn; check obs/gradients.
#   C=FAIL               →  Reward pipeline broken; fix _check_bob_success first.
#   B=FAIL               →  goal_distance() has a frame bug; fix before anything else.
#   D=FAIL               →  validate_goal() broken; Alice never produces valid goals.
# ─────────────────────────────────────────────────────────────────────────────

NUM_ENVS=16    # small: these are quick diagnostic checks, not training runs
SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)

echo "[INFO] Job ID:   $SLURM_JOB_ID"
echo "[INFO] Node:     $SLURMD_NODENAME"
echo "[INFO] Started:  $(date)"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found at $(pwd)/$SIF_IMAGE"
    echo "[INFO]  Pull it with: apptainer pull isaac-lab.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0"
    exit 1
fi

RUN() {
    local step="$1"; shift
    apptainer exec --nv \
        --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
        --bind "$PROJECT_ROOT/.cache":/root/.cache \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
        "$SIF_IMAGE" \
        /workspace/isaaclab/isaaclab.sh -p \
            /workspace/isaaclab/user_project/asyncDualPlayPPO/tests/test_abc.py \
        --step "$step" \
        --num_envs "$NUM_ENVS" \
        --headless \
        "$@" \
        2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                       -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
}

# ── Test A: wide-threshold PPO (step 1 @ 20 cm) ──────────────────────────────
echo ""
echo "████  TEST A: Step 1 wide threshold (pos_threshold=0.20 m, 50 iters)  ████"
RUN 1 --num_iterations 50 --pos_threshold 0.20

# ── Test B: goal distance sanity ─────────────────────────────────────────────
echo ""
echo "████  TEST B: Step 4 goal distance check (5 iters)  ████"
RUN 4 --num_iterations 5

# ── Test C: reward pipeline ───────────────────────────────────────────────────
echo ""
echo "████  TEST C: Step 5 reward pipeline (DummyBobWrapper, 10 iters)  ████"
RUN 5 --num_iterations 10

# ── Test D: movement detection ────────────────────────────────────────────────
echo ""
echo "████  TEST D: Step 6 Alice movement detection (5 iters)  ████"
RUN 6 --num_iterations 5

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== DIAGNOSTIC CHECKLIST ==="
echo "  Test A (wide PPO):    grep 'STEP 1 PASSED'       → SR climbed with 20cm threshold"
echo "                        grep 'Max reward seen'      → must be > 0"
echo "  Test B (goal dist):   grep '\[DistCheck\]'        → all lines must show ✓"
echo "  Test C (reward pipe): grep 'STEP 5 PASSED'        → SR > 90%"
echo "  Test D (move detect): grep '\[MoveCheck\]'        → measured_dist ≈ 0.300 and ✓"
echo "                        grep 'Valid Goal'            → at least some present"
echo "============================="
echo ""
echo "[INFO] Finished: $(date)"
