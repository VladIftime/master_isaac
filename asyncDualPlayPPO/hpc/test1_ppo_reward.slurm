#!/bin/bash
#SBATCH --job-name=asp_diag_t1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM, 1536 GB node RAM / 8 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%j-diag-t1.out

# TEST 1 — PPO Reward Pipeline
# Verifies Bob's sparse reward fires when the object reaches goal.
#   Watch: Bob Rew > 0 and SR > 0 from iteration 1.
#          DummyBobWrapper teleports target → goal at Bob step 50.

set -o pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)
TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="diag_hpc_$TS"
NFS_DIAG_DIR="$PROJECT_ROOT/runs/$EXP_ROOT"

LOCAL_RUNS="${TMPDIR:-/tmp}/diag_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

echo "[INFO] Diagnostic Run: $EXP_ROOT"
echo "[INFO] Local Scratch:  $LOCAL_RUNS"
echo "[INFO] NFS Output:     $NFS_DIAG_DIR"

# ── Cleanup Hook ─────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[INFO] Syncing all diagnostic results to NFS..."
    mkdir -p "$PROJECT_ROOT/runs"
    rsync -a --no-perms "$LOCAL_RUNS/" "$NFS_DIAG_DIR/" 2>/dev/null
    echo "[INFO] Sync complete. Results available in: $NFS_DIAG_DIR"
}
trap cleanup EXIT

# ── Helper: Run HPC Test ─────────────────────────────────────────────────────
run_test_hpc() {
    local test_num="$1"
    local test_name="$2"
    local test_exp="test${test_num}_$(echo "$test_name" | tr ' ' '_' | tr -d '()')"
    shift 2
    local args=("$@")

    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  TEST ${test_num}: ${test_name}"
    echo "  ARGS: ${args[*]}"
    echo "══════════════════════════════════════════════════════════════════════"

    apptainer exec --nv \
        --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
        --bind "$LOCAL_RUNS":"$PROJECT_ROOT/runs" \
        --bind "$LOCAL_RUNS":/workspace/isaaclab/logs \
        --bind "$PROJECT_ROOT/.cache":/root/.cache \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
        "$SIF_IMAGE" \
        /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/user_project/asyncDualPlayPPO/train.py \
        --headless \
        --exp_name "$EXP_ROOT/$test_exp" \
        "${args[@]}" \
        2>&1 | grep -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                       -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"

    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "  → Test ${test_num} PASSED"
    else
        echo "  → Test ${test_num} FAILED (exit $exit_code)"
    fi
}

# ── Preflight ────────────────────────────────────────────────────────────────
if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found at $PROJECT_ROOT!"
    exit 1
fi

# ── Execute ──────────────────────────────────────────────────────────────────
run_test_hpc 1 "PPO Reward Pipeline" \
    --num_envs 16 \
    --max_iterations 50 \
    --save_interval 0 \
    --test_bob_reward

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  TEST 1 COMPLETE"
echo "══════════════════════════════════════════════════════════════════════"
