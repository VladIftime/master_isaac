#!/bin/bash
#SBATCH --job-name=asp_diag_t3
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM, 1536 GB node RAM / 8 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%j-diag-t3.out

# TEST 3 — PPO vs ABC Tug-of-War (Mini-ASP)
# Verifies loss magnitudes are balanced and ABC warmup fires.
#
# What to watch in TensorBoard:
#   Loss/Bob/Surrogate vs Loss/Bob/ABC  — should be within ~10×
#   Metrics/ABC/IsWarm                  — crosses 1.0 once Alice reward > abc_warmup_threshold
#   Metrics/ABC/BufferSize              — fills as Alice succeeds

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
run_test_hpc 3 "PPO vs ABC Tug-of-War" \
    --num_envs 64 \
    --max_iterations 300 \
    --save_interval 0

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  TEST 3 COMPLETE"
echo "══════════════════════════════════════════════════════════════════════"
