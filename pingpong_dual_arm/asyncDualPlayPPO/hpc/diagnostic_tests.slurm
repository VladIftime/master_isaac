#!/bin/bash
# Combined suite — run all three tests sequentially in one job.
# For individual submissions use: test1_ppo_reward.slurm,
#   test2_alice_exploration.slurm, test3_asp_tug_of_war.slurm
#SBATCH --job-name=asp_diag
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1   # 96 GB VRAM, 1536 GB node RAM / 8 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm-%j-diag.out

set -o pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SIF_IMAGE="isaac-lab.sif"
PROJECT_ROOT=$(pwd)
TS="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="diag_hpc_$TS"
NFS_DIAG_DIR="$PROJECT_ROOT/runs/$EXP_ROOT"

# Local scratch for high-performance I/O during test
LOCAL_RUNS="${TMPDIR:-/tmp}/diag_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_RUNS"

echo "[INFO] Diagnostic Run: $EXP_ROOT"
echo "[INFO] Local Scratch:  $LOCAL_RUNS"
echo "[INFO] NFS Output:     $NFS_DIAG_DIR"

# ── Cleanup Hook ─────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[INFO] Syncing all diagnostic results to NFS..."
    # Ensure the parent runs directory exists
    mkdir -p "$PROJECT_ROOT/runs"
    # Sync everything from local scratch to NFS diag directory
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

    # We map $LOCAL_RUNS to /workspace/isaaclab/logs inside the container.
    # train.py uses logs/ internally or the --exp_name path relative to it.
    # However, hpc scripts bind $LOCAL_RUNS to $PROJECT_ROOT/runs.
    
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
        2>&1 | grep --line-buffered -v -e "\[Lula\] Joint .* is specified as a mimic joint" \
                       -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined"
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "  → Test ${test_num} PASSED"
    else
        echo "  → Test ${test_num} FAILED (exit $exit_code)"
    fi
}

# ── Execute Suite ────────────────────────────────────────────────────────────

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found at $PROJECT_ROOT!"
    exit 1
fi

# TEST 1 — PPO Reward Pipeline - PASSED
# run_test_hpc 1 "PPO Reward Pipeline" \
#     --num_envs 16 \
#     --max_iterations 50 \
#     --save_interval 0 \
#     --test_bob_reward

# TEST 2 — Alice Exploration Sandbox
# Uses DiagnosticAliceWrapper: alice_pos_req=0.02m, _MIN_XY_DISP=0.03m.
# Watch [AliceDisp] lines: avg XY ≈ 0.00 → Alice not touching object;
#                          avg XY > 0.02 but valid=0 → filter/frame bug.
run_test_hpc 2 "Alice Exploration Sandbox" \
    --num_envs 32 \
    --max_iterations 200 \
    --save_interval 0 \
    --diag_alice_exploration

# TEST 3 — PPO vs ABC Tug-of-War (Mini-ASP)
run_test_hpc 3 "PPO vs ABC Tug-of-War" \
    --num_envs 64 \
    --max_iterations 300 \
    --save_interval 0

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  DIAGNOSTIC SUITE COMPLETE"
echo "══════════════════════════════════════════════════════════════════════"
