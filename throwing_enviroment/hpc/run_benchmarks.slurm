#!/bin/bash
#SBATCH --job-name=throw_bench
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j-benchmarks.out

set -o pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT=$(pwd)
SIF_IMAGE="$(dirname "$PROJECT_ROOT")/asyncDualPlayPPO/isaac-lab.sif"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOCAL_RESULTS="${TMPDIR:-/tmp}/benchmarks_${SLURM_JOB_ID}"
OUTPUT_DIR="$PROJECT_ROOT/logs/benchmarks/$TIMESTAMP"
mkdir -p "$LOCAL_RESULTS"
mkdir -p "$PROJECT_ROOT/.cache/warp"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/data"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/cache"
mkdir -p "$PROJECT_ROOT/.isaac_cache/kit/logs"

CONTAINER_PROJECT="/workspace/isaaclab/user_project/throwing_enviroment"
CONTAINER_OUTPUT="$CONTAINER_PROJECT/logs/benchmarks/$TIMESTAMP"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

cleanup() {
    echo "[INFO] Results at: $OUTPUT_DIR"
}
trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT JOB INFO
# ═══════════════════════════════════════════════════════════════════════════════

echo "==================================================================="
echo "  Isaac Lab — Performance Benchmarks"
echo "==================================================================="
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Node           : $SLURMD_NODENAME"
echo "  GPU            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Project root   : $PROJECT_ROOT"
echo "  Container      : $SIF_IMAGE"
echo "  Output dir     : $OUTPUT_DIR"
echo "==================================================================="

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════

if [ ! -f "$SIF_IMAGE" ]; then
    echo "[ERROR] Container image $SIF_IMAGE not found!"
    echo "[INFO]  Pull it with: apptainer pull isaac-lab.sif docker://nvcr.io/nvidia/isaac-lab:2.3.0"
    exit 1
fi

run_benchmark() {
    local SCRIPT_NAME="$1"
    local EXTRA_ARGS="$2"
    local TEST_NUM="$3"
    local TEST_NAME="$4"

    echo ""
    echo "==================================================================="
    echo "  [$TEST_NUM] $TEST_NAME"
    echo "==================================================================="

    apptainer exec --nv \
        --pwd "$CONTAINER_PROJECT" \
        --env PYTHONPATH="$CONTAINER_PROJECT:$CONTAINER_PROJECT/source/Throwing:$CONTAINER_PROJECT/scripts" \
        --env WARP_CACHE_PATH=/root/.cache/warp \
        --bind "$PROJECT_ROOT":"$CONTAINER_PROJECT" \
        --bind "$PROJECT_ROOT/.cache":/root/.cache \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
        --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
        "$SIF_IMAGE" \
        /workspace/isaaclab/isaaclab.sh -p "$CONTAINER_PROJECT/scripts/$SCRIPT_NAME" \
            --output_dir "$CONTAINER_OUTPUT" \
            $EXTRA_ARGS \
        2>&1 | tee "$OUTPUT_DIR/${SCRIPT_NAME%.py}.log"

    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[WARNING] $TEST_NAME exited with code $EXIT_CODE"
    else
        echo "[OK] $TEST_NAME completed."
    fi
    return $EXIT_CODE
}

# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

mkdir -p "$OUTPUT_DIR"
FAILED=0

run_benchmark "benchmark_scalability.py" \
    "--env_counts 128,1024,4096,8192 --num_steps 100 --warmup_steps 10" \
    "1/4" "Scalability & Throughput Testing"
[ $? -ne 0 ] && FAILED=$((FAILED + 1))

run_benchmark "benchmark_zero_copy.py" \
    "--num_envs 1024 --num_steps 200 --warmup_steps 10" \
    "2/4" "Zero-Copy Architecture Ablation"
[ $? -ne 0 ] && FAILED=$((FAILED + 1))

run_benchmark "benchmark_direct_vs_manager.py" \
    "--num_envs 1024 --num_steps 200 --warmup_steps 10" \
    "3/4" "DirectRLEnv vs Manager-Based Overhead"
[ $? -ne 0 ] && FAILED=$((FAILED + 1))

run_benchmark "benchmark_tiled_camera.py" \
    "--env_counts 128,512,1024,2048 --num_steps 50 --warmup_steps 10" \
    "4/4" "Tiled Rendering Performance"
[ $? -ne 0 ] && FAILED=$((FAILED + 1))

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "==================================================================="
echo "  BENCHMARK SUITE COMPLETE"
echo "==================================================================="
echo "  Total failed: $FAILED / 4"
echo "  Results dir : $OUTPUT_DIR"
echo "  CSV files   : scalability.csv, zero_copy.csv,"
echo "                direct_vs_manager.csv, tiled_camera.csv"
echo "  PNG charts  : scalability.png, zero_copy.png,"
echo "                direct_vs_manager.png, tiled_camera.png"
echo "==================================================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
