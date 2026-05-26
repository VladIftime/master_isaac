#!/bin/bash
# Usage: ./run_push.sh <num_envs> [exp_name] [max_iterations] [mig_uuid]
# Logs:    ~/master/master_isaac/logs/<exp_name>_<timestamp>.log
# TensorBoard: ~/master/master_isaac/runs/<exp_name>/summary
# Survives SSH drops via screen.

NUM_ENVS=${1:-64}
EXP_NAME=${2:-"push_${NUM_ENVS}env"}
MAX_ITERS=${3:-1000}

if [ -z "${4:-}" ]; then
    if [ "$NUM_ENVS" -ge 512 ]; then
        MIG_UUID="MIG-03c7a8b7-dd9c-5bda-82fb-1cf2e9c3a34d"
    elif [ "$NUM_ENVS" -ge 256 ]; then
        MIG_UUID="MIG-0fb6bae2-fd70-5e49-921b-9d8c9f43e593"
    else
        MIG_UUID="MIG-1c3fba4e-6023-5e36-a52b-b8c2bca9be44"
    fi
else
    MIG_UUID="$4"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
RUNS_DIR="$PROJECT_ROOT/runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXP_NAME}_${TIMESTAMP}.log"

# MUST create dirs before screen auto-wrap or apptainer mount check
mkdir -p "$LOG_DIR" "$RUNS_DIR" \
    "$PROJECT_ROOT/.cache" \
    "$PROJECT_ROOT/.isaac_cache/kit/data" \
    "$PROJECT_ROOT/.isaac_cache/kit/cache" \
    "$PROJECT_ROOT/.isaac_cache/kit/logs"

echo "================================================"
echo " train_push.py (Push-PPO Baseline)"
echo "   Envs:      $NUM_ENVS"
echo "   MaxIters:  $MAX_ITERS"
echo "   Exp:       $EXP_NAME"
echo "   MIG:       $MIG_UUID"
echo "   Log:       $LOG_FILE"
echo "   TB:        $RUNS_DIR/$EXP_NAME/summary"
echo "================================================"

# Auto-wrap in screen if not already inside one
if [ -z "$STY" ] && [ -z "$TMUX" ]; then
    echo "[INFO] Launching inside screen to survive SSH drops..."
    SCREEN_NAME=$(echo "$EXP_NAME" | tr '/' '_')
    screen -dmS "$SCREEN_NAME" bash -c "cd '$SCRIPT_DIR' && bash '$0' '$NUM_ENVS' '$EXP_NAME' '$MAX_ITERS' '$MIG_UUID'"
    echo "[INFO] Reattach:  screen -r $SCREEN_NAME"
    echo "[INFO] Tail log:  tail -f $LOG_FILE"
    exit 0
fi

cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES="$MIG_UUID" \
apptainer exec --nv --pwd /workspace/isaaclab/user_project/asyncDualPlayPPO \
    --overlay "$PROJECT_ROOT/curobo_overlay.img":ro \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$PROJECT_ROOT/isaac-lab.sif" \
    /workspace/isaaclab/isaaclab.sh -p train_push.py \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --save_interval 10 \
    --exp_name "$EXP_NAME" \
    --headless \
    2>&1 | grep --line-buffered -v \
        -e "\[Lula\] Joint .* is specified as a mimic joint" \
        -e "Warning: link 'robotiq_coupler' material 'flat_black' undefined" \
        -e "rootless{.*} ignoring" \
    | tee "$LOG_FILE"
