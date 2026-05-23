#!/bin/bash
# Usage: ./run_curobo.sh <num_envs> [exp_name] [max_iterations] [nsteps] [mig_uuid]
#   ./run_curobo.sh 64                  # smoke test (defaults: 1g MIG, 5 iters)
#   ./run_curobo.sh 512 my_run 100000 300 MIG-03c7a8b7-dd9c-5bda-82fb-1cf2e9c3a34d

NUM_ENVS=${1:-64}
EXP_NAME=${2:-"curobo_${NUM_ENVS}env"}
MAX_ITERS=${3:-1000}
NSTEPS=${4:-300}
# Auto-pick MIG: >= 512 envs -> 7g.80gb, >= 256 -> 3g.40gb, else -> 1g.20gb
if [ -z "${5:-}" ]; then
    if [ "$NUM_ENVS" -ge 512 ]; then
        MIG_UUID="MIG-03c7a8b7-dd9c-5bda-82fb-1cf2e9c3a34d"   # 7g.80gb
    elif [ "$NUM_ENVS" -ge 256 ]; then
        MIG_UUID="MIG-0fb6bae2-fd70-5e49-921b-9d8c9f43e593"   # 3g.40gb (GPU1)
    else
        MIG_UUID="MIG-1c3fba4e-6023-5e36-a52b-b8c2bca9be44"   # 1g.20gb
    fi
else
    MIG_UUID="$5"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "================================================"
echo " train_curobo.py (ASP)"
echo "   Envs:      $NUM_ENVS"
echo "   Nsteps:    $NSTEPS"
echo "   MaxIters:  $MAX_ITERS"
echo "   Exp:       $EXP_NAME"
echo "   MIG:       $MIG_UUID"
echo "================================================"

cd "$SCRIPT_DIR"

CUDA_VISIBLE_DEVICES="$MIG_UUID" \
apptainer exec --nv --pwd /workspace/isaaclab/user_project/asyncDualPlayPPO \
    --overlay "$PROJECT_ROOT/curobo_overlay.img":ro \
    --bind "$PROJECT_ROOT":/workspace/isaaclab/user_project/asyncDualPlayPPO \
    --bind "$PROJECT_ROOT/.cache":/root/.cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/data":/isaac-sim/kit/data \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/cache":/isaac-sim/kit/cache \
    --bind "$PROJECT_ROOT/.isaac_cache/kit/logs":/isaac-sim/kit/logs \
    "$PROJECT_ROOT/isaac-lab.sif" \
    /workspace/isaaclab/isaaclab.sh -p train_curobo.py \
    --num_envs "$NUM_ENVS" \
    --nsteps "$NSTEPS" \
    --max_iterations "$MAX_ITERS" \
    --save_interval 10 \
    --exp_name "$EXP_NAME" \
    --headless \
    --arm_config rotated
