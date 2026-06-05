#!/usr/bin/env bash
# run_diagnostics.sh — execute all 4 pre-pipeline diagnostic tests in sequence.
# Each test is a hard gate: the script stops on the first failure.
#
# Usage (from master_isaac/ root):
#   bash asyncDualPlayPPO/diagnostics/run_diagnostics.sh
#
# To run a specific test only:
#   bash asyncDualPlayPPO/diagnostics/run_diagnostics.sh --test 1
#
# Logs: asyncDualPlayPPO/diagnostics/runs/diag_<timestamp>/
# TensorBoard: tensorboard --logdir asyncDualPlayPPO/diagnostics/runs/diag_<timestamp>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_BASE="$SCRIPT_DIR/runs/diag_$TS"
mkdir -p "$LOG_BASE"

ONLY_TEST=""
if [[ "${1:-}" == "--test" && -n "${2:-}" ]]; then
    ONLY_TEST="$2"
fi

PASS=0
FAIL=0
RESULTS=()

run_step() {
    local label="$1"
    shift
    echo ""
    echo "──────────────────────────────────────────────────────────────────"
    echo "  $label"
    echo "  CMD: $*"
    echo "──────────────────────────────────────────────────────────────────"
    local start; start="$(date +%s)"
    set +e
    "$@"
    local rc=$?
    set -e
    local elapsed=$(( $(date +%s) - start ))
    if [[ $rc -eq 0 ]]; then
        RESULTS+=("  PASS  $label  (${elapsed}s)")
        ((PASS++)) || true
    else
        RESULTS+=("  FAIL  $label  (exit $rc, ${elapsed}s)")
        ((FAIL++)) || true
        echo ""
        echo "STOPPED: $label failed (exit $rc)."
        print_summary
        exit $rc
    fi
}

print_summary() {
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  DIAGNOSTIC RESULTS"
    echo "══════════════════════════════════════════════════════════════════"
    for r in "${RESULTS[@]}"; do echo "$r"; done
    echo ""
    echo "  Passed: $PASS / $(( PASS + FAIL ))"
    echo "  TensorBoard: tensorboard --logdir $LOG_BASE"
    echo "══════════════════════════════════════════════════════════════════"
}

cd "$ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Reward Pipeline Integrity via Teleportation
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY_TEST" || "$ONLY_TEST" == "1" ]]; then
    run_step "Test 1: Reward Pipeline (teleport)" \
        python -m asyncDualPlayPPO.train_curobo \
            --test_reward_pipeline \
            --num_envs 16 \
            --headless \
            --exp_name "diag_${TS}/t1_reward_pipeline" \
            2>&1 | tee "$LOG_BASE/test1.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Alice Exploration Sandbox (200 iters, random Bob)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY_TEST" || "$ONLY_TEST" == "2" ]]; then
    run_step "Test 2: Alice Sandbox (200 iters)" \
        python -m asyncDualPlayPPO.train_curobo \
            --alice_sandbox \
            --num_envs 32 \
            --max_iterations 200 \
            --headless \
            --exp_name "diag_${TS}/t2_alice_sandbox" \
            --save_interval 0 \
            2>&1 | tee "$LOG_BASE/test2_train.log"

    run_step "Test 2: Alice Sandbox analysis" \
        python -m asyncDualPlayPPO.diagnostics.test_alice_sandbox \
            --log_dir "asyncDualPlayPPO/runs/diag_${TS}/t2_alice_sandbox/summary" \
            2>&1 | tee "$LOG_BASE/test2_check.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — PPO/ABC Balance + checkpoint round-trip (50 iters, full pipeline)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY_TEST" || "$ONLY_TEST" == "3" ]]; then
    run_step "Test 3: PPO/ABC Balance run (50 iters)" \
        python -m asyncDualPlayPPO.train_curobo \
            --num_envs 32 \
            --max_iterations 50 \
            --headless \
            --exp_name "diag_${TS}/t3_ppo_abc" \
            --save_interval 50 \
            2>&1 | tee "$LOG_BASE/test3_train.log"

    run_step "Test 3a: PPO/ABC log analysis" \
        python -m asyncDualPlayPPO.diagnostics.test_ppo_abc_balance \
            --log_dir "asyncDualPlayPPO/runs/diag_${TS}/t3_ppo_abc/summary" \
            2>&1 | tee "$LOG_BASE/test3a.log"

    run_step "Test 3b: Buffer checkpoint round-trip" \
        python -m asyncDualPlayPPO.diagnostics.test_checkpoint_chain \
            --log_dir "asyncDualPlayPPO/runs/diag_${TS}/t3_ppo_abc/bob" \
            2>&1 | tee "$LOG_BASE/test3b.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Goal Encoder integration + latent space
#
# Test 4a uses the 50-iter checkpoint from Test 3 (forward pass + grad flow).
# Test 4b (t-SNE silhouette) requires ≥500 iterations to form meaningful
# clusters and is SKIPPED during CI unless LONG_RUN_CKPT is set, e.g.:
#
#   LONG_RUN_CKPT=runs/production/bob/model_500.pt \
#   LONG_RUN_LOG=runs/production/summary \
#       bash diagnostics/run_diagnostics.sh --test 4
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$ONLY_TEST" || "$ONLY_TEST" == "4" ]]; then
    CKPT="asyncDualPlayPPO/runs/diag_${TS}/t3_ppo_abc/bob/model_50.pt"
    CFG="asyncDualPlayPPO/cfg/ppo/ppo_continuous.yaml"

    run_step "Test 4a: GoalEncoder integration (forward + grad)" \
        python -m asyncDualPlayPPO.diagnostics.test_abc_goal_encoder \
            --ckpt "$CKPT" --cfg "$CFG" \
            2>&1 | tee "$LOG_BASE/test4a.log"

    # Test 4b: t-SNE latent space — only runs when a long-run checkpoint is provided.
    # In CI this always skips (sys.exit(0)) because the 50-iter model has no latent structure.
    _4B_CKPT="${LONG_RUN_CKPT:-$CKPT}"
    _4B_LOG="${LONG_RUN_LOG:-asyncDualPlayPPO/runs/diag_${TS}/t3_ppo_abc/summary}"
    run_step "Test 4b: GoalEncoder latent space (t-SNE + noise)" \
        python -m asyncDualPlayPPO.diagnostics.test_goal_encoder_latent \
            --ckpt "$_4B_CKPT" --cfg "$CFG" \
            --log_dir "$_4B_LOG" \
            --save_plot "$LOG_BASE/goal_encoder_tsne.png" \
            2>&1 | tee "$LOG_BASE/test4b.log"
fi

print_summary
[[ $FAIL -eq 0 ]]
