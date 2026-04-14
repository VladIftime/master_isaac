#!/usr/bin/env bash
# run_diagnostic_tests.sh
#
# Runs the three ASP diagnostic tests in sequence, saving full logs for each.
# All tests run headless — no display required.
#
# Usage (from master_isaac/):
#   bash asyncDualPlayPPO/run_diagnostic_tests.sh
#
# Logs:  asyncDualPlayPPO/runs/diag_<timestamp>/test{1,2,3}.log
# TensorBoard (all three runs combined):
#   tensorboard --logdir asyncDualPlayPPO/runs/diag_<timestamp>

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$HOME/env_isaaclab/bin/activate"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/runs/diag_$TS"
mkdir -p "$LOG_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
RESULTS=()

run_test() {
    local test_num="$1"
    local test_name="$2"
    local log_file="$LOG_DIR/test${test_num}.log"
    shift 2
    local cmd=("$@")

    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  TEST ${test_num}: ${test_name}"
    echo "  CMD : ${cmd[*]}"
    echo "  LOG : ${log_file}"
    echo "══════════════════════════════════════════════════════════════════════"

    local start
    start="$(date +%s)"

    # Run from project root; tee so progress is visible and also saved.
    # PIPESTATUS[0] captures Python's exit code before tee can mask it.
    cd "$PROJECT_ROOT"
    set +e
    "${cmd[@]}" 2>&1 | tee "$log_file"
    local py_exit=${PIPESTATUS[0]}
    set -e

    local status
    if [[ $py_exit -eq 0 ]]; then
        status="PASSED"
        ((PASS++)) || true
    else
        status="FAILED (exit ${py_exit})"
        ((FAIL++)) || true
    fi

    local elapsed=$(( $(date +%s) - start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    RESULTS+=("  Test ${test_num} [${test_name}]: ${status}  (${mins}m ${secs}s)")
    echo ""
    echo "  → Test ${test_num} ${status} in ${mins}m ${secs}s"
}

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [[ -f "$VENV" ]]; then
    # shellcheck source=/dev/null
    source "$VENV"
else
    echo "[WARNING] venv not found at $VENV — assuming Python is already on PATH"
fi

echo "Diagnostic run: $TS"
echo "Logs → $LOG_DIR"
echo "TensorBoard: tensorboard --logdir $LOG_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — PPO Reward Pipeline
# Verifies sparse rewards flow correctly:
#   DummyBobWrapper teleports target→goal at Bob step 50.
#   Expected: Reward/Bob > 0 and Metrics/Bob/SuccessRate > 0 from iteration 1.
# 50 iterations is enough to confirm; stop early if SR reaches 1.0.
# ─────────────────────────────────────────────────────────────────────────────
run_test 1 "PPO Reward Pipeline (--test_bob_reward)" \
    python -m asyncDualPlayPPO.train \
        --headless \
        --num_envs 16 \
        --max_iterations 50 \
        --exp_name "diag_${TS}/test1_reward_pipeline" \
        --save_interval 0 \
        --test_bob_reward

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Alice Exploration Sandbox
# Verifies Alice generates valid goals before entropy collapses.
#   Watch: Metrics/Alice/ValidGoals should climb.
#          Alice/EntropyCoef anneals 1.0 → 0.05 over 300 iters (safe here).
#          Policy/mean_noise_std in Alice's TensorBoard run.
# ─────────────────────────────────────────────────────────────────────────────
run_test 2 "Alice Exploration Sandbox" \
    python -m asyncDualPlayPPO.train \
        --headless \
        --num_envs 32 \
        --max_iterations 200 \
        --exp_name "diag_${TS}/test2_alice_exploration" \
        --save_interval 0

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — PPO vs ABC Tug-of-War (Mini-ASP)
# Verifies loss magnitudes are balanced and ABC warmup fires.
#   Watch: Loss/Bob/Surrogate vs Loss/Bob/ABC (should be within ~10×).
#          Metrics/ABC/IsWarm crosses 1.0 once Alice's reward > abc_warmup_threshold.
#          Metrics/ABC/BufferSize fills as Alice succeeds.
# ─────────────────────────────────────────────────────────────────────────────
run_test 3 "PPO vs ABC Tug-of-War (Mini-ASP)" \
    python -m asyncDualPlayPPO.train \
        --headless \
        --num_envs 64 \
        --max_iterations 300 \
        --exp_name "diag_${TS}/test3_ppo_abc_balance" \
        --save_interval 0

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  DIAGNOSTIC SUITE COMPLETE"
echo "══════════════════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do echo "$r"; done
echo ""
echo "  Passed: $PASS / $(( PASS + FAIL ))"
echo ""
echo "  Logs saved to:"
echo "    $LOG_DIR/test1.log"
echo "    $LOG_DIR/test2.log"
echo "    $LOG_DIR/test3.log"
echo ""
echo "  TensorBoard (all tests):"
echo "    tensorboard --logdir $LOG_DIR"
echo "══════════════════════════════════════════════════════════════════════"

# Exit non-zero if any test failed so CI/callers can detect failure
[[ $FAIL -eq 0 ]]
