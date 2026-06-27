#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
    echo "Usage: $0 <folder_path> [--num-tests N] [--max-pushes N] [--max-tries N] [--timeout HOURS] [--out-dir DIR] [--gym]"
    echo ""
    echo "  folder_path   Path to a run date folder (e.g. runs/ppo_pbrs_reward/26.06.24)"
    echo "  --num-tests   Number of test scenes to run (default: 30)"
    echo "  --max-pushes  Max pushes per test (default: 30)"
    echo "  --max-tries   Max retries per test (default: 20)"
    echo "  --timeout     Timeout in hours per model (default: 2)"
    echo "  --out-dir     Output directory (default: validation_results_YYMMDD inside folder)"
    echo "  --gym         Validate using gym-pusht 2D environment instead of Isaac Lab"
    exit 1
}

NUM_TESTS=30
MAX_PUSHES=30
MAX_TRIES=20
TIMEOUT_HOURS=2
OUT_DIR=""
FOLDER_PATH=""
GYM_MODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-tests)   NUM_TESTS="$2";   shift 2 ;;
        --max-pushes)  MAX_PUSHES="$2";  shift 2 ;;
        --max-tries)   MAX_TRIES="$2";   shift 2 ;;
        --timeout)     TIMEOUT_HOURS="$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        --gym)         GYM_MODE=1;       shift ;;
        -h|--help)     usage ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [[ -z "$FOLDER_PATH" ]]; then
                FOLDER_PATH="$1"
                shift
            else
                echo "Error: unexpected argument '$1'"
                usage
            fi
            ;;
    esac
done

if [[ -z "$FOLDER_PATH" ]]; then
    echo "Error: folder path required"
    usage
fi

if [[ ! -d "$FOLDER_PATH" ]]; then
    echo "Error: $FOLDER_PATH is not a directory"
    exit 1
fi

FOLDER_PATH="$(realpath "$FOLDER_PATH")"

if [[ -z "$OUT_DIR" ]]; then
    FOLDER_BASENAME="$(basename "$FOLDER_PATH")"
    if [[ "$FOLDER_BASENAME" =~ ^([0-9]{2})\.([0-9]{2})\.([0-9]{2})$ ]]; then
        DATE_STR="${BASH_REMATCH[1]}${BASH_REMATCH[2]}${BASH_REMATCH[3]}"
    else
        DATE_STR="$(date +%y%m%d)"
    fi
    OUT_DIR="${FOLDER_PATH}/validation_results_${DATE_STR}"
fi

mkdir -p "$OUT_DIR"
echo "================================================================================"
echo "  VALIDATION FOLDER RUNNER"
echo "================================================================================"
echo "  Input folder:  $FOLDER_PATH"
echo "  Output dir:    $OUT_DIR"
echo "  Num tests:     $NUM_TESTS"
echo "  Max pushes:    $MAX_PUSHES"
echo "  Max tries:     $MAX_TRIES"
echo "  Timeout/model: ${TIMEOUT_HOURS}h"
echo "================================================================================"

echo ""
echo "Scanning for models..."

MODELS=()
MODEL_TYPES=()
CHKPT_PATHS=()

_scan_dir() {
    local scan_root="$1"
    local prefix="${2:-}"
    for subdir in "$scan_root"/*/; do
        [[ -d "$subdir" ]] || continue
        local name
        name="$(basename "$subdir")"
        local label="${prefix}${name}"

        [[ "$name" =~ ^(anal_|analysis|logs|comparison|csv|validation_results) ]] && continue

        if [[ -d "${subdir}bob" ]]; then
            local chkpt=""
            for cand in "${subdir}bob/model_best.pt" "${subdir}bob/latest_checkpoint.pt"; do
                if [[ -f "$cand" ]]; then chkpt="$cand"; break; fi
            done
            if [[ -n "$chkpt" ]]; then
                MODELS+=("$label")
                MODEL_TYPES+=("asp")
                CHKPT_PATHS+=("$chkpt")
            fi
        elif [[ -d "${subdir}agent" ]]; then
            local chkpt=""
            for cand in "${subdir}agent/model_best.pt" "${subdir}agent/latest_checkpoint.pt"; do
                if [[ -f "$cand" ]]; then chkpt="$cand"; break; fi
            done
            if [[ -n "$chkpt" ]]; then
                MODELS+=("$label")
                MODEL_TYPES+=("ppo")
                CHKPT_PATHS+=("$chkpt")
            fi
        elif compgen -G "${subdir}latest_checkpoint.zip" > /dev/null 2>&1; then
            local chkpt=""
            for cand in "${subdir}latest_checkpoint.zip"; do
                if [[ -f "$cand" ]]; then chkpt="$cand"; break; fi
            done
            if [[ -n "$chkpt" ]]; then
                MODELS+=("$label")
                MODEL_TYPES+=("sac")
                CHKPT_PATHS+=("$chkpt")
            fi
        fi
    done
}

_scan_dir "$FOLDER_PATH"
if [[ -d "${FOLDER_PATH}/runs" ]]; then
    _scan_dir "${FOLDER_PATH}/runs"
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo ""
    echo "No models with checkpoints found in $FOLDER_PATH"
    exit 1
fi

echo ""
echo "Found ${#MODELS[@]} model(s):"
for i in "${!MODELS[@]}"; do
    printf "  [%s] %-40s  %s\n" "${MODEL_TYPES[$i]}" "${MODELS[$i]}" "${CHKPT_PATHS[$i]}"
done

cd "$ROOT"
TIMEOUT_SEC=$((TIMEOUT_HOURS * 3600))

ABORT=0
trap 'echo; echo "[ABORT] Interrupted. Stopping after current model."; ABORT=1' INT TERM

CSV_FILES=()
LABELS=()
FAILED_MODELS=()

for i in "${!MODELS[@]}"; do
    [[ $ABORT -eq 1 ]] && break
    model="${MODELS[$i]}"
    type="${MODEL_TYPES[$i]}"
    chkpt="${CHKPT_PATHS[$i]}"
    csv_file="${OUT_DIR}/${model}.csv"

    echo ""
    echo "============================================"
    echo "[$((i+1))/${#MODELS[@]}] Validating: $model ($type)"
    echo "  Checkpoint: $chkpt"
    echo "  CSV output: $csv_file"
    echo "============================================"

    if [[ "$type" == "asp" ]]; then
        if [[ "$GYM_MODE" -eq 1 ]]; then
            cmd="timeout \"$TIMEOUT_SEC\" python -m asyncDualPlayPPO.tests.validate_pusht_gym \
                --chkpt-bob \"$chkpt\" \
                --num-tests \"$NUM_TESTS\" --max-pushes \"$MAX_PUSHES\" --max-tries \"$MAX_TRIES\" \
                --csv \"$csv_file\""
        else
            cmd="timeout \"$TIMEOUT_SEC\" python -m asyncDualPlayPPO.tests.validate_push_asp \
                --chkpt_bob \"$chkpt\" \
                --num_tests \"$NUM_TESTS\" --max_pushes \"$MAX_PUSHES\" --max_tries \"$MAX_TRIES\" \
                --headless --csv \"$csv_file\""
        fi
    elif [[ "$type" == "ppo" ]]; then
        if [[ "$GYM_MODE" -eq 1 ]]; then
            cmd="timeout \"$TIMEOUT_SEC\" python -m asyncDualPlayPPO.tests.validate_pusht_gym \
                --chkpt \"$chkpt\" \
                --num_tests \"$NUM_TESTS\" --max_pushes \"$MAX_PUSHES\" --max_tries \"$MAX_TRIES\" \
                --csv \"$csv_file\""
        else
            cmd="timeout \"$TIMEOUT_SEC\" python -m asyncDualPlayPPO.tests.validate_push \
                --chkpt \"$chkpt\" \
                --rel-obs --rel-act \
                --num_tests \"$NUM_TESTS\" --max_pushes \"$MAX_PUSHES\" --max_tries \"$MAX_TRIES\" \
                --headless --csv \"$csv_file\""
        fi
    elif [[ "$type" == "sac" ]]; then
        if [[ "$GYM_MODE" -eq 1 ]]; then
            cmd="timeout \"$TIMEOUT_SEC\" python -m asyncDualPlayPPO.tests.validate_pusht_gym \
                --chkpt-sac \"$chkpt\" \
                --num_tests \"$NUM_TESTS\" --max_pushes \"$MAX_PUSHES\" --max_tries \"$MAX_TRIES\" \
                --csv \"$csv_file\""
        else
            cmd="timeout \"$TIMEOUT_SEC\" python -m asyncDualPlayPPO.tests.validate_push_sac \
                --chkpt \"$chkpt\" \
                --rel-act \
                --num_tests \"$NUM_TESTS\" --max_pushes \"$MAX_PUSHES\" --max_tries \"$MAX_TRIES\" \
                --headless --csv \"$csv_file\""
        fi
    fi
    echo "  CMD: $cmd"

    set +e
    eval "$cmd"
    rc=$?
    set -e

    if [[ "$rc" -eq 0 ]] && [[ -f "$csv_file" ]]; then
        CSV_FILES+=("$csv_file")
        LABELS+=("$model")
        echo "  [OK] $model completed successfully"
    elif [[ "$rc" -eq 124 ]]; then
        echo "  [TIMEOUT] $model exceeded ${TIMEOUT_HOURS}h limit"
        FAILED_MODELS+=("$model")
    else
        echo "  [FAIL] $model failed with exit code $rc"
        FAILED_MODELS+=("$model")
    fi
done

echo ""
echo "================================================================================"
echo "  COMPLETE: ${#CSV_FILES[@]}/${#MODELS[@]} models validated"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo "  Failed: ${FAILED_MODELS[*]}"
fi
echo "================================================================================"

if [[ ${#CSV_FILES[@]} -ge 2 ]]; then
    echo ""
    echo "Generating comparison plots..."
    python asyncDualPlayPPO/tests/plot_validation.py \
        --csvs "${CSV_FILES[@]}" \
        --labels "${LABELS[@]}" \
        -o "$OUT_DIR"
    echo "[OK] Comparison plots saved to $OUT_DIR/"
elif [[ ${#CSV_FILES[@]} -eq 1 ]]; then
    echo ""
    echo "Generating single-model plots..."
    python asyncDualPlayPPO/tests/plot_validation.py \
        --single "${CSV_FILES[0]}" \
        --rot-threshold 0.2 \
        -o "$OUT_DIR"
    echo "[OK] Plots saved to $OUT_DIR/"
else
    echo ""
    echo "No CSVs to plot."
fi

echo ""
echo "Output: $OUT_DIR"
ls -la "$OUT_DIR/"

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    exit 1
fi
