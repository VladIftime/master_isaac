#!/bin/bash
# ── submit_all.sh — launch the thesis training/validation suite as job arrays ──
#
# Reads hpc/params/manifest.txt and submits each params file as a throttled
# SLURM job array using the matching template, memory tier and walltime.
#
# Usage (run from the project root, where isaac-lab.sif lives):
#   ./hpc/submit_all.sh --list                 # show phases + job counts
#   ./hpc/submit_all.sh --phase phase1          # submit one phase
#   ./hpc/submit_all.sh --phase phase1 --phase phase3
#   ./hpc/submit_all.sh --phase phase2 --throttle 2
#   ./hpc/submit_all.sh --phase phase1 --dry-run
#   ./hpc/submit_all.sh --phase phase1_validate # validation for a phase
#
# Recommended order (fairshare/age): phase1 + phase3 first, then phase2,
# then phase4 (after phase1 picks the best env count), then phase5_*.
set -euo pipefail

PROJECT_ROOT=$(pwd)
MANIFEST="$PROJECT_ROOT/hpc/params/manifest.txt"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/$USER/final_results_thesis}"
MAX_RESUBMITS=10

PHASES=()
THROTTLE_OVERRIDE=""
DRYRUN=false
LIST=false

while [ $# -gt 0 ]; do
    case "$1" in
        --phase)     PHASES+=("$2"); shift 2 ;;
        --throttle)  THROTTLE_OVERRIDE="$2"; shift 2 ;;
        --dry-run)   DRYRUN=true; shift ;;
        --list)      LIST=true; shift ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$MANIFEST" ]; then
    echo "[ERROR] manifest not found: $MANIFEST"
    echo "        run:  python3 hpc/gen_params.py"
    exit 1
fi

if [ "$LIST" = true ]; then
    printf "%-28s %-38s %-5s %-10s %-4s %s\n" PARAMS TEMPLATE MEM TIME THR PHASE
    while read -r pf tpl mem tlim thr phase; do
        [[ "$pf" =~ ^#.*$ || -z "$pf" ]] && continue
        n=$(grep -cve '^\s*$' "$PROJECT_ROOT/$pf" 2>/dev/null || echo 0)
        printf "%-28s %-38s %-5s %-10s %-4s %s (%s jobs)\n" "$(basename "$pf")" "$tpl" "$mem" "$tlim" "$thr" "$phase" "$n"
    done < "$MANIFEST"
    exit 0
fi

if [ ${#PHASES[@]} -eq 0 ]; then
    echo "[ERROR] no --phase given. Use --list to see phases."
    exit 1
fi

# ── Preflight ────────────────────────────────────────────────────────────────
[ -f "$PROJECT_ROOT/isaac-lab.sif" ]      || { echo "[ERROR] isaac-lab.sif missing in $PROJECT_ROOT"; exit 1; }
[ -f "$PROJECT_ROOT/curobo_overlay.img" ] || { echo "[ERROR] curobo_overlay.img missing in $PROJECT_ROOT"; exit 1; }
mkdir -p "$RESULTS_ROOT" || { echo "[ERROR] cannot create $RESULTS_ROOT"; exit 1; }
echo "[INFO] Results root: $RESULTS_ROOT"

want_phase() {
    local p="$1"
    for want in "${PHASES[@]}"; do [ "$want" = "$p" ] && return 0; done
    return 1
}

submitted=0
while read -r pf tpl mem tlim thr phase; do
    [[ "$pf" =~ ^#.*$ || -z "$pf" ]] && continue
    want_phase "$phase" || continue

    params_abs="$PROJECT_ROOT/$pf"
    tpl_abs="$PROJECT_ROOT/$tpl"
    [ -f "$params_abs" ] || { echo "[WARN] missing params: $params_abs"; continue; }
    [ -f "$tpl_abs" ]    || { echo "[WARN] missing template: $tpl_abs"; continue; }

    n=$(grep -cve '^\s*$' "$params_abs")
    [ "$n" -gt 0 ] || { echo "[WARN] empty params: $params_abs"; continue; }
    thr="${THROTTLE_OVERRIDE:-$thr}"

    cmd=(sbatch --array="1-${n}%${thr}" --mem="$mem" --time="$tlim"
         --export=ALL,PARAMS="$params_abs",JOB_MEM="$mem",JOB_TIME="$tlim",TEMPLATE="$tpl_abs",MAX_RESUBMITS="$MAX_RESUBMITS",RESULTS_ROOT="$RESULTS_ROOT"
         "$tpl_abs")

    echo "[SUBMIT] $phase : $(basename "$pf")  (${n} tasks, throttle ${thr}, mem ${mem}, time ${tlim})"
    if [ "$DRYRUN" = true ]; then
        printf '           %q ' "${cmd[@]}"; echo
    else
        "${cmd[@]}"
    fi
    submitted=$((submitted + 1))
done < "$MANIFEST"

echo "[INFO] ${submitted} array(s) $([ "$DRYRUN" = true ] && echo 'previewed' || echo 'submitted')."
