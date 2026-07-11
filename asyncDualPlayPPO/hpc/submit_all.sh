#!/bin/bash
# ── submit_all.sh — launch the thesis training/validation suite as job arrays ──
#
# Reads hpc/params/manifest.txt and submits each params file as a throttled
# SLURM job array using the matching template, memory tier and walltime.
#
# Two safety/convenience features:
#   • Fail-fast brake — the first non-timeout crash of any job trips an ABORTED
#     sentinel, scancels every job in the campaign, and writes $HOME/FAIL_YYMMDD.log.
#   • Idempotent reconcile (default) — for each config it SKIPS lines whose
#     run dir has a .done marker, or a .running marker whose job-id is still in
#     squeue; only the remaining indices are submitted (explicit index list).
#     Re-running after a crash/cancel therefore resubmits just the gaps.
#
# Usage (run from the project root, where isaac-lab.sif lives):
#   ./hpc/submit_all.sh --list                 # show phases + job counts
#   ./hpc/submit_all.sh --phase phase1          # submit one phase
#   ./hpc/submit_all.sh --phase phase1 --phase phase3
#   ./hpc/submit_all.sh --phase phase2 --throttle 2
#   ./hpc/submit_all.sh --phase phase1 --dry-run
#   ./hpc/submit_all.sh --phase phase1 --no-reconcile   # force full 1-N submit
#   ./hpc/submit_all.sh --phase phase1 --campaign mycamp # join a named campaign
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
RECONCILE=true
CAMPAIGN_ID=""

while [ $# -gt 0 ]; do
    case "$1" in
        --phase)        PHASES+=("$2"); shift 2 ;;
        --throttle)     THROTTLE_OVERRIDE="$2"; shift 2 ;;
        --dry-run)      DRYRUN=true; shift ;;
        --list)         LIST=true; shift ;;
        --no-reconcile) RECONCILE=false; shift ;;
        --campaign)     CAMPAIGN_ID="$2"; shift 2 ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown arg: $1"; exit 1 ;;
    esac
done

if [ ! -f "$MANIFEST" ]; then
    echo "[ERROR] manifest not found: $MANIFEST"
    echo "        run:  python3 hpc/gen_params.py"
    exit 1
fi

if [ "$LIST" = true ]; then
    printf "%-30s %-38s %-5s %-10s %-4s %s\n" PARAMS TEMPLATE MEM TIME THR PHASE
    while read -r pf tpl mem tlim thr phase; do
        [[ "$pf" =~ ^#.*$ || -z "$pf" ]] && continue
        n=$(grep -cve '^[[:space:]]*$' "$PROJECT_ROOT/$pf" 2>/dev/null || echo 0)
        printf "%-30s %-38s %-5s %-10s %-4s %s (%s jobs)\n" "$(basename "$pf")" "$tpl" "$mem" "$tlim" "$thr" "$phase" "$n"
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
mkdir -p "$PROJECT_ROOT/runs" 2>/dev/null   # nested bind mount-point for validate.slurm
touch "$HOME/.write_test_$$" 2>/dev/null && rm -f "$HOME/.write_test_$$" || { echo "[ERROR] \$HOME not writable (FAIL log target)"; exit 1; }

# ── Campaign setup (fail-fast brake domain) ──────────────────────────────────
[ -n "$CAMPAIGN_ID" ] || CAMPAIGN_ID="$(date +%y%m%d_%H%M%S)_$$"
CAMPAIGN_DIR="$RESULTS_ROOT/.campaign/$CAMPAIGN_ID"
FAIL_LOG="$HOME/FAIL_$(date +%y%m%d).log"
mkdir -p "$CAMPAIGN_DIR/jobids"
# A reused campaign id must not carry a stale ABORTED sentinel.
rm -f "$CAMPAIGN_DIR/ABORTED" 2>/dev/null; rmdir "$CAMPAIGN_DIR/aborting" 2>/dev/null || true
echo "[INFO] Results root : $RESULTS_ROOT"
echo "[INFO] Campaign     : $CAMPAIGN_ID  ($CAMPAIGN_DIR)"
echo "[INFO] FAIL log     : $FAIL_LOG"
echo "[INFO] Reconcile    : $RECONCILE"

want_phase() {
    local p="$1"
    for want in "${PHASES[@]}"; do [ "$want" = "$p" ] && return 0; done
    return 1
}

# exp_from_line — EXP_NAME is the token immediately before the first --flag,
# or the last token if there are none. Works for single/self/gym/validate lines.
exp_from_line() {
    local prev="" w
    for w in $1; do
        case "$w" in --*) echo "$prev"; return ;; esac
        prev="$w"
    done
    echo "$prev"
}

# config_active — true if this config is done, or running with a live job-id.
config_state() {   # echoes: done | running | idle
    local exp="$1" rundir="$RESULTS_ROOT/$exp"
    if [ -f "$rundir/.done" ]; then echo done; return; fi
    if [ -f "$rundir/.running" ]; then
        local jid; jid=$(cat "$rundir/.running" 2>/dev/null)
        if [ -n "$jid" ] && squeue -h -j "$jid" -o "%i" 2>/dev/null | grep -q .; then
            echo running; return
        fi
    fi
    echo idle
}

submitted=0
while read -r pf tpl mem tlim thr phase; do
    [[ "$pf" =~ ^#.*$ || -z "$pf" ]] && continue
    want_phase "$phase" || continue

    params_abs="$PROJECT_ROOT/$pf"
    tpl_abs="$PROJECT_ROOT/$tpl"
    [ -f "$params_abs" ] || { echo "[WARN] missing params: $params_abs"; continue; }
    [ -f "$tpl_abs" ]    || { echo "[WARN] missing template: $tpl_abs"; continue; }

    n=$(grep -cve '^[[:space:]]*$' "$params_abs")
    [ "$n" -gt 0 ] || { echo "[WARN] empty params: $params_abs"; continue; }
    thr="${THROTTLE_OVERRIDE:-$thr}"

    # ── Reconcile: build the index list to submit ────────────────────────────
    is_validate=false; [[ "$tpl" == *validate.slurm ]] && is_validate=true
    if [ "$RECONCILE" = true ] && [ "$is_validate" = false ]; then
        indices=""; skip_done=0; skip_run=0
        idx=0
        while IFS= read -r line; do
            [[ -z "${line// }" ]] && continue
            idx=$((idx + 1))
            exp=$(exp_from_line "$line")
            case "$(config_state "$exp")" in
                done)    skip_done=$((skip_done + 1)) ;;
                running) skip_run=$((skip_run + 1)) ;;
                *)       indices="${indices:+$indices,}$idx" ;;
            esac
        done < "$params_abs"
        if [ -z "$indices" ]; then
            echo "[SKIP]   $phase : $(basename "$pf")  (all $n done/running: ${skip_done} done, ${skip_run} running)"
            continue
        fi
        array_spec="${indices}%${thr}"
        note="reconcile: submit [$indices]; skip ${skip_done} done, ${skip_run} running"
    else
        array_spec="1-${n}%${thr}"
        note="full 1-${n}"
    fi

    cmd=(sbatch --array="$array_spec" --mem="$mem" --time="$tlim"
         --export=ALL,PARAMS="$params_abs",JOB_MEM="$mem",JOB_TIME="$tlim",TEMPLATE="$tpl_abs",MAX_RESUBMITS="$MAX_RESUBMITS",RESULTS_ROOT="$RESULTS_ROOT",CAMPAIGN_DIR="$CAMPAIGN_DIR",FAIL_LOG="$FAIL_LOG",ABORT_ON_FAIL=1
         "$tpl_abs")

    echo "[SUBMIT] $phase : $(basename "$pf")  (array=$array_spec, mem=$mem, time=$tlim) — $note"
    if [ "$DRYRUN" = true ]; then
        printf '           %q ' "${cmd[@]}"; echo
    else
        SB_OUT=$("${cmd[@]}" 2>&1); echo "           $SB_OUT"
        JID=$(echo "$SB_OUT" | awk '/Submitted batch job/{print $4}')
        [ -n "$JID" ] && touch "$CAMPAIGN_DIR/jobids/$JID" 2>/dev/null
    fi
    submitted=$((submitted + 1))
done < "$MANIFEST"

echo "[INFO] ${submitted} array(s) $([ "$DRYRUN" = true ] && echo 'previewed' || echo 'submitted')."
[ "$DRYRUN" = false ] && echo "[INFO] Manual abort: scancel \$(ls $CAMPAIGN_DIR/jobids) ; touch $CAMPAIGN_DIR/ABORTED"
