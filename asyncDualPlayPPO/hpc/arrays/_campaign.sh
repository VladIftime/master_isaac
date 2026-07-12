#!/bin/bash
# ── _campaign.sh — shared campaign coordination for array training templates ──
#
# Provides a fail-fast brake (first non-timeout crash cancels the whole launch
# and writes $HOME/FAIL_YYMMDD.log) and per-config reconcile markers
# (.running / .done) consumed by submit_all.sh.
#
# Requires these to be set before sourcing/calling:
#   PROJECT_ROOT NFS_RUN_DIR EXP_NAME SCRIPT PARAMS
#   (campaign vars exported by submit_all.sh: CAMPAIGN_DIR FAIL_LOG ABORT_ON_FAIL)
# All functions no-op safely when CAMPAIGN_DIR is unset (manual single sbatch).

campaign_start() {
    JOB_START=$(date +%s)
    if [ -n "${CAMPAIGN_DIR:-}" ]; then
        mkdir -p "$CAMPAIGN_DIR/jobids" 2>/dev/null
        mkdir -p "/scratch/$USER/slurm_logs" 2>/dev/null
        # Self-abort: a queued task that starts after the brake tripped exits now
        # (backstop in case scancel had not yet reached it).
        if [ -f "$CAMPAIGN_DIR/ABORTED" ]; then
            echo "[ABORT] Campaign already aborted — skipping $EXP_NAME (task $SLURM_ARRAY_TASK_ID)."
            exit 0
        fi
        touch "$CAMPAIGN_DIR/jobids/$SLURM_JOB_ID" 2>/dev/null
    fi
    # Per-config running marker (stores jobid for stale detection by the launcher).
    mkdir -p "$NFS_RUN_DIR" 2>/dev/null
    echo "$SLURM_JOB_ID" > "$NFS_RUN_DIR/.running" 2>/dev/null
}

campaign_clear_running() {
    [ -n "${NFS_RUN_DIR:-}" ] && rm -f "$NFS_RUN_DIR/.running" 2>/dev/null
    return 0
}

campaign_aborted() {
    [ -n "${CAMPAIGN_DIR:-}" ] && [ -f "$CAMPAIGN_DIR/ABORTED" ]
}

# campaign_trip_abort <exit_code> — first failer cancels every campaign job and
# writes the FAIL log; later failers only append a follow-up line.
campaign_trip_abort() {
    local exit_code="$1"
    [ "${ABORT_ON_FAIL:-1}" = "1" ] || return 0
    [ -n "${CAMPAIGN_DIR:-}" ] || return 0
    local elapsed=$(( $(date +%s) - ${JOB_START:-$(date +%s)} ))
    local flog="${FAIL_LOG:-$HOME/FAIL_$(date +%y%m%d).log}"
    if mkdir "$CAMPAIGN_DIR/aborting" 2>/dev/null; then
        touch "$CAMPAIGN_DIR/ABORTED"
        local out_file
        out_file=$(ls -1 "$PROJECT_ROOT"/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-*.out 2>/dev/null | head -1)
        {
            echo "==================================================================="
            echo "[FAIL] $(date '+%Y-%m-%d %H:%M:%S')  campaign=$(basename "$CAMPAIGN_DIR")"
            echo "  exp_name  : $EXP_NAME"
            echo "  script    : $SCRIPT"
            echo "  params    : $PARAMS  (line $SLURM_ARRAY_TASK_ID)"
            echo "  job       : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  (jobid $SLURM_JOB_ID)"
            echo "  node      : ${SLURMD_NODENAME:-?}   exit=$exit_code   elapsed=${elapsed}s"
            echo "  --- tail of ${out_file:-<no slurm out found>} ---"
            [ -n "$out_file" ] && tail -n 40 "$out_file" 2>/dev/null
            echo "==================================================================="
        } >> "$flog" 2>/dev/null
        local ids
        ids=$(ls -1 "$CAMPAIGN_DIR/jobids" 2>/dev/null | grep -v "^${SLURM_JOB_ID}$")
        if [ -n "$ids" ]; then
            echo "[ABORT] Cancelling sibling campaign jobs: $(echo $ids | tr '\n' ' ')"
            scancel $ids 2>/dev/null
        fi
        echo "[ABORT] Campaign tripped by $EXP_NAME (exit $exit_code). See $flog"
    else
        echo "[FAIL-followup] $(date '+%F %T') $EXP_NAME exit=$exit_code job=$SLURM_JOB_ID" >> "$flog" 2>/dev/null
    fi
}
