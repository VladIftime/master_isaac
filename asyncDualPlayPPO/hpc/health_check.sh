#!/bin/bash
# ── experiment liveness monitor — run every 3h via cron ──────────────────────
# Expected usage:
#   */3 * * * * /home3/s3426394/master_isaac/asyncDualPlayPPO/hpc/health_check.sh
# ──────────────────────────────────────────────────────────────────────────────
SCRATCH="/scratch/$USER/final_results_thesis"
CAMPAIGN_DIR="$SCRATCH/.campaign"
LOG="$HOME/experiment_health_$(date +%y%m%d).log"

{
    echo "=== $(date +'%Y-%m-%d %H:%M') ==="

    # ── running / pending jobs ──────────────────────────────────────────
    RUNNING=$(squeue -u "$USER" -h -t RUNNING 2>/dev/null | wc -l)
    PENDING=$(squeue -u "$USER" -h -t PENDING 2>/dev/null | wc -l)
    echo "Jobs: ${RUNNING} running, ${PENDING} pending"

    # ── last 5 running job names + time ──────────────────────────────────
    squeue -u "$USER" -h -t RUNNING -o "%.12j %.8T %.10M" 2>/dev/null | tail -5

    # ── completions in last 6h ───────────────────────────────────────────
    DONE24=$(find "$SCRATCH" -name .done -mmin -360 2>/dev/null | wc -l)
    echo "Completions (6h): $DONE24"

    # ── ABORTED sentinel? ────────────────────────────────────────────────
    ABORTED=$(find "$CAMPAIGN_DIR" -name ABORTED -mmin -360 2>/dev/null)
    if [ -n "$ABORTED" ]; then
        echo "!!! ABORTED sentinel found: $ABORTED !!!"
    fi

    # ── recent TensorBoard writes (any model still logging?) ─────────────
    TB=$(find "$SCRATCH" -name "events.out.*" -mmin -120 2>/dev/null | head -5)
    if [ -z "$TB" ]; then
        echo "!!! No TensorBoard writes in last 2h !!!"
    else
        echo "TB activity (last 5):"
        echo "$TB" | while read f; do echo "  $f"; done
    fi

    # ── disk usage ───────────────────────────────────────────────────────
    du -sh "$SCRATCH" 2>/dev/null

    # ── stalled jobs? (>6h running, no .done change) ────────────────────
    STALLED=$(squeue -u "$USER" -h -t RUNNING -o "%.12j %.10M" 2>/dev/null | \
        awk '$2 ~ /-/ || ($2 !~ /-/ && int(substr($2,1,index($2,":")-1)*60 + substr($2,index($2,":")+1)) > 360) {print}')
    if [ -n "$STALLED" ]; then
        echo "Jobs running >6h:"
        echo "$STALLED"
    fi

    echo "---"
} >> "$LOG" 2>&1
