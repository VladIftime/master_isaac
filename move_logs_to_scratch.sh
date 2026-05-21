#!/bin/bash

# Define directories
SRC_LOGS="/home3/s3426394/master_isaac/asyncDualPlayPPO/logs"
SRC_RUNS="/home3/s3426394/master_isaac/asyncDualPlayPPO/runs"

# Add date YY.MM.DD to the destination directory
DATE_STR=$(date +'%y.%m.%d')
DEST_DIR="/scratch/s3426394/master_isaac_archive/$DATE_STR"

# Create destination directories if they don't exist
mkdir -p "$DEST_DIR/logs"
mkdir -p "$DEST_DIR/runs"

LOG_FILE="/home3/s3426394/master_isaac/movement.log"
echo "--- $(date +'%Y-%m-%d %H:%M:%S') ---" >> "$LOG_FILE"

# Use rsync to move the files. 
# --remove-source-files will delete the files in the source AFTER they have been successfully copied.
# This is safer than 'mv' in case the script is interrupted.
# We exclude analyze_training.py so it isn't moved or deleted from the source logs directory.
echo "Archiving logs from $SRC_LOGS to $DEST_DIR/logs/" >> "$LOG_FILE"
rsync -av --exclude 'analyze_training.py' --remove-source-files "$SRC_LOGS/" "$DEST_DIR/logs/" >> "$LOG_FILE" 2>&1

# Move runs ONLY if they have a .done file indicating completion
for run_dir in "$SRC_RUNS"/*/; do
    if [ -d "$run_dir" ] && [ -f "${run_dir}.done" ]; then
        run_name=$(basename "$run_dir")
        echo "Archiving completed run: $run_name" >> "$LOG_FILE"
        rsync -av --remove-source-files "$run_dir" "$DEST_DIR/runs/$run_name/" >> "$LOG_FILE" 2>&1
    fi
done

# Remove empty directories left behind by rsync in the source
# Directories containing files (like analyze_training.py) will not be empty, so they won't be deleted.
find "$SRC_LOGS" -mindepth 1 -type d -empty -delete 2>/dev/null
find "$SRC_RUNS" -mindepth 1 -type d -empty -delete 2>/dev/null
