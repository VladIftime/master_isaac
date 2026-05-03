#!/bin/bash

# Usage: ./restore_slurm_logs.sh YY.MM.DD
if [ -z "$1" ]; then
    echo "Usage: ./restore_slurm_logs.sh YY.MM.DD"
    echo "Example: ./restore_slurm_logs.sh $(date +'%y.%m.%d')"
    exit 1
fi

DATE_STR="$1"
SRC_DIR="/home3/s3426394/master_isaac/asyncDualPlayPPO/logs/$DATE_STR"
DEST_DIR="/home3/s3426394/master_isaac/asyncDualPlayPPO"

# Fallback: check if they are in scratch (since the first version of the script put them there)
if [ ! -d "$SRC_DIR" ] && [ -d "/scratch/s3426394/master_isaac_archive/$DATE_STR/slurm_logs" ]; then
    SRC_DIR="/scratch/s3426394/master_isaac_archive/$DATE_STR/slurm_logs"
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Directory $SRC_DIR does not exist."
    exit 1
fi

echo "Restoring slurm logs from $SRC_DIR..."
# We use rsync to move them back safely
rsync -a --remove-source-files "$SRC_DIR/"slurm-*.out "$DEST_DIR/" 2>/dev/null || true

# Remove the directory if empty
find "$SRC_DIR" -maxdepth 0 -empty -delete 2>/dev/null
echo "Done."
