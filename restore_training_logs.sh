#!/bin/bash

# Usage: ./restore_training_logs.sh YY.MM.DD
if [ -z "$1" ]; then
    echo "Usage: ./restore_training_logs.sh YY.MM.DD"
    echo "Example: ./restore_training_logs.sh $(date +'%y.%m.%d')"
    exit 1
fi

DATE_STR="$1"
SRC_ARCHIVE="/scratch/s3426394/master_isaac_archive/$DATE_STR"
DEST_LOGS="/home3/s3426394/master_isaac/asyncDualPlayPPO/logs"
DEST_RUNS="/home3/s3426394/master_isaac/asyncDualPlayPPO/runs"

if [ ! -d "$SRC_ARCHIVE" ]; then
    echo "Error: Archive directory $SRC_ARCHIVE does not exist."
    exit 1
fi

echo "Restoring training logs and runs from $DATE_STR..."
if [ -d "$SRC_ARCHIVE/logs" ]; then
    rsync -a --remove-source-files "$SRC_ARCHIVE/logs/" "$DEST_LOGS/"
fi
if [ -d "$SRC_ARCHIVE/runs" ]; then
    rsync -a --remove-source-files "$SRC_ARCHIVE/runs/" "$DEST_RUNS/"
fi

# Clean up empty directories
find "$SRC_ARCHIVE" -mindepth 1 -type d -empty -delete 2>/dev/null
find "$SRC_ARCHIVE" -maxdepth 0 -empty -delete 2>/dev/null
echo "Done."
