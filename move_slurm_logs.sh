#!/bin/bash

# Define source directory for slurm logs
SRC_DIR="/home3/s3426394/master_isaac/asyncDualPlayPPO"

# Add date YY.MM.DD to the destination directory
DATE_STR=$(date +'%y.%m.%d')
DEST_DIR="/home3/s3426394/master_isaac/asyncDualPlayPPO/logs/$DATE_STR"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Use a for loop to iterate over ONLY slurm-*.out files in the root of asyncDualPlayPPO directory.
for log_file in "$SRC_DIR"/slurm-*.out; do
    # Skip if no files match the glob
    [ -e "$log_file" ] || continue

    # Check for completion markers in the file (using tail to avoid searching massive files entirely)
    if tail -n 50 "$log_file" | grep -E -q "(no resubmission|chained next job|Resubmission failed)"; then
        # If the marker is found, safely move the file using rsync
        rsync -a --remove-source-files "$log_file" "$DEST_DIR/"
    fi
done
