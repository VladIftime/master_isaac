#!/bin/bash

# Define source directory for slurm logs
SRC_DIR="/home3/s3426394/master_isaac/asyncDualPlayPPO"

# Add date YY.MM.DD to the destination directory
DATE_STR=$(date +'%y.%m.%d')
DEST_DIR="/home3/s3426394/master_isaac/asyncDualPlayPPO/logs/$DATE_STR"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Use rsync to safely move ONLY slurm-*.out files from the root of asyncDualPlayPPO directory.
# --remove-source-files deletes the original files after successful copy.
rsync -a --remove-source-files --include="slurm-*.out" --exclude="*" "$SRC_DIR/" "$DEST_DIR/"
