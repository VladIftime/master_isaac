#!/bin/bash

# Save the current directory
CURRENT_DIR=$(pwd)

# Navigate to the Isaac Sim folder
cd ~/.local/share/ov/pkg/isaac-sim-4.2.0 || exit 1

# Source the setup script
source ~/.local/share/ov/pkg/isaac-sim-4.2.0/setup_conda_env.sh

# Return to the original directory
cd "$CURRENT_DIR"
