#!/usr/bin/env bash
set -euo pipefail

# Usage: Tools/pose_datafeeder.sh <device> <csv_path>
# Example:
#   Tools/pose_datafeeder.sh 1 /workspaces/CameraSensor/replay/2025-07-31_16-54-02/Pose_0.csv

DEVICE=${1:-1}
CSV_PATH=${2:-/workspaces/CameraSensor/replay/2025-07-31_16-54-02/Pose_0.csv}

# Resolve directory of this script so it works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start the receiver first so it is ready for incoming data.
python3 "$SCRIPT_DIR/pose_datafeeder_receive.py" --device "$DEVICE" &
RECEIVER_PID=$!

# Give the receiver a moment to connect before sending data.
sleep 1

# Send pose data from the CSV file.
python3 "$SCRIPT_DIR/pose_datafeeder_send.py" --device "$DEVICE" --csv "$CSV_PATH"

# Wait for the receiver to finish processing.
wait $RECEIVER_PID
