#!/usr/bin/env bash
#
# setup_yolo_project.sh
#
# Setup YOLO project for yolo-model-manager.
# This script handles system dependencies, virtual environment activation,
# and Python package installation, including a fix for TensorRT bindings.

set -Eeuo pipefail

# --- Configuration ---
VENV_DIR="${VENV_DIR:-$HOME/yolo-venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"
DIST_PACKAGES_PATH="/usr/lib/python3.10/dist-packages"
TENSORRT_PTH_FILE="$VENV_DIR/lib/python3.10/site-packages/tensorrt.pth"

# --- Utility Functions ---
log() {
  echo "--> $*"
}

die() {
  echo "Error: $*" >&2
  exit 1
}

# --- Main Script ---

log "Starting project setup..."

log "Updating package list and installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y libxcb-cursor0

log "Activating virtual environment at $VENV_DIR..."
if [[ ! -d "$VENV_DIR" ]]; then
  die "Virtual environment not found at $VENV_DIR. Please create it first."
fi

source "$VENV_DIR/bin/activate"

log "Installing Python packages from $REQ_FILE..."
if [[ ! -f "$REQ_FILE" ]]; then
  die "requirements.txt not found at $REQ_FILE."
fi
uv pip install -r "$REQ_FILE"

log "Checking for TensorRT dist-packages path in venv..."
if [[ ! -f "$TENSORRT_PTH_FILE" ]]; then
  log "Creating .pth file to link to TensorRT system bindings..."
  echo "$DIST_PACKAGES_PATH" > "$TENSORRT_PTH_FILE"
else
  log "TensorRT .pth file already exists. Skipping."
fi

log "Setup complete."
log "To use the environment, run: source $VENV_DIR/bin/activate"
