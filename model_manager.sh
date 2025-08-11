#!/usr/bin/env bash

# Path to the virtual environment
VENV_DIR="$HOME/yolo-venv"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Run the Python application
python -m model_manager

# Deactivate the virtual environment after the script finishes
deactivate