#!/usr/bin/env bash

# download_coco128.sh - Download and set up the COCO128 dataset and YAML for YOLO
# This script:
#   - Downloads coco128.zip
#   - Extracts to ./coco128/
#   - Downloads coco128.yaml
#   - Adjusts it for local use (relative paths)
#   - Stores the YAML inside ./coco128/

set -euo pipefail

# Constants
DATASET_URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
YAML_URL="https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco128.yaml"
ZIP_FILE="coco128.zip"
DATASET_DIR="coco128"
YAML_FILE="${DATASET_DIR}/coco128.yaml"

echo "[INFO] Downloading COCO128 dataset..."
if [ -f "$ZIP_FILE" ]; then
    echo "[SKIP] $ZIP_FILE already exists."
else
    curl -L -o "$ZIP_FILE" "$DATASET_URL"
    echo "[OK] Downloaded $ZIP_FILE"
fi

echo "[INFO] Extracting dataset to $DATASET_DIR..."
if [ -d "$DATASET_DIR" ]; then
    echo "[SKIP] $DATASET_DIR already exists."
else
    unzip -q "$ZIP_FILE"
    echo "[OK] Extracted to $DATASET_DIR"
fi

echo "[INFO] Downloading coco128.yaml..."
curl -s -L -o "$YAML_FILE" "$YAML_URL"
echo "[OK] Saved $YAML_FILE"

echo "[INFO] Adjusting paths in coco128.yaml..."
# Clean and modify the YAML file to be local-relative
# - path: . (relative to the YAML file itself)
# - train: images/train2017
# - val: images/train2017
# - test: (empty or unset)

# Replace or insert lines using awk
awk '
BEGIN { in_names = 0 }
/^path:/ { print "path: ."; next }
/^train:/ { print "train: images/train2017"; next }
/^val:/ { print "val: images/train2017"; next }
/^test:/ { print "test:"; next }
{ print }
' "$YAML_FILE" > "${YAML_FILE}.tmp" && mv "${YAML_FILE}.tmp" "$YAML_FILE"

echo "[SUCCESS] COCO128 dataset and YAML are ready."
echo "YAML path: $YAML_FILE"
