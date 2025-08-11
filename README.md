# YOLO Model Manager

A desktop GUI application to simplify the management, downloading, and TensorRT engine conversion of Ultralytics YOLO models. Built with PySide6, this tool provides a user-friendly graphical interface, eliminating the need for complex command-line operations for.

### Note:
* The .engine files created do not utilize the DLA engines on Jetson NX or AGX. 
* The .engine files are Ultralytics format; While they are in TensorRT format, they may not be compatible with the NVIDIA DeepStream framework.
* It may take significant time to download the YOLO weight files or convert the models to .engine files. These may take several minutes, depending on which options are selected. 
* It can take several seconds to launch the application

## Features

- **Model Management**: Load, view, and manage a curated list of YOLO models from a `models.json` file.
- **Model Download**: Download models directly from the Ultralytics repository.
- **TensorRT Engine Builder**: Convert downloaded models to TensorRT engines with support for FP32, FP16, and INT8 precision.
- **INT8 Calibration**: Select a calibration dataset (e.g., COCO128) using a YAML file to build highly-optimized INT8 engines.
- **System Information**: Displays the availability of a CUDA-enabled GPU and TensorRT.

## Prerequisites

- Ubuntu 22.04 (recommended)
- NVIDIA GPU with CUDA drivers installed
- Python 3.10+
- `uv` package installer (recommended for speed)
- Ultralytics Yolo

## Setup

Use the provided scripts to set up the project environment.

### 1. Project Setup Script

The `setup_yolo_project.sh` script handles the creation of a Python virtual environment, installs the required packages, and configures the environment for TensorRT.

```bash
# Run the setup script
./setup_yolo_project.sh
```
### 2. INT8 Calibration Data
To build INT8 engines, you need a calibration dataset. The download_coco128.sh script will download and configure a small dataset for this purpose.
```Bash
./download_coco128.sh
```
## Usage
### Running the Application
Activate the virtual environment, and start the application:

```Bash
source ~/yolo-venv/bin/activate
python -m model_manager
```
or, you can run the provided shell script which sets the virtual environment and launches the application:
```Bash
./model_manager.sh
```

## Application Workflow
Select a Model: Use the drop-down menus to choose a task, a model version, and specific model.

**Download:** If the model is not downloaded, click the "Download Model" button. The progress will be shown in the log.

**Build Engine**:

Select the desired precision (FP32, FP16, or INT8).

For INT8, click "Load Calibration YAML..." and select the coco128/coco128.yaml file.

Click the "Build" button for the desired precision to start the conversion process.

## Scripts
**setup_yolo_project.sh**: A comprehensive script for setting up the virtual environment and installing all necessary dependencies.

**download_coco128.sh**: Downloads the COCO128 dataset and prepares a calibration YAML file for use with INT8 engine building.

## Release Notes
** Initial Release, August 2025 **
* Tested on Jetson Orin Nano, JetPack 6.2.1
