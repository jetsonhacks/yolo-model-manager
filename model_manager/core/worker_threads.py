import sys
import os
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot
from ultralytics import YOLO, settings


class StreamRedirect(QObject):
    """
    Redirects stdout and stderr to a Qt signal.
    """

    output = Signal(str)

    def __init__(self, old_stream):
        super().__init__()
        self.old_stream = old_stream

    def write(self, text):
        if text.strip():
            self.output.emit(text)
        self.old_stream.write(text)

    def flush(self):
        self.old_stream.flush()


class ModelDownloadWorker(QObject):
    """Worker for downloading models."""

    download_complete = Signal(bool, str)
    output = Signal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    @Slot()
    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirect(old_stdout)
        sys.stderr = StreamRedirect(old_stderr)
        sys.stdout.output.connect(self.output.emit)
        sys.stderr.output.connect(self.output.emit)

        try:
            weights_dir = Path(settings["weights_dir"])
            weights_dir.mkdir(parents=True, exist_ok=True)
            model_path = weights_dir / f"{self.model_name}.pt"

            self.output.emit(f"Starting to download model {self.model_name}...")
            # Ultralytics will automatically download the model if not present.
            YOLO(str(model_path))

            if model_path.exists():
                self.output.emit(
                    f"Download of {self.model_name} complete and stored at {model_path}."
                )
                self.download_complete.emit(True, "Download successful.")
            else:
                self.output.emit("Download failed. Model file not found.")
                self.download_complete.emit(False, "Download failed.")
        except Exception as e:
            self.output.emit(f"An error occurred during download: {str(e)}")
            self.download_complete.emit(False, str(e))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class EngineExportWorker(QObject):
    """Worker for converting a model to a TensorRT engine."""

    done = Signal(bool, str, str, str)
    output = Signal(str)

    def __init__(
        self,
        model_name: str,
        precision: str = "fp16",
        device: str = "cuda",
        calibration_yaml_path: Optional[Path] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.precision = precision
        self.device = device
        self.calibration_yaml_path = calibration_yaml_path

    @Slot()
    def run(self):
        temp_yaml_file = None
        try:
            weights_dir = Path(settings["weights_dir"])
            model_path = weights_dir / f"{self.model_name}.pt"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at {model_path}. Please download the model first."
                )

            self.output.emit(
                f"Starting to build TensorRT engine for {self.model_name} at {self.precision} precision..."
            )

            cmd = [
                "yolo",
                "export",
                f"model={str(model_path)}",
                "format=engine",
                f"device={self.device}",
            ]

            if self.precision == "fp16":
                cmd.append("half")
            elif self.precision == "int8":
                if (
                    not self.calibration_yaml_path
                    or not self.calibration_yaml_path.exists()
                ):
                    raise ValueError(
                        "INT8 calibration requires a valid calibration YAML file to be selected."
                    )

                # Dynamically modify the YAML file to use absolute paths
                with open(self.calibration_yaml_path, "r") as f:
                    data = yaml.safe_load(f)

                yaml_dir = self.calibration_yaml_path.parent
                if "path" in data:
                    data["path"] = str((yaml_dir / data["path"]).resolve())
                if "train" in data and not os.path.isabs(data["train"]):
                    data["train"] = str((yaml_dir / data["train"]).resolve())
                if "val" in data and not os.path.isabs(data["val"]):
                    data["val"] = str((yaml_dir / data["val"]).resolve())

                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".yaml"
                ) as temp_f:
                    yaml.dump(data, temp_f)
                    temp_yaml_file = temp_f.name

                cmd.append("int8")
                cmd.append(f"data={temp_yaml_file}")

            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                for line in proc.stdout:
                    self.output.emit(line.strip())

            if proc.returncode == 0:
                generic_engine_path = (
                    Path(model_path.parent) / f"{model_path.stem}.engine"
                )
                if generic_engine_path.exists():
                    new_engine_path = generic_engine_path.with_name(
                        f"{generic_engine_path.stem}-{self.precision}.engine"
                    )
                    os.rename(generic_engine_path, new_engine_path)
                    self.output.emit(
                        f"Engine build successful. Renamed to {new_engine_path.name}."
                    )
                    self.done.emit(
                        True,
                        self.model_name,
                        self.precision,
                        "Engine build successful.",
                    )
                else:
                    self.output.emit("Engine build failed. Engine file not found.")
                    self.done.emit(
                        False,
                        self.model_name,
                        self.precision,
                        "Failed to create TensorRT engine.",
                    )
            else:
                self.output.emit(
                    f"Engine build failed with exit code {proc.returncode}."
                )
                self.done.emit(
                    False, self.model_name, self.precision, "Engine build failed."
                )

        except (ValueError, FileNotFoundError, SyntaxError) as e:
            self.output.emit(f"Error: {str(e)}")
            self.done.emit(False, self.model_name, self.precision, str(e))
        except Exception as e:
            self.output.emit(f"An unexpected error occurred: {str(e)}")
            self.done.emit(False, self.model_name, self.precision, str(e))
        finally:
            if temp_yaml_file and os.path.exists(temp_yaml_file):
                os.remove(temp_yaml_file)
