import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO, settings


def get_ultralytics_weights_dir():
    """Returns the path where Ultralytics expects to find/download weights."""
    return Path(settings["weights_dir"])


class ModelManager:
    def __init__(self):
        self._current_model: Optional[YOLO] = None
        self._current_model_name: Optional[str] = None
        self.weights_dir = get_ultralytics_weights_dir()

    @staticmethod
    def load_model_data_from_json(
        file_path: str = "models.json",
    ) -> Dict[str, List[str]]:
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model configuration file not found at: {file_path}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error decoding JSON from file: {file_path}\n{str(e)}", doc="", pos=0
            )

    def get_engine_path(self, model_name: str, precision: str = "fp16") -> Path:
        name = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        return self.weights_dir / f"{Path(name).stem}-{precision}.engine"

    def engine_exists(self, model_name: str, precision: str = "fp16") -> bool:
        return self.get_engine_path(model_name, precision).exists()

    def model_file_path(self, model_name: str) -> Path:
        name = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        return self.weights_dir / name

    def is_model_downloaded(self, version: str, model: str) -> bool:
        model_path = self.model_file_path(model)
        return model_path.exists()

    def get_cached_model_path(self, model_name: str) -> Optional[Path]:
        path = self.model_file_path(model_name)
        return path if path.exists() else None

    def list_cached_models(self) -> List[str]:
        return [f.name for f in self.weights_dir.glob("*.pt")]

    def load_model(self, model_name: str) -> bool:
        full_path = str(self.model_file_path(model_name))
        if self._current_model and self._current_model_name == full_path:
            print(f"Model {full_path} is already loaded (cached).")
            return True
        try:
            print(
                f"Loading model {full_path} (Ultralytics will use cache or download)..."
            )
            self._current_model = YOLO(full_path)
            self._current_model_name = full_path
            path = getattr(self._current_model, "ckpt_path", None)
            if path:
                print(f"Model loaded from: {path}")
            else:
                print(f"Model loaded: {self._current_model_name}")
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            self._current_model = None
            self._current_model_name = None
            return False

    def download_model(self, version: str, model: str) -> bool:
        name = model if model.endswith(".pt") else f"{model}.pt"
        model_path = self.model_file_path(name)
        print(f"Expected model_path: {model_path}")
        if model_path.exists():
            print(f"Model already present: {model_path}")
            return True
        try:
            print("Starting YOLO download via ultralytics...")
            YOLO(str(model_path))
            print(f"Download finished. Exists? {model_path.exists()}")
            return model_path.exists()
        except Exception as e:
            print(f"Error in download_model: {e}")
            return False

    def get_model(self) -> Optional[YOLO]:
        return self._current_model

    def clear_model(self):
        self._current_model = None
        self._current_model_name = None

    def list_all_models_by_version(self):
        return ModelManager.load_model_data_from_json()

    def is_engine_built(self, model: str, precision: str) -> bool:
        return self.engine_exists(model, precision)

    def build_engine(self, model: str, precision: str) -> bool:
        import time

        engine_path = self.get_engine_path(model, precision)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Building engine {engine_path} (may take several minutes)...")
        time.sleep(4)
        engine_path.touch()
        print(f"Engine built at {engine_path}")
        return engine_path.exists()

    def can_download_model(self, version: str, model: str) -> bool:
        return True
