import sys
import json
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMessageBox
from model_manager.ui.model_manager_window import ModelManagerWindow


def main():
    """
    Main entry point for the YOLO Model Manager application.
    """

    model_data_path = Path("models.json")
    if not model_data_path.exists():
        QMessageBox.critical(
            None,
            "Startup Error",
            f"Required configuration file not found: '{model_data_path}'\n"
            "Please ensure 'models.json' is in the application directory.",
        )
        sys.exit(1)

    try:
        with open(model_data_path, "r") as f:
            model_data = json.load(f)

        app = QApplication(sys.argv)
        app.setApplicationName("Model Manager")  # Set the application name here
        main_window = ModelManagerWindow(model_data=model_data)
        main_window.show()
        sys.exit(app.exec())

    except Exception as e:
        print(f"Startup Error: {e}")
        QMessageBox.critical(None, "Startup Error", f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
