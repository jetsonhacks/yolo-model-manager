from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMenuBar,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QFileDialog,
    QFrame,
    QSizePolicy,
    QProgressBar,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread, QObject, Signal, Slot
import torch
from pathlib import Path
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

from ..core.model_manager import ModelManager
from ..core.worker_threads import EngineExportWorker, ModelDownloadWorker

BTN_H = 28  # uniform button height for better vertical alignment


def is_gpu_available():
    return torch.cuda.is_available()


def is_tensorrt_installed():
    try:
        import tensorrt

        return True
    except ImportError:
        return False


def status_icon(ok: bool) -> QLabel:
    """Returns a QLabel with a colored check or cross for status."""
    label = QLabel()
    icon = "✔" if ok else "✘"
    color = "#27ae60" if ok else "#c0392b"
    label.setText(
        f'<span style="font-size: 18px; color:{color}; font-weight:bold;">{icon}</span>'
    )
    label.setAlignment(Qt.AlignVCenter)
    return label


def status_markup_old(present: bool) -> str:
    icon = "✔" if present else "✘"
    text = "Built" if present else "Needs Building"
    color = "#27ae60" if present else "#c0392b"
    return f'<span style="color:{color}; font-weight:600;">{icon} {text}</span>'


def status_markup(present: bool) -> str:
    icon = "✔" if present else "✘"
    text = "Built" if present else "Needs Building"
    color = "#27ae60" if present else "#c0392b"

    return f'<span style="color:{color}; font-size:18px;">{icon}</span> <span style="font-weight:600; color:black;">{text}</span>'


class ModelManagerWindow(QMainWindow):
    def __init__(self, parent=None, model_data=None):
        super().__init__(parent)
        self.model_data = model_data or {}
        self.model_manager = ModelManager()

        self.worker_thread = None
        self.worker = None
        self.calibration_yaml_path = None

        self._setup_ui()
        self._setup_connections()  # This will be done in the next step
        self._update_tasks()

    def _setup_ui(self):
        self.setWindowTitle("YOLO Model Manager")
        self.setMinimumSize(868, 720)

        # Global style: larger, bold group titles
        self.setStyleSheet(
            """
        QGroupBox {
            font-size: 12pt;
            font-weight: bold;
            /* Create space for the title */
            margin-top: 12px;
            border: 1px solid #C0C0C0;
            border-radius: 8px;
        }
        QGroupBox::title {
            /* Position the title within the margin space */
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 3px 0 3px;
        }
        #engineCard {
            border: 1px solid rgba(0,0,0,0.18);
            border-radius: 8px;
            background: rgba(255,255,255,0.6);
        }
        #engineCard QLabel {
            padding: 0px;
        }
        """
        )

        menubar = QMenuBar(None)
        menubar.setNativeMenuBar(True)
        help_menu = menubar.addMenu("Help")
        help_action = help_menu.addAction("About")
        help_action.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "About",
                "YOLO Model Manager\nVersion 1.0\nDeveloped by JetsonHacks",
            )
        )
        details_menu = menubar.addMenu("Show Details")

        # Create a central widget to hold your main UI layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Apply the main layout to the central widget, not the QMainWindow
        root = QVBoxLayout(central_widget)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ====== TOP ROW with 3 columns ======
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        root.addLayout(top_row)

        # =========== Model Selection ===========
        model_box = QGroupBox("Model Selection")
        model_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        model_layout = QVBoxLayout(model_box)
        model_layout.setSpacing(8)

        self.version_combo = QComboBox()
        self.version_combo.addItems(list(self.model_data.keys()))
        self.task_combo = QComboBox()
        self.model_combo = QComboBox()

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Model Status:"))
        self.model_status_label = QLabel()
        self.model_status_label.setStyleSheet("font-weight:700;")
        status_row.addWidget(self.model_status_label)
        status_row.addStretch()

        self.download_btn = QPushButton("Download Model")
        self.download_btn.setFixedHeight(BTN_H)

        model_layout.addWidget(QLabel("Task:"))
        model_layout.addWidget(self.task_combo)
        model_layout.addWidget(QLabel("Model Version:"))
        model_layout.addWidget(self.version_combo)
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addLayout(status_row)
        model_layout.addWidget(self.download_btn)
        model_layout.addStretch()

        # =========== Engine Build ===========
        engine_box = QGroupBox("Engine Build")
        engine_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        engine_box.setContentsMargins(10, 8, 10, 10)
        engine_layout = QVBoxLayout(engine_box)
        engine_layout.setSpacing(10)

        def make_precision_card(title: str, btn_attr: str, lbl_attr: str) -> QFrame:
            card = QFrame()
            card.setObjectName("engineCard")
            card.setFrameShape(QFrame.StyledPanel)
            card.setFrameShadow(QFrame.Raised)
            card.setMinimumWidth(316)

            v = QVBoxLayout(card)
            v.setContentsMargins(10, 10, 8, 10)  # Left, Top, Right, Bottom
            v.setSpacing(6)

            v.addWidget(QLabel(f"<b>{title} Engine</b>"))

            row = QHBoxLayout()
            row.setSpacing(8)

            # Content on the left
            left = QHBoxLayout()
            left.setSpacing(6)
            left.addWidget(QLabel("• Engine file:"))
            lbl_status = QLabel(status_markup(False))
            setattr(self, lbl_attr, lbl_status)
            left.addWidget(lbl_status)

            row.addLayout(left)

            # We will use addStretch() to push the button to the right
            row.addStretch()

            # Right-aligned button
            btn = QPushButton("Build")
            btn.setFixedHeight(BTN_H)
            setattr(self, btn_attr, btn)
            row.addWidget(btn)

            v.addLayout(row)
            return card

        def make_int8_card() -> QFrame:
            card = QFrame()
            card.setObjectName("engineCard")
            card.setFrameShape(QFrame.StyledPanel)
            card.setFrameShadow(QFrame.Raised)
            card.setMinimumWidth(316)
            v = QVBoxLayout(card)
            v.setContentsMargins(10, 10, 10, 10)
            v.setSpacing(6)
            v.addWidget(QLabel("<b>INT8 Engine</b>"))
            row_status = QHBoxLayout()
            row_status.setSpacing(8)
            left = QHBoxLayout()
            left.setSpacing(6)
            left.addWidget(QLabel("• Engine file:"))
            self.lbl_int8_status = QLabel(status_markup(False))
            left.addWidget(self.lbl_int8_status)
            row_status.addLayout(left)
            row_status.addStretch()
            self.btn_build_int8 = QPushButton("Build")
            self.btn_build_int8.setFixedHeight(BTN_H)
            row_status.addWidget(
                self.btn_build_int8, 0, Qt.AlignVCenter | Qt.AlignRight
            )
            v.addLayout(row_status)
            v.addSpacing(4)
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setFrameShadow(QFrame.Sunken)
            v.addWidget(sep)
            v.addSpacing(4)
            row_cal = QHBoxLayout()
            row_cal.setSpacing(6)
            row_cal.addWidget(QLabel("• Calibration:"))
            self.lbl_calib = QLabel("<i>None selected</i>")
            self.lbl_calib.setTextFormat(Qt.RichText)
            self.lbl_calib.setToolTip(
                "Calibration dataset YAML used for INT8 quantization."
            )
            row_cal.addWidget(self.lbl_calib)
            row_cal.addStretch()
            v.addLayout(row_cal)
            self.btn_pick_calib = QPushButton("Load Calibration YAML ...")
            self.btn_pick_calib.setFixedHeight(BTN_H)
            v.addWidget(self.btn_pick_calib, alignment=Qt.AlignLeft)
            return card

        engine_layout.addWidget(
            make_precision_card("FP32", "btn_build_fp32", "lbl_fp32_status")
        )
        engine_layout.addWidget(
            make_precision_card("FP16", "btn_build_fp16", "lbl_fp16_status")
        )
        engine_layout.addWidget(make_int8_card())
        engine_layout.addStretch()

        # =========== System Info ===========
        system_info_box = QGroupBox("System Info")
        system_info_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        env_layout = QVBoxLayout(system_info_box)
        env_layout.setSpacing(8)

        row_gpu = QHBoxLayout()
        row_gpu.addWidget(QLabel("GPU Available:"))
        self.gpu_stat_icon = status_icon(is_gpu_available())
        row_gpu.addWidget(self.gpu_stat_icon)
        self.gpu_stat_text = QLabel(f"<b>{'Yes' if is_gpu_available() else 'No'}</b>")
        row_gpu.addWidget(self.gpu_stat_text)
        row_gpu.addStretch()

        row_trt = QHBoxLayout()
        row_trt.addWidget(QLabel("TensorRT Installed:"))
        self.trt_stat_icon = status_icon(is_tensorrt_installed())
        row_trt.addWidget(self.trt_stat_icon)
        self.trt_stat_text = QLabel(
            f"<b>{'Yes' if is_tensorrt_installed() else 'No'}</b>"
        )
        row_trt.addWidget(self.trt_stat_text)
        row_trt.addStretch()

        self.open_models_btn = QPushButton("Open Models Folder")
        self.open_models_btn.setFixedHeight(BTN_H)

        env_layout.addLayout(row_gpu)
        env_layout.addLayout(row_trt)
        env_layout.addWidget(self.open_models_btn)
        env_layout.addStretch()

        top_row.addWidget(model_box, 3)
        top_row.addWidget(engine_box, 4)
        top_row.addWidget(system_info_box, 2)

        # ====== Build Log ======
        log_box = QGroupBox("Build Log")
        log_layout = QVBoxLayout(log_box)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(220)
        self.log_output.setStyleSheet(
            "background-color: #f0f0f0; font-family: 'Courier New', Courier, monospace;"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()

        log_btn_row = QHBoxLayout()
        log_btn_row.addStretch()
        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.setFixedHeight(BTN_H)
        log_btn_row.addWidget(self.clear_log_btn)

        log_layout.addWidget(self.log_output)
        log_layout.addWidget(self.progress_bar)
        log_layout.addLayout(log_btn_row)
        root.addWidget(log_box, 1)

    def _setup_connections(self):
        self.open_models_btn.clicked.connect(self._open_models_directory)
        self.clear_log_btn.clicked.connect(self.log_output.clear)

        # Connect model selection comboboxes
        self.version_combo.currentTextChanged.connect(self._update_tasks)
        self.task_combo.currentTextChanged.connect(self._update_models)
        self.model_combo.currentTextChanged.connect(self._update_model_status)

        # Connect model download button
        self.download_btn.clicked.connect(self._download_model)

        # Connect engine build buttons
        self.btn_build_fp32.clicked.connect(lambda: self._build_engine("fp32"))
        self.btn_build_fp16.clicked.connect(lambda: self._build_engine("fp16"))
        self.btn_build_int8.clicked.connect(lambda: self._build_engine("int8"))

        # Connect INT8 calibration file selection
        self.btn_pick_calib.clicked.connect(self._select_calibration_data)

    def _update_tasks(self):
        version = self.version_combo.currentText()
        models = self.model_data.get(version, [])
        tasks = []
        for m in models:
            if "-pose" in m:
                tasks.append("Pose")
            elif "-seg" in m:
                tasks.append("Segmentation")
            elif "-obb" in m:
                tasks.append("OBB")
            elif "-cls" in m:
                tasks.append("Classification")
            else:
                tasks.append("Detection")
        task_set = sorted(
            set(tasks),
            key=lambda x: (
                ["Detection", "Segmentation", "Pose", "OBB", "Classification"].index(x)
                if x in ["Detection", "Segmentation", "Pose", "OBB", "Classification"]
                else x
            ),
        )
        self.task_combo.blockSignals(True)
        self.task_combo.clear()
        self.task_combo.addItems(task_set)
        self.task_combo.blockSignals(False)
        self._update_models()

    def _update_models(self):
        version = self.version_combo.currentText()
        task = self.task_combo.currentText()
        models = self.model_data.get(version, [])
        filtered = []
        for m in models:
            if task == "Detection" and not any(
                x in m for x in ("-seg", "-pose", "-obb", "-cls")
            ):
                filtered.append(m)
            elif task == "Segmentation" and "-seg" in m:
                filtered.append(m)
            elif task == "Pose" and "-pose" in m:
                filtered.append(m)
            elif task == "OBB" and "-obb" in m:
                filtered.append(m)
            elif task == "Classification" and "-cls" in m:
                filtered.append(m)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(filtered)
        self.model_combo.blockSignals(False)
        self._update_model_status()

    def _update_model_status(self):
        model = self.model_combo.currentText()
        if not model or not self.model_manager:
            self.model_status_label.setText("")
            self.download_btn.setEnabled(False)
            self.lbl_fp32_status.setText(status_markup(False))
            self.btn_build_fp32.setEnabled(False)
            self.lbl_fp16_status.setText(status_markup(False))
            self.btn_build_fp16.setEnabled(False)
            self.lbl_int8_status.setText(status_markup(False))
            self.btn_build_int8.setEnabled(False)
            return

        is_downloaded = self.model_manager.is_model_downloaded(
            "Yolov8", model
        )  # Version is hardcoded
        self.model_status_label.setText(
            f"{status_icon(is_downloaded).text()} {'Downloaded' if is_downloaded else 'Not Downloaded'}"
        )
        self.download_btn.setText(
            "Re-download Model" if is_downloaded else "Download Model"
        )
        self.download_btn.setEnabled(True)

        is_gpu_ok = is_gpu_available() and is_tensorrt_installed()

        # Update each precision card
        for prec, lbl_name, btn_name in [
            ("fp32", "lbl_fp32_status", "btn_build_fp32"),
            ("fp16", "lbl_fp16_status", "btn_build_fp16"),
        ]:
            is_built = self.model_manager.engine_exists(model, prec)
            lbl = getattr(self, lbl_name)
            btn = getattr(self, btn_name)

            lbl.setText(status_markup(is_built))
            btn.setText("Rebuild" if is_built else "Build")

            can_build = is_downloaded and is_gpu_ok
            btn.setEnabled(can_build)

        # Handle INT8 separately due to calibration file requirement
        is_int8_built = self.model_manager.engine_exists(model, "int8")
        self.lbl_int8_status.setText(status_markup(is_int8_built))
        self.btn_build_int8.setText("Rebuild" if is_int8_built else "Build")

        can_build_int8 = (
            is_downloaded and is_gpu_ok and (self.calibration_yaml_path is not None)
        )
        self.btn_build_int8.setEnabled(can_build_int8)

        # Tooltips to explain why a button might be disabled
        if not is_gpu_ok:
            tooltip_text = "Building a TensorRT engine requires a CUDA-enabled GPU and TensorRT to be installed."
            self.btn_build_fp32.setToolTip(tooltip_text)
            self.btn_build_fp16.setToolTip(tooltip_text)
            self.btn_build_int8.setToolTip(tooltip_text)
        elif not is_downloaded:
            tooltip_text = "Please download the model first before building an engine."
            self.btn_build_fp32.setToolTip(tooltip_text)
            self.btn_build_fp16.setToolTip(tooltip_text)
            self.btn_build_int8.setToolTip(tooltip_text)
        elif "INT8" in self.btn_build_int8.text() and not self.calibration_yaml_path:
            self.btn_build_int8.setToolTip(
                "Select a calibration YAML to enable INT8 build."
            )

    @Slot()
    def _download_model(self):
        model_name = self.model_combo.currentText()
        if not model_name:
            QMessageBox.warning(
                self, "No Model Selected", "Please select a model to download."
            )
            return

        self.log_output.clear()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self._set_ui_enabled(False)
        self.log_output.append(f"Starting model download for {model_name}...")

        self.worker_thread = QThread()
        self.worker = ModelDownloadWorker(model_name=model_name)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.download_complete.connect(self._on_download_complete)
        self.worker.output.connect(self._append_log)
        self.worker_thread.start()

    def _build_engine(self, precision):
        model_name = self.model_combo.currentText()
        if not model_name:
            QMessageBox.warning(
                self, "No Model Selected", "Please select a model to build the engine."
            )
            return

        self.log_output.clear()
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)
        self._set_ui_enabled(False)
        self.log_output.append(
            f"Starting engine build for {model_name} with precision {precision}..."
        )

        calibration_yaml_path = (
            self.calibration_yaml_path if precision.lower() == "int8" else None
        )

        self.worker_thread = QThread()
        self.worker = EngineExportWorker(
            model_name=model_name,
            precision=precision.lower(),
            device="cuda:0",
            calibration_yaml_path=calibration_yaml_path,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.done.connect(self._on_engine_built)
        self.worker.output.connect(self._append_log)
        self.worker_thread.start()

    @Slot(bool, str)
    def _on_download_complete(self, success, message):
        self._common_task_cleanup(success, message)
        if success:
            QMessageBox.information(self, "Download Complete", message)
        else:
            QMessageBox.critical(self, "Download Failed", message)
        self._update_model_status()

    @Slot(bool, str, str, str)
    def _on_engine_built(self, success, model_name, precision, message):
        self._common_task_cleanup(success, message)
        if success:
            QMessageBox.information(
                self,
                "Task Complete",
                f"Engine ({precision}) for {model_name} built successfully.",
            )
        else:
            QMessageBox.critical(
                self,
                "Task Failed",
                f"Failed to build engine for {model_name}.\n\n{message}",
            )
        self._update_model_status()

    def _common_task_cleanup(self, success, message):
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)
        self._set_ui_enabled(True)
        if self.worker_thread:
            # Check if the worker thread is running before quitting
            # The worker might have already finished and quit the thread
            if self.worker_thread.isRunning():
                self.worker_thread.quit()
                self.worker_thread.wait()

        self.log_output.append("\n--- Task Finished ---")
        if success:
            self.log_output.append(f"✅ {message}")
        else:
            self.log_output.append(f"❌ {message}")

    @Slot(str)
    def _append_log(self, text):
        self.log_output.append(text)
        QApplication.processEvents()

    @Slot()
    def _open_models_directory(self):
        models_dir = self.model_manager.weights_dir

        # Ensure the directory exists before trying to open it
        if not models_dir.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            QMessageBox.information(
                self,
                "Folder Created",
                f"The models folder was created at {models_dir}.",
            )

        # Convert the pathlib.Path object to an absolute QUrl and open it
        absolute_path = models_dir.absolute()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(absolute_path)))

    def _set_ui_enabled(self, enabled):
        self.version_combo.setEnabled(enabled)
        self.task_combo.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.download_btn.setEnabled(enabled)
        self.btn_build_fp32.setEnabled(enabled)
        self.btn_build_fp16.setEnabled(enabled)
        self.btn_build_int8.setEnabled(enabled)
        self.btn_pick_calib.setEnabled(enabled)
        self.open_models_btn.setEnabled(enabled)
        self.clear_log_btn.setEnabled(enabled)

    @Slot()
    def _select_calibration_data(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration YAML File", "", "YAML Files (*.yaml)"
        )
        if file:
            self.calibration_yaml_path = Path(file)
            self.lbl_calib.setText(f"<i>{self.calibration_yaml_path.name}</i>")
            self.lbl_calib.setToolTip(str(self.calibration_yaml_path))
            # Refresh the UI status to enable the INT8 build button
            self._update_model_status()
        else:
            self.calibration_yaml_path = None
            self.lbl_calib.setText("<i>None selected</i>")
            self.lbl_calib.setToolTip(
                "Calibration dataset YAML used for INT8 quantization."
            )
            self._update_model_status()

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Task in Progress",
                "A task is currently running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                # The workers do not currently have a stop method, so we just
                # warn the user and quit the thread. The process may continue.
                # A more robust solution would be to implement a stop method.
                if self.worker and hasattr(self.worker, "stop"):
                    self.worker.stop()
                self.worker_thread.quit()
                self.worker_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
