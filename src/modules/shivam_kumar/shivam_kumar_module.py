from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal
import numpy as np
import imageio.v2 as imageio

from modules.i_image_module import IImageModule


class ShivamKumarControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Transformation: Contrast Stretching"))

        self.apply_button = QPushButton("Apply Contrast Stretching")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self.apply)
        layout.addStretch()

    def apply(self):
        self.process_requested.emit({})


class ShivamKumarImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self.controls = None

    def get_name(self):
        return "Shivam Kumar Module"

    def get_supported_formats(self):
        return ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None):
        if self.controls is None:
            self.controls = ShivamKumarControlsWidget(module_manager, parent)
            self.controls.process_requested.connect(self.handle_processing)
        return self.controls

    def handle_processing(self, params):
        self.controls.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path):
        image = imageio.imread(file_path)
        metadata = {"name": file_path.split("/")[-1]}
        return True, image, metadata, None

    def process_image(self, image_data, metadata, params=None):
        return self.contrast_stretching(image_data)

    def contrast_stretching(self, image):
        img = image.astype(np.float32)

        factor = 1.5
        result = 128 + factor * (img - 128)

        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)