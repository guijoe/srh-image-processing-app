from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Signal
import numpy as np
import imageio.v2 as imageio
import os

from modules.i_image_module import IImageModule


class ShivamKumarControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose Transformation:"))

        self.contrast_button = QPushButton("Apply Contrast Stretching")
        layout.addWidget(self.contrast_button)

        self.blur_button = QPushButton("Apply Gaussian Blur")
        layout.addWidget(self.blur_button)

        self.edge_button = QPushButton("Apply Edge Detection")
        layout.addWidget(self.edge_button)

        self.contrast_button.clicked.connect(lambda: self.apply("contrast"))
        self.blur_button.clicked.connect(lambda: self.apply("blur"))
        self.edge_button.clicked.connect(lambda: self.apply("edge"))

        layout.addStretch()

    def apply(self, operation):
        self.process_requested.emit({"operation": operation})


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
        metadata = {"name": os.path.basename(file_path)}
        return True, image, metadata, None

    def process_image(self, image_data, metadata, params=None):
        operation = params.get("operation", "contrast") if params else "contrast"

        if operation == "blur":
            return self.gaussian_blur(image_data)

        if operation == "edge":
            return self.edge_detection(image_data)

        return self.contrast_stretching(image_data)

    def contrast_stretching(self, image):
        img = image.astype(np.float32)

        factor = 1.5
        result = 128 + factor * (img - 128)

        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)

    def gaussian_blur(self, image):
        img = image.astype(np.float32)

        if img.ndim == 2:
            padded = np.pad(img, 1, mode="edge")

            result = (
                1 * padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + 1 * padded[:-2, 2:] +
                2 * padded[1:-1, :-2] + 4 * padded[1:-1, 1:-1] + 2 * padded[1:-1, 2:] +
                1 * padded[2:, :-2] + 2 * padded[2:, 1:-1] + 1 * padded[2:, 2:]
            ) / 16

        else:
            padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")

            result = (
                1 * padded[:-2, :-2, :] + 2 * padded[:-2, 1:-1, :] + 1 * padded[:-2, 2:, :] +
                2 * padded[1:-1, :-2, :] + 4 * padded[1:-1, 1:-1, :] + 2 * padded[1:-1, 2:, :] +
                1 * padded[2:, :-2, :] + 2 * padded[2:, 1:-1, :] + 1 * padded[2:, 2:, :]
            ) / 16

        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)

    def edge_detection(self, image):
        img = image.astype(np.float32)

        if img.ndim == 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img

        padded = np.pad(gray, 1, mode="edge")

        gx = (
            -1 * padded[:-2, :-2] + 1 * padded[:-2, 2:] +
            -2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:] +
            -1 * padded[2:, :-2] + 1 * padded[2:, 2:]
        )

        gy = (
            -1 * padded[:-2, :-2] + -2 * padded[:-2, 1:-1] + -1 * padded[:-2, 2:] +
             1 * padded[2:, :-2] +  2 * padded[2:, 1:-1] +  1 * padded[2:, 2:]
        )

        edges = np.sqrt(gx ** 2 + gy ** 2)

        if edges.max() != 0:
            edges = edges / edges.max() * 255

        edges = edges.astype(np.uint8)

        if image.ndim == 3:
            edges = np.stack([edges, edges, edges], axis=2)

        return edges