from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
from PySide6.QtCore import Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule


# --- Parameter Widgets ---

class BaseParamsWidget(QWidget):
    """Base class for parameter widgets."""
    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    """Placeholder for operations with no parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class BoxBlurParamsWidget(BaseParamsWidget):
    """Widget for Box Blur parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Kernel Size (odd number):"))
        self.size_spinbox = QDoubleSpinBox()
        self.size_spinbox.setMinimum(3.0)
        self.size_spinbox.setMaximum(21.0)
        self.size_spinbox.setValue(3.0)
        self.size_spinbox.setSingleStep(2.0)
        self.size_spinbox.setDecimals(0)
        layout.addWidget(self.size_spinbox)

        hint = QLabel("Larger = more blur")
        hint.setStyleSheet("font-style: italic; color: gray; font-size: 10px;")
        layout.addWidget(hint)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'kernel_size': int(self.size_spinbox.value())}


class GammaCorrectionParamsWidget(BaseParamsWidget):
    """Widget for Gamma Correction parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Gamma Value:"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(5.0)
        self.gamma_spinbox.setValue(1.0)
        self.gamma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.gamma_spinbox)

        hint = QLabel("< 1.0 = brighter, > 1.0 = darker")
        hint.setStyleSheet("font-style: italic; color: gray; font-size: 10px;")
        layout.addWidget(hint)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'gamma': self.gamma_spinbox.value()}


# --- Control Panel ---

class AnunithaControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Control Panel</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        # Stacked widget to hold parameter UIs
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Define operations and their parameter widgets
        operations = {
            "Image Negation": NoParamsWidget,
            "Gamma Correction": GammaCorrectionParamsWidget,
            "Box Blur": BoxBlurParamsWidget,
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply_clicked(self):
        operation_name = self.operation_selector.currentText()
        active_widget = self.param_widgets[operation_name]
        params = active_widget.get_params()
        params['operation'] = operation_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])


# --- Main Module ---

class AnunithaImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Anunitha Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = AnunithaControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)

            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                pass  # RGB or RGBA, keep as is
            elif image_data.ndim == 2:
                image_data = image_data[np.newaxis, :]  # Add channel dimension

            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')

        if operation == "Image Negation":
            # Invert the image: subtract each pixel value from the max possible value
            if processed_data.dtype == np.uint8:
                processed_data = np.uint8(255) - processed_data
            else:
                max_val = np.max(processed_data)
                processed_data = max_val - processed_data

        elif operation == "Gamma Correction":
            gamma = params.get('gamma', 1.0)

            # Convert to float for calculation
            img_float = processed_data.astype(np.float64)

            # Normalize to [0, 1]
            max_val = np.max(img_float)
            if max_val > 0:
                normalized = img_float / max_val

                # Apply gamma: output = input ^ gamma
                corrected = np.power(normalized, gamma)

                # Scale back to original range
                processed_data = corrected * max_val

        elif operation == "Box Blur":
            kernel_size = params.get('kernel_size', 3)
            # Make sure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

            img_float = processed_data.astype(np.float64)
            pad = kernel_size // 2

            # Handle both grayscale (2D) and color (3D) images
            if img_float.ndim == 2:
                # Pad the image with edge values so borders aren't black
                padded = np.pad(img_float, pad, mode='edge')
                result = np.zeros_like(img_float)
                # Slide the kernel across every pixel and average
                for i in range(img_float.shape[0]):
                    for j in range(img_float.shape[1]):
                        region = padded[i:i + kernel_size, j:j + kernel_size]
                        result[i, j] = np.mean(region)
                processed_data = result
            elif img_float.ndim == 3:
                # Apply blur to each color channel separately
                result = np.zeros_like(img_float)
                for c in range(img_float.shape[2]):
                    channel = img_float[:, :, c]
                    padded = np.pad(channel, pad, mode='edge')
                    for i in range(channel.shape[0]):
                        for j in range(channel.shape[1]):
                            region = padded[i:i + kernel_size, j:j + kernel_size]
                            result[i, j, c] = np.mean(region)
                processed_data = result

        # Ensure output dtype matches input
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data