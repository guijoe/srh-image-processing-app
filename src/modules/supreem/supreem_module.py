import numpy as np
import imageio
from scipy.ndimage import convolve, median_filter, map_coordinates

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QStackedWidget, QPushButton
from PySide6.QtCore import Qt, Signal

from modules.i_image_module import IImageModule

# --- Parameter Widgets ---

class BaseParameterWidget(QWidget):
    def get_parameters(self) -> dict:
        raise NotImplementedError

class NoParameterWidget(BaseParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_parameters(self) -> dict:
        return {}

class GaussianBlurParameterWidget(BaseParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Sigma (Standard Deviation):"))
        self.sigma_input = QDoubleSpinBox()
        self.sigma_input.setRange(0.1, 25.0)
        self.sigma_input.setValue(1.0)
        layout.addWidget(self.sigma_input)
        layout.addStretch()

    def get_parameters(self) -> dict:
        return {'sigma': self.sigma_input.value()}

class PencilSketchParameterWidget(BaseParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Blur Radius (3-99):"))
        self.radius_input = QDoubleSpinBox()
        self.radius_input.setRange(3, 99)
        self.radius_input.setValue(15)
        layout.addWidget(self.radius_input)
        layout.addStretch()

    def get_parameters(self) -> dict:
        return {'radius': int(self.radius_input.value())}

class ManualIntensityParameterWidget(BaseParameterWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Contrast Factor (0.1 to 5.0):"))
        self.contrast_input = QDoubleSpinBox()
        self.contrast_input.setRange(0.1, 5.0)
        self.contrast_input.setValue(1.0)
        layout.addWidget(self.contrast_input)
        layout.addWidget(QLabel("Brightness Offset (-255 to 255):"))
        self.brightness_input = QDoubleSpinBox()
        self.brightness_input.setRange(-255.0, 255.0)
        self.brightness_input.setValue(0.0)
        layout.addWidget(self.brightness_input)
        layout.addStretch()

    def get_parameters(self) -> dict:
        return {'contrast_factor': self.contrast_input.value(), 'brightness_offset': self.brightness_input.value()}

# --- Control Widget ---

class SupreemControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.parameter_widgets = {}
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Supreem Control Panel</h3>"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)
        self.parameter_stack = QStackedWidget()
        layout.addWidget(self.parameter_stack)

        operations = {
            "Gaussian Blur": GaussianBlurParameterWidget,
            "Sobel Edge Detect": NoParameterWidget,
            "Astro Negative": NoParameterWidget,
            "Space Debris Cleaner": NoParameterWidget,
            "Pencil Sketch": PencilSketchParameterWidget,
            "Manual Intensity": ManualIntensityParameterWidget
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.parameter_stack.addWidget(widget)
            self.parameter_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        self.apply_button.clicked.connect(self._handle_apply)
        layout.addWidget(self.apply_button)
        self.operation_selector.currentTextChanged.connect(
            lambda text: self.parameter_stack.setCurrentWidget(self.parameter_widgets[text])
        )

    def _handle_apply(self):
        name = self.operation_selector.currentText()
        params = self.parameter_widgets[name].get_parameters()
        params['operation'] = name
        self.process_requested.emit(params)

# --- Main Module ---

class SupreemImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self.controls_widget = None

    def get_name(self) -> str: return "Supreem Module"
    def get_supported_formats(self) -> list[str]: return ["png", "jpg", "jpeg", "bmp", "tiff"]

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            return True, image_data, {'name': file_path.split('/')[-1]}, None
        except: return False, None, {}, None

    def create_control_widget(self, parent=None, module_manager=None):
        if not self.controls_widget:
            self.controls_widget = SupreemControlsWidget(module_manager, parent)
            self.controls_widget.process_requested.connect(module_manager.apply_processing_to_current_image)
        return self.controls_widget

    def process_image(self, image_data: np.ndarray, metadata: dict, parameters: dict) -> np.ndarray:
        processed_data = image_data.astype(float)
        operation = parameters.get('operation')
        is_color = processed_data.ndim == 3 and processed_data.shape[2] >= 3
        grayscale_base = np.dot(processed_data[..., :3], [0.299, 0.587, 0.114]) if is_color else processed_data

        if operation == "Gaussian Blur":
            sigma = parameters.get('sigma', 1.0)
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            axis = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
            kernel = np.exp(-axis**2 / (2 * sigma**2)); kernel /= kernel.sum()
            if is_color:
                for i in range(processed_data.shape[2]):
                    processed_data[..., i] = convolve(convolve(processed_data[..., i], kernel[None, :], mode='reflect'), kernel[:, None], mode='reflect')
            else:
                processed_data = convolve(convolve(processed_data, kernel[None, :], mode='reflect'), kernel[:, None], mode='reflect')

        elif operation == "Sobel Edge Detect":
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            gradient = np.sqrt(convolve(grayscale_base, kernel_x, mode='reflect')**2 + convolve(grayscale_base, kernel_y, mode='reflect')**2)
            processed_data = np.stack([gradient]*image_data.shape[2], axis=-1) if is_color else gradient

        elif operation == "Astro Negative":
            processed_data[..., :3] = 255 - processed_data[..., :3] if is_color else 255 - processed_data

        elif operation == "Space Debris Cleaner":
            if is_color:
                for i in range(3): processed_data[..., i] = median_filter(processed_data[..., i], size=3)
            else: processed_data = median_filter(processed_data, size=3)

        elif operation == "Pencil Sketch":
            radius = parameters.get('radius', 5)
            inverted = 255 - grayscale_base
            blurred = convolve(inverted, np.ones((radius, radius)) / (radius * radius), mode='reflect')
            sketch = np.clip(np.divide(grayscale_base, (255 - blurred + 0.01)) * 255, 0, 255)
            processed_data = np.stack([sketch]*image_data.shape[2], axis=-1) if is_color else sketch

        elif operation == "Manual Intensity":
            processed_data[..., :3] = (processed_data[..., :3] * parameters.get('contrast_factor', 1.0)) + parameters.get('brightness_offset', 0.0)
            processed_data = np.clip(processed_data, 0, 255)

        return processed_data.astype(image_data.dtype)