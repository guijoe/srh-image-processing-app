from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
from PySide6.QtCore import Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# --- Parameter Widgets for Different Operations ---
class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
    """A placeholder widget for operations with no parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class GaussianParamsWidget(BaseParamsWidget):
    """A widget for Gaussian blur parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Sigma (Standard Deviation):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(0.1)
        self.sigma_spinbox.setMaximum(25.0)
        self.sigma_spinbox.setValue(1.0)
        self.sigma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.sigma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'sigma': self.sigma_spinbox.value()}

class ContrastStretchParamsWidget(BaseParamsWidget):
    """A widget for contrast stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Lower Percentile:"))
        self.low_spinbox = QDoubleSpinBox()
        self.low_spinbox.setMinimum(0.0)
        self.low_spinbox.setMaximum(49.0)
        self.low_spinbox.setValue(2.0)
        self.low_spinbox.setSingleStep(0.5)
        layout.addWidget(self.low_spinbox)

        layout.addWidget(QLabel("Upper Percentile:"))
        self.high_spinbox = QDoubleSpinBox()
        self.high_spinbox.setMinimum(51.0)
        self.high_spinbox.setMaximum(100.0)
        self.high_spinbox.setValue(98.0)
        self.high_spinbox.setSingleStep(0.5)
        layout.addWidget(self.high_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'low_percentile': self.low_spinbox.value(),
            'high_percentile': self.high_spinbox.value(),
        }

def _dtype_limits(image: np.ndarray) -> tuple[float, float]:
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        return float(info.min), float(info.max)
    return float(np.nanmin(image)), float(np.nanmax(image))

def _restore_dtype(image: np.ndarray, original: np.ndarray) -> np.ndarray:
    low, high = _dtype_limits(original)
    if high <= low:
        return image.astype(original.dtype)
    return np.clip(image, low, high).astype(original.dtype)

def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[:, :, :3].astype(float)
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return image.astype(float)

def _gaussian_kernel(sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.1)
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

def _convolve_1d_reflect(channel: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * channel.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(channel, pad_width, mode='reflect')
    result = np.zeros_like(channel, dtype=float)

    for index, weight in enumerate(kernel):
        slices = [slice(None)] * channel.ndim
        slices[axis] = slice(index, index + channel.shape[axis])
        result += weight * padded[tuple(slices)]

    return result

def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma)
    image_float = image.astype(float)

    if image_float.ndim == 3 and image_float.shape[2] in [3, 4]:
        channels = []
        for channel_index in range(image_float.shape[2]):
            channel = image_float[:, :, channel_index]
            blurred = _convolve_1d_reflect(channel, kernel, axis=0)
            blurred = _convolve_1d_reflect(blurred, kernel, axis=1)
            channels.append(blurred)
        return np.stack(channels, axis=-1)

    blurred = _convolve_1d_reflect(image_float, kernel, axis=0)
    return _convolve_1d_reflect(blurred, kernel, axis=1)

def _contrast_stretch(image: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    image_float = image.astype(float)
    output_low, output_high = _dtype_limits(image)

    if np.issubdtype(image.dtype, np.integer):
        output_low, output_high = _dtype_limits(image)
    else:
        output_low, output_high = 0.0, 1.0

    if image_float.ndim == 3 and image_float.shape[2] == 4:
        alpha = image_float[:, :, 3:4]
        image_float = image_float[:, :, :3]
    else:
        alpha = None

    low_value = np.percentile(image_float, low_percentile)
    high_value = np.percentile(image_float, high_percentile)
    if high_value <= low_value:
        stretched = image_float.copy()
    else:
        stretched = (image_float - low_value) / (high_value - low_value)
        stretched = stretched * (output_high - output_low) + output_low

    stretched = np.clip(stretched, output_low, output_high)
    if alpha is not None:
        stretched = np.concatenate([stretched, alpha], axis=2)
    return stretched

def _sketch_filter(image: np.ndarray) -> np.ndarray:
    gray = _rgb_to_gray(image)
    input_low, input_high = _dtype_limits(image)
    if input_high > input_low:
        gray_255 = (gray - input_low) / (input_high - input_low) * 255.0
    else:
        gray_255 = gray.copy()

    inverted = 255.0 - gray_255
    blurred = _gaussian_blur(inverted, sigma=6.0)
    sketch_255 = gray_255 * 255.0 / (255.0 - blurred + 1.0)
    sketch_255 = np.clip(sketch_255, 0, 255)

    if input_high > input_low:
        sketch = sketch_255 / 255.0 * (input_high - input_low) + input_low
    else:
        sketch = sketch_255

    if image.ndim == 3 and image.shape[2] in [3, 4]:
        sketch_rgb = np.stack([sketch, sketch, sketch], axis=-1)
        if image.shape[2] == 4:
            sketch_rgb = np.concatenate([sketch_rgb, image[:, :, 3:4].astype(float)], axis=2)
        return sketch_rgb

    return sketch

# Define a custom control widget
class ayush_muleyControlsWidget(QWidget):
    # Signal to request processing from the module manager
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

        # Stacked widget to hold the parameter UIs
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Define operations and their corresponding parameter widgets
        operations = {
            "Gaussian Blur": GaussianParamsWidget,
            "Contrast Stretching": ContrastStretchParamsWidget,
            "Sketch Filter": NoParamsWidget,
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
        params['operation'] = operation_name # Add operation name to params
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])

class ayush_muleyImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "ayush_muley Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = ayush_muleyControlsWidget(module_manager, parent)
            # The widget's signal is connected to the module's handler
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        # Here, the module needs a way to trigger processing in the main app
        # The control widget now has a valid reference to the module manager
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            # Ensure 2D images are correctly shaped (e.g., handle grayscale vs RGB)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]: # RGB or RGBA
                # napari handles this well, but for processing, sometimes a single channel is needed
                pass
            elif image_data.ndim == 2: # Grayscale
                image_data = image_data[np.newaxis, :] # Add a channel dimension for consistency if desired
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {'name': file_path.split('/')[-1]}
            # Add more metadata: original_shape, file_size, etc.
            return True, image_data, metadata, None # Session ID generated by store
        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        operation = params.get('operation')

        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            processed_data = _gaussian_blur(image_data, sigma)
        elif operation == "Contrast Stretching":
            low_percentile = params.get('low_percentile', 2.0)
            high_percentile = params.get('high_percentile', 98.0)
            processed_data = _contrast_stretch(image_data, low_percentile, high_percentile)
        elif operation == "Sketch Filter":
            processed_data = _sketch_filter(image_data)
        else:
            processed_data = image_data.copy()

        return _restore_dtype(processed_data, image_data)
