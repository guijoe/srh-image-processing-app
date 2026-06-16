from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout, QSpinBox
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule

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

class PowerLawParamsWidget(BaseParamsWidget):
    """A widget for Power Law (Gamma) Transformation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Gamma:"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(5.0)
        self.gamma_spinbox.setValue(1.0)
        self.gamma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.gamma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'gamma': self.gamma_spinbox.value()}

class ConvolutionParamsWidget(BaseParamsWidget):
    """A widget for defining a 3x3 convolution kernel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("3x3 Kernel:"))
        
        grid_layout = QGridLayout()
        self.kernel_inputs = []
        for r in range(3):
            row_inputs = []
            for c in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setMinimum(-100.0)
                spinbox.setMaximum(100.0)
                spinbox.setValue(0.0)
                if r == 1 and c == 1:
                    spinbox.setValue(1.0)
                grid_layout.addWidget(spinbox, r, c)
                row_inputs.append(spinbox)
            self.kernel_inputs.append(row_inputs)
        layout.addLayout(grid_layout)

    def get_params(self) -> dict:
        kernel = np.array([[spinbox.value() for spinbox in row] for row in self.kernel_inputs])
        return {'kernel': kernel}


class SketchParamsWidget(BaseParamsWidget):
    """A widget for Sketch filter parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Blur Kernel Size (odd number):"))
        self.blur_size_spinbox = QSpinBox()
        self.blur_size_spinbox.setMinimum(3)
        self.blur_size_spinbox.setMaximum(51)
        self.blur_size_spinbox.setValue(21)
        self.blur_size_spinbox.setSingleStep(2)  # Keep it odd
        layout.addWidget(self.blur_size_spinbox)

        layout.addWidget(QLabel("Intensity (scale factor):"))
        self.intensity_spinbox = QDoubleSpinBox()
        self.intensity_spinbox.setMinimum(100.0)
        self.intensity_spinbox.setMaximum(512.0)
        self.intensity_spinbox.setValue(256.0)
        self.intensity_spinbox.setSingleStep(16.0)
        layout.addWidget(self.intensity_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        blur_size = self.blur_size_spinbox.value()
        # Ensure kernel size is odd
        if blur_size % 2 == 0:
            blur_size += 1
        return {
            'blur_size': blur_size,
            'intensity': self.intensity_spinbox.value()
        }


class EmbossParamsWidget(BaseParamsWidget):
    """A widget for Emboss filter parameters."""
    DIRECTIONS = {
        "Top-Left":     np.array([[-2, -1,  0], [-1,  1,  1], [ 0,  1,  2]]),
        "Top-Right":    np.array([[ 0, -1, -2], [ 1,  1, -1], [ 2,  1,  0]]),
        "Bottom-Left":  np.array([[ 0,  1,  2], [-1,  1,  1], [-2, -1,  0]]),
        "Bottom-Right": np.array([[ 2,  1,  0], [ 1,  1, -1], [ 0, -1, -2]]),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Light Direction:"))
        self.direction_combo = QComboBox()
        for direction in self.DIRECTIONS:
            self.direction_combo.addItem(direction)
        layout.addWidget(self.direction_combo)

        layout.addWidget(QLabel("Depth (brightness offset):"))
        self.depth_spinbox = QSpinBox()
        self.depth_spinbox.setMinimum(0)
        self.depth_spinbox.setMaximum(255)
        self.depth_spinbox.setValue(128)
        self.depth_spinbox.setSingleStep(8)
        layout.addWidget(self.depth_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        direction = self.direction_combo.currentText()
        return {
            'kernel': self.DIRECTIONS[direction],
            'depth': self.depth_spinbox.value()
        }

class ContrastStretchingParamsWidget(BaseParamsWidget):
    """A widget for Contrast Stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Input for the new minimum value
        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setMinimum(0.0)
        self.min_spinbox.setMaximum(255.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)

        # Input for the new maximum value
        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setMinimum(0.0)
        self.max_spinbox.setMaximum(255.0)
        self.max_spinbox.setValue(255.0)
        layout.addWidget(self.max_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'new_min': self.min_spinbox.value(),
            'new_max': self.max_spinbox.value()
        }

# Define a custom control widget
class NizarControlsWidget(QWidget):
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

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Sketch": SketchParamsWidget,
            "Emboss": EmbossParamsWidget,
            "Contrast Stretching": ContrastStretchingParamsWidget,
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

class NizarImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Nizar Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = NizarControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                pass
            elif image_data.ndim == 2:
                image_data = image_data[np.newaxis, :]
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')

        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)

        elif operation == "Median Filter":
            filter_size = params.get('filter_size', 3)
            if filter_size <= 1:
                return processed_data
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                channels = []
                for i in range(processed_data.shape[2]):
                    channels.append(skimage.filters.median(processed_data[:,:,i], footprint=skimage.morphology.disk(int(filter_size/2))))
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = skimage.filters.median(processed_data, footprint=skimage.morphology.disk(int(filter_size/2)))

        elif operation == "Sobel Edge Detect":
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:,:,:3])
            else:
                grayscale_img = processed_data
            processed_data = skimage.filters.sobel(grayscale_img)

        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            input_float = processed_data.astype(float)
            max_val = np.max(input_float)
            if max_val > 0:
                normalized = input_float / max_val
                gamma_corrected = np.power(normalized, gamma)
                processed_data = gamma_corrected * max_val

        elif operation == "Convolution":
            kernel = params.get('kernel')
            if kernel is not None:
                input_float = processed_data.astype(float)
                if input_float.ndim == 3 and input_float.shape[2] in [3, 4]:
                    channels = []
                    for i in range(input_float.shape[2]):
                        channels.append(convolve(input_float[:,:,i], kernel, mode='reflect'))
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = convolve(input_float, kernel, mode='reflect')

        elif operation == "Sketch":
            blur_size = params.get('blur_size', 21)
            intensity = params.get('intensity', 256.0)

            # Convert to grayscale for the sketch effect
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                gray = rgb2gray(processed_data[:,:,:3])
                # rgb2gray returns [0,1] float; scale to [0,255]
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = processed_data.astype(np.uint8)

            # Invert → blur → invert → divide (dodge blend)
            inverted = 255 - gray
            blurred = skimage.filters.gaussian(inverted.astype(float), sigma=blur_size / 6.0, preserve_range=True)
            inverted_blurred = 255 - blurred
            # Avoid division by zero
            inverted_blurred = np.clip(inverted_blurred, 1, 255)
            sketch = np.clip(gray.astype(float) / inverted_blurred * intensity, 0, 255).astype(np.uint8)

            # If input was colour, return a 3-channel grayscale so downstream stays consistent
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                processed_data = np.stack([sketch, sketch, sketch], axis=-1)
            else:
                processed_data = sketch

        elif operation == "Emboss":
            kernel = params.get('kernel')
            depth  = params.get('depth', 128)

            if kernel is not None:
                input_float = processed_data.astype(float)

                if input_float.ndim == 3 and input_float.shape[2] in [3, 4]:
                    channels = []
                    for i in range(input_float.shape[2]):
                        channels.append(convolve(input_float[:,:,i], kernel, mode='reflect'))
                    embossed = np.stack(channels, axis=-1)
                else:
                    embossed = convolve(input_float, kernel, mode='reflect')

                # Shift mid-point so neutral areas appear as mid-gray
                processed_data = np.clip(embossed + depth, 0, 255)
        
        elif operation == "Contrast Stretching":
            # Ensure we are working with a floating point image for calculations
            img_float = processed_data.astype(float)

            # Get parameters from the UI
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)

            # Get current image intensity range
            current_min = np.min(img_float)
            current_max = np.max(img_float)

            # Avoid division by zero if the image is flat
            if current_max == current_min:
                return processed_data # Return original image

            # Apply the linear stretching formula
            processed_data = (img_float - current_min) * \
                             ((new_max - new_min) / (current_max - current_min)) + new_min

            # Clip values to be safe, though the formula should handle it
            processed_data = np.clip(processed_data, new_min, new_max)
            
        # Restore original dtype
        processed_data = processed_data.astype(image_data.dtype)
        return processed_data
