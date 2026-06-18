from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
import skimage.filters
from skimage.color import rgb2gray
from scipy.ndimage import convolve

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
                # Set center to 1.0 for an identity-like default
                if r == 1 and c == 1:
                    spinbox.setValue(1.0)
                grid_layout.addWidget(spinbox, r, c)
                row_inputs.append(spinbox)
            self.kernel_inputs.append(row_inputs)
        layout.addLayout(grid_layout)

    def get_params(self) -> dict:
        kernel = np.array([[spinbox.value() for spinbox in row] for row in self.kernel_inputs])
        return {'kernel': kernel}
    
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

class SolarizationParamsWidget(BaseParamsWidget):
    """A widget for Solarization parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Input for the threshold value
        layout.addWidget(QLabel("Threshold (0-255):"))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setMinimum(0.0)
        self.threshold_spinbox.setMaximum(255.0)
        self.threshold_spinbox.setValue(128.0)
        layout.addWidget(self.threshold_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'threshold': self.threshold_spinbox.value()
        }

class DuotoneParamsWidget(BaseParamsWidget):
    """A widget for the Black/Blue Duotone operation. No parameters needed."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Maps shadows to black and highlights to blue.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

# Define a custom control widget
class GreshaControlsWidget(QWidget):
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
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Contrast Stretching": ContrastStretchingParamsWidget,
            "Solarization": SolarizationParamsWidget,
            "Black/Blue Duotone": DuotoneParamsWidget,
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

class GreshaImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "gresha Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = GreshaControlsWidget(module_manager, parent)
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
        processed_data = image_data.copy()

        operation = params.get('operation')

        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            # skimage.filters.gaussian expects float data
            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)
        elif operation == "Sobel Edge Detect":
            # Sobel works on 2D (grayscale) images. Convert if necessary.
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:,:,:3])
            else:
                grayscale_img = processed_data
            
            processed_data = skimage.filters.sobel(grayscale_img)
        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            # Normalize to [0, 1]
            input_float = processed_data.astype(float)
            max_val = np.max(input_float)
            if max_val > 0:
                normalized = input_float / max_val
                # Apply gamma correction
                gamma_corrected = np.power(normalized, gamma)
                # Scale back to original range
                processed_data = gamma_corrected * max_val
        elif operation == "Convolution":
            kernel = params.get('kernel')
            if kernel is not None:
                # Convolve works best on float images
                input_float = processed_data.astype(float)
                if input_float.ndim == 3 and input_float.shape[2] in [3, 4]: # RGB/RGBA
                    channels = []
                    for i in range(input_float.shape[2]):
                        channels.append(convolve(input_float[:,:,i], kernel, mode='reflect'))
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = convolve(input_float, kernel, mode='reflect')
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
        elif operation == "Solarization":
            # Work in float for the comparison/inversion math
            img_float = processed_data.astype(float)

            # Get threshold from the UI
            threshold = params.get('threshold', 128.0)

            # The image's max representable intensity (e.g. 255 for uint8).
            # Falls back to the image's own max if dtype isn't an integer type.
            if np.issubdtype(image_data.dtype, np.integer):
                max_value = float(np.iinfo(image_data.dtype).max)
            else:
                max_value = float(np.max(img_float)) if np.max(img_float) > 0 else 255.0

            # Invert only the pixels above the threshold; leave the rest untouched
            processed_data = np.where(img_float > threshold, max_value - img_float, img_float)

            # Clip values to be safe
            processed_data = np.clip(processed_data, 0, max_value)
        elif operation == "Black/Blue Duotone":
            # Determine the max representable intensity for normalization
            if np.issubdtype(image_data.dtype, np.integer):
                max_value = float(np.iinfo(image_data.dtype).max)
            else:
                max_value = float(np.max(processed_data)) if np.max(processed_data) > 0 else 255.0

            # Convert to a normalized grayscale intensity map in [0, 1]
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                # rgb2gray returns floats already normalized to [0, 1] when input is uint8/float
                gray = rgb2gray(processed_data[:, :, :3].astype(float) / max_value)
            else:
                gray = processed_data.astype(float) / max_value

            # Duotone color endpoints (RGB, 0-255): shadows = black, highlights = blue
            shadow_color = np.array([0.0, 0.0, 0.0])
            highlight_color = np.array([0.0, 0.0, 255.0])

            # Linearly interpolate each channel between shadow and highlight colors
            # gray has shape (H, W); expand to (H, W, 1) to broadcast against the 3 color channels
            gray_expanded = gray[..., np.newaxis]
            duotone = shadow_color * (1 - gray_expanded) + highlight_color * gray_expanded

            # Rescale from 0-255 range back into the image's native intensity range if needed
            processed_data = duotone * (max_value / 255.0)
            processed_data = np.clip(processed_data, 0, max_value)

        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data