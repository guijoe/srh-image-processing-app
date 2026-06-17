from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
#mport skimage.filters
#import skimage.morphology
#from skimage.color import rgb2gray
#from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# --- Parameter Widgets for Different Operations ---

# In your_module.py, add this class near the other *ParamsWidget classes

class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError

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

# Define a custom control widget
class SupreemControlsWidget(QWidget):
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
            "Contrast Stretching": ContrastStretchingParamsWidget
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

class SupreemImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Supreem Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = SupreemControlsWidget(module_manager, parent)
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
        # Work with float for precision
        processed_data = image_data.astype(np.float64)
        operation = params.get('operation')

        if operation == "Gaussian Blur":
            # Using a simple box blur approach with numpy as a pure numpy approximation
            sigma = params.get('sigma', 1.0)
            size = int(2 * np.ceil(2 * sigma) + 1)
            x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
            g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
            kernel = g / g.sum()
            # Padding to handle edges
            pad = size // 2
            padded = np.pad(processed_data, ((pad, pad), (pad, pad), (0, 0)) if processed_data.ndim == 3 else ((pad, pad), (pad, pad)), mode='reflect')
            output = np.zeros_like(processed_data)
            for i in range(processed_data.shape[0]):
                for j in range(processed_data.shape[1]):
                    region = padded[i:i+size, j:j+size]
                    if processed_data.ndim == 3:
                        output[i, j] = np.sum(region * kernel[..., np.newaxis], axis=(0, 1))
                    else:
                        output[i, j] = np.sum(region * kernel)
            processed_data = output

        elif operation == "Sobel Edge Detect":
            # Pure numpy gradient computation
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            # Simple grayscale conversion if RGB
            if processed_data.ndim == 3:
                processed_data = np.dot(processed_data[..., :3], [0.299, 0.587, 0.114])
            
            from numpy.lib.stride_tricks import sliding_window_view
            padded = np.pad(processed_data, 1, mode='reflect')
            windows = sliding_window_view(padded, (3, 3))
            Gx = np.sum(windows * Kx, axis=(2, 3))
            Gy = np.sum(windows * Ky, axis=(2, 3))
            processed_data = np.sqrt(Gx**2 + Gy**2)

        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            max_val = np.max(processed_data)
            if max_val > 0:
                processed_data = np.power(processed_data / max_val, gamma) * max_val

        elif operation == "Contrast Stretching":
            new_min, new_max = params.get('new_min', 0.0), params.get('new_max', 255.0)
            c_min, c_max = np.min(processed_data), np.max(processed_data)
            if c_max != c_min:
                processed_data = (processed_data - c_min) * ((new_max - new_min) / (c_max - c_min)) + new_min
            processed_data = np.clip(processed_data, new_min, new_max)

        return processed_data.astype(image_data.dtype)