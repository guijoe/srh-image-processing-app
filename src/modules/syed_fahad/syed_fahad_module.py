from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# --- Parameter Widgets for Different Operations ---
class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError

class ConvolutionParamsWidget(BaseParamsWidget):
    """A widget for 2D Convolution kernel weights (3x3 default sharpening matrix)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("3x3 Convolution Kernel Matrix:"))
        grid_layout = QGridLayout()
        
        # Default sharpening kernel configuration
        default_kernel = [
            [ 0.0, -1.0,  0.0],
            [-1.0,  5.0, -1.0],
            [ 0.0, -1.0,  0.0]
        ]
        
        self.cells = []
        for row in range(3):
            row_cells = []
            for col in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setMinimum(-50.0)
                spinbox.setMaximum(50.0)
                spinbox.setSingleStep(0.5)
                spinbox.setValue(default_kernel[row][col])
                grid_layout.addWidget(spinbox, row, col)
                row_cells.append(spinbox)
            self.cells.append(row_cells)
            
        layout.addLayout(grid_layout)
        layout.addStretch()

    def get_params(self) -> dict:
        kernel_matrix = np.array([[self.cells[r][c].value() for c in range(3)] for r in range(3)])
        return {
            'kernel': kernel_matrix
        }

class GaussianBlurParamsWidget(BaseParamsWidget):
    """A widget for Gaussian Blur parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Kernel Size (Must be odd):"))
        self.size_spinbox = QDoubleSpinBox()
        self.size_spinbox.setMinimum(3.0)
        self.size_spinbox.setMaximum(25.0)
        self.size_spinbox.setSingleStep(2.0)
        self.size_spinbox.setValue(5.0)
        layout.addWidget(self.size_spinbox)

        layout.addWidget(QLabel("Sigma (Standard Deviation):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(0.1)
        self.sigma_spinbox.setMaximum(10.0)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setValue(1.0)
        layout.addWidget(self.sigma_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        # Ensure size is an odd integer
        size = int(self.size_spinbox.value())
        if size % 2 == 0:
            size += 1
        return {
            'kernel_size': size,
            'sigma': self.sigma_spinbox.value()
        }

class GammaCorrectionParamsWidget(BaseParamsWidget):
    """A widget for Gamma-based Color Correction parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Gamma Value (O = c * s^gamma):"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(10.0)
        self.gamma_spinbox.setSingleStep(0.1)
        self.gamma_spinbox.setValue(1.0)
        layout.addWidget(self.gamma_spinbox)

        layout.addWidget(QLabel("Scaling Constant (c):"))
        self.c_spinbox = QDoubleSpinBox()
        self.c_spinbox.setMinimum(0.1)
        self.c_spinbox.setMaximum(3.0)
        self.c_spinbox.setSingleStep(0.1)
        self.c_spinbox.setValue(1.0)
        layout.addWidget(self.c_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'gamma': self.gamma_spinbox.value(),
            'c': self.c_spinbox.value()
        }

class NeonBloomParamsWidget(BaseParamsWidget):
    """A widget for Neon Bloom Edge Detection parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Edge Threshold (0.0 - 1.0):"))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setMinimum(0.0)
        self.threshold_spinbox.setMaximum(1.0)
        self.threshold_spinbox.setSingleStep(0.05)
        self.threshold_spinbox.setValue(0.15)
        layout.addWidget(self.threshold_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'edge_threshold': self.threshold_spinbox.value()
        }

# Define a custom control widget
class syed_fahadControlsWidget(QWidget):
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
            "2D Convolution": ConvolutionParamsWidget,
            "Gaussian Blur": GaussianBlurParamsWidget,
            "Gamma Correction": GammaCorrectionParamsWidget,
            "Neon Bloom Edge Detection": NeonBloomParamsWidget
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

class syed_fahadImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "syed_fahad Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = syed_fahadControlsWidget(module_manager, parent)
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
                pass #image_data = image_data[:,:,np.newaxis] # Add a channel dimension for consistency if desired
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

        # Unified logic to scale input to float array safely
        is_uint8 = np.issubdtype(image_data.dtype, np.integer)
        max_val = 255.0 if is_uint8 else 1.0
        img_float = processed_data.astype(float) / max_val

        if operation == "2D Convolution":
            kernel = params.get('kernel', np.array([[0,0,0],[0,1,0],[0,0,0]]))
            img_h, img_w = img_float.shape[0], img_float.shape[1]
            kh, kw = kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            
            if img_float.ndim == 3:
                padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
                conv_out = np.zeros_like(img_float)
                for i in range(img_h):
                    for j in range(img_w):
                        roi = padded_img[i:i+kh, j:j+kw, :]
                        conv_out[i, j, :] = np.sum(roi * kernel[:, :, np.newaxis], axis=(0, 1))
            else:
                padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
                conv_out = np.zeros_like(img_float)
                for i in range(img_h):
                    for j in range(img_w):
                        roi = padded_img[i:i+kh, j:j+kw]
                        conv_out[i, j] = np.sum(roi * kernel)

            processed_data = np.clip(conv_out, 0.0, 1.0) * max_val

        elif operation == "Gaussian Blur":
            kernel_size = params.get('kernel_size', 5)
            sigma = params.get('sigma', 1.0)
            
            # Generate mathematical Gaussian distribution matrix
            ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
            x, y = np.meshgrid(ax, ax)
            gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            gaussian_kernel /= np.sum(gaussian_kernel)
            
            img_h, img_w = img_float.shape[0], img_float.shape[1]
            kh, kw = gaussian_kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            
            if img_float.ndim == 3:
                padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
                conv_out = np.zeros_like(img_float)
                for i in range(img_h):
                    for j in range(img_w):
                        roi = padded_img[i:i+kh, j:j+kw, :]
                        conv_out[i, j, :] = np.sum(roi * gaussian_kernel[:, :, np.newaxis], axis=(0, 1))
            else:
                padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
                conv_out = np.zeros_like(img_float)
                for i in range(img_h):
                    for j in range(img_w):
                        roi = padded_img[i:i+kh, j:j+kw]
                        conv_out[i, j] = np.sum(roi * gaussian_kernel)
                        
            processed_data = np.clip(conv_out, 0.0, 1.0) * max_val

        elif operation == "Gamma Correction":
            gamma = params.get('gamma', 1.0)
            c = params.get('c', 1.0)
            
            # Mathematical Power Law: O = c * (s ^ gamma)
            corrected = c * np.power(img_float, gamma)
            processed_data = np.clip(corrected, 0.0, 1.0) * max_val

        elif operation == "Neon Bloom Edge Detection":
            edge_threshold = params.get('edge_threshold', 0.15)
            
            # Laplacian kernel for second-order spatial derivative edge extraction
            laplacian_kernel = np.array([[ 0,  1,  0],
                                         [ 1, -4,  1],
                                         [ 0,  1,  0]])
            
            img_h, img_w = img_float.shape[0], img_float.shape[1]
            kh, kw = laplacian_kernel.shape
            pad_h, pad_w = kh // 2, kw // 2
            
            # Perform edge detection convolution
            if img_float.ndim == 3:
                padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
                edges = np.zeros_like(img_float)
                for i in range(img_h):
                    for j in range(img_w):
                        roi = padded_img[i:i+kh, j:j+kw, :]
                        edges[i, j, :] = np.sum(roi * laplacian_kernel[:, :, np.newaxis], axis=(0, 1))
                # Compress 3-channel edges down to spatial frame intensity mapping
                edge_intensity = np.mean(edges, axis=2)
            else:
                padded_img = np.pad(img_float, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
                edge_intensity = np.zeros_like(img_float)
                for i in range(img_h):
                    for j in range(img_w):
                        roi = padded_img[i:i+kh, j:j+kw]
                        edge_intensity[i, j] = np.abs(np.sum(roi * laplacian_kernel))

            # Normalize negative edges and isolate significant contrast differences
            edge_intensity = np.abs(edge_intensity)
            edge_mask = np.where(edge_intensity > edge_threshold, edge_intensity, 0.0)
            edge_mask = np.clip(edge_mask * 3.0, 0.0, 1.0) # Amplify to establish bloom visibility
            
            # Translate into high-grade CMYK profile: Cyan=0, Magenta=intensity, Yellow=0, Key=(1 - intensity)
            C = np.zeros_like(edge_mask)
            M = edge_mask
            Y = np.zeros_like(edge_mask)
            K = 1.0 - edge_mask
            
            # Convert CMYK profile arrays directly back into standard viewable RGB space channels
            R = (1.0 - C) * (1.0 - K)
            G = (1.0 - M) * (1.0 - K)
            B = (1.0 - Y) * (1.0 - K)
            
            # Package into standard 3D layout shape representation
            neon_img = np.stack([R, G, B], axis=2)
            processed_data = np.clip(neon_img, 0.0, 1.0) * max_val

        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data