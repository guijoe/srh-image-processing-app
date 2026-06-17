from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio
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


class MangaEffectParamsWidget(BaseParamsWidget):
    """A widget for Manga/JoJo-inspired effect parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Poster Levels:"))
        self.levels_spinbox = QDoubleSpinBox()
        self.levels_spinbox.setMinimum(2.0)
        self.levels_spinbox.setMaximum(12.0)
        self.levels_spinbox.setValue(5.0)
        self.levels_spinbox.setSingleStep(1.0)
        layout.addWidget(self.levels_spinbox)

        layout.addWidget(QLabel("Edge Strength:"))
        self.edge_spinbox = QDoubleSpinBox()
        self.edge_spinbox.setMinimum(0.0)
        self.edge_spinbox.setMaximum(5.0)
        self.edge_spinbox.setValue(2.0)
        self.edge_spinbox.setSingleStep(0.1)
        layout.addWidget(self.edge_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "levels": int(self.levels_spinbox.value()),
            "edge_strength": self.edge_spinbox.value()
        }


class VintageCartoonParamsWidget(BaseParamsWidget):
    """Parameters for Vintage Cartoon effect."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Brightness Threshold:"))

        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setMinimum(0.0)
        self.threshold_spinbox.setMaximum(1.0)
        self.threshold_spinbox.setValue(0.5)
        self.threshold_spinbox.setSingleStep(0.05)

        layout.addWidget(self.threshold_spinbox)

        layout.addWidget(QLabel("Edge Threshold:"))

        self.edge_spinbox = QDoubleSpinBox()
        self.edge_spinbox.setMinimum(0.01)
        self.edge_spinbox.setMaximum(1.0)
        self.edge_spinbox.setValue(0.08)
        self.edge_spinbox.setSingleStep(0.01)

        layout.addWidget(self.edge_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "brightness_threshold": self.threshold_spinbox.value(),
            "edge_threshold": self.edge_spinbox.value(),
        }


class VibrationBlurParamsWidget(BaseParamsWidget):
    """Parameters for Vibrating Blur effect."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Blur Radius / Shake Distance:"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(1.0)
        self.radius_spinbox.setMaximum(30.0)
        self.radius_spinbox.setValue(6.0)
        self.radius_spinbox.setSingleStep(1.0)
        layout.addWidget(self.radius_spinbox)

        layout.addWidget(QLabel("Intensity:"))
        self.intensity_spinbox = QDoubleSpinBox()
        self.intensity_spinbox.setMinimum(0.1)
        self.intensity_spinbox.setMaximum(3.0)
        self.intensity_spinbox.setValue(1.0)
        self.intensity_spinbox.setSingleStep(0.1)
        layout.addWidget(self.intensity_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "radius": int(self.radius_spinbox.value()),
            "intensity": self.intensity_spinbox.value()
        }


# --- Operations mapping (module-level, not inside a class) ---

OPERATIONS = {
    "Gaussian Blur": GaussianParamsWidget,
    "Sobel Edge Detect": NoParamsWidget,
    "Power Law (Gamma)": PowerLawParamsWidget,
    "Convolution": ConvolutionParamsWidget,
    "Manga Poster Effect": MangaEffectParamsWidget,
    "Vintage Cartoon": VintageCartoonParamsWidget,
    "Vibration Blur": VibrationBlurParamsWidget,
}


# --- Control Widget ---

class KennethControlsWidget(QWidget):
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

        for name, widget_class in OPERATIONS.items():
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


# --- Image Module ---

class KennethImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Kenneth Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = KennethControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 2:
                pass
            elif image_data.ndim == 3 and image_data.shape[2] not in [3, 4]:
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
            processed_data = skimage.filters.gaussian(
                processed_data.astype(float), sigma=sigma, preserve_range=True
            )

        elif operation == "Median Filter":
            filter_size = params.get('filter_size', 3)
            if filter_size <= 1:
                return processed_data
            footprint = skimage.morphology.disk(int(filter_size / 2))
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                channels = [
                    skimage.filters.median(processed_data[:, :, i], footprint=footprint)
                    for i in range(processed_data.shape[2])
                ]
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = skimage.filters.median(processed_data, footprint=footprint)

        elif operation == "Sobel Edge Detect":
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:, :, :3])
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
                    channels = [
                        convolve(input_float[:, :, i], kernel, mode='reflect')
                        for i in range(input_float.shape[2])
                    ]
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = convolve(input_float, kernel, mode='reflect')

        elif operation == "Manga Poster Effect":
            img_float = processed_data.astype(float)
            levels = params.get("levels", 5)
            edge_strength = params.get("edge_strength", 2.0)

            # Normalize to 0-255
            img_min = np.min(img_float)
            img_max = np.max(img_float)
            if img_max != img_min:
                img_float = (img_float - img_min) * (255.0 / (img_max - img_min))

            # Posterization
            step = 256.0 / levels
            posterized = np.floor(img_float / step) * step

            # Grayscale for edge detection
            if img_float.ndim == 3 and img_float.shape[2] in [3, 4]:
                gray = rgb2gray(img_float[:, :, :3] / 255.0)
            else:
                gray = img_float / 255.0

            # Sobel edges -> dark ink lines
            edges = skimage.filters.sobel(gray)
            edge_mask = 1.0 - np.clip(edges * edge_strength, 0, 1)

            # Apply edge mask
            if posterized.ndim == 3 and posterized.shape[2] in [3, 4]:
                processed_data = posterized.copy()
                processed_data[:, :, :3] = posterized[:, :, :3] * edge_mask[:, :, np.newaxis]
            else:
                processed_data = posterized * edge_mask

            processed_data = np.clip(processed_data, 0, 255)
        
        elif operation == "Vintage Cartoon":
            brightness_threshold = params.get("brightness_threshold", 0.5)
            edge_threshold = params.get("edge_threshold", 0.08)

            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                gray = rgb2gray(processed_data[:, :, :3])
            else:
                gray = processed_data.astype(float)

            gray = gray.astype(float)

            gray_min = np.min(gray)
            gray_max = np.max(gray)

            if gray_max != gray_min:
                gray = (gray - gray_min) / (gray_max - gray_min)
            else:
                return processed_data

            cartoon = np.where(gray > brightness_threshold, 255, 0).astype(np.uint8)

            edges = skimage.filters.sobel(gray)
            cartoon[edges > edge_threshold] = 0

            # Prevent completely black or completely white image
            if cartoon.min() == cartoon.max():
                cartoon[0, 0] = 0
                cartoon[-1, -1] = 255

                if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                            cartoon_rgb = np.stack([cartoon, cartoon, cartoon], axis=-1)

            if image_data.shape[2] == 4:
                    alpha = image_data[:, :, 3]
                    cartoon_rgb = np.dstack([cartoon_rgb, alpha])

                    processed_data = cartoon_rgb
            else:
                processed_data = cartoon

        elif operation == "Vibrating Blur":
            radius = params.get("radius", 10)

            img_float = processed_data.astype(float)

            shifts = [
                (0, 0),
                (radius, 0),
                (-radius, 0),
                (0, radius),
                (0, -radius),
                (radius, radius),
                (-radius, -radius),
                (radius, -radius),
                (-radius, radius),
            ]

            blurred = np.zeros_like(img_float, dtype=float)

            for dy, dx in shifts:
                shifted = np.roll(img_float, shift=(dy, dx), axis=(0, 1))
                blurred += shifted

            processed_data = blurred / len(shifts)
            processed_data = np.clip(processed_data, 0, 255)

        # Restore original dtype
        processed_data = processed_data.astype(image_data.dtype)
        return processed_data
