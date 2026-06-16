from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget,
    QDoubleSpinBox, QSpinBox, QGridLayout
)
from PySide6.QtCore import Signal
import numpy as np
import imageio.v3 as iio

from modules.i_image_module import IImageModule


# ============================================================
# NumPy-only image processing algorithms
# ============================================================

def _as_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64, copy=False)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def _split_alpha(image: np.ndarray):
    """Return RGB/gray image and alpha channel if present."""
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    return image, None


def _restore_alpha(image: np.ndarray, alpha):
    if alpha is not None:
        return np.dstack([image, alpha])
    return image


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale using luminance weights."""
    base, _ = _split_alpha(image)
    if base.ndim == 2:
        return base
    if base.ndim == 3 and base.shape[2] >= 3:
        r = base[:, :, 0].astype(np.float64)
        g = base[:, :, 1].astype(np.float64)
        b = base[:, :, 2].astype(np.float64)
        return 0.299 * r + 0.587 * g + 0.114 * b
    return base.astype(np.float64)


def _apply_per_channel(image: np.ndarray, function):
    """Apply a 2-D function to grayscale or every RGB channel."""
    base, alpha = _split_alpha(image)
    if base.ndim == 2:
        result = function(base)
    elif base.ndim == 3:
        channels = [function(base[:, :, c]) for c in range(base.shape[2])]
        result = np.dstack(channels)
    else:
        result = function(base)
    return _restore_alpha(_to_uint8(result), alpha)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Objective enhancement algorithm.
    It redistributes intensity values using histogram + cumulative distribution function.
    """
    def equalize_2d(channel):
        channel_u8 = _to_uint8(channel)
        hist = np.bincount(channel_u8.ravel(), minlength=256)
        cdf = hist.cumsum()
        nonzero = cdf[cdf > 0]
        if len(nonzero) == 0:
            return channel_u8
        cdf_min = nonzero[0]
        total = channel_u8.size
        if total == cdf_min:
            return channel_u8
        lookup = np.round((cdf - cdf_min) / (total - cdf_min) * 255)
        lookup = np.clip(lookup, 0, 255).astype(np.uint8)
        return lookup[channel_u8]
    return _apply_per_channel(image, equalize_2d)


def contrast_stretching(
    image: np.ndarray,
    new_min: float = 0.0,
    new_max: float = 255.0,
) -> np.ndarray:
    """
    Contrast stretching with user-controlled output range.

    Formula:
        out = (pixel - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    If new_min=0 and new_max=255, the image uses the full 8-bit intensity range.
    Smaller output ranges make the image flatter; higher ranges make it brighter.
    """
    new_min = float(np.clip(new_min, 0, 255))
    new_max = float(np.clip(new_max, 0, 255))
    if new_max <= new_min:
        new_max = min(255.0, new_min + 1.0)

    def stretch_2d(channel):
        channel = _as_float(channel)
        old_min = channel.min()
        old_max = channel.max()
        if old_max == old_min:
            return np.full_like(channel, new_min, dtype=np.float64)

        normalized = (channel - old_min) / (old_max - old_min)
        return normalized * (new_max - new_min) + new_min

    return _apply_per_channel(image, stretch_2d)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Power-law intensity transform: output = 255 * (input / 255) ** gamma."""
    gamma = max(float(gamma), 0.01)
    base, alpha = _split_alpha(image)
    normalized = np.clip(_as_float(base) / 255.0, 0, 1)
    corrected = 255.0 * np.power(normalized, gamma)
    return _restore_alpha(_to_uint8(corrected), alpha)


def image_negative(image: np.ndarray) -> np.ndarray:
    """Invert intensity values: output = 255 - input."""
    base, alpha = _split_alpha(image)
    return _restore_alpha(_to_uint8(255 - base), alpha)


def convolve_2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution using NumPy padding and kernel sliding."""
    channel = _as_float(channel)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    output = np.zeros_like(channel, dtype=np.float64)

    # Convolution flips the kernel compared with correlation.
    flipped = np.flipud(np.fliplr(kernel))
    for r in range(kh):
        for c in range(kw):
            output += flipped[r, c] * padded[r:r + channel.shape[0], c:c + channel.shape[1]]
    return output


def custom_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a user-specified 3x3 convolution kernel."""
    return _apply_per_channel(image, lambda ch: convolve_2d(ch, kernel))


def mean_filter(image: np.ndarray) -> np.ndarray:
    """Average smoothing filter implemented as convolution."""
    kernel = np.ones((3, 3), dtype=np.float64) / 9.0
    return custom_convolution(image, kernel)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a normalized 2-D Gaussian kernel."""
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Smooth image using a Gaussian-weighted convolution kernel."""
    sigma = max(float(sigma), 0.1)
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(size, sigma)
    return custom_convolution(image, kernel)


def median_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Remove salt-and-pepper noise by replacing each pixel with local median."""
    size = int(size)
    if size % 2 == 0:
        size += 1
    size = max(size, 3)

    def median_2d(channel):
        channel = _as_float(channel)
        pad = size // 2
        padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="reflect")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        return np.median(windows, axis=(-1, -2))

    return _apply_per_channel(image, median_2d)


def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    """Detect edges using Sobel x and y gradient masks."""
    gray = _to_gray(image)
    gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float64)
    gy_kernel = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float64)
    gx = convolve_2d(gray, gx_kernel)
    gy = convolve_2d(gray, gy_kernel)
    magnitude = np.sqrt(gx**2 + gy**2)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0
    return _to_uint8(magnitude)


def laplacian_sharpening(image: np.ndarray, amount: float = 0.7) -> np.ndarray:
    """Sharpen image by subtracting the Laplacian second derivative response."""
    amount = float(amount)
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float64)
    def sharpen_2d(channel):
        lap = convolve_2d(channel, laplacian_kernel)
        return _as_float(channel) - amount * lap
    return _apply_per_channel(image, sharpen_2d)


# ============================================================
# PySide control widgets
# ============================================================

class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class ContrastStretchParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.new_min = QDoubleSpinBox()
        self.new_min.setMinimum(0.0)
        self.new_min.setMaximum(255.0)
        self.new_min.setDecimals(2)
        self.new_min.setSingleStep(1.0)
        self.new_min.setValue(0.0)
        layout.addWidget(self.new_min)

        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.new_max = QDoubleSpinBox()
        self.new_max.setMinimum(0.0)
        self.new_max.setMaximum(255.0)
        self.new_max.setDecimals(2)
        self.new_max.setSingleStep(1.0)
        self.new_max.setValue(255.0)
        layout.addWidget(self.new_max)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "new_min": self.new_min.value(),
            "new_max": self.new_max.value(),
        }


class GammaParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Gamma:"))
        self.gamma = QDoubleSpinBox()
        self.gamma.setMinimum(0.10)
        self.gamma.setMaximum(5.00)
        self.gamma.setSingleStep(0.10)
        self.gamma.setValue(1.00)
        layout.addWidget(self.gamma)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"gamma": self.gamma.value()}


class GaussianParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Sigma:"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setMinimum(0.10)
        self.sigma.setMaximum(10.00)
        self.sigma.setSingleStep(0.10)
        self.sigma.setValue(1.20)
        layout.addWidget(self.sigma)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"sigma": self.sigma.value()}


class MedianParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Window size:"))
        self.size = QSpinBox()
        self.size.setMinimum(3)
        self.size.setMaximum(15)
        self.size.setSingleStep(2)
        self.size.setValue(3)
        layout.addWidget(self.size)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"size": self.size.value()}


class LaplacianParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Sharpen amount:"))
        self.amount = QDoubleSpinBox()
        self.amount.setMinimum(0.10)
        self.amount.setMaximum(3.00)
        self.amount.setSingleStep(0.10)
        self.amount.setValue(0.70)
        layout.addWidget(self.amount)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"amount": self.amount.value()}


class ConvolutionParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("3x3 Kernel:"))
        grid = QGridLayout()
        default_kernel = [[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]]
        self.inputs = []
        for r in range(3):
            row = []
            for c in range(3):
                box = QDoubleSpinBox()
                box.setMinimum(-20.0)
                box.setMaximum(20.0)
                box.setDecimals(2)
                box.setSingleStep(0.25)
                box.setValue(default_kernel[r][c])
                grid.addWidget(box, r, c)
                row.append(box)
            self.inputs.append(row)
        layout.addLayout(grid)
        layout.addStretch()

    def get_params(self) -> dict:
        kernel = np.array([[box.value() for box in row] for row in self.inputs], dtype=np.float64)
        return {"kernel": kernel}


class KenChegeControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Ken Chege</h3>"))
        layout.addWidget(QLabel("Operation:"))

        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Histogram Equalization": NoParamsWidget,
            "Contrast Stretching": ContrastStretchParamsWidget,
            "Gamma Correction": GammaParamsWidget,
            "Image Negative": NoParamsWidget,
            "Gaussian Blur": GaussianParamsWidget,
            "Mean Filter": NoParamsWidget,
            "Median Filter": MedianParamsWidget,
            "Sobel Edge Detection": NoParamsWidget,
            "Laplacian Sharpening": LaplacianParamsWidget,
            "Convolution": ConvolutionParamsWidget,
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)
        layout.addStretch()

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply_clicked(self):
        operation = self.operation_selector.currentText()
        params = self.param_widgets[operation].get_params()
        params["operation"] = operation
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation: str):
        if operation in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation])


class KenChegeImageModule(IImageModule):
    def __init__(self):
        self._controls_widget = None

    def get_name(self) -> str:
        return "Ken Chege"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = KenChegeControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = iio.imread(file_path)
            if image_data.dtype != np.uint8:
                image_data = _to_uint8(image_data)
            metadata = {
                "name": file_path.split("/")[-1],
                "contrast_limits": (0, 255),
            }
            return True, image_data, metadata, None
        except Exception as exc:
            print(f"Could not load image: {exc}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        operation = params.get("operation")

        if operation == "Histogram Equalization":
            return histogram_equalization(image_data)
        if operation == "Contrast Stretching":
            return contrast_stretching(
                image_data,
                params.get("new_min", 0.0),
                params.get("new_max", 255.0),
            )
        if operation == "Gamma Correction":
            return gamma_correction(image_data, params.get("gamma", 1.0))
        if operation == "Image Negative":
            return image_negative(image_data)
        if operation == "Gaussian Blur":
            return gaussian_blur(image_data, params.get("sigma", 1.2))
        if operation == "Mean Filter":
            return mean_filter(image_data)
        if operation == "Median Filter":
            return median_filter(image_data, params.get("size", 3))
        if operation == "Sobel Edge Detection":
            return sobel_edge_detection(image_data)
        if operation == "Laplacian Sharpening":
            return laplacian_sharpening(image_data, params.get("amount", 0.7))
        if operation == "Convolution":
            return custom_convolution(image_data, params.get("kernel"))

        return image_data.copy()
