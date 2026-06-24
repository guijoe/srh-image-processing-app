from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QDoubleSpinBox, QSpinBox
)
from PySide6.QtCore import Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule


# =============================================================
# Helper functions — pure NumPy only
# =============================================================

def _to_float(arr):
    return arr.astype(np.float32)

def _safe_output(result, original_dtype):
    result = np.nan_to_num(result, nan=0.0, posinf=255.0, neginf=0.0)
    result = np.clip(result, 0, 255)
    return result.astype(original_dtype)

def _split_alpha(image):
    """If RGBA, split into RGB and alpha. Otherwise return image and None."""
    if image.ndim == 3 and image.shape[2] == 4:
        return image[..., :3], image[..., 3:4]
    return image, None

def _restore_alpha(rgb, alpha):
    """Re-attach alpha channel if one was split off."""
    if alpha is not None:
        return np.concatenate([rgb, alpha], axis=2)
    return rgb

def _to_grayscale(rgb_float):
    """Convert (H, W, 3) float image to (H, W) grayscale using standard weights."""
    return 0.299 * rgb_float[..., 0] + 0.587 * rgb_float[..., 1] + 0.114 * rgb_float[..., 2]

def _gaussian_kernel(size, sigma):
    """Build a normalized 2D Gaussian kernel."""
    if size < 1:
        size = 1
    if size % 2 == 0:
        size += 1
    half = size // 2
    xs = np.arange(-half, half + 1, dtype=np.float32)
    ys = np.arange(-half, half + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()

def _convolve_channel(channel, kernel):
    """
    Convolve a single 2D channel (H, W) with a 2D kernel.
    Uses reflect padding and iterates over kernel positions with NumPy slicing.
    """
    h, w = channel.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    result = np.zeros((h, w), dtype=np.float32)

    for i in range(kh):
        for j in range(kw):
            result += padded[i:i + h, j:j + w] * kernel[i, j]

    return result

def _blur_image(img_float, kernel_size, sigma):
    """Apply Gaussian blur to a 2D or 3D float image."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = _gaussian_kernel(kernel_size, sigma)

    if img_float.ndim == 2:
        return _convolve_channel(img_float, kernel)

    # Process each channel separately for RGB
    channels = [_convolve_channel(img_float[..., c], kernel) for c in range(img_float.shape[2])]
    return np.stack(channels, axis=2)

def _equalize_channel(channel_uint8):
    """Histogram equalization on a single uint8 channel."""
    flat = channel_uint8.flatten()
    hist = np.bincount(flat, minlength=256).astype(np.float32)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]
    total = float(flat.size)

    if (total - cdf_min) == 0:
        return channel_uint8.astype(np.float32)

    lut = (cdf - cdf_min) * 255.0 / (total - cdf_min)
    lut = np.clip(lut, 0, 255)
    return lut[flat].reshape(channel_uint8.shape)


# =============================================================
# Parameter Widgets — one class per operation
# =============================================================

class _BaseParams(QWidget):
    def get_params(self):
        raise NotImplementedError


class NoParamsWidget(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lbl = QLabel("No parameters for this operation.")
        lbl.setStyleSheet("font-style: italic; color: gray;")
        lay.addWidget(lbl)
        lay.addStretch()

    def get_params(self):
        return {}


class LogTransformParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Scale (Strength):"))
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.1, 10.0)
        self.scale.setValue(1.0)
        self.scale.setSingleStep(0.1)
        lay.addWidget(self.scale)
        lay.addStretch()

    def get_params(self):
        return {'scale': self.scale.value()}


class GammaParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Gamma:"))
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.1, 5.0)
        self.gamma.setValue(1.0)
        self.gamma.setSingleStep(0.1)
        lay.addWidget(self.gamma)
        lay.addStretch()

    def get_params(self):
        return {'gamma': self.gamma.value()}


class GaussianParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Kernel Size (odd):"))
        self.ksize = QSpinBox()
        self.ksize.setRange(1, 21)
        self.ksize.setValue(5)
        self.ksize.setSingleStep(2)
        lay.addWidget(self.ksize)
        lay.addWidget(QLabel("Sigma:"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 10.0)
        self.sigma.setValue(1.0)
        self.sigma.setSingleStep(0.1)
        lay.addWidget(self.sigma)
        lay.addStretch()

    def get_params(self):
        k = self.ksize.value()
        if k % 2 == 0:
            k += 1
        return {'kernel_size': k, 'sigma': self.sigma.value()}


class SobelParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Threshold (0 - 255):"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 255.0)
        self.threshold.setValue(30.0)
        self.threshold.setSingleStep(5.0)
        lay.addWidget(self.threshold)
        lay.addStretch()

    def get_params(self):
        return {'threshold': self.threshold.value()}


class CLAHEParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Clip Limit:"))
        self.clip = QDoubleSpinBox()
        self.clip.setRange(0.5, 10.0)
        self.clip.setValue(2.0)
        self.clip.setSingleStep(0.5)
        lay.addWidget(self.clip)
        lay.addWidget(QLabel("Tile Size:"))
        self.tile = QSpinBox()
        self.tile.setRange(4, 64)
        self.tile.setValue(8)
        self.tile.setSingleStep(4)
        lay.addWidget(self.tile)
        lay.addStretch()

    def get_params(self):
        return {'clip_limit': self.clip.value(), 'tile_size': self.tile.value()}


class UnsharpParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Kernel Size (odd):"))
        self.ksize = QSpinBox()
        self.ksize.setRange(1, 21)
        self.ksize.setValue(5)
        self.ksize.setSingleStep(2)
        lay.addWidget(self.ksize)
        lay.addWidget(QLabel("Sigma:"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 10.0)
        self.sigma.setValue(1.0)
        self.sigma.setSingleStep(0.1)
        lay.addWidget(self.sigma)
        lay.addWidget(QLabel("Amount:"))
        self.amount = QDoubleSpinBox()
        self.amount.setRange(0.1, 5.0)
        self.amount.setValue(1.5)
        self.amount.setSingleStep(0.1)
        lay.addWidget(self.amount)
        lay.addStretch()

    def get_params(self):
        k = self.ksize.value()
        if k % 2 == 0:
            k += 1
        return {'kernel_size': k, 'sigma': self.sigma.value(), 'amount': self.amount.value()}


class LaplacianParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Strength:"))
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.1, 5.0)
        self.strength.setValue(1.0)
        self.strength.setSingleStep(0.1)
        lay.addWidget(self.strength)
        lay.addStretch()

    def get_params(self):
        return {'strength': self.strength.value()}


class CartoonParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Color Levels:"))
        self.levels = QSpinBox()
        self.levels.setRange(2, 32)
        self.levels.setValue(8)
        lay.addWidget(self.levels)
        lay.addWidget(QLabel("Edge Threshold:"))
        self.edge_thresh = QDoubleSpinBox()
        self.edge_thresh.setRange(0.0, 255.0)
        self.edge_thresh.setValue(50.0)
        self.edge_thresh.setSingleStep(5.0)
        lay.addWidget(self.edge_thresh)
        lay.addWidget(QLabel("Blur Kernel Size (odd):"))
        self.blur_k = QSpinBox()
        self.blur_k.setRange(1, 15)
        self.blur_k.setValue(5)
        self.blur_k.setSingleStep(2)
        lay.addWidget(self.blur_k)
        lay.addStretch()

    def get_params(self):
        k = self.blur_k.value()
        if k % 2 == 0:
            k += 1
        return {
            'color_levels': self.levels.value(),
            'edge_threshold': self.edge_thresh.value(),
            'blur_kernel_size': k,
        }


class TealOrangeParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Intensity (0.0 - 1.0):"))
        self.intensity = QDoubleSpinBox()
        self.intensity.setRange(0.0, 1.0)
        self.intensity.setValue(0.3)
        self.intensity.setSingleStep(0.05)
        lay.addWidget(self.intensity)
        lay.addStretch()

    def get_params(self):
        return {'intensity': self.intensity.value()}


class RetinexParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Sigma:"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.5, 20.0)
        self.sigma.setValue(5.0)
        self.sigma.setSingleStep(0.5)
        lay.addWidget(self.sigma)
        lay.addWidget(QLabel("Gain:"))
        self.gain = QDoubleSpinBox()
        self.gain.setRange(0.1, 5.0)
        self.gain.setValue(1.0)
        self.gain.setSingleStep(0.1)
        lay.addWidget(self.gain)
        lay.addStretch()

    def get_params(self):
        return {'sigma': self.sigma.value(), 'gain': self.gain.value()}


class TiltShiftParams(_BaseParams):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Focus Position (0.0 - 1.0):"))
        self.focus_pos = QDoubleSpinBox()
        self.focus_pos.setRange(0.0, 1.0)
        self.focus_pos.setValue(0.5)
        self.focus_pos.setSingleStep(0.05)
        lay.addWidget(self.focus_pos)
        lay.addWidget(QLabel("Focus Width (0.0 - 1.0):"))
        self.focus_width = QDoubleSpinBox()
        self.focus_width.setRange(0.01, 1.0)
        self.focus_width.setValue(0.2)
        self.focus_width.setSingleStep(0.05)
        lay.addWidget(self.focus_width)
        lay.addWidget(QLabel("Blur Strength:"))
        self.blur_strength = QDoubleSpinBox()
        self.blur_strength.setRange(0.5, 10.0)
        self.blur_strength.setValue(3.0)
        self.blur_strength.setSingleStep(0.5)
        lay.addWidget(self.blur_strength)
        lay.addStretch()

    def get_params(self):
        return {
            'focus_position': self.focus_pos.value(),
            'focus_width': self.focus_width.value(),
            'blur_strength': self.blur_strength.value(),
        }


# =============================================================
# Control Widget
# =============================================================

class FarzamCinematicControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Farzam Cinematic</h3>"))
        layout.addWidget(QLabel("Operation:"))

        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Operations in order — key is the exact string used in process_image
        operations = {
            "Log Transform":              LogTransformParams,
            "Gamma Correction":           GammaParams,
            "Histogram Equalization":     NoParamsWidget,
            "Gaussian Blur":              GaussianParams,
            "Sobel Edge Detection":       SobelParams,
            "CLAHE":                      CLAHEParams,
            "Unsharp Mask":               UnsharpParams,
            "Laplacian Sharpening":       LaplacianParams,
            "Cartoon Effect":             CartoonParams,
            "Teal & Orange Color Grading": TealOrangeParams,
            "Retinex":                    RetinexParams,
            "Tilt-Shift":                 TiltShiftParams,
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
        op_name = self.operation_selector.currentText()
        params = self.param_widgets[op_name].get_params()
        params['operation'] = op_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, op_name):
        if op_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[op_name])


# =============================================================
# Main Module Class
# =============================================================

class FarzamCinematicImageModule(IImageModule):

    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Farzam Cinematic"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "tiff", "tif"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = FarzamCinematicControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        operation = params.get('operation')

        if operation == "Log Transform":
            return self._apply_log_transform(image_data, params)
        elif operation == "Gamma Correction":
            return self._apply_gamma_correction(image_data, params)
        elif operation == "Histogram Equalization":
            return self._apply_histogram_equalization(image_data, params)
        elif operation == "Gaussian Blur":
            return self._apply_gaussian_blur(image_data, params)
        elif operation == "Sobel Edge Detection":
            return self._apply_sobel(image_data, params)
        elif operation == "CLAHE":
            return self._apply_clahe(image_data, params)
        elif operation == "Unsharp Mask":
            return self._apply_unsharp_mask(image_data, params)
        elif operation == "Laplacian Sharpening":
            return self._apply_laplacian_sharpening(image_data, params)
        elif operation == "Cartoon Effect":
            return self._apply_cartoon(image_data, params)
        elif operation == "Teal & Orange Color Grading":
            return self._apply_teal_orange(image_data, params)
        elif operation == "Retinex":
            return self._apply_retinex(image_data, params)
        elif operation == "Tilt-Shift":
            return self._apply_tilt_shift(image_data, params)

        return image_data.copy()

    # ----------------------------------------------------------
    # 1. Log Transform
    # ----------------------------------------------------------
    def _apply_log_transform(self, image_data, params):
        scale = params.get('scale', 1.0)
        original_dtype = image_data.dtype

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        # c normalizes log output so that input=255 maps to 255
        c = 255.0 / np.log1p(255.0)
        log_img = scale * c * np.log1p(img)

        result = _restore_alpha(log_img, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 2. Gamma Correction
    # ----------------------------------------------------------
    def _apply_gamma_correction(self, image_data, params):
        gamma = params.get('gamma', 1.0)
        if gamma <= 0:
            gamma = 1.0
        original_dtype = image_data.dtype

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        # Normalize to [0, 1], apply power, scale back
        normalized = img / 255.0
        corrected = np.power(normalized, gamma) * 255.0

        result = _restore_alpha(corrected, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 3. Histogram Equalization
    # ----------------------------------------------------------
    def _apply_histogram_equalization(self, image_data, params):
        original_dtype = image_data.dtype

        if image_data.ndim == 2:
            img_uint8 = np.clip(image_data, 0, 255).astype(np.uint8)
            equalized = _equalize_channel(img_uint8)
            return _safe_output(equalized, original_dtype)

        # Color image: equalize luminance only to preserve color ratios
        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        luminance = _to_grayscale(img)                              # (H, W)
        lum_uint8 = np.clip(luminance, 0, 255).astype(np.uint8)
        equalized_lum = _equalize_channel(lum_uint8).astype(np.float32)

        # Scale each channel by how much the luminance changed
        ratio = np.where(luminance > 0, equalized_lum / (luminance + 1e-6), 1.0)
        ratio = ratio[..., np.newaxis]                              # (H, W, 1)

        result_rgb = img * ratio
        result = _restore_alpha(result_rgb, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 4. Gaussian Blur
    # ----------------------------------------------------------
    def _apply_gaussian_blur(self, image_data, params):
        kernel_size = params.get('kernel_size', 5)
        sigma = params.get('sigma', 1.0)
        original_dtype = image_data.dtype

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        blurred = _blur_image(img, kernel_size, sigma)

        result = _restore_alpha(blurred, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 5. Sobel Edge Detection
    # ----------------------------------------------------------
    def _apply_sobel(self, image_data, params):
        threshold = params.get('threshold', 30.0)
        original_dtype = image_data.dtype

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        gray = img if img.ndim == 2 else _to_grayscale(img)

        sobel_x = np.array([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1],
        ], dtype=np.float32)

        sobel_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ], dtype=np.float32)

        gx = _convolve_channel(gray, sobel_x)
        gy = _convolve_channel(gray, sobel_y)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Normalize to 0-255
        max_mag = np.max(magnitude)
        if max_mag > 0:
            magnitude = magnitude / max_mag * 255.0

        edges = np.where(magnitude >= threshold, magnitude, 0.0)

        if image_data.ndim == 2:
            return _safe_output(edges, original_dtype)

        # Return as RGB so napari displays it correctly
        edges_rgb = np.stack([edges, edges, edges], axis=2)
        result = _restore_alpha(edges_rgb, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 6. CLAHE
    # ----------------------------------------------------------
    def _apply_clahe(self, image_data, params):
        clip_limit = params.get('clip_limit', 2.0)
        tile_size = max(4, params.get('tile_size', 8))
        original_dtype = image_data.dtype

        if image_data.ndim == 2:
            img_uint8 = np.clip(image_data, 0, 255).astype(np.uint8)
            result = self._clahe_on_channel(img_uint8, clip_limit, tile_size)
            return _safe_output(result, original_dtype)

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        luminance = _to_grayscale(img)
        lum_uint8 = np.clip(luminance, 0, 255).astype(np.uint8)
        equalized_lum = self._clahe_on_channel(lum_uint8, clip_limit, tile_size).astype(np.float32)

        ratio = np.where(luminance > 0, equalized_lum / (luminance + 1e-6), 1.0)
        ratio = ratio[..., np.newaxis]

        result_rgb = img * ratio
        result = _restore_alpha(result_rgb, alpha)
        return _safe_output(result, original_dtype)

    def _clahe_on_channel(self, channel, clip_limit, tile_size):
        """Apply CLAHE tile by tile on a single uint8 channel."""
        h, w = channel.shape
        result = np.zeros((h, w), dtype=np.float32)

        n_tiles_y = max(1, h // tile_size)
        n_tiles_x = max(1, w // tile_size)

        for ty in range(n_tiles_y):
            y_start = ty * tile_size
            y_end = y_start + tile_size if ty < n_tiles_y - 1 else h

            for tx in range(n_tiles_x):
                x_start = tx * tile_size
                x_end = x_start + tile_size if tx < n_tiles_x - 1 else w

                tile = channel[y_start:y_end, x_start:x_end]
                lut = self._clahe_lut(tile, clip_limit)
                result[y_start:y_end, x_start:x_end] = lut[tile]

        return result

    def _clahe_lut(self, tile, clip_limit):
        """Build a clipped and equalized lookup table for one tile."""
        flat = tile.flatten()
        total = float(flat.size)

        hist = np.bincount(flat, minlength=256).astype(np.float32)

        # Clip bins and redistribute the excess evenly
        clip_value = clip_limit * total / 256.0
        excess = np.sum(np.maximum(hist - clip_value, 0.0))
        hist = np.minimum(hist, clip_value)
        hist += excess / 256.0

        cdf = np.cumsum(hist)
        cdf_min = cdf[0]
        cdf_range = cdf[-1] - cdf_min

        if cdf_range == 0:
            return np.arange(256, dtype=np.float32)

        lut = (cdf - cdf_min) * 255.0 / cdf_range
        return np.clip(lut, 0, 255)

    # ----------------------------------------------------------
    # 7. Unsharp Mask
    # ----------------------------------------------------------
    def _apply_unsharp_mask(self, image_data, params):
        kernel_size = params.get('kernel_size', 5)
        sigma = params.get('sigma', 1.0)
        amount = params.get('amount', 1.5)
        original_dtype = image_data.dtype

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        blurred = _blur_image(img, kernel_size, sigma)

        # Sharpened = original + amount * (original - blurred)
        mask = img - blurred
        sharpened = img + amount * mask

        result = _restore_alpha(sharpened, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 8. Laplacian Sharpening
    # ----------------------------------------------------------
    def _apply_laplacian_sharpening(self, image_data, params):
        strength = params.get('strength', 1.0)
        original_dtype = image_data.dtype

        # Simple 4-neighbour Laplacian kernel
        laplacian_kernel = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0],
        ], dtype=np.float32)

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        if img.ndim == 2:
            lap = _convolve_channel(img, laplacian_kernel)
            sharpened = img + strength * lap
        else:
            channels = []
            for c in range(img.shape[2]):
                lap = _convolve_channel(img[..., c], laplacian_kernel)
                channels.append(img[..., c] + strength * lap)
            sharpened = np.stack(channels, axis=2)

        result = _restore_alpha(sharpened, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 9. Cartoon Effect
    # ----------------------------------------------------------
    def _apply_cartoon(self, image_data, params):
        color_levels = max(2, params.get('color_levels', 8))
        edge_threshold = params.get('edge_threshold', 50.0)
        blur_k = params.get('blur_kernel_size', 5)
        original_dtype = image_data.dtype

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        # Step 1: Smooth to reduce noise before quantizing
        smoothed = _blur_image(img, blur_k, 1.0)

        # Step 2: Quantize colors into flat areas
        level_size = 255.0 / color_levels
        quantized = np.floor(smoothed / level_size) * level_size

        # Step 3: Detect edges on original grayscale
        gray = img if img.ndim == 2 else _to_grayscale(img)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        gx = _convolve_channel(gray, sobel_x)
        gy = _convolve_channel(gray, sobel_y)
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        max_mag = np.max(magnitude)
        if max_mag > 0:
            magnitude = magnitude / max_mag * 255.0

        # Step 4: Paint strong edges black on the quantized image
        edge_mask = magnitude >= edge_threshold    # shape (H, W)
        result = quantized.copy()
        result[edge_mask] = 0.0                   # broadcast sets all channels to 0

        result = _restore_alpha(result, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 10. Teal & Orange Color Grading
    # ----------------------------------------------------------
    def _apply_teal_orange(self, image_data, params):
        intensity = params.get('intensity', 0.3)
        original_dtype = image_data.dtype

        if image_data.ndim == 2:
            # Grayscale has no color channels to grade
            return image_data.copy()

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb) / 255.0              # work in [0, 1]

        # Per-pixel luminance tells us if a pixel is a shadow or highlight
        lum = _to_grayscale(img * 255.0) / 255.0  # (H, W), range [0, 1]
        lum = lum[..., np.newaxis]                 # (H, W, 1) for broadcasting

        # Shadows → teal: pull R down, push B up
        # Highlights → orange: pull B down, push R up
        shadow_shift    = np.array([-intensity * 0.5,  intensity * 0.2,  intensity * 0.5])
        highlight_shift = np.array([ intensity * 0.5,  intensity * 0.2, -intensity * 0.5])

        shadow_weight    = 1.0 - lum   # dark pixels get full shadow shift
        highlight_weight = lum          # bright pixels get full highlight shift

        graded = img + shadow_weight * shadow_shift + highlight_weight * highlight_shift
        result = _restore_alpha(graded * 255.0, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 11. Retinex (Single-Scale)
    # ----------------------------------------------------------
    def _apply_retinex(self, image_data, params):
        sigma = params.get('sigma', 5.0)
        gain = params.get('gain', 1.0)
        original_dtype = image_data.dtype

        # Cap kernel size so convolution stays fast enough for interactive use
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 31)

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        # +1 offset prevents log(0)
        img_safe = img + 1.0
        blurred = _blur_image(img_safe, kernel_size, sigma) + 1.0

        # SSR: log(image) - log(estimated_illumination)
        retinex = np.log(img_safe) - np.log(blurred)
        retinex *= gain

        # Normalize result to 0-255
        r_min = np.min(retinex)
        r_max = np.max(retinex)
        if r_max - r_min > 0:
            retinex = (retinex - r_min) / (r_max - r_min) * 255.0
        else:
            retinex = np.zeros_like(retinex)

        result = _restore_alpha(retinex, alpha)
        return _safe_output(result, original_dtype)

    # ----------------------------------------------------------
    # 12. Tilt-Shift
    # ----------------------------------------------------------
    def _apply_tilt_shift(self, image_data, params):
        focus_position = params.get('focus_position', 0.5)   # 0.0 = top, 1.0 = bottom
        focus_width    = params.get('focus_width', 0.2)      # fraction of image height
        blur_strength  = params.get('blur_strength', 3.0)
        original_dtype = image_data.dtype

        rgb, alpha = _split_alpha(image_data)
        img = _to_float(rgb)

        h = img.shape[0]

        # Build a blurred version of the whole image
        kernel_size = int(blur_strength * 4) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 21)
        blurred = _blur_image(img, kernel_size, blur_strength)

        # Focus band in pixel rows
        focus_center = focus_position * h
        focus_half   = focus_width * h / 2.0

        row_indices = np.arange(h, dtype=np.float32)
        distance    = np.abs(row_indices - focus_center)

        # Rows inside the focus band get weight 0 (sharp).
        # Rows outside transition linearly to weight 1 (fully blurred).
        transition = max(1.0, focus_half)
        blur_weight = np.clip((distance - focus_half) / transition, 0.0, 1.0)

        # Reshape for broadcasting
        if img.ndim == 2:
            blur_weight = blur_weight[:, np.newaxis]
        else:
            blur_weight = blur_weight[:, np.newaxis, np.newaxis]

        tilt_shifted = img * (1.0 - blur_weight) + blurred * blur_weight

        result = _restore_alpha(tilt_shifted, alpha)
        return _safe_output(result, original_dtype)
