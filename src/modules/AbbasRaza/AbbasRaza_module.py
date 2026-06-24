# =============================================================================
# File: src/modules/AbbasRaza/AbbasRaza_module.py
#
# Autonomous Vehicle Vision Processing Module
# Author : Abbas Raza
# Purpose: Image processing operations commonly used in self-driving car pipelines.
#          Drop-in replacement for sample_module.py – fully compatible with the
#          SRH Image Processing App architecture.
#
# NOTE: All OpenCV (cv2) calls have been replaced with NumPy / SciPy equivalents.
# =============================================================================

# --- Standard / third-party imports ------------------------------------------
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider, QPushButton,
    QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout, QSpinBox,
)
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve, median_filter, gaussian_filter

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore


# =============================================================================
# SECTION 1 – Parameter Widgets
# =============================================================================

class BaseParamsWidget(QWidget):
    """Base class for all parameter widgets – enforces get_params() interface."""

    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    """Placeholder for operations that require no user-adjustable parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class GaussianBlurParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Kernel Size (odd, e.g. 3, 5, 7):"))
        self.ksize_spinbox = QSpinBox()
        self.ksize_spinbox.setMinimum(1)
        self.ksize_spinbox.setMaximum(31)
        self.ksize_spinbox.setValue(5)
        self.ksize_spinbox.setSingleStep(2)
        layout.addWidget(self.ksize_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        ksize = self.ksize_spinbox.value()
        if ksize % 2 == 0:
            ksize += 1
        return {'ksize': ksize}


class BinaryThresholdParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Threshold Value (0–255):"))
        self.thresh_spinbox = QSpinBox()
        self.thresh_spinbox.setMinimum(0)
        self.thresh_spinbox.setMaximum(255)
        self.thresh_spinbox.setValue(127)
        layout.addWidget(self.thresh_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'threshold': self.thresh_spinbox.value()}


class SobelParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Combined", "X-direction", "Y-direction"])
        layout.addWidget(self.direction_combo)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'direction': self.direction_combo.currentText()}


class CannyParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Lower Threshold (0–255):"))
        self.low_spinbox = QSpinBox()
        self.low_spinbox.setMinimum(0)
        self.low_spinbox.setMaximum(255)
        self.low_spinbox.setValue(50)
        layout.addWidget(self.low_spinbox)
        layout.addWidget(QLabel("Upper Threshold (0–255):"))
        self.high_spinbox = QSpinBox()
        self.high_spinbox.setMinimum(0)
        self.high_spinbox.setMaximum(255)
        self.high_spinbox.setValue(150)
        layout.addWidget(self.high_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'canny_low': self.low_spinbox.value(),
            'canny_high': self.high_spinbox.value(),
        }


class MorphologyParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(["Erosion", "Dilation", "Opening", "Closing"])
        layout.addWidget(self.op_combo)
        layout.addWidget(QLabel("Kernel Size (odd):"))
        self.ksize_spinbox = QSpinBox()
        self.ksize_spinbox.setMinimum(3)
        self.ksize_spinbox.setMaximum(21)
        self.ksize_spinbox.setValue(5)
        self.ksize_spinbox.setSingleStep(2)
        layout.addWidget(self.ksize_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        ksize = self.ksize_spinbox.value()
        if ksize % 2 == 0:
            ksize += 1
        return {'morph_op': self.op_combo.currentText(), 'morph_ksize': ksize}


class PerspectiveParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Source trapezoid (fraction of image dimensions):"))

        layout.addWidget(QLabel("Top-left X offset (0–0.5):"))
        self.tl_spinbox = QDoubleSpinBox()
        self.tl_spinbox.setMinimum(0.0)
        self.tl_spinbox.setMaximum(0.5)
        self.tl_spinbox.setValue(0.35)
        self.tl_spinbox.setSingleStep(0.05)
        layout.addWidget(self.tl_spinbox)

        layout.addWidget(QLabel("Bottom-left X offset (0–0.5):"))
        self.bl_spinbox = QDoubleSpinBox()
        self.bl_spinbox.setMinimum(0.0)
        self.bl_spinbox.setMaximum(0.5)
        self.bl_spinbox.setValue(0.1)
        self.bl_spinbox.setSingleStep(0.05)
        layout.addWidget(self.bl_spinbox)

        layout.addWidget(QLabel("Top Y (horizon, 0–1):"))
        self.top_y_spinbox = QDoubleSpinBox()
        self.top_y_spinbox.setMinimum(0.0)
        self.top_y_spinbox.setMaximum(1.0)
        self.top_y_spinbox.setValue(0.6)
        self.top_y_spinbox.setSingleStep(0.05)
        layout.addWidget(self.top_y_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'persp_tl_offset': self.tl_spinbox.value(),
            'persp_bl_offset': self.bl_spinbox.value(),
            'persp_top_y': self.top_y_spinbox.value(),
        }


class ContrastParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Contrast Factor α (0.1–4.0):"))
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setMinimum(0.1)
        self.alpha_spinbox.setMaximum(4.0)
        self.alpha_spinbox.setValue(1.5)
        self.alpha_spinbox.setSingleStep(0.1)
        layout.addWidget(self.alpha_spinbox)
        layout.addWidget(QLabel("Brightness Offset β (-100–100):"))
        self.beta_spinbox = QDoubleSpinBox()
        self.beta_spinbox.setMinimum(-100.0)
        self.beta_spinbox.setMaximum(100.0)
        self.beta_spinbox.setValue(0.0)
        self.beta_spinbox.setSingleStep(5.0)
        layout.addWidget(self.beta_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'contrast_alpha': self.alpha_spinbox.value(),
            'contrast_beta': self.beta_spinbox.value(),
        }


class NoiseReductionParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Median Filter", "Gaussian Filter"])
        layout.addWidget(self.filter_combo)
        layout.addWidget(QLabel("Kernel Size (odd):"))
        self.ksize_spinbox = QSpinBox()
        self.ksize_spinbox.setMinimum(3)
        self.ksize_spinbox.setMaximum(21)
        self.ksize_spinbox.setValue(5)
        self.ksize_spinbox.setSingleStep(2)
        layout.addWidget(self.ksize_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        ksize = self.ksize_spinbox.value()
        if ksize % 2 == 0:
            ksize += 1
        return {'noise_filter': self.filter_combo.currentText(), 'noise_ksize': ksize}


# =============================================================================
# SECTION 2 – Controls Widget
# =============================================================================

class AbbasRazaControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>AV Vision Control Panel</h3>"))
        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Grayscale Conversion":        NoParamsWidget,
            "Gaussian Blur":               GaussianBlurParamsWidget,
            "Binary Thresholding":         BinaryThresholdParamsWidget,
            "Sobel Edge Detection":        SobelParamsWidget,
            "Canny Edge Detection":        CannyParamsWidget,
            "Morphological Operations":    MorphologyParamsWidget,
            "Perspective Transform (BEV)": PerspectiveParamsWidget,
            "Histogram Equalization":      NoParamsWidget,
            "Contrast Enhancement":        ContrastParamsWidget,
            "Noise Reduction":             NoiseReductionParamsWidget,
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


# =============================================================================
# SECTION 3 – Pure-NumPy helper utilities
# (All cv2 color-conversion and data-type helpers replaced)
# =============================================================================

def _to_gray_uint8(image_data: np.ndarray) -> np.ndarray:
    """
    Convert any ndarray (2-D grayscale, RGB, or RGBA) to uint8 grayscale
    using pure NumPy/skimage – no OpenCV required.
    """
    if image_data.ndim == 2:
        gray = image_data
    elif image_data.ndim == 3 and image_data.shape[2] in (3, 4):
        gray_float = rgb2gray(image_data[:, :, :3])   # [0, 1] float64
        gray = (gray_float * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {image_data.shape}")
    return np.clip(gray, 0, 255).astype(np.uint8)


def _to_uint8(image_data: np.ndarray) -> np.ndarray:
    """
    Normalise any numeric ndarray to uint8 [0, 255].
    Float images assumed to be in [0, 1]; integer images clipped to [0, 255].
    """
    if image_data.dtype in (np.float32, np.float64) and image_data.max() <= 1.0:
        return (image_data * 255).astype(np.uint8)
    return np.clip(image_data, 0, 255).astype(np.uint8)


def _match_channels(result: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Ensure `result` has the same number of channels as `original`.
    - If original was grayscale (ndim=2), collapse result to 2-D.
    - If original had an alpha channel, reattach the original alpha.
    - Otherwise keep result as-is (RGB).
    Returns an array with the same dtype as original.
    """
    if original.ndim == 2:
        # Collapse to grayscale by averaging channels if needed
        if result.ndim == 3:
            result = result.mean(axis=2)
        return result.astype(original.dtype)

    if result.ndim == 2:
        # Expand grayscale result back to the original channel count
        result = np.stack([result] * original.shape[2], axis=-1)

    if original.ndim == 3 and original.shape[2] == 4 and result.shape[2] == 3:
        # Re-attach the original alpha channel
        result = np.concatenate(
            [result, original[:, :, 3:4].astype(result.dtype)], axis=2
        )

    return result.astype(original.dtype)


# =============================================================================
# SECTION 4 – Pure-NumPy / SciPy image operation helpers
# =============================================================================

def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Return a 1-D Gaussian kernel (not normalised; caller normalises)."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _ksize_to_sigma(ksize: int) -> float:
    """Replicate OpenCV's default sigma formula: sigma = 0.3*(ksize/2-1)+0.8."""
    return 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8


def _apply_gaussian_blur(img_uint8: np.ndarray, ksize: int) -> np.ndarray:
    """
    Gaussian blur via scipy.ndimage.gaussian_filter.
    sigma derived from ksize the same way OpenCV does when sigmaX=0.
    Works on 2-D and 3-D (multi-channel) arrays.
    """
    sigma = _ksize_to_sigma(ksize)
    if img_uint8.ndim == 2:
        blurred = gaussian_filter(img_uint8.astype(np.float64), sigma=sigma)
    else:
        # Apply independently per channel
        blurred = np.stack(
            [gaussian_filter(img_uint8[:, :, c].astype(np.float64), sigma=sigma)
             for c in range(img_uint8.shape[2])],
            axis=2,
        )
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _sobel_kernels():
    """Return the 3×3 Sobel kernels Kx and Ky as float64 arrays."""
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)
    return Kx, Ky


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Linearly scale any float array to the full [0, 255] uint8 range."""
    arr = np.abs(arr).astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def _perspective_transform_numpy(
    img: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """
    Pure-NumPy perspective (projective) warp.

    Computes the 3×3 homography H that maps src → dst, then applies it
    via inverse mapping (for each destination pixel, look up source pixel).

    Parameters
    ----------
    img : (H, W) or (H, W, C) uint8 array
    src : (4, 2) float32 array – source quadrilateral corners
    dst : (4, 2) float32 array – destination quadrilateral corners

    Returns
    -------
    warped : same shape and dtype as img
    """
    h, w = img.shape[:2]

    # ---- 1. Build the homography using the DLT (Direct Linear Transform) ----
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1,  0,  0,  0, u * x, u * y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A, dtype=np.float64)

    # H is the right singular vector corresponding to the smallest singular value
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]          # normalise so bottom-right entry is 1

    # ---- 2. Inverse warp: for every (u, v) in dst, find (x, y) in src ------
    H_inv = np.linalg.inv(H)

    # Build a grid of all destination pixel coordinates
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(uu)
    dst_coords = np.stack([uu.ravel(), vv.ravel(), ones.ravel()], axis=0).astype(np.float64)

    # Map to source coordinates
    src_coords = H_inv @ dst_coords
    src_coords /= src_coords[2:3, :]          # homogeneous divide

    src_x = src_coords[0].reshape(h, w)
    src_y = src_coords[1].reshape(h, w)

    # ---- 3. Nearest-neighbour sampling (no external deps) -------------------
    src_xi = np.round(src_x).astype(np.int32)
    src_yi = np.round(src_y).astype(np.int32)

    # Mask pixels that fall outside the source image
    valid = (src_xi >= 0) & (src_xi < w) & (src_yi >= 0) & (src_yi < h)

    if img.ndim == 2:
        warped = np.zeros((h, w), dtype=img.dtype)
        warped[valid] = img[src_yi[valid], src_xi[valid]]
    else:
        warped = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
        warped[valid] = img[src_yi[valid], src_xi[valid]]

    return warped


def _histogram_equalize_channel(channel: np.ndarray) -> np.ndarray:
    """
    Equalise a single uint8 channel using the standard CDF method.
    Pure NumPy – no OpenCV required.
    """
    hist, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    # Normalise: map CDF to [0, 255], skipping zero-count bins
    cdf_min = cdf[cdf > 0].min()
    n_pixels = channel.size
    lut = np.zeros(256, dtype=np.uint8)
    non_zero = cdf > 0
    lut[non_zero] = np.round(
        (cdf[non_zero] - cdf_min) / (n_pixels - cdf_min) * 255
    ).astype(np.uint8)
    return lut[channel]


def _rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB → YCbCr using the BT.601 matrix (pure NumPy)."""
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    Y  =  0.299    * r + 0.587    * g + 0.114    * b
    Cb = -0.168736 * r - 0.331264 * g + 0.5      * b + 128.0
    Cr =  0.5      * r - 0.418688 * g - 0.081312 * b + 128.0
    ycbcr = np.stack([Y, Cb, Cr], axis=2)
    return np.clip(ycbcr, 0, 255).astype(np.uint8)


def _ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Convert uint8 YCbCr → RGB (inverse of BT.601, pure NumPy)."""
    Y  = ycbcr[:, :, 0].astype(np.float64)
    Cb = ycbcr[:, :, 1].astype(np.float64) - 128.0
    Cr = ycbcr[:, :, 2].astype(np.float64) - 128.0
    r = Y + 1.402    * Cr
    g = Y - 0.344136 * Cb - 0.714136 * Cr
    b = Y + 1.772    * Cb
    rgb = np.stack([r, g, b], axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _canny_numpy(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    """
    Full Canny edge detector implemented in pure NumPy / SciPy.

    Steps:
      1. Gaussian pre-blur (sigma≈1.4, equivalent to ksize=5 in OpenCV)
      2. Sobel gradient magnitude and angle
      3. Non-maximum suppression
      4. Double-threshold hysteresis
    """
    # --- Step 1: Gaussian pre-blur -------------------------------------------
    blurred = gaussian_filter(gray.astype(np.float64), sigma=1.4)

    # --- Step 2: Sobel gradients ---------------------------------------------
    Kx, Ky = _sobel_kernels()
    Gx = convolve(blurred, Kx)
    Gy = convolve(blurred, Ky)
    magnitude = np.hypot(Gx, Gy)
    angle = np.arctan2(Gy, Gx) * 180.0 / np.pi  # degrees
    angle[angle < 0] += 180.0

    # --- Step 3: Non-maximum suppression -------------------------------------
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # Quantise angle to 4 directions: 0°, 45°, 90°, 135°
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            m = magnitude[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            suppressed[i, j] = m if (m >= n1 and m >= n2) else 0.0

    # --- Step 4: Double-threshold + hysteresis -------------------------------
    strong_val = 255
    weak_val   = 25

    strong_mask = suppressed >= high
    weak_mask   = (suppressed >= low) & ~strong_mask

    result = np.zeros_like(suppressed, dtype=np.uint8)
    result[strong_mask] = strong_val
    result[weak_mask]   = weak_val

    # Promote weak pixels that are 8-connected to a strong pixel
    from scipy.ndimage import label
    labeled, _ = label(result == weak_val)
    for region_id in range(1, labeled.max() + 1):
        region_mask = labeled == region_id
        # Dilate region by 1 pixel and check for strong neighbour
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(region_mask)
        if np.any(result[dilated] == strong_val):
            result[region_mask] = strong_val
        else:
            result[region_mask] = 0

    return result


# =============================================================================
# SECTION 5 – Controls Widget
# =============================================================================

class AbbasRazaImageModule(IImageModule):
    """
    Autonomous Vehicle Vision Processing Module.
    All image operations use pure NumPy / SciPy – no OpenCV dependency.
    """

    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "AV Vision Module (Abbas Raza)"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = AbbasRazaControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(
                self._handle_processing_request
            )
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(
                params
            )

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as exc:
            print(f"Error loading image {file_path}: {exc}")
            return False, None, {}, None

    # ------------------------------------------------------------------
    # Core processing dispatcher
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_data: np.ndarray,
        metadata: dict,
        params: dict,
    ) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation', '')

        try:
            if operation == "Grayscale Conversion":
                processed_data = self._op_grayscale(processed_data)

            elif operation == "Gaussian Blur":
                ksize = params.get('ksize', 5)
                processed_data = self._op_gaussian_blur(processed_data, ksize)

            elif operation == "Binary Thresholding":
                threshold = params.get('threshold', 127)
                processed_data = self._op_binary_threshold(processed_data, threshold)

            elif operation == "Sobel Edge Detection":
                direction = params.get('direction', 'Combined')
                processed_data = self._op_sobel(processed_data, direction)

            elif operation == "Canny Edge Detection":
                low  = params.get('canny_low',  50)
                high = params.get('canny_high', 150)
                processed_data = self._op_canny(processed_data, low, high)

            elif operation == "Morphological Operations":
                morph_op = params.get('morph_op', 'Erosion')
                ksize    = params.get('morph_ksize', 5)
                processed_data = self._op_morphology(processed_data, morph_op, ksize)

            elif operation == "Perspective Transform (BEV)":
                tl_offset = params.get('persp_tl_offset', 0.35)
                bl_offset = params.get('persp_bl_offset', 0.10)
                top_y     = params.get('persp_top_y',     0.60)
                processed_data = self._op_perspective(
                    processed_data, tl_offset, bl_offset, top_y
                )

            elif operation == "Histogram Equalization":
                processed_data = self._op_histogram_eq(processed_data)

            elif operation == "Contrast Enhancement":
                alpha = params.get('contrast_alpha', 1.5)
                beta  = params.get('contrast_beta',  0.0)
                processed_data = self._op_contrast(processed_data, alpha, beta)

            elif operation == "Noise Reduction":
                noise_filter = params.get('noise_filter', 'Median Filter')
                ksize        = params.get('noise_ksize',  5)
                processed_data = self._op_noise_reduction(
                    processed_data, noise_filter, ksize
                )

            else:
                print(f"[AbbasRaza] Unknown operation: '{operation}'")

        except Exception as exc:
            print(f"[AbbasRaza] Error during '{operation}': {exc}")
            return image_data.copy()

        return processed_data.astype(image_data.dtype)

    # ==========================================================================
    # SECTION 6 – Private operation implementations (NumPy / SciPy only)
    # ==========================================================================

    # --------------------------------------------------------------------------
    # 1. Grayscale Conversion
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_grayscale(img: np.ndarray) -> np.ndarray:
        """
        Convert RGB/RGBA to grayscale luminance using skimage rgb2gray.
        If already grayscale, return a copy unchanged.
        """
        if img.ndim == 2:
            return img.copy()

        gray_float = rgb2gray(img[:, :, :3])   # [0, 1] float64

        if img.dtype == np.uint8:
            return (gray_float * 255).astype(np.uint8)
        return gray_float.astype(img.dtype)

    # --------------------------------------------------------------------------
    # 2. Gaussian Blur
    # Purpose: Smooth image / suppress noise before edge detection.
    # Replaces: cv2.GaussianBlur
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
        """
        Apply Gaussian blur via scipy.ndimage.gaussian_filter.
        Sigma is derived from ksize using OpenCV's formula so results are
        visually equivalent.
        """
        img8 = _to_uint8(img)
        blurred = _apply_gaussian_blur(img8, ksize)
        return _match_channels(blurred, img)

    # --------------------------------------------------------------------------
    # 3. Binary Thresholding
    # Purpose: Isolate lane markings from road background.
    # Replaces: cv2.threshold (THRESH_BINARY)
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_binary_threshold(img: np.ndarray, threshold: int) -> np.ndarray:
        """
        THRESH_BINARY: pixels > threshold → 255, else → 0.
        Operates on a grayscale version of the input.
        """
        gray = _to_gray_uint8(img)

        # Pure NumPy threshold
        binary = np.where(gray > threshold, np.uint8(255), np.uint8(0))

        return _match_channels(binary, img)

    # --------------------------------------------------------------------------
    # 4. Sobel Edge Detection
    # Purpose: Detect lane boundaries along X, Y, or both.
    # Replaces: cv2.Sobel
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_sobel(img: np.ndarray, direction: str) -> np.ndarray:
        """
        Compute Sobel gradients using scipy.ndimage.convolve with the
        standard 3×3 Sobel kernels.
        """
        gray = _to_gray_uint8(img).astype(np.float64)
        Kx, Ky = _sobel_kernels()

        if direction == "X-direction":
            sobel = convolve(gray, Kx)
        elif direction == "Y-direction":
            sobel = convolve(gray, Ky)
        else:
            gx = convolve(gray, Kx)
            gy = convolve(gray, Ky)
            sobel = np.hypot(gx, gy)

        sobel_norm = _normalize_to_uint8(sobel)
        return _match_channels(sobel_norm, img)

    # --------------------------------------------------------------------------
    # 5. Canny Edge Detection
    # Purpose: Robust multi-stage edge detection for AV pipelines.
    # Replaces: cv2.GaussianBlur + cv2.Canny
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_canny(img: np.ndarray, low: int, high: int) -> np.ndarray:
        """
        Full Canny pipeline (blur → gradients → NMS → hysteresis)
        implemented with NumPy and SciPy only.
        """
        gray = _to_gray_uint8(img)
        edges = _canny_numpy(gray, low, high)
        return _match_channels(edges, img)

    # --------------------------------------------------------------------------
    # 6. Morphological Operations
    # Purpose: Noise removal and lane segmentation refinement.
    # Replaces: cv2.getStructuringElement + cv2.morphologyEx
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_morphology(
        img: np.ndarray, op_name: str, ksize: int
    ) -> np.ndarray:
        """
        Erosion, Dilation, Opening, Closing using skimage.morphology.
        Rectangular structuring element matches OpenCV's MORPH_RECT.
        Operations applied per-channel for colour images.
        """
        # Rectangular structuring element (same as cv2.MORPH_RECT)
        selem = skimage.morphology.rectangle(ksize, ksize)

        img8 = _to_uint8(img)

        def _apply_channel(ch):
            if op_name == "Erosion":
                return skimage.morphology.erosion(ch, selem)
            elif op_name == "Dilation":
                return skimage.morphology.dilation(ch, selem)
            elif op_name == "Opening":
                return skimage.morphology.opening(ch, selem)
            elif op_name == "Closing":
                return skimage.morphology.closing(ch, selem)
            return ch

        if img8.ndim == 2:
            result = _apply_channel(img8)
        else:
            result = np.stack(
                [_apply_channel(img8[:, :, c]) for c in range(img8.shape[2])],
                axis=2,
            )

        return _match_channels(result.astype(np.uint8), img)

    # --------------------------------------------------------------------------
    # 7. Perspective Transform (Bird's Eye View)
    # Purpose: Convert road view to top-down for lane geometry.
    # Replaces: cv2.getPerspectiveTransform + cv2.warpPerspective
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_perspective(
        img: np.ndarray,
        tl_offset: float,
        bl_offset: float,
        top_y: float,
    ) -> np.ndarray:
        """
        Bird's-eye warp using a pure-NumPy DLT homography solver
        and inverse nearest-neighbour mapping.
        """
        h, w = img.shape[:2]

        src = np.float32([
            [(0.5 - tl_offset) * w, top_y * h],
            [(0.5 + tl_offset) * w, top_y * h],
            [(0.5 + bl_offset) * w, h],
            [(0.5 - bl_offset) * w, h],
        ])

        dst = np.float32([
            [0,     0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0,     h - 1],
        ])

        img8 = _to_uint8(img)
        warped = _perspective_transform_numpy(img8, src, dst)
        return _match_channels(warped, img)

    # --------------------------------------------------------------------------
    # 8. Histogram Equalization
    # Purpose: Improve contrast under variable lighting conditions.
    # Replaces: cv2.equalizeHist + cv2.cvtColor (YCrCb)
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_histogram_eq(img: np.ndarray) -> np.ndarray:
        """
        Equalise histogram for grayscale images.
        For colour images, equalise only the Y (luminance) channel in YCbCr
        space to avoid colour distortion – mirrors the YCrCb approach in OpenCV.
        """
        img8 = _to_uint8(img)

        if img8.ndim == 2:
            eq = _histogram_equalize_channel(img8)
            return _match_channels(eq, img)

        # Colour: work in YCbCr
        rgb = img8[:, :, :3]
        ycbcr = _rgb_to_ycbcr(rgb)
        ycbcr[:, :, 0] = _histogram_equalize_channel(ycbcr[:, :, 0])
        eq_rgb = _ycbcr_to_rgb(ycbcr)

        if img8.shape[2] == 4:
            eq_rgba = np.concatenate(
                [eq_rgb, img8[:, :, 3:4]], axis=2
            )
            return _match_channels(eq_rgba, img)

        return _match_channels(eq_rgb, img)

    # --------------------------------------------------------------------------
    # 9. Contrast Enhancement
    # Purpose: Improve visibility of road markings (linear scaling).
    # Replaces: cv2.convertScaleAbs
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_contrast(
        img: np.ndarray, alpha: float, beta: float
    ) -> np.ndarray:
        """
        Linear contrast: output = α × input + β, clipped to [0, 255].
        Pure NumPy – no OpenCV required.
        """
        img8 = _to_uint8(img)
        enhanced = np.clip(alpha * img8.astype(np.float64) + beta, 0, 255).astype(np.uint8)
        return _match_channels(enhanced, img)

    # --------------------------------------------------------------------------
    # 10. Noise Reduction
    # Purpose: Prepare images for robust feature extraction.
    # Replaces: cv2.medianBlur + cv2.GaussianBlur
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_noise_reduction(
        img: np.ndarray, filter_name: str, ksize: int
    ) -> np.ndarray:
        """
        Median filter  → scipy.ndimage.median_filter  (non-linear, edge-preserving)
        Gaussian filter → scipy.ndimage.gaussian_filter (fast, Gaussian noise)
        Both handle 2-D and multi-channel arrays natively.
        """
        img8 = _to_uint8(img)

        if filter_name == "Median Filter":
            if img8.ndim == 2:
                result = median_filter(img8, size=ksize)
            else:
                # Apply median per channel
                result = np.stack(
                    [median_filter(img8[:, :, c], size=ksize)
                     for c in range(img8.shape[2])],
                    axis=2,
                )
        else:
            # Gaussian filter
            sigma = _ksize_to_sigma(ksize)
            if img8.ndim == 2:
                result = gaussian_filter(img8.astype(np.float64), sigma=sigma)
            else:
                result = np.stack(
                    [gaussian_filter(img8[:, :, c].astype(np.float64), sigma=sigma)
                     for c in range(img8.shape[2])],
                    axis=2,
                )
            result = np.clip(result, 0, 255)

        return _match_channels(result.astype(np.uint8), img)


# =============================================================================
# SECTION 7 – Module Registration
# =============================================================================

def register_module():
    """Factory function called by the app's plugin loader."""
    return AbbasRazaImageModule()
