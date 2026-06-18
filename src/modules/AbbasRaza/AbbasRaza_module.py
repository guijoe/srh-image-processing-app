# =============================================================================
# File: src/modules/AbbasRaza/AbbasRaza_module.py
#
# Autonomous Vehicle Vision Processing Module
# Author : Abbas Raza
# Purpose: Image processing operations commonly used in self-driving car pipelines.
#          Drop-in replacement for sample_module.py – fully compatible with the
#          SRH Image Processing App architecture.
# =============================================================================

# --- Standard / third-party imports (same set as sample_module.py + cv2) -----
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider, QPushButton,
    QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout, QSpinBox,
)
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio
import cv2                        # OpenCV – core dependency for AV vision ops
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore


# =============================================================================
# SECTION 1 – Parameter Widgets
# Each class mirrors the BaseParamsWidget pattern from sample_module.py.
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


# -----------------------------------------------------------------------------
# 1. Gaussian Blur params
# -----------------------------------------------------------------------------
class GaussianBlurParamsWidget(BaseParamsWidget):
    """Controls for Gaussian blur kernel size (sigma)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Kernel Size (odd, e.g. 3, 5, 7):"))
        self.ksize_spinbox = QSpinBox()
        self.ksize_spinbox.setMinimum(1)
        self.ksize_spinbox.setMaximum(31)
        self.ksize_spinbox.setValue(5)
        self.ksize_spinbox.setSingleStep(2)   # keep it odd
        layout.addWidget(self.ksize_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        ksize = self.ksize_spinbox.value()
        # Ensure odd value (required by cv2.GaussianBlur)
        if ksize % 2 == 0:
            ksize += 1
        return {'ksize': ksize}


# -----------------------------------------------------------------------------
# 2. Binary Thresholding params
# -----------------------------------------------------------------------------
class BinaryThresholdParamsWidget(BaseParamsWidget):
    """Controls for binary threshold value (0-255)."""

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


# -----------------------------------------------------------------------------
# 3. Sobel Edge Detection params
# -----------------------------------------------------------------------------
class SobelParamsWidget(BaseParamsWidget):
    """Controls for Sobel direction: X, Y, or Combined."""

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


# -----------------------------------------------------------------------------
# 4. Canny Edge Detection params
# -----------------------------------------------------------------------------
class CannyParamsWidget(BaseParamsWidget):
    """Controls for Canny lower / upper thresholds."""

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


# -----------------------------------------------------------------------------
# 5. Morphological Operations params
# -----------------------------------------------------------------------------
class MorphologyParamsWidget(BaseParamsWidget):
    """Controls for morphological operation type and kernel size."""

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
        return {
            'morph_op': self.op_combo.currentText(),
            'morph_ksize': ksize,
        }


# -----------------------------------------------------------------------------
# 6. Perspective Transformation (Bird's Eye View) params
# -----------------------------------------------------------------------------
class PerspectiveParamsWidget(BaseParamsWidget):
    """
    Controls for the four source-point offsets used to build the bird's-eye
    perspective warp.  Values are expressed as fractions of image width/height
    so they adapt to any image size.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(
            "Source trapezoid (fraction of image dimensions):"
        ))

        # Top-left X offset from centre as fraction of width
        layout.addWidget(QLabel("Top-left X offset (0–0.5):"))
        self.tl_spinbox = QDoubleSpinBox()
        self.tl_spinbox.setMinimum(0.0)
        self.tl_spinbox.setMaximum(0.5)
        self.tl_spinbox.setValue(0.35)
        self.tl_spinbox.setSingleStep(0.05)
        layout.addWidget(self.tl_spinbox)

        # Bottom-left X offset as fraction of width
        layout.addWidget(QLabel("Bottom-left X offset (0–0.5):"))
        self.bl_spinbox = QDoubleSpinBox()
        self.bl_spinbox.setMinimum(0.0)
        self.bl_spinbox.setMaximum(0.5)
        self.bl_spinbox.setValue(0.1)
        self.bl_spinbox.setSingleStep(0.05)
        layout.addWidget(self.bl_spinbox)

        # Top Y as fraction of image height (where horizon sits)
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


# -----------------------------------------------------------------------------
# 7. Contrast Enhancement params
# -----------------------------------------------------------------------------
class ContrastParamsWidget(BaseParamsWidget):
    """Controls for a linear contrast factor (alpha) and brightness (beta)."""

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


# -----------------------------------------------------------------------------
# 8. Noise Reduction params
# -----------------------------------------------------------------------------
class NoiseReductionParamsWidget(BaseParamsWidget):
    """Controls for noise-reduction method and kernel size."""

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
        return {
            'noise_filter': self.filter_combo.currentText(),
            'noise_ksize': ksize,
        }


# =============================================================================
# SECTION 2 – Controls Widget  (mirrors SampleControlsWidget exactly)
# =============================================================================

class AbbasRazaControlsWidget(QWidget):
    """
    Control panel for the Autonomous Vehicle Vision Processing Module.
    Follows the exact same structure as SampleControlsWidget:
      – QComboBox for operation selection
      – QStackedWidget for per-operation parameter panels
      – QPushButton to trigger processing
    """

    # Signal forwarded to the module manager to apply processing
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

        # Stacked widget – one pane per operation
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Map operation display-names → their parameter widget class
        # (Order determines dropdown order)
        operations = {
            "Grayscale Conversion":            NoParamsWidget,
            "Gaussian Blur":                   GaussianBlurParamsWidget,
            "Binary Thresholding":             BinaryThresholdParamsWidget,
            "Sobel Edge Detection":            SobelParamsWidget,
            "Canny Edge Detection":            CannyParamsWidget,
            "Morphological Operations":        MorphologyParamsWidget,
            "Perspective Transform (BEV)":     PerspectiveParamsWidget,
            "Histogram Equalization":          NoParamsWidget,
            "Contrast Enhancement":            ContrastParamsWidget,
            "Noise Reduction":                 NoiseReductionParamsWidget,
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)

        # Wire signals
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(
            self._on_operation_changed
        )

    def _on_apply_clicked(self):
        """Collect params from the active widget and emit process_requested."""
        operation_name = self.operation_selector.currentText()
        active_widget = self.param_widgets[operation_name]
        params = active_widget.get_params()
        params['operation'] = operation_name   # include name so module can dispatch
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        """Switch the stacked widget to the matching parameter pane."""
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])


# =============================================================================
# SECTION 3 – Helper: safe grayscale conversion
# =============================================================================

def _to_gray_uint8(image_data: np.ndarray) -> np.ndarray:
    """
    Convert any ndarray (2-D grayscale, RGB, or RGBA) to a uint8 grayscale
    image suitable for use with OpenCV functions.
    """
    if image_data.ndim == 2:
        # Already grayscale – just ensure uint8
        gray = image_data
    elif image_data.ndim == 3 and image_data.shape[2] in (3, 4):
        # Convert RGB/RGBA → grayscale via skimage (handles float/uint8 input)
        gray_float = rgb2gray(image_data[:, :, :3])
        gray = (gray_float * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image shape: {image_data.shape}")

    # Guarantee uint8 range
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _to_bgr_uint8(image_data: np.ndarray) -> np.ndarray:
    """
    Convert any ndarray to a BGR uint8 image for OpenCV.
    Grayscale input is replicated to 3 channels.
    """
    orig_max = image_data.max()
    if orig_max <= 1.0 and image_data.dtype in (np.float32, np.float64):
        img8 = (image_data * 255).astype(np.uint8)
    else:
        img8 = np.clip(image_data, 0, 255).astype(np.uint8)

    if img8.ndim == 2:
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    elif img8.shape[2] == 4:                       # RGBA → BGR
        return cv2.cvtColor(img8, cv2.COLOR_RGBA2BGR)
    elif img8.shape[2] == 3:                       # RGB → BGR
        return cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)
    return img8


def _bgr_to_output(bgr: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR uint8 result back to the same channel layout and
    dtype as the original input image.
    """
    if original.ndim == 2:
        # Caller expects grayscale
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif original.ndim == 3 and original.shape[2] == 4:
        # Re-attach original alpha channel
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = np.dstack([rgb, original[:, :, 3]])
    else:
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return out.astype(original.dtype)


# =============================================================================
# SECTION 4 – Main Module Class
# =============================================================================

class AbbasRazaImageModule(IImageModule):
    """
    Autonomous Vehicle Vision Processing Module.

    Registers itself with the SRH Image Processing App under the name
    "AV Vision Module (Abbas Raza)" which will appear in the module
    dropdown automatically when the app discovers modules at startup.
    """

    def __init__(self):
        super().__init__()
        self._controls_widget = None   # created lazily on first request

    # ------------------------------------------------------------------
    # IImageModule interface
    # ------------------------------------------------------------------

    def get_name(self) -> str:
        """Display name shown in the module dropdown."""
        return "AV Vision Module (Abbas Raza)"

    def get_supported_formats(self) -> list[str]:
        """Standard raster formats handled by imageio/OpenCV."""
        return ["png", "jpg", "jpeg", "bmp", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        """
        Lazily instantiate and return the control panel widget.
        Connects the widget's process_requested signal to _handle_processing_request.
        """
        if self._controls_widget is None:
            self._controls_widget = AbbasRazaControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(
                self._handle_processing_request
            )
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        """
        Forwarded by the control widget – asks the module manager to run
        process_image() on the currently active image.
        """
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(
                params
            )

    def load_image(self, file_path: str):
        """
        Load an image from disk via imageio (same logic as sample_module.py).
        Returns (success, image_data, metadata, session_id).
        """
        try:
            image_data = imageio.imread(file_path)

            if image_data.ndim == 3 and image_data.shape[2] in (3, 4):
                # RGB / RGBA – no extra handling needed
                pass
            elif image_data.ndim == 2:
                # Grayscale – keep as 2-D (consistent with rest of module)
                pass
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

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
        """
        Dispatch the requested operation and return the processed ndarray.
        All operations handle both grayscale (2-D) and RGB/RGBA (3-D) input.
        Output dtype always matches input dtype.
        """
        # Work on a copy so the original image_data is never mutated
        processed_data = image_data.copy()
        operation = params.get('operation', '')

        try:
            # ----------------------------------------------------------------
            # 1. Grayscale Conversion
            # Purpose: Reduce colour image to luminance for downstream ops.
            # ----------------------------------------------------------------
            if operation == "Grayscale Conversion":
                processed_data = self._op_grayscale(processed_data)

            # ----------------------------------------------------------------
            # 2. Gaussian Blur
            # Purpose: Smooth image / suppress noise before edge detection.
            # ----------------------------------------------------------------
            elif operation == "Gaussian Blur":
                ksize = params.get('ksize', 5)
                processed_data = self._op_gaussian_blur(processed_data, ksize)

            # ----------------------------------------------------------------
            # 3. Binary Thresholding
            # Purpose: Isolate lane markings from road background.
            # ----------------------------------------------------------------
            elif operation == "Binary Thresholding":
                threshold = params.get('threshold', 127)
                processed_data = self._op_binary_threshold(
                    processed_data, threshold
                )

            # ----------------------------------------------------------------
            # 4. Sobel Edge Detection
            # Purpose: Detect lane boundaries along X, Y, or both.
            # ----------------------------------------------------------------
            elif operation == "Sobel Edge Detection":
                direction = params.get('direction', 'Combined')
                processed_data = self._op_sobel(processed_data, direction)

            # ----------------------------------------------------------------
            # 5. Canny Edge Detection
            # Purpose: Robust multi-stage edge detection for AV pipelines.
            # ----------------------------------------------------------------
            elif operation == "Canny Edge Detection":
                low  = params.get('canny_low',  50)
                high = params.get('canny_high', 150)
                processed_data = self._op_canny(processed_data, low, high)

            # ----------------------------------------------------------------
            # 6. Morphological Operations
            # Purpose: Noise removal and lane segmentation refinement.
            # ----------------------------------------------------------------
            elif operation == "Morphological Operations":
                morph_op = params.get('morph_op', 'Erosion')
                ksize    = params.get('morph_ksize', 5)
                processed_data = self._op_morphology(
                    processed_data, morph_op, ksize
                )

            # ----------------------------------------------------------------
            # 7. Perspective Transform (Bird's Eye View)
            # Purpose: Convert road view to top-down for lane geometry.
            # ----------------------------------------------------------------
            elif operation == "Perspective Transform (BEV)":
                tl_offset = params.get('persp_tl_offset', 0.35)
                bl_offset = params.get('persp_bl_offset', 0.10)
                top_y     = params.get('persp_top_y',     0.60)
                processed_data = self._op_perspective(
                    processed_data, tl_offset, bl_offset, top_y
                )

            # ----------------------------------------------------------------
            # 8. Histogram Equalization
            # Purpose: Improve contrast under variable lighting conditions.
            # ----------------------------------------------------------------
            elif operation == "Histogram Equalization":
                processed_data = self._op_histogram_eq(processed_data)

            # ----------------------------------------------------------------
            # 9. Contrast Enhancement
            # Purpose: Improve visibility of road markings (linear scaling).
            # ----------------------------------------------------------------
            elif operation == "Contrast Enhancement":
                alpha = params.get('contrast_alpha', 1.5)
                beta  = params.get('contrast_beta',  0.0)
                processed_data = self._op_contrast(processed_data, alpha, beta)

            # ----------------------------------------------------------------
            # 10. Noise Reduction
            # Purpose: Prepare images for robust feature extraction.
            # ----------------------------------------------------------------
            elif operation == "Noise Reduction":
                noise_filter = params.get('noise_filter', 'Median Filter')
                ksize        = params.get('noise_ksize',  5)
                processed_data = self._op_noise_reduction(
                    processed_data, noise_filter, ksize
                )

            else:
                print(f"[AbbasRaza] Unknown operation: '{operation}'")

        except Exception as exc:
            # On any processing error, return the unmodified copy gracefully
            print(f"[AbbasRaza] Error during '{operation}': {exc}")
            return image_data.copy()

        # Ensure output dtype matches input dtype (same convention as sample_module.py)
        processed_data = processed_data.astype(image_data.dtype)
        return processed_data

    # ==========================================================================
    # SECTION 5 – Private operation implementations
    # Each method accepts any valid ndarray and returns an ndarray of the same
    # spatial shape (and matching or compatible dtype).
    # ==========================================================================

    # --------------------------------------------------------------------------
    # 1. Grayscale Conversion
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_grayscale(img: np.ndarray) -> np.ndarray:
        """
        Convert RGB/RGBA to grayscale luminance image.
        If already grayscale, return a copy unchanged.
        """
        if img.ndim == 2:
            # Already grayscale – nothing to do
            return img.copy()

        # Convert via skimage rgb2gray (handles any numeric dtype)
        gray_float = rgb2gray(img[:, :, :3])   # returns [0,1] float64

        # Scale back to the original value range
        if img.dtype == np.uint8:
            return (gray_float * 255).astype(np.uint8)
        else:
            # Float input – keep normalised [0, 1]
            return gray_float.astype(img.dtype)

    # --------------------------------------------------------------------------
    # 2. Gaussian Blur
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
        """
        Apply Gaussian blur using cv2.GaussianBlur.
        sigma=0 lets OpenCV compute sigma from ksize automatically.
        Handles RGB, RGBA, and grayscale.
        """
        bgr = _to_bgr_uint8(img)
        blurred_bgr = cv2.GaussianBlur(bgr, (ksize, ksize), sigmaX=0)
        return _bgr_to_output(blurred_bgr, img)

    # --------------------------------------------------------------------------
    # 3. Binary Thresholding
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_binary_threshold(img: np.ndarray, threshold: int) -> np.ndarray:
        """
        Apply THRESH_BINARY: pixels above threshold → 255, else → 0.
        Works on grayscale; for colour images, converts to gray first.
        """
        gray = _to_gray_uint8(img)
        _, binary = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY
        )
        # Return as the same channel layout as the original
        if img.ndim == 2:
            return binary.astype(img.dtype)
        # Replicate single channel to match original channel count
        out = np.stack([binary] * img.shape[2], axis=-1)
        return out.astype(img.dtype)

    # --------------------------------------------------------------------------
    # 4. Sobel Edge Detection
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_sobel(img: np.ndarray, direction: str) -> np.ndarray:
        """
        Compute Sobel gradients in X, Y, or combined (magnitude).
        Output is a normalised uint8 grayscale (or replicated to RGB if input
        was colour, so the viewer can display it consistently).
        """
        gray = _to_gray_uint8(img)

        if direction == "X-direction":
            # Gradient in X – detects vertical edges (lane lines)
            sobel = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
        elif direction == "Y-direction":
            # Gradient in Y – detects horizontal edges
            sobel = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
        else:
            # Combined magnitude: sqrt(Gx² + Gy²)
            gx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
            sobel = np.hypot(gx, gy)

        # Normalise to 0-255
        sobel_norm = cv2.normalize(
            np.abs(sobel), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        if img.ndim == 2:
            return sobel_norm.astype(img.dtype)

        out = np.stack([sobel_norm] * img.shape[2], axis=-1)
        return out.astype(img.dtype)

    # --------------------------------------------------------------------------
    # 5. Canny Edge Detection
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_canny(img: np.ndarray, low: int, high: int) -> np.ndarray:
        """
        Industry-standard Canny edge detector.
        Applies Gaussian pre-blur (ksize=5) for noise robustness.
        """
        gray = _to_gray_uint8(img)

        # Pre-blur reduces false edges from sensor noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=low, threshold2=high)

        if img.ndim == 2:
            return edges.astype(img.dtype)

        out = np.stack([edges] * img.shape[2], axis=-1)
        return out.astype(img.dtype)

    # --------------------------------------------------------------------------
    # 6. Morphological Operations
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_morphology(
        img: np.ndarray, op_name: str, ksize: int
    ) -> np.ndarray:
        """
        Apply erosion, dilation, opening, or closing with a rectangular kernel.
        Operates per-channel so colour images are handled correctly.
        """
        # Build structuring element
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (ksize, ksize)
        )

        # Map name → OpenCV morph type
        morph_map = {
            "Erosion":  cv2.MORPH_ERODE,
            "Dilation": cv2.MORPH_DILATE,
            "Opening":  cv2.MORPH_OPEN,
            "Closing":  cv2.MORPH_CLOSE,
        }
        morph_type = morph_map.get(op_name, cv2.MORPH_ERODE)

        bgr = _to_bgr_uint8(img)
        result_bgr = cv2.morphologyEx(bgr, morph_type, kernel)
        return _bgr_to_output(result_bgr, img)

    # --------------------------------------------------------------------------
    # 7. Perspective Transformation (Bird's Eye View)
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_perspective(
        img: np.ndarray,
        tl_offset: float,
        bl_offset: float,
        top_y: float,
    ) -> np.ndarray:
        """
        Warp the image to a bird's-eye (top-down) view using a four-point
        perspective transform – the standard first step in AV lane detection.

        Source trapezoid (fractions of image W/H):
            TL = (0.5 - tl_offset, top_y)    TR = (0.5 + tl_offset, top_y)
            BL = (0.5 - bl_offset, 1.0  )    BR = (0.5 + bl_offset, 1.0  )  (bottom)

        Destination rectangle spans the full image.
        """
        h, w = img.shape[:2]

        # Convert fractional coords → pixel coords
        src = np.float32([
            [(0.5 - tl_offset) * w, top_y * h],   # top-left
            [(0.5 + tl_offset) * w, top_y * h],   # top-right
            [(0.5 + bl_offset) * w, h],            # bottom-right (road edge)
            [(0.5 - bl_offset) * w, h],            # bottom-left  (road edge)
        ])

        dst = np.float32([
            [0,     0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0,     h - 1],
        ])

        # Compute and apply the perspective warp matrix
        M = cv2.getPerspectiveTransform(src, dst)
        bgr = _to_bgr_uint8(img)
        warped_bgr = cv2.warpPerspective(bgr, M, (w, h))
        return _bgr_to_output(warped_bgr, img)

    # --------------------------------------------------------------------------
    # 8. Histogram Equalization
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_histogram_eq(img: np.ndarray) -> np.ndarray:
        """
        Equalise histogram to improve contrast under poor lighting.
        For colour images, convert to YCrCb, equalise the Y channel only,
        then convert back – this avoids colour distortion.
        """
        bgr = _to_bgr_uint8(img)

        if img.ndim == 2:
            # Grayscale: direct equalisation
            eq = cv2.equalizeHist(bgr[:, :, 0])
            return eq.astype(img.dtype)

        # Colour: equalise luminance channel only (avoids hue shifts)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        eq_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return _bgr_to_output(eq_bgr, img)

    # --------------------------------------------------------------------------
    # 9. Contrast Enhancement
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_contrast(
        img: np.ndarray, alpha: float, beta: float
    ) -> np.ndarray:
        """
        Linear contrast enhancement: output = α × input + β
        α > 1 increases contrast; β shifts brightness.
        Result is clamped to [0, 255].
        """
        bgr = _to_bgr_uint8(img)

        # cv2.convertScaleAbs clips to uint8 range automatically
        enhanced = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
        return _bgr_to_output(enhanced, img)

    # --------------------------------------------------------------------------
    # 10. Noise Reduction
    # --------------------------------------------------------------------------
    @staticmethod
    def _op_noise_reduction(
        img: np.ndarray, filter_name: str, ksize: int
    ) -> np.ndarray:
        """
        Apply median or Gaussian filter to reduce sensor / compression noise.
        Median filter is non-linear and preserves edges better.
        Gaussian filter is faster and suitable when noise is Gaussian.
        """
        bgr = _to_bgr_uint8(img)

        if filter_name == "Median Filter":
            # cv2.medianBlur requires odd ksize; handles colour natively
            result = cv2.medianBlur(bgr, ksize)
        else:
            # Gaussian: sigmaX=0 → derived from ksize
            result = cv2.GaussianBlur(bgr, (ksize, ksize), sigmaX=0)

        return _bgr_to_output(result, img)


# =============================================================================
# SECTION 6 – Module Registration
# =============================================================================
# The SRH Image Processing App's module loader calls register_module() on every
# file it discovers under src/modules/.  Return the module instance here.
# This mirrors the registration pattern used in sample_module.py.
# =============================================================================

def register_module():
    """
    Factory function called by the app's plugin loader.
    Returns an instance of AbbasRazaImageModule so it appears in the
    module dropdown automatically.
    """
    return AbbasRazaImageModule()