# maarij_forensic_module.py
# Educational Forensic Image Processing Module — Maarij
# All image processing uses NumPy only (no OpenCV, skimage, scipy, PIL, etc.)
# Results are visual aids for educational purposes — NOT legal or forensic evidence.

import numpy as np
import imageio

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QStackedWidget, QDoubleSpinBox,
    QSpinBox, QFormLayout
)
from PySide6.QtCore import Signal

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore


# ═══════════════════════════════════════════════════════════
# Helper Functions — NumPy only, no external image libraries
# ═══════════════════════════════════════════════════════════

def _to_grayscale(image):
    """Convert any image (2D, 3D RGB, or RGBA) to 2D float32 grayscale."""
    if image.ndim == 2:
        return image.astype(np.float32)
    # Standard luminance weights for RGB
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _safe_odd(size, image):
    """
    Return an odd integer >= 1 that is smaller than the image's smallest dimension.
    Prevents kernel sizes from exceeding image size.
    """
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    min_dim = min(image.shape[0], image.shape[1])
    if size >= min_dim:
        size = max(1, min_dim - 2)
        if size % 2 == 0:
            size -= 1
        size = max(1, size)
    return size


def _gaussian_kernel(size, sigma):
    """Create a normalized 2D Gaussian kernel of the given size and sigma."""
    size = int(size)
    if size < 1:
        size = 1
    if size % 2 == 0:
        size += 1
    if sigma is None or sigma <= 0:
        sigma = size / 6.0
    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            dx = i - center
            dy = j - center
            kernel[i, j] = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    total = np.sum(kernel)
    if total > 0:
        kernel /= total
    return kernel


def _convolve_channel(channel, kernel):
    """
    Convolve a single 2D channel with a 2D kernel.
    Loops over kernel elements (kh*kw iterations, each a fast NumPy array op).
    This is readable and efficient for small kernels.
    """
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(channel.astype(np.float32),
                    ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    h, w = channel.shape
    result = np.zeros((h, w), dtype=np.float32)
    # Each iteration adds one kernel-element's contribution to the whole image at once
    for i in range(kh):
        for j in range(kw):
            result += kernel[i, j] * padded[i:i + h, j:j + w]
    return result


def _gaussian_blur(image, kernel_size, sigma=None):
    """Apply Gaussian blur to a 2D or 3D image using NumPy convolution."""
    kernel_size = _safe_odd(kernel_size, image)
    if sigma is None or sigma <= 0:
        sigma = kernel_size / 6.0
    kernel = _gaussian_kernel(kernel_size, sigma)
    img = image.astype(np.float32)
    if img.ndim == 2:
        return _convolve_channel(img, kernel)
    # Blur each channel independently
    channels = []
    for c in range(img.shape[2]):
        channels.append(_convolve_channel(img[:, :, c], kernel))
    return np.stack(channels, axis=2)


def _normalize_to_uint8(image):
    """Stretch image values to fill 0–255 range."""
    mn, mx = np.min(image), np.max(image)
    if mx == mn:
        return np.zeros_like(image, dtype=np.uint8)
    out = (image - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _gray_to_output(gray_2d, original_image):
    """
    Match a 2D grayscale result to the channel count of the original image.
    Ensures napari gets a consistent array shape.
    """
    gray = np.clip(gray_2d, 0, 255).astype(np.float32)
    if original_image.ndim == 2:
        return gray
    if original_image.shape[2] == 4:
        # Preserve original alpha channel
        alpha = original_image[:, :, 3:4].astype(np.float32)
        rgb = np.stack([gray, gray, gray], axis=2)
        return np.concatenate([rgb, alpha], axis=2)
    # 3-channel RGB output
    return np.stack([gray, gray, gray], axis=2)


def _erode_binary(mask, kernel_size):
    """
    Binary erosion: a pixel stays True only if ALL pixels in the window are True.
    Removes small isolated spots from a binary mask.
    """
    if kernel_size < 1:
        return mask.copy()
    pad = kernel_size // 2
    padded = np.pad(mask.astype(np.float32), pad, mode='constant', constant_values=0)
    h, w = mask.shape
    count = np.zeros((h, w), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            count += padded[i:i + h, j:j + w]
    return count >= float(kernel_size * kernel_size)


def _dilate_binary(mask, kernel_size):
    """
    Binary dilation: a pixel becomes True if ANY pixel in the window is True.
    Restores eroded regions to their approximate original size.
    """
    if kernel_size < 1:
        return mask.copy()
    pad = kernel_size // 2
    padded = np.pad(mask.astype(np.float32), pad, mode='constant', constant_values=0)
    h, w = mask.shape
    count = np.zeros((h, w), dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            count += padded[i:i + h, j:j + w]
    return count > 0


def _create_motion_blur_kernel(length, angle_degrees):
    """
    Create a motion blur PSF (point spread function) kernel.
    Draws a line of `length` pixels at the given angle.
    """
    length = max(1, int(length))
    size = length * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle_rad = np.deg2rad(float(angle_degrees))
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Sample points along the line and set them in the kernel
    num_samples = length * 4
    for t_idx in range(num_samples + 1):
        t = (t_idx / max(1, num_samples) - 0.5) * length
        x = int(round(center + t * cos_a))
        y = int(round(center + t * sin_a))
        if 0 <= y < size and 0 <= x < size:
            kernel[y, x] = 1.0
    total = np.sum(kernel)
    if total > 0:
        kernel /= total
    else:
        # Fallback: identity kernel so image is unchanged
        kernel[center, center] = 1.0
    return kernel


# ═══════════════════════════════════════════════════════════
# Parameter Widget Classes — one per operation
# ═══════════════════════════════════════════════════════════

class BaseParamsWidget(QWidget):
    """Base class — every params widget must implement get_params()."""
    def get_params(self) -> dict:
        raise NotImplementedError


class FingerprintParamsWidget(BaseParamsWidget):
    """Parameters for Fingerprint Enhancement."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.contrast_strength = QDoubleSpinBox()
        self.contrast_strength.setRange(0.5, 3.0)
        self.contrast_strength.setValue(1.5)
        self.contrast_strength.setSingleStep(0.1)
        layout.addRow("Contrast Strength:", self.contrast_strength)

        self.blur_kernel_size = QSpinBox()
        self.blur_kernel_size.setRange(1, 15)
        self.blur_kernel_size.setValue(3)
        self.blur_kernel_size.setSingleStep(2)
        layout.addRow("Blur Kernel Size:", self.blur_kernel_size)

        self.ridge_threshold = QSpinBox()
        self.ridge_threshold.setRange(0, 255)
        self.ridge_threshold.setValue(0)
        layout.addRow("Ridge Threshold:", self.ridge_threshold)

    def get_params(self) -> dict:
        return {
            'contrast_strength': self.contrast_strength.value(),
            'blur_kernel_size': self.blur_kernel_size.value(),
            'ridge_threshold': self.ridge_threshold.value(),
        }


class ELAParamsWidget(BaseParamsWidget):
    """Parameters for ELA — Forgery Detection (NumPy block residual approximation)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.block_size = QComboBox()
        self.block_size.addItems(['4', '8', '16'])
        self.block_size.setCurrentText('8')
        layout.addRow("Block Size:", self.block_size)

        self.amplification = QDoubleSpinBox()
        self.amplification.setRange(1.0, 20.0)
        self.amplification.setValue(5.0)
        self.amplification.setSingleStep(0.5)
        layout.addRow("Amplification:", self.amplification)

        self.threshold = QSpinBox()
        self.threshold.setRange(0, 255)
        self.threshold.setValue(10)
        layout.addRow("Threshold:", self.threshold)

    def get_params(self) -> dict:
        return {
            'block_size': int(self.block_size.currentText()),
            'amplification': self.amplification.value(),
            'threshold': self.threshold.value(),
        }


class SecurityDocumentParamsWidget(BaseParamsWidget):
    """Parameters for Security Document Analysis."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.detail_strength = QDoubleSpinBox()
        self.detail_strength.setRange(0.5, 5.0)
        self.detail_strength.setValue(2.0)
        self.detail_strength.setSingleStep(0.1)
        layout.addRow("Detail Strength:", self.detail_strength)

        self.edge_threshold = QSpinBox()
        self.edge_threshold.setRange(0, 255)
        self.edge_threshold.setValue(30)
        layout.addRow("Edge Threshold:", self.edge_threshold)

        self.grid_size = QSpinBox()
        self.grid_size.setRange(3, 15)
        self.grid_size.setValue(7)
        self.grid_size.setSingleStep(2)
        layout.addRow("Grid Size:", self.grid_size)

    def get_params(self) -> dict:
        return {
            'detail_strength': self.detail_strength.value(),
            'edge_threshold': self.edge_threshold.value(),
            'grid_size': self.grid_size.value(),
        }


class WienerDeblurParamsWidget(BaseParamsWidget):
    """Parameters for Surveillance Deblur — Wiener Filter."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.blur_length = QSpinBox()
        self.blur_length.setRange(1, 31)
        self.blur_length.setValue(10)
        layout.addRow("Blur Length:", self.blur_length)

        self.blur_angle = QSpinBox()
        self.blur_angle.setRange(0, 180)
        self.blur_angle.setValue(0)
        layout.addRow("Blur Angle (°):", self.blur_angle)

        self.noise_ratio = QDoubleSpinBox()
        self.noise_ratio.setRange(0.0001, 0.1)
        self.noise_ratio.setValue(0.01)
        self.noise_ratio.setSingleStep(0.001)
        self.noise_ratio.setDecimals(4)
        layout.addRow("Noise Ratio:", self.noise_ratio)

    def get_params(self) -> dict:
        return {
            'blur_length': self.blur_length.value(),
            'blur_angle': self.blur_angle.value(),
            'noise_ratio': self.noise_ratio.value(),
        }


class LatentRecoveryParamsWidget(BaseParamsWidget):
    """Parameters for Latent Image Recovery."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.local_contrast = QDoubleSpinBox()
        self.local_contrast.setRange(0.5, 5.0)
        self.local_contrast.setValue(2.0)
        self.local_contrast.setSingleStep(0.1)
        layout.addRow("Local Contrast:", self.local_contrast)

        self.detail_strength = QDoubleSpinBox()
        self.detail_strength.setRange(0.0, 5.0)
        self.detail_strength.setValue(2.0)
        self.detail_strength.setSingleStep(0.1)
        layout.addRow("Detail Strength:", self.detail_strength)

        self.threshold = QSpinBox()
        self.threshold.setRange(0, 255)
        self.threshold.setValue(0)
        layout.addRow("Threshold:", self.threshold)

    def get_params(self) -> dict:
        return {
            'local_contrast': self.local_contrast.value(),
            'detail_strength': self.detail_strength.value(),
            'threshold': self.threshold.value(),
        }


class IRInkParamsWidget(BaseParamsWidget):
    """Parameters for IR Ink Detection (color channel proxy analysis)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.channel_diff = QComboBox()
        self.channel_diff.addItems(['R-G', 'R-B', 'G-B'])
        layout.addRow("Channel Difference:", self.channel_diff)

        self.enhancement_strength = QDoubleSpinBox()
        self.enhancement_strength.setRange(0.5, 10.0)
        self.enhancement_strength.setValue(3.0)
        self.enhancement_strength.setSingleStep(0.5)
        layout.addRow("Enhancement Strength:", self.enhancement_strength)

        self.threshold = QSpinBox()
        self.threshold.setRange(0, 255)
        self.threshold.setValue(20)
        layout.addRow("Threshold:", self.threshold)

    def get_params(self) -> dict:
        return {
            'channel_diff': self.channel_diff.currentText(),
            'enhancement_strength': self.enhancement_strength.value(),
            'threshold': self.threshold.value(),
        }


class BitPlaneParamsWidget(BaseParamsWidget):
    """Parameters for Bit-Plane Slicing."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.bit_plane = QSpinBox()
        self.bit_plane.setRange(0, 7)
        self.bit_plane.setValue(7)
        layout.addRow("Bit Plane (0=LSB, 7=MSB):", self.bit_plane)

        self.channel = QComboBox()
        self.channel.addItems(['Grayscale', 'Red', 'Green', 'Blue'])
        layout.addRow("Channel:", self.channel)

    def get_params(self) -> dict:
        return {
            'bit_plane': self.bit_plane.value(),
            'channel': self.channel.currentText(),
        }


class FrequencyForgeryParamsWidget(BaseParamsWidget):
    """Parameters for Frequency Forgery Detection."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.frequency_mode = QComboBox()
        self.frequency_mode.addItems(['High Pass', 'Low Pass', 'Band Pass'])
        layout.addRow("Frequency Mode:", self.frequency_mode)

        self.radius = QSpinBox()
        self.radius.setRange(1, 200)
        self.radius.setValue(30)
        layout.addRow("Radius (pixels):", self.radius)

        self.amplification = QDoubleSpinBox()
        self.amplification.setRange(1.0, 20.0)
        self.amplification.setValue(5.0)
        self.amplification.setSingleStep(0.5)
        layout.addRow("Amplification:", self.amplification)

    def get_params(self) -> dict:
        return {
            'frequency_mode': self.frequency_mode.currentText(),
            'radius': self.radius.value(),
            'amplification': self.amplification.value(),
        }


class PRNUParamsWidget(BaseParamsWidget):
    """Parameters for PRNU Camera Fingerprinting (single-image noise residual)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.denoise_kernel_size = QSpinBox()
        self.denoise_kernel_size.setRange(3, 15)
        self.denoise_kernel_size.setValue(5)
        self.denoise_kernel_size.setSingleStep(2)
        layout.addRow("Denoise Kernel Size:", self.denoise_kernel_size)

        self.residual_strength = QDoubleSpinBox()
        self.residual_strength.setRange(1.0, 20.0)
        self.residual_strength.setValue(5.0)
        self.residual_strength.setSingleStep(0.5)
        layout.addRow("Residual Strength:", self.residual_strength)

        self.display_mode = QComboBox()
        self.display_mode.addItems(['Residual', 'Normalized Residual'])
        layout.addRow("Display Mode:", self.display_mode)

    def get_params(self) -> dict:
        return {
            'denoise_kernel_size': self.denoise_kernel_size.value(),
            'residual_strength': self.residual_strength.value(),
            'display_mode': self.display_mode.currentText(),
        }


class BloodSpatterParamsWidget(BaseParamsWidget):
    """Parameters for Blood Spatter Analysis (candidate stain detection)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.red_dominance = QDoubleSpinBox()
        self.red_dominance.setRange(1.0, 3.0)
        self.red_dominance.setValue(1.5)
        self.red_dominance.setSingleStep(0.1)
        layout.addRow("Red Dominance:", self.red_dominance)

        self.darkness_threshold = QSpinBox()
        self.darkness_threshold.setRange(0, 255)
        self.darkness_threshold.setValue(200)
        layout.addRow("Darkness Threshold:", self.darkness_threshold)

        self.min_region_size = QSpinBox()
        self.min_region_size.setRange(1, 500)
        self.min_region_size.setValue(10)
        layout.addRow("Minimum Region Size:", self.min_region_size)

    def get_params(self) -> dict:
        return {
            'red_dominance': self.red_dominance.value(),
            'darkness_threshold': self.darkness_threshold.value(),
            'min_region_size': self.min_region_size.value(),
        }


# ═══════════════════════════════════════════════════════════
# Controls Widget — connects UI to module manager
# ═══════════════════════════════════════════════════════════

class MaarijForensicControlsWidget(QWidget):
    # Signal emitted when user clicks Apply, carries the params dict
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Maarij Forensic</h3>"))
        layout.addWidget(QLabel("<i>Visual analysis only — not forensic evidence</i>"))
        layout.addWidget(QLabel("Operation:"))

        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        # Stacked widget shows only the params for the selected operation
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Map operation names to their parameter widget classes
        operations = {
            "Fingerprint Enhancement": FingerprintParamsWidget,
            "ELA — Forgery Detection": ELAParamsWidget,
            "Security Document Analysis": SecurityDocumentParamsWidget,
            "Surveillance Deblur — Wiener Filter": WienerDeblurParamsWidget,
            "Latent Image Recovery": LatentRecoveryParamsWidget,
            "IR Ink Detection": IRInkParamsWidget,
            "Bit-Plane Slicing": BitPlaneParamsWidget,
            "Frequency Forgery Detection": FrequencyForgeryParamsWidget,
            "PRNU Camera Fingerprinting": PRNUParamsWidget,
            "Blood Spatter Analysis": BloodSpatterParamsWidget,
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
        params['operation'] = operation_name  # Add operation key so process_image knows what to run
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])


# ═══════════════════════════════════════════════════════════
# Main Module Class
# ═══════════════════════════════════════════════════════════

class MaarijForensicImageModule(IImageModule):

    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Maarij Forensic"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = MaarijForensicControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                pass  # Keep RGB or RGBA as is
            elif image_data.ndim == 2:
                pass  # Keep grayscale as (H, W)
            else:
                print(f"Warning: Unexpected image shape {image_data.shape}")
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')

        if operation == "Fingerprint Enhancement":
            processed_data = self._apply_fingerprint_enhancement(processed_data, params)

        elif operation == "ELA — Forgery Detection":
            processed_data = self._apply_ela(processed_data, params)

        elif operation == "Security Document Analysis":
            processed_data = self._apply_security_document_analysis(processed_data, params)

        elif operation == "Surveillance Deblur — Wiener Filter":
            processed_data = self._apply_wiener_deblur(processed_data, params)

        elif operation == "Latent Image Recovery":
            processed_data = self._apply_latent_recovery(processed_data, params)

        elif operation == "IR Ink Detection":
            processed_data = self._apply_ir_ink_proxy(processed_data, params)

        elif operation == "Bit-Plane Slicing":
            processed_data = self._apply_bit_plane_slicing(processed_data, params)

        elif operation == "Frequency Forgery Detection":
            processed_data = self._apply_frequency_analysis(processed_data, params)

        elif operation == "PRNU Camera Fingerprinting":
            processed_data = self._apply_prnu_residual(processed_data, params)

        elif operation == "Blood Spatter Analysis":
            processed_data = self._apply_candidate_stain_analysis(processed_data, params)

        # Safety cleanup before returning
        processed_data = np.nan_to_num(processed_data, nan=0.0, posinf=255.0, neginf=0.0)
        processed_data = np.clip(processed_data, 0, 255)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data

    # ─────────────────────────────────────────────────────────
    # Operation 1: Fingerprint Enhancement
    # Enhances ridge/valley contrast — visual aid only.
    # ─────────────────────────────────────────────────────────
    def _apply_fingerprint_enhancement(self, image_data, params):
        contrast_strength = float(params.get('contrast_strength', 1.5))
        blur_kernel_size = int(params.get('blur_kernel_size', 3))
        ridge_threshold = int(params.get('ridge_threshold', 0))

        # Step 1: Convert to grayscale
        gray = _to_grayscale(image_data)

        # Step 2: Stretch contrast so ridges and valleys span the full range
        mn, mx = np.min(gray), np.max(gray)
        if mx > mn:
            gray = (gray - mn) / (mx - mn) * 255.0
        else:
            return _gray_to_output(gray, image_data)

        # Step 3: Create a smoothed (blurred) version of the image
        k = _safe_odd(blur_kernel_size, image_data)
        blurred = _gaussian_blur(gray, k)

        # Step 4: High-frequency detail = original minus blurred
        detail = gray - blurred

        # Step 5: Add amplified detail back to enhance ridges
        enhanced = gray + contrast_strength * detail

        # Step 6: Optional threshold to focus only on prominent ridges
        if ridge_threshold > 0:
            enhanced = np.where(enhanced >= ridge_threshold, enhanced, 0.0)

        result = np.clip(enhanced, 0, 255)
        return _gray_to_output(result, image_data)

    # ─────────────────────────────────────────────────────────
    # Operation 2: ELA — Forgery Detection
    # NumPy ELA-like block residual — NOT true JPEG ELA.
    # True ELA requires JPEG re-encoding with a codec; this is an approximation.
    # Shows regions with high local inconsistency (possible inconsistencies).
    # ─────────────────────────────────────────────────────────
    def _apply_ela(self, image_data, params):
        block_size = int(params.get('block_size', 8))
        amplification = float(params.get('amplification', 5.0))
        threshold = int(params.get('threshold', 10))

        block_size = max(1, block_size)

        # Work on grayscale for a clear residual map
        gray = _to_grayscale(image_data)
        h, w = gray.shape

        # Build a block-averaged (quantized) version of the image
        # Each block is replaced with its mean — this approximates block compression
        approximated = gray.copy()
        for row in range(0, h, block_size):
            for col in range(0, w, block_size):
                block = gray[row:row + block_size, col:col + block_size]
                approximated[row:row + block_size, col:col + block_size] = np.mean(block)

        # Residual shows how much each pixel differs from its block's average
        residual = np.abs(gray - approximated)

        # Amplify and suppress values below threshold
        amplified = residual * amplification
        amplified[amplified < threshold] = 0

        result = np.clip(amplified, 0, 255)
        return _gray_to_output(result, image_data)

    # ─────────────────────────────────────────────────────────
    # Operation 3: Security Document Analysis
    # Highlights possible visual inconsistencies — not proof of forgery.
    # ─────────────────────────────────────────────────────────
    def _apply_security_document_analysis(self, image_data, params):
        detail_strength = float(params.get('detail_strength', 2.0))
        edge_threshold = int(params.get('edge_threshold', 30))
        grid_size = int(params.get('grid_size', 7))

        gray = _to_grayscale(image_data)
        k = _safe_odd(grid_size, image_data)

        # Step 1: Blur to create a smooth reference
        blurred = _gaussian_blur(gray, k)

        # Step 2: High-frequency detail map
        detail = np.abs(gray - blurred) * detail_strength

        # Step 3: Sobel edge detection (NumPy convolution)
        sobel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]], dtype=np.float32)
        gx = _convolve_channel(gray, sobel_x)
        gy = _convolve_channel(gray, sobel_y)
        edges = np.sqrt(gx * gx + gy * gy)

        # Step 4: Grid-based local variance map
        # High variance in a grid cell may indicate suspicious local changes
        h, w = gray.shape
        gs = max(1, int(grid_size))
        variance_map = np.zeros((h, w), dtype=np.float32)
        for row in range(0, h, gs):
            for col in range(0, w, gs):
                block = gray[row:row + gs, col:col + gs]
                block_var = float(np.var(block))
                variance_map[row:row + gs, col:col + gs] = block_var

        # Normalize variance map to 0-255
        v_max = np.max(variance_map)
        if v_max > 0:
            variance_map = variance_map / v_max * 255.0

        # Step 5: Combine all maps into a single suspicious-region map
        combined = detail + edges
        combined[combined < edge_threshold] = 0
        combined = combined + variance_map * 0.3

        result = np.clip(combined, 0, 255)
        return _gray_to_output(result, image_data)

    # ─────────────────────────────────────────────────────────
    # Operation 4: Surveillance Deblur — Wiener Filter
    # Uses NumPy FFT for Wiener deconvolution of motion blur.
    # ─────────────────────────────────────────────────────────
    def _apply_wiener_deblur(self, image_data, params):
        blur_length = int(params.get('blur_length', 10))
        blur_angle = float(params.get('blur_angle', 0))
        noise_ratio = float(params.get('noise_ratio', 0.01))

        # Ensure noise ratio is positive to avoid divide-by-zero
        noise_ratio = max(1e-6, noise_ratio)

        # Build the motion blur PSF (point spread function)
        psf = _create_motion_blur_kernel(blur_length, blur_angle)

        def deblur_one_channel(channel_2d):
            """Apply Wiener deconvolution to a single 2D channel."""
            h, w = channel_2d.shape

            # Place PSF in an array the same size as the channel
            psf_padded = np.zeros((h, w), dtype=np.float32)
            ph, pw = psf.shape
            # Copy PSF to top-left, then roll so its center lands at pixel (0,0)
            psf_padded[:ph, :pw] = psf
            psf_padded = np.roll(psf_padded, -(ph // 2), axis=0)
            psf_padded = np.roll(psf_padded, -(pw // 2), axis=1)

            # FFT of image and PSF
            F_img = np.fft.fft2(channel_2d.astype(np.float32))
            F_psf = np.fft.fft2(psf_padded)

            # Wiener filter: H* / (|H|^2 + K)  where K is the noise ratio
            H_conj = np.conj(F_psf)
            H_sq = np.abs(F_psf) ** 2
            wiener = H_conj / (H_sq + noise_ratio)

            # Apply filter and go back to spatial domain
            F_restored = F_img * wiener
            restored = np.fft.ifft2(F_restored).real
            return restored

        img_float = image_data.astype(np.float32)

        if img_float.ndim == 2:
            result = deblur_one_channel(img_float)
            return _gray_to_output(np.clip(result, 0, 255), image_data)

        # Process RGB channels separately; skip alpha if present
        num_color_ch = min(3, img_float.shape[2])
        channels_out = []
        for c in range(num_color_ch):
            channels_out.append(deblur_one_channel(img_float[:, :, c]))

        result = np.stack(channels_out, axis=2)

        # Preserve alpha channel if original image had one
        if img_float.shape[2] == 4:
            alpha = img_float[:, :, 3:4]
            result = np.concatenate([result, alpha], axis=2)

        return np.clip(result, 0, 255)

    # ─────────────────────────────────────────────────────────
    # Operation 5: Latent Image Recovery
    # Amplifies weak, low-contrast details to make them visible.
    # ─────────────────────────────────────────────────────────
    def _apply_latent_recovery(self, image_data, params):
        local_contrast = float(params.get('local_contrast', 2.0))
        detail_strength = float(params.get('detail_strength', 2.0))
        threshold = int(params.get('threshold', 0))

        # Step 1: Convert to grayscale
        gray = _to_grayscale(image_data)

        # Step 2: Stretch contrast to use the full 0-255 range
        mn, mx = np.min(gray), np.max(gray)
        if mx > mn:
            gray = (gray - mn) / (mx - mn) * 255.0
        else:
            return _gray_to_output(gray, image_data)

        # Step 3: Local mean using a large Gaussian blur
        large_k = _safe_odd(15, image_data)
        local_mean = _gaussian_blur(gray, large_k)

        # Step 4: Local deviation from the mean
        deviation = gray - local_mean

        # Step 5: Amplify local contrast
        enhanced = local_mean + local_contrast * deviation
        enhanced = np.clip(enhanced, 0, 255)

        # Step 6: Extract and add fine detail using a small blur
        small_k = _safe_odd(3, image_data)
        small_blur = _gaussian_blur(enhanced, small_k)
        fine_detail = enhanced - small_blur
        enhanced = enhanced + detail_strength * fine_detail

        # Step 7: Optional threshold to suppress background noise
        if threshold > 0:
            enhanced = np.where(enhanced >= threshold, enhanced, 0.0)

        result = np.clip(enhanced, 0, 255)
        return _gray_to_output(result, image_data)

    # ─────────────────────────────────────────────────────────
    # Operation 6: IR Ink Detection (Color Channel Proxy Analysis)
    # NOTE: Real IR ink detection requires an IR sensor/camera.
    # This analyzes color channel differences as a visual proxy only.
    # ─────────────────────────────────────────────────────────
    def _apply_ir_ink_proxy(self, image_data, params):
        channel_diff = params.get('channel_diff', 'R-G')
        enhancement_strength = float(params.get('enhancement_strength', 3.0))
        threshold = int(params.get('threshold', 20))

        # Grayscale images have no color channel to compare — return contrast enhanced version
        if image_data.ndim == 2:
            gray = image_data.astype(np.float32)
            mn, mx = np.min(gray), np.max(gray)
            if mx > mn:
                result = (gray - mn) / (mx - mn) * 255.0
            else:
                result = gray.copy()
            return np.clip(result, 0, 255)

        img = image_data.astype(np.float32)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # Compute the selected channel difference
        if channel_diff == 'R-G':
            diff = r - g
        elif channel_diff == 'R-B':
            diff = r - b
        else:  # G-B
            diff = g - b

        # Keep only positive differences (one channel stronger than the other)
        positive_diff = np.maximum(diff, 0.0)

        # Amplify to make subtle differences visible
        amplified = positive_diff * enhancement_strength

        # Suppress values below threshold
        amplified[amplified < threshold] = 0

        result = np.clip(amplified, 0, 255)
        return _gray_to_output(result, image_data)

    # ─────────────────────────────────────────────────────────
    # Operation 7: Bit-Plane Slicing
    # Extracts a single bit layer from a selected channel.
    # ─────────────────────────────────────────────────────────
    def _apply_bit_plane_slicing(self, image_data, params):
        bit_index = int(params.get('bit_plane', 7))
        channel_name = params.get('channel', 'Grayscale')

        # Clamp bit index to valid range 0-7
        bit_index = max(0, min(7, bit_index))

        # Convert to uint8 so bitwise operations work correctly
        img_uint8 = np.clip(image_data, 0, 255).astype(np.uint8)

        # Select the channel to analyze
        if image_data.ndim == 2 or channel_name == 'Grayscale':
            channel = _to_grayscale(img_uint8).astype(np.uint8)
        elif channel_name == 'Red' and image_data.ndim == 3:
            channel = img_uint8[:, :, 0]
        elif channel_name == 'Green' and image_data.ndim == 3:
            channel = img_uint8[:, :, 1]
        elif channel_name == 'Blue' and image_data.ndim == 3:
            channel = img_uint8[:, :, 2]
        else:
            channel = _to_grayscale(img_uint8).astype(np.uint8)

        # Extract the bit: shift right to bring desired bit to position 0, then mask
        bit_plane = (channel >> bit_index) & 1

        # Scale: 0 → 0, 1 → 255 for a visible black-and-white result
        result = (bit_plane * 255).astype(np.float32)
        return _gray_to_output(result, image_data)

    # ─────────────────────────────────────────────────────────
    # Operation 8: Frequency Forgery Detection
    # Shows possible texture inconsistencies via FFT frequency filtering.
    # ─────────────────────────────────────────────────────────
    def _apply_frequency_analysis(self, image_data, params):
        frequency_mode = params.get('frequency_mode', 'High Pass')
        radius = int(params.get('radius', 30))
        amplification = float(params.get('amplification', 5.0))

        radius = max(1, radius)

        # Step 1: Convert to grayscale
        gray = _to_grayscale(image_data)
        h, w = gray.shape

        # Step 2: Compute 2D FFT and shift DC component to the center
        F = np.fft.fft2(gray)
        F_shifted = np.fft.fftshift(F)

        # Step 3: Build a grid of distances from the center frequency (DC)
        cy, cx = h // 2, w // 2
        y_indices = np.arange(h) - cy
        x_indices = np.arange(w) - cx
        # meshgrid creates 2D grids of x and y coordinates
        xx, yy = np.meshgrid(x_indices, y_indices)
        dist = np.sqrt(xx * xx + yy * yy)

        # Step 4: Build frequency mask based on selected mode
        if frequency_mode == 'Low Pass':
            # Keep only low frequencies (near center)
            mask = (dist <= radius).astype(np.float32)
        elif frequency_mode == 'High Pass':
            # Keep only high frequencies (far from center)
            mask = (dist > radius).astype(np.float32)
        else:  # Band Pass
            # Keep a ring of frequencies between radius and radius*2
            inner = radius
            outer = radius * 2
            mask = ((dist > inner) & (dist <= outer)).astype(np.float32)

        # Step 5: Apply mask and inverse FFT to get back to spatial domain
        F_filtered = F_shifted * mask
        F_back = np.fft.ifftshift(F_filtered)
        spatial_result = np.fft.ifft2(F_back)

        # Step 6: Take magnitude and amplify
        magnitude = np.abs(spatial_result)
        amplified = magnitude * amplification

        result = np.clip(amplified, 0, 255)
        return _gray_to_output(result, image_data)

    # ─────────────────────────────────────────────────────────
    # Operation 9: PRNU Camera Fingerprinting
    # NOTE: True PRNU requires many images from the same camera.
    # This is a single-image estimated noise residual only.
    # Shows the estimated noise pattern — not camera identification.
    # ─────────────────────────────────────────────────────────
    def _apply_prnu_residual(self, image_data, params):
        kernel_size = int(params.get('denoise_kernel_size', 5))
        residual_strength = float(params.get('residual_strength', 5.0))
        display_mode = params.get('display_mode', 'Residual')

        k = _safe_odd(kernel_size, image_data)
        img_float = image_data.astype(np.float32)

        # Step 1: Create a smoothed (denoised) version with Gaussian blur
        denoised = _gaussian_blur(img_float, k)

        # Step 2: Noise residual = original - smoothed
        residual = img_float - denoised

        # Step 3: Scale by residual strength
        residual = residual * residual_strength

        if display_mode == 'Normalized Residual':
            # Normalize each channel to 0-255 so subtle patterns are visible
            if residual.ndim == 2:
                mn, mx = np.min(residual), np.max(residual)
                if mx > mn:
                    residual = (residual - mn) / (mx - mn) * 255.0
                else:
                    residual = np.zeros_like(residual)
            else:
                for c in range(min(3, residual.shape[2])):
                    ch = residual[:, :, c]
                    mn, mx = np.min(ch), np.max(ch)
                    if mx > mn:
                        residual[:, :, c] = (ch - mn) / (mx - mn) * 255.0
                    else:
                        residual[:, :, c] = 0.0
        else:
            # Plain residual: shift to 128 so negative values are visible as gray
            residual = residual + 128.0

        result = np.clip(residual, 0, 255)

        # If grayscale input, return grayscale output matched to original shape
        if image_data.ndim == 2:
            return result
        return result

    # ─────────────────────────────────────────────────────────
    # Operation 10: Blood Spatter Analysis
    # Shows candidate stain regions based on color dominance.
    # NOT a confirmation of blood — only a visual estimation tool.
    # ─────────────────────────────────────────────────────────
    def _apply_candidate_stain_analysis(self, image_data, params):
        red_dominance = float(params.get('red_dominance', 1.5))
        darkness_threshold = int(params.get('darkness_threshold', 200))
        min_region_size = int(params.get('min_region_size', 10))

        # Grayscale images have no color channel — return copy unchanged
        if image_data.ndim == 2:
            return image_data.astype(np.float32)

        img = image_data.astype(np.float32)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        # Average of green and blue as reference brightness
        avg_gb = (g + b) / 2.0

        # Condition 1: Red must clearly dominate green and blue
        red_dominates = r > (red_dominance * avg_gb + 1e-6)

        # Condition 2: Pixel must not be too bright (dark red / crimson range)
        not_too_bright = r < float(darkness_threshold)

        # Combine into a binary candidate mask
        candidate_mask = red_dominates & not_too_bright

        # Clean the mask: erosion removes small spots, dilation restores larger ones
        erosion_size = max(1, int(np.sqrt(float(min_region_size)) / 2.0))
        if erosion_size >= 2:
            eroded = _erode_binary(candidate_mask, erosion_size)
            cleaned_mask = _dilate_binary(eroded, erosion_size)
        else:
            cleaned_mask = candidate_mask

        # Create overlay: highlight candidate stain regions in red tint
        output = img.copy()
        output[:, :, 0] = np.where(cleaned_mask, np.minimum(r * 1.5, 255.0), r)
        output[:, :, 1] = np.where(cleaned_mask, g * 0.4, g)
        output[:, :, 2] = np.where(cleaned_mask, b * 0.4, b)
        # Alpha channel (index 3) is not touched — remains unchanged if present

        return np.clip(output, 0, 255)
