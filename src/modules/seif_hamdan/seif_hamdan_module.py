from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
from PySide6.QtCore import Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule


class BaseParamsWidget(QWidget):
    """Base class for parameter widgets."""
    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    """Widget for transformations with no parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class ContrastStretchingParamsWidget(BaseParamsWidget):
    """Widget for Contrast Stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setMinimum(0.0)
        self.min_spinbox.setMaximum(255.0)
        self.min_spinbox.setValue(0.0)
        self.min_spinbox.setSingleStep(1.0)
        layout.addWidget(self.min_spinbox)

        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setMinimum(0.0)
        self.max_spinbox.setMaximum(255.0)
        self.max_spinbox.setValue(255.0)
        self.max_spinbox.setSingleStep(1.0)
        layout.addWidget(self.max_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "new_min": self.min_spinbox.value(),
            "new_max": self.max_spinbox.value()
        }


class GammaParamsWidget(BaseParamsWidget):
    """Widget for Gamma Correction."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Gamma Value:"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.05)
        self.gamma_spinbox.setMaximum(5.0)
        self.gamma_spinbox.setValue(0.7)
        self.gamma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.gamma_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "gamma": self.gamma_spinbox.value()
        }


class NeonGlowParamsWidget(BaseParamsWidget):
    """Widget for Neon Edge Glow."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Edge Threshold (0-255):"))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setMinimum(0.0)
        self.threshold_spinbox.setMaximum(255.0)
        self.threshold_spinbox.setValue(35.0)
        self.threshold_spinbox.setSingleStep(5.0)
        layout.addWidget(self.threshold_spinbox)

        layout.addWidget(QLabel("Glow Strength:"))
        self.strength_spinbox = QDoubleSpinBox()
        self.strength_spinbox.setMinimum(0.1)
        self.strength_spinbox.setMaximum(5.0)
        self.strength_spinbox.setValue(2.0)
        self.strength_spinbox.setSingleStep(0.1)
        layout.addWidget(self.strength_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "threshold": self.threshold_spinbox.value(),
            "strength": self.strength_spinbox.value()
        }


class PixelationParamsWidget(BaseParamsWidget):
    """Widget for Pixelation / Mosaic."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Block Size:"))
        self.block_spinbox = QDoubleSpinBox()
        self.block_spinbox.setMinimum(2.0)
        self.block_spinbox.setMaximum(80.0)
        self.block_spinbox.setValue(12.0)
        self.block_spinbox.setSingleStep(1.0)
        layout.addWidget(self.block_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "block_size": int(self.block_spinbox.value())
        }


class PosterizationParamsWidget(BaseParamsWidget):
    """Widget for Posterization."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Number of Intensity Levels:"))
        self.levels_spinbox = QDoubleSpinBox()
        self.levels_spinbox.setMinimum(2.0)
        self.levels_spinbox.setMaximum(16.0)
        self.levels_spinbox.setValue(5.0)
        self.levels_spinbox.setSingleStep(1.0)
        layout.addWidget(self.levels_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "levels": int(self.levels_spinbox.value())
        }


# -------------------------------
# Control Widget
# -------------------------------

class SeifControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Seif NumPy Image Transformations</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Contrast Stretching": ContrastStretchingParamsWidget,
            "Gamma Correction": GammaParamsWidget,
            "Histogram Equalization": NoParamsWidget,
            "Neon Edge Glow": NeonGlowParamsWidget,
            "Pixelation / Mosaic": PixelationParamsWidget,
            "Posterization": PosterizationParamsWidget,
            "Negative Image": NoParamsWidget,
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
        params["operation"] = operation_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])


# -------------------------------
# Main Image Module
# -------------------------------

class SeifImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Seif Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = SeifControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)

        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {"name": file_path.split("/")[-1]}
            return True, image_data, metadata, None

        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    # -------------------------------
    # Helper Functions
    # -------------------------------

    def _split_alpha(self, image: np.ndarray):
        """
        If image has an alpha channel, separate it so transformations do not destroy transparency.
        """
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, :3], image[:, :, 3]
        return image, None

    def _restore_alpha(self, image: np.ndarray, alpha):
        """
        Add alpha channel back after processing.
        """
        if alpha is not None:
            return np.dstack((image, alpha))
        return image

    def _clip_uint8(self, image: np.ndarray):
        """
        Keep pixel values inside valid 8-bit range.
        """
        return np.clip(image, 0, 255)

    # -------------------------------
    # NumPy Image Transformations
    # -------------------------------

    def contrast_stretching(self, image: np.ndarray, new_min: float, new_max: float):
        """
        Rescales the image intensity range.

        Formula:
        output = (pixel - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
        """
        img, alpha = self._split_alpha(image)
        img_float = img.astype(float)

        old_min = np.min(img_float)
        old_max = np.max(img_float)

        if old_max == old_min:
            return image

        stretched = (img_float - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
        stretched = self._clip_uint8(stretched)

        return self._restore_alpha(stretched, alpha)

    def gamma_correction(self, image: np.ndarray, gamma: float):
        """
        Applies power-law transformation.

        Formula:
        output = 255 * (input / 255) ^ gamma
        """
        img, alpha = self._split_alpha(image)
        img_float = img.astype(float)

        normalized = img_float / 255.0
        corrected = 255.0 * np.power(normalized, gamma)
        corrected = self._clip_uint8(corrected)

        return self._restore_alpha(corrected, alpha)

    def histogram_equalization(self, image: np.ndarray):
        """
        Improves contrast by spreading intensity values using the cumulative distribution function.
        Applied channel-by-channel for color images.
        """
        img, alpha = self._split_alpha(image)
        img_uint8 = self._clip_uint8(img).astype(np.uint8)

        def equalize_channel(channel):
            histogram = np.bincount(channel.flatten(), minlength=256)
            cdf = histogram.cumsum()

            nonzero_cdf = cdf[cdf > 0]
            if len(nonzero_cdf) == 0:
                return channel

            cdf_min = nonzero_cdf[0]
            total_pixels = channel.size

            if total_pixels == cdf_min:
                return channel

            lookup_table = ((cdf - cdf_min) / (total_pixels - cdf_min)) * 255
            lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)

            return lookup_table[channel]

        if img_uint8.ndim == 2:
            equalized = equalize_channel(img_uint8)
        else:
            channels = []
            for c in range(img_uint8.shape[2]):
                channels.append(equalize_channel(img_uint8[:, :, c]))
            equalized = np.stack(channels, axis=-1)

        return self._restore_alpha(equalized.astype(float), alpha)

    def neon_edge_glow(self, image: np.ndarray, threshold: float, strength: float):
        """
        Creates a neon glow effect by estimating edges from pixel intensity differences.
        This uses simple NumPy gradient calculations.
        """
        img, alpha = self._split_alpha(image)
        img_float = img.astype(float)

        if img_float.ndim == 3:
            gray = (
                0.299 * img_float[:, :, 0] +
                0.587 * img_float[:, :, 1] +
                0.114 * img_float[:, :, 2]
            )
        else:
            gray = img_float

        dx = np.zeros_like(gray)
        dy = np.zeros_like(gray)

        dx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
        dy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])

        gradient = np.sqrt(dx ** 2 + dy ** 2)

        max_gradient = np.max(gradient)
        if max_gradient > 0:
            gradient = (gradient / max_gradient) * 255

        edge_mask = gradient > threshold
        glow = (gradient / 255.0) * strength

        if img_float.ndim == 2:
            result = img_float + edge_mask * glow * 255
        else:
            result = img_float.copy()

            # Cyan / blue neon glow
            result[:, :, 0] = result[:, :, 0] + edge_mask * glow * 20
            result[:, :, 1] = result[:, :, 1] + edge_mask * glow * 180
            result[:, :, 2] = result[:, :, 2] + edge_mask * glow * 255

        result = self._clip_uint8(result)
        return self._restore_alpha(result, alpha)

    def pixelation(self, image: np.ndarray, block_size: int):
        """
        Pixelates the image by replacing each block with its average color.
        This connects to sampling and resolution.
        """
        img, alpha = self._split_alpha(image)
        img_float = img.astype(float)

        h, w = img_float.shape[:2]
        result = img_float.copy()

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = img_float[y:y + block_size, x:x + block_size]
                block_mean = np.mean(block, axis=(0, 1))
                result[y:y + block_size, x:x + block_size] = block_mean

        result = self._clip_uint8(result)
        return self._restore_alpha(result, alpha)

    def posterization(self, image: np.ndarray, levels: int):
        """
        Reduces the number of intensity levels.
        This connects to quantization.
        """
        img, alpha = self._split_alpha(image)
        img_float = img.astype(float)

        levels = max(2, levels)
        step = 255 / (levels - 1)

        posterized = np.round(img_float / step) * step
        posterized = self._clip_uint8(posterized)

        return self._restore_alpha(posterized, alpha)

    def negative_image(self, image: np.ndarray):
        """
        Creates a negative image by inverting intensity values.
        Formula:
        output = 255 - input
        """
        img, alpha = self._split_alpha(image)
        img_float = img.astype(float)

        negative = 255 - img_float
        negative = self._clip_uint8(negative)

        return self._restore_alpha(negative, alpha)

    # -------------------------------
    # Main Processing Function
    # -------------------------------

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        operation = params.get("operation")
        processed_data = image_data.copy()

        if operation == "Contrast Stretching":
            processed_data = self.contrast_stretching(
                processed_data,
                params.get("new_min", 0.0),
                params.get("new_max", 255.0)
            )

        elif operation == "Gamma Correction":
            processed_data = self.gamma_correction(
                processed_data,
                params.get("gamma", 1.0)
            )

        elif operation == "Histogram Equalization":
            processed_data = self.histogram_equalization(processed_data)

        elif operation == "Neon Edge Glow":
            processed_data = self.neon_edge_glow(
                processed_data,
                params.get("threshold", 35.0),
                params.get("strength", 2.0)
            )

        elif operation == "Pixelation / Mosaic":
            processed_data = self.pixelation(
                processed_data,
                params.get("block_size", 12)
            )

        elif operation == "Posterization":
            processed_data = self.posterization(
                processed_data,
                params.get("levels", 5)
            )

        elif operation == "Negative Image":
            processed_data = self.negative_image(processed_data)

        if np.issubdtype(image_data.dtype, np.integer):
            processed_data = np.clip(processed_data, 0, 255)

        processed_data = processed_data.astype(image_data.dtype)

        return processed_data