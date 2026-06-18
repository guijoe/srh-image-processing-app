from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout, QSpinBox, QCheckBox
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore



def _manual_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2D convolution with reflection padding (beginner-friendly loops)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return output


class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
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


class SolarizationParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Threshold:"))
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setMinimum(0)
        self.threshold_spinbox.setMaximum(255)
        self.threshold_spinbox.setValue(128)
        layout.addWidget(self.threshold_spinbox)
        hint = QLabel("Pixels below threshold stay the same; above are inverted.")
        hint.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(hint)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'threshold': self.threshold_spinbox.value()}



class KaleidoscopeParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Number of Sectors:"))
        self.sector_spinbox = QSpinBox()
        self.sector_spinbox.setMinimum(3)
        self.sector_spinbox.setMaximum(20)
        self.sector_spinbox.setValue(6)
        layout.addWidget(self.sector_spinbox)
        hint = QLabel("3 = triangle, 6 = flower, 12 = intricate kaleidoscope")
        hint.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(hint)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'segments': self.sector_spinbox.value()}


class PixelSortingParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.reverse_checkbox = QCheckBox("Sort Descending (Dark to Bright)")
        layout.addWidget(self.reverse_checkbox)
        hint = QLabel("Unchecked: Bright to Dark (standard glitch)")
        hint.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(hint)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'reverse': self.reverse_checkbox.isChecked()}



class UlasControlsWidget(QWidget):
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

        operations = {
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Thermal Heatmap": NoParamsWidget,
            "Solarization": SolarizationParamsWidget,
            "Kaleidoscope": KaleidoscopeParamsWidget,
            "Pixel Sorting (Glitch)": PixelSortingParamsWidget,
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



class UlasImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Ulas Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = UlasControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                pass
            elif image_data.ndim == 2:
                image_data = image_data[np.newaxis, :]
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        single_channel = (image_data.ndim == 3 and image_data.shape[0] == 1)
        working = image_data[0] if single_channel else image_data
        processed_data = working.copy()

        operation = params.get('operation')

        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)

        elif operation == "Sobel Edge Detect":
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:,:,:3])
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
                    channels = []
                    for i in range(input_float.shape[2]):
                        channels.append(convolve(input_float[:,:,i], kernel, mode='reflect'))
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = convolve(input_float, kernel, mode='reflect')

        elif operation == "Thermal Heatmap":
            if working.ndim == 3 and working.shape[2] in (3, 4):
                gray = working[:, :, :3].astype(float).mean(axis=2)
            else:
                gray = working.astype(float)
            gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
            control_points = [0, 64, 128, 192, 255]
            xs = np.arange(256)
            red_lut = np.interp(xs, control_points, [0, 80, 255, 255, 255])
            green_lut = np.interp(xs, control_points, [0, 0, 0, 180, 255])
            blue_lut = np.interp(xs, control_points, [0, 120, 0, 0, 200])
            processed_data = np.stack([
                red_lut[gray_u8],
                green_lut[gray_u8],
                blue_lut[gray_u8],
            ], axis=-1)

        elif operation == "Solarization":
            threshold = params.get('threshold', 128)
            if np.issubdtype(working.dtype, np.integer):
                max_val = np.iinfo(working.dtype).max
            else:
                max_val = 255
            processed_data = np.where(working < threshold, working, max_val - working)

        elif operation == "Kaleidoscope":
            segments = params.get('segments', 6)
            H, W = working.shape[:2]
            cx, cy = W / 2.0, H / 2.0
            y, x = np.ogrid[:H, :W]
            x_rel = x - cx
            y_rel = y - cy
            r = np.sqrt(x_rel**2 + y_rel**2)
            theta = np.arctan2(y_rel, x_rel)

            sector_angle = 2 * np.pi / segments
            theta_norm = theta % (2 * np.pi)
            theta_folded = theta_norm % sector_angle
            sector_idx = (theta_norm / sector_angle).astype(int)
            theta_folded = np.where(sector_idx % 2 == 0, theta_folded, sector_angle - theta_folded)
            src_theta = theta_folded + (sector_angle / 2)

            src_x = cx + r * np.cos(src_theta)
            src_y = cy + r * np.sin(src_theta)
            src_x = np.clip(src_x, 0, W - 1).astype(int)
            src_y = np.clip(src_y, 0, H - 1).astype(int)

            if working.ndim == 3:
                processed_data = working[src_y, src_x, :]
            else:
                processed_data = working[src_y, src_x]

        elif operation == "Pixel Sorting (Glitch)":
            reverse = params.get('reverse', False)
            if working.ndim == 3 and working.shape[2] >= 3:
                brightness = working[:, :, :3].mean(axis=2)
            else:
                brightness = working

            sorted_output = np.zeros_like(working)
            for i in range(working.shape[0]):
                row = working[i]
                b_row = brightness[i]
                sort_idx = np.argsort(b_row)
                if reverse:
                    sort_idx = sort_idx[::-1]
                sorted_output[i] = row[sort_idx]
            processed_data = sorted_output

        processed_data = processed_data.astype(image_data.dtype)

        if single_channel and processed_data.ndim == 2:
            processed_data = processed_data[np.newaxis, :]

        return processed_data