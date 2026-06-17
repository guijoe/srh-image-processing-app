import numpy as np
import imageio
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget)
from PySide6.QtCore import Qt, Signal
from modules.i_image_module import IImageModule

# --- Custom Parameter Widgets ---

class KaleidoscopeParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.label = QLabel("Segments (8):")
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(2, 16)
        self.slider.setValue(8)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.label.setText(f"Segments ({value}):")

    def get_params(self) -> dict:
        return {'segments': self.slider.value()}

class FilmGrainParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.intensity_label = QLabel("Grain Intensity (20):")
        layout.addWidget(self.intensity_label)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(1, 80)
        self.intensity_slider.setValue(20)
        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        layout.addWidget(self.intensity_slider)

        self.size_label = QLabel("Grain Size (1.0 — fine):")
        layout.addWidget(self.size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 30)
        self.size_slider.setValue(1)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        layout.addWidget(self.size_slider)

    def _on_intensity_changed(self, value):
        self.intensity_label.setText(f"Grain Intensity ({value}):")

    def _on_size_changed(self, value):
        self.size_label.setText(f"Grain Size ({value / 10.0:.1f} — {'fine' if value <= 5 else 'coarse'}):")

    def get_params(self) -> dict:
        return {
            'intensity': self.intensity_slider.value(),
            'grain_size': self.size_slider.value() / 10.0
        }

class LaplacianParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Kernel Type:"))
        self.kernel_selector = QComboBox()
        self.kernel_selector.addItems(["4-Neighbor (Cross)", "8-Neighbor (Full)"])
        layout.addWidget(self.kernel_selector)

        self.label = QLabel("Edge Threshold (0 = Raw):")
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.label.setText(f"Edge Threshold ({value}):")

    def get_params(self) -> dict:
        return {
            'kernel_type': self.kernel_selector.currentIndex(),
            'threshold': self.slider.value()
        }


class GaussianBlurParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.sigma_label = QLabel("Sigma / Spread (1.0):")
        layout.addWidget(self.sigma_label)
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setRange(1, 50)
        self.sigma_slider.setValue(10)
        self.sigma_slider.valueChanged.connect(self._on_sigma_changed)
        layout.addWidget(self.sigma_slider)

        self.size_label = QLabel("Kernel Size (5x5):")
        layout.addWidget(self.size_label)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(3, 21)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        layout.addWidget(self.size_slider)

    def _on_sigma_changed(self, value):
        self.sigma_label.setText(f"Sigma / Spread ({value / 10.0:.1f}):")

    def _on_size_changed(self, value):
        if value % 2 == 0:
            value += 1
            self.size_slider.setValue(value)
        self.size_label.setText(f"Kernel Size ({value}x{value}):")

    def get_params(self) -> dict:
        return {'sigma': self.sigma_slider.value() / 10.0,
                'kernel_size': self.size_slider.value()}

class UnsharpMaskParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.strength_label = QLabel("Sharpening Strength (1.0x):")
        layout.addWidget(self.strength_label)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(1, 50)
        self.strength_slider.setValue(10)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        layout.addWidget(self.strength_slider)

        self.sigma_label = QLabel("Blur Radius / Sigma (1.0):")
        layout.addWidget(self.sigma_label)
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setRange(1, 30)
        self.sigma_slider.setValue(10)
        self.sigma_slider.valueChanged.connect(self._on_sigma_changed)
        layout.addWidget(self.sigma_slider)

    def _on_strength_changed(self, value):
        self.strength_label.setText(f"Sharpening Strength ({value / 10.0:.1f}x):")

    def _on_sigma_changed(self, value):
        self.sigma_label.setText(f"Blur Radius / Sigma ({value / 10.0:.1f}):")

    def get_params(self) -> dict:
        return {'strength': self.strength_slider.value() / 10.0,
                'sigma': self.sigma_slider.value() / 10.0}

class HistogramEQParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Redistributes pixel intensities\nto maximise contrast using\nthe cumulative distribution function."))
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class VignetteParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.strength_label = QLabel("Vignette Strength (1.5):")
        layout.addWidget(self.strength_label)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(1, 80)
        self.strength_slider.setValue(15)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        layout.addWidget(self.strength_slider)

    def _on_strength_changed(self, value):
        self.strength_label.setText(f"Vignette Strength ({value / 10.0:.1f}):")

    def get_params(self) -> dict:
        return {'strength': self.strength_slider.value() / 10.0}

# --- Main Module UI ---

class TaseebControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Taseeb Image Suite</h3>"))

        layout.addWidget(QLabel("Select Filter Matrix:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Kaleidoscope": KaleidoscopeParamsWidget,
            "Film Grain": FilmGrainParamsWidget,
            "Gaussian Blur": GaussianBlurParamsWidget,
            "Unsharp Masking": UnsharpMaskParamsWidget,
            "Histogram Equalization": HistogramEQParamsWidget,
            "Vignette Effect": VignetteParamsWidget,
            "Laplacian Edge Detection": LaplacianParamsWidget
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Execute Processing")
        self.apply_button.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold;")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply_clicked(self):
        op = self.operation_selector.currentText()
        params = self.param_widgets[op].get_params()
        params['operation'] = op
        self.process_requested.emit(params)

    def _on_operation_changed(self, name):
        self.params_stack.setCurrentWidget(self.param_widgets[name])

# --- Main Module Logic ---

class TaseebImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Taseeb Toolkit"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = TaseebControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 3 and image_data.shape[2] == 4:
                image_data = image_data[:, :, :3]

            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error: {e}")
            return False, None, {}, None

    def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        k = size // 2
        coords = np.arange(-k, k + 1, dtype=float)
        x, y = np.meshgrid(coords, coords)
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def _execute_convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Vectorized sliding-window convolution using NumPy only."""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2

        extended_grid = np.pad(image, ((ph, ph), (pw, pw)), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(extended_grid, (kh, kw))
        return np.einsum('ijkl,kl->ij', windows, kernel)

    def _process_channel(self, matrix: np.ndarray, op: str, params: dict) -> np.ndarray:
        if op == "Kaleidoscope":
            h, w = matrix.shape
            cy, cx = h / 2.0, w / 2.0
            n = params.get('segments', 8)
            y_idx, x_idx = np.indices((h, w))
            r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
            theta = np.arctan2(y_idx - cy, x_idx - cx) % (2 * np.pi)
            sector = 2 * np.pi / n
            theta_mod = theta % sector
            theta_folded = np.where(theta_mod < sector / 2, theta_mod, sector - theta_mod)
            src_x = np.clip((r * np.cos(theta_folded) + cx).astype(int), 0, w - 1)
            src_y = np.clip((r * np.sin(theta_folded) + cy).astype(int), 0, h - 1)
            result = matrix[src_y, src_x]

        elif op == "Film Grain":
            intensity = params.get('intensity', 20.0)
            grain_size = params.get('grain_size', 1.0)
            noise = np.random.normal(0, intensity, matrix.shape)
            if grain_size > 1.0:
                k_size = max(3, int(6 * grain_size + 1) | 1)
                kernel = self._gaussian_kernel(k_size, grain_size)
                noise = self._execute_convolution(noise, kernel)
                std = noise.std()
                if std > 0:
                    noise = noise / std * intensity
            result = matrix + noise

        elif op == "Laplacian Edge Detection":
            if params.get('kernel_type', 0) == 1:
                kernel = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=float)
            else:
                kernel = np.array([[ 0, -1,  0],
                                   [-1,  4, -1],
                                   [ 0, -1,  0]], dtype=float)
            result = np.abs(self._execute_convolution(matrix, kernel))
            threshold = params.get('threshold', 0)
            if threshold > 0:
                result = np.where(result >= threshold, result, 0.0)

        elif op == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            k_size = params.get('kernel_size', 5)
            if k_size % 2 == 0:
                k_size += 1
            kernel = self._gaussian_kernel(k_size, sigma)
            result = self._execute_convolution(matrix, kernel)

        elif op == "Unsharp Masking":
            strength = params.get('strength', 1.0)
            sigma = params.get('sigma', 1.0)
            k_size = max(3, int(6 * sigma + 1) | 1)
            kernel = self._gaussian_kernel(k_size, sigma)
            blurred = self._execute_convolution(matrix, kernel)
            result = matrix + strength * (matrix - blurred)

        elif op == "Histogram Equalization":
            flat = np.clip(matrix, 0, 255).astype(np.uint8).flatten()
            hist, _ = np.histogram(flat, bins=256, range=(0, 256))
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0].min()
            lut = (cdf - cdf_min) / max(matrix.size - cdf_min, 1) * 255.0
            result = lut[np.clip(matrix, 0, 255).astype(np.uint8)]

        elif op == "Vignette Effect":
            h, w = matrix.shape
            strength = params.get('strength', 1.5)
            cy, cx = h / 2.0, w / 2.0
            y, x = np.ogrid[:h, :w]
            dist_sq = ((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2
            mask = np.exp(-strength * dist_sq / 2.0)
            result = matrix * mask

        else:
            return matrix

        return result

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        op = params.get('operation')
        matrix = image_data.astype(float)

        if matrix.ndim == 3:
            result = np.stack(
                [self._process_channel(matrix[:, :, ch], op, params) for ch in range(matrix.shape[2])],
                axis=-1
            )
        else:
            result = self._process_channel(matrix, op, params)

        return np.clip(result, 0, 255).astype(image_data.dtype)