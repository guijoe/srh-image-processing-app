from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
from PySide6.QtCore import Signal
import numpy as np
import imageio  # For general image loading (can use Pillow too)
from skimage.color import rgb2gray
from modules.i_image_module import IImageModule


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


class LocalStretchParamsWidget(BaseParamsWidget):
    """A widget for the moving window size parameter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Local Window Neighborhood Size:"))
        self.window_spinbox = QDoubleSpinBox()
        self.window_spinbox.setMinimum(3.0)
        self.window_spinbox.setMaximum(101.0)
        self.window_spinbox.setValue(15.0)
        self.window_spinbox.setSingleStep(2.0)  # Enforce odd steps
        layout.addWidget(self.window_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        w_size = int(self.window_spinbox.value())
        # Enforce that the sliding dimension must be odd to have an exact center
        if w_size % 2 == 0:
            w_size += 1
        return {'window_size': w_size}


class HDRParamsWidget(BaseParamsWidget):
    """A widget for Logarithmic and Gamma Scaling fusion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Log Intensity Scaling Factor (k):"))
        self.k_spinbox = QDoubleSpinBox()
        self.k_spinbox.setMinimum(1.0)
        self.k_spinbox.setMaximum(100.0)
        self.k_spinbox.setValue(10.0)
        layout.addWidget(self.k_spinbox)

        layout.addWidget(QLabel("Gamma Factor:"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(5.0)
        self.gamma_spinbox.setValue(1.0)
        self.gamma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.gamma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'k_factor': self.k_spinbox.value(),
            'gamma': self.gamma_spinbox.value()
        }


class FrequencyParamsWidget(BaseParamsWidget):
    """A widget for setting boundaries in the frequency spectrum grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Notch Brush Radius (Pixels):"))
        self.inner_spinbox = QDoubleSpinBox()
        self.inner_spinbox.setMinimum(1.0)
        self.inner_spinbox.setMaximum(20.0)
        self.inner_spinbox.setValue(3.0)
        layout.addWidget(self.inner_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'r_inner': self.inner_spinbox.value()}


class SobelParamsWidget(BaseParamsWidget):
    """A widget for the Gradient blending factor multiplier."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Edge Boosting Multiplier (Alpha):"))
        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setMinimum(0.0)
        self.alpha_spinbox.setMaximum(10.0)
        self.alpha_spinbox.setValue(1.0)
        self.alpha_spinbox.setSingleStep(0.1)
        layout.addWidget(self.alpha_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'alpha': self.alpha_spinbox.value()}


class ColorParamsWidget(BaseParamsWidget):
    """A widget for adjusting color spectrum midpoint thresholds."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Red Component Level:"))
        self.r_spinbox = QDoubleSpinBox()
        self.r_spinbox.setMinimum(0.1)
        self.r_spinbox.setMaximum(3.0)
        self.r_spinbox.setValue(1.5)  # Boost red for a warm nebula feel
        layout.addWidget(self.r_spinbox)

        layout.addWidget(QLabel("Green Component Level:"))
        self.g_spinbox = QDoubleSpinBox()
        self.g_spinbox.setMinimum(0.1)
        self.g_spinbox.setMaximum(3.0)
        self.g_spinbox.setValue(0.8)  # Lower green slightly
        layout.addWidget(self.g_spinbox)

        layout.addWidget(QLabel("Blue Component Level:"))
        self.b_spinbox = QDoubleSpinBox()
        self.b_spinbox.setMinimum(0.1)
        self.b_spinbox.setMaximum(3.0)
        self.b_spinbox.setValue(0.5)  # Keep dark areas cool, but subtle
        layout.addWidget(self.b_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'alpha_r': self.r_spinbox.value(),
            'alpha_g': self.g_spinbox.value(),
            'alpha_b': self.b_spinbox.value()
        }


class ThresholdParamsWidget(BaseParamsWidget):
    """A widget for the upper and lower binary clipping range."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Minimum Filtering Cutoff (0-255):"))
        self.low_spinbox = QDoubleSpinBox()
        self.low_spinbox.setMinimum(0.0)
        self.low_spinbox.setMaximum(255.0)
        self.low_spinbox.setValue(50.0)
        layout.addWidget(self.low_spinbox)

        layout.addWidget(QLabel("Maximum Filtering Cutoff (0-255):"))
        self.high_spinbox = QDoubleSpinBox()
        self.high_spinbox.setMinimum(0.0)
        self.high_spinbox.setMaximum(255.0)
        self.high_spinbox.setValue(200.0)
        layout.addWidget(self.high_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            't_low': self.low_spinbox.value(),
            't_high': self.high_spinbox.value()
        }


# Define a custom control widget
class AyeshaControlsWidget(QWidget):
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
            "Local Contrast Stretch": LocalStretchParamsWidget,
            "Space Debris Cleaner": NoParamsWidget,
            "Galactic HDR Scaling": HDRParamsWidget,
            "Frequency Noise Stripper": FrequencyParamsWidget,
            "Sobel Edge Sharpener": SobelParamsWidget,
            "Laplacian Sharpener": NoParamsWidget,
            "Astro Negative Invert": NoParamsWidget,
            "Color Palette": ColorParamsWidget,
            "Space Object Isolator": ThresholdParamsWidget
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
        params['operation'] = operation_name  # Add operation name to params
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])


class AyeshaImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Ayesha Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]  # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = AyeshaControlsWidget(module_manager, parent)
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
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:  # RGB or RGBA
                # napari handles this well, but for processing, sometimes a single channel is needed
                # Drop alpha channel if it exists, then convert RGB to Grayscale
                image_data = rgb2gray(image_data[:, :, :3])
                # Scale back up to uint8 values if rgb2gray turned it into 0.0 - 1.0 floats
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255).astype(np.uint8)

            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None  # Session ID generated by store

        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict, orig_dtype=None) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')

        # -------------------------------------------------------------
        # 1. Local Contrast Stretch
        # -------------------------------------------------------------
        if operation == "Local Contrast Stretch":
            img_float = processed_data.astype(float)
            window_size = params.get('window_size', 15)
            half = window_size // 2
            h, w = img_float.shape[:2]

            padded = np.pad(img_float,((half, half), (half, half)), mode='edge')
            out = np.zeros_like(img_float)

            for i in range(h):
                for j in range(w):
                    patch = padded[i: i + window_size, j: j + window_size]
                    p_min, p_max = np.min(patch), np.max(patch)
                    if p_max > p_min:
                        out[i, j] = (img_float[i, j] - p_min) * (255.0 / (p_max - p_min))
                    else:
                        out[i, j] = img_float[i, j]

            processed_data = np.clip(out, 0, 255)
        # -------------------------------------------------------------
        # 2. Space Debris Cleaner
        # -------------------------------------------------------------
        elif operation == "Space Debris Cleaner":
            img_float = processed_data.astype(float)
            h, w = img_float.shape[:2]
            padded = np.pad(processed_data, ((1, 1), (1, 1)), 'edge')
            out = np.zeros_like(img_float)

            for i in range(h):
                for j in range(w):
                    out[i, j] = np.median(padded[i:i + 3, j:j + 3])

            processed_data = np.clip(out, 0, 255)

        # -------------------------------------------------------------
        # 3. Galactic HDR Scaling
        # -------------------------------------------------------------
        elif operation == "Galactic HDR Scaling":
            img_float = processed_data.astype(float)
            k = params.get('k_factor', 10.0)
            gamma = params.get('gamma', 1.0)

            max_val = np.max(img_float)
            if max_val > 0:
                c_factor = 255.0 / np.log(1.0 + k * max_val)
                log_mapped = c_factor * np.log(1.0 + k * processed_data)

                normalized = log_mapped / 255.0
                processed_data = np.power(normalized, gamma) * 255.0
            else:
                processed_data = img_float

        # -------------------------------------------------------------
        # 4. Frequency Noise Stripper
        # -------------------------------------------------------------
        elif operation == "Frequency Noise Stripper":
            img_float = processed_data.astype(float)
            M, N = img_float.shape
            P, Q = 2 * M, 2 * N

            # Pad to P x Q (double size) to avoid circular convolution wraparound
            f_padded = np.pad(img_float, ((0, M), (0, N)), 'constant', constant_values=0)

            # Create a matrix of the same size P x Q where every pixel is 1 or -1
            x = np.arange(P).reshape(-1, 1)
            y = np.arange(Q).reshape(1, -1)
            centering_matrix = (-1) ** (x + y)
            f_centered = f_padded * centering_matrix

            # Spectral domain forward FFT shift
            F = np.fft.fft2(f_centered)
            F_magnitude = np.abs(F)

            # Initialize a completely clear filter mask (pass everything)
            H = np.ones((P, Q))

            # Set an aggressive peak threshold to pick out extreme periodic noise spikes
            threshold = np.percentile(F_magnitude, 99.98)

            # Define a protection bubble around the center coordinates
            u0, v0 = P // 2, Q // 2
            safe_radius = 20

            # Compute the distance from center map
            D_center = np.sqrt((x - u0) ** 2 + (y - v0) ** 2)

            # Isolate outlier spikes that reside completely outside our central protection zone
            spike_locations = (F_magnitude > threshold) & (D_center > safe_radius)

            # Retrieve the exact row/column coordinate locations of the noise spikes
            spike_indices = np.argwhere(spike_locations)

            # Read brush size parameter from our UI slider
            brush_radius = int(params.get('r_inner', 3.0))

            # Dynamic Notch Painting: Loop over each detected point and zero it out safely
            for r, c in spike_indices:
                row_start = max(0, r - brush_radius)
                row_end = min(P, r + brush_radius + 1)
                col_start = max(0, c - brush_radius)
                col_end = min(Q, c + brush_radius + 1)

                H[row_start:row_end, col_start:col_end] = 0.0

            # Multiply spectrum grids directly (Convolution Theorem)
            G = F * H

            # Compute Inverse DFT.
            # The result is complex, we take the Real part.
            g_complex = np.fft.ifft2(G)
            g_real = np.real(g_complex)

            # The result is still multiplied by (-1)^(x+y), so we multiply again to reverse it.
            g_padded = g_real * centering_matrix

            # Crop padding borders off
            processed_data = g_padded[0:M, 0:N]

        # -------------------------------------------------------------
        # 5. Sobel Edge Sharpener
        # -------------------------------------------------------------
        elif operation == "Sobel Edge Sharpener":
            img_float = processed_data.astype(float)
            alpha = params.get('alpha', 1.0)
            h, w = img_float.shape

            w_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            w_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

            padded = np.pad(img_float, ((1, 1), (1, 1)), 'edge')
            G_x = np.zeros_like(img_float)
            G_y = np.zeros_like(img_float)

            for i in range(h):
                for j in range(w):
                    patch = padded[i:i + 3, j:j + 3]
                    G_x[i, j] = np.sum(patch * w_x)
                    G_y[i, j] = np.sum(patch * w_y)

            magnitude = np.sqrt(G_x ** 2 + G_y ** 2)
            if np.max(magnitude) > 0:
                magnitude = (magnitude / np.max(magnitude)) * 255.0

            processed_data = img_float + alpha * magnitude

        # -------------------------------------------------------------
        # 6. Laplacian Sharpener
        # -------------------------------------------------------------
        elif operation == "Laplacian Sharpener":
            img_float = processed_data.astype(float)
            h, w = img_float.shape[:2]
            w_laplacian = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

            padded = np.pad(img_float, ((1, 1), (1, 1)), 'edge')
            out = np.zeros_like(img_float)

            for i in range(h):
                for j in range(w):
                    out[i, j] = np.sum(padded[i:i + 3, j:j + 3] * w_laplacian)
            processed_data = np.clip(out, 0, 255)

        # -------------------------------------------------------------
        # 7. Astro Negative Invert
        # -------------------------------------------------------------
        elif operation == "Astro Negative Invert":
            img_float = processed_data.astype(float)
            processed_data = 255.0 - img_float

        # -------------------------------------------------------------
        # 8. Color Palette
        # -------------------------------------------------------------
        elif operation == "Color Palette":
            img_float = processed_data.astype(float)
            # Normalize original image to a crisp 0.0 - 1.0 spectrum scale
            norm_img = img_float / 255.0

            # Extract user weighting parameters from UI
            alpha_r = params.get('alpha_r', 1.5)
            alpha_g = params.get('alpha_g', 0.8)
            alpha_b = params.get('alpha_b', 0.5)

            # Build a classic astrophysical gas palette curve:
            # Dark space stays black/deep red, mid-tones glow vibrant orange,
            # and bright stellar structures burn bright gold/white.
            R_channel = np.clip((norm_img ** 1.2) * alpha_r * 255.0, 0, 255)
            G_channel = np.clip((norm_img ** 2.0) * alpha_g * 255.0, 0, 255)
            B_channel = np.clip((norm_img ** 3.5) * alpha_b * 255.0, 0, 255)

            processed_data = np.stack([R_channel, G_channel, B_channel], axis=-1)

        # -------------------------------------------------------------
        # 9. Space Object Isolator
        # -------------------------------------------------------------
        elif operation == "Space Object Isolator":
            img_float = processed_data.astype(float)
            t_low = params.get('t_low', 50.0)
            t_high = params.get('t_high', 200.0)

            binary_mask = np.zeros_like(img_float)
            binary_mask[(img_float >= t_low) & (img_float <= t_high)] = 255.0
            processed_data = binary_mask

        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = np.clip(processed_data, 0, 255)
        processed_data = processed_data.astype(image_data.dtype)
        return processed_data

