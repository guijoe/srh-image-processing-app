from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio  # For general image loading

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore


# ====================================================================== #
#  Pure-math helper functions (no scipy / skimage – only numpy)           #
# ====================================================================== #

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA image to grayscale using the luminosity formula.

    Y = 0.2989·R + 0.5870·G + 0.1140·B
    """
    if img.ndim == 3 and img.shape[2] >= 3:
        return (0.2989 * img[:, :, 0].astype(np.float64)
                + 0.5870 * img[:, :, 1].astype(np.float64)
                + 0.1140 * img[:, :, 2].astype(np.float64))
    return img.astype(np.float64)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Perform a straightforward 2-D spatial convolution.

    For every output pixel (x, y):
        out(x, y) = Σ_i Σ_j  image(x+i, y+j) · kernel(i, j)
    where the kernel is centred on the pixel.  Zero-padding is used at
    the borders.
    """
    img = image.astype(np.float64)
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Zero-pad the input image
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    rows, cols = img.shape
    output = np.zeros_like(img, dtype=np.float64)

    for x in range(rows):
        for y in range(cols):
            region = padded[x:x + kh, y:y + kw]
            output[x, y] = np.sum(region * kernel)

    return output


def _median_filter_2d(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Median filter – pure sliding-window implementation.

    Output(x, y) = median({ Input(x+i, y+j) | -k ≤ i, j ≤ k })
    where k = kernel_size // 2.
    """
    img = image.astype(np.float64)
    k = kernel_size // 2
    rows, cols = img.shape

    # Zero-pad
    padded = np.pad(img, ((k, k), (k, k)), mode='constant')
    output = np.zeros_like(img, dtype=np.float64)

    for x in range(rows):
        for y in range(cols):
            neighbourhood = padded[x:x + kernel_size, y:y + kernel_size]
            output[x, y] = np.median(neighbourhood)

    return output


def _build_gaussian_kernel(sigma: float, size: int = None) -> np.ndarray:
    """Build a 2-D Gaussian kernel from the formula:

        G(x, y) = (1 / (2·π·σ²)) · exp( -(x² + y²) / (2·σ²) )

    *size* defaults to 6·σ + 1 (rounded to the nearest odd integer) so
    that the kernel captures ±3σ.
    """
    if size is None:
        size = int(np.ceil(6 * sigma)) | 1          # ensure odd
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()   # normalise so pixel brightness is preserved
    return kernel


def _dft2d(image: np.ndarray) -> np.ndarray:
    """Compute the 2-D Discrete Fourier Transform.

        F(u, v) = Σ_{x=0}^{M-1} Σ_{y=0}^{N-1}
                      f(x, y) · exp( -j·2π·(u·x/M + v·y/N) )

    Uses numpy's FFT (Cooley-Tukey algorithm) which computes the same
    mathematical transform in O(MN·log(MN)) instead of O(M²N²).
    """
    return np.fft.fft2(image.astype(np.float64))


def _idft2d(F: np.ndarray) -> np.ndarray:
    """Compute the 2-D Inverse Discrete Fourier Transform.

        f(x, y) = (1/(M·N)) Σ_{u=0}^{M-1} Σ_{v=0}^{N-1}
                      F(u, v) · exp( +j·2π·(u·x/M + v·y/N) )

    Uses numpy's inverse FFT for the same mathematical result.
    """
    return np.fft.ifft2(F)

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

class NoiseGenerationParamsWidget(BaseParamsWidget):
    """A widget for generating noise."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Noise Parameters:"))
        
        self.mean_spinbox = QDoubleSpinBox()
        self.mean_spinbox.setMinimum(-100.0)
        self.mean_spinbox.setMaximum(100.0)
        self.mean_spinbox.setValue(0.0)
        self.mean_spinbox.setSingleStep(0.1)
        layout.addWidget(self.mean_spinbox)
        
        self.std_dev_spinbox = QDoubleSpinBox()
        self.std_dev_spinbox.setMinimum(0.0)
        self.std_dev_spinbox.setMaximum(100.0)
        self.std_dev_spinbox.setValue(0.1)
        self.std_dev_spinbox.setSingleStep(0.1)
        layout.addWidget(self.std_dev_spinbox)
        
        layout.addStretch()
        
    def get_params(self) -> dict:
        return {
            'mean': self.mean_spinbox.value(),
            'std_dev': self.std_dev_spinbox.value()
        }

class SaturationBoostParamsWidget(BaseParamsWidget):
    """Widget for per-pixel saturation boost.

    Each R, G, B channel is multiplied by a scalar k so that the
    saturation is boosted.  k is expressed as a percentage (0-50 %).
    A value of 0 % means no change; 50 % means each channel value is
    scaled by 1.5.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Saturation Boost k (%):"))

        # Slider from 0 to 50 (represents percentage)
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(0)
        self.k_slider.setMaximum(50)
        self.k_slider.setValue(10)
        self.k_slider.setTickInterval(5)
        self.k_slider.setTickPosition(QSlider.TicksBelow)

        self.k_label = QLabel(f"{self.k_slider.value()} %")
        self.k_slider.valueChanged.connect(lambda v: self.k_label.setText(f"{v} %"))

        row = QHBoxLayout()
        row.addWidget(self.k_slider)
        row.addWidget(self.k_label)
        layout.addLayout(row)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'k_percent': self.k_slider.value()}


class MedianFilterParamsWidget(BaseParamsWidget):
    """Widget for median-filter denoising.

    The user selects the kernel (window) size – must be an odd integer.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Kernel Size (odd):"))

        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(3)
        self.size_spinbox.setMaximum(15)
        self.size_spinbox.setSingleStep(2)  # keep it odd
        self.size_spinbox.setValue(3)
        layout.addWidget(self.size_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        size = self.size_spinbox.value()
        # Enforce odd
        if size % 2 == 0:
            size += 1
        return {'filter_size': size}


class FourierEdgeDetectParamsWidget(BaseParamsWidget):
    """Widget for FFT-based high-pass edge detection.

    The user controls the cut-off radius of the high-pass filter in the
    frequency domain.  Pixels within the radius around the DC component
    are zeroed, preserving only high-frequency (edge) information.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("High-Pass Cut-off Radius (px):"))

        self.radius_spinbox = QSpinBox()
        self.radius_spinbox.setMinimum(1)
        self.radius_spinbox.setMaximum(200)
        self.radius_spinbox.setValue(30)
        layout.addWidget(self.radius_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'cutoff_radius': self.radius_spinbox.value()}


# Define a custom control widget
class KushagraControlsWidget(QWidget):
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
            "Saturation Boost": SaturationBoostParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Median Filter": MedianFilterParamsWidget,
            "Fourier Edge Detect": FourierEdgeDetectParamsWidget,
            "Gaussian Blur": GaussianParamsWidget,
            "Noise Generation": NoiseGenerationParamsWidget,
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
        params['operation'] = operation_name # Add operation name to params
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])

class KushagraImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Kushagra Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = KushagraControlsWidget(module_manager, parent)
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
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]: # RGB or RGBA
                # napari handles this well, but for processing, sometimes a single channel is needed
                pass
            elif image_data.ndim == 2: # Grayscale
                image_data = image_data[np.newaxis, :] # Add a channel dimension for consistency if desired
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {'name': file_path.split('/')[-1]}
            # Add more metadata: original_shape, file_size, etc.
            return True, image_data, metadata, None # Session ID generated by store
        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()

        operation = params.get('operation')

        # ------------------------------------------------------------------ #
        # 1. Saturation Boost – per-pixel RGB scaling
        #    [R_new, G_new, B_new] = k · [R, G, B]
        #    k = 1 + k_percent / 100   (k_percent ∈ [0, 50])
        # ------------------------------------------------------------------ #
        if operation == "Saturation Boost":
            k_percent = params.get('k_percent', 10)
            # k_percent is 0–50 → multiplier 1.00 – 1.50
            k = 1.0 + k_percent / 100.0

            input_float = processed_data.astype(np.float64)

            # Per-pixel: multiply every R, G, B value by k
            boosted = k * input_float

            # Clamp to the valid range of the original dtype
            if np.issubdtype(image_data.dtype, np.integer):
                max_val = np.iinfo(image_data.dtype).max
            else:
                max_val = 1.0
            processed_data = np.clip(boosted, 0, max_val)

        # ------------------------------------------------------------------ #
        # 2. Sobel Edge Detection (pure math)
        #    Gx = [[-1, 0, 1],      Gy = [[-1, -2, -1],
        #          [-2, 0, 2],             [ 0,  0,  0],
        #          [-1, 0, 1]]             [ 1,  2,  1]]
        #    G  = sqrt(Gx² + Gy²)
        # ------------------------------------------------------------------ #
        elif operation == "Sobel Edge Detect":
            # Convert to grayscale
            gray = _to_grayscale(processed_data)

            # Define the two 3×3 Sobel kernels
            Gx = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float64)

            Gy = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float64)

            # Convolve with each kernel using our pure-math convolution
            edge_x = _convolve2d(gray, Gx)
            edge_y = _convolve2d(gray, Gy)

            # Gradient magnitude:  G = sqrt(Gx² + Gy²)
            processed_data = np.sqrt(edge_x ** 2 + edge_y ** 2)

        # ------------------------------------------------------------------ #
        # 3. Power Law (Gamma) Correction
        #    O = c · s^γ
        #    where s is the normalised input [0,1] and c scales back.
        # ------------------------------------------------------------------ #
        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            input_float = processed_data.astype(np.float64)

            # c = max value (so we normalise to [0,1], apply γ, scale back)
            c = np.max(input_float)
            if c > 0:
                s = input_float / c          # normalise
                output = c * np.power(s, gamma)   # O = c · s^γ
                processed_data = output

        # ------------------------------------------------------------------ #
        # 4. Median Filter (Denoising) – pure sliding-window
        #    Output(x,y) = median({ Input(x+i, y+j) | -k ≤ i,j ≤ k })
        # ------------------------------------------------------------------ #
        elif operation == "Median Filter":
            filter_size = params.get('filter_size', 3)
            if filter_size <= 1:
                return processed_data  # No change

            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                # Apply to each colour channel independently
                channels = []
                for ch in range(processed_data.shape[2]):
                    channels.append(
                        _median_filter_2d(processed_data[:, :, ch], filter_size)
                    )
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = _median_filter_2d(processed_data, filter_size)

        # ------------------------------------------------------------------ #
        # 6. Fourier Transform Edge Detection (spatial → spectral → spatial)
        #    F(u,v) = Σ_x Σ_y f(x,y)·exp(-j·2π·(ux/M + vy/N))
        #
        #    Uses np.fft.fft2 / np.fft.ifft2 (Cooley-Tukey FFT) which
        #    computes the same DFT summation in O(MN·log(MN)).
        #
        #    Steps:
        #      a) Convert to grayscale
        #      b) Zero-pad to 3× the image size (image placed at centre)
        #      c) Compute 2-D FFT (forward DFT)
        #      d) Apply circular high-pass mask (zero inside cutoff radius)
        #      e) Inverse FFT back to spatial domain
        #      f) Crop the result back to the original image size
        # ------------------------------------------------------------------ #
        elif operation == "Fourier Edge Detect":
            cutoff = params.get('cutoff_radius', 30)

            # (a) Grayscale
            gray = _to_grayscale(processed_data)
            orig_rows, orig_cols = gray.shape

            # (b) Zero-pad to 3× the image size; original sits in the middle
            pad_rows, pad_cols = orig_rows * 3, orig_cols * 3
            padded = np.zeros((pad_rows, pad_cols), dtype=np.float64)
            start_r, start_c = orig_rows, orig_cols   # top-left of centre block
            padded[start_r:start_r + orig_rows, start_c:start_c + orig_cols] = gray

            # (c) Forward 2-D FFT
            F = np.fft.fft2(padded)

            # (d) High-pass mask in frequency domain
            crow, ccol = pad_rows // 2, pad_cols // 2
            Y, X = np.ogrid[:pad_rows, :pad_cols]
            dist = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
            mask = np.ones((pad_rows, pad_cols), dtype=np.float64)
            mask[dist <= cutoff] = 0.0          # suppress low frequencies

            # Shift DC to centre, apply mask, shift back
            F_shifted = np.fft.fftshift(F)
            F_filtered = F_shifted * mask
            F_unshifted = np.fft.ifftshift(F_filtered)

            # (e) Inverse 2-D FFT
            spatial = np.fft.ifft2(F_unshifted)

            # (f) Crop back to the original image region and take magnitude
            cropped = np.abs(spatial[start_r:start_r + orig_rows,
                                     start_c:start_c + orig_cols])
            processed_data = cropped

        # ------------------------------------------------------------------ #
        #  Gaussian Blur – pure-math spatial convolution
        #  G(x,y) = (1/(2πσ²))·exp(-(x²+y²)/(2σ²))
        # ------------------------------------------------------------------ #
        elif operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            g_kernel = _build_gaussian_kernel(sigma)

            input_float = processed_data.astype(np.float64)
            if input_float.ndim == 3 and input_float.shape[2] in [3, 4]:
                channels = []
                for ch in range(input_float.shape[2]):
                    channels.append(_convolve2d(input_float[:, :, ch], g_kernel))
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = _convolve2d(input_float, g_kernel)

        # ------------------------------------------------------------------ #
        # Ensure output dimensions match input dimensions
        # (Napari crashes if a 3D layer is updated with 2D data in-place)
        # ------------------------------------------------------------------ #
        if image_data.ndim == 3 and processed_data.ndim == 2:
            num_channels = image_data.shape[2]
            # Copy grayscale data into R, G, B
            channels = [processed_data] * min(3, num_channels)
            # If original had an alpha channel, preserve it
            if num_channels == 4:
                channels.append(image_data[:, :, 3].astype(np.float64))
            processed_data = np.stack(channels, axis=-1)

        # Ensure output data type is consistent
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data