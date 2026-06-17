from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# --- Parameter Widgets for Different Operations ---

def bilinear_interpolate(img, x, y):
    h, w = img.shape[:2]

    x = np.clip(x, 0, w - 2)
    y = np.clip(y, 0, h - 2)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    if img.ndim == 3:
        dx = dx[..., None]
        dy = dy[..., None]

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    return ((1 - dx) * (1 - dy) * Ia +
            dx * (1 - dy) * Ib +
            (1 - dx) * dy * Ic +
            dx * dy * Id)

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

class RotationsWidget(BaseParamsWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel("Angle(0-360):"))
            self.angle_spinbox = QDoubleSpinBox()
            self.angle_spinbox.setMinimum(0)
            self.angle_spinbox.setMaximum(360)
            self.angle_spinbox.setValue(0)
            self.angle_spinbox.setSingleStep(1)
            layout.addWidget(self.angle_spinbox)
            layout.addStretch()
        def get_params(self) -> dict:
            return {'angle': self.angle_spinbox.value()}

class ScalingsWidget(BaseParamsWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel("XScaling(%):"))
            self.xscale_spinbox = QDoubleSpinBox()
            self.xscale_spinbox.setMinimum(1)
            self.xscale_spinbox.setMaximum(1000)
            self.xscale_spinbox.setValue(100)
            self.xscale_spinbox.setSingleStep(1)
            layout.addWidget(self.xscale_spinbox)

            layout.addWidget(QLabel("YScaling(%):"))
            self.yscale_spinbox = QDoubleSpinBox()
            self.yscale_spinbox.setMinimum(1)
            self.yscale_spinbox.setMaximum(1000)
            self.yscale_spinbox.setValue(100)
            self.yscale_spinbox.setSingleStep(1)
            layout.addWidget(self.yscale_spinbox)
            layout.addStretch()
        def get_params(self) -> dict:
            return {'xscale': self.xscale_spinbox.value(),
                    'yscale': self.yscale_spinbox.value()}
        
class WaveDistortionWidget(BaseParamsWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel("Amplitude:"))
            self.amplitude_spinbox = QDoubleSpinBox()
            self.amplitude_spinbox.setMinimum(0)
            self.amplitude_spinbox.setMaximum(1000)
            self.amplitude_spinbox.setValue(100)
            self.amplitude_spinbox.setSingleStep(1)
            layout.addWidget(self.amplitude_spinbox)

            layout.addWidget(QLabel("Wavelength:"))
            self.wavelength_spinbox = QDoubleSpinBox()
            self.wavelength_spinbox.setMinimum(0)
            self.wavelength_spinbox.setMaximum(1000)
            self.wavelength_spinbox.setValue(100)
            self.wavelength_spinbox.setSingleStep(1)
            layout.addWidget(self.wavelength_spinbox)
            

            layout.addWidget(QLabel("direction:"))
            self.direction = QComboBox()
            self.direction.addItems(["vertical","horizontal"])
            layout.addWidget(self.direction)

            layout.addStretch()
        def get_params(self) -> dict:
            return {'amplitude': self.amplitude_spinbox.value(),
                    'wavelength': self.wavelength_spinbox.value(),
                    'direction': self.direction.currentText()
                
                    }
        
class DrosteWidget(BaseParamsWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel("scale(0.1):"))
            self.scale_spinbox = QDoubleSpinBox()
            self.scale_spinbox.setMinimum(0.1)
            self.scale_spinbox.setMaximum(1)
            self.scale_spinbox.setValue(1)
            self.scale_spinbox.setSingleStep(0.1)
            layout.addWidget(self.scale_spinbox)

            layout.addWidget(QLabel("iterations:"))
            self.iteration_spinbox = QDoubleSpinBox()
            self.iteration_spinbox.setMinimum(1)
            self.iteration_spinbox.setMaximum(1000)
            self.iteration_spinbox.setValue(100)
            self.iteration_spinbox.setSingleStep(1)
            layout.addWidget(self.iteration_spinbox)
            layout.addStretch()
        def get_params(self) -> dict:
            return {'scale': self.scale_spinbox.value(),
                    'iterations': self.iteration_spinbox.value()}
        
class SelectiveBlurWidget(BaseParamsWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel("radius:"))
            self.radius_spinbox = QDoubleSpinBox()
            self.radius_spinbox.setMinimum(1)
            self.radius_spinbox.setMaximum(1000)
            self.radius_spinbox.setValue(1)
            self.radius_spinbox.setSingleStep(1) 
            layout.addWidget(self.radius_spinbox)

            layout.addWidget(QLabel("kernel:"))
            self.kernel_spinbox = QDoubleSpinBox()
            self.kernel_spinbox.setMinimum(1)
            self.kernel_spinbox.setMaximum(13)
            self.kernel_spinbox.setValue(7)
            self.kernel_spinbox.setSingleStep(1)
            layout.addWidget(self.kernel_spinbox)

            layout.addWidget(QLabel("xcenter:"))
            self.xcenter_spinbox = QDoubleSpinBox()
            self.xcenter_spinbox.setMinimum(0)
            self.xcenter_spinbox.setMaximum(2000)
            self.xcenter_spinbox.setValue(1)
            self.xcenter_spinbox.setSingleStep(1) 
            layout.addWidget(self.xcenter_spinbox)

            layout.addWidget(QLabel("ycenter:"))
            self.ycenter_spinbox = QDoubleSpinBox()
            self.ycenter_spinbox.setMinimum(0)
            self.ycenter_spinbox.setMaximum(2000)
            self.ycenter_spinbox.setValue(1)
            self.ycenter_spinbox.setSingleStep(1) 
            layout.addWidget(self.ycenter_spinbox)

            layout.addStretch()
        def get_params(self) -> dict:
            return {'radius': self.radius_spinbox.value(),
                    'kernel': self.kernel_spinbox.value(),
                    'xcenter': self.xcenter_spinbox.value(),
                    'ycenter': self.ycenter_spinbox.value()}



# Define a custom control widget
class GrigorijsControlsWidget(QWidget):
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

            "Rotation": RotationsWidget,
            "Scaling": ScalingsWidget,
            "Wave distortion": WaveDistortionWidget,
            "Droste effect": DrosteWidget,
            "Selective blur":SelectiveBlurWidget
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

class GrigorijsImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Grigorijs Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = GrigorijsControlsWidget(module_manager, parent)
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



        if operation == "Rotation":

            angle = params.get('angle', 0.0)
            angle = np.radians(angle)
            height, width = image_data.shape[:2]
            cos_t = abs(np.cos(angle))
            sin_t = abs(np.sin(angle))

            new_w = int(height * sin_t + width * cos_t)
            new_h = int(height * cos_t + width * sin_t)
            cx_out, cy_out = new_w / 2, new_h / 2
            cx_in, cy_in = width / 2, height / 2

            output = np.zeros((new_h, new_w, *image_data.shape[2:]), dtype=image_data.dtype)

            cos_t = np.cos(angle)
            sin_t = np.sin(angle)

            for y_out in range(new_h):
                for x_out in range(new_w):

                    x_shift = x_out - cx_out
                    y_shift = y_out - cy_out

                    x_src = cos_t * x_shift + sin_t * y_shift + cx_in
                    y_src = -sin_t * x_shift + cos_t * y_shift + cy_in

                    if 0 <= x_src < width - 1 and 0 <= y_src < height - 1:
                        output[y_out, x_out] = bilinear_interpolate(
                            image_data,
                            x_src,
                            y_src
                        )
        elif operation == "Scaling":
            xscale = params.get('xscale', 100.0)/100
            yscale = params.get('yscale', 100.0)/100
            height, width = image_data.shape[:2]
            new_w = int(width * xscale)
            new_h = int(height * yscale)
            channels = image_data.shape[2]
            output = np.zeros((new_h, new_w, channels), dtype=image_data.dtype)
            for y_out in range(new_h):
                for x_out in range(new_w):

                    x_src = x_out / xscale
                    y_src = y_out / yscale

                    if 0 <= x_src < width - 1 and 0 <= y_src < height - 1:
                        output[y_out, x_out] = bilinear_interpolate(
                            image_data,
                            x_src,
                            y_src
                        )

        elif operation == "Wave distortion":
            height, width = image_data.shape[:2]
            amplitude = params.get('amplitude', 0.0)
            wavelength = params.get('wavelength', 0.0)
            direction = params.get('direction', 0.0)

            # Output image
            output = np.zeros_like(image_data)

            # Coordinate grid
            y, x = np.indices((height, width))

            # Avoid division errors
            wavelength = max(wavelength, 1)

            if direction == "horizontal":
                # Shift rows left/right using sine wave
                shift_x = amplitude * np.sin(2 * np.pi * y / wavelength)

                x_src = x - shift_x
                y_src = y

            else:
                # Shift columns up/down using sine wave
                shift_y = amplitude * np.sin(2 * np.pi * x / wavelength)

                x_src = x
                y_src = y - shift_y

            output = bilinear_interpolate(image_data, x_src, y_src)

        elif operation == "Droste effect":
            scale = params.get('scale', 1.0)
            iterations = int(params.get('iterations', 0))

            def resize_nearest(image, new_h, new_w):
                height, weight = image.shape[:2]

                y = np.linspace(0, height - 1, new_h).astype(int)
                x = np.linspace(0, weight - 1, new_w).astype(int)

                return image[y[:, None], x[None, :]]

  
            output = image_data.copy()
            current = image_data.copy()

            height, weight = image_data.shape[:2]

            for _ in range(iterations):
                new_h = max(1, int(current.shape[0] * scale))
                new_w = max(1, int(current.shape[1] * scale))

                current = resize_nearest(current, new_h, new_w)

                y0 = (height - new_h) // 2
                x0 = (weight - new_w) // 2

                output[y0:y0 + new_h, x0:x0 + new_w] = current



        elif operation == "Selective blur":
            height, width = image_data.shape[:2]
            ycenter = params.get('ycenter', 500)
            xcenter = params.get('xcenter', 500)
            radius = params.get('radius', 1)
            kernel = int(params.get('kernel', 7))
            def box_blur(image, k):
                """Simple fast blur using box averaging (NumPy only)."""
                pad = k // 2
                padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

                out = np.zeros_like(image, dtype=np.float32)

                for y in range(image.shape[0]):
                    for x in range(image.shape[1]):
                        out[y, x] = np.mean(
                            padded[y:y+k, x:x+k],
                            axis=(0, 1)
                        )

                return out

            # 1. blurred version
            blurred = box_blur(image_data, kernel)

            # 2. coordinate grid
            y, x = np.ogrid[:height, :width]

            # 3. circular mask
            dist = (x - xcenter)**2 + (y - ycenter)**2
            mask = dist <= radius**2

            # 4. output
            output = blurred.copy()
            output[mask] = image_data[mask]

        processed_data = output
        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data