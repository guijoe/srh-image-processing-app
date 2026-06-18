from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)


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


class PixelationParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Pixel Size:"))

        self.pixel_spinbox = QDoubleSpinBox()
        self.pixel_spinbox.setMinimum(2)
        self.pixel_spinbox.setMaximum(20)
        self.pixel_spinbox.setValue(8)

        layout.addWidget(self.pixel_spinbox)

    def get_params(self):
        return {
            "pixel_size": int(self.pixel_spinbox.value())
        }
    
# Define a custom control widget
class BoitumeloControlsWidget(QWidget):
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
            "Image Negative": NoParamsWidget,
            "Histogram Equalization": NoParamsWidget,
            "Pixelation": PixelationParamsWidget,
            "Sky Sports 2004" : NoParamsWidget,
            "Game Boy Camera": NoParamsWidget,
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

class BoitumeloImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Boitumelo Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = BoitumeloControlsWidget(module_manager, parent)
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

        # Work on a copy so the original image is not changed
        processed_data = image_data.copy()

        operation = params.get('operation')

        # Invert image colours
        if operation == "Image Negative":

            max_val = np.iinfo(image_data.dtype).max
            processed_data = max_val - processed_data

        # Spread pixel values across the full intensity range
        elif operation == "Histogram Equalization":

            flat = processed_data.flatten()

            hist = np.bincount(flat, minlength=256)

            cdf = hist.cumsum()

            cdf_normalized = (cdf - cdf.min()) * 255
            cdf_normalized = cdf_normalized / (cdf.max() - cdf.min())

            processed_data = cdf_normalized[flat].reshape(processed_data.shape)

        # Adjustable pixelation effect
        elif operation == "Pixelation":

            processed_data = processed_data.astype(float)

            pixel_size = params.get("pixel_size", 8)

            original_height = image_data.shape[0]
            original_width = image_data.shape[1]

            processed_data = processed_data[::pixel_size, ::pixel_size]

            processed_data = processed_data.repeat(pixel_size, axis=0)
            processed_data = processed_data.repeat(pixel_size, axis=1)

            processed_data = processed_data[:original_height, :original_width]

            processed_data = np.clip(processed_data, 0, 255)

        # Inspired by early 2000s football broadcasts
        elif operation == "Sky Sports 2004":

            processed_data = processed_data.astype(float)

            original_height = image_data.shape[0]
            original_width = image_data.shape[1]

            # Make the image look lower resolution
            pixel_size = 6
            processed_data = processed_data[::pixel_size, ::pixel_size]

            processed_data = processed_data.repeat(pixel_size, axis=0).repeat(pixel_size, axis=1)

            processed_data = processed_data[:original_height, :original_width]

            # Desaturate the colours slightly
            gray = processed_data.mean(axis=2, keepdims=True)
            processed_data = gray + (processed_data - gray) * 0.65

            # Small green tint
            processed_data[:, :, 1] *= 1.08

            # Simulate colour bleed
            processed_data[:, :, 1] = (
                processed_data[:, :, 1] * 0.8 +
                np.roll(processed_data[:, :, 1], 1, axis=1) * 0.2
            )

            # Add scanlines
            processed_data[::2] *= 0.65

            # Add noise
            noise = np.random.normal(0, 8, processed_data.shape)
            processed_data = processed_data + noise

            processed_data = np.clip(processed_data, 0, 255)


        # Inspired by the original Game Boy Camera
        elif operation == "Game Boy Camera":

            processed_data = processed_data.astype(float)

            # Convert to grayscale
            gray = processed_data.mean(axis=2)

            # Limit image to 4 shades
            gray = np.floor(gray / 64) * 64

            # Add slight green tint
            processed_data = np.stack([
                gray * 0.90,
                gray * 1.00,
                gray * 0.90
            ], axis=2)

            processed_data = np.clip(processed_data, 0, 255)

        return processed_data