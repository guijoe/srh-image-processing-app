from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QSpinBox
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# Custom UI Parameter Widget for Min/Max Operations
class IntensityParamsWidget(QWidget):
    """A widget specifically for capturing user-defined I_min and I_max values."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Minimum Intensity Spinbox
        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.imin_spinbox = QSpinBox()
        self.imin_spinbox.setMinimum(0)
        self.imin_spinbox.setMaximum(255)
        self.imin_spinbox.setValue(0)
        layout.addWidget(self.imin_spinbox)
        
        # Maximum Intensity Spinbox
        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.imax_spinbox = QSpinBox()
        self.imax_spinbox.setMinimum(0)
        self.imax_spinbox.setMaximum(255)
        self.imax_spinbox.setValue(255)
        layout.addWidget(self.imax_spinbox)
        
        layout.addStretch()

    def get_params(self) -> dict:
        """Packages the UI inputs to send to the processing function."""
        return {
            'i_min': self.imin_spinbox.value(),
            'i_max': self.imax_spinbox.value()
        }

# Main Control Panel Widget
class ShravanControlsWidget(QWidget):
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

        # 3 transformations
        operations = {
            "Contrast Stretch": IntensityParamsWidget,
            "Logarithmic Depth Correction": IntensityParamsWidget,
            "Bleaching  Map": IntensityParamsWidget,
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

# Core Plugin
class ShravanModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Shravan Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = ShravanControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image: {e}")
            return False, None, {}, None

    # NumPy processing
    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        operation = params.get('operation')
        i_min = params.get('i_min', 0)
        i_max = params.get('i_max', 255)
        
        # Work on a copy to avoid converting original state
        img_float = image_data.astype(np.float64)

        if operation == "Contrast Stretch":
            restored = np.copy(img_float)
            
            if img_float.ndim == 2:
                A, B = np.percentile(img_float, 2), np.percentile(img_float, 98)
                if A >= B: 
                    A, B = np.min(img_float), np.max(img_float)
                if A < B: 
                    clipped = np.clip(img_float, A, B)
                    restored = ((clipped - A) / (B - A)) * (i_max - i_min) + i_min
                    
            else:
                for c in range(min(3, img_float.shape[2])): 
                    channel = img_float[:, :, c]
                    A, B = np.percentile(channel, 2), np.percentile(channel, 98)
                    if A >= B: 
                        A, B = np.min(channel), np.max(channel)
                    if A < B:
                        clipped = np.clip(channel, A, B)
                        restored[:, :, c] = ((clipped - A) / (B - A)) * (i_max - i_min) + i_min
            
            # Convert to final array 
            final_output = np.clip(restored, 0, 255).astype(np.uint8)
            
            
            # this forces the top-left corner to have absolute black and absolute white
            # because of napari?
            final_output[0, 0] = 0     # pure black
            final_output[0, 1] = 255   # pure white
            
            return final_output

        elif operation == "Logarithmic Depth Correction":
            log_transformed = np.log1p(img_float)
            A, B = np.min(log_transformed), np.max(log_transformed)
            
            if A == B:
                return np.full_like(image_data, i_min).astype(np.uint8)
                
            scaled_log = ((log_transformed - A) / (B - A)) * (i_max - i_min) + i_min
            return np.clip(scaled_log, 0, 255).astype(np.uint8)

        elif operation == "Bleaching Map":
            # Handle grayscale images
            if img_float.ndim == 2:
                gray = img_float
            # Handle RGB images using standard luminosity weights
            else:
                gray = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
            
            threshold = np.percentile(gray, 85)
            bleach_mask = gray > threshold
            
            creative_map = np.copy(img_float)
            creative_map[~bleach_mask] = np.maximum(creative_map[~bleach_mask] * 0.3, i_min)
            
            # Apply max to all channels for RGB, or just the single matrix for grayscale
            if img_float.ndim == 3:
                creative_map[bleach_mask] = [i_max, i_max, i_max][:creative_map.shape[2]]
            else:
                creative_map[bleach_mask] = i_max
                
            return np.clip(creative_map, 0, 255).astype(np.uint8)
       # Force napari to turn off auto-scaling by anchoring the dynamic range
        final_output = np.clip(restored, 0, 255).astype(np.uint8)
        
        # Apply anchors to all channels if it's RGB
        if final_output.ndim == 3:
            final_output[0, 0] = [0, 0, 0][:final_output.shape[2]]
            final_output[0, 1] = [255, 255, 255][:final_output.shape[2]]
        else:
            final_output[0, 0] = 0
            final_output[0, 1] = 255
            
        return final_output

        return image_data