import numpy as np
import imageio
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSlider, 
                             QPushButton, QComboBox, QStackedWidget)
from PySide6.QtCore import Qt, Signal
from modules.i_image_module import IImageModule

# --- Custom Parameter Widgets ---

class BrightnessParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Brightness Offset (-255 to 255):"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-255, 255)
        self.slider.setValue(0)
        layout.addWidget(self.slider)
        
    def get_params(self) -> dict:
        return {'value': self.slider.value()}

class ContrastParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.min_label = QLabel("New Min Intensity (Target: 0):")
        layout.addWidget(self.min_label)
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 127)
        self.min_slider.setValue(0)
        self.min_slider.valueChanged.connect(self._on_min_changed)
        layout.addWidget(self.min_slider)

        self.max_label = QLabel("New Max Intensity (Target: 255):")
        layout.addWidget(self.max_label)
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(128, 255)
        self.max_slider.setValue(255)
        self.max_slider.valueChanged.connect(self._on_max_changed)
        layout.addWidget(self.max_slider)

    def _on_min_changed(self, value):
        self.min_label.setText(f"New Min Intensity (Target: {value}):")

    def _on_max_changed(self, value):
        self.max_label.setText(f"New Max Intensity (Target: {value}):")
        
    def get_params(self) -> dict:
        return {'new_min': self.min_slider.value(), 'new_max': self.max_slider.value()}

class SwirlParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Swirl Twist Strength (Factor: 0.0):")
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100) # Re-mapped to 0.0 -> 10.0
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.label.setText(f"Swirl Twist Strength (Factor: {value / 10.0:.1f}):")
        
    def get_params(self) -> dict:
        return {'strength': self.slider.value() / 10.0}

class PixelateParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Pixel Block Size (1px - No Effect):")
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 32)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.label.setText(f"Pixel Block Size ({value}px x {value}px):")
        
    def get_params(self) -> dict:
        return {'block_size': self.slider.value()}

class GaussianBlurParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel("Blur Intensity (Sigma: 1.0):")
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 50) 
        self.slider.setValue(10)     
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.label.setText(f"Blur Intensity (Sigma: {value / 10.0:.1f}):")
        
    def get_params(self) -> dict:
        return {'sigma': self.slider.value() / 10.0}

class SharpenParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel("Sharpen Strength (Factor: 1.0):")
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 30) 
        self.slider.setValue(10)    
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.label.setText(f"Sharpen Strength (Factor: {value / 10.0:.1f}):")
        
    def get_params(self) -> dict:
        return {'factor': self.slider.value() / 10.0}

class SobelParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel("Edge Thresholding (0 = Show All):")
        layout.addWidget(self.label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider)

    def _on_value_changed(self, value):
        self.label.setText(f"Edge Thresholding ({value}):")
        
    def get_params(self) -> dict:
        return {'threshold': self.slider.value()}

# --- Main Module UI ---

class ILFControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>ILF Image Suite</h3>"))

        layout.addWidget(QLabel("Select Algorithm:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Brightness Adjustment": BrightnessParamsWidget,
            "Contrast Stretching": ContrastParamsWidget,
            "Swirl Distortion Filter": SwirlParamsWidget,
            "Block Pixelation Filter": PixelateParamsWidget,
            "Dynamic Gaussian Blur": GaussianBlurParamsWidget,
            "Adjustable Sharpen Filter": SharpenParamsWidget,
            "Sobel Edge Detection": SobelParamsWidget
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Run Algorithm")
        self.apply_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
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

class IlfImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "ILF Processing Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = ILFControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 3:
                image_data = np.mean(image_data, axis=2).astype(np.uint8)
            metadata = {'name': file_path.split('/')[-1], 'type': 'Grayscale'}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error: {e}")
            return False, None, {}, None

    def _manual_convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        img_h, img_w = image.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(image, dtype=float)
        
        for i in range(img_h):
            for j in range(img_w):
                region = padded_img[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(region * kernel)
        return output

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        op = params.get('operation')
        img = image_data.astype(float)
        h, w = img.shape
        
        if op == "Brightness Adjustment":
            val = params.get('value', 0)
            processed = img + val

        elif op == "Contrast Stretching":
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)
            current_min = np.min(img)
            current_max = np.max(img)
            if current_max == current_min:
                return image_data
            processed = (img - current_min) * ((new_max - new_min) / (current_max - current_min)) + new_min

        elif op == "Swirl Distortion Filter":
            strength = params.get('strength', 0.0)
            if strength == 0.0:
                return image_data
            
            processed = np.zeros_like(img)
            cx, cy = w / 2.0, h / 2.0
            # Maximum radius bounding limit setup
            max_radius = np.sqrt(cx**2 + cy**2)
            
            for y in range(h):
                for x in range(w):
                    dx = x - cx
                    dy = y - cy
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r < max_radius:
                        # Angle increases the closer it gets to the center point
                        theta = strength * np.exp(-r / (max_radius / 3.0))
                        current_angle = np.arctan2(dy, dx) + theta
                        
                        src_x = int(cx + r * np.cos(current_angle))
                        src_y = int(cy + r * np.sin(current_angle))
                        
                        # Boundary clipping validation
                        if 0 <= src_x < w and 0 <= src_y < h:
                            processed[y, x] = img[src_y, src_x]
                        else:
                            processed[y, x] = img[y, x]
                    else:
                        processed[y, x] = img[y, x]

        elif op == "Block Pixelation Filter":
            block_size = params.get('block_size', 1)
            if block_size <= 1:
                return image_data
            
            processed = np.copy(img)
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    # Bound slice windows inside spatial arrays cleanly
                    y_end = min(y + block_size, h)
                    x_end = min(x + block_size, w)
                    
                    block_mean = np.mean(img[y:y_end, x:x_end])
                    processed[y:y_end, x:x_end] = block_mean

        elif op == "Dynamic Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            size = 5
            kernel = np.zeros((size, size))
            center = size // 2
            for x in range(size):
                for y in range(size):
                    dx = x - center
                    dy = y - center
                    kernel[x, y] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            kernel /= np.sum(kernel)
            processed = self._manual_convolve(img, kernel)

        elif op == "Adjustable Sharpen Filter":
            factor = params.get('factor', 1.0)
            kernel = np.array([[ 0,       -factor,  0], 
                               [-factor, 1 + 4*factor, -factor], 
                               [ 0,       -factor,  0]])
            processed = self._manual_convolve(img, kernel)

        elif op == "Sobel Edge Detection":
            threshold = params.get('threshold', 0)
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            Gx = self._manual_convolve(img, Kx)
            Gy = self._manual_convolve(img, Ky)
            processed = np.sqrt(Gx**2 + Gy**2)
            if threshold > 0:
                processed = np.where(processed >= threshold, processed, 0.0)
        
        else:
            return image_data

        return np.clip(processed, 0, 255).astype(image_data.dtype)