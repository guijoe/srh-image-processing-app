from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio                             
from modules.ash import filtering_utils
from modules.ash import gaussian_blur
from modules.ash import sobel_edge
from modules.ash import median_filter
from modules.ash import gamma_transform
from modules.ash import vintage_filter
from modules.ash import vignette_filter
from modules.ash import chromatic_aberration
from modules.ash import anime_filter
from modules.ash import fisheye_filter
from modules.ash import glow_filter
from modules.ash import halation_filter
from modules.ash import kaleidoscope_filter
from modules.ash import pixel_art_filter
from modules.ash import oil_painting
from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore
class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError
class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("No settings needed for this filter.")
        layout.addWidget(label)
    def get_params(self) -> dict:
        return {}
class GaussianParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Blur Strength (Sigma):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setValue(1.0)
        layout.addWidget(self.sigma_spinbox)
    def get_params(self) -> dict:
        return {'sigma': self.sigma_spinbox.value()}
class PowerLawParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Gamma Value:"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setValue(1.0)
        layout.addWidget(self.gamma_spinbox)
    def get_params(self) -> dict:
        return {'gamma': self.gamma_spinbox.value()}
class VintageParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Vintage Intensity:"))
        self.intensity_spinbox = QDoubleSpinBox()
        self.intensity_spinbox.setMinimum(0.0)
        self.intensity_spinbox.setMaximum(1.0)
        self.intensity_spinbox.setValue(1.0)
        self.intensity_spinbox.setSingleStep(0.1)
        layout.addWidget(self.intensity_spinbox)
    def get_params(self) -> dict:
        return {'intensity': self.intensity_spinbox.value()}
class VignetteParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Vignette Strength:"))
        self.strength_spinbox = QDoubleSpinBox()
        self.strength_spinbox.setMinimum(0.1)
        self.strength_spinbox.setMaximum(5.0)
        self.strength_spinbox.setValue(1.0)
        self.strength_spinbox.setSingleStep(0.1)
        layout.addWidget(self.strength_spinbox)
    def get_params(self) -> dict:
        return {'strength': self.strength_spinbox.value()}
class ChromaticParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Color Shift (Pixels):"))
        self.shift_spinbox = QDoubleSpinBox()
        self.shift_spinbox.setMinimum(0.0)
        self.shift_spinbox.setMaximum(100.0)
        self.shift_spinbox.setValue(5.0)
        self.shift_spinbox.setSingleStep(1.0)
        layout.addWidget(self.shift_spinbox)
    def get_params(self) -> dict:
        return {'shift_amount': self.shift_spinbox.value()}
class AnimeParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Color Levels (Lower = Flatter):"))
        self.levels_spinbox = QDoubleSpinBox()
        self.levels_spinbox.setMinimum(2.0)
        self.levels_spinbox.setMaximum(20.0)
        self.levels_spinbox.setValue(5.0)
        self.levels_spinbox.setSingleStep(1.0)
        layout.addWidget(self.levels_spinbox)
        layout.addWidget(QLabel("Edge Threshold (Lower = More Lines):"))
        self.edge_spinbox = QDoubleSpinBox()
        self.edge_spinbox.setMinimum(0.01)
        self.edge_spinbox.setMaximum(1.0)
        self.edge_spinbox.setValue(0.2)
        self.edge_spinbox.setSingleStep(0.05)
        layout.addWidget(self.edge_spinbox)
    def get_params(self) -> dict:
        return {
            'levels': self.levels_spinbox.value(),
            'threshold': self.edge_spinbox.value()
        }
class FisheyeParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Distortion (Negative = Bulge):"))
        self.dist_spinbox = QDoubleSpinBox()
        self.dist_spinbox.setMinimum(-1.0)
        self.dist_spinbox.setMaximum(1.0)
        self.dist_spinbox.setValue(-0.3)
        self.dist_spinbox.setSingleStep(0.1)
        layout.addWidget(self.dist_spinbox)
    def get_params(self) -> dict:
        return {'distortion': self.dist_spinbox.value()}
class GlowParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Glow Intensity:"))
        self.intensity_spinbox = QDoubleSpinBox()
        self.intensity_spinbox.setMinimum(0.0)
        self.intensity_spinbox.setMaximum(2.0)
        self.intensity_spinbox.setValue(0.6)
        self.intensity_spinbox.setSingleStep(0.1)
        layout.addWidget(self.intensity_spinbox)
        layout.addWidget(QLabel("Blur Radius (Keep low, it's slow!):"))
        self.blur_spinbox = QDoubleSpinBox()
        self.blur_spinbox.setMinimum(0.1)
        self.blur_spinbox.setMaximum(5.0)
        self.blur_spinbox.setValue(2.0)
        self.blur_spinbox.setSingleStep(0.5)
        layout.addWidget(self.blur_spinbox)
    def get_params(self) -> dict:
        return {
            'intensity': self.intensity_spinbox.value(),
            'blur_amount': self.blur_spinbox.value()
        }
class HalationParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Highlight Threshold:"))
        self.thresh_spinbox = QDoubleSpinBox()
        self.thresh_spinbox.setMinimum(0.1)
        self.thresh_spinbox.setMaximum(0.9)
        self.thresh_spinbox.setValue(0.7)
        self.thresh_spinbox.setSingleStep(0.1)
        layout.addWidget(self.thresh_spinbox)
        layout.addWidget(QLabel("Red Glow Intensity:"))
        self.intensity_spinbox = QDoubleSpinBox()
        self.intensity_spinbox.setMinimum(0.0)
        self.intensity_spinbox.setMaximum(2.0)
        self.intensity_spinbox.setValue(0.8)
        self.intensity_spinbox.setSingleStep(0.1)
        layout.addWidget(self.intensity_spinbox)
    def get_params(self) -> dict:
        return {
            'threshold': self.thresh_spinbox.value(),
            'intensity': self.intensity_spinbox.value()
        }
class OilPaintingParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Brush Radius (Warning: Very Slow!):"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(1.0)
        self.radius_spinbox.setMaximum(5.0)
        self.radius_spinbox.setValue(2.0)
        self.radius_spinbox.setSingleStep(1.0)
        layout.addWidget(self.radius_spinbox)
    def get_params(self) -> dict:
        return {'radius': int(self.radius_spinbox.value())}
class KaleidoscopeParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Mirror Segments:"))
        self.segments_spinbox = QDoubleSpinBox()
        self.segments_spinbox.setMinimum(2.0)
        self.segments_spinbox.setMaximum(12.0)
        self.segments_spinbox.setValue(6.0)
        self.segments_spinbox.setSingleStep(1.0)
        layout.addWidget(self.segments_spinbox)
    def get_params(self) -> dict:
        return {'segments': self.segments_spinbox.value()}
class PixelArtParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Pixel Size (Blockiness):"))
        self.pixel_spinbox = QDoubleSpinBox()
        self.pixel_spinbox.setMinimum(2.0)
        self.pixel_spinbox.setMaximum(32.0)
        self.pixel_spinbox.setValue(8.0)
        self.pixel_spinbox.setSingleStep(2.0)
        layout.addWidget(self.pixel_spinbox)
        layout.addWidget(QLabel("Color Palette Levels:"))
        self.colors_spinbox = QDoubleSpinBox()
        self.colors_spinbox.setMinimum(2.0)
        self.colors_spinbox.setMaximum(16.0)
        self.colors_spinbox.setValue(4.0)
        self.colors_spinbox.setSingleStep(1.0)
        layout.addWidget(self.colors_spinbox)
    def get_params(self) -> dict:
        return {
            'pixel_size': int(self.pixel_spinbox.value()),
            'color_levels': self.colors_spinbox.value()
        }
class ConvolutionParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("3x3 Custom Filter (Kernel):"))
        grid = QGridLayout()
        self.inputs = []
        for row in range(3):
            row_data = []
            for col in range(3):
                box = QDoubleSpinBox()
                box.setMinimum(-100.0)
                box.setValue(0.0)
                if row == 1 and col == 1:
                    box.setValue(1.0)
                grid.addWidget(box, row, col)
                row_data.append(box)
            self.inputs.append(row_data)
        layout.addLayout(grid)
    def get_params(self) -> dict:
        kernel = np.array([[b.value() for b in row] for row in self.inputs])
        return {'kernel': kernel}
class AshControlsWidget(QWidget):
    process_requested = Signal(dict)
    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Ash's Filters</h2>"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)
        self.add_op("Gaussian Blur", GaussianParamsWidget)
        self.add_op("Sobel Edge Detect", NoParamsWidget)
        self.add_op("Power Law (Gamma)", PowerLawParamsWidget)
        self.add_op("Vintage (Sepia)", VintageParamsWidget)
        self.add_op("Cinematic Vignette", VignetteParamsWidget)
        self.add_op("Chromatic Aberration", ChromaticParamsWidget)
        self.add_op("Anime / Cartoon", AnimeParamsWidget)
        self.add_op("Fisheye Lens", FisheyeParamsWidget)
        self.add_op("Cinematic Glow", GlowParamsWidget)
        self.add_op("Film Halation", HalationParamsWidget)
        self.add_op("Oil Painting", OilPaintingParamsWidget)
        self.add_op("Kaleidoscope", KaleidoscopeParamsWidget)
        self.add_op("Pixel Art / 8-Bit", PixelArtParamsWidget)
        self.add_op("Convolution", ConvolutionParamsWidget)
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.clicked.connect(self._on_click)
        layout.addWidget(self.apply_btn)
        self.operation_selector.currentTextChanged.connect(self._on_change)
    def add_op(self, name, widget_class):
        widget = widget_class()
        self.params_stack.addWidget(widget)
        self.param_widgets[name] = widget
        self.operation_selector.addItem(name)
    def _on_click(self):
        name = self.operation_selector.currentText()
        params = self.param_widgets[name].get_params()
        params['operation'] = name
        self.process_requested.emit(params)
    def _on_change(self, name):
        self.params_stack.setCurrentWidget(self.param_widgets[name])
class AshImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls = None
    def get_name(self) -> str:
        return "Ash's Module"
    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp"]
    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls is None:
            self._controls = AshControlsWidget(module_manager, parent)
            self._controls.process_requested.connect(self._handle_request)
        return self._controls
    def _handle_request(self, params: dict):
        if self._controls and self._controls.module_manager:
            self._controls.module_manager.apply_processing_to_current_image(params)
    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 2:
                image_data = image_data[np.newaxis, :]
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Oops! Could not load the image: {e}")
            return False, None, {}, None
    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        original_shape = image_data.shape
        temp_data = image_data.copy()
        if temp_data.ndim == 3 and temp_data.shape[0] == 1:
            temp_data = np.squeeze(temp_data, axis=0)
        name = params.get('operation')
        result = temp_data                             
        if name == "Gaussian Blur":
            s = params.get('sigma', 1.0)
            result = gaussian_blur.apply_gaussian_blur(temp_data, s)
        elif name == "Sobel Edge Detect":
            result = sobel_edge.apply_sobel_edge_detection(temp_data)
        elif name == "Power Law (Gamma)":
            g = params.get('gamma', 1.0)
            result = gamma_transform.apply_gamma_transformation(temp_data, g)
        elif name == "Vintage (Sepia)":
            intensity = params.get('intensity', 1.0)
            result = vintage_filter.apply_vintage_filter(temp_data, intensity)
        elif name == "Cinematic Vignette":
            strength = params.get('strength', 1.0)
            result = vignette_filter.apply_vignette_filter(temp_data, strength)
        elif name == "Chromatic Aberration":
            shift = params.get('shift_amount', 5.0)
            result = chromatic_aberration.apply_chromatic_aberration(temp_data, shift)
        elif name == "Anime / Cartoon":
            levels = params.get('levels', 5.0)
            threshold = params.get('threshold', 0.2)
            result = anime_filter.apply_anime_filter(temp_data, levels, threshold)
        elif name == "Fisheye Lens":
            dist = params.get('distortion', -0.3)
            result = fisheye_filter.apply_fisheye_filter(temp_data, dist)
        elif name == "Cinematic Glow":
            intensity = params.get('intensity', 0.6)
            blur = params.get('blur_amount', 2.0)
            result = glow_filter.apply_glow_filter(temp_data, intensity, blur)
        elif name == "Film Halation":
            thresh = params.get('threshold', 0.7)
            intensity = params.get('intensity', 0.8)
            result = halation_filter.apply_halation_filter(temp_data, thresh, intensity)
        elif name == "Oil Painting":
            radius = params.get('radius', 2)
            result = oil_painting.apply_oil_painting(temp_data, radius)
        elif name == "Kaleidoscope":
            segments = params.get('segments', 6.0)
            result = kaleidoscope_filter.apply_kaleidoscope(temp_data, segments)
        elif name == "Pixel Art / 8-Bit":
            pixel_size = params.get('pixel_size', 8)
            colors = params.get('color_levels', 4.0)
            result = pixel_art_filter.apply_pixel_art(temp_data, pixel_size, colors)
        elif name == "Convolution":
            k = params.get('kernel')
            if k is not None:
                img_f = temp_data.astype(float)
                if img_f.ndim == 3 and (img_f.shape[2] == 3 or img_f.shape[2] == 4):
                    num_channels = img_f.shape[2]
                    channel_results = []
                    for c in range(num_channels):
                        channel_data = img_f[:, :, c]
                        filtered_channel = filtering_utils.manual_convolve(channel_data, k)
                        channel_results.append(filtered_channel)
                    result = np.stack(channel_results, axis=-1)
                else:
                    result = filtering_utils.manual_convolve(img_f, k)
        if result.shape != original_shape:
            if len(original_shape) == 3 and (original_shape[2] == 3 or original_shape[2] == 4):
                if result.ndim == 2:
                    new_result = np.zeros(original_shape, dtype=result.dtype)
                    num_channels = original_shape[2]
                    for c in range(min(3, num_channels)):
                        new_result[:, :, c] = result
                    if num_channels == 4:
                        max_val = np.max(result)
                        new_result[:, :, 3] = max_val if max_val > 0 else 1.0
                    result = new_result
            try:
                final_output = result.reshape(original_shape)
            except Exception:
                final_output = result
        else:
            final_output = result
        return final_output.astype(image_data.dtype)
