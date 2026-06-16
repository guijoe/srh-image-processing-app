from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
from PySide6.QtCore import Signal
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
        label = QLabel("This transformation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class KaleidoscopeParamsWidget(BaseParamsWidget):
    """A widget for the Kaleidoscope Mirror transformation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Mirror Segments:"))
        self.segments_spinbox = QDoubleSpinBox()
        self.segments_spinbox.setMinimum(3)
        self.segments_spinbox.setMaximum(16)
        self.segments_spinbox.setDecimals(0)
        self.segments_spinbox.setValue(8)
        self.segments_spinbox.setSingleStep(1)
        layout.addWidget(self.segments_spinbox)

        layout.addWidget(QLabel("Zoom:"))
        self.zoom_spinbox = QDoubleSpinBox()
        self.zoom_spinbox.setMinimum(0.5)
        self.zoom_spinbox.setMaximum(2.0)
        self.zoom_spinbox.setValue(1.0)
        self.zoom_spinbox.setSingleStep(0.1)
        layout.addWidget(self.zoom_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'segments': int(self.segments_spinbox.value()),
            'zoom': self.zoom_spinbox.value(),
        }

class NeonEdgeBloomParamsWidget(BaseParamsWidget):
    """A widget for the Neon Edge Bloom transformation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Edge Strength:"))
        self.edge_strength_spinbox = QDoubleSpinBox()
        self.edge_strength_spinbox.setMinimum(0.5)
        self.edge_strength_spinbox.setMaximum(8.0)
        self.edge_strength_spinbox.setValue(3.0)
        self.edge_strength_spinbox.setSingleStep(0.25)
        layout.addWidget(self.edge_strength_spinbox)

        layout.addWidget(QLabel("Glow Amount:"))
        self.glow_amount_spinbox = QDoubleSpinBox()
        self.glow_amount_spinbox.setMinimum(0.0)
        self.glow_amount_spinbox.setMaximum(1.0)
        self.glow_amount_spinbox.setValue(0.65)
        self.glow_amount_spinbox.setSingleStep(0.05)
        layout.addWidget(self.glow_amount_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'edge_strength': self.edge_strength_spinbox.value(),
            'glow_amount': self.glow_amount_spinbox.value(),
        }

class VortexSwirlParamsWidget(BaseParamsWidget):
    """A widget for the Vortex Swirl transformation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Swirl Strength:"))
        self.swirl_strength_spinbox = QDoubleSpinBox()
        self.swirl_strength_spinbox.setMinimum(-12.0)
        self.swirl_strength_spinbox.setMaximum(12.0)
        self.swirl_strength_spinbox.setValue(5.0)
        self.swirl_strength_spinbox.setSingleStep(0.5)
        layout.addWidget(self.swirl_strength_spinbox)

        layout.addWidget(QLabel("Radius:"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(0.2)
        self.radius_spinbox.setMaximum(2.0)
        self.radius_spinbox.setValue(1.0)
        self.radius_spinbox.setSingleStep(0.1)
        layout.addWidget(self.radius_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'swirl_strength': self.swirl_strength_spinbox.value(),
            'radius': self.radius_spinbox.value(),
        }

class ThermalTopographyParamsWidget(BaseParamsWidget):
    """A widget for the Thermal Topography transformation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Band Count:"))
        self.band_count_spinbox = QDoubleSpinBox()
        self.band_count_spinbox.setMinimum(3)
        self.band_count_spinbox.setMaximum(24)
        self.band_count_spinbox.setDecimals(0)
        self.band_count_spinbox.setValue(10)
        self.band_count_spinbox.setSingleStep(1)
        layout.addWidget(self.band_count_spinbox)

        layout.addWidget(QLabel("Color Intensity:"))
        self.color_intensity_spinbox = QDoubleSpinBox()
        self.color_intensity_spinbox.setMinimum(0.2)
        self.color_intensity_spinbox.setMaximum(2.0)
        self.color_intensity_spinbox.setValue(1.0)
        self.color_intensity_spinbox.setSingleStep(0.1)
        layout.addWidget(self.color_intensity_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'band_count': int(self.band_count_spinbox.value()),
            'color_intensity': self.color_intensity_spinbox.value(),
        }

# Define a custom control widget
class faaz_hassanControlsWidget(QWidget):
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
            "Kaleidoscope Mirror": KaleidoscopeParamsWidget,
            "Neon Edge Bloom": NeonEdgeBloomParamsWidget,
            "Vortex Swirl": VortexSwirlParamsWidget,
            "Thermal Topography": ThermalTopographyParamsWidget,
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

class faaz_hassanImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "faaz_hassan Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = faaz_hassanControlsWidget(module_manager, parent)
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

    def _get_display_max(self, image_data: np.ndarray) -> float:
        """Return the value that represents full brightness for this image."""
        if np.issubdtype(image_data.dtype, np.integer):
            return float(np.iinfo(image_data.dtype).max)
        max_value = float(np.max(image_data)) if image_data.size else 1.0
        return 1.0 if max_value <= 1.0 else max_value

    def _to_unit_float(self, image_data: np.ndarray) -> np.ndarray:
        """Convert image values to a clean 0..1 float range."""
        display_max = self._get_display_max(image_data)
        if display_max == 0:
            return image_data.astype(float)
        return np.clip(image_data.astype(float) / display_max, 0.0, 1.0)

    def _from_unit_float(self, image_data: np.ndarray, unit_data: np.ndarray) -> np.ndarray:
        """Convert 0..1 float image values back to the original image type."""
        display_max = self._get_display_max(image_data)
        restored = np.clip(unit_data, 0.0, 1.0) * display_max
        if np.issubdtype(image_data.dtype, np.integer):
            restored = np.rint(restored)
        return restored.astype(image_data.dtype)

    def _split_color_and_alpha(self, unit_data: np.ndarray):
        """Separate editable color channels from an optional alpha channel."""
        if unit_data.ndim == 3 and unit_data.shape[2] == 4:
            return unit_data[:, :, :3], unit_data[:, :, 3:4]
        return unit_data, None

    def _restore_alpha(self, color_data: np.ndarray, alpha_data: np.ndarray) -> np.ndarray:
        if alpha_data is None:
            return color_data
        return np.concatenate([color_data, alpha_data], axis=2)

    def _brightness(self, color_data: np.ndarray) -> np.ndarray:
        """Create a grayscale brightness image using only NumPy."""
        if color_data.ndim == 3 and color_data.shape[2] >= 3:
            return (
                0.299 * color_data[:, :, 0]
                + 0.587 * color_data[:, :, 1]
                + 0.114 * color_data[:, :, 2]
            )
        return color_data

    def _nearest_sample(self, image_data: np.ndarray, source_y: np.ndarray, source_x: np.ndarray) -> np.ndarray:
        """Sample pixels from floating-point coordinates with nearest-neighbor lookup."""
        height, width = image_data.shape[:2]
        y_indices = np.clip(np.rint(source_y).astype(int), 0, height - 1)
        x_indices = np.clip(np.rint(source_x).astype(int), 0, width - 1)
        return image_data[y_indices, x_indices]

    def _kaleidoscope_mirror(self, image_data: np.ndarray, segments: int, zoom: float) -> np.ndarray:
        unit_data = self._to_unit_float(image_data)
        color_data, alpha_data = self._split_color_and_alpha(unit_data)
        height, width = color_data.shape[:2]

        y_grid, x_grid = np.indices((height, width))
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
        y_shifted = y_grid - center_y
        x_shifted = x_grid - center_x

        radius = np.sqrt(x_shifted ** 2 + y_shifted ** 2) / max(zoom, 0.01)
        angle = np.arctan2(y_shifted, x_shifted)
        segment_angle = 2.0 * np.pi / max(segments, 1)
        mirrored_angle = np.mod(angle, segment_angle)
        mirrored_angle = np.minimum(mirrored_angle, segment_angle - mirrored_angle)

        source_x = center_x + radius * np.cos(mirrored_angle)
        source_y = center_y + radius * np.sin(mirrored_angle)
        color_result = self._nearest_sample(color_data, source_y, source_x)

        if alpha_data is not None:
            alpha_result = self._nearest_sample(alpha_data, source_y, source_x)
            color_result = self._restore_alpha(color_result, alpha_result)

        return self._from_unit_float(image_data, color_result)

    def _neon_edge_bloom(self, image_data: np.ndarray, edge_strength: float, glow_amount: float) -> np.ndarray:
        unit_data = self._to_unit_float(image_data)
        color_data, alpha_data = self._split_color_and_alpha(unit_data)
        brightness = self._brightness(color_data)

        horizontal_edges = np.abs(np.roll(brightness, -1, axis=1) - np.roll(brightness, 1, axis=1))
        vertical_edges = np.abs(np.roll(brightness, -1, axis=0) - np.roll(brightness, 1, axis=0))
        edge_map = np.sqrt(horizontal_edges ** 2 + vertical_edges ** 2)
        edge_map = np.clip(edge_map * edge_strength, 0.0, 1.0)

        glow = edge_map.copy()
        for distance in range(1, 4):
            glow += (
                np.roll(edge_map, distance, axis=0)
                + np.roll(edge_map, -distance, axis=0)
                + np.roll(edge_map, distance, axis=1)
                + np.roll(edge_map, -distance, axis=1)
            ) / (4.0 * (distance + 1.0))
        glow = np.clip(glow * glow_amount, 0.0, 1.0)

        if color_data.ndim == 2:
            neon_color = glow
            color_result = np.clip(color_data * (1.0 - glow) + neon_color, 0.0, 1.0)
        else:
            neon_palette = np.stack([
                np.clip(edge_map + 0.35 * brightness, 0.0, 1.0),
                np.clip(0.45 * glow + 0.95 * (1.0 - brightness), 0.0, 1.0),
                np.clip(glow + 0.55 * brightness, 0.0, 1.0),
            ], axis=2)
            color_result = np.clip(color_data * (1.0 - glow[:, :, None]) + neon_palette * glow[:, :, None], 0.0, 1.0)

        color_result = self._restore_alpha(color_result, alpha_data)
        return self._from_unit_float(image_data, color_result)

    def _vortex_swirl(self, image_data: np.ndarray, swirl_strength: float, radius_scale: float) -> np.ndarray:
        unit_data = self._to_unit_float(image_data)
        color_data, alpha_data = self._split_color_and_alpha(unit_data)
        height, width = color_data.shape[:2]

        y_grid, x_grid = np.indices((height, width))
        center_y = (height - 1) / 2.0
        center_x = (width - 1) / 2.0
        y_shifted = y_grid - center_y
        x_shifted = x_grid - center_x
        radius = np.sqrt(x_shifted ** 2 + y_shifted ** 2)
        max_radius = max(np.sqrt(center_x ** 2 + center_y ** 2) * radius_scale, 1.0)

        fade = np.clip(1.0 - radius / max_radius, 0.0, 1.0)
        angle = swirl_strength * fade ** 2
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)

        source_x = center_x + x_shifted * cos_angle - y_shifted * sin_angle
        source_y = center_y + x_shifted * sin_angle + y_shifted * cos_angle
        color_result = self._nearest_sample(color_data, source_y, source_x)

        if alpha_data is not None:
            alpha_result = self._nearest_sample(alpha_data, source_y, source_x)
            color_result = self._restore_alpha(color_result, alpha_result)

        return self._from_unit_float(image_data, color_result)

    def _thermal_topography(self, image_data: np.ndarray, band_count: int, color_intensity: float) -> np.ndarray:
        unit_data = self._to_unit_float(image_data)
        color_data, alpha_data = self._split_color_and_alpha(unit_data)
        brightness = self._brightness(color_data)

        bands = np.floor(brightness * band_count) / max(band_count - 1, 1)
        contour_lines = np.abs(np.sin(brightness * band_count * np.pi))
        contour_lines = np.where(contour_lines < 0.18, 0.18, 1.0)

        red = np.clip(1.8 * bands - 0.35, 0.0, 1.0)
        green = np.clip(1.6 - np.abs(2.0 * bands - 1.1) * 1.5, 0.0, 1.0)
        blue = np.clip(1.25 - 2.0 * bands, 0.0, 1.0)
        thermal_color = np.stack([red, green, blue], axis=2)

        thermal_color = np.clip(thermal_color * color_intensity, 0.0, 1.0)
        thermal_color *= contour_lines[:, :, None]

        color_result = self._restore_alpha(thermal_color, alpha_data)
        return self._from_unit_float(image_data, color_result)

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()

        operation = params.get('operation')

        if operation == "Kaleidoscope Mirror":
            segments = params.get('segments', 8)
            zoom = params.get('zoom', 1.0)
            processed_data = self._kaleidoscope_mirror(processed_data, segments, zoom)
        elif operation == "Neon Edge Bloom":
            edge_strength = params.get('edge_strength', 3.0)
            glow_amount = params.get('glow_amount', 0.65)
            processed_data = self._neon_edge_bloom(processed_data, edge_strength, glow_amount)
        elif operation == "Vortex Swirl":
            swirl_strength = params.get('swirl_strength', 5.0)
            radius = params.get('radius', 1.0)
            processed_data = self._vortex_swirl(processed_data, swirl_strength, radius)
        elif operation == "Thermal Topography":
            band_count = params.get('band_count', 10)
            color_intensity = params.get('color_intensity', 1.0)
            processed_data = self._thermal_topography(processed_data, band_count, color_intensity)


        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data
