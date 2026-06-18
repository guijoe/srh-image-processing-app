from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout, QSpinBox, QCheckBox
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# --- Parameter Widgets for Different Operations ---
class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError
class ContrastStretchParamsWidget(BaseParamsWidget):
    """A widget for contrast stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Lower Percentile:"))
        self.lower_spinbox = QDoubleSpinBox()
        self.lower_spinbox.setMinimum(0.0)
        self.lower_spinbox.setMaximum(100.0)
        self.lower_spinbox.setValue(2.0)
        self.lower_spinbox.setSingleStep(0.1)
        layout.addWidget(self.lower_spinbox)

        layout.addWidget(QLabel("Upper Percentile:"))
        self.upper_spinbox = QDoubleSpinBox()
        self.upper_spinbox.setMinimum(0.0)
        self.upper_spinbox.setMaximum(100.0)
        self.upper_spinbox.setValue(98.0)
        self.upper_spinbox.setSingleStep(0.1)
        layout.addWidget(self.upper_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'lower_percentile': self.lower_spinbox.value(),
            'upper_percentile': self.upper_spinbox.value()
        }
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
   
    #Also known ans Perona Malik diffusion
class AnisotropicDiffusionParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Anisotropic Diffusion Parameters:"))

        layout.addWidget(QLabel("Iterations:"))
        self.iterations_spinbox = QDoubleSpinBox()
        self.iterations_spinbox.setMinimum(1)
        self.iterations_spinbox.setMaximum(100)
        self.iterations_spinbox.setValue(10)
        self.iterations_spinbox.setSingleStep(1)
        layout.addWidget(self.iterations_spinbox)

        layout.addWidget(QLabel("Kappa (Conductance):"))
        self.kappa_spinbox = QDoubleSpinBox()
        self.kappa_spinbox.setMinimum(0.1)
        self.kappa_spinbox.setMaximum(100.0)
        self.kappa_spinbox.setValue(20.0)
        self.kappa_spinbox.setSingleStep(0.1)
        layout.addWidget(self.kappa_spinbox)

        layout.addWidget(QLabel("Gamma (Lambda):"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(0.25)
        self.gamma_spinbox.setValue(0.1)
        self.gamma_spinbox.setSingleStep(0.01)
        layout.addWidget(self.gamma_spinbox)

        layout.addWidget(QLabel("Equation:")) #There were 2 different equations proposed by Perona and Malik. This allows users to choose which one to use.
        self.option_combobox = QComboBox()
        self.option_combobox.addItems(["1: exp(-(x/K)^2)", "2: 1 / (1 + (x/K)^2)"])
        layout.addWidget(self.option_combobox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'iterations': int(self.iterations_spinbox.value()),
            'kappa': self.kappa_spinbox.value(),
            'gamma': self.gamma_spinbox.value(),
            'equation': 1 if self.option_combobox.currentIndex() == 0 else 2
        }
    

class FrangiParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.black_ridges_checkbox = QCheckBox("Detect Dark Structures (Black Ridges)")
        self.black_ridges_checkbox.setChecked(False) # Default detects bright veins
        layout.addWidget(self.black_ridges_checkbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'black_ridges': self.black_ridges_checkbox.isChecked()}


class UnsharpMaskParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Radius (Blur Size):"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(0.1)
        self.radius_spinbox.setMaximum(10.0)
        self.radius_spinbox.setValue(1.0)
        self.radius_spinbox.setSingleStep(0.1)
        layout.addWidget(self.radius_spinbox)

        layout.addWidget(QLabel("Amount (Sharpen Strength):"))
        self.amount_spinbox = QDoubleSpinBox()
        self.amount_spinbox.setMinimum(0.1)
        self.amount_spinbox.setMaximum(10.0)
        self.amount_spinbox.setValue(1.0)
        self.amount_spinbox.setSingleStep(0.1)
        layout.addWidget(self.amount_spinbox)
        
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'radius': self.radius_spinbox.value(),
            'amount': self.amount_spinbox.value()
        }

class ConvolutionParamsWidget(BaseParamsWidget):
    """A widget for defining a 3x3 convolution kernel."""
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
                # Set center to 1.0 for an identity-like default
                if r == 1 and c == 1:
                    spinbox.setValue(1.0)
                grid_layout.addWidget(spinbox, r, c)
                row_inputs.append(spinbox)
            self.kernel_inputs.append(row_inputs)
        layout.addLayout(grid_layout)

    def get_params(self) -> dict:
        kernel = np.array([[spinbox.value() for spinbox in row] for row in self.kernel_inputs])
        return {'kernel': kernel}

# Define a custom control widget
class AlpTuncerControlsWidget(QWidget):
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
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Contrast Stretching": ContrastStretchParamsWidget,
            "Perona-Malik Diffusion": AnisotropicDiffusionParamsWidget,
            "Frangi Vesselness": FrangiParamsWidget,
            "Unsharp Masking": UnsharpMaskParamsWidget,
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

class AlpTuncerImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "AlpTuncer Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = AlpTuncerControlsWidget(module_manager, parent)
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

        min_orig = np.min(image_data)
        max_orig = np.max(image_data)

        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)

            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)
        elif operation == "Median Filter":
            filter_size = params.get('filter_size', 3)
            if filter_size <= 1: return processed_data 

            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]: 

                channels = []
                for i in range(processed_data.shape[2]):
                    channels.append(skimage.filters.median(processed_data[:,:,i], footprint=skimage.morphology.disk(int(filter_size/2))))
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = skimage.filters.median(processed_data, footprint=skimage.morphology.disk(int(filter_size/2)))
        elif operation == "Sobel Edge Detect":

            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:,:,:3])
            else:
                grayscale_img = processed_data
            
            processed_data = skimage.filters.sobel(grayscale_img)
        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            # Normalize to [0, 1]
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
        elif operation == "Contrast Stretching":
            lower_percentile = params.get('lower_percentile', 2.0)
            upper_percentile = params.get('upper_percentile', 98.0)
            # Compute percentiles
            lower_val = np.percentile(processed_data, lower_percentile)
            upper_val = np.percentile(processed_data, upper_percentile)
            # Stretch contrast
            if upper_val > lower_val: # Avoid division by zero
                processed_data = (processed_data - lower_val) * (255.0 / (upper_val - lower_val))
                processed_data = np.clip(processed_data, 0, 255) # Ensure values are in valid range
        elif operation == "Perona-Malik Diffusion":
            iterations = params.get('iterations', 10)
            kappa = params.get('kappa', 15.0)
            gamma = params.get('gamma', 0.1)
            option = params.get('option', 1)

            img_float = processed_data.astype(np.float64)

            if img_float.ndim == 3:
                pad_width = ((1, 1), (1, 1), (0, 0))
            else:
                pad_width = ((1, 1), (1, 1))

            for _ in range(iterations):
                padded = np.pad(img_float, pad_width, mode='reflect')

                #North South East West differences to compute the diffusion coefficients

                C = padded[1:-1, 1:-1]
                N = padded[:-2, 1:-1]
                S = padded[2:, 1:-1]
                E = padded[1:-1, 2:]
                W = padded[1:-1, :-2]

                deltaN = N - C
                deltaS = S - C
                deltaE = E - C
                deltaW = W - C

                if option == 1:
                    cN = np.exp(-(deltaN / kappa)**2)
                    cS = np.exp(-(deltaS / kappa)**2)
                    cE = np.exp(-(deltaE / kappa)**2)
                    cW = np.exp(-(deltaW / kappa)**2)
                else:
                    cN = 1.0 / (1.0 + (deltaN / kappa)**2)
                    cS = 1.0 / (1.0 + (deltaS / kappa)**2)
                    cE = 1.0 / (1.0 + (deltaE / kappa)**2)
                    cW = 1.0 / (1.0 + (deltaW / kappa)**2)

                img_float += gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

            processed_data = np.clip(img_float, 0, 255)
        elif operation == "Frangi Vesselness":
            black_ridges = params.get('black_ridges', False)
            
            # Frangi requires 2D grayscale
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:,:,:3])
            else:
                grayscale_img = processed_data.astype(float)
                

            filtered = skimage.filters.frangi(grayscale_img, black_ridges=black_ridges)
            
            
            f_min = np.min(filtered)
            f_max = np.max(filtered)
            if f_max > f_min:
                processed_data = ((filtered - f_min) / (f_max - f_min)) * max_orig
            else:
                processed_data = filtered

            #The output is strictly a 2D map, even if you put an RGB image in.
            
        elif operation == "Unsharp Masking":
            radius = params.get('radius', 1.0)
            amount = params.get('amount', 1.0)
            

            img_norm = (processed_data.astype(float) - min_orig) / (max_orig - min_orig + 1e-8)
            
            sharpened = skimage.filters.unsharp_mask(img_norm, radius=radius, amount=amount, preserve_range=False)
            
            # Map back to original scale
            processed_data = (sharpened * (max_orig - min_orig)) + min_orig
            processed_data = np.clip(processed_data, min_orig, max_orig)

        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        if image_data.ndim == 3 and processed_data.ndim == 2:
            processed_data = np.stack([processed_data] * image_data.shape[2], axis=-1)

        # Final cast back to original data type
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data