from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
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
    
class RainbowParamsWidget(BaseParamsWidget):
    """Filtre arc-en-ciel — mélange de couleurs selon la position verticale."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Intensité (0.0 = original, 1.0 = full rainbow) :"))
        self.intensity_spinbox = QDoubleSpinBox()
        self.intensity_spinbox.setMinimum(0.0)
        self.intensity_spinbox.setMaximum(1.0)
        self.intensity_spinbox.setValue(0.5)
        self.intensity_spinbox.setSingleStep(0.1)
        layout.addWidget(self.intensity_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'intensity': self.intensity_spinbox.value()}
    
class ShadesOfBlueParamsWidget(BaseParamsWidget): #wigdet for the blue filter
    """Filtre tons bleus — renforce le canal bleu, réduit rouge et vert."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Intensité (0.0 = original, 1.0 = full blue) :"))
        self.intensity_spinbox = QDoubleSpinBox()
        self.intensity_spinbox.setMinimum(0.0)
        self.intensity_spinbox.setMaximum(1.0)
        self.intensity_spinbox.setValue(0.5)
        self.intensity_spinbox.setSingleStep(0.1)
        layout.addWidget(self.intensity_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'intensity': self.intensity_spinbox.value()}
    

class SaturationParamsWidget(BaseParamsWidget):
    """Filtre saturation — contrôle l'intensité des couleurs."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Saturation :"))

        # QSlider travaille avec des entiers, donc on multiplie par 10
        # 0 → 0.0, 10 → 1.0 (original), 30 → 3.0 (max)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(30)
        self.slider.setValue(10)  # 1.0 par défaut = original
        layout.addWidget(self.slider)

        # Label qui affiche la valeur en temps réel
        self.value_label = QLabel("1.0")
        layout.addWidget(self.value_label)

        # Mettre à jour le label quand le slider bouge
        self.slider.valueChanged.connect(
            lambda v: self.value_label.setText(f"{v / 10:.1f}")
        )

        layout.addStretch()

    def get_params(self) -> dict:
        return {'saturation': self.slider.value() / 10}  # reconvertir en float
    
class VignetteParamsWidget(BaseParamsWidget):
    """Filtre vignette — assombrit les bords progressivement."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Intensité :"))

        # 0 = pas de vignette, 30 = vignette très forte
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(30)
        self.slider.setValue(15)
        layout.addWidget(self.slider)

        self.value_label = QLabel("1.5")
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(
            lambda v: self.value_label.setText(f"{v / 10:.1f}")
        )

        layout.addStretch()

    def get_params(self) -> dict:
        return {'strength': self.slider.value() / 10}
    
class ContrastStretchingParamsWidget(BaseParamsWidget):
    """A widget for Contrast Stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Input for the new minimum value
        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setMinimum(0.0)
        self.min_spinbox.setMaximum(255.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)

        # Input for the new maximum value
        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setMinimum(0.0)
        self.max_spinbox.setMaximum(255.0)
        self.max_spinbox.setValue(255.0)
        layout.addWidget(self.max_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'new_min': self.min_spinbox.value(),
            'new_max': self.max_spinbox.value()
        }

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
class GabrielControlsWidget(QWidget):
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
            "Contrast Stretching": ContrastStretchingParamsWidget,
            "Rainbow": RainbowParamsWidget,
            "Shades of Blue": ShadesOfBlueParamsWidget,
            "Saturation": SaturationParamsWidget,
            "Vignette": VignetteParamsWidget,
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

class GabrielImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Gabriel Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = GabrielControlsWidget(module_manager, parent)
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

        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            # skimage.filters.gaussian expects float data
            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)
        elif operation == "Median Filter":
            filter_size = params.get('filter_size', 3)
            if filter_size <= 1: return processed_data # No change
            # skimage.filters.median
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]: # RGB/RGBA
                # Apply to each channel
                channels = []
                for i in range(processed_data.shape[2]):
                    channels.append(skimage.filters.median(processed_data[:,:,i], footprint=skimage.morphology.disk(int(filter_size/2))))
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = skimage.filters.median(processed_data, footprint=skimage.morphology.disk(int(filter_size/2)))
        elif operation == "Sobel Edge Detect":
            # Sobel works on 2D (grayscale) images. Convert if necessary.
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
                # Apply gamma correction
                gamma_corrected = np.power(normalized, gamma)
                # Scale back to original range
                processed_data = gamma_corrected * max_val
        elif operation == "Convolution":
            kernel = params.get('kernel')
            if kernel is not None:
                # Convolve works best on float images
                input_float = processed_data.astype(float)
                if input_float.ndim == 3 and input_float.shape[2] in [3, 4]: # RGB/RGBA
                    channels = []
                    for i in range(input_float.shape[2]):
                        channels.append(convolve(input_float[:,:,i], kernel, mode='reflect'))
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = convolve(input_float, kernel, mode='reflect')
        
        elif operation == "Contrast Stretching":
            # Ensure we are working with a floating point image for calculations
            img_float = processed_data.astype(float)

            # Get parameters from the UI
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)

            # Get current image intensity range
            current_min = np.min(img_float)
            current_max = np.max(img_float)

            # Avoid division by zero if the image is flat
            if current_max == current_min:
                return processed_data # Return original image

            # Apply the linear stretching formula
            processed_data = (img_float - current_min) * \
                             ((new_max - new_min) / (current_max - current_min)) + new_min

            # Clip values to be safe, though the formula should handle it
            processed_data = np.clip(processed_data, new_min, new_max)

        #Logique pour le filtre arc-en-ciel !!! --------->
        elif operation == "Rainbow":
            intensity = params.get('intensity', 0.5)

            # On s'assure d'avoir une image RGB
            if processed_data.ndim == 2:
                # Grayscale → on convertit en RGB
                img_rgb = np.stack([processed_data] * 3, axis=-1).astype(float)
            else:
                img_rgb = processed_data[:, :, :3].astype(float)

            h, w = img_rgb.shape[:2]

            # --- Créer le masque arc-en-ciel ---
            # Pour chaque ligne y, on calcule une teinte HSV
            # y_norm va de 0.0 (haut) à 1.0 (bas)
            y_norm = np.linspace(0.0, 1.0, h)  # shape: (h,)

            # Convertir HSV → RGB manuellement avec NumPy
            # H = y_norm (teinte), S = 1.0 (saturé), V = 1.0 (lumineux)
            hue = y_norm  # shape: (h,)

            # Formule HSV → RGB (NumPy pur, pas de skimage)
            i_h = (hue * 6.0).astype(int) % 6       # secteur (0 à 5)
            f   = (hue * 6.0) - (hue * 6.0).astype(int)  # fraction dans le secteur
            q   = 1.0 - f
            
            # Les 6 secteurs de la roue des couleurs
            # Chaque secteur = une transition entre deux couleurs primaires
            rgb_sectors = np.array([
                [[1, 0, 0], [1, 0, 0]],   # secteur 0: rouge
            ])

            # Version plus lisible : on calcule R, G, B pour chaque ligne
            r = np.where(i_h == 0, 1.0,
                np.where(i_h == 1, q,
                np.where(i_h == 2, 0.0,
                np.where(i_h == 3, 0.0,
                np.where(i_h == 4, f,
                1.0)))))  # secteur 5

            g = np.where(i_h == 0, f,
                np.where(i_h == 1, 1.0,
                np.where(i_h == 2, 1.0,
                np.where(i_h == 3, q,
                np.where(i_h == 4, 0.0,
                0.0)))))

            b = np.where(i_h == 0, 0.0,
                np.where(i_h == 1, 0.0,
                np.where(i_h == 2, f,
                np.where(i_h == 3, 1.0,
                np.where(i_h == 4, 1.0,
                q)))))

            # rainbow_map shape: (h, 3) — une couleur par ligne
            rainbow_map = np.stack([r, g, b], axis=-1) * 255.0

            # Étendre à toute la largeur : (h, 1, 3) → broadcast → (h, w, 3)
            rainbow_map = rainbow_map[:, np.newaxis, :] * np.ones((1, w, 1))

            # --- Mélange avec l'image originale ---
            # intensity=0 → image originale, intensity=1 → full rainbow
            processed_data = (1 - intensity) * img_rgb + intensity * rainbow_map
            processed_data = np.clip(processed_data, 0, 255)
        
        elif operation == "Shades of Blue":
            intensity = params.get('intensity', 0.5)

            # S'assurer qu'on a une image RGB
            if processed_data.ndim == 2:
                img_rgb = np.stack([processed_data] * 3, axis=-1).astype(float)
            else:
                img_rgb = processed_data[:, :, :3].astype(float)

            # Copie pour ne pas modifier l'original
            blue_img = img_rgb.copy()

            # Les facteurs varient selon l'intensité
            # intensity=0 → facteurs neutres (1.0, 1.0, 1.0) = image originale
            # intensity=1 → facteurs forts (0.2, 0.4, 1.4) = full blue
            r_factor = 1.0 - (0.8 * intensity)   # 1.0 → 0.2
            g_factor = 1.0 - (0.6 * intensity)   # 1.0 → 0.4
            b_factor = 1.0 + (0.4 * intensity)   # 1.0 → 1.4

            blue_img[:, :, 0] = img_rgb[:, :, 0] * r_factor  # canal Rouge
            blue_img[:, :, 1] = img_rgb[:, :, 1] * g_factor  # canal Vert
            blue_img[:, :, 2] = img_rgb[:, :, 2] * b_factor  # canal Bleu

            processed_data = np.clip(blue_img, 0, 255)

        elif operation == "Saturation":
            saturation = params.get('saturation', 1.0)

            # S'assurer qu'on a une image RGB
            if processed_data.ndim == 2:
                # Grayscale → déjà désaturé, rien à faire
                img_rgb = np.stack([processed_data] * 3, axis=-1).astype(float)
            else:
                img_rgb = processed_data[:, :, :3].astype(float)

            # Étape 1 : calculer la luminosité perçue de chaque pixel
            # Coefficients de luminance (perception humaine)
            # shape: (h, w) — une valeur par pixel
            luminance = (0.299 * img_rgb[:, :, 0] +
                        0.587 * img_rgb[:, :, 1] +
                        0.114 * img_rgb[:, :, 2])

            # Étape 2 : étendre luminance à 3 canaux pour le broadcast
            # shape: (h, w) → (h, w, 3)
            luminance_3ch = luminance[:, :, np.newaxis]

            # Étape 3 : interpolation linéaire
            # saturation=0 → tout gris
            # saturation=1 → original
            # saturation>1 → couleurs amplifiées
            processed_data = luminance_3ch + saturation * (img_rgb - luminance_3ch)

            processed_data = np.clip(processed_data, 0, 255)
        elif operation == "Vignette":
            strength = params.get('strength', 1.5)

            if processed_data.ndim == 2:
                img = processed_data.astype(float)
            else:
                img = processed_data[:, :, :3].astype(float)

            h, w = img.shape[:2]

            # Étape 1 : grilles de coordonnées
            # X shape: (h, w), Y shape: (h, w)
            X, Y = np.meshgrid(np.arange(w), np.arange(h))

            # Étape 2 : normaliser autour du centre (-1 à 1)
            cx, cy = w / 2, h / 2
            x_norm = (X - cx) / cx
            y_norm = (Y - cy) / cy

            # Étape 3 : distance euclidienne au centre
            # shape: (h, w)
            distance = np.sqrt(x_norm ** 2 + y_norm ** 2)

            # Étape 4 : masque — plus on est loin du centre, plus c'est sombre
            # On normalise par sqrt(2) (distance max au coin)
            # strength contrôle la rapidité de l'assombrissement
            mask = 1 - np.clip((distance / np.sqrt(2)) ** (1 / strength), 0, 1)

            # Étape 5 : appliquer le masque
            # mask shape: (h, w) → (h, w, 1) pour le broadcast sur les 3 canaux
            if img.ndim == 3:
                mask = mask[:, :, np.newaxis]

            processed_data = img * mask
            processed_data = np.clip(processed_data, 0, 255)


        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data