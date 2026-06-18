import numpy as np
import imageio
from PySide6.QtWidgets import QWidget

from modules.i_image_module import IImageModule
from modules.david.widgets import DavidControlsWidget
from modules.david.effects import (
    normalize,
    apply_sepia,
    apply_channel_swap,
    apply_vignette,
    apply_negative,
    apply_emboss,
    apply_polar_transform,
    apply_glitch,
)


class DavidImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "David's Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff", "tif"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = DavidControlsWidget(module_manager, parent)
        return self._controls_widget

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict | None = None) -> np.ndarray:
        if params is None:
            return image_data

        operation = params.get('operation')

        original_dtype = image_data.dtype
        img_float = normalize(image_data)

        result = self._apply_effect(img_float, operation, params)

        result = self._match_dimensions(result, image_data)

        if original_dtype == np.uint8:
            return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
        elif original_dtype == np.uint16:
            return (np.clip(result, 0.0, 1.0) * 65535).astype(np.uint16)
        else:
            return result.astype(original_dtype)

    def _apply_effect(self, image: np.ndarray, operation: str, params: dict) -> np.ndarray:
        if operation == "Sepia Tone":
            return apply_sepia(image, intensity=params.get('intensity', 1.0))

        elif operation == "Channel Swap":
            return apply_channel_swap(image, swap_mode=params.get('swap_mode', 'R<->B'))

        elif operation == "Vignette":
            return apply_vignette(
                image,
                strength=params.get('strength', 0.8),
                radius=params.get('radius', 1.0),
            )

        elif operation == "Negative":
            return apply_negative(image)

        elif operation == "Emboss / Bas-Relief":
            return apply_emboss(
                image,
                direction=params.get('direction', 'Top-Left'),
                strength=params.get('strength', 1.0),
            )

        elif operation == "Polar Transform":
            return apply_polar_transform(
                image,
                direction=params.get('direction', 'To Polar'),
            )

        elif operation == "Glitch Art":
            return apply_glitch(
                image,
                intensity=params.get('intensity', 0.5),
                seed=params.get('seed', 42),
            )

        return image


    def _match_dimensions(self, result: np.ndarray, original: np.ndarray) -> np.ndarray:
        if result.ndim == original.ndim:
            return result

        if original.ndim == 3 and original.shape[2] in [3, 4] and result.ndim == 2:
            rgb = np.stack([result, result, result], axis=-1)
            if original.shape[2] == 4:
                alpha = np.ones_like(result)
                rgb = np.dstack([rgb, alpha])
            return rgb

        if original.ndim == 3 and original.shape[0] == 1 and result.ndim == 2:
            return result[np.newaxis, :]

        return result
