"""
SRH ImageViewer – Maha Ibrahim Module
Course : Imaging Technologies
Student: Maha Ibrahim | SRH University of Applied Sciences Berlin

Transformations implemented (pure NumPy):
  1. Histogram Equalization  → cat.jpg       (flat grayscale, reveals fur texture)
  2. Contrast Stretching     → fullmoon.jpg  (overexposed/warm moon, reveals surface)
  3. Laplacian Edge Enhance  → moon.jpg      (half-moon, sharpens crater rims)
  4. Unsharp Mask (Sharpen)  → flowers.jpg   (magnolia petals + fine branch detail)
  5. Median Filter           → statue.jpg    (very noisy low-light museum photo)
  6. Grayscale Conversion    → portrait.jpg  (colorful autumn Berlin street)
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox,
)

try:
    from modules.i_image_module import IImageModule, BaseParamsWidget, NoParamsWidget
except ImportError:
    class IImageModule:
        pass
    class BaseParamsWidget(QWidget):
        def get_params(self) -> dict:
            return {}
    class NoParamsWidget(BaseParamsWidget):
        pass


# ===========================================================================
# Parameter Widgets
# ===========================================================================

class HistogramEqualizationParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Mode:"))
        self.combo = QComboBox()
        self.combo.addItems(["Grayscale", "Per Channel (RGB)"])
        layout.addWidget(self.combo)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"mode": self.combo.currentText()}


class ContrastStretchingParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(0.0, 254.0)
        self.min_spin.setValue(0.0)
        layout.addWidget(self.min_spin)
        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(1.0, 255.0)
        self.max_spin.setValue(255.0)
        layout.addWidget(self.max_spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"new_min": self.min_spin.value(), "new_max": self.max_spin.value()}


class LaplacianParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Enhancement Strength (0.0 - 1.0):"))
        self.blend = QDoubleSpinBox()
        self.blend.setRange(0.0, 1.0)
        self.blend.setSingleStep(0.1)
        self.blend.setValue(0.5)
        layout.addWidget(self.blend)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"blend": self.blend.value()}


class SharpeningParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Strength (0.5 - 5.0):"))
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.5, 5.0)
        self.strength.setSingleStep(0.5)
        self.strength.setValue(1.5)
        layout.addWidget(self.strength)
        layout.addWidget(QLabel("Blur Radius (1 - 10):"))
        self.radius = QSpinBox()
        self.radius.setRange(1, 10)
        self.radius.setValue(3)
        layout.addWidget(self.radius)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"strength": self.strength.value(), "radius": self.radius.value()}


class MedianFilterParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Kernel Size (odd, 3-15):"))
        self.size = QSpinBox()
        self.size.setRange(3, 15)
        self.size.setSingleStep(2)
        self.size.setValue(3)
        layout.addWidget(self.size)
        layout.addStretch()

    def get_params(self) -> dict:
        k = self.size.value()
        if k % 2 == 0:
            k += 1
        return {"kernel_size": k}


# ===========================================================================
# Core image processing functions  (pure NumPy)
# ===========================================================================

def _apply_per_channel(image: np.ndarray, func):
    if image.ndim == 2:
        return func(image)
    return np.stack([func(image[:, :, c]) for c in range(image.shape[2])], axis=2)


# 1. Histogram Equalization
def _equalize_channel(ch: np.ndarray) -> np.ndarray:
    ch = ch.astype(np.uint8)
    hist, _ = np.histogram(ch.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = int(cdf[cdf > 0].min())
    n = int(ch.size)
    lut = np.round((cdf - cdf_min) / max(n - cdf_min, 1) * 255).astype(np.uint8)
    return lut[ch]

def histogram_equalization(image: np.ndarray, mode: str = "Grayscale") -> np.ndarray:
    """
    Histogram Equalization (lecture slide 54).
    Formula: s_i = cumul(i) / number_of_pixels * 255
    """
    orig_dtype = image.dtype
    if mode == "Per Channel (RGB)" and image.ndim == 3:
        out = np.stack([_equalize_channel(image[:, :, c])
                        for c in range(image.shape[2])], axis=2)
    else:
        if image.ndim == 3:
            gray = (0.2989 * image[:, :, 0].astype(float) +
                    0.5870 * image[:, :, 1].astype(float) +
                    0.1140 * image[:, :, 2].astype(float)).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        eq = _equalize_channel(gray)
        out = np.stack([eq, eq, eq], axis=2) if image.ndim == 3 else eq
    return out.astype(orig_dtype)


# 2. Contrast Stretching
def contrast_stretching(image: np.ndarray,
                        new_min: float = 0.0,
                        new_max: float = 255.0) -> np.ndarray:
    """
    Linear Min-Max Contrast Stretching (lecture slide 49).
    Formula: s = (r - curr_min) * (new_max - new_min) / (curr_max - curr_min) + new_min
    """
    img_f = image.astype(float)
    c_min, c_max = img_f.min(), img_f.max()
    if c_max == c_min:
        return image
    stretched = (img_f - c_min) * ((new_max - new_min) / (c_max - c_min)) + new_min
    return np.clip(stretched, 0, 255).astype(image.dtype)


# 3. Laplacian Edge Enhancement
def laplacian_edge_enhancement(image: np.ndarray, blend: float = 0.5) -> np.ndarray:
    """
    Laplacian Edge Enhancement (lecture slides 70-72).
    Kernel:  [ 0  1  0 ]
             [ 1 -4  1 ]
             [ 0  1  0 ]
    Output = original - blend * laplacian(original)
    """
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=float)
    def enhance(ch):
        ch_f = ch.astype(float)
        padded = np.pad(ch_f, 1, mode="reflect")
        lap = np.zeros_like(ch_f)
        for i in range(3):
            for j in range(3):
                lap += kernel[i, j] * padded[i:i+ch_f.shape[0], j:j+ch_f.shape[1]]
        return np.clip(ch_f - blend * lap, 0, 255)
    return _apply_per_channel(image, enhance).astype(image.dtype)


# 4. Unsharp Mask (Sharpening)
def _gaussian_kernel_1d(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    sigma = max(radius / 3.0, 0.5)
    x = np.arange(size) - radius
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()

def _gauss_blur(ch: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    pad = len(k1d) // 2
    out = np.pad(ch, pad, mode="reflect")
    row = sum(w * out[pad:-pad, i:i+ch.shape[1]] for i, w in enumerate(k1d))
    out2 = np.pad(row, pad, mode="reflect")
    return sum(w * out2[i:i+ch.shape[0], pad:-pad] for i, w in enumerate(k1d))

def unsharp_mask(image: np.ndarray, strength: float = 1.5, radius: int = 3) -> np.ndarray:
    """
    Unsharp Mask Sharpening (derivative-based, lecture slide 63).
    mask      = original - gaussian_blur(original)
    sharpened = original + strength * mask
    """
    k1d = _gaussian_kernel_1d(radius)
    def sharpen(ch):
        f = ch.astype(float)
        return np.clip(f + strength * (f - _gauss_blur(f, k1d)), 0, 255)
    return _apply_per_channel(image, sharpen).astype(image.dtype)


# 5. Median Filter
def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Spatial Median Filter (lecture slide 57 — spatial neighbourhood).
    Uses NumPy stride tricks for efficient sliding-window computation.
    """
    k, pad = kernel_size, kernel_size // 2
    def med(ch):
        padded = np.pad(ch.astype(float), pad, mode="reflect")
        H, W = ch.shape
        windows = np.lib.stride_tricks.as_strided(
            padded,
            shape=(H, W, k, k),
            strides=padded.strides + padded.strides
        )
        return np.median(windows.reshape(H, W, k * k), axis=2)
    return np.clip(_apply_per_channel(image, med), 0, 255).astype(image.dtype)


# 6. Grayscale Conversion
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Grayscale Conversion using ITU-R BT.601 weights (lecture slide 37).
    Y = 0.2989*R + 0.5870*G + 0.1140*B
    """
    if image.ndim == 2:
        return image
    gray = np.clip(
        0.2989 * image[:,:,0].astype(float) +
        0.5870 * image[:,:,1].astype(float) +
        0.1140 * image[:,:,2].astype(float),
        0, 255
    ).astype(image.dtype)
    return np.stack([gray, gray, gray], axis=2)


# ===========================================================================
# Controls Widget
# ===========================================================================

class MahaIbrahimControlsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Maha Ibrahim Module</b>"))
        layout.addWidget(QLabel("Operation:"))
        self.combo = QComboBox()
        self.operations = {
            "Histogram Equalization":     HistogramEqualizationParamsWidget,
            "Contrast Stretching":        ContrastStretchingParamsWidget,
            "Laplacian Edge Enhancement": LaplacianParamsWidget,
            "Unsharp Mask (Sharpen)":     SharpeningParamsWidget,
            "Median Filter":              MedianFilterParamsWidget,
            "Grayscale Conversion":       NoParamsWidget,
        }
        for name in self.operations:
            self.combo.addItem(name)
        layout.addWidget(self.combo)
        self._params_area = QVBoxLayout()
        layout.addLayout(self._params_area)
        self._current_widget = None
        self.combo.currentTextChanged.connect(self._swap_params)
        self._swap_params(self.combo.currentText())
        layout.addStretch()

    def _swap_params(self, name: str):
        if self._current_widget is not None:
            self._params_area.removeWidget(self._current_widget)
            self._current_widget.setParent(None)
            self._current_widget.deleteLater()
        self._current_widget = self.operations.get(name, NoParamsWidget)()
        self._params_area.addWidget(self._current_widget)

    def get_params(self) -> dict:
        params = {"operation": self.combo.currentText()}
        if self._current_widget:
            params.update(self._current_widget.get_params())
        return params


# ===========================================================================
# Main Module Class
# ===========================================================================

class MahaIbrahimImageModule(IImageModule):

    def get_name(self) -> str:
        return "Maha Ibrahim"

    def get_controls_widget(self, parent=None) -> QWidget:
        return MahaIbrahimControlsWidget(parent)

    def process_image(self, image_data: np.ndarray,
                      metadata: dict, params: dict) -> np.ndarray:
        op  = params.get("operation", "")
        img = image_data.copy()

        if op == "Histogram Equalization":
            return histogram_equalization(img, params.get("mode", "Grayscale"))
        elif op == "Contrast Stretching":
            return contrast_stretching(img, params.get("new_min", 0.0),
                                            params.get("new_max", 255.0))
        elif op == "Laplacian Edge Enhancement":
            return laplacian_edge_enhancement(img, params.get("blend", 0.5))
        elif op == "Unsharp Mask (Sharpen)":
            return unsharp_mask(img, params.get("strength", 1.5),
                                     params.get("radius", 3))
        elif op == "Median Filter":
            return median_filter(img, params.get("kernel_size", 3))
        elif op == "Grayscale Conversion":
            return to_grayscale(img)

        return img