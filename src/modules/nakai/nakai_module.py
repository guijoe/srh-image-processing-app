from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
)
from PySide6.QtCore import Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule


# ---------------- Parameter widgets ----------------
class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("No parameters for this operation."))
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class ContrastStretchParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("New Min (0-255):"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(0, 255)
        self.min_spin.setValue(0)
        layout.addWidget(self.min_spin)

        layout.addWidget(QLabel("New Max (0-255):"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0, 255)
        self.max_spin.setValue(255)
        layout.addWidget(self.max_spin)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"new_min": float(self.min_spin.value()), "new_max": float(self.max_spin.value())}


class UnsharpParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Blur kernel size (odd: 3,5,7,...):"))
        self.ksize = QDoubleSpinBox()
        self.ksize.setRange(3, 25)
        self.ksize.setSingleStep(2)
        self.ksize.setValue(5)
        layout.addWidget(self.ksize)

        layout.addWidget(QLabel("Amount (0.5 to 2.0 typical):"))
        self.amount = QDoubleSpinBox()
        self.amount.setRange(0.0, 5.0)
        self.amount.setSingleStep(0.1)
        self.amount.setValue(1.2)
        layout.addWidget(self.amount)

        layout.addStretch()

    def get_params(self) -> dict:
        k = int(self.ksize.value())
        if k % 2 == 0:
            k += 1
        return {"ksize": k, "amount": float(self.amount.value())}


class PortraitBlurParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Blur Sigma (strength):"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.5, 25.0)
        self.sigma.setValue(8.0)
        layout.addWidget(self.sigma)

        layout.addWidget(QLabel("Focus radius (% of image min side):"))
        self.radius = QDoubleSpinBox()
        self.radius.setRange(5.0, 80.0)
        self.radius.setValue(30.0)
        layout.addWidget(self.radius)

        layout.addWidget(QLabel("Feather (%):"))
        self.feather = QDoubleSpinBox()
        self.feather.setRange(1.0, 50.0)
        self.feather.setValue(12.0)
        layout.addWidget(self.feather)

        layout.addWidget(QLabel("Center X (%):"))
        self.cx = QDoubleSpinBox()
        self.cx.setRange(0.0, 100.0)
        self.cx.setValue(50.0)
        layout.addWidget(self.cx)

        layout.addWidget(QLabel("Center Y (%):"))
        self.cy = QDoubleSpinBox()
        self.cy.setRange(0.0, 100.0)
        self.cy.setValue(45.0)
        layout.addWidget(self.cy)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "sigma": float(self.sigma.value()),
            "radius_pct": float(self.radius.value()),
            "feather_pct": float(self.feather.value()),
            "cx_pct": float(self.cx.value()),
            "cy_pct": float(self.cy.value()),
        }


class SwirlParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Strength (bigger = more twist):"))
        self.strength = QDoubleSpinBox()
        self.strength.setRange(0.0, 20.0)
        self.strength.setValue(6.0)
        layout.addWidget(self.strength)

        layout.addWidget(QLabel("Radius (% of image min side):"))
        self.radius = QDoubleSpinBox()
        self.radius.setRange(5.0, 100.0)
        self.radius.setValue(70.0)
        layout.addWidget(self.radius)

        layout.addWidget(QLabel("Center X (%):"))
        self.cx = QDoubleSpinBox()
        self.cx.setRange(0.0, 100.0)
        self.cx.setValue(50.0)
        layout.addWidget(self.cx)

        layout.addWidget(QLabel("Center Y (%):"))
        self.cy = QDoubleSpinBox()
        self.cy.setRange(0.0, 100.0)
        self.cy.setValue(50.0)
        layout.addWidget(self.cy)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "strength": float(self.strength.value()),
            "swirl_radius_pct": float(self.radius.value()),
            "swirl_cx_pct": float(self.cx.value()),
            "swirl_cy_pct": float(self.cy.value()),
        }


# ---------------- Controls widget ----------------
class NakkzControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Nakai Module</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Contrast Stretching": ContrastStretchParamsWidget,
            "Histogram Equalization": NoParamsWidget,
            "Unsharp Mask Sharpen": UnsharpParamsWidget,
            "Portrait Blur (Radial)": PortraitBlurParamsWidget,
            "Swirl Illusion": SwirlParamsWidget,
        }

        for name, widget_cls in operations.items():
            w = widget_cls()
            self.params_stack.addWidget(w)
            self.param_widgets[name] = w
            self.operation_selector.addItem(name)

        self.apply_btn = QPushButton("Apply Processing")
        layout.addWidget(self.apply_btn)
        layout.addStretch()

        self.apply_btn.clicked.connect(self._on_apply)
        self.operation_selector.currentTextChanged.connect(self._on_op_changed)

    def _on_op_changed(self, op_name: str):
        self.params_stack.setCurrentWidget(self.param_widgets[op_name])

    def _on_apply(self):
        op = self.operation_selector.currentText()
        params = self.param_widgets[op].get_params()
        params["operation"] = op
        self.process_requested.emit(params)


# ---------------- Numpy algorithms ----------------
def _clip_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    if x.max() <= 1.0:
        x *= 255.0
    return _clip_uint8(x)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return _ensure_uint8(img)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= (k.sum() + 1e-8)
    return k


def _convolve1d_reflect(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    pad = len(k) // 2
    pad_width = [(0, 0)] * img.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(img, pad_width, mode="reflect")

    def conv_line(line):
        return np.convolve(line, k, mode="valid")

    return np.apply_along_axis(conv_line, axis, padded)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    x = img.astype(np.float32)
    k = _gaussian_kernel1d(sigma)

    if x.ndim == 2:
        x = _convolve1d_reflect(x, k, axis=1)
        x = _convolve1d_reflect(x, k, axis=0)
        return x

    out = x.copy()
    for c in range(x.shape[2]):
        ch = x[..., c]
        ch = _convolve1d_reflect(ch, k, axis=1)
        ch = _convolve1d_reflect(ch, k, axis=0)
        out[..., c] = ch
    return out


def portrait_blur_radial(img: np.ndarray, sigma: float, radius_pct: float, feather_pct: float,
                         cx_pct: float, cy_pct: float) -> np.ndarray:
    img8 = _ensure_uint8(img)
    h, w = img8.shape[:2]

    blurred = gaussian_blur(img8, sigma)

    cx = (cx_pct / 100.0) * (w - 1)
    cy = (cy_pct / 100.0) * (h - 1)

    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    min_side = float(min(h, w))
    r0 = (radius_pct / 100.0) * min_side
    feather = (feather_pct / 100.0) * min_side

    t = (dist - r0) / (feather + 1e-6)
    t = np.clip(t, 0.0, 1.0)
    alpha = t * t * (3 - 2 * t)  # smoothstep

    if img8.ndim == 2:
        out = (1 - alpha) * img8.astype(np.float32) + alpha * blurred
        return _clip_uint8(out).astype(img.dtype)

    alpha3 = alpha[..., None]
    out = (1 - alpha3) * img8.astype(np.float32) + alpha3 * blurred
    return _clip_uint8(out).astype(img.dtype)


def swirl_warp(img: np.ndarray, strength: float = 6.0, radius_pct: float = 70.0,
               cx_pct: float = 50.0, cy_pct: float = 50.0) -> np.ndarray:
    img8 = _ensure_uint8(img)
    h, w = img8.shape[:2]

    cx = (cx_pct / 100.0) * (w - 1)
    cy = (cy_pct / 100.0) * (h - 1)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xx - cx
    y = yy - cy
    r = np.sqrt(x * x + y * y)

    R = (radius_pct / 100.0) * min(h, w)
    t = np.clip((R - r) / (R + 1e-6), 0.0, 1.0)
    angle = strength * (t * t)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    xs = cos_a * x - sin_a * y + cx
    ys = sin_a * x + cos_a * y + cy

    xs_i = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    ys_i = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)

    out = img8[ys_i, xs_i]
    return out.astype(img.dtype)


def contrast_stretch(img: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    x = img.astype(np.float32)
    cur_min = float(x.min())
    cur_max = float(x.max())
    if cur_max == cur_min:
        return img
    y = (x - cur_min) * ((new_max - new_min) / (cur_max - cur_min)) + new_min
    y = np.clip(y, new_min, new_max)
    return y.astype(img.dtype)


def hist_equalize(img: np.ndarray) -> np.ndarray:
    img8 = _to_uint8(img)

    if img8.ndim == 2:
        g = img8
        hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-12)
        lut = np.floor(255 * cdf + 0.5).astype(np.uint8)
        return lut[g].astype(img.dtype)

    rgb = img8[..., :3].astype(np.float32)
    y = rgb.mean(axis=2).astype(np.uint8)

    hist = np.bincount(y.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-12)
    lut = np.floor(255 * cdf + 0.5).astype(np.uint8)

    y_eq = lut[y].astype(np.float32)
    ratio = (y_eq / (y.astype(np.float32) + 1e-6))[..., None]
    out = np.clip(rgb * ratio, 0, 255).astype(np.uint8)

    if img8.shape[2] == 4:
        out = np.concatenate([out, img8[..., 3:4]], axis=2)

    return out.astype(img.dtype)


def _pad_reflect(a: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(a, ((pad, pad), (pad, pad)), mode="reflect")


def _box_blur(gray: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    g = gray.astype(np.float32)
    p = _pad_reflect(g, pad)
    out = np.zeros_like(g, dtype=np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = p[i:i + k, j:j + k].mean()
    return out


def unsharp_mask(img: np.ndarray, ksize: int, amount: float) -> np.ndarray:
    img8 = _to_uint8(img)

    if img8.ndim == 2:
        g = img8.astype(np.float32)
        blur = _box_blur(g, ksize)
        sharp = g + amount * (g - blur)
        return np.clip(sharp, 0, 255).astype(img.dtype)

    rgb = img8[..., :3].astype(np.float32)
    y = rgb.mean(axis=2)
    blur = _box_blur(y, ksize)
    y_sharp = y + amount * (y - blur)
    ratio = (y_sharp / (y + 1e-6))[..., None]
    out = np.clip(rgb * ratio, 0, 255).astype(np.uint8)

    if img8.shape[2] == 4:
        out = np.concatenate([out, img8[..., 3:4]], axis=2)

    return out.astype(img.dtype)


# ---------------- Module class ----------------
class NakkzImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls = None

    def get_name(self) -> str:
        return "Nakai Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls is None:
            self._controls = NakkzControlsWidget(module_manager, parent)
            self._controls.process_requested.connect(self._handle_request)
        return self._controls

    def _handle_request(self, params: dict):
        if self._controls and self._controls.module_manager:
            self._controls.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            img = imageio.imread(file_path)
            meta = {"name": file_path.split("\\")[-1]}
            return True, img, meta, None
        except Exception as e:
            print("Load error:", e)
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        op = params.get("operation", "")
        out = image_data.copy()

        if op == "Contrast Stretching":
            out = contrast_stretch(out, params.get("new_min", 0.0), params.get("new_max", 255.0))

        elif op == "Histogram Equalization":
            out = hist_equalize(out)

        elif op == "Unsharp Mask Sharpen":
            out = unsharp_mask(out, int(params.get("ksize", 5)), float(params.get("amount", 1.2)))

        elif op == "Portrait Blur (Radial)":
            out = portrait_blur_radial(
                out,
                sigma=float(params.get("sigma", 8.0)),
                radius_pct=float(params.get("radius_pct", 30.0)),
                feather_pct=float(params.get("feather_pct", 12.0)),
                cx_pct=float(params.get("cx_pct", 50.0)),
                cy_pct=float(params.get("cy_pct", 45.0)),
            )

        elif op == "Swirl Illusion":
            out = swirl_warp(
                out,
                strength=float(params.get("strength", 6.0)),
                radius_pct=float(params.get("swirl_radius_pct", 70.0)),
                cx_pct=float(params.get("swirl_cx_pct", 50.0)),
                cy_pct=float(params.get("swirl_cy_pct", 50.0)),
            )

        return out
