"""
SRH ImageViewer - Football Vision Analytics Module
Author: Computer Science Capstone Candidate
Description: A high-performance, purely NumPy-driven image processing module 
             tailored for football match imagery analytics. Fully compliant 
             with all abstract framework requirements.
"""

import os
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSlider, QPushButton, QComboBox, QDoubleSpinBox)
from PySide6.QtCore import Qt

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# =============================================================================
# LOCAL PARAMETER WIDGET CLASSES
# =============================================================================

class NoParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("No configuration parameters required."))
        layout.addStretch()
    def get_params(self) -> dict:
        return {}


class ContrastStretchingParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Target Floor Intensity (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setRange(0.0, 255.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)
        layout.addWidget(QLabel("Target Ceiling Intensity (0-255):"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setRange(0.0, 255.0)
        self.max_spinbox.setValue(255.0)
        layout.addWidget(self.max_spinbox)
        layout.addStretch()
        
    def get_params(self) -> dict:
        return {'new_min': self.min_spinbox.value(), 'new_max': self.max_spinbox.value()}


class PowerLawParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Gamma Scaling Multiplier:"))
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(1, 50)
        self.gamma_slider.setValue(10)
        self.lbl = QLabel("1.0")
        self.gamma_slider.valueChanged.connect(lambda v: self.lbl.setText(f"{v/10.0:.1f}"))
        hb = QHBoxLayout()
        hb.addWidget(self.gamma_slider)
        hb.addWidget(self.lbl)
        layout.addLayout(hb)
        layout.addStretch()
        
    def get_params(self) -> dict:
        return {'gamma_val': float(self.gamma_slider.value() / 10.0)}


class GaussianParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Gaussian Kernel Window Dimensions:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(3, 31)
        self.slider.setValue(5)
        self.slider.setSingleStep(2)
        self.lbl = QLabel("5x5")
        self.slider.valueChanged.connect(self._verify_odd_constraint)
        hb = QHBoxLayout()
        hb.addWidget(self.slider)
        hb.addWidget(self.lbl)
        layout.addLayout(hb)
        layout.addStretch()
        
    def _verify_odd_constraint(self, value: int):
        if value % 2 == 0:
            self.slider.setValue(value + 1)
        self.lbl.setText(f"{self.slider.value()}x{self.slider.value()}")
        
    def get_params(self) -> dict:
        return {'blur_kernel': self.slider.value()}


class BinaryThresholdParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Binary Cut-Off Level:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(127)
        self.lbl = QLabel("127")
        self.slider.valueChanged.connect(lambda v: self.lbl.setText(str(v)))
        hb = QHBoxLayout()
        hb.addWidget(self.slider)
        hb.addWidget(self.lbl)
        layout.addLayout(hb)
        layout.addStretch()
        
    def get_params(self) -> dict:
        return {'threshold_val': self.slider.value()}


# =============================================================================
# CONTROL PANEL PANEL REGISTRATION
# =============================================================================

class FootballVisionControlsWidget(QWidget):
    def __init__(self, module_manager, module_instance, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.module_instance = module_instance
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Operation:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems([
            "Original Matrix",
            "Contrast Stretching",
            "Histogram Equalization",
            "Power Law (Gamma)",
            "Gaussian Blur",
            "Sobel Edge Detect",
            "Binary Thresholding",
            "Green Pitch Detection",
            "Heatmap Color Mapping"
        ])
        layout.addWidget(self.op_combo)
        
        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)
        self.param_layout.setContentsMargins(0, 10, 0, 10)
        layout.addWidget(self.param_container)
        
        self.param_widgets = {
            "Original Matrix": NoParamsWidget,
            "Contrast Stretching": ContrastStretchingParamsWidget,
            "Histogram Equalization": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Binary Thresholding": BinaryThresholdParamsWidget,
            "Green Pitch Detection": NoParamsWidget,
            "Heatmap Color Mapping": NoParamsWidget
        }
        
        self.current_param_widget = None
        self.op_combo.currentTextChanged.connect(self._on_operation_changed)
        self._on_operation_changed(self.op_combo.currentText())
        
        self.apply_btn = QPushButton("Apply Processing")
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.apply_btn)
        layout.addStretch()

    def _on_operation_changed(self, text):
        if self.current_param_widget:
            self.param_layout.removeWidget(self.current_param_widget)
            self.current_param_widget.deleteLater()
        widget_class = self.param_widgets.get(text, NoParamsWidget)
        self.current_param_widget = widget_class()
        self.param_layout.addWidget(self.current_param_widget)

    def _on_apply_clicked(self):
        params = {}
        if self.current_param_widget:
            params = self.current_param_widget.get_params()
        params['operation'] = self.op_combo.currentText()
        self.module_manager.apply_processing_to_current_image(params)


# =============================================================================
# MASTER CORE MODULE IMPLEMENTATION (FULLY ABSTRACT COMPLIANT)
# =============================================================================

class FootballVisionImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self.name = "Football Vision Module"

    def get_name(self) -> str:
        return self.name

    def get_supported_formats(self) -> list:
        """Satisfies the framework's required abstract base method constraint."""
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    def create_control_widget(self, module_manager) -> QWidget:
        return FootballVisionControlsWidget(module_manager, self)

    def load_image(self, file_path: str):
        try:
            img = Image.open(file_path)
            image_data = np.array(img)
            metadata = {'source': file_path, 'name': os.path.basename(file_path)}
            session_id = str(os.path.basename(file_path))
            return True, image_data, metadata, session_id
        except Exception as e:
            print(f"Error loading image in module: {e}")
            return False, None, {}, ""

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        if image_data is None or image_data.size == 0:
            return image_data
        processed_data = image_data.copy()
        operation = params.get('operation', 'Original Matrix')
        try:
            if operation == "Original Matrix": return processed_data
            elif operation == "Contrast Stretching": return self._apply_contrast_stretching(processed_data, params)
            elif operation == "Histogram Equalization": return self._apply_histogram_equalization(processed_data)
            elif operation == "Power Law (Gamma)": return self._apply_gamma_correction(processed_data, params)
            elif operation == "Gaussian Blur": return self._apply_gaussian_blur(processed_data, params)
            elif operation == "Sobel Edge Detect": return self._apply_sobel_edges(processed_data)
            elif operation == "Binary Thresholding": return self._apply_binary_threshold(processed_data, params)
            elif operation == "Green Pitch Detection": return self._advanced_pitch_detection(processed_data)
            elif operation == "Heatmap Color Mapping": return self._advanced_heatmap_mapping(processed_data)
        except Exception as err:
            print(f"[PIPELINE RUNTIME FAULT] Failure during {operation}: {str(err)}")
            return image_data
        return processed_data

    # Algorithms remain pure NumPy execution blocks
    def _apply_contrast_stretching(self, img: np.ndarray, params: dict) -> np.ndarray:
        img_float = img.astype(float)
        new_min, new_max = params.get('new_min', 0.0), params.get('new_max', 255.0)
        if len(img_float.shape) == 3:
            out = np.zeros_like(img_float)
            for c in range(img_float.shape[2]):
                c_min, c_max = np.min(img_float[..., c]), np.max(img_float[..., c])
                if c_max == c_min: out[..., c] = img_float[..., c]
                else: out[..., c] = (img_float[..., c] - c_min) * ((new_max - new_min) / (c_max - c_min)) + new_min
            return np.clip(out, new_min, new_max).astype(img.dtype)
        else:
            c_min, c_max = np.min(img_float), np.max(img_float)
            if c_max == c_min: return img
            out = (img_float - c_min) * ((new_max - new_min) / (c_max - c_min)) + new_min
            return np.clip(out, new_min, new_max).astype(img.dtype)

    def _apply_histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            r, g, b = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
            y = 0.299 * r + 0.587 * g + 0.114 * b
            hist = np.bincount(y.astype(np.uint8).flatten(), minlength=256)
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            lut = np.ma.filled(cdf_m, 0).astype(np.uint8)
            y_equalized = lut[y.astype(np.uint8)].astype(float)
            scale_idx = y != 0
            scale = np.ones_like(y)
            scale[scale_idx] = y_equalized[scale_idx] / y[scale_idx]
            out = np.zeros_like(img)
            out[..., 0] = np.clip(r * scale, 0, 255)
            out[..., 1] = np.clip(g * scale, 0, 255)
            out[..., 2] = np.clip(b * scale, 0, 255)
            return out.astype(np.uint8)
        else:
            hist = np.bincount(img.flatten(), minlength=256)
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            lut = np.ma.filled(cdf_m, 0).astype(np.uint8)
            return lut[img]

    def _apply_gamma_correction(self, img: np.ndarray, params: dict) -> np.ndarray:
        gamma = params.get('gamma_val', 1.0)
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype(np.uint8)
        return lut[img]

    def _apply_gaussian_blur(self, img: np.ndarray, params: dict) -> np.ndarray:
        k_size = params.get('blur_kernel', 5)
        sigma = max(k_size / 6.0, 0.5)
        ax = np.linspace(-(k_size // 2), k_size // 2, k_size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel_2d = np.outer(gauss, gauss)
        kernel = kernel_2d / np.sum(kernel_2d)
        def conv2d(ch: np.ndarray, k: np.ndarray) -> np.ndarray:
            kh, kw = k.shape
            h, w = ch.shape
            padded = np.pad(ch, ((kh//2, kh//2), (kw//2, kw//2)), mode='edge')
            from numpy.lib.stride_tricks import as_strided
            shape = (h, w, kh, kw)
            strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
            sub_matrices = as_strided(padded, shape=shape, strides=strides)
            return np.einsum('ijmn,mn->ij', sub_matrices, k).astype(np.uint8)
        if len(img.shape) == 3:
            out = np.zeros_like(img)
            for c in range(img.shape[2]): out[..., c] = conv2d(img[..., c], kernel)
            return out
        return conv2d(img, kernel)

    def _apply_sobel_edges(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3: gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
        else: gray = img.copy()
        g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        h, w = gray.shape
        padded = np.pad(gray, 1, mode='edge')
        from numpy.lib.stride_tricks import as_strided
        shape = (h, w, 3, 3)
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        sub_matrices = as_strided(padded, shape=shape, strides=strides)
        grad_x = np.einsum('ijmn,mn->ij', sub_matrices, g_x)
        grad_y = np.einsum('ijmn,mn->ij', sub_matrices, g_y)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        if magnitude.max() > 0: magnitude = (magnitude / magnitude.max()) * 255
        return magnitude.astype(np.uint8)

    def _apply_binary_threshold(self, img: np.ndarray, params: dict) -> np.ndarray:
        thresh = params.get('threshold_val', 127)
        if len(img.shape) == 3: gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
        else: gray = img
        return np.where(gray >= thresh, np.uint8(255), np.uint8(0))

    def _advanced_pitch_detection(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) != 3: return img
        r, g, b = img[..., 0] / 255.0, img[..., 1] / 255.0, img[..., 2] / 255.0
        mx, mn = np.max(img, axis=2) / 255.0, np.min(img, axis=2) / 255.0
        df = mx - mn
        h = np.zeros_like(mx)
        idx = df != 0
        h[idx & (mx == r)] = (60 * ((g[idx & (mx == r)] - b[idx & (mx == r)]) / df[idx & (mx == r)]) % 360)
        h[idx & (mx == g)] = (60 * ((b[idx & (mx == g)] - r[idx & (mx == g)]) / df[idx & (mx == g)]) + 120)
        h[idx & (mx == b)] = (60 * ((r[idx & (mx == b)] - g[idx & (mx == b)]) / df[idx & (mx == b)]) + 240)
        s = np.zeros_like(mx)
        s[mx != 0] = df[mx != 0] / mx[mx != 0]
        h_scaled, s_scaled, v_scaled = (h / 2.0).astype(np.uint8), (s * 255.0).astype(np.uint8), (mx * 255.0).astype(np.uint8)
        pitch_mask = (h_scaled >= 35) & (h_scaled <= 85) & (s_scaled >= 40) & (v_scaled >= 40)
        out = img.copy()
        out[~pitch_mask] = 0
        return out

    def _advanced_heatmap_mapping(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3: gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
        else: gray = img
        smooth_gray = self._apply_gaussian_blur(gray, {'blur_kernel': 15})
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            r = int(np.clip(255 * (i - 127) / 128 if i >= 127 else 0, 0, 255))
            g = int(np.clip(255 * (127 - abs(i - 127)) / 127 if i > 0 else 0, 0, 255))
            b = int(np.clip(255 * (127 - i) / 128 if i <= 127 else 0, 0, 255))
            lut[i] = [r, g, b]
        return lut[smooth_gray]