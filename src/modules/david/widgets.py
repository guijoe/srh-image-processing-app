from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QDoubleSpinBox, QSpinBox, QProgressBar
)
from PySide6.QtCore import Signal, QThread


class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError


class SepiaParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Intensity (0 = original, 1 = full sepia):"))
        self.intensity = QDoubleSpinBox()
        self.intensity.setMinimum(0.0)
        self.intensity.setMaximum(1.0)
        self.intensity.setValue(1.0)
        self.intensity.setSingleStep(0.1)
        layout.addWidget(self.intensity)

        layout.addStretch()

    def get_params(self) -> dict:
        return {'intensity': self.intensity.value()}


class ChannelSwapParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Swap Mode:"))
        self.swap_mode = QComboBox()
        self.swap_mode.addItems(["R<->B", "R<->G", "G<->B", "Rotate"])
        layout.addWidget(self.swap_mode)

        info = QLabel(
            "R<->B: Swap red and blue (most dramatic)\n"
            "R<->G: Swap red and green\n"
            "G<->B: Swap green and blue\n"
            "Rotate: Shift all channels cyclically"
        )
        info.setStyleSheet("color: gray; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()

    def get_params(self) -> dict:
        return {'swap_mode': self.swap_mode.currentText()}


class VignetteParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Strength (edge darkness):"))
        self.strength = QDoubleSpinBox()
        self.strength.setMinimum(0.0)
        self.strength.setMaximum(1.0)
        self.strength.setValue(0.8)
        self.strength.setSingleStep(0.1)
        layout.addWidget(self.strength)

        layout.addWidget(QLabel("Radius (size of bright center):"))
        self.radius = QDoubleSpinBox()
        self.radius.setMinimum(0.2)
        self.radius.setMaximum(3.0)
        self.radius.setValue(1.0)
        self.radius.setSingleStep(0.1)
        layout.addWidget(self.radius)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'strength': self.strength.value(),
            'radius': self.radius.value(),
        }


class NegativeParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        info = QLabel("Inverts all pixel values.\nNo parameters needed.")
        info.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(info)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class EmbossParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Light Direction:"))
        self.direction = QComboBox()
        self.direction.addItems(["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"])
        layout.addWidget(self.direction)

        layout.addWidget(QLabel("Strength:"))
        self.strength = QDoubleSpinBox()
        self.strength.setMinimum(0.1)
        self.strength.setMaximum(5.0)
        self.strength.setValue(1.0)
        self.strength.setSingleStep(0.1)
        layout.addWidget(self.strength)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'direction': self.direction.currentText(),
            'strength': self.strength.value(),
        }


class PolarParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Transform Direction:"))
        self.direction = QComboBox()
        self.direction.addItems(["To Polar", "From Polar"])
        layout.addWidget(self.direction)

        info = QLabel(
            "To Polar: Unwrap circles into straight lines\n"
            "From Polar: Wrap image into circular pattern\n\n"
            "Note: This effect is slow on large images."
        )
        info.setStyleSheet("color: gray; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()

    def get_params(self) -> dict:
        return {'direction': self.direction.currentText()}


class GlitchParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Intensity (corruption amount):"))
        self.intensity = QDoubleSpinBox()
        self.intensity.setMinimum(0.1)
        self.intensity.setMaximum(1.0)
        self.intensity.setValue(0.5)
        self.intensity.setSingleStep(0.1)
        layout.addWidget(self.intensity)

        layout.addWidget(QLabel("Random Seed (change for different glitch):"))
        self.seed = QSpinBox()
        self.seed.setMinimum(0)
        self.seed.setMaximum(9999)
        self.seed.setValue(42)
        layout.addWidget(self.seed)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'intensity': self.intensity.value(),
            'seed': self.seed.value(),
        }


class ProcessingWorker(QThread):
    finished = Signal()

    def __init__(self, module_manager, params, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.params = params

    def run(self):
        self.module_manager.apply_processing_to_current_image(self.params)
        self.finished.emit()


OPERATIONS = {
    "Sepia Tone": SepiaParamsWidget,
    "Channel Swap": ChannelSwapParamsWidget,
    "Vignette": VignetteParamsWidget,
    "Negative": NegativeParamsWidget,
    "Emboss / Bas-Relief": EmbossParamsWidget,
    "Polar Transform": PolarParamsWidget,
    "Glitch Art": GlitchParamsWidget,
}


class DavidControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<h3>David's Module</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        for name, widget_class in OPERATIONS.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setFormat("Processing...")
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.apply_button.clicked.connect(self._on_apply)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply(self):
        if self._worker and self._worker.isRunning():
            return

        operation_name = self.operation_selector.currentText()
        params = self.param_widgets[operation_name].get_params()
        params['operation'] = operation_name

        self.apply_button.setEnabled(False)
        self.apply_button.setText("Processing...")
        self.progress_bar.show()

        self._worker = ProcessingWorker(self.module_manager, params)
        self._worker.finished.connect(self._on_processing_done)
        self._worker.start()

    def _on_processing_done(self):
        self.progress_bar.hide()
        self.apply_button.setEnabled(True)
        self.apply_button.setText("Apply Processing")
        self._worker = None

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])
