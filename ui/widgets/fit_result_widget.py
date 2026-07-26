from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
from PyQt5.QtCore import pyqtSignal
from models.data_models import FeatureType


class FitResultWidget(QWidget):
    feature_changed = pyqtSignal(FeatureType)
    fit_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.btn_fit = QPushButton("\U0001f3af \u6267\u884c\u62df\u5408")
        self.btn_fit.setStyleSheet("""
            QPushButton { background: #3b82f6; color: white;
                          border: none; padding: 6px 16px;
                          border-radius: 4px; font-size: 10pt; font-weight: bold; }
            QPushButton:hover { background: #2563eb; }
        """)
        self.btn_fit.clicked.connect(self.fit_clicked.emit)
        layout.addWidget(self.btn_fit)

        ft_label = QLabel("\u7279\u5f81\u7c7b\u578b:")
        ft_label.setStyleSheet("color: #9898b8; font-size: 9pt;")
        layout.addWidget(ft_label)

        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["\u578b\u8154(Cavity)", "\u578b\u82af(Core)"])
        self.feature_combo.currentIndexChanged.connect(self._on_change)
        self.feature_combo.setStyleSheet("""
            QComboBox { background: #252542; color: #e8e8f0;
                       border: none; padding: 4px 8px;
                       border-radius: 3px; font-size: 9pt; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background: #1e1e36; color: #e8e8f0;
                                         selection-background-color: #3b82f6; }
        """)
        layout.addWidget(self.feature_combo)

        sep = QLabel("\u2014 \u62df\u5408\u7ed3\u679c \u2014")
        sep.setStyleSheet("color: #6b6b8a; font-size: 9pt; padding: 4px 0;")
        layout.addWidget(sep)

        self._labels = {}
        for key, txt in [("cx", "X:"), ("cy", "Y:"), ("r", "R:"),
                          ("roundness", "\u5706\u5ea6:"), ("rmse", "RMSE:")]:
            lbl = QLabel(f"{txt} \u2014")
            lbl.setStyleSheet("color: #6b6b8a; font-size: 9pt;")
            layout.addWidget(lbl)
            self._labels[key] = lbl

        layout.addStretch()

    def _on_change(self, idx):
        ft = FeatureType.CAVITY if idx == 0 else FeatureType.CORE
        self.feature_changed.emit(ft)

    def get_feature_type(self):
        return FeatureType.CAVITY if self.feature_combo.currentIndex() == 0 else FeatureType.CORE

    def update_results(self, result):
        vals = {"cx": f"X: {result.fitted.cx:.4f}",
                "cy": f"Y: {result.fitted.cy:.4f}",
                "r": f"R: {result.fitted.radius:.4f}",
                "roundness": f"\u5706\u5ea6: {result.roundness:.4f}",
                "rmse": f"RMSE: {result.fitted.rmse:.6f}"}
        for key, text in vals.items():
            self._labels[key].setText(text)
            self._labels[key].setStyleSheet("color: #e8e8f0; font-size: 9pt;")
