import os, json, tempfile
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QMessageBox, QFileDialog, QPushButton)
from PyQt5.QtCore import Qt

from qfluentwidgets import FluentWindow, FluentIcon

from ui.theme_manager import setup_theme
from ui.widgets.data_import_widget import DataImportWidget
from ui.widgets.fit_result_widget import FitResultWidget
from ui.widgets.chart_widget import ChartWidget
from ui.widgets.advice_table_widget import AdviceTableWidget

from core.circle_fitting import fit_least_squares
from core.deviation_analysis import analyze
from core.mold_advisor import apply_advice, get_advice_table
from core.report_generator import generate_and_save
from models.data_models import FeatureType


class MainWindow(FluentWindow):
    def __init__(self):
        setup_theme(dark=True)
        super().__init__()
        self._result = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        self.resize(1280, 860)
        self.setWindowTitle("\u6ce8\u5851\u4ef6\u6a21\u5177\u4fee\u6a21\u5206\u6790\u5de5\u5177")

        self.data_import = DataImportWidget()
        self.fit_result = FitResultWidget()
        self.chart = ChartWidget()
        self.advice_table = AdviceTableWidget()

        main_page = QWidget()
        main_page.setObjectName("main_page")
        main_layout = QVBoxLayout(main_page)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(self.chart)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.data_import)
        right_layout.addWidget(self.fit_result)
        right_layout.addWidget(self.advice_table, 1)
        splitter.addWidget(right)
        right.setMinimumWidth(280)

        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btns = {}
        for key, txt, slot in [
            ("pdf", "\U0001f4c4 \u5bfc\u51fa PDF", self._export_pdf),
            ("csv", "\U0001f4ca \u5bfc\u51fa CSV", self._export_csv),
            ("save", "\U0001f4be \u4fdd\u5b58\u4f1a\u8bdd", self._save_session),
            ("load", "\U0001f4c2 \u52a0\u8f7d\u4f1a\u8bdd", self._load_session),
        ]:
            btn = QPushButton(txt)
            btn.clicked.connect(slot)
            btn.setEnabled(False)
            btn.setStyleSheet("""
                QPushButton { background: #252542; color: #e8e8f0;
                              border: none; padding: 6px 14px;
                              border-radius: 4px; font-size: 9pt; }
                QPushButton:hover { background: #2d2d50; }
                QPushButton:disabled { color: #6b6b8a; }
            """)
            self._btns[key] = btn
            toolbar.addWidget(btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        self.addSubInterface(main_page, FluentIcon.HOME, "\u5206\u6790")

    def _connect_signals(self):
        self.fit_result.fit_clicked.connect(self._run_analysis)
        self.fit_result.feature_changed.connect(self._on_feature_change)

    def _run_analysis(self):
        pts = self.data_import.points
        if len(pts) < 3:
            QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u5148\u5bfc\u5165\u81f3\u5c11 3 \u4e2a\u6d4b\u91cf\u70b9")
            return
        fitted = fit_least_squares(pts)
        if fitted is None:
            QMessageBox.warning(self, "\u9519\u8bef", "\u62df\u5408\u5931\u8d25")
            return
        ft = self.fit_result.get_feature_type()
        self._result = analyze(pts, fitted, ft)
        apply_advice(self._result.polar_points, ft)
        self.fit_result.update_results(self._result)
        self.chart.set_result(self._result)
        self.advice_table.update_data(self._result.polar_points)
        for btn in self._btns.values():
            btn.setEnabled(True)

    def _on_feature_change(self, ft):
        if self._result is not None:
            self._result.feature_type = ft
            apply_advice(self._result.polar_points, ft)
            self.advice_table.update_data(self._result.polar_points)

    def _export_pdf(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "\u4fdd\u5b58 PDF", "", "PDF (*.pdf)")
        if not path:
            return
        try:
            self.chart._render()
            self.chart.fig.canvas.draw()
            import numpy as np
            buf = np.array(self.chart.fig.canvas.renderer.buffer_rgba())
            import cv2
            chart_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
            generate_and_save(self._result, chart_img, path)
            QMessageBox.information(self, "\u5b8c\u6210", f"PDF \u5df2\u4fdd\u5b58: {path}")
        except Exception as e:
            QMessageBox.warning(self, "\u5bfc\u51fa\u5931\u8d25", str(e))

    def _export_csv(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "\u4fdd\u5b58 CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["no", "angle", "delta_r", "x", "y", "advice"])
                w.writeheader()
                w.writerows(get_advice_table(self._result.polar_points))
            QMessageBox.information(self, "\u5b8c\u6210", f"CSV \u5df2\u4fdd\u5b58: {path}")
        except Exception as e:
            QMessageBox.warning(self, "\u5bfc\u51fa\u5931\u8d25", str(e))

    def _save_session(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(self, "\u4fdd\u5b58\u4f1a\u8bdd", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = {
                "result": self._result.to_dict(),
                "points": [{"x": p.x, "y": p.y} for p in self._result.points],
                "polar": [{"theta": p.theta, "delta_r": p.delta_r, "advice": p.advice}
                          for p in self._result.polar_points],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "\u5b8c\u6210", "\u4f1a\u8bdd\u5df2\u4fdd\u5b58")
        except Exception as e:
            QMessageBox.warning(self, "\u4fdd\u5b58\u5931\u8d25", str(e))

    def _load_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "\u52a0\u8f7d\u4f1a\u8bdd", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            from models.data_models import MeasurePoint, FittedCircle
            pts = [MeasurePoint(p["x"], p["y"]) for p in data["points"]]
            rd = data["result"]
            ft = FeatureType.CAVITY if rd.get("feature_type") == "\u578b\u8154" else FeatureType.CORE
            fitted = FittedCircle(rd["cx"], rd["cy"], rd["radius"], rd.get("rmse", 0))
            self._result = analyze(pts, fitted, ft)
            self._result.roundness = rd["roundness"]
            self._result.peak_error = rd["peak_error"]
            self._result.valley_error = rd["valley_error"]
            apply_advice(self._result.polar_points, ft)
            self.fit_result.update_results(self._result)
            self.chart.set_result(self._result)
            self.advice_table.update_data(self._result.polar_points)
            for btn in self._btns.values():
                btn.setEnabled(True)
            QMessageBox.information(self, "\u5b8c\u6210", "\u4f1a\u8bdd\u5df2\u52a0\u8f7d")
        except Exception as e:
            QMessageBox.warning(self, "\u52a0\u8f7d\u5931\u8d25", str(e))
