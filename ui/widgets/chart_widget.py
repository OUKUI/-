import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QCheckBox, QLabel, QComboBox
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._view_mode = "polar"
        self._k = 20.0
        self._show_fitted = True
        self._show_profile = True
        self._show_center = True
        self._show_labels = True
        self._show_dist_labels = False
        self._unit = "mm"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.fig = Figure(tight_layout=True, facecolor="#12121e")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #12121e;")

        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)

        lbl_k = QLabel("\u653e\u5927 k:")
        lbl_k.setStyleSheet("color: #9898b8; font-size: 9pt;")
        ctrl.addWidget(lbl_k)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 500)
        self.slider.setValue(20)
        self.slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self.slider)

        self.lbl_k = QLabel("20")
        self.lbl_k.setStyleSheet("color: #e8e8f0; font-size: 9pt;")
        ctrl.addWidget(self.lbl_k)

        sep = QLabel("|")
        sep.setStyleSheet("color: #333355; font-size: 11pt;")
        ctrl.addWidget(sep)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["\u6781\u5750\u6807\u504f\u5dee", "\u5e73\u9762\u5750\u6807"])
        self.view_combo.currentIndexChanged.connect(self._on_view_change)
        self.view_combo.setStyleSheet("""
            QComboBox { background: #252542; color: #e8e8f0;
                       border: none; padding: 3px 8px;
                       border-radius: 3px; font-size: 9pt; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background: #1e1e36; color: #e8e8f0;
                                         selection-background-color: #3b82f6; }
        """)
        ctrl.addWidget(self.view_combo)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #333355; font-size: 11pt;")
        ctrl.addWidget(sep2)

        lbl0 = QLabel("0\u00b0:")
        lbl0.setStyleSheet("color: #9898b8; font-size: 9pt;")
        ctrl.addWidget(lbl0)

        self._zero_combo = QComboBox()
        self._zero_combo.addItems(["\u53f3(3\u70b9)", "\u4e0a(12\u70b9)", "\u5de6(9\u70b9)", "\u4e0b(6\u70b9)"])
        self._zero_combo.currentIndexChanged.connect(lambda: self._render())
        self._zero_combo.setStyleSheet("""
            QComboBox { background: #252542; color: #e8e8f0;
                       border: none; padding: 3px 6px;
                       border-radius: 3px; font-size: 9pt; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView { background: #1e1e36; color: #e8e8f0;
                                         selection-background-color: #3b82f6; }
        """)
        ctrl.addWidget(self._zero_combo)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        toggles = QHBoxLayout()
        toggles.setSpacing(6)
        self._checkboxes = {}
        for key, txt in [("fitted", "\u62df\u5408\u5706"), ("profile", "\u5b9e\u6d4b\u8f6e\u5ed3"),
                          ("center", "\u5706\u5fc3"), ("labels", "\u504f\u5dee\u6807\u6ce8"),
                          ("dist", "\u8ddd\u5706\u5fc3\u8ddd")]:
            cb = QCheckBox(txt)
            cb.setChecked(True)
            cb.stateChanged.connect(lambda _, k=key: self._toggle(k))
            cb.setStyleSheet("color: #e8e8f0; font-size: 8pt;")
            toggles.addWidget(cb)
            self._checkboxes[key] = cb
        self._checkboxes["dist"].setChecked(False)
        toggles.addStretch()
        layout.addLayout(toggles)

        layout.addWidget(self.canvas)

    def set_result(self, result):
        self._result = result
        self._render()

    def _on_slider(self, val):
        self._k = val
        self.lbl_k.setText(str(val))
        self._render()

    def _on_view_change(self, idx):
        self._view_mode = "polar" if idx == 0 else "xy"
        self._render()

    def _toggle(self, name):
        state = self._checkboxes[name].isChecked()
        m = {"fitted": "_show_fitted", "profile": "_show_profile", "center": "_show_center",
             "labels": "_show_labels", "dist": "_show_dist_labels"}
        setattr(self, m[name], state)
        self._render()

    def _render(self):
        self.fig.clear()
        if not self._result:
            self.canvas.draw_idle()
            return

        r = self._result
        polar = r.polar_points
        thetas = np.array([p.theta for p in polar])
        deltas = np.array([p.delta_r for p in polar])
        thetas_loop = np.append(thetas, thetas[0] + 2 * np.pi)
        deltas_loop = np.append(deltas, deltas[0])

        if self._view_mode == "polar":
            self._render_polar(thetas, deltas, thetas_loop, deltas_loop, r)
        else:
            self._render_xy(r)

        self.canvas.draw_idle()

    def _render_polar(self, thetas, deltas, thetas_loop, deltas_loop, r):
        ax = self.fig.add_subplot(111, polar=True)
        ax.set_facecolor("#12121e")
        zero_map = {"\u53f3(3\u70b9)": 0, "\u4e0a(12\u70b9)": 90,
                     "\u5de6(9\u70b9)": 180, "\u4e0b(6\u70b9)": 270}
        ax.set_theta_offset(np.radians(zero_map.get(self._zero_combo.currentText(), 0)))

        max_err = max(abs(r.peak_error), abs(r.valley_error), 1e-6)
        rlim = max_err * 1.4 * (self._k / 20.0)
        offset = rlim
        dev_shift = deltas + offset

        if self._show_profile:
            ax.fill_between(thetas, offset, dev_shift, alpha=0.3,
                            where=(deltas >= 0), color="#f87171")
            ax.fill_between(thetas, offset, dev_shift, alpha=0.3,
                            where=(deltas < 0), color="#4ade80")
            ax.plot(thetas, dev_shift, "-", color="#93c5fd", lw=1.5)
            ax.plot(thetas, dev_shift, ".", color="#e0e7ff", ms=3)

        if self._show_fitted:
            ax.plot(thetas_loop, np.full_like(thetas_loop, offset), "-",
                    color="#555577", lw=1, alpha=0.5)

        if self._show_center:
            ax.plot(thetas_loop, np.full_like(thetas_loop, r.peak_error + offset),
                    ":", color="#f87171", lw=0.6)
            ax.plot(thetas_loop, np.full_like(thetas_loop, r.valley_error + offset),
                    ":", color="#4ade80", lw=0.6)

        if self._show_labels:
            self._annotate_extremes(ax, thetas, deltas, offset, r)

        if self._show_dist_labels and hasattr(r, 'polar_points'):
            n = len(thetas)
            outer_r = 2 * offset
            label_gap = outer_r * 0.20
            thetas_deg = np.degrees(thetas) % 360
            closest_0 = np.argmin(np.minimum(thetas_deg, 360 - thetas_deg))
            for k in range(n):
                idx = (closest_0 + k) % n
                t = thetas[idx]
                r_val = dev_shift[idx]
                stagger = (k % 5) * label_gap * 0.5
                label_r = outer_r + stagger
                ax.plot([t, t], [r_val, outer_r], "-",
                        color="#6b6b8a", lw=0.3, alpha=0.35)
                num = k + 1
                ax.annotate(f"#{num} {r.polar_points[idx].r:.3f}",
                            xy=(t, label_r), fontsize=4.5,
                            color="#6b6b8a", ha="center", va="bottom")

        max_r = 2 * offset + (offset * 0.6 if self._show_dist_labels else 0)
        ax.set_ylim(0, max_r)
        ax.set_yticks(np.linspace(0, 2 * offset, 5))
        ax.set_yticklabels([f"{v - offset:.3f}" for v in np.linspace(0, 2 * offset, 5)],
                           color="#9898b8", fontsize=6)
        ax.tick_params(colors="#9898b8", labelsize=7)
        ax.grid(True, alpha=0.15, color="#555577")

        info = f"偏差 [k={self._k}]  圆度={r.roundness:.4f}{r.unit}  Peak=+{r.peak_error:.4f}  Valley={r.valley_error:.4f}"
        ax.set_title(info, color="#e8e8f0", fontsize=9, pad=12)

    def _render_xy(self, r):
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#12121e")
        pts = r.points
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]

        if self._show_fitted:
            circle = plt.Circle((r.fitted.cx, r.fitted.cy), r.fitted.radius,
                                fill=False, linestyle="--", color="#555577", lw=1)
            ax.add_patch(circle)

        if self._show_profile:
            polar = r.polar_points
            oxs = [p.x for p in polar] + [polar[0].x]
            oys = [p.y for p in polar] + [polar[0].y]
            ax.plot(oxs, oys, "-", color="#93c5fd", lw=1.5)
            ax.scatter(xs, ys, c="#e0e7ff", s=8)

        if self._show_center:
            ax.scatter([r.fitted.cx], [r.fitted.cy], c="#fbbf24", marker="+", s=60, lw=1.5)

        ax.set_aspect("equal")
        ax.tick_params(colors="#9898b8", labelsize=7)
        ax.set_title(f"XY 视图  [R={r.fitted.radius:.2f}]", color="#e8e8f0", fontsize=10)
        ax.grid(True, alpha=0.1, color="#333355")

    def _annotate_extremes(self, ax, thetas, deltas, offset, r):
        peak_idx = np.argmax(deltas)
        valley_idx = np.argmin(deltas)
        for idx, val, clr, lbl in [
            (peak_idx, r.peak_error, "#f87171", f"+{r.peak_error:.4f}"),
            (valley_idx, r.valley_error, "#4ade80", f"{r.valley_error:.4f}"),
        ]:
            ax.annotate(lbl, xy=(thetas[idx], deltas[idx] + offset),
                        fontsize=7, color=clr, ha="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e", ec=clr, lw=0.5))


import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
