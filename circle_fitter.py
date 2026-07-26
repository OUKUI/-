"""
圆度拟合工具 - Circle Roundness Fitter
Fluent Design · Dark Mode · High DPI
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import os
import json
import math
import re
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

COLORS = {
    "bg_dark": "#1a1a2e",
    "bg_card": "#1e1e36",
    "bg_surface": "#252542",
    "bg_hover": "#2d2d50",
    "accent": "#60a5fa",
    "accent_hover": "#3b82f6",
    "success": "#4ade80",
    "warning": "#fbbf24",
    "error": "#f87171",
    "text_primary": "#e8e8f0",
    "text_secondary": "#9898b8",
    "text_muted": "#6b6b8a",
    "border": "#333355",
    "canvas_bg": "#12121e",
}

# ── Circle Fitting Algorithms ─────────────────────────────────────────────────

class CircleFitResult:
    """Result of a circle fit operation."""
    def __init__(self, cx, cy, radius, points, method="least_squares"):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.points = np.array(points, dtype=np.float64)
        self.method = method
        self._compute_errors()

    def _compute_errors(self):
        if len(self.points) < 3:
            self.errors = np.array([])
            self.rmse = 0.0
            self.roundness = 0.0
            self.peak_error = 0.0
            self.valley_error = 0.0
            return
        dx = self.points[:, 0] - self.cx
        dy = self.points[:, 1] - self.cy
        distances = np.sqrt(dx * dx + dy * dy)
        self.errors = distances - self.radius
        self.rmse = float(np.sqrt(np.mean(self.errors ** 2)))
        self.peak_error = float(np.max(self.errors))
        self.valley_error = float(np.min(self.errors))
        self.roundness = float(np.max(distances) - np.min(distances))

    def to_dict(self):
        return {
            "cx": round(self.cx, 4),
            "cy": round(self.cy, 4),
            "radius": round(self.radius, 4),
            "roundness": round(self.roundness, 4),
            "rmse": round(self.rmse, 6),
            "peak_error": round(self.peak_error, 4),
            "valley_error": round(self.valley_error, 4),
            "method": self.method,
            "n_points": len(self.points),
            "timestamp": datetime.now().isoformat(),
        }


def fit_circle_least_squares(points):
    """
    Algebraic least-squares circle fit.
    Minimizes Σ((x²+y²) - A·x - B·y + C)²
    """
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 3:
        return None

    x, y = pts[:, 0], pts[:, 1]
    z = x * x + y * y

    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs

    cx = a / 2.0
    cy = b / 2.0
    radius = np.sqrt(cx * cx + cy * cy + c)

    return CircleFitResult(cx, cy, radius, pts, "least_squares")


def fit_circle_taubin(points):
    """
    Taubin method (geometric fit).
    Uses renormalized least squares for robustness to noise.
    """
    result = fit_circle_least_squares(points)
    if result is not None:
        result.method = "taubin"
    return result


# ── Image Processing ──────────────────────────────────────────────────────────

class ImageProcessor:
    @staticmethod
    def detect_edges_opencv(image, low_thresh=50, high_thresh=150):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        return edges

    @staticmethod
    def find_circle_contours(edges, min_radius_ratio=0.05, max_radius_ratio=0.95):
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = edges.shape
        max_dim = max(h, w)
        valid = []
        for cnt in contours:
            if len(cnt) < 20:
                continue
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.3:
                _, radius = cv2.minEnclosingCircle(cnt)
                r_ratio = radius / max_dim
                if min_radius_ratio < r_ratio < max_radius_ratio:
                    valid.append((cnt, circularity, radius))
        valid.sort(key=lambda x: -x[1])
        return [v[0] for v in valid[:5]]

    @staticmethod
    def sample_contour_points(contour, n_samples=200):
        contour = contour.squeeze()
        if len(contour.shape) != 2:
            return np.array([])
        if len(contour) > n_samples:
            indices = np.linspace(0, len(contour) - 1, n_samples, dtype=int)
            return contour[indices].astype(np.float64)
        return contour.astype(np.float64)

    @staticmethod
    def hough_circles(image, dp=1.2, min_dist=50, param1=100, param2=30, min_radius=10, max_radius=500):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
            param1=param1, param2=param2,
            minRadius=min_radius, maxRadius=max_radius
        )
        if circles is not None:
            return np.round(circles[0]).astype(int)
        return None


# ── Main Application ──────────────────────────────────────────────────────────

class CircleFitterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window config
        self.title("圆度拟合工具 · Circle Roundness Fitter")
        self.geometry("1280x900")
        self.minsize(960, 680)

        # High DPI scaling
        try:
            ctk.set_widget_scaling(1.0)
            ctk.set_window_scaling(1.0)
        except Exception:
            pass

        # State
        self.original_image = None
        self.display_image = None
        self.display_photo = None
        self.canvas_image = None
        self.points = []  # user-selected or auto-detected points
        self.fit_result = None
        self.zoom_scale = 1.0
        self.pan_offset = [0, 0]
        self.image_position = [0, 0]
        self.image_size = [0, 0]
        self.is_dragging = False
        self.drag_start = None
        self.mode = "auto"  # "auto" or "manual"
        self.selected_method = "Taubin"
        self.auto_fit_enabled = True
        self.show_errors = True
        self.show_center = True
        self.show_contours = True
        self.edge_cache = None
        self._drag_start = None
        self._was_drag = False
        self.showing_profile = False
        self._profile_img = None

        # Setup UI
        self.setup_ui()

        # Keyboard shortcuts
        self.bind("<Control-o>", lambda e: self.load_image())
        self.bind("<Control-r>", lambda e: self.reset_all())
        self.bind("<Control-f>", lambda e: self.fit_circle())
        self.bind("<Control-v>", lambda e: self.paste_data())

    # ── UI Setup ───────────────────────────────────────────────────────────

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ── Row 0: Title bar ──
        tb = ctk.CTkFrame(self, height=26, corner_radius=0, fg_color=COLORS["bg_dark"])
        tb.grid(row=0, column=0, sticky="ew")
        tb.grid_propagate(False)
        ctk.CTkLabel(tb, text="✦ 圆度拟合 · Circle Roundness Fitter",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left", padx=12, pady=3)

        # ── Row 1: Toolbar ──
        bar = ctk.CTkFrame(self, height=32, corner_radius=0, fg_color=COLORS["bg_card"])
        bar.grid(row=1, column=0, sticky="ew")
        bar.grid_propagate(False)
        BS = {"height": 26, "border_width": 0, "corner_radius": 5}
        ZS = {"width": 24, "height": 22, "border_width": 0, "corner_radius": 4}

        btns = [
            ("📂", self.load_image), ("📋", self.paste_data),
            ("🎯", self.fit_circle), ("↺", self.reset_all),
            ("📈", self.show_profile), ("💾", self.export_results),
        ]
        self._tool_btns = []
        for txt, cmd in btns:
            bg = COLORS["accent"] if txt == "🎯" else COLORS["bg_surface"]
            hc = COLORS["accent_hover"] if txt == "🎯" else COLORS["bg_hover"]
            tc = "#ffffff" if txt == "🎯" else COLORS["text_primary"]
            st = "disabled" if txt in ("🎯", "📈", "💾") else "normal"
            b = ctk.CTkButton(bar, text=txt, command=cmd, fg_color=bg,
                              hover_color=hc, text_color=tc,
                              state=st, width=32, **BS)
            b.pack(side="left", padx=1, pady=3)
            self._tool_btns.append(b)
        self.btn_fit, self.btn_profile, self.btn_export = self._tool_btns[2], self._tool_btns[4], self._tool_btns[5]

        ctk.CTkLabel(bar, text="│", font=ctk.CTkFont(size=12),
                     text_color=COLORS["border"]).pack(side="left", padx=4)

        # Zoom controls
        ctk.CTkLabel(bar, text="缩放", font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_muted"]).pack(side="left", padx=(2, 0))
        for t, c in [("⊏⊐", self._zoom_fit), ("＋", self._zoom_in)]:
            ctk.CTkButton(bar, text=t, command=c, fg_color=COLORS["bg_surface"],
                          hover_color=COLORS["bg_hover"], text_color=COLORS["text_primary"],
                          **ZS).pack(side="left", padx=1)
        self._zoom_entry = ctk.CTkEntry(bar, width=42, height=20,
                                         font=ctk.CTkFont(size=9),
                                         fg_color=COLORS["canvas_bg"],
                                         text_color=COLORS["text_primary"],
                                         border_width=0, corner_radius=3,
                                         justify="center")
        self._zoom_entry.insert(0, "100")
        self._zoom_entry.pack(side="left", padx=1)
        self._zoom_entry.bind("<Return>", self._zoom_apply)
        self._zoom_entry.bind("<FocusOut>", self._zoom_apply)
        ctk.CTkLabel(bar, text="%", font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_muted"]).pack(side="left")
        ctk.CTkButton(bar, text="－", command=self._zoom_out,
                      fg_color=COLORS["bg_surface"], hover_color=COLORS["bg_hover"],
                      text_color=COLORS["text_primary"], **ZS).pack(side="left", padx=1)
        ctk.CTkButton(bar, text="⟲", command=self._zoom_reset,
                      fg_color=COLORS["bg_surface"], hover_color=COLORS["bg_hover"],
                      text_color=COLORS["text_primary"], **ZS).pack(side="left", padx=1)

        ctk.CTkLabel(bar, text="│", font=ctk.CTkFont(size=12),
                     text_color=COLORS["border"]).pack(side="left", padx=4)

        # Mode
        ctk.CTkLabel(bar, text="模式", font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_muted"]).pack(side="left")
        self.mode_var = ctk.StringVar(value="auto")
        for t, v in [("自动", "auto"), ("手动", "manual")]:
            ctk.CTkRadioButton(bar, text=t, variable=self.mode_var, value=v,
                               command=self.on_mode_change, fg_color=COLORS["accent"],
                               text_color=COLORS["text_primary"],
                               font=ctk.CTkFont(size=9)).pack(side="left", padx=(2, 0))

        ctk.CTkLabel(bar, text="算法", font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_muted"]).pack(side="left", padx=(6, 0))
        self.method_var = ctk.StringVar(value="Taubin")
        ctk.CTkOptionMenu(bar, variable=self.method_var, values=["Taubin", "L-S"],
                          command=self.on_method_change,
                          fg_color=COLORS["bg_surface"], button_color=COLORS["accent"],
                          button_hover_color=COLORS["accent_hover"],
                          text_color=COLORS["text_primary"],
                          dropdown_fg_color=COLORS["bg_surface"],
                          dropdown_hover_color=COLORS["bg_hover"],
                          dropdown_text_color=COLORS["text_primary"],
                          width=60, height=20, corner_radius=4).pack(side="left", padx=(2, 0))

        ctk.CTkLabel(bar, text="│", font=ctk.CTkFont(size=12),
                     text_color=COLORS["border"]).pack(side="left", padx=4)

        # View selector
        ctk.CTkLabel(bar, text="视图", font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_muted"]).pack(side="left")
        self._profile_view = ctk.StringVar(value="full")
        ctk.CTkOptionMenu(bar, variable=self._profile_view,
                          values=["三面板", "极坐标轮廓", "极坐标偏差", "线性偏差"],
                          command=lambda v: self._on_view_change(v),
                          fg_color=COLORS["bg_surface"], button_color=COLORS["accent"],
                          button_hover_color=COLORS["accent_hover"],
                          text_color=COLORS["text_primary"],
                          dropdown_fg_color=COLORS["bg_surface"],
                          dropdown_hover_color=COLORS["bg_hover"],
                          dropdown_text_color=COLORS["text_primary"],
                          width=80, height=20, corner_radius=4).pack(side="left", padx=(2, 0))

        # ── Row 2: Canvas + right panel ──
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Canvas
        canvas_container = ctk.CTkFrame(self, fg_color=COLORS["canvas_bg"])
        canvas_container.grid(row=2, column=0, sticky="nsew")
        canvas_container.grid_columnconfigure(0, weight=1)
        canvas_container.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_container, bg=COLORS["canvas_bg"],
                                highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Right panel (overlay on canvas)
        self._right_panel = ctk.CTkFrame(canvas_container, width=180,
                                          fg_color=COLORS["bg_card"], corner_radius=0)
        self._right_panel.grid(row=0, column=0, sticky="ne")
        self._right_panel.grid_propagate(False)
        self._build_right_panel(self._right_panel)

        # Canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # ── Row 3: Status bar ──
        sb = ctk.CTkFrame(self, height=22, corner_radius=0, fg_color=COLORS["bg_dark"])
        sb.grid(row=3, column=0, sticky="ew")
        sb.grid_propagate(False)
        self.status_label = ctk.CTkLabel(sb, text="就绪 · Ready",
                                          font=ctk.CTkFont(size=9),
                                          text_color=COLORS["text_secondary"])
        self.status_label.pack(side="left", padx=10, pady=2)
        self.coord_label = ctk.CTkLabel(sb, text="", font=ctk.CTkFont(size=9),
                                         text_color=COLORS["text_muted"])
        self.coord_label.pack(side="right", padx=10, pady=2)

    def _build_right_panel(self, parent):
        P, S, F = 10, 3, 9

        # Display
        ctk.CTkLabel(parent, text="显示", font=ctk.CTkFont(size=F, weight="bold"),
                     text_color=COLORS["text_muted"]).pack(anchor="w", padx=P, pady=(6, 2))
        self.var_center = ctk.BooleanVar(value=True)
        self.var_contours = ctk.BooleanVar(value=True)
        self.var_errors = ctk.BooleanVar(value=True)
        self.var_edges = ctk.BooleanVar(value=False)
        self.var_label_idx = ctk.BooleanVar(value=True)
        self.var_label_xy = ctk.BooleanVar(value=False)
        self.var_label_dist = ctk.BooleanVar(value=True)
        self.var_label_dev = ctk.BooleanVar(value=True)
        for t, v in [("圆心", self.var_center), ("轮廓", self.var_contours),
                     ("偏差", self.var_errors), ("边缘", self.var_edges)]:
            ctk.CTkCheckBox(parent, text=t, variable=v, command=self.redraw,
                            fg_color=COLORS["accent"], text_color=COLORS["text_primary"],
                            font=ctk.CTkFont(size=F)).pack(anchor="w", padx=P)
        ctk.CTkLabel(parent, text="标注:", font=ctk.CTkFont(size=F, weight="bold"),
                     text_color=COLORS["text_muted"]).pack(anchor="w", padx=P, pady=(2, 0))
        for t, v in [("点号", self.var_label_idx), ("坐标(XY)", self.var_label_xy),
                     ("径向距", self.var_label_dist), ("偏差值", self.var_label_dev)]:
            ctk.CTkCheckBox(parent, text=t, variable=v, command=self.redraw,
                            fg_color=COLORS["accent"], text_color=COLORS["text_primary"],
                            font=ctk.CTkFont(size=F)).pack(anchor="w", padx=P)

        ctk.CTkFrame(parent, height=1, fg_color=COLORS["border"]).pack(fill="x", padx=P, pady=S)

        # Results
        ctk.CTkLabel(parent, text="结果", font=ctk.CTkFont(size=F, weight="bold"),
                     text_color=COLORS["text_muted"]).pack(anchor="w", padx=P, pady=(2, 2))
        rf = ctk.CTkFrame(parent, fg_color=COLORS["bg_surface"], corner_radius=4)
        rf.pack(fill="x", padx=P, pady=(0, 4))
        self.result_widgets = {}
        for k, l in [("n_points", "点数"), ("cx", "X"), ("cy", "Y"),
                      ("radius", "半径"), ("roundness", "圆度"), ("rmse", "RMSE")]:
            r = ctk.CTkFrame(rf, fg_color="transparent")
            r.pack(fill="x", padx=6, pady=0)
            ctk.CTkLabel(r, text=l, font=ctk.CTkFont(size=F),
                         text_color=COLORS["text_secondary"]).pack(side="left")
            v = ctk.CTkLabel(r, text="—", font=ctk.CTkFont(size=F, weight="bold"),
                             text_color=COLORS["text_primary"])
            v.pack(side="right")
            self.result_widgets[k] = v

    def set_status(self, text):
        self.status_label.configure(text=text)

    # ── Event Handlers ────────────────────────────────────────────────────

    def on_mode_change(self):
        self.mode = self.mode_var.get()
        if self.mode == "manual":
            self.canvas.configure(cursor="crosshair")
            self.set_status("手动模式：点击图像添加测量点  |  Manual: click to add points")
        else:
            self.canvas.configure(cursor="crosshair")
            self.set_status("自动模式：加载图像后自动检测边缘  |  Auto: load image to detect edges")

    def on_method_change(self, choice):
        self.selected_method = choice
        if self.fit_result is not None and len(self.points) >= 3:
            self.fit_circle()

    def on_canvas_configure(self, event):
        self.redraw()

    def on_mouse_move(self, event):
        if self.original_image is None:
            return
        img_x, img_y = self.canvas_to_image(event.x, event.y)
        if 0 <= img_x < self.original_image.shape[1] and 0 <= img_y < self.original_image.shape[0]:
            self.coord_label.configure(
                text=f"坐标  Coord: ({int(img_x)}, {int(img_y)})"
            )

    def on_canvas_press(self, event):
        self._drag_start = (event.x, event.y)
        self._moved = False

    def on_canvas_drag(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._moved = self._moved or abs(dx) > 2 or abs(dy) > 2
        self.pan_offset[0] += dx
        self.pan_offset[1] += dy
        self._drag_start = (event.x, event.y)
        if self._moved:
            self.redraw()

    def on_canvas_release(self, event):
        if self._drag_start is None:
            return
        if not self._moved and self.mode == "manual" and self.original_image is not None:
            img_x, img_y = self.canvas_to_image(event.x, event.y)
            h, w = self.original_image.shape[:2]
            if 0 <= img_x < w and 0 <= img_y < h:
                self.points.append((float(img_x), float(img_y)))
                self.redraw()
                self.set_status(f"已添加点 {len(self.points)}  |  Point added ({int(img_x)}, {int(img_y)})")
                if len(self.points) >= 3:
                    self.btn_fit.configure(state="normal")
        self._drag_start = None

    def on_right_click(self, event):
        if self.original_image is None:
            return
        self.pan_offset = [0, 0]
        self.zoom_scale = 1.0
        self.redraw()
        self.set_status("视图重置  |  View reset")

    def _update_zoom_entry(self):
        try:
            self._zoom_entry.delete(0, "end")
            self._zoom_entry.insert(0, f"{self.zoom_scale * 100:.0f}")
        except Exception:
            pass

    def _zoom_fit(self):
        if self.original_image is not None:
            cw = max(self.canvas.winfo_width(), 1)
            ch = max(self.canvas.winfo_height(), 1)
            h, w = self.original_image.shape[:2]
            self.zoom_scale = min(cw / w, ch / h)
            self._update_zoom_entry()
        self.pan_offset = [0, 0]
        self.redraw()

    def _zoom_apply(self, _event=None):
        try:
            val = self._zoom_entry.get().strip().rstrip("%")
            pct = float(val)
            self.zoom_scale = max(0.1, min(20.0, pct / 100.0))
            self._update_zoom_entry()
            self.redraw()
        except (ValueError, AttributeError):
            self._update_zoom_entry()

    def _zoom_in(self):
        self.zoom_scale = min(20.0, self.zoom_scale * 1.25)
        self._update_zoom_entry()
        self.redraw()

    def _zoom_out(self):
        self.zoom_scale = max(0.1, self.zoom_scale / 1.25)
        self._update_zoom_entry()
        self.redraw()

    def _zoom_reset(self):
        self.zoom_scale = 1.0
        self.pan_offset = [0, 0]
        self._update_zoom_entry()
        self.redraw()

    def _on_view_change(self, label):
        view_map = {"三面板": "full", "极坐标轮廓": "profile",
                     "极坐标偏差": "deviation_polar", "线性偏差": "deviation_linear"}
        mapped = view_map.get(label, "full")
        self._profile_view.set(label)
        if self.fit_result is not None:
            self.show_profile()

    def on_mousewheel(self, event):
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_scale *= scale_factor
        self.zoom_scale = max(0.1, min(20.0, self.zoom_scale))
        self._update_zoom_entry()
        self.redraw()

    # ── Coordinate Transformations ───────────────────────────────────────

    def _get_display_scale(self):
        if self.original_image is None:
            return 1.0
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        h, w = self.original_image.shape[:2]
        if w < 1 or h < 1 or cw < 1 or ch < 1:
            return 1.0
        return min(cw / w, ch / h) * self.zoom_scale

    def image_to_canvas(self, x, y):
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        s = self._get_display_scale()
        h, w = self.original_image.shape[:2] if self.original_image is not None else (1, 1)
        return (
            cw / 2 + self.pan_offset[0] + (x - w / 2) * s,
            ch / 2 + self.pan_offset[1] + (y - h / 2) * s,
        )

    def canvas_to_image(self, cx, cy):
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        s = self._get_display_scale()
        h, w = self.original_image.shape[:2] if self.original_image is not None else (1, 1)
        return (
            (cx - cw / 2 - self.pan_offset[0]) / s + w / 2,
            (cy - ch / 2 - self.pan_offset[1]) / s + h / 2,
        )

    # ── Image Loading ────────────────────────────────────────────────────

    def load_image(self, filepath=None):
        if filepath is None:
            filepath = filedialog.askopenfilename(
                title="选择图像  Select Image",
                filetypes=[
                    ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                    ("所有文件", "*.*"),
                ],
            )
        if not filepath:
            return

        img = cv2.imread(filepath)
        if img is None:
            messagebox.showerror("错误", "无法加载图像文件")
            return

        self.original_image = img
        self.points = []
        self.fit_result = None
        self.pan_offset = [0, 0]
        self.btn_fit.configure(state="normal")
        self.btn_export.configure(state="disabled")
        self.btn_profile.configure(state="disabled")
        self.set_status(f"已加载: {os.path.basename(filepath)}  ({img.shape[1]}×{img.shape[0]})")
        self.clear_results()

        if self.mode == "auto":
            self.auto_detect_and_fit()

        self.redraw()
        self.after(100, self._zoom_fit)

    def _apply_coords(self, coords):
        """Apply parsed coordinates and run fit."""
        self.mode_var.set("manual")
        self.mode = "manual"

        if self.original_image is not None:
            img_h, img_w = self.original_image.shape[:2]
            in_bounds = all(0 <= x < img_w and 0 <= y < img_h for x, y in coords)
            bounds_msg = "" if in_bounds else "（部分超出图像边界）"
        else:
            bounds_msg = ""

        self.set_status(f"已应用 {len(coords)} 个坐标点  |  Applied {len(coords)} points {bounds_msg}")
        self.points = coords
        self.fit_result = None
        self.btn_fit.configure(state="normal")
        self.btn_export.configure(state="disabled")
        self.clear_results()
        self.fit_circle()
        self.redraw()

    def _parse_coord_text(self, text):
        """Parse coordinate text into list of (x,y) tuples."""
        lines = text.strip().splitlines()
        coords = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'[\t,; ]+', line)
            nums = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                try:
                    nums.append(float(p))
                except ValueError:
                    nums = []
                    break
            if len(nums) >= 2:
                coords.append((nums[0], nums[1]))
        return coords

    def paste_data(self, _event=None):
        """Open paste dialog, optionally pre-filled with clipboard content."""
        initial = ""
        try:
            initial = self.clipboard_get()
        except tk.TclError:
            pass
        PasteDialog(self, initial_text=initial)


    def auto_detect_and_fit(self):
        if self.original_image is None:
            return
        self.set_status("正在检测边缘...  Detecting edges...")
        self.update()

        # Multi-strategy detection
        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        best_contour = None

        self.edge_cache = None

        # Strategy 1: Canny + contour detection
        for low in [30, 50, 80]:
            edges = cv2.Canny(blurred, low, low * 3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = img.shape[:2]
            max_dim = max(h, w)
            candidates = []
            for cnt in contours:
                if len(cnt) < 20:
                    continue
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                _, radius = cv2.minEnclosingCircle(cnt)
                r_ratio = radius / max_dim
                if circularity > 0.4 and 0.03 < r_ratio < 0.98:
                    candidates.append((cnt, circularity, radius, len(cnt)))
            if candidates:
                self.edge_cache = edges
                candidates.sort(key=lambda x: -x[1] * x[3])
                best_contour = candidates[0][0]
                break

        # Strategy 2: Hough Circles if contour fails
        if best_contour is None:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=50,
                param1=80, param2=30, minRadius=10, maxRadius=max_dim // 2
            )
            if circles is not None:
                circles = np.round(circles[0]).astype(int)
                if len(circles) > 0:
                    cx, cy, r = circles[0]
                    mask = np.zeros_like(gray)
                    cv2.circle(mask, (cx, cy), r, 255, 3)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        best_contour = contours[0]

        if best_contour is not None:
            pts = ImageProcessor.sample_contour_points(best_contour, 200)
            if len(pts) >= 3:
                self.points = [(p[0], p[1]) for p in pts]
                self.set_status(f"检测到 {len(self.points)} 个边缘点  |  Detected {len(self.points)} edge points")
                self.fit_circle()
                return

        self.set_status("未检测到明显的圆形特征，请切换到手动模式  |  No circular feature detected, try manual mode")

    def fit_circle(self):
        if len(self.points) < 3:
            messagebox.showwarning("提示", "至少需要3个测量点  |  Need at least 3 points")
            return

        if self.selected_method == "Taubin":
            result = fit_circle_taubin(self.points)
        else:
            result = fit_circle_least_squares(self.points)

        if result is None:
            messagebox.showerror("错误", "拟合失败  |  Fit failed")
            return

        self.fit_result = result
        self.btn_export.configure(state="normal")
        self.btn_profile.configure(state="normal")
        self.update_results(result)
        self.redraw()
        self.set_status(
            f"拟合完成  |  Fit complete  —  "
            f"圆度 Roundness: {result.roundness:.4f} px  "
            f"|  RMSE: {result.rmse:.4f}"
        )

    def reset_all(self):
        self.points = []
        self.fit_result = None
        self.zoom_scale = 1.0
        self.pan_offset = [0, 0]
        self._close_profile()
        self.btn_fit.configure(state="normal" if self.original_image is not None else "disabled")
        self.btn_export.configure(state="disabled")
        self.btn_profile.configure(state="disabled")
        self.clear_results()
        self.redraw()
        self.set_status("已清除  |  Cleared")

    # ── Results Display ──────────────────────────────────────────────────

    def update_results(self, result):
        d = result.to_dict()
        self.result_widgets["n_points"].configure(text=str(d["n_points"]))
        self.result_widgets["cx"].configure(text=f"{d['cx']:.4f}")
        self.result_widgets["cy"].configure(text=f"{d['cy']:.4f}")
        self.result_widgets["radius"].configure(text=f"{d['radius']:.4f}")
        self.result_widgets["roundness"].configure(text=f"{d['roundness']:.4f}")

        rmse_val = d["rmse"]
        rmse_color = COLORS["success"] if rmse_val < 0.1 else (COLORS["warning"] if rmse_val < 0.5 else COLORS["error"])
        self.result_widgets["rmse"].configure(text=f"{rmse_val:.6f}", text_color=rmse_color)

    def _show_point_popup(self):
        if self.fit_result is None:
            return
        r = self.fit_result
        dx = r.points[:, 0] - r.cx
        dy = r.points[:, 1] - r.cy
        dist = np.sqrt(dx * dx + dy * dy)

        win = ctk.CTkToplevel(self)
        win.title("点数据  Point Data")
        win.geometry("440x400")
        win.minsize(300, 200)
        win.transient(self)
        win.grab_set()

        frame = ctk.CTkScrollableFrame(win, fg_color=COLORS["bg_card"], corner_radius=0)
        frame.pack(fill="both", expand=True, padx=0, pady=0)

        hdr = ctk.CTkFrame(frame, fg_color="transparent")
        hdr.pack(fill="x", padx=8, pady=(6, 2))
        for txt, w in [("#", 30), ("X", 60), ("Y", 60), ("R-距", 65), ("偏差", 65)]:
            ctk.CTkLabel(hdr, text=txt, font=ctk.CTkFont(size=10, weight="bold"),
                         text_color=COLORS["text_muted"], width=w).pack(side="left")

        for i, pt in enumerate(r.points):
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=1)
            vals = [str(i + 1), f"{pt[0]:.3f}", f"{pt[1]:.3f}",
                    f"{dist[i]:.3f}", f"{r.errors[i]:+.3f}"]
            for j, v in enumerate(vals):
                clr = COLORS["text_secondary"]
                if j == 4:
                    e = r.errors[i]
                    clr = COLORS["success"] if abs(e) < r.rmse * 1.5 else (
                        COLORS["warning"] if e > 0 else COLORS["error"])
                ctk.CTkLabel(row, text=v, font=ctk.CTkFont(size=10, family="Consolas"),
                             text_color=clr, width=[30, 60, 60, 65, 65][j]).pack(side="left")

        ctk.CTkButton(win, text="关闭  Close", command=win.destroy,
                      fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
                      text_color="#ffffff", height=30, corner_radius=8,
                      ).pack(pady=(0, 8))

    def clear_results(self):
        for key, widget in self.result_widgets.items():
            widget.configure(text="—")
            if key == "rmse":
                widget.configure(text_color=COLORS["text_primary"])

    def export_results(self):
        if not self.fit_result:
            return
        filepath = filedialog.asksaveasfilename(
            title="导出结果  Export Results",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv"), ("所有文件", "*.*")],
        )
        if not filepath:
            return
        data = self.fit_result.to_dict()
        if self.original_image is not None:
            data["image"] = {
                "width": self.original_image.shape[1],
                "height": self.original_image.shape[0],
            }
            import hashlib
            data["image_hash"] = hashlib.md5(self.original_image.tobytes()).hexdigest()

        if filepath.endswith(".csv"):
            import csv
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for k, v in data.items():
                    if isinstance(v, dict):
                        for sk, sv in v.items():
                            writer.writerow([f"{k}_{sk}", sv])
                    else:
                        writer.writerow([k, v])
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        self.set_status(f"结果已导出  |  Exported to {os.path.basename(filepath)}")

    def show_profile(self):
        """Toggle roundness profile display on the canvas."""
        if self.showing_profile:
            self._close_profile()
            return
        if self.fit_result is None:
            return
        r = self.fit_result
        vlabel = self._profile_view.get()
        vmap = {"三面板": "full", "极坐标轮廓": "profile",
                "极坐标偏差": "deviation_polar", "线性偏差": "deviation_linear"}
        view = vmap.get(vlabel, "full")

        angles = np.arctan2(r.points[:, 1] - r.cy, r.points[:, 0] - r.cx)
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]
        errors_sorted = r.errors[sort_idx]
        angles_loop = np.append(angles_sorted, angles_sorted[0] + 2 * np.pi)
        errors_loop = np.append(errors_sorted, errors_sorted[0])
        max_err = max(abs(r.peak_error), abs(r.valley_error), 1e-6)

        fig = plt.figure(figsize=(7, 6) if view != "full" else (10, 7), facecolor='#1a1a2e')

        if view == "full":
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[3, 1], hspace=0.25, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0], polar=True)
            ax1.set_facecolor('#12121e')
            profile_r = r.radius + errors_sorted
            ax1.plot(angles_loop, np.full_like(angles_loop, r.radius), '--', color='#555577', lw=1, alpha=0.6)
            ax1.plot(angles_sorted, profile_r, 'o-', color='#60a5fa', lw=1.5, ms=3)
            ax1.fill(angles_loop, r.radius + errors_loop, alpha=0.12, color='#60a5fa')
            ax1.plot(angles_loop, np.full_like(angles_loop, r.radius + r.peak_error), ':', color='#f87171', lw=0.6)
            ax1.plot(angles_loop, np.full_like(angles_loop, r.radius + r.valley_error), ':', color='#4ade80', lw=0.6)
            ax1.set_title(f'Profile  [R={r.radius:.2f}]', color='#e8e8f0', fontsize=10, pad=12)
            ax1.tick_params(colors='#9898b8', labelsize=7)
            ax1.grid(True, alpha=0.15, color='#555577')

            ax2 = fig.add_subplot(gs[0, 1], polar=True)
            ax2.set_facecolor('#12121e')
            rlim = max_err * 1.4
            offset = rlim
            dev_shift = errors_sorted + offset
            ax2.fill_between(angles_sorted, offset, dev_shift, alpha=0.3,
                             where=(errors_sorted >= 0), color='#f87171')
            ax2.fill_between(angles_sorted, offset, dev_shift, alpha=0.3,
                             where=(errors_sorted < 0), color='#4ade80')
            ax2.plot(angles_sorted, dev_shift, '-', color='#93c5fd', lw=1.5)
            ax2.plot(angles_sorted, dev_shift, '.', color='#e0e7ff', ms=3)
            ax2.plot(angles_loop, np.full_like(angles_loop, offset), '-',
                     color='#555577', lw=1, alpha=0.5)
            ax2.plot(angles_loop, np.full_like(angles_loop, r.peak_error + offset),
                     ':', color='#f87171', lw=0.6)
            ax2.plot(angles_loop, np.full_like(angles_loop, r.valley_error + offset),
                     ':', color='#4ade80', lw=0.6)
            ax2.set_ylim(0, 2 * offset)
            tick_vals = np.linspace(0, 2 * offset, 5)
            ax2.set_yticks(tick_vals)
            ax2.set_yticklabels([f'{v - offset:.2f}' for v in tick_vals], color='#9898b8', fontsize=6)
            ax2.set_title(f'Deviation (polar)  [±{rlim:.4f}]', color='#e8e8f0', fontsize=10, pad=12)
            ax2.tick_params(colors='#9898b8', labelsize=7)
            ax2.grid(True, alpha=0.15, color='#555577')

            ax3 = fig.add_subplot(gs[1, :])
            ax3.set_facecolor('#12121e')
            ax3.axhline(0, color='#555577', lw=1, ls='--', alpha=0.5)
            ax3.fill_between(angles_sorted, errors_sorted, 0, alpha=0.3, where=(errors_sorted >= 0), color='#f87171')
            ax3.fill_between(angles_sorted, errors_sorted, 0, alpha=0.3, where=(errors_sorted < 0), color='#4ade80')
            ax3.plot(angles_sorted, errors_sorted, '-', color='#60a5fa', lw=1.5)
            ax3.plot(angles_sorted, errors_sorted, '.', color='#93c5fd', ms=4)
            ax3.axhline(r.peak_error, color='#f87171', lw=0.6, ls=':')
            ax3.axhline(r.valley_error, color='#4ade80', lw=0.6, ls=':')
            ax3.set_xlim(-0.1, 2 * np.pi + 0.1)
            ax3.set_title('Deviation (linear)', color='#e8e8f0', fontsize=11)
            ax3.set_xlabel('Angle (rad)', color='#9898b8', fontsize=8)
            ax3.set_ylabel('Error', color='#9898b8', fontsize=8)
            ax3.tick_params(colors='#9898b8', labelsize=7)
            ax3.grid(True, alpha=0.15, color='#555577')

        elif view == "profile":
            ax = fig.add_subplot(111, polar=True)
            ax.set_facecolor('#12121e')
            profile_r = r.radius + errors_sorted
            ax.plot(angles_loop, np.full_like(angles_loop, r.radius), '--', color='#555577', lw=1, alpha=0.6)
            ax.plot(angles_sorted, profile_r, 'o-', color='#60a5fa', lw=2, ms=4)
            ax.fill(angles_loop, r.radius + errors_loop, alpha=0.15, color='#60a5fa')
            ax.plot(angles_loop, np.full_like(angles_loop, r.radius + r.peak_error), ':', color='#f87171', lw=0.8)
            ax.plot(angles_loop, np.full_like(angles_loop, r.radius + r.valley_error), ':', color='#4ade80', lw=0.8)
            ax.set_title('Roundness Profile', color='#e8e8f0', fontsize=13, pad=16)
            ax.tick_params(colors='#9898b8', labelsize=8)
            ax.grid(True, alpha=0.15, color='#555577')

        elif view == "deviation_polar":
            rlim = max_err * 1.4
            offset = rlim
            dev_shift = errors_sorted + offset

            ax = fig.add_subplot(111, polar=True)
            ax.set_facecolor('#12121e')
            ax.fill_between(angles_sorted, offset, dev_shift, alpha=0.3,
                            where=(errors_sorted >= 0), color='#f87171')
            ax.fill_between(angles_sorted, offset, dev_shift, alpha=0.3,
                            where=(errors_sorted < 0), color='#4ade80')
            ax.plot(angles_sorted, dev_shift, '-', color='#93c5fd', lw=2)
            ax.plot(angles_sorted, dev_shift, '.', color='#e0e7ff', ms=4)
            ax.plot(angles_loop, np.full_like(angles_loop, offset), '-',
                    color='#555577', lw=1, alpha=0.5)
            ax.plot(angles_loop, np.full_like(angles_loop, r.peak_error + offset),
                    ':', color='#f87171', lw=0.8)
            ax.plot(angles_loop, np.full_like(angles_loop, r.valley_error + offset),
                    ':', color='#4ade80', lw=0.8)
            ax.set_ylim(0, 2 * offset)
            ax.set_yticks(np.linspace(0, 2 * offset, 5))
            ax.set_yticklabels([f'{v - offset:.2f}' for v in np.linspace(0, 2 * offset, 5)],
                               color='#9898b8', fontsize=7)
            ax.set_title(f'Deviation (polar)  [±{rlim:.4f}]', color='#e8e8f0', fontsize=12, pad=16)
            ax.tick_params(colors='#9898b8', labelsize=8)
            ax.grid(True, alpha=0.2, color='#555577')

        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor('#12121e')
            ax.axhline(0, color='#555577', lw=1, ls='--', alpha=0.5)
            ax.fill_between(angles_sorted, errors_sorted, 0, alpha=0.3,
                            where=(errors_sorted >= 0), color='#f87171')
            ax.fill_between(angles_sorted, errors_sorted, 0, alpha=0.3,
                            where=(errors_sorted < 0), color='#4ade80')
            ax.plot(angles_sorted, errors_sorted, '-', color='#60a5fa', lw=2)
            ax.plot(angles_sorted, errors_sorted, '.', color='#93c5fd', ms=4)
            ax.axhline(r.peak_error, color='#f87171', lw=0.6, ls=':')
            ax.axhline(r.valley_error, color='#4ade80', lw=0.6, ls=':')
            ax.set_xlim(-0.1, 2 * np.pi + 0.1)
            ax.set_title('Deviation (linear)', color='#e8e8f0', fontsize=13)
            ax.set_xlabel('Angle (rad)', color='#9898b8', fontsize=9)
            ax.set_ylabel('Error', color='#9898b8', fontsize=9)
            ax.tick_params(colors='#9898b8', labelsize=8)
            ax.grid(True, alpha=0.15, color='#555577')

        summary = (
            f"C: ({r.cx:.3f}, {r.cy:.3f})  R: {r.radius:.3f}  "
            f"Peak: +{r.peak_error:.4f}  Valley: {r.valley_error:.4f}  "
            f"Roundness: {r.roundness:.4f}  RMSE: {r.rmse:.6f}"
        )
        fig.text(0.5, 0.005, summary, ha='center', va='bottom', color='#9898b8', fontsize=8)
        plt.tight_layout(rect=[0, 0.04, 1, 1])

        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        self._profile_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2RGB)
        self.showing_profile = True
        plt.close(fig)
        self.set_status(f"轮廓图 [{view}] 已显示  |  Profile displayed")
        self.redraw()

    def _close_profile(self):
        self.showing_profile = False
        self._profile_img = None
        self.redraw()

    # ── Canvas Drawing ───────────────────────────────────────────────────

    def redraw(self, _event=None):
        self.canvas.delete("all")

        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)

        # ── Profile view ──
        if self.showing_profile and self._profile_img is not None:
            ph, pw = self._profile_img.shape[:2]
            s = min(cw / pw, ch / ph)
            nw, nh = int(pw * s), int(ph * s)
            disp = cv2.resize(self._profile_img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            img_pil = Image.fromarray(disp)
            self.display_photo = ImageTk.PhotoImage(img_pil)
            self.canvas.create_image(cw / 2, ch / 2, image=self.display_photo, anchor="center")
            self.canvas.create_text(
                cw / 2, ch - 12,
                text="✕ 关闭轮廓  Close Profile （点击「轮廓图」或「清除」）",
                fill=COLORS["text_muted"], font=("Segoe UI", 11), anchor="center",
            )
            return

        # ── No image, but may have pasted data ──
        if self.original_image is None:
            self.image_size = [cw, ch]
            if self.points and len(self.points) >= 3 and self.fit_result is not None:
                r = self.fit_result
                nz = max(cw, ch)
                padding = 60
                xs = [p[0] for p in r.points]
                ys = [p[1] for p in r.points]
                data_w = max(xs) - min(xs) if xs else 1
                data_h = max(ys) - min(ys) if ys else 1
                data_w = max(data_w, 1)
                data_h = max(data_h, 1)
                margin = 0.15
                draw_scale = min((cw - 2 * padding) / (data_w * (1 + 2 * margin)),
                                 (ch - 2 * padding) / (data_h * (1 + 2 * margin))) * self.zoom_scale
                ox = (min(xs) + max(xs)) / 2 - data_w * (1 + 2 * margin) / 2
                oy = (min(ys) + max(ys)) / 2 - data_h * (1 + 2 * margin) / 2

                # Create blank canvas image
                disp = np.full((ch, cw, 3), 18, dtype=np.uint8)

                # Draw grid
                for gx in range(0, cw, 50):
                    cv2.line(disp, (gx, 0), (gx, ch), (30, 30, 50), 1)
                for gy in range(0, ch, 50):
                    cv2.line(disp, (0, gy), (cw, gy), (30, 30, 50), 1)

                # Draw points
                dxs = r.points[:, 0] - r.cx
                dys = r.points[:, 1] - r.cy
                dxs = r.points[:, 0] - r.cx
                dys = r.points[:, 1] - r.cy
                rad_dist = np.sqrt(dxs * dxs + dys * dys)
                for i, pt in enumerate(r.points):
                    px = int(padding + (pt[0] - ox) * draw_scale)
                    py = int(padding + (pt[1] - oy) * draw_scale)
                    if 0 <= px < cw and 0 <= py < ch:
                        cv2.circle(disp, (px, py), 3, (0, 200, 255), -1)
                        parts = []
                        if self.var_label_idx.get():
                            parts.append(f"P{i+1}")
                        if self.var_label_xy.get():
                            parts.append(f"({pt[0]:.1f},{pt[1]:.1f})")
                        if self.var_label_dist.get():
                            parts.append(f"R{rad_dist[i]:.2f}")
                        if self.var_label_dev.get():
                            parts.append(f"d{r.errors[i]:+.2f}")
                        if parts:
                            cv2.putText(disp, " ".join(parts), (px + 6, py - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 220), 1)

                # Draw fitted circle
                dcx = int(padding + (r.cx - ox) * draw_scale)
                dcy = int(padding + (r.cy - oy) * draw_scale)
                dr = int(r.radius * draw_scale)
                if 0 <= dcx < cw and 0 <= dcy < ch:
                    cv2.circle(disp, (dcx, dcy), dr, (255, 140, 0), 2)
                    cl = 10
                    cv2.line(disp, (dcx - cl, dcy), (dcx + cl, dcy), (0, 255, 255), 2)
                    cv2.line(disp, (dcx, dcy - cl), (dcx, dcy + cl), (0, 255, 255), 2)
                    cv2.circle(disp, (dcx, dcy), 4, (0, 255, 255), -1)

                img_pil = Image.fromarray(disp)
                self.display_photo = ImageTk.PhotoImage(img_pil)
                self.canvas.create_image(0, 0, image=self.display_photo, anchor="nw")
            else:
                self.canvas.create_text(
                    cw / 2, ch / 2,
                    text="📋 点击「粘贴数据」导入坐标点  |  Click Paste Data to import",
                    fill=COLORS["text_muted"],
                    font=("Segoe UI", 14),
                    anchor="center",
                )
            return

        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        self.image_size = [w, h]

        # Scale image to fit canvas
        scale = min(cw / w, ch / h) * self.zoom_scale
        new_w, new_h = int(w * scale), int(h * scale)
        img_display = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Overlay points
        if self.points and self.mode == "manual":
            overlay = img_display.copy()
            for i, pt in enumerate(self.points):
                px, py = int(pt[0] * scale), int(pt[1] * scale)
                cv2.circle(overlay, (px, py), max(3, int(2 * scale / max(w, h) * 100)),
                           (0, 200, 255), -1)
                cv2.circle(overlay, (px, py), max(6, int(5 * scale / max(w, h) * 100)),
                           (255, 255, 255), 1)
                if self.fit_result is not None:
                    parts = []
                    if self.var_label_idx.get():
                        parts.append(f"P{i+1}")
                    if self.var_label_xy.get():
                        parts.append(f"({pt[0]:.1f},{pt[1]:.1f})")
                    if self.var_label_dist.get():
                        dx_pt = pt[0] - self.fit_result.cx
                        dy_pt = pt[1] - self.fit_result.cy
                        parts.append(f"R{math.hypot(dx_pt, dy_pt):.2f}")
                    if self.var_label_dev.get():
                        if self.fit_result is not None and i < len(self.fit_result.errors):
                            parts.append(f"d{self.fit_result.errors[i]:+.2f}")
                    if parts:
                        cv2.putText(overlay, " ".join(parts), (px + 6, py - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 220), 1)
            img_display = overlay

        # Overlay fit result
        if self.fit_result is not None and len(self.fit_result.points) >= 3:
            overlay = img_display.copy()
            r = self.fit_result
            s = scale

            # Transform center and radius to display coordinates (0..new_w, 0..new_h)
            disp_cx = r.cx * s
            disp_cy = r.cy * s
            disp_r = r.radius * s

            if self.mode == "auto" and self.var_contours.get():
                for pt in r.points:
                    px = int(pt[0] * s)
                    py = int(pt[1] * s)
                    cv2.circle(overlay, (px, py), 1, (0, 220, 120), -1)

            cv2.circle(overlay,
                       (int(disp_cx), int(disp_cy)),
                       int(disp_r),
                       (255, 140, 0), max(2, int(2.5 * s)))

            if self.var_center.get():
                cl = max(10, int(12 * s))
                cv2.line(overlay,
                         (int(disp_cx) - cl, int(disp_cy)),
                         (int(disp_cx) + cl, int(disp_cy)),
                         (0, 255, 255), max(1, int(2 * s)))
                cv2.line(overlay,
                         (int(disp_cx), int(disp_cy) - cl),
                         (int(disp_cx), int(disp_cy) + cl),
                         (0, 255, 255), max(1, int(2 * s)))
                cv2.circle(overlay, (int(disp_cx), int(disp_cy)),
                           max(3, int(3 * s)), (0, 255, 255), -1)

            if self.var_errors.get() and self.mode == "manual":
                for pt, err in zip(r.points, r.errors):
                    px = int(pt[0] * s)
                    py = int(pt[1] * s)
                    color = (0, 200, 100) if abs(err) < r.rmse * 1.5 else (0, 100, 255) if err > 0 else (255, 100, 0)
                    dx_pt = pt[0] - r.cx
                    dy_pt = pt[1] - r.cy
                    dist = math.hypot(dx_pt, dy_pt)
                    if dist > 0:
                        proj_x = int((r.cx + dx_pt / dist * r.radius) * s)
                        proj_y = int((r.cy + dy_pt / dist * r.radius) * s)
                        cv2.line(overlay, (px, py), (proj_x, proj_y), color, 1)

            img_display = overlay

        # Edge overlay (outside fit-result block so it works independently)
        if self.var_edges.get() and self.edge_cache is not None:
            eh, ew = self.edge_cache.shape[:2]
            edge_rgb = cv2.cvtColor(self.edge_cache, cv2.COLOR_GRAY2BGR)
            edge_rgb = cv2.resize(edge_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            edge_mask = edge_rgb > 0
            edge_overlay = img_display.copy()
            edge_overlay[edge_mask] = (0, 220, 255)
            cv2.addWeighted(edge_overlay, 0.3, img_display, 0.7, 0, img_display)

        # Convert to PhotoImage
        img_pil = Image.fromarray(img_display)
        self.display_photo = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(cw / 2 + self.pan_offset[0], ch / 2 + self.pan_offset[1],
                                 image=self.display_photo, anchor="center")

        # Update zoom entry
        self._update_zoom_entry()

        # Help text overlay
        help_lines = ["🔄 滚轮缩放 · 拖拽平移 · 右键复位"]
        if self.mode == "manual":
            help_lines.append(f"⏺ 单击加点 (已选 {len(self.points)} 个)")

        if help_lines:
            text = "  |  ".join(help_lines)
            self.canvas.create_text(
                16, ch - 16, text=text, anchor="sw",
                fill=COLORS["text_muted"], font=("Segoe UI", 11),
            )


# ── Entry Point ───────────────────────────────────────────────────────────────



class PasteDialog:
    """Modal dialog for pasting/editing coordinate data."""

    def __init__(self, parent, initial_text=""):
        self.parent = parent
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("粘贴数据 · Paste Data")
        self.dialog.geometry("560x460")
        self.dialog.minsize(400, 320)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        px = parent.winfo_x() + (parent.winfo_width() - 560) // 2
        py = parent.winfo_y() + (parent.winfo_height() - 460) // 2
        self.dialog.geometry(f"+{max(0,px)}+{max(0,py)}")

        self._build_ui(initial_text)
        self.dialog.focus_set()
        self.dialog.wait_window()

    def _build_ui(self, initial_text):
        d = self.dialog

        ctk.CTkLabel(
            d, text="粘贴坐标数据  ·  Paste Coordinate Data",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS["text_primary"],
        ).pack(anchor="w", padx=24, pady=(20, 4))

        ctk.CTkLabel(
            d, text="每行至少两个数值（x y），支持 Tab / 逗号 / 空格 / 分号 分隔",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_secondary"],
        ).pack(anchor="w", padx=24, pady=(0, 12))

        text_frame = ctk.CTkFrame(d, fg_color=COLORS["bg_surface"], corner_radius=8)
        text_frame.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        self.textbox = ctk.CTkTextbox(
            text_frame, font=ctk.CTkFont(size=13, family="Consolas"),
            fg_color=COLORS["canvas_bg"], text_color=COLORS["text_primary"],
            border_width=0, corner_radius=6,
        )
        self.textbox.pack(fill="both", expand=True, padx=8, pady=8)
        if initial_text:
            self.textbox.insert("1.0", initial_text)
            self.textbox.edit_modified(False)

        bottom = ctk.CTkFrame(d, fg_color="transparent")
        bottom.pack(fill="x", padx=20, pady=(0, 16))

        self.info_label = ctk.CTkLabel(
            bottom, text="解析点数: 0",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_secondary"],
        )
        self.info_label.pack(side="left")

        btn_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_frame.pack(side="right")

        ctk.CTkButton(
            btn_frame, text="取消  Cancel",
            command=self.dialog.destroy,
            fg_color=COLORS["bg_surface"], hover_color=COLORS["bg_hover"],
            text_color=COLORS["text_primary"],
            width=90, height=34, corner_radius=8, border_width=0,
        ).pack(side="left", padx=(0, 8))

        self.ok_btn = ctk.CTkButton(
            btn_frame, text="✓  确定  Apply",
            command=self._apply,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            text_color="#ffffff",
            width=100, height=34, corner_radius=8, border_width=0,
            state="disabled",
        )
        self.ok_btn.pack(side="left")

        self.textbox.bind("<KeyRelease>", self._on_text_change)
        self.textbox.bind("<<Modified>>", self._on_text_change)
        self._on_text_change()

        self.dialog.bind("<Control-Return>", lambda e: self._apply())
        self.dialog.bind("<Escape>", lambda e: self.dialog.destroy())

    def _on_text_change(self, _event=None):
        text = self.textbox.get("1.0", "end-1c")
        coords = self.parent._parse_coord_text(text)
        count = len(coords)
        self.info_label.configure(text=f"解析点数: {count}")
        if count >= 3:
            self.ok_btn.configure(state="normal", text=f"✓  确定 ({count}点)")
        else:
            self.ok_btn.configure(state="disabled", text="✓  确定  Apply")

    def _apply(self):
        text = self.textbox.get("1.0", "end-1c")
        coords = self.parent._parse_coord_text(text)
        if len(coords) < 3:
            return
        self.parent._apply_coords(coords)
        self.dialog.destroy()

if __name__ == "__main__":
    app = CircleFitterApp()
    app.mainloop()
