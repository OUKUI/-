import numpy as np
from datetime import datetime

class CircleFitResult:
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
            "cx": round(self.cx, 4), "cy": round(self.cy, 4),
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
    result = fit_circle_least_squares(points)
    if result is not None:
        result.method = "taubin"
    return result
