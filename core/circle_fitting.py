import numpy as np
from models.data_models import MeasurePoint, FittedCircle


def fit_least_squares(points):
    pts = np.array([[p.x, p.y] for p in points], dtype=np.float64)
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
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((distances - radius) ** 2)))
    return FittedCircle(cx=float(cx), cy=float(cy), radius=float(radius), rmse=rmse)
