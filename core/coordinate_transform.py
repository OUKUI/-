import numpy as np
from models.data_models import MeasurePoint, FittedCircle, PolarPoint


def to_polar(points, fitted, theta_offset=0.0):
    polar = []
    for p in points:
        dx = p.x - fitted.cx
        dy = p.y - fitted.cy
        theta = np.arctan2(dy, dx) + theta_offset
        r = np.sqrt(dx * dx + dy * dy)
        delta_r = r - fitted.radius
        polar.append(PolarPoint(
            theta=theta,
            theta_deg=np.degrees(theta) % 360,
            r=r,
            delta_r=delta_r,
            x=p.x,
            y=p.y,
        ))
    polar.sort(key=lambda pp: pp.theta)
    return polar


def sort_by_angle(polar_points):
    return sorted(polar_points, key=lambda p: p.theta)
