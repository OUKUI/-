import numpy as np
from models.data_models import FittedCircle, PolarPoint, AnalysisResult, FeatureType


def analyze(points, fitted, feature_type=FeatureType.CAVITY, unit="mm"):
    polar = []
    for p in points:
        dx = p.x - fitted.cx
        dy = p.y - fitted.cy
        theta = np.arctan2(dy, dx)
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

    errors = np.array([pp.delta_r for pp in polar])
    roundness = float(np.max(errors) - np.min(errors))
    peak_error = float(np.max(errors))
    valley_error = float(np.min(errors))

    return AnalysisResult(
        points=points,
        fitted=fitted,
        polar_points=polar,
        roundness=roundness,
        peak_error=peak_error,
        valley_error=valley_error,
        feature_type=feature_type,
        unit=unit,
    )
