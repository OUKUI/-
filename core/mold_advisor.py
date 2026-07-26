from models.data_models import FeatureType, PolarPoint


def get_advice(delta_r, feature_type):
    if feature_type == FeatureType.CAVITY:
        if delta_r > 0:
            return "该方向需缩小型腔"
        else:
            return "该方向需放大型腔"
    else:
        if delta_r > 0:
            return "该方向需放大型芯"
        else:
            return "该方向需缩小型芯"


def apply_advice(polar_points, feature_type):
    for pp in polar_points:
        pp.advice = get_advice(pp.delta_r, feature_type)
    return polar_points


def get_advice_table(polar_points):
    import numpy as np
    n = len(polar_points)
    if n == 0:
        return []
    thetas = np.array([pp.theta for pp in polar_points])
    thetas_deg = np.degrees(thetas) % 360
    closest_0 = int(np.argmin(np.minimum(thetas_deg, 360 - thetas_deg)))
    rows = []
    for k in range(n):
        idx = (closest_0 + k) % n
        pp = polar_points[idx]
        rows.append({
            "no": k + 1,
            "angle": f"{pp.theta_deg:.2f}",
            "delta_r": f"{pp.delta_r:+.4f}",
            "x": f"{pp.x:.4f}",
            "y": f"{pp.y:.4f}",
            "advice": pp.advice,
        })
    return rows
