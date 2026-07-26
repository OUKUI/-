from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime


class FeatureType(Enum):
    CAVITY = "型腔"
    CORE = "型芯"


@dataclass
class MeasurePoint:
    x: float
    y: float


@dataclass
class FittedCircle:
    cx: float
    cy: float
    radius: float
    rmse: float


@dataclass
class PolarPoint:
    theta: float          # angle in radians
    theta_deg: float      # angle in degrees
    r: float              # actual radius
    delta_r: float        # deviation from fitted circle
    x: float              # original x
    y: float              # original y
    advice: str = ""      # mold repair advice


@dataclass
class AnalysisResult:
    points: list
    fitted: FittedCircle
    polar_points: list
    roundness: float
    peak_error: float
    valley_error: float
    feature_type: FeatureType
    created_at: str = ""
    unit: str = "mm"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self):
        return {
            "cx": round(self.fitted.cx, 4),
            "cy": round(self.fitted.cy, 4),
            "radius": round(self.fitted.radius, 4),
            "roundness": round(self.roundness, 4),
            "rmse": round(self.fitted.rmse, 6),
            "peak_error": round(self.peak_error, 4),
            "valley_error": round(self.valley_error, 4),
            "feature_type": self.feature_type.value,
            "n_points": len(self.points),
            "unit": self.unit,
            "created_at": self.created_at,
        }
