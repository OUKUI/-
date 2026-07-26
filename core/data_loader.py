import re
import pandas as pd
import numpy as np
from models.data_models import MeasurePoint


def parse_clipboard_text(text):
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
            coords.append(MeasurePoint(nums[0], nums[1]))
    return coords


def load_csv(filepath):
    df = pd.read_csv(filepath)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise ValueError("CSV must have at least 2 numeric columns")
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    points = [MeasurePoint(row[x_col], row[y_col]) for _, row in df.iterrows()]
    return points


def validate_points(points, min_count=3):
    if len(points) < min_count:
        return False, f"Need at least {min_count} points, got {len(points)}"
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    if max(xs) - min(xs) < 1e-10 and max(ys) - min(ys) < 1e-10:
        return False, "All points are identical"
    return True, "OK"
