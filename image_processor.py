import cv2
import numpy as np
import math

class ImageProcessor:
    @staticmethod
    def detect_edges_opencv(image, low_thresh=50, high_thresh=150):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, low_thresh, high_thresh)

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
