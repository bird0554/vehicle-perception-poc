from __future__ import annotations

import math

import cv2
import numpy as np


def fit_ground_homography(
    image_points_px: list[tuple[float, float]],
    world_points_mm: list[tuple[float, float]],
) -> tuple[list[list[float]], float]:
    if len(image_points_px) != len(world_points_mm) or len(image_points_px) < 4:
        raise ValueError("Calibration requires at least four paired points")

    image_points = np.asarray(image_points_px, dtype=np.float64)
    world_points = np.asarray(world_points_mm, dtype=np.float64)
    if not np.isfinite(image_points).all() or not np.isfinite(world_points).all():
        raise ValueError("Calibration points must be finite numbers")
    if len(np.unique(image_points, axis=0)) < 4 or len(np.unique(world_points, axis=0)) < 4:
        raise ValueError("Calibration points must be distinct")

    matrix, _ = cv2.findHomography(image_points, world_points, method=0)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Calibration points do not define a usable homography")
    if np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Calibration points are degenerate")

    projected = cv2.perspectiveTransform(image_points.reshape(1, -1, 2), matrix)[0]
    distances = np.linalg.norm(projected - world_points, axis=1)
    median_error = float(np.median(distances))
    return matrix.tolist(), median_error


def map_image_point(point_px: tuple[float, float], homography: list[list[float]]) -> tuple[float, float]:
    matrix = np.asarray(homography, dtype=np.float64)
    point = np.asarray(point_px, dtype=np.float64).reshape(1, 1, 2)
    mapped = cv2.perspectiveTransform(point, matrix)[0, 0]
    if not np.isfinite(mapped).all():
        raise ValueError("Point cannot be mapped using this calibration")
    return float(mapped[0]), float(mapped[1])


def map_image_polygon(
    points_px: list[tuple[float, float]], homography: list[list[float]]
) -> list[tuple[float, float]]:
    matrix = np.asarray(homography, dtype=np.float64)
    points = np.asarray(points_px, dtype=np.float64).reshape(1, -1, 2)
    mapped = cv2.perspectiveTransform(points, matrix)[0]
    if not np.isfinite(mapped).all():
        raise ValueError("Polygon cannot be mapped using this calibration")
    return [(float(point[0]), float(point[1])) for point in mapped]


def polygon_dimensions_mm(polygon: list[tuple[float, float]]) -> dict[str, float]:
    if len(polygon) < 3:
        raise ValueError("A projected outline requires at least three points")
    x_coordinates = [point[0] for point in polygon]
    y_coordinates = [point[1] for point in polygon]
    return {
        "extent_x_mm": round(max(x_coordinates) - min(x_coordinates), 1),
        "extent_y_mm": round(max(y_coordinates) - min(y_coordinates), 1),
    }


def polygon_centroid(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    if len(polygon) < 3:
        raise ValueError("A polygon centroid requires at least three points")
    twice_area = 0.0
    x_sum = 0.0
    y_sum = 0.0
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        x_sum += (x1 + x2) * cross
        y_sum += (y1 + y2) * cross
    if abs(twice_area) < 1e-9:
        raise ValueError("Polygon area is too small to estimate a centroid")
    return x_sum / (3.0 * twice_area), y_sum / (3.0 * twice_area)


def heading_degrees(rear_mm: tuple[float, float], front_mm: tuple[float, float]) -> float:
    dx = front_mm[0] - rear_mm[0]
    dy = front_mm[1] - rear_mm[1]
    if math.hypot(dx, dy) < 1e-6:
        raise ValueError("Front and rear points are too close to estimate heading")
    return math.degrees(math.atan2(dy, dx)) % 360
