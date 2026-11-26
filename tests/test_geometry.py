from __future__ import annotations

import pytest

from app.geometry import fit_ground_homography, heading_degrees, map_image_point, polygon_dimensions_mm


def test_homography_maps_pixels_to_ground_millimeters() -> None:
    image_points = [(0, 0), (640, 0), (640, 360), (0, 360)]
    world_points = [(0, 0), (32000, 0), (32000, 18000), (0, 18000)]
    homography, error = fit_ground_homography(image_points, world_points)

    assert error == pytest.approx(0.0, abs=1e-6)
    assert map_image_point((320, 180), homography) == pytest.approx((16000, 9000))


def test_homography_rejects_repeated_points() -> None:
    points = [(0, 0), (1, 0), (1, 0), (0, 1)]
    with pytest.raises(ValueError, match="distinct"):
        fit_ground_homography(points, points)


def test_homography_rejects_collinear_points() -> None:
    points = [(0, 0), (1, 1), (2, 2), (3, 3)]
    with pytest.raises(ValueError):
        fit_ground_homography(points, points)


def test_heading_and_polygon_dimensions() -> None:
    assert heading_degrees((0, 0), (0, 1)) == pytest.approx(90.0)
    assert polygon_dimensions_mm([(0, 0), (10, 0), (10, 20), (0, 20)]) == {
        "extent_x_mm": 10.0,
        "extent_y_mm": 20.0,
    }


def test_polygon_dimensions_report_world_axis_extents_for_rotated_outline() -> None:
    assert polygon_dimensions_mm([(0, 0), (10, 10), (0, 20), (-10, 10)]) == {
        "extent_x_mm": 20.0,
        "extent_y_mm": 20.0,
    }
