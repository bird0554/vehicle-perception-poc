from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from app.config import Settings
from app.detector import OnnxYoloDetector, UnavailableDetector, build_detector


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def run(self, output_names: object, inputs: object) -> list[np.ndarray]:
        return [self.output]


def test_detector_decodes_declared_xyxy_output_contract() -> None:
    settings = Settings(
        database_path=Path("unused.sqlite3"),
        class_names=("vehicle_body", "wheel"),
        model_width=320,
        model_height=320,
        confidence_threshold=0.4,
    )
    detector = object.__new__(OnnxYoloDetector)
    detector._settings = settings
    detector._session = FakeSession(
        np.asarray([[[10, 20, 100, 120, 0.9, 1], [0, 0, 20, 20, 0.2, 0], [1, 1, 5, 5, 0.8, 9]]], dtype=np.float32)
    )
    detector._input_name = "images"

    result = detector.detect(np.zeros((160, 640, 3), dtype=np.uint8))

    assert len(result) == 1
    assert result[0].class_name == "wheel"
    assert result[0].bbox_px == pytest.approx((20, 10, 200, 60))


def test_detector_rejects_unsupported_output_shape() -> None:
    settings = Settings(
        database_path=Path("unused.sqlite3"),
        model_width=32,
        model_height=32,
    )
    detector = object.__new__(OnnxYoloDetector)
    detector._settings = settings
    detector._session = FakeSession(np.zeros((1, 84, 8400), dtype=np.float32))
    detector._input_name = "images"

    with pytest.raises(ValueError, match="up to 128 outline points"):
        detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))


def test_detector_decodes_polygon_points_when_present() -> None:
    settings = Settings(
        database_path=Path("unused.sqlite3"),
        class_names=("vehicle_footprint",),
        model_width=32,
        model_height=32,
    )
    detector = object.__new__(OnnxYoloDetector)
    detector._settings = settings
    detector._session = FakeSession(
        np.asarray([[[4, 4, 28, 28, 0.9, 0, 4, 4, 28, 4, 28, 28, 4, 28]]], dtype=np.float32)
    )
    detector._input_name = "images"

    result = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert result[0].outline_px == ((4.0, 4.0), (28.0, 4.0), (28.0, 28.0), (4.0, 28.0))


def test_build_detector_is_explicit_when_model_is_missing() -> None:
    detector = build_detector(Settings(database_path=Path("unused.sqlite3")))
    assert isinstance(detector, UnavailableDetector)
    assert detector.reason == "model_not_configured"
