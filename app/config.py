from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CLASS_NAMES = (
    "vehicle_body",
    "vehicle_footprint",
    "front",
    "rear",
    "wheel",
    "mirror",
    "roof_rack",
    "antenna",
)


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path("data/vehicle_perception.sqlite3")
    model_path: Path | None = None
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    model_width: int = 640
    model_height: int = 640
    confidence_threshold: float = 0.35
    max_frame_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 24_000_000
    max_calibration_error_mm: float = 10.0
    max_highlight_fraction: float = 0.12
    min_blur_score: float = 15.0
    min_frame_width: int = 640
    min_frame_height: int = 360
    required_views: tuple[str, ...] = ("front", "rear", "left", "right", "overhead")
    parking_zone_mm: tuple[float, float, float, float] | None = None
    expected_heading_degrees: float | None = None
    heading_tolerance_degrees: float = 10.0

    def __post_init__(self) -> None:
        if self.model_width < 1 or self.model_height < 1:
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence threshold must be between 0 and 1")
        if self.max_frame_bytes < 1 or self.max_image_pixels < 1:
            raise ValueError("frame size limits must be positive")
        if not math.isfinite(self.max_calibration_error_mm) or self.max_calibration_error_mm <= 0:
            raise ValueError("maximum calibration error must be a positive finite number")
        if (
            not math.isfinite(self.max_highlight_fraction)
            or not 0 <= self.max_highlight_fraction <= 1
            or not math.isfinite(self.min_blur_score)
            or self.min_blur_score < 0
        ):
            raise ValueError("image quality thresholds are outside their valid range")
        if self.min_frame_width < 1 or self.min_frame_height < 1:
            raise ValueError("minimum frame dimensions must be positive")
        allowed_views = {"front", "rear", "left", "right", "overhead", "other"}
        if not self.required_views or not set(self.required_views) <= allowed_views:
            raise ValueError("required views must contain supported view names")
        if len(set(self.required_views)) != len(self.required_views):
            raise ValueError("required views must be unique")
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("model class names must be non-empty and unique")
        if self.parking_zone_mm is not None:
            if len(self.parking_zone_mm) != 4 or not all(math.isfinite(value) for value in self.parking_zone_mm):
                raise ValueError("parking zone must contain four finite coordinates")
            x_min, y_min, x_max, y_max = self.parking_zone_mm
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("parking zone minimum coordinates must be below maximum coordinates")
        if self.expected_heading_degrees is not None and not 0 <= self.expected_heading_degrees < 360:
            raise ValueError("expected heading must be in the range 0 to less than 360")
        if not 0 <= self.heading_tolerance_degrees <= 180:
            raise ValueError("heading tolerance must be between 0 and 180 degrees")

    @classmethod
    def from_env(cls) -> "Settings":
        model_value = os.getenv("ONNX_MODEL_PATH", "").strip()
        class_value = os.getenv("MODEL_CLASS_NAMES", ",".join(DEFAULT_CLASS_NAMES))
        required_views = os.getenv("REQUIRED_VIEWS", "front,rear,left,right,overhead")
        zone_value = os.getenv("PARKING_ZONE_MM", "").strip()
        zone = None
        if zone_value:
            values = tuple(float(value.strip()) for value in zone_value.split(","))
            if len(values) != 4:
                raise ValueError("PARKING_ZONE_MM must contain xmin,ymin,xmax,ymax")
            zone = values

        heading_value = os.getenv("EXPECTED_HEADING_DEGREES", "").strip()
        return cls(
            database_path=Path(os.getenv("DATABASE_PATH", "data/vehicle_perception.sqlite3")),
            model_path=Path(model_value) if model_value else None,
            class_names=tuple(name.strip() for name in class_value.split(",") if name.strip()),
            model_width=int(os.getenv("MODEL_WIDTH", "640")),
            model_height=int(os.getenv("MODEL_HEIGHT", "640")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
            max_frame_bytes=int(os.getenv("MAX_FRAME_BYTES", str(15 * 1024 * 1024))),
            max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "24000000")),
            max_calibration_error_mm=float(os.getenv("MAX_CALIBRATION_ERROR_MM", "10")),
            max_highlight_fraction=float(os.getenv("MAX_HIGHLIGHT_FRACTION", "0.12")),
            min_blur_score=float(os.getenv("MIN_BLUR_SCORE", "15")),
            min_frame_width=int(os.getenv("MIN_FRAME_WIDTH", "640")),
            min_frame_height=int(os.getenv("MIN_FRAME_HEIGHT", "360")),
            required_views=tuple(view.strip() for view in required_views.split(",") if view.strip()),
            parking_zone_mm=zone,
            expected_heading_degrees=float(heading_value) if heading_value else None,
            heading_tolerance_degrees=float(os.getenv("HEADING_TOLERANCE_DEGREES", "10")),
        )
