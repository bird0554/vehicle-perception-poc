from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.config import Settings


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_px: tuple[float, float, float, float]
    outline_px: tuple[tuple[float, float], ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Detector(Protocol):
    @property
    def available(self) -> bool: ...

    @property
    def reason(self) -> str | None: ...

    def detect(self, image_bgr: np.ndarray) -> list[Detection]: ...


class UnavailableDetector:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def available(self) -> bool:
        return False

    @property
    def reason(self) -> str:
        return self._reason

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        return []


class OnnxYoloDetector:
    """Loads ONNX models that emit box rows with optional polygon vertices."""

    def __init__(self, settings: Settings) -> None:
        if settings.model_path is None or not settings.model_path.is_file():
            raise FileNotFoundError("ONNX model file is missing")
        import onnxruntime as ort

        self._settings = settings
        self._session = ort.InferenceSession(
            str(Path(settings.model_path)), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def available(self) -> bool:
        return True

    @property
    def reason(self) -> None:
        return None

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        source_height, source_width = image_bgr.shape[:2]
        resized = cv2.resize(
            image_bgr,
            (self._settings.model_width, self._settings.model_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        output = self._session.run(None, {self._input_name: tensor})[0]
        rows = np.asarray(output, dtype=np.float32)
        if rows.ndim == 3 and rows.shape[0] == 1:
            rows = rows[0]
        extra_values = rows.shape[1] - 6 if rows.ndim == 2 else -1
        outline_points = extra_values // 2
        if (
            rows.ndim != 2
            or extra_values < 0
            or extra_values % 2
            or (extra_values and not 4 <= outline_points <= 128)
        ):
            raise ValueError("ONNX output must have six fields and up to 128 outline points")

        scale_x = source_width / self._settings.model_width
        scale_y = source_height / self._settings.model_height
        detections: list[Detection] = []
        for row in rows:
            if not np.isfinite(row).all():
                continue
            x1, y1, x2, y2, confidence, class_id_value = map(float, row[:6])
            if not class_id_value.is_integer() or confidence > 1.0:
                continue
            class_id = int(class_id_value)
            if confidence < self._settings.confidence_threshold:
                continue
            if class_id < 0 or class_id >= len(self._settings.class_names):
                continue
            bbox = (
                min(max(x1 * scale_x, 0.0), float(source_width)),
                min(max(y1 * scale_y, 0.0), float(source_height)),
                min(max(x2 * scale_x, 0.0), float(source_width)),
                min(max(y2 * scale_y, 0.0), float(source_height)),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            outline = None
            if extra_values:
                outline = tuple(
                    (
                        min(max(float(point[0]) * scale_x, 0.0), float(source_width)),
                        min(max(float(point[1]) * scale_y, 0.0), float(source_height)),
                    )
                    for point in row[6:].reshape(-1, 2)
                )
            detections.append(
                Detection(
                    class_name=self._settings.class_names[class_id],
                    confidence=round(confidence, 5),
                    bbox_px=bbox,
                    outline_px=outline,
                )
            )
        return detections


def build_detector(settings: Settings) -> Detector:
    if settings.model_path is None:
        return UnavailableDetector("model_not_configured")
    try:
        return OnnxYoloDetector(settings)
    except Exception as exc:
        return UnavailableDetector(f"model_load_failed:{type(exc).__name__}")
