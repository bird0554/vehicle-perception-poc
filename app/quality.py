from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from app.config import Settings


@dataclass(frozen=True)
class FrameQuality:
    width: int
    height: int
    mean_luminance: float
    highlight_fraction: float
    shadow_fraction: float
    blur_score: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def assess_frame(image_bgr: np.ndarray, settings: Settings) -> FrameQuality:
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    luminance = gray.astype(np.float32)
    highlight_fraction = float(np.mean(luminance >= 250))
    shadow_fraction = float(np.mean(luminance <= 5))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons: list[str] = []

    if width < settings.min_frame_width or height < settings.min_frame_height:
        reasons.append("low_resolution")
    if highlight_fraction > settings.max_highlight_fraction:
        reasons.append("highlight_clipping")
    if shadow_fraction > settings.max_highlight_fraction:
        reasons.append("shadow_clipping")
    if blur_score < settings.min_blur_score:
        reasons.append("motion_blur_or_low_detail")

    return FrameQuality(
        width=width,
        height=height,
        mean_luminance=round(float(np.mean(luminance)), 2),
        highlight_fraction=round(highlight_fraction, 5),
        shadow_fraction=round(shadow_fraction, 5),
        blur_score=round(blur_score, 2),
        passed=not reasons,
        reasons=tuple(reasons),
    )
