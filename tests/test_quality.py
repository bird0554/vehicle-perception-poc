from __future__ import annotations

from pathlib import Path

import numpy as np

from app.config import Settings
from app.quality import assess_frame


def test_quality_gate_marks_saturated_sunlit_frame() -> None:
    settings = Settings(
        database_path=Path("unused.sqlite3"),
        min_frame_width=64,
        min_frame_height=64,
        min_blur_score=0,
        max_highlight_fraction=0.1,
    )
    image = np.full((120, 160, 3), 255, dtype=np.uint8)

    result = assess_frame(image, settings)

    assert not result.passed
    assert "highlight_clipping" in result.reasons


def test_quality_gate_accepts_detailed_frame() -> None:
    settings = Settings(
        database_path=Path("unused.sqlite3"),
        min_frame_width=64,
        min_frame_height=64,
        min_blur_score=0,
    )
    generator = np.random.default_rng(10)
    image = generator.integers(30, 220, size=(120, 160, 3), dtype=np.uint8)

    result = assess_frame(image, settings)

    assert result.passed
    assert result.width == 160
    assert result.height == 120
