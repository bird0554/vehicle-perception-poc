from __future__ import annotations

import cv2
import numpy as np


def make_test_image() -> bytes:
    generator = np.random.default_rng(7)
    image = generator.integers(30, 220, size=(360, 640, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert success
    return encoded.tobytes()
