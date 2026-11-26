from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, FiniteFloat


FrameView = Literal["front", "rear", "left", "right", "overhead", "other"]


class CreateScanRequest(BaseModel):
    vehicle_reference: str | None = Field(default=None, max_length=80)
    trigger_source: str = Field(default="api", max_length=40)


class EntryTriggerRequest(BaseModel):
    sensor_name: str = Field(default="entry_sensor", max_length=80)
    active: bool = True
    vehicle_reference: str | None = Field(default=None, max_length=80)


class CalibrationRequest(BaseModel):
    camera_id: str = Field(min_length=1, max_length=80)
    image_points_px: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=4, max_length=64)
    world_points_mm: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=4, max_length=64)


class CalibrationResponse(BaseModel):
    camera_id: str
    homography: list[list[float]]
    reprojection_error_mm: float
    point_count: int
    updated_at: datetime
