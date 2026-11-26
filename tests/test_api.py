from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.detector import Detection, UnavailableDetector
from app.main import create_app
from conftest import make_test_image


class FakeDetector:
    available = True
    reason = None

    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, image: Any) -> list[Detection]:
        return self.detections


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_path": tmp_path / "scan.sqlite3",
        "required_views": ("front",),
        "min_frame_width": 32,
        "min_frame_height": 32,
        "min_blur_score": 0,
        "max_highlight_fraction": 0.12,
    }
    values.update(overrides)
    return Settings(**values)


def calibrate(client: TestClient, camera_id: str = "cam-front") -> None:
    response = client.post(
        "/api/v1/calibrations",
        json={
            "camera_id": camera_id,
            "image_points_px": [[0, 0], [640, 0], [640, 360], [0, 360]],
            "world_points_mm": [[0, 0], [32000, 0], [32000, 18000], [0, 18000]],
        },
    )
    assert response.status_code == 200, response.text


def upload_frame(client: TestClient, scan_id: str, view: str = "front", camera_id: str = "cam-front", image: bytes | None = None) -> dict[str, Any]:
    response = client.post(
        f"/api/v1/scans/{scan_id}/frames",
        data={"camera_id": camera_id, "view": view},
        files={"file": ("frame.jpg", image or make_test_image(), "image/jpeg")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_scan_lifecycle_calibrated_pose_and_features(tmp_path: Path) -> None:
    detections = [
        Detection("vehicle_body", 0.94, (180, 80, 420, 300)),
        Detection(
            "vehicle_footprint",
            0.91,
            (200, 100, 400, 260),
            ((200, 100), (400, 100), (400, 260), (200, 260)),
        ),
        Detection("front", 0.90, (260, 200, 340, 240)),
        Detection("rear", 0.89, (260, 80, 340, 120)),
        Detection("wheel", 0.88, (190, 220, 230, 260)),
        Detection("mirror", 0.86, (150, 140, 180, 170)),
        Detection("roof_rack", 0.84, (240, 130, 360, 150)),
        Detection("antenna", 0.82, (300, 105, 310, 120)),
    ]
    settings = settings_for(
        tmp_path,
        parking_zone_mm=(0, 0, 32000, 18000),
        expected_heading_degrees=90.0,
        heading_tolerance_degrees=5.0,
    )
    client = TestClient(create_app(settings, FakeDetector(detections)))
    calibrate(client)

    trigger = client.post(
        "/api/v1/triggers/entry",
        json={"sensor_name": "photoeye-1", "active": True, "vehicle_reference": "poc-car-01"},
    )
    assert trigger.status_code == 201
    scan_id = trigger.json()["scan_id"]
    frame = upload_frame(client, scan_id)
    assert frame["quality"]["passed"] is True

    completed = client.post(f"/api/v1/scans/{scan_id}/complete")
    assert completed.status_code == 200
    scan = completed.json()
    result = scan["result"]
    assert scan["status"] == "ready"
    assert result["position_mm"] == {"x": 15000.0, "y": 9000.0}
    assert result["orientation"]["yaw_degrees"] == 90.0
    assert result["features"]["wheel"]["status"] == "observed"
    assert result["features"]["antenna"]["status"] == "observed"
    assert result["parking_compliance"]["status"] == "estimated_compliant"
    assert result["body_envelope"]["measurement_status"] == "estimated_ground_footprint"
    assert client.get(f"/api/v1/scans/{scan_id}/result").json() == result
    assert client.post(f"/api/v1/scans/{scan_id}/complete").status_code == 409


def test_missing_model_and_calibration_never_create_physical_result(tmp_path: Path) -> None:
    client = TestClient(create_app(settings_for(tmp_path), UnavailableDetector("model_not_configured")))
    scan_id = client.post("/api/v1/scans", json={"trigger_source": "test"}).json()["scan_id"]
    upload_frame(client, scan_id)

    result = client.post(f"/api/v1/scans/{scan_id}/complete").json()["result"]

    assert result["status"] == "not_ready"
    assert result["position_mm"] is None
    assert result["features"]["front"]["status"] == "unknown"
    assert "detector_unavailable" in result["errors"]
    assert "camera_calibration_missing" in result["errors"]


def test_saturated_frame_does_not_drive_pose_or_features(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, max_highlight_fraction=0.1)
    detections = [Detection("vehicle_body", 0.95, (100, 80, 300, 300)), Detection("wheel", 0.8, (120, 200, 150, 240))]
    client = TestClient(create_app(settings, FakeDetector(detections)))
    calibrate(client)
    scan_id = client.post("/api/v1/scans", json={}).json()["scan_id"]
    ok, encoded = cv2.imencode(".jpg", np.full((360, 640, 3), 255, dtype=np.uint8))
    assert ok
    frame = upload_frame(client, scan_id, image=encoded.tobytes())
    assert "highlight_clipping" in frame["quality"]["reasons"]

    result = client.post(f"/api/v1/scans/{scan_id}/complete").json()["result"]

    assert result["position_mm"] is None
    assert result["features"]["wheel"]["status"] == "unknown"
    assert len(result["features"]["wheel"]["quality_rejected_observations"]) == 1
    assert result["status"] == "degraded"


def test_inactive_entry_trigger_and_invalid_image_are_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(settings_for(tmp_path), FakeDetector([])))
    assert client.post("/api/v1/triggers/entry", json={"active": False}).status_code == 409
    scan_id = client.post("/api/v1/scans", json={}).json()["scan_id"]
    response = client.post(
        f"/api/v1/scans/{scan_id}/frames",
        data={"camera_id": "cam-front", "view": "front"},
        files={"file": ("broken.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_frame_upload_after_completion_is_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(settings_for(tmp_path), FakeDetector([])))
    scan_id = client.post("/api/v1/scans", json={}).json()["scan_id"]
    assert client.post(f"/api/v1/scans/{scan_id}/complete").status_code == 200
    response = client.post(
        f"/api/v1/scans/{scan_id}/frames",
        data={"camera_id": "cam-front", "view": "front"},
        files={"file": ("frame.jpg", make_test_image(), "image/jpeg")},
    )
    assert response.status_code == 409


def test_empty_scan_and_missing_vehicle_detection_are_degraded(tmp_path: Path) -> None:
    client = TestClient(create_app(settings_for(tmp_path), FakeDetector([])))
    empty_scan_id = client.post("/api/v1/scans", json={}).json()["scan_id"]
    empty_result = client.post(f"/api/v1/scans/{empty_scan_id}/complete").json()["result"]
    assert empty_result["status"] == "degraded"
    assert "no_frames" in empty_result["errors"]

    scan_id = client.post("/api/v1/scans", json={}).json()["scan_id"]
    upload_frame(client, scan_id)
    result = client.post(f"/api/v1/scans/{scan_id}/complete").json()["result"]
    assert result["status"] == "degraded"
    assert "vehicle_not_detected" in result["errors"]


def test_decoded_pixel_limit_and_naive_timestamp_are_rejected(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, max_image_pixels=1000)
    client = TestClient(create_app(settings, FakeDetector([])))
    scan_id = client.post("/api/v1/scans", json={}).json()["scan_id"]
    response = client.post(
        f"/api/v1/scans/{scan_id}/frames",
        data={"camera_id": "cam-front", "view": "front"},
        files={"file": ("large.jpg", make_test_image(), "image/jpeg")},
    )
    assert response.status_code == 413

    normal_settings = settings_for(tmp_path / "timestamp")
    normal_client = TestClient(create_app(normal_settings, FakeDetector([])))
    normal_scan = normal_client.post("/api/v1/scans", json={}).json()["scan_id"]
    timestamp_response = normal_client.post(
        f"/api/v1/scans/{normal_scan}/frames",
        data={"camera_id": "cam-front", "view": "front", "captured_at": "2026-11-26T10:30:00"},
        files={"file": ("frame.jpg", make_test_image(), "image/jpeg")},
    )
    assert timestamp_response.status_code == 422


def test_finalizing_scan_rejects_late_frames_and_second_completion(tmp_path: Path) -> None:
    client = TestClient(create_app(settings_for(tmp_path), FakeDetector([])))
    scan_id = client.post("/api/v1/scans", json={}).json()["scan_id"]
    repository = client.app.state.repository
    repository.claim_scan_completion(scan_id)

    frame_response = client.post(
        f"/api/v1/scans/{scan_id}/frames",
        data={"camera_id": "cam-front", "view": "front"},
        files={"file": ("frame.jpg", make_test_image(), "image/jpeg")},
    )
    completion_response = client.post(f"/api/v1/scans/{scan_id}/complete")

    assert frame_response.status_code == 409
    assert completion_response.status_code == 409
    repository.release_scan_completion(scan_id)


def test_three_point_footprint_cannot_make_scan_ready(tmp_path: Path) -> None:
    detections = [
        Detection("vehicle_body", 0.9, (100, 80, 300, 280)),
        Detection("vehicle_footprint", 0.9, (100, 80, 300, 280), ((100, 80), (300, 80), (200, 280))),
        Detection("front", 0.9, (180, 200, 220, 240)),
        Detection("rear", 0.9, (180, 100, 220, 140)),
    ]
    client = TestClient(create_app(settings_for(tmp_path), FakeDetector(detections)))
    calibrate(client)
    scan_id = client.post("/api/v1/scans", json={}).json()["scan_id"]
    upload_frame(client, scan_id)

    result = client.post(f"/api/v1/scans/{scan_id}/complete").json()["result"]

    assert result["status"] == "degraded"
    assert "ground_footprint_polygon_unavailable" in result["errors"]
