from __future__ import annotations

import uuid
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.detector import Detector, build_detector
from app.geometry import fit_ground_homography
from app.processing import build_scan_result
from app.quality import assess_frame
from app.schemas import CalibrationRequest, CreateScanRequest, EntryTriggerRequest
from app.storage import Repository


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    repository = Repository(config.database_path)
    model = detector or build_detector(config)
    application = FastAPI(
        title="Vehicle Perception POC API",
        version="0.1.0",
        description="Camera-led vehicle positioning and feature visibility validation service.",
    )
    application.state.settings = config
    application.state.repository = repository
    application.state.detector = model

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @application.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "detector_available": model.available,
            "detector_reason": model.reason,
        }

    @application.get("/api/v1/system")
    def system_status() -> dict[str, object]:
        calibrations = repository.list_calibrations()
        return {
            "service": "vehicle-perception-poc",
            "version": "0.1.0",
            "image_ingest": "http_upload",
            "detector_available": model.available,
            "detector_reason": model.reason,
            "calibrated_cameras": [item["camera_id"] for item in calibrations],
            "required_views": list(config.required_views),
            "raw_frames_persisted": False,
        }

    @application.post("/api/v1/scans", status_code=201)
    def create_scan(request: CreateScanRequest) -> dict[str, object]:
        return repository.create_scan(
            str(uuid.uuid4()), request.vehicle_reference, request.trigger_source
        )

    @application.post("/api/v1/triggers/entry", status_code=201)
    def vehicle_entry(request: EntryTriggerRequest) -> dict[str, object]:
        if not request.active:
            raise HTTPException(status_code=409, detail="entry sensor state is inactive")
        return repository.create_scan(
            str(uuid.uuid4()), request.vehicle_reference, f"sensor:{request.sensor_name}"
        )

    @application.get("/api/v1/scans")
    def list_scans(limit: int = 50) -> list[dict[str, object]]:
        if limit < 1 or limit > 200:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
        return repository.list_scans(limit)

    @application.get("/api/v1/scans/{scan_id}")
    def get_scan(scan_id: str) -> dict[str, object]:
        scan = repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        scan["frames"] = repository.get_frames(scan_id)
        return scan

    @application.post("/api/v1/scans/{scan_id}/frames", status_code=201)
    async def upload_frame(
        scan_id: str,
        camera_id: str = Form(min_length=1, max_length=80),
        view: str = Form(min_length=1, max_length=20),
        file: UploadFile = File(...),
        captured_at: datetime | None = Form(default=None),
    ) -> dict[str, object]:
        scan = repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        if scan["status"] != "acquiring":
            raise HTTPException(status_code=409, detail="scan is not accepting frames")
        allowed_views = {"front", "rear", "left", "right", "overhead", "other"}
        if view not in allowed_views:
            raise HTTPException(status_code=422, detail=f"view must be one of {sorted(allowed_views)}")

        payload = await file.read(config.max_frame_bytes + 1)
        await file.close()
        if len(payload) > config.max_frame_bytes:
            raise HTTPException(status_code=413, detail="image exceeds MAX_FRAME_BYTES")
        if not payload:
            raise HTTPException(status_code=400, detail="image is empty")

        try:
            with Image.open(BytesIO(payload)) as image_header:
                width, height = image_header.size
                if width < 1 or height < 1 or width * height > config.max_image_pixels:
                    raise HTTPException(status_code=413, detail="decoded image exceeds MAX_IMAGE_PIXELS")
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=400, detail="uploaded file is not a supported image") from exc
        except OSError as exc:
            raise HTTPException(status_code=400, detail="uploaded image header is invalid") from exc

        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="uploaded file is not a supported image")

        quality = assess_frame(image, config).to_dict()
        detections = []
        detector_error = None
        if model.available:
            try:
                detections = [
                    detection.to_dict()
                    for detection in await run_in_threadpool(model.detect, image)
                ]
            except Exception as exc:
                detector_error = type(exc).__name__

        if captured_at is not None and captured_at.utcoffset() is None:
            raise HTTPException(status_code=422, detail="captured_at must include a timezone")
        timestamp = (captured_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        try:
            return repository.add_frame(
                frame_id=str(uuid.uuid4()),
                scan_id=scan_id,
                camera_id=camera_id,
                view=view,
                captured_at=timestamp,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                quality=quality,
                detections=detections,
                detector_error=detector_error,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="scan not found") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @application.post("/api/v1/calibrations")
    def set_calibration(request: CalibrationRequest) -> dict[str, object]:
        if len(request.image_points_px) != len(request.world_points_mm):
            raise HTTPException(status_code=422, detail="image and world point counts must match")
        try:
            homography, error_mm = fit_ground_homography(
                request.image_points_px, request.world_points_mm
            )
        except (ValueError, cv2.error) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if error_mm > config.max_calibration_error_mm:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "calibration_reprojection_error_exceeded",
                    "measured_mm": round(error_mm, 3),
                    "limit_mm": config.max_calibration_error_mm,
                },
            )
        return repository.save_calibration(
            request.camera_id, homography, error_mm, len(request.image_points_px)
        )

    @application.get("/api/v1/calibrations")
    def list_calibrations() -> list[dict[str, object]]:
        return repository.list_calibrations()

    @application.get("/api/v1/calibrations/{camera_id}")
    def get_calibration(camera_id: str) -> dict[str, object]:
        calibration = repository.get_calibration(camera_id)
        if calibration is None:
            raise HTTPException(status_code=404, detail="camera calibration not found")
        return calibration

    @application.post("/api/v1/scans/{scan_id}/complete")
    def complete_scan(scan_id: str) -> dict[str, object]:
        try:
            scan = repository.claim_scan_completion(scan_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="scan not found") from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409, detail="scan is already being finalized or complete"
            ) from exc
        try:
            calibrations = {item["camera_id"]: item for item in repository.list_calibrations()}
            result = build_scan_result(
                scan=scan,
                frames=repository.get_frames(scan_id),
                calibrations=calibrations,
                detector_available=model.available,
                detector_reason=model.reason,
                settings=config,
            )
            return repository.complete_scan(scan_id, result)
        except Exception:
            repository.release_scan_completion(scan_id)
            raise

    @application.get("/api/v1/scans/{scan_id}/result")
    def get_result(scan_id: str) -> dict[str, object]:
        scan = repository.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail="scan not found")
        if scan["result"] is None:
            raise HTTPException(status_code=409, detail="scan has not been completed")
        return scan["result"]

    return application


app = create_app()
