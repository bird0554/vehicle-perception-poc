from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from app.config import Settings
from app.geometry import (
    heading_degrees,
    map_image_point,
    map_image_polygon,
    polygon_centroid,
    polygon_dimensions_mm,
)


FEATURE_CLASSES = ("front", "rear", "wheel", "mirror", "roof_rack", "antenna")


def _center(bbox: list[float] | tuple[float, ...]) -> tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2


def _angular_difference(first: float, second: float) -> float:
    return abs((first - second + 180) % 360 - 180)


def _has_valid_outline(detection: dict[str, Any]) -> bool:
    outline = detection.get("outline_px")
    if not outline or not 4 <= len(outline) <= 128:
        return False
    if not all(math.isfinite(float(value)) for point in outline for value in point):
        return False
    try:
        polygon_centroid([(float(point[0]), float(point[1])) for point in outline])
    except ValueError:
        return False
    return True


def build_scan_result(
    *,
    scan: dict[str, Any],
    frames: list[dict[str, Any]],
    calibrations: dict[str, dict[str, Any]],
    detector_available: bool,
    detector_reason: str | None,
    settings: Settings,
) -> dict[str, Any]:
    errors: list[str] = []
    seen_views = {frame["view"] for frame in frames}
    if not frames:
        errors.append("no_frames")
    missing_views = sorted(set(settings.required_views) - seen_views)
    if missing_views:
        errors.append("missing_required_views")
    if not detector_available:
        errors.append("detector_unavailable")
    if any(not frame["quality"]["passed"] for frame in frames):
        errors.append("frame_quality_gate_failed")
    if any(frame["camera_id"] not in calibrations for frame in frames):
        errors.append("camera_calibration_missing")
    for frame in frames:
        if frame.get("detector_error"):
            errors.append("frame_inference_failed")
            break

    accepted_frames = [frame for frame in frames if frame["quality"]["passed"]]
    feature_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    uncertain_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    body_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    footprint_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []

    for frame in frames:
        accepted = frame["quality"]["passed"] and detector_available and not frame.get("detector_error")
        for detection in frame["detections"]:
            class_name = str(detection["class_name"])
            observation = {
                "frame_id": frame["frame_id"],
                "camera_id": frame["camera_id"],
                "view": frame["view"],
                "confidence": float(detection["confidence"]),
                "bbox_px": detection["bbox_px"],
            }
            if class_name in FEATURE_CLASSES:
                (feature_observations if accepted else uncertain_observations)[class_name].append(observation)
            if class_name == "vehicle_body":
                if accepted:
                    body_candidates.append((float(detection["confidence"]), frame, detection))
            if class_name == "vehicle_footprint" and accepted:
                footprint_candidates.append((float(detection["confidence"]), frame, detection))

    features = {
        name: {
            "status": "observed" if feature_observations[name] else "unknown",
            "observations": feature_observations[name],
            "quality_rejected_observations": uncertain_observations[name],
        }
        for name in FEATURE_CLASSES
    }

    position_mm = None
    body_envelope = None
    parking_outline_mm = None
    pose_source = None
    position_reference = "unknown"
    calibrated_footprints = [
        item for item in footprint_candidates if item[1]["camera_id"] in calibrations
    ]
    valid_footprints = [item for item in calibrated_footprints if _has_valid_outline(item[2])]
    calibrated_bodies = [item for item in body_candidates if item[1]["camera_id"] in calibrations]
    envelope_candidate = None
    if valid_footprints:
        envelope_candidate = max(valid_footprints, key=lambda item: item[0])
    elif calibrated_bodies:
        envelope_candidate = max(calibrated_bodies, key=lambda item: item[0])
    elif calibrated_footprints:
        envelope_candidate = max(calibrated_footprints, key=lambda item: item[0])

    if envelope_candidate:
        _, body_frame, body_detection = envelope_candidate
        calibration = calibrations.get(body_frame["camera_id"])
        if calibration:
            x1, y1, x2, y2 = map(float, body_detection["bbox_px"])
            is_footprint = body_detection["class_name"] == "vehicle_footprint"
            detected_outline = body_detection.get("outline_px")
            has_outline = is_footprint and _has_valid_outline(body_detection)
            outline_px = (
                [(float(point[0]), float(point[1])) for point in detected_outline]
                if has_outline
                else [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            )
            if has_outline:
                reference = polygon_centroid(outline_px)
            elif is_footprint:
                reference = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                reference = ((x1 + x2) / 2, y2)
            position_x, position_y = map_image_point(reference, calibration["homography"])
            position_mm = {"x": round(position_x, 1), "y": round(position_y, 1)}
            outline_mm = map_image_polygon(outline_px, calibration["homography"])
            if has_outline:
                parking_outline_mm = outline_mm
            body_envelope = {
                "outline_mm": [[round(x, 1), round(y, 1)] for x, y in outline_mm],
                **polygon_dimensions_mm(outline_mm),
                "measurement_status": (
                    "estimated_ground_footprint"
                    if has_outline
                    else "approximate_image_plane_projection"
                ),
                "source_class": body_detection["class_name"],
                "camera_id": body_frame["camera_id"],
                "frame_id": body_frame["frame_id"],
            }
            if has_outline:
                position_reference = "vehicle_footprint_polygon_centroid"
            elif is_footprint:
                position_reference = "vehicle_footprint_bbox_center"
            else:
                position_reference = "vehicle_body_bbox_bottom_center"
            pose_source = {"camera_id": body_frame["camera_id"], "frame_id": body_frame["frame_id"]}

    orientation = None
    if detector_available:
        for frame in accepted_frames:
            calibration = calibrations.get(frame["camera_id"])
            if calibration is None:
                continue
            front_detections = [item for item in frame["detections"] if item["class_name"] == "front"]
            rear_detections = [item for item in frame["detections"] if item["class_name"] == "rear"]
            if not front_detections or not rear_detections:
                continue
            front = max(front_detections, key=lambda item: item["confidence"])
            rear = max(rear_detections, key=lambda item: item["confidence"])
            front_mm = map_image_point(_center(front["bbox_px"]), calibration["homography"])
            rear_mm = map_image_point(_center(rear["bbox_px"]), calibration["homography"])
            orientation = {
                "yaw_degrees": round(heading_degrees(rear_mm, front_mm), 2),
                "method": "front_to_rear_feature_centers",
                "measurement_status": "estimated_from_image_feature_centers",
                "camera_id": frame["camera_id"],
                "frame_id": frame["frame_id"],
            }
            break

    if detector_available and not body_candidates and not footprint_candidates:
        errors.append("vehicle_not_detected")
    if detector_available:
        if not calibrated_footprints:
            errors.append("calibrated_ground_footprint_unavailable")
        elif not valid_footprints:
            errors.append("ground_footprint_polygon_unavailable")
        if position_mm is None:
            errors.append("calibrated_position_unavailable")
        if orientation is None:
            errors.append("orientation_unavailable")

    parking = {"status": "unknown", "reasons": ["parking_zone_not_configured"]}
    if settings.parking_zone_mm is not None:
        if not detector_available:
            parking = {"status": "unknown", "reasons": ["detector_unavailable"]}
        elif any(frame.get("detector_error") for frame in frames):
            parking = {"status": "unknown", "reasons": ["frame_inference_failed"]}
        elif missing_views:
            parking = {"status": "unknown", "reasons": ["required_views_missing"]}
        elif any(not frame["quality"]["passed"] for frame in frames):
            parking = {"status": "unknown", "reasons": ["frame_quality_gate_failed"]}
        elif any(frame["camera_id"] not in calibrations for frame in frames):
            parking = {"status": "unknown", "reasons": ["camera_calibration_missing"]}
        elif parking_outline_mm is None:
            parking = {"status": "unknown", "reasons": ["ground_footprint_detection_unavailable"]}
        elif orientation is None:
            parking = {"status": "unknown", "reasons": ["orientation_unavailable"]}
        else:
            x_min, y_min, x_max, y_max = settings.parking_zone_mm
            points_inside = all(
                x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
                for point in parking_outline_mm
            )
            heading_ok = True
            if settings.expected_heading_degrees is not None:
                heading_ok = (
                    _angular_difference(
                        orientation["yaw_degrees"], settings.expected_heading_degrees
                    )
                    <= settings.heading_tolerance_degrees
                )
            compliant = points_inside and heading_ok
            reasons = []
            if not points_inside:
                reasons.append("outline_outside_parking_zone")
            if not heading_ok:
                reasons.append("heading_outside_tolerance")
            parking = {
                "status": "estimated_compliant" if compliant else "estimated_non_compliant",
                "basis": "vehicle_footprint_detection_and_ground_homography",
                "reasons": reasons,
            }

    if not detector_available:
        status = "not_ready"
    else:
        status = "ready" if not errors else "degraded"

    return {
        "scan_id": scan["scan_id"],
        "status": status,
        "trigger_source": scan["trigger_source"],
        "vehicle_reference": scan["vehicle_reference"],
        "frame_count": len(frames),
        "view_coverage": sorted(seen_views),
        "missing_required_views": missing_views,
        "detector": {
            "available": detector_available,
            "reason": detector_reason,
        },
        "position_mm": position_mm,
        "position_reference": position_reference,
        "orientation": orientation,
        "body_envelope": body_envelope,
        "parking_compliance": parking,
        "features": features,
        "quality": {
            "passed_frames": sum(1 for frame in frames if frame["quality"]["passed"]),
            "failed_frames": sum(1 for frame in frames if not frame["quality"]["passed"]),
            "frame_diagnostics": [
                {
                    "frame_id": frame["frame_id"],
                    "camera_id": frame["camera_id"],
                    "view": frame["view"],
                    "passed": frame["quality"]["passed"],
                    "reasons": frame["quality"]["reasons"],
                    "highlight_fraction": frame["quality"]["highlight_fraction"],
                    "blur_score": frame["quality"]["blur_score"],
                }
                for frame in frames
            ],
        },
        "calibration": {
            "available_cameras": sorted(calibrations),
            "pose_source": pose_source,
        },
        "errors": sorted(set(errors)),
    }
