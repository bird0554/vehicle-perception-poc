from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        if str(database_path) != ":memory:":
            database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    scan_id TEXT PRIMARY KEY,
                    vehicle_reference TEXT,
                    trigger_source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    result_json TEXT
                );
                CREATE TABLE IF NOT EXISTS frames (
                    frame_id TEXT PRIMARY KEY,
                    scan_id TEXT NOT NULL REFERENCES scans(scan_id),
                    camera_id TEXT NOT NULL,
                    view TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    quality_json TEXT NOT NULL,
                    detections_json TEXT NOT NULL,
                    detector_error TEXT
                );
                CREATE INDEX IF NOT EXISTS frames_scan_id ON frames(scan_id);
                CREATE TABLE IF NOT EXISTS calibrations (
                    camera_id TEXT PRIMARY KEY,
                    homography_json TEXT NOT NULL,
                    reprojection_error_mm REAL NOT NULL,
                    point_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def create_scan(self, scan_id: str, vehicle_reference: str | None, trigger_source: str) -> dict[str, Any]:
        created_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO scans(scan_id, vehicle_reference, trigger_source, status, created_at) "
                "VALUES (?, ?, ?, 'acquiring', ?)",
                (scan_id, vehicle_reference, trigger_source, created_at),
            )
        return {
            "scan_id": scan_id,
            "vehicle_reference": vehicle_reference,
            "trigger_source": trigger_source,
            "status": "acquiring",
            "created_at": created_at,
            "completed_at": None,
            "result": None,
        }

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
        if row is None:
            return None
        return self._scan_row(row)

    def list_scans(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._scan_row(row) for row in rows]

    @staticmethod
    def _scan_row(row: sqlite3.Row) -> dict[str, Any]:
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return {
            "scan_id": row["scan_id"],
            "vehicle_reference": row["vehicle_reference"],
            "trigger_source": row["trigger_source"],
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "result": result,
        }

    def add_frame(
        self,
        *,
        frame_id: str,
        scan_id: str,
        camera_id: str,
        view: str,
        captured_at: str,
        width: int,
        height: int,
        quality: dict[str, Any],
        detections: list[dict[str, Any]],
        detector_error: str | None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            scan = connection.execute(
                "SELECT status FROM scans WHERE scan_id = ?", (scan_id,)
            ).fetchone()
            if scan is None:
                raise KeyError(scan_id)
            if scan["status"] != "acquiring":
                raise RuntimeError("scan is not accepting frames")
            connection.execute(
                """INSERT INTO frames(
                    frame_id, scan_id, camera_id, view, captured_at, width, height,
                    quality_json, detections_json, detector_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    frame_id,
                    scan_id,
                    camera_id,
                    view,
                    captured_at,
                    width,
                    height,
                    json.dumps(quality),
                    json.dumps(detections),
                    detector_error,
                ),
            )
        return {
            "frame_id": frame_id,
            "scan_id": scan_id,
            "camera_id": camera_id,
            "view": view,
            "captured_at": captured_at,
            "width": width,
            "height": height,
            "quality": quality,
            "detections": detections,
            "detector_error": detector_error,
        }

    def get_frames(self, scan_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM frames WHERE scan_id = ? ORDER BY captured_at, frame_id", (scan_id,)
            ).fetchall()
        return [
            {
                "frame_id": row["frame_id"],
                "scan_id": row["scan_id"],
                "camera_id": row["camera_id"],
                "view": row["view"],
                "captured_at": row["captured_at"],
                "width": row["width"],
                "height": row["height"],
                "quality": json.loads(row["quality_json"]),
                "detections": json.loads(row["detections_json"]),
                "detector_error": row["detector_error"],
            }
            for row in rows
        ]

    def save_calibration(
        self,
        camera_id: str,
        homography: list[list[float]],
        reprojection_error_mm: float,
        point_count: int,
    ) -> dict[str, Any]:
        updated_at = utc_now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO calibrations(camera_id, homography_json, reprojection_error_mm, point_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    homography_json = excluded.homography_json,
                    reprojection_error_mm = excluded.reprojection_error_mm,
                    point_count = excluded.point_count,
                    updated_at = excluded.updated_at""",
                (camera_id, json.dumps(homography), reprojection_error_mm, point_count, updated_at),
            )
        return {
            "camera_id": camera_id,
            "homography": homography,
            "reprojection_error_mm": reprojection_error_mm,
            "point_count": point_count,
            "updated_at": updated_at,
        }

    def get_calibration(self, camera_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM calibrations WHERE camera_id = ?", (camera_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "camera_id": row["camera_id"],
            "homography": json.loads(row["homography_json"]),
            "reprojection_error_mm": row["reprojection_error_mm"],
            "point_count": row["point_count"],
            "updated_at": row["updated_at"],
        }

    def list_calibrations(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT camera_id FROM calibrations ORDER BY camera_id").fetchall()
        return [calibration for row in rows if (calibration := self.get_calibration(row["camera_id"]))]

    def complete_scan(self, scan_id: str, result: dict[str, Any]) -> dict[str, Any]:
        completed_at = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE scans SET status = ?, completed_at = ?, result_json = ? "
                "WHERE scan_id = ? AND status = 'finalizing'",
                (result["status"], completed_at, json.dumps(result), scan_id),
            )
            if connection.total_changes == 0:
                raise RuntimeError("scan is not being finalized")
        scan = self.get_scan(scan_id)
        if scan is None:
            raise KeyError(scan_id)
        return scan

    def claim_scan_completion(self, scan_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM scans WHERE scan_id = ?", (scan_id,)
            ).fetchone()
            if row is None:
                raise KeyError(scan_id)
            if row["status"] != "acquiring":
                raise RuntimeError("scan is not accepting completion")
            connection.execute(
                "UPDATE scans SET status = 'finalizing' WHERE scan_id = ?", (scan_id,)
            )
        scan = self.get_scan(scan_id)
        if scan is None:
            raise KeyError(scan_id)
        return scan

    def release_scan_completion(self, scan_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE scans SET status = 'acquiring' WHERE scan_id = ? AND status = 'finalizing'",
                (scan_id,),
            )
