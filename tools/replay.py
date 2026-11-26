from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


VALID_VIEWS = {"front", "rear", "left", "right", "overhead", "other"}


def parse_image_name(path: Path) -> tuple[str, str]:
    parts = path.stem.split("__", maxsplit=2)
    if len(parts) < 2 or parts[0] not in VALID_VIEWS:
        raise ValueError(f"Expected <view>__<camera_id>__<name>.jpg, got {path.name}")
    return parts[0], parts[1]


def replay_directory(api_url: str, directory: Path, vehicle_reference: str | None) -> dict[str, object]:
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise ValueError(f"No supported images found in {directory}")
    parsed = [(path, *parse_image_name(path)) for path in images]

    with httpx.Client(base_url=api_url.rstrip("/"), timeout=60.0) as client:
        response = client.post(
            "/api/v1/scans",
            json={"vehicle_reference": vehicle_reference, "trigger_source": "dataset_replay"},
        )
        response.raise_for_status()
        scan = response.json()
        for path, view, camera_id in parsed:
            with path.open("rb") as image_file:
                response = client.post(
                    f"/api/v1/scans/{scan['scan_id']}/frames",
                    data={"camera_id": camera_id, "view": view},
                    files={"file": (path.name, image_file, "application/octet-stream")},
                )
            response.raise_for_status()
        response = client.post(f"/api/v1/scans/{scan['scan_id']}/complete")
        response.raise_for_status()
        return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a folder of vehicle camera frames through the POC API")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--vehicle-reference")
    args = parser.parse_args()
    result = replay_directory(args.api_url, args.directory, args.vehicle_reference)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
