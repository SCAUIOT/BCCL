"""Validate the system multipart files + metadata contract without ZIP staging."""
from __future__ import annotations
import hashlib
import math
import re
from pathlib import Path
from .config import digest


def save_multipart(metadata, files, destination, max_images, max_bytes):
    if not isinstance(metadata, dict) or not isinstance(metadata.get("experiment"), dict):
        raise ValueError("metadata.experiment is required for resumable experiments")
    rows = metadata.get("samples")
    if not isinstance(rows, list) or not rows or len(rows) != len(files) or len(rows) > max_images:
        raise ValueError("metadata.samples must match the ordered files list and batch limit")
    activity = metadata.get("activity")
    if not isinstance(activity, list) or len(activity) != len(rows):
        raise ValueError("metadata.activity must have one entry per uploaded image")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen, clean, total = set(), [], 0
    for row, file, active in zip(rows, files, activity):
        sid = row.get("sample_index")
        if type(sid) is not int or sid <= 0 or sid in seen:
            raise ValueError("Invalid or duplicate sample index")
        seen.add(sid)
        filename = row.get("filename", "")
        if not re.fullmatch(r"frame_[0-9]{12}\.jpg", filename) or filename != f"frame_{sid:012d}.jpg":
            raise ValueError("Unsafe image name or sample index mismatch")
        if file.filename != filename:
            raise ValueError("Multipart file order differs from metadata.samples")
        if "path" in row:
            raise ValueError("Client filesystem paths must not be sent in metadata")
        for key in ("avg_conf", "det_count", "non_lie_ratio", "recorded_time"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Non-finite {key}")
        if float(active) != float(row["non_lie_ratio"]):
            raise ValueError("Activity order differs from sample metadata")
        expected_size = row.get("size")
        if type(expected_size) is not int or expected_size < 1 or total + expected_size > max_bytes:
            raise ValueError("Upload exceeds configured byte limit")
        image, hasher, count = destination / filename, hashlib.sha256(), 0
        with image.open("wb") as target:
            while chunk := file.stream.read(1024 * 1024):
                count += len(chunk)
                if count > expected_size or total + count > max_bytes:
                    raise ValueError("Image exceeds declared or allowed size")
                hasher.update(chunk)
                target.write(chunk)
        if count != expected_size or hasher.hexdigest() != row.get("sha256"):
            raise ValueError("Image size or checksum mismatch")
        total += count
        clean.append({k: v for k, v in row.items() if k not in ("filename", "size", "sha256")}
                     | {"path": str(image.resolve())})
    return dict(metadata["experiment"]), clean, digest(metadata)
