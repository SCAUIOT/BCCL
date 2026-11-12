from __future__ import annotations

from pathlib import Path
import os
import shutil
import cv2

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return [p for p in sorted(path.rglob('*')) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def list_videos(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def read_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f'Failed to read image: {path}')
    return img


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    """Hard-link *src* → *dst*; fall back to copy on cross-filesystem error."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(str(src), str(dst))
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    """Legacy: always copies.  Prefer :func:`_hardlink_or_copy` where
    same-filesystem hard-links are safe (zero extra disk usage)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
