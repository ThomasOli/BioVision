#!/usr/bin/env python3
"""Prepare immutable model assets required by packaged one-shot annotation."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
YOLO_FILENAME = "yolov8s-worldv2.pt"
YOLO_SHA256 = "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"
CLIP_FILENAME = "ViT-B-32.pt"
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
SAM2_FILENAME = "sam2_b.pt"
SAM2_SHA256 = "39722bb0ce2a086058cf64e50dffd6f9e9931b5fcbee79e33b30441c6c40264d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> bool:
    return path.is_file() and _sha256(path) == expected


def _reject_bad_existing(path: Path, expected: str) -> None:
    if path.exists() and not _verify(path, expected):
        raise RuntimeError(
            f"Release asset {path.name} exists but its SHA-256 is not the pinned value {expected}. "
            "Remove or replace the file before packaging."
        )


def _prepare_yolo_world() -> Path:
    target = ROOT / YOLO_FILENAME
    _reject_bad_existing(target, YOLO_SHA256)
    if not target.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        resolved = Path(
            attempt_download_asset(YOLO_FILENAME, repo="ultralytics/assets", release="v8.4.0")
        ).resolve()
        if resolved != target:
            shutil.copy2(resolved, target)
    if not _verify(target, YOLO_SHA256):
        raise RuntimeError(f"Downloaded {YOLO_FILENAME} failed its pinned SHA-256 check.")
    return target


def _prepare_clip() -> Path:
    target = ROOT / CLIP_FILENAME
    _reject_bad_existing(target, CLIP_SHA256)
    if not target.exists():
        from clip.clip import _MODELS, _download

        with tempfile.TemporaryDirectory(prefix="biovision-clip-") as temp_dir:
            downloaded = Path(_download(_MODELS["ViT-B/32"], temp_dir))
            if _sha256(downloaded) != CLIP_SHA256:
                raise RuntimeError(f"Downloaded {CLIP_FILENAME} failed its pinned SHA-256 check.")
            shutil.copy2(downloaded, target)
    if not _verify(target, CLIP_SHA256):
        raise RuntimeError(f"Prepared {CLIP_FILENAME} failed its pinned SHA-256 check.")
    return target


def _prepare_sam2() -> Path:
    target = ROOT / SAM2_FILENAME
    _reject_bad_existing(target, SAM2_SHA256)
    if not target.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        resolved = Path(
            attempt_download_asset(SAM2_FILENAME, repo="ultralytics/assets", release="v8.4.0")
        ).resolve()
        if resolved != target:
            shutil.copy2(resolved, target)
    if not _verify(target, SAM2_SHA256):
        raise RuntimeError(f"Downloaded {SAM2_FILENAME} failed its pinned SHA-256 check.")
    return target


def main() -> None:
    assets = (_prepare_yolo_world(), _prepare_clip(), _prepare_sam2())
    for asset in assets:
        print(f"Release asset ready: {asset.name} sha256={_sha256(asset)}")


if __name__ == "__main__":
    main()
