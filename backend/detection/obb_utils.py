"""Shared Ultralytics OBB parsing and detection-profile helpers.

Ultralytics already applies oriented NMS for ``task="obb"``.  Consumers of
these helpers should not apply a second AABB-envelope suppression pass: two
valid, slender, rotated specimens can have nearly identical AABBs while their
oriented intersection is small.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterator, Optional

import numpy as np


def resolve_obb_model_config_path(model_path):
    """Return the configuration carried by an OBB model artifact or alias.

    New immutable artifacts store ``obb_config.json`` beside ``model.pt``.
    The remaining candidates retain compatibility with the promoted alias and
    with pre-registry Ultralytics ``weights/best.pt`` outputs.
    """
    model_dir = os.path.dirname(os.path.abspath(os.fspath(model_path)))
    candidates = (
        os.path.join(model_dir, "obb_config.json"),
        os.path.join(model_dir, "session_obb_detector_config.json"),
        os.path.join(os.path.dirname(model_dir), "obb_config.json"),
        os.path.join(model_dir, "obb_training", "session_obb", "obb_config.json"),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_obb_nms_iou(model_path, default=0.3):
    """Load an artifact's NMS preference, returning a finite fallback."""
    fallback = float(default)
    config_path = resolve_obb_model_config_path(model_path)
    if config_path is None:
        return fallback
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            value = float(json.load(handle).get("nms_iou", fallback))
        return value if math.isfinite(value) else fallback
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


def load_obb_confidence_threshold(model_path, default=0.3):
    """Load an artifact's validation-calibrated confidence threshold."""
    fallback = float(default)
    config_path = resolve_obb_model_config_path(model_path)
    if config_path is None:
        return fallback
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            value = float(json.load(handle).get("confidence_threshold", fallback))
        return value if math.isfinite(value) and 0.0 < value < 1.0 else fallback
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


def resolve_obb_detection_preset(
    conf_threshold=0.3,
    nms_iou=0.3,
    max_objects=20,
    detection_preset="balanced",
    imgsz=None,
):
    """Resolve the shared OBB inference profile used by all entry points."""
    preset = str(detection_preset or "balanced").strip().lower()
    conf = float(conf_threshold)
    iou = float(nms_iou)
    top_k = int(max_objects)
    resolved_imgsz = int(imgsz) if imgsz is not None else 640

    if preset == "custom":
        conf = min(max(conf, 0.01), 0.99)
        iou = min(max(iou, 0.05), 0.95)
        top_k = min(max(top_k, 1), 250)
        resolved_imgsz = 1280 if resolved_imgsz >= 1280 else 960 if resolved_imgsz >= 960 else 640
    elif preset == "precision":
        conf = max(conf, 0.45)
        top_k = min(top_k, 8)
        resolved_imgsz = 640
    elif preset == "recall":
        conf = min(conf, 0.2)
        top_k = max(top_k, 30)
        resolved_imgsz = 960 if resolved_imgsz < 960 else resolved_imgsz
    elif preset == "single_object":
        conf = max(conf, 0.35)
        top_k = 1
        resolved_imgsz = 640
    else:
        preset = "balanced"
        conf = max(0.3, min(conf, 0.9))
        top_k = max(1, min(top_k, 25))
        resolved_imgsz = 640 if resolved_imgsz not in (640, 960, 1280) else resolved_imgsz

    return {
        "preset": preset,
        "conf": conf,
        "iou": iou,
        "top_k": max(1, top_k),
        "imgsz": resolved_imgsz,
        "allow_relaxed_retry": False,
    }


def _build_canonical_obb_from_xywhr(cx, cy, width, height, angle_rad):
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    return np.asarray(
        [
            [float(cx) + cos_a * (-half_w) - sin_a * (-half_h), float(cy) + sin_a * (-half_w) + cos_a * (-half_h)],
            [float(cx) + cos_a * (half_w) - sin_a * (-half_h), float(cy) + sin_a * (half_w) + cos_a * (-half_h)],
            [float(cx) + cos_a * (half_w) - sin_a * (half_h), float(cy) + sin_a * (half_w) + cos_a * (half_h)],
            [float(cx) + cos_a * (-half_w) - sin_a * (half_h), float(cy) + sin_a * (-half_w) + cos_a * (half_h)],
        ],
        dtype=np.float32,
    )


def _as_corner_array(corners):
    arr = np.asarray(corners, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError("expected 4 OBB corners")
    return arr


def _roll_corners_to_top_left(corners):
    pts = _as_corner_array(corners)
    top_left_idx = min(range(4), key=lambda idx: (float(pts[idx][1]), float(pts[idx][0])))
    return np.roll(pts, -top_left_idx, axis=0)


def _is_valid_canonical_obb(corners, tolerance=1e-3):
    pts = _as_corner_array(corners)
    if len({(round(float(x), 4), round(float(y), 4)) for x, y in pts}) != 4:
        return False

    area2 = 0.0
    for idx in range(4):
        x1, y1 = pts[idx]
        x2, y2 = pts[(idx + 1) % 4]
        area2 += float(x1) * float(y2) - float(y1) * float(x2)
    if abs(area2) <= tolerance:
        return False

    if float(pts[0][1] + pts[1][1]) / 2.0 > float(pts[2][1] + pts[3][1]) / 2.0 + tolerance:
        return False
    if float(pts[0][0] + pts[3][0]) / 2.0 > float(pts[1][0] + pts[2][0]) / 2.0 + tolerance:
        return False

    return all(
        float(np.linalg.norm(pts[(idx + 1) % 4] - pts[idx])) > tolerance
        for idx in range(4)
    )


def _canonicalize_by_row_sort(corners):
    pts = _as_corner_array(corners)
    sorted_idx = sorted(range(4), key=lambda idx: (float(pts[idx][1]), float(pts[idx][0])))
    top = sorted((pts[sorted_idx[0]], pts[sorted_idx[1]]), key=lambda point: float(point[0]))
    bottom = sorted((pts[sorted_idx[2]], pts[sorted_idx[3]]), key=lambda point: float(point[0]))
    return np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def _canonicalize_by_angle_sort(corners):
    pts = _as_corner_array(corners)
    center = pts.mean(axis=0)
    order = sorted(
        range(4),
        key=lambda idx: math.atan2(float(pts[idx][1] - center[1]), float(pts[idx][0] - center[0])),
    )
    ordered = pts[order]
    rolled = _roll_corners_to_top_left(ordered)
    clockwise = np.asarray([rolled[0], rolled[3], rolled[2], rolled[1]], dtype=np.float32)
    counter_clockwise = np.asarray([rolled[0], rolled[1], rolled[2], rolled[3]], dtype=np.float32)
    clockwise_valid = _is_valid_canonical_obb(clockwise)
    counter_clockwise_valid = _is_valid_canonical_obb(counter_clockwise)
    if clockwise_valid and not counter_clockwise_valid:
        return clockwise
    if counter_clockwise_valid and not clockwise_valid:
        return counter_clockwise
    return clockwise if clockwise_valid else counter_clockwise


def _canonicalize_by_min_area_rect(corners):
    try:
        import cv2
    except Exception:
        return None
    rect = cv2.minAreaRect(_as_corner_array(corners))
    return _canonicalize_by_angle_sort(cv2.boxPoints(rect))


def canonicalize_detector_obb_corners(corners, xywhr=None):
    """Return corners in LT, RT, RB, LB order without altering geometry."""
    pts = _as_corner_array(corners)
    if xywhr is not None:
        xywhr_arr = np.asarray(xywhr, dtype=np.float32).reshape(-1)
        if xywhr_arr.shape[0] >= 5 and np.all(np.isfinite(xywhr_arr[:5])):
            candidate = _build_canonical_obb_from_xywhr(*xywhr_arr[:5])
            if _is_valid_canonical_obb(candidate):
                return candidate.tolist()

    for candidate in (_canonicalize_by_row_sort(pts), _canonicalize_by_angle_sort(pts)):
        if _is_valid_canonical_obb(candidate):
            return candidate.tolist()
    candidate = _canonicalize_by_min_area_rect(pts)
    if candidate is not None and _is_valid_canonical_obb(candidate):
        return candidate.tolist()
    return pts.tolist()


def _tensor_value(value):
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def iter_ultralytics_obb(result: Any, max_objects: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield a stable representation of every OBB in one Ultralytics result."""
    boxes_obj = getattr(result, "obb", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return

    names = getattr(result, "names", {}) or {}
    limit = len(boxes_obj)
    if max_objects is not None:
        limit = min(limit, max(1, int(max_objects)))

    for index in range(limit):
        try:
            xywhr = _tensor_value(boxes_obj.xywhr[index])
            xywhr = np.asarray(xywhr, dtype=np.float32).reshape(-1).tolist()
            if len(xywhr) < 5:
                continue
            raw_corners = _tensor_value(boxes_obj.xyxyxyxy[index])
            corners = canonicalize_detector_obb_corners(raw_corners, xywhr=xywhr)
            class_id = int(boxes_obj.cls[index]) if getattr(boxes_obj, "cls", None) is not None else 0
            confidence = float(boxes_obj.conf[index]) if getattr(boxes_obj, "conf", None) is not None else 0.0
            if isinstance(names, dict):
                class_name = names.get(class_id, "specimen")
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                class_name = names[class_id]
            else:
                class_name = "specimen"
            yield {
                "corners": corners,
                "angle_rad": float(xywhr[4]),
                "angle": float(xywhr[4]) * 180.0 / math.pi,
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": confidence,
            }
        except (IndexError, TypeError, ValueError):
            # A malformed detector row should not discard other valid rows.
            continue
