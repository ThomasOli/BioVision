#!/usr/bin/env python3
"""
Export BioVision session annotations to YOLO training format.

Key rules:
- Detection training uses only finalized accepted boxes.
- Draft auto-detected boxes are ignored until image finalization.
- Synthetic images are generated from finalized SAM2 segment crops on synthetic
  canvases (not on top of original images), with non-overlap constraints.
"""

import json
import math
import os
import random
import shutil
import glob
import sys

import cv2
import numpy as np

import sys as _sys, os as _os
_BACKEND_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _BACKEND_ROOT not in _sys.path:
    _sys.path.insert(0, _BACKEND_ROOT)

from bv_utils.image_utils import safe_imread, safe_imwrite
from bv_utils.orientation_utils import resolve_session_augmentation_profile


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _safe_int(value, default=0):
    try:
        return int(round(float(value)))
    except Exception:
        return default


def _safe_float(value, default=None):
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _normalize_category_targets(raw_categories, fallback):
    if isinstance(raw_categories, (list, tuple)):
        values = [str(c).strip().lower() for c in raw_categories if str(c).strip()]
        if values:
            return values
    return [str(c).strip().lower() for c in fallback if str(c).strip()]


def _normalize_landmarks(raw_landmarks):
    if not isinstance(raw_landmarks, list):
        return []
    normalized = []
    for idx, lm in enumerate(raw_landmarks):
        if not isinstance(lm, dict):
            continue
        lm_id = _safe_int(lm.get("id"), None)
        skipped = bool(lm.get("isSkipped"))
        x = _safe_float(lm.get("x"), None)
        y = _safe_float(lm.get("y"), None)
        if lm_id is None:
            continue
        if not skipped and (x is None or y is None):
            continue
        normalized.append(
            {
                "id": lm_id,
                "x": -1.0 if skipped or x is None else float(x),
                "y": -1.0 if skipped or y is None else float(y),
                **({"isSkipped": True} if skipped else {}),
            }
        )
    return normalized


def _box_signature(box):
    if not isinstance(box, dict):
        return None
    left = _safe_int(box.get("left"), None)
    top = _safe_int(box.get("top"), None)
    width = _safe_int(box.get("width"), None)
    height = _safe_int(box.get("height"), None)
    if left is None or top is None or width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        return None
    return (left, top, width, height)


# -----------------------------------------------------------------------------
# Orientation / pose helpers
# -----------------------------------------------------------------------------

def _load_head_tail_ids(session_dir):
    """
    Read session.json and return (head_id, tail_id).
    """
    session_path = os.path.join(session_dir, "session.json")
    if not os.path.exists(session_path):
        return None, None
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
    except Exception:
        return None, None

    template = session.get("landmarkTemplate", [])
    if not isinstance(template, list):
        return None, None

    orientation_policy = session.get("orientationPolicy", {})
    if not isinstance(orientation_policy, dict):
        orientation_policy = {}
    head_targets = set(
        _normalize_category_targets(
            orientation_policy.get("headCategories"),
            ["head"],
        )
    )
    tail_targets = set(
        _normalize_category_targets(
            orientation_policy.get("tailCategories"),
            ["tail"],
        )
    )

    head_id = None
    tail_id = None
    for lm in template:
        try:
            idx = int(lm.get("index"))
        except (TypeError, ValueError):
            continue
        cat = str(lm.get("category", "")).strip().lower()
        if cat in head_targets and head_id is None:
            head_id = idx
        elif cat in tail_targets and tail_id is None:
            tail_id = idx
    return head_id, tail_id


def _load_orientation_anchor_ids(session_dir):
    session_path = os.path.join(session_dir, "session.json")
    if not os.path.exists(session_path):
        return [], []
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
        orientation_policy = session.get("orientationPolicy", {})
        if not isinstance(orientation_policy, dict):
            return [], []
        anterior = orientation_policy.get("anteriorAnchorIds", [])
        posterior = orientation_policy.get("posteriorAnchorIds", [])
        anterior_ids = []
        for value in anterior:
            parsed = _safe_int(value, None)
            if parsed is None or parsed <= 0 or parsed in anterior_ids:
                continue
            anterior_ids.append(parsed)
        posterior_ids = []
        for value in posterior:
            parsed = _safe_int(value, None)
            if parsed is None or parsed <= 0 or parsed in posterior_ids:
                continue
            posterior_ids.append(parsed)
        anterior_ids.sort()
        posterior_ids.sort()
        return anterior_ids, posterior_ids
    except Exception:
        return [], []


def _load_session_orientation_policy(session_dir):
    session_path = os.path.join(session_dir, "session.json")
    if not os.path.exists(session_path):
        return {}
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
        orientation_policy = session.get("orientationPolicy", {})
        return orientation_policy if isinstance(orientation_policy, dict) else {}
    except Exception:
        return {}


def _resolve_detector_class_config(orientation_policy, fallback_mode="invariant"):
    mode = str(
        (orientation_policy or {}).get("mode", fallback_mode or "invariant")
    ).strip().lower()
    bilateral_axis = _resolve_bilateral_class_axis(orientation_policy)
    if mode == "directional":
        return True, ["left", "right"], mode, bilateral_axis
    if mode == "bilateral":
        return True, ["up", "down"], mode, bilateral_axis
    return False, ["specimen"], mode, bilateral_axis


def _get_box_explicit_orientation(box_dict, orientation_policy=None, minimum_hint_confidence=0.35):
    mode = str((orientation_policy or {}).get("mode", "invariant")).strip().lower()
    bilateral_axis = _resolve_bilateral_class_axis(orientation_policy)

    override = str(box_dict.get("orientation_override", "")).strip().lower()
    if override == "uncertain":
        return None
    if mode == "bilateral":
        if override in {"up", "down"}:
            return override
    elif override in {"left", "right"}:
        return override

    hint = box_dict.get("orientation_hint", {})
    if isinstance(hint, dict):
        hint_orientation = str(hint.get("orientation", "")).strip().lower()
        hint_confidence = _safe_float(hint.get("confidence"), None)
        if hint_confidence is None or hint_confidence >= float(minimum_hint_confidence):
            if mode == "bilateral":
                if hint_orientation in {"up", "down"}:
                    return hint_orientation
            elif hint_orientation in {"left", "right"}:
                return hint_orientation

    class_id = box_dict.get("class_id", None)
    try:
        class_id = int(class_id) if class_id is not None else None
    except Exception:
        class_id = None
    if class_id is None:
        return None
    if mode == "bilateral":
        return "up" if class_id == 0 else "down"
    if mode in {"directional", "bilateral"}:
        return "left" if class_id == 0 else "right"
    return None


def _compute_box_orientation(box_dict, head_id, tail_id):
    """
    Determine left/right specimen orientation from box landmarks.

    Returns:
      "left" | "right" | None
    """
    explicit = _get_box_explicit_orientation(box_dict, {"mode": "directional"})
    if explicit in {"left", "right"}:
        return explicit

    landmarks = [
        lm for lm in box_dict.get("landmarks", [])
        if not lm.get("isSkipped")
        and lm.get("x", -1) >= 0
        and lm.get("y", -1) >= 0
    ]
    if len(landmarks) < 2:
        return None

    head_lm = None
    if head_id is not None:
        head_lm = next((lm for lm in landmarks if int(lm.get("id", -1)) == int(head_id)), None)
    if head_lm is None:
        try:
            head_lm = min(landmarks, key=lambda lm: int(lm.get("id", 0)))
        except Exception:
            return None

    tail_lm = None
    if tail_id is not None:
        tail_lm = next((lm for lm in landmarks if int(lm.get("id", -1)) == int(tail_id)), None)
    if tail_lm is None:
        others = [lm for lm in landmarks if int(lm.get("id", -99999)) != int(head_lm.get("id", -99998))]
        if not others:
            return None
        hx, hy = float(head_lm["x"]), float(head_lm["y"])
        tail_lm = max(others, key=lambda lm: math.hypot(float(lm["x"]) - hx, float(lm["y"]) - hy))

    return "left" if float(head_lm["x"]) < float(tail_lm["x"]) else "right"


def _resolve_bilateral_class_axis(orientation_policy):
    return "vertical_obb"


def _get_box_vertical_axis(box_dict):
    obb_corners = box_dict.get("obbCorners") or box_dict.get("obb_corners")
    if isinstance(obb_corners, list) and len(obb_corners) == 4:
        try:
            cp0, cp1, cp2, cp3 = [
                (float(point[0]), float(point[1])) for point in obb_corners
            ]
        except Exception:
            cp0 = cp1 = cp2 = cp3 = None
    else:
        cp0 = cp1 = cp2 = cp3 = None

    if cp0 is None:
        xyxy = _box_xyxy_from_xywh_box(box_dict) or _box_xyxy_from_meta_box(box_dict)
        if xyxy is None:
            return None
        left, top, right, bottom = xyxy
        cp0 = (left, top)
        cp1 = (right, top)
        cp2 = (right, bottom)
        cp3 = (left, bottom)

    center_x = (cp0[0] + cp1[0] + cp2[0] + cp3[0]) / 4.0
    center_y = (cp0[1] + cp1[1] + cp2[1] + cp3[1]) / 4.0
    top_mid_x = (cp0[0] + cp1[0]) / 2.0
    top_mid_y = (cp0[1] + cp1[1]) / 2.0
    up_x = top_mid_x - center_x
    up_y = top_mid_y - center_y
    mag = math.hypot(up_x, up_y)
    if mag <= 1e-6:
        return None
    return (center_x, center_y), (up_x / mag, up_y / mag)


def _resolve_landmark_centroid(box_dict, landmark_ids):
    if not landmark_ids:
        return None
    pts = []
    for lm in box_dict.get("landmarks", []):
        if lm.get("isSkipped"):
            continue
        lm_id = _safe_int(lm.get("id"), None)
        x = _safe_float(lm.get("x"), None)
        y = _safe_float(lm.get("y"), None)
        if lm_id is None or x is None or y is None:
            continue
        if int(lm_id) in landmark_ids:
            pts.append((float(x), float(y)))
    if not pts:
        return None
    return (
        sum(point[0] for point in pts) / float(len(pts)),
        sum(point[1] for point in pts) / float(len(pts)),
    )


def _compute_box_bilateral_orientation(box_dict, head_id, tail_id, orientation_policy=None):
    explicit = _get_box_explicit_orientation(
        box_dict,
        {"mode": "bilateral", "bilateralClassAxis": "vertical_obb"},
    )
    if explicit in {"up", "down"}:
        return explicit

    anterior_ids = set((orientation_policy or {}).get("anteriorAnchorIds") or [])
    posterior_ids = set((orientation_policy or {}).get("posteriorAnchorIds") or [])
    head_xy = _resolve_landmark_centroid(box_dict, anterior_ids)
    tail_xy = _resolve_landmark_centroid(box_dict, posterior_ids)
    points = (head_xy, tail_xy) if head_xy and tail_xy else _extract_head_tail_points(box_dict, head_id, tail_id)
    if not points:
        return None
    axis = _get_box_vertical_axis(box_dict)
    if axis is None:
        return "up" if float(points[0][1]) < float(points[1][1]) else "down"

    (center_x, center_y), (up_x, up_y) = axis
    head_xy, tail_xy = points
    head_proj = (head_xy[0] - center_x) * up_x + (head_xy[1] - center_y) * up_y
    tail_proj = (tail_xy[0] - center_x) * up_x + (tail_xy[1] - center_y) * up_y
    if abs(head_proj - tail_proj) <= 1e-6:
        return "up" if float(head_xy[1]) < float(tail_xy[1]) else "down"
    return "up" if head_proj > tail_proj else "down"


def _extract_head_tail_points(box_dict, head_id, tail_id):
    """
    Return absolute (head_xy, tail_xy) points from a box landmark set.
    Returns None when unavailable.
    """
    landmarks = [
        lm for lm in box_dict.get("landmarks", [])
        if not lm.get("isSkipped")
        and lm.get("x", -1) >= 0
        and lm.get("y", -1) >= 0
    ]
    if len(landmarks) < 2:
        return None

    head_lm = None
    if head_id is not None:
        head_lm = next((lm for lm in landmarks if int(lm.get("id", -1)) == int(head_id)), None)
    if head_lm is None:
        try:
            head_lm = min(landmarks, key=lambda lm: int(lm.get("id", 0)))
        except Exception:
            return None

    tail_lm = None
    if tail_id is not None:
        tail_lm = next((lm for lm in landmarks if int(lm.get("id", -1)) == int(tail_id)), None)
    if tail_lm is None:
        others = [lm for lm in landmarks if int(lm.get("id", -99999)) != int(head_lm.get("id", -99998))]
        if not others:
            return None
        hx, hy = float(head_lm["x"]), float(head_lm["y"])
        tail_lm = max(others, key=lambda lm: math.hypot(float(lm["x"]) - hx, float(lm["y"]) - hy))

    return (
        (float(head_lm["x"]), float(head_lm["y"])),
        (float(tail_lm["x"]), float(tail_lm["y"])),
    )


def _compute_box_directional_orientation(box_dict, orientation_policy=None, head_id=None, tail_id=None):
    explicit = _get_box_explicit_orientation(box_dict, {"mode": "directional"})
    if explicit in {"left", "right"}:
        return explicit

    anterior_ids = set((orientation_policy or {}).get("anteriorAnchorIds") or [])
    posterior_ids = set((orientation_policy or {}).get("posteriorAnchorIds") or [])
    head_xy = _resolve_landmark_centroid(box_dict, anterior_ids)
    tail_xy = _resolve_landmark_centroid(box_dict, posterior_ids)
    if head_xy is not None and tail_xy is not None:
        obb_corners = box_dict.get("obbCorners") or box_dict.get("obb_corners")
        if isinstance(obb_corners, list) and len(obb_corners) == 4:
            try:
                cp0, cp1, cp2, cp3 = [(float(point[0]), float(point[1])) for point in obb_corners]
                center_x = (cp0[0] + cp1[0] + cp2[0] + cp3[0]) / 4.0
                center_y = (cp0[1] + cp1[1] + cp2[1] + cp3[1]) / 4.0
                right_mid_x = (cp1[0] + cp2[0]) / 2.0
                right_mid_y = (cp1[1] + cp2[1]) / 2.0
                axis_x = right_mid_x - center_x
                axis_y = right_mid_y - center_y
                mag = math.hypot(axis_x, axis_y)
                if mag > 1e-6:
                    axis_x /= mag
                    axis_y /= mag
                    head_proj = (head_xy[0] - center_x) * axis_x + (head_xy[1] - center_y) * axis_y
                    tail_proj = (tail_xy[0] - center_x) * axis_x + (tail_xy[1] - center_y) * axis_y
                    if abs(head_proj - tail_proj) > 1e-6:
                        return "left" if head_proj < tail_proj else "right"
            except Exception:
                pass
        return "left" if float(head_xy[0]) < float(tail_xy[0]) else "right"

    return _compute_box_orientation(box_dict, head_id, tail_id)


def _norm_path(path_value):
    if not path_value:
        return ""
    try:
        return os.path.normcase(os.path.abspath(str(path_value)))
    except Exception:
        return str(path_value)


def _box_xyxy_from_meta_box(box_dict):
    if not isinstance(box_dict, dict):
        return None
    left = _safe_float(box_dict.get("left"), None)
    top = _safe_float(box_dict.get("top"), None)
    right = _safe_float(box_dict.get("right"), None)
    bottom = _safe_float(box_dict.get("bottom"), None)
    if left is None or top is None or right is None or bottom is None:
        return None
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _box_xyxy_from_xywh_box(box_dict):
    if not isinstance(box_dict, dict):
        return None
    left = _safe_float(box_dict.get("left"), None)
    top = _safe_float(box_dict.get("top"), None)
    width = _safe_float(box_dict.get("width"), None)
    height = _safe_float(box_dict.get("height"), None)
    if left is None or top is None or width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        return None
    return (left, top, left + width, top + height)


def _xyxy_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _build_anchor_index(positives, head_id, tail_id):
    """
    Build per-image anchor lookup from finalized positive boxes.
    """
    index = {}
    for sample in positives:
        img_key = _norm_path(sample.get("image_path"))
        if not img_key:
            continue
        entries = index.setdefault(img_key, [])
        for box in sample.get("boxes", []):
            points = _extract_head_tail_points(box, head_id, tail_id)
            if not points:
                continue
            xyxy = _box_xyxy_from_xywh_box(box)
            if xyxy is None:
                continue
            entries.append(
                {
                    "xyxy": xyxy,
                    "head_xy": points[0],
                    "tail_xy": points[1],
                }
            )
    return index


def _build_source_box_index(
    positives,
    orientation_class_enabled=False,
    head_id=None,
    tail_id=None,
    orientation_policy=None,
):
    index = {}
    for sample in positives:
        img_key = _norm_path(sample.get("image_path"))
        if not img_key:
            continue
        entries = index.setdefault(img_key, [])
        for box in sample.get("boxes", []):
            xyxy = _box_xyxy_from_xywh_box(box)
            if xyxy is None:
                continue
            source_class_id = None
            if orientation_class_enabled:
                source_class_id = _resolve_obb_class_id(
                    box,
                    orientation_class_enabled=True,
                    head_id=head_id,
                    tail_id=tail_id,
                    orientation_policy=orientation_policy,
                )
            entries.append({"xyxy": xyxy, "class_id": source_class_id})
    return index


# -----------------------------------------------------------------------------
# Finalization helpers
# -----------------------------------------------------------------------------

def _load_finalized_filenames(session_dir):
    path = os.path.join(session_dir, "finalized_images.json")
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(str(x) for x in data)
    except Exception:
        pass
    return set()


def _normalize_box(box):
    sig = _box_signature(box)
    if sig is None:
        return None
    left, top, width, height = sig
    orientation_override_raw = str(
        box.get("orientation_override", box.get("orientationOverride", ""))
    ).strip().lower()
    orientation_override = (
        orientation_override_raw
        if orientation_override_raw in {"left", "right", "up", "down", "uncertain"}
        else None
    )
    out = {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        **({"orientation_override": orientation_override} if orientation_override else {}),
        "landmarks": _normalize_landmarks(box.get("landmarks", [])),
    }
    orientation_hint = box.get("orientation_hint")
    if isinstance(orientation_hint, dict):
        hint_orientation = str(orientation_hint.get("orientation", "")).strip().lower()
        if hint_orientation in {"left", "right", "up", "down"}:
            out["orientation_hint"] = {
                "orientation": hint_orientation,
                **(
                    {"confidence": float(orientation_hint.get("confidence"))}
                    if _safe_float(orientation_hint.get("confidence"), None) is not None
                    else {}
                ),
                **(
                    {"source": str(orientation_hint.get("source"))}
                    if orientation_hint.get("source")
                    else {}
                ),
            }
    # Preserve OBB geometry fields so export_obb_dataset can use the 4-corner format.
    obb_corners = box.get("obbCorners") or box.get("obb_corners")
    if obb_corners and len(obb_corners) == 4:
        out["obbCorners"] = [[float(c[0]), float(c[1])] for c in obb_corners]
    angle_val = box.get("angle")
    if angle_val is not None:
        try:
            out["angle"] = float(angle_val)
        except (TypeError, ValueError):
            pass
    class_id_val = box.get("class_id")
    if class_id_val is not None:
        try:
            out["class_id"] = int(class_id_val)
        except (TypeError, ValueError):
            pass
    return out


def _get_finalized_boxes(label_data, image_filename, finalized_set):
    """
    Resolve finalized accepted boxes from label JSON.

    Returns:
      (is_finalized, boxes, used_fallback_to_draft)
    """
    finalized_detection = label_data.get("finalizedDetection")
    finalized_flag = bool(
        isinstance(finalized_detection, dict)
        and finalized_detection.get("isFinalized")
    )
    is_finalized = finalized_flag or (image_filename in finalized_set)
    if not is_finalized:
        return False, [], False

    accepted = []
    used_fallback = False
    draft_boxes = []
    draft_by_signature = {}
    for b in label_data.get("boxes", []):
        nb = _normalize_box(b)
        if not nb:
            continue
        draft_boxes.append(nb)
        sig = _box_signature(nb)
        if sig is None:
            continue
        draft_by_signature.setdefault(sig, []).append(nb)

    if isinstance(finalized_detection, dict):
        accepted_raw = finalized_detection.get("acceptedBoxes", [])
        if isinstance(accepted_raw, list):
            for b in accepted_raw:
                nb = _normalize_box(b)
                if nb:
                    # Backfill landmarks from draft boxes when finalized snapshot
                    # was stored as geometry-only.
                    if not nb.get("landmarks") or not nb.get("obbCorners"):
                        sig = _box_signature(nb)
                        candidates = draft_by_signature.get(sig, []) if sig else []
                        if candidates:
                            best = max(
                                candidates,
                                key=lambda x: len(x.get("landmarks", [])),
                            )
                            if best.get("landmarks") and not nb.get("landmarks"):
                                nb["landmarks"] = [dict(lm) for lm in best["landmarks"]]
                            if not nb.get("orientation_override") and best.get("orientation_override"):
                                nb["orientation_override"] = best.get("orientation_override")
                            # Backfill OBB geometry from draft box when not in accepted snapshot
                            if not nb.get("obbCorners") and best.get("obbCorners"):
                                nb["obbCorners"] = list(best["obbCorners"])
                            if nb.get("angle") is None and best.get("angle") is not None:
                                nb["angle"] = best["angle"]
                            if nb.get("class_id") is None and best.get("class_id") is not None:
                                nb["class_id"] = best["class_id"]
                    accepted.append(nb)

    # Backward compatibility: older finalized sessions may not have
    # finalizedDetection.acceptedBoxes yet.
    if not accepted:
        used_fallback = True
        accepted.extend([dict(b) for b in draft_boxes])

    accepted.sort(key=lambda b: (b["left"], b["top"], b["width"], b["height"]))
    return True, accepted, used_fallback


def _reset_output_dataset_dir(out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)


def _sample_orientation_classes(sample, head_id, tail_id, orientation_policy=None):
    classes = set()
    for box in sample.get("boxes", []):
        class_id = _resolve_obb_class_id(
            box,
            orientation_class_enabled=True,
            head_id=head_id,
            tail_id=tail_id,
            orientation_policy=orientation_policy,
        )
        classes.add(int(class_id))
    return classes


def _resolve_obb_class_id(
    box,
    orientation_class_enabled=False,
    head_id=None,
    tail_id=None,
    orientation_policy=None,
):
    """
    Resolve class id used by OBB export.
    Priority:
      1) explicit box.class_id
      2) stored orientation metadata
      3) orientation from landmarks (legacy compatibility)
      3) default 0
    """
    if not orientation_class_enabled:
        return 0

    class_id = box.get("class_id", None)
    if class_id is not None:
        try:
            return int(class_id)
        except Exception:
            pass

    mode = str((orientation_policy or {}).get("mode", "directional")).strip().lower()
    bilateral_axis = _resolve_bilateral_class_axis(orientation_policy)
    explicit_orientation = _get_box_explicit_orientation(box, orientation_policy)
    if mode == "bilateral" and bilateral_axis == "vertical_obb":
        if explicit_orientation == "down":
            return 1
        if explicit_orientation == "up":
            return 0
    else:
        if explicit_orientation == "right":
            return 1
        if explicit_orientation == "left":
            return 0

    if mode == "bilateral":
        orientation = _compute_box_bilateral_orientation(box, head_id, tail_id)
        if orientation == "down":
            return 1
        if orientation == "up":
            return 0
        return 0

    orientation = _compute_box_directional_orientation(
        box,
        orientation_policy=orientation_policy,
        head_id=head_id,
        tail_id=tail_id,
    )
    if orientation == "right":
        return 1
    if orientation == "left":
        return 0
    return 0


def _sample_class_histogram(
    sample,
    orientation_class_enabled=False,
    head_id=None,
    tail_id=None,
    orientation_policy=None,
):
    """
    Count class instances for a sample based on export class-id resolution.
    Returns dict[class_id -> instance_count].
    """
    hist = {}
    for box in sample.get("boxes", []):
        width = _safe_float(box.get("width"), 0.0)
        height = _safe_float(box.get("height"), 0.0)
        if width is None or height is None or width <= 0 or height <= 0:
            continue
        class_id = _resolve_obb_class_id(
            box,
            orientation_class_enabled=orientation_class_enabled,
            head_id=head_id,
            tail_id=tail_id,
            orientation_policy=orientation_policy,
        )
        hist[class_id] = hist.get(class_id, 0) + 1
    return hist


def _merge_class_histograms(target, source):
    for class_id, count in (source or {}).items():
        target[int(class_id)] = target.get(int(class_id), 0) + int(count)
    return target


def _box_rotation_degrees(box):
    angle_val = box.get("angle")
    if angle_val is not None:
        try:
            return float(angle_val)
        except Exception:
            pass
    obb_corners = box.get("obbCorners") or box.get("obb_corners")
    if not isinstance(obb_corners, list) or len(obb_corners) != 4:
        return None
    try:
        p0 = obb_corners[0]
        p1 = obb_corners[1]
        return math.degrees(math.atan2(float(p1[1]) - float(p0[1]), float(p1[0]) - float(p0[0])))
    except Exception:
        return None


def _sample_has_rotated_obb(sample, threshold_deg=3.0):
    for box in sample.get("boxes", []):
        obb_corners = box.get("obbCorners") or box.get("obb_corners")
        if not isinstance(obb_corners, list) or len(obb_corners) != 4:
            continue
        angle = _box_rotation_degrees(box)
        if angle is None:
            return True
        angle_mod = abs(float(angle)) % 180.0
        dist_to_axis = min(abs(angle_mod - 0.0), abs(angle_mod - 90.0), abs(angle_mod - 180.0))
        if dist_to_axis > float(threshold_deg):
            return True
    return False


def _select_obb_val_indices(
    samples,
    val_ratio,
    seed,
    orientation_class_enabled=False,
    head_id=None,
    tail_id=None,
    orientation_policy=None,
    minority_small_cutoff=20,
    minority_target_ratio=0.15,
    minority_min_ratio=0.10,
    minority_max_ratio=0.20,
):
    """
    Select val indices for OBB export with optional minority-protection policy.

    Minority policy:
      - Determine minority classes dynamically from class instance counts.
      - When minority count is small (<= cutoff), keep minority val instances
        near target_ratio, bounded to [minority_min_ratio, minority_max_ratio].
      - Selection is image-level but instance-aware (best effort).
    """
    total = len(samples)
    if total <= 0:
        return set(), 0, {
            "minority_rule_applied": False,
            "minority_class_ids": [],
            "minority_total_instances": 0,
            "minority_val_instances": 0,
        }

    if total == 1:
        return set(), 0, {
            "minority_rule_applied": False,
            "minority_class_ids": [],
            "minority_total_instances": 0,
            "minority_val_instances": 0,
        }

    val_count = max(1, int(total * val_ratio))
    val_count = min(val_count, total)
    rng = random.Random(seed)
    shuffled_indices = list(range(total))
    rng.shuffle(shuffled_indices)
    rotated_sample_flags = [_sample_has_rotated_obb(sample) for sample in samples]
    rotated_indices = [idx for idx in shuffled_indices if rotated_sample_flags[idx]]

    if not orientation_class_enabled:
        selected = []
        selected_set = set()
        if rotated_indices and val_count > 0:
            selected.append(rotated_indices[0])
            selected_set.add(rotated_indices[0])
        for idx in shuffled_indices:
            if len(selected) >= val_count:
                break
            if idx in selected_set:
                continue
            selected.append(idx)
            selected_set.add(idx)
        val_set = set(selected[:val_count])
        return val_set, val_count, {
            "minority_rule_applied": False,
            "minority_class_ids": [],
            "minority_total_instances": 0,
            "minority_val_instances": 0,
            "rotated_real_images_total": int(len(rotated_indices)),
            "rotated_real_images_val": int(sum(1 for idx in val_set if rotated_sample_flags[idx])),
        }

    sample_hists = [
        _sample_class_histogram(
            s,
            orientation_class_enabled=orientation_class_enabled,
            head_id=head_id,
            tail_id=tail_id,
            orientation_policy=orientation_policy,
        )
        for s in samples
    ]

    class_counts = {}
    for hist in sample_hists:
        for class_id, count in hist.items():
            class_counts[class_id] = class_counts.get(class_id, 0) + int(count)

    present_classes = sorted([cid for cid, cnt in class_counts.items() if cnt > 0])
    if not present_classes:
        return set(shuffled_indices[:val_count]), val_count, {
            "minority_rule_applied": False,
            "minority_class_ids": [],
            "minority_total_instances": 0,
            "minority_val_instances": 0,
        }

    min_nonzero = min(class_counts[cid] for cid in present_classes)
    minority_class_ids = sorted([cid for cid in present_classes if class_counts[cid] == min_nonzero])
    minority_total_instances = int(sum(class_counts[cid] for cid in minority_class_ids))

    # Apply minority rule only when minority class is truly scarce.
    minority_rule_applied = bool(
        len(present_classes) >= 2 and min_nonzero <= int(minority_small_cutoff)
    )

    selected = []
    selected_set = set()

    class_presence_observed = {cid: 0 for cid in present_classes}

    def _add_index(idx):
        if idx in selected_set or len(selected) >= val_count:
            return False
        selected.append(idx)
        selected_set.add(idx)
        hist = sample_hists[idx]
        for cid in present_classes:
            if hist.get(cid, 0) > 0:
                class_presence_observed[cid] += int(hist.get(cid, 0))
        return True

    if rotated_indices and val_count > 0:
        _add_index(rotated_indices[0])

    # Image-level minority split: collect all images that contain ≥1 minority class box,
    # then allocate exactly floor(count × minority_max_ratio) of them to val.
    # Images not selected for val are locked into training (excluded from random fill).
    minority_image_indices: list[int] = []
    minority_image_train_set: set[int] = set()
    n_minority_val: int = 0

    if minority_rule_applied:
        minority_image_indices = [
            idx for idx in shuffled_indices
            if any(sample_hists[idx].get(cid, 0) > 0 for cid in minority_class_ids)
        ]
        n_minority_val = int(math.floor(len(minority_image_indices) * float(minority_max_ratio)))
        # Add the first n_minority_val minority images to val.
        for idx in minority_image_indices[:n_minority_val]:
            _add_index(idx)
        # Lock remaining minority images into training (they must not go to val via random fill).
        minority_image_train_set = set(minority_image_indices[n_minority_val:])

    # Enforce class presence in val where capacity allows.
    if len(selected) < val_count:
        for cid in present_classes:
            if len(selected) >= val_count:
                break
            if class_presence_observed.get(cid, 0) > 0:
                continue
            # Do not force minority-class images into val when the 80/20 split
            # allocated 0 val slots for them (i.e. too few minority images to split).
            if minority_rule_applied and cid in minority_class_ids and n_minority_val == 0:
                continue
            candidate = None
            candidate_count = 0
            for idx in shuffled_indices:
                if idx in selected_set:
                    continue
                count = int(sample_hists[idx].get(cid, 0))
                if count > candidate_count:
                    candidate = idx
                    candidate_count = count
            if candidate is not None and candidate_count > 0:
                _add_index(candidate)

    # Fill remaining val slots randomly, skipping images locked into training by the minority rule.
    if len(selected) < val_count:
        for idx in shuffled_indices:
            if len(selected) >= val_count:
                break
            if idx in selected_set:
                continue
            if minority_image_train_set and idx in minority_image_train_set:
                continue
            _add_index(idx)

    val_set = set(selected[:val_count])

    minority_val_instances = 0
    if minority_class_ids:
        for idx in val_set:
            hist = sample_hists[idx]
            for cid in minority_class_ids:
                minority_val_instances += int(hist.get(cid, 0))

    return val_set, val_count, {
        "minority_rule_applied": bool(minority_rule_applied),
        "minority_class_ids": minority_class_ids,
        "minority_total_instances": minority_total_instances,
        "minority_val_instances": int(minority_val_instances),
        "rotated_real_images_total": int(len(rotated_indices)),
        "rotated_real_images_val": int(sum(1 for idx in val_set if rotated_sample_flags[idx])),
    }


# -----------------------------------------------------------------------------
# Synthetic generation from finalized segment crops
# -----------------------------------------------------------------------------

def _collect_finalized_segments(session_dir, anchor_index=None, finalized_images=None):
    """
    Collect finalized SAM2 segments (RGBA).
    Only segments with accepted_by_user=true are used.

    When anchor_index is provided, attempts to attach head/tail anchor points
    (in segment crop coordinates) for orientation-aware synthetic labeling.

    When finalized_images is a frozenset of normalized source image paths,
    segments whose source_image is not in the set are skipped (stale segment
    guard for deleted or reverted images).
    """
    seg_dir = os.path.join(session_dir, "segments")
    if not os.path.isdir(seg_dir):
        return [], {
            "segments_total": 0,
            "segments_with_anchors": 0,
            "segments_missing_anchors": 0,
        }

    segments = []
    with_anchor = 0
    without_anchor = 0
    for fname in sorted(os.listdir(seg_dir)):
        if not fname.endswith("_fg.png"):
            continue
        fg_path = os.path.join(seg_dir, fname)
        base = fname[:-7]  # strip "_fg.png"
        meta_path = os.path.join(seg_dir, f"{base}_meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if not bool(meta.get("accepted_by_user", False)):
                continue
            if meta.get("mask_source") == "rectangle_fallback":
                continue
            if finalized_images is not None:
                if _norm_path(meta.get("source_image", "")) not in finalized_images:
                    continue
        except Exception:
            continue

        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        if fg is None or fg.ndim != 3 or fg.shape[2] != 4:
            continue
        if np.count_nonzero(fg[:, :, 3] > 10) < 20:
            continue
        seg_entry = {"id": base, "rgba": fg}

        # Optional source box metadata from finalized accepted boxes.
        if anchor_index:
            img_key = _norm_path(meta.get("source_image"))
            candidates = anchor_index.get(img_key, [])
            seg_box = _box_xyxy_from_meta_box(meta.get("box"))
            crop_origin = meta.get("crop_origin", [0, 0])
            try:
                cx1 = float(crop_origin[0])
                cy1 = float(crop_origin[1])
            except Exception:
                cx1, cy1 = 0.0, 0.0

            best = None
            best_iou = 0.0
            for cand in candidates:
                iou = _xyxy_iou(seg_box, cand.get("xyxy"))
                if iou > best_iou:
                    best_iou = iou
                    best = cand

            if best is not None and best_iou >= 0.50:
                source_class_id = best.get("class_id")
                if source_class_id is not None:
                    seg_entry["source_class_id"] = int(source_class_id)
                    with_anchor += 1
                else:
                    without_anchor += 1
            else:
                without_anchor += 1

        segments.append(seg_entry)

    return segments, {
        "segments_total": len(segments),
        "segments_with_anchors": with_anchor,
        "segments_missing_anchors": without_anchor,
    }


def _prepare_segment_chip(fg_rgba, pad_ratio=0.2, head_tail_fg=None):
    """
    Tight-crop around alpha mask and add transparent padding.

    Returns:
      chip_rgba, head_tail_chip
      where head_tail_chip uses chip coordinates after tight-crop + pad.
    """
    alpha = fg_rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return fg_rgba.copy(), head_tail_fg

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    tight = fg_rgba[y1:y2, x1:x2].copy()
    h, w = tight.shape[:2]
    pad = max(2, int(max(h, w) * float(pad_ratio)))
    out = np.zeros((h + 2 * pad, w + 2 * pad, 4), dtype=np.uint8)
    out[pad:pad + h, pad:pad + w] = tight

    remapped = None
    if head_tail_fg is not None:
        try:
            (hx, hy), (tx, ty) = head_tail_fg
            remapped = (
                (float(hx) - x1 + pad, float(hy) - y1 + pad),
                (float(tx) - x1 + pad, float(ty) - y1 + pad),
            )
        except Exception:
            remapped = None
    return out, remapped


def _augment_segment_chip(
    chip_rgba,
    rng,
    orientation_schema="invariant",
    scale_range=(0.65, 1.35),
    rot_range=(-60.0, 60.0),
    flip_prob=0.5,
    head_tail_chip=None,
):
    """
    Apply orientation and shape augmentations to one chip.

    Returns:
      aug_rgba, head_tail_aug
    """
    aug = chip_rgba.copy()
    did_flip = False
    points = None
    if head_tail_chip is not None:
        try:
            (hx, hy), (tx, ty) = head_tail_chip
            points = np.array([[float(hx), float(hy)], [float(tx), float(ty)]], dtype=np.float32)
        except Exception:
            points = None

    mirror_axis = None
    if orientation_schema == "directional":
        if rng.random() < flip_prob:
            did_flip = True
            mirror_axis = "horizontal"
            h0, w0 = aug.shape[:2]
            aug = cv2.flip(aug, 1)
            if points is not None:
                points[:, 0] = (w0 - 1) - points[:, 0]
    elif orientation_schema == "axial":
        if rng.random() < flip_prob:
            mirror_axis = "horizontal"
            h0, w0 = aug.shape[:2]
            aug = cv2.flip(aug, 1)
            if points is not None:
                points[:, 0] = (w0 - 1) - points[:, 0]
        if rng.random() < flip_prob:
            mirror_axis = "vertical" if mirror_axis is None else mirror_axis
            h0, _w0 = aug.shape[:2]
            aug = cv2.flip(aug, 0)
            if points is not None:
                points[:, 1] = (h0 - 1) - points[:, 1]
    elif orientation_schema == "invariant":
        if rng.random() < flip_prob:
            mirror_axis = "horizontal"
            h0, w0 = aug.shape[:2]
            aug = cv2.flip(aug, 1)
            if points is not None:
                points[:, 0] = (w0 - 1) - points[:, 0]
        if rng.random() < flip_prob:
            mirror_axis = "vertical" if mirror_axis is None else mirror_axis
            h0, _w0 = aug.shape[:2]
            aug = cv2.flip(aug, 0)
            if points is not None:
                points[:, 1] = (h0 - 1) - points[:, 1]

    # Random scaling
    scale = rng.uniform(scale_range[0], scale_range[1])
    h, w = aug.shape[:2]
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    aug = cv2.resize(aug, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if points is not None:
        sx = float(nw) / max(1.0, float(w))
        sy = float(nh) / max(1.0, float(h))
        points[:, 0] *= sx
        points[:, 1] *= sy

    # Random rotation with expanded canvas to avoid clipping.
    if orientation_schema == "bilateral":
        angle = 180.0 if rng.random() < 0.5 else 0.0
    elif orientation_schema == "axial":
        angle = 180.0 if rng.random() < 0.5 else 0.0
    elif orientation_schema == "invariant":
        angle = rng.uniform(-180.0, 180.0)
    else:
        angle = rng.uniform(rot_range[0], rot_range[1])
    h, w = aug.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    aug = cv2.warpAffine(
        aug,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    if points is not None:
        pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        points = (M @ pts_h.T).T

    # Mild color jitter on foreground only.
    jitter = rng.uniform(0.88, 1.12)
    alpha = aug[:, :, 3] > 0
    if np.any(alpha):
        rgb = aug[:, :, :3].astype(np.float32)
        rgb[alpha] = np.clip(rgb[alpha] * jitter, 0, 255)
        aug[:, :, :3] = rgb.astype(np.uint8)

    head_tail_aug = None
    if points is not None:
        head_tail_aug = (
            (float(points[0, 0]), float(points[0, 1])),
            (float(points[1, 0]), float(points[1, 1])),
        )

    return aug, head_tail_aug, {
        "flipped": did_flip,
        "mirror_axis": mirror_axis,
        "rotation_deg": angle,
        "is_half_turn": abs(abs(angle) - 180.0) <= 1e-6,
    }


def _random_canvas_background(width, height, rng):
    """
    Create a synthetic background canvas with no real-image pixels.
    Seven modes weighted toward spatial complexity so the OBB detector must
    learn to distinguish objects by shape and texture rather than exploiting
    low-entropy contrast between a studio specimen and a flat field.

    Weights: solid=1, linear_gradient=3, radial_gradient=2, multi_gradient=2,
             perlin_noise=3, coarse_noise=2, vignette=1  (total=14)
    """
    # Derive a numpy Generator from the Python rng so array-generation calls work.
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    _MODES = [
        "solid", "linear_gradient", "radial_gradient",
        "multi_gradient", "perlin_noise", "coarse_noise", "vignette",
    ]
    _WEIGHTS = [1, 3, 2, 2, 3, 2, 1]
    _CUMULATIVE = []
    _s = 0
    for w in _WEIGHTS:
        _s += w
        _CUMULATIVE.append(_s)
    r_draw = int(np_rng.integers(0, _s))
    mode = next(m for m, thresh in zip(_MODES, _CUMULATIVE) if r_draw < thresh)

    def _rand_color():
        return np.array([int(np_rng.integers(15, 221)) for _ in range(3)], dtype=np.float32)

    if mode == "solid":
        c = _rand_color()
        return np.full((height, width, 3), c.astype(np.uint8), dtype=np.uint8)

    if mode == "linear_gradient":
        c1, c2 = _rand_color(), _rand_color()
        direction = int(np_rng.integers(0, 4))   # 0=H  1=V  2=diag TL→BR  3=diag TR→BL
        if direction == 0:
            t = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
            bg = c1 * (1 - t) + c2 * t
            bg = np.repeat(bg, height, axis=0)
        elif direction == 1:
            t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
            bg = c1 * (1 - t) + c2 * t
            bg = np.repeat(bg, width, axis=1)
        elif direction == 2:
            tx = np.linspace(0, 1, width, dtype=np.float32)
            ty = np.linspace(0, 1, height, dtype=np.float32)
            t = ((tx[None, :] + ty[:, None]) / 2.0)[:, :, None]
            bg = c1 * (1 - t) + c2 * t
        else:
            tx = np.linspace(1, 0, width, dtype=np.float32)
            ty = np.linspace(0, 1, height, dtype=np.float32)
            t = ((tx[None, :] + ty[:, None]) / 2.0)[:, :, None]
            bg = c1 * (1 - t) + c2 * t
        return np.clip(bg, 0, 255).astype(np.uint8)

    if mode == "radial_gradient":
        c1, c2 = _rand_color(), _rand_color()
        cx = float(np_rng.uniform(0.2, 0.8)) * width
        cy = float(np_rng.uniform(0.2, 0.8)) * height
        xs = np.arange(width, dtype=np.float32) - cx
        ys = np.arange(height, dtype=np.float32) - cy
        dist = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
        t = np.clip(dist / (np.sqrt(width ** 2 + height ** 2) / 2.0), 0, 1)[:, :, None]
        bg = c1 * (1 - t) + c2 * t
        return np.clip(bg, 0, 255).astype(np.uint8)

    if mode == "multi_gradient":
        n_stops = int(np_rng.integers(3, 5))
        stops = [_rand_color() for _ in range(n_stops)]
        horizontal = bool(np_rng.random() < 0.5)
        size = width if horizontal else height
        positions = [min(int(i * size / (n_stops - 1)), size - 1) for i in range(n_stops)]
        line = np.zeros((size, 3), dtype=np.float32)
        for i in range(n_stops - 1):
            s, e = positions[i], positions[i + 1]
            span = max(1, e - s)
            t = np.linspace(0, 1, span, dtype=np.float32)[:, None]
            line[s:e] = stops[i] * (1 - t) + stops[i + 1] * t
        line[positions[-1]] = stops[-1]
        if horizontal:
            bg = np.repeat(line[None, :, :], height, axis=0)
        else:
            bg = np.repeat(line[:, None, :], width, axis=1)
        return np.clip(bg, 0, 255).astype(np.uint8)

    if mode == "perlin_noise":
        # Multi-octave fractal noise: sum 4 octaves of bilinearly-upsampled
        # random patches (no scipy required).
        base = _rand_color()
        acc = np.zeros((height, width, 3), dtype=np.float32)
        amplitude = 55.0
        for octave in range(4):
            divisor = 2 ** (octave + 3)
            h_s = max(2, height // divisor)
            w_s = max(2, width // divisor)
            patch = np_rng.integers(-64, 65, size=(h_s, w_s, 3)).astype(np.float32)
            up = cv2.resize(patch, (width, height), interpolation=cv2.INTER_LINEAR)
            acc += up * (amplitude / (2 ** octave))
        acc = acc / (amplitude * 2.0) + base[None, None, :]
        return np.clip(acc, 0, 255).astype(np.uint8)

    if mode == "coarse_noise":
        # Large random patches resized and blurred → blotchy low-frequency texture.
        base = int(np_rng.integers(30, 201))
        patch_px = int(np_rng.integers(16, 65))
        hp = max(2, (height + patch_px - 1) // patch_px)
        wp = max(2, (width + patch_px - 1) // patch_px)
        lo, hi = max(0, base - 60), min(255, base + 60)
        patches = np_rng.integers(lo, hi + 1, size=(hp, wp, 3)).astype(np.uint8)
        up = cv2.resize(patches, (width, height), interpolation=cv2.INTER_LINEAR)
        ksize = (patch_px | 1, patch_px | 1)
        return cv2.GaussianBlur(up, ksize, patch_px / 3.0)

    # vignette — bright centre fading to dark edge with variable gamma falloff
    c_center = _rand_color()
    c_edge = _rand_color() * float(np_rng.uniform(0.05, 0.45))
    xs = np.linspace(-1, 1, width, dtype=np.float32)
    ys = np.linspace(-1, 1, height, dtype=np.float32)
    dist = np.clip(np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2), 0, 1)
    t = (dist ** float(np_rng.uniform(0.6, 2.2)))[:, :, None]
    bg = c_center * (1 - t) + c_edge * t
    return np.clip(bg, 0, 255).astype(np.uint8)



def _overlaps_with_gap(candidate, placed, min_gap_px=8):
    """
    True if candidate overlaps any existing box, considering minimum gap.
    """
    cx1, cy1, cx2, cy2 = candidate
    for px1, py1, px2, py2 in placed:
        if (
            (cx2 + min_gap_px) <= px1
            or (px2 + min_gap_px) <= cx1
            or (cy2 + min_gap_px) <= py1
            or (py2 + min_gap_px) <= cy1
        ):
            continue
        return True
    return False


def _place_chip(canvas, chip_rgba, placed_boxes, rng, min_gap_px=8, max_attempts=80):
    """
    Place an RGBA chip on canvas with strict non-overlap.
    Returns {"bbox": (x1, y1, x2, y2), "offset": (x, y)} or None.
    """
    h, w = chip_rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    if h <= 1 or w <= 1 or h >= ch or w >= cw:
        return None

    alpha = chip_rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None

    tight_x1 = int(xs.min())
    tight_y1 = int(ys.min())
    tight_x2 = int(xs.max()) + 1
    tight_y2 = int(ys.max()) + 1

    for _ in range(max_attempts):
        x = rng.randint(0, cw - w)
        y = rng.randint(0, ch - h)
        cand = (
            x + tight_x1,
            y + tight_y1,
            x + tight_x2,
            y + tight_y2,
        )
        if _overlaps_with_gap(cand, placed_boxes, min_gap_px=min_gap_px):
            continue

        roi = canvas[y:y + h, x:x + w].astype(np.float32)
        fg = chip_rgba[:, :, :3].astype(np.float32)
        # Feather the binary SAM2 mask edge to avoid hard dark halos where the
        # object silhouette meets the gradient background.
        alpha_raw = chip_rgba[:, :, 3].astype(np.float32)
        alpha_soft = cv2.GaussianBlur(alpha_raw, (0, 0), sigmaX=2.0, sigmaY=2.0)
        a = (alpha_soft / 255.0)[:, :, None]
        blended = fg * a + roi * (1.0 - a)
        canvas[y:y + h, x:x + w] = blended.astype(np.uint8)
        placed_boxes.append(cand)
        return {"bbox": cand, "offset": (x, y)}

    return None


# -----------------------------------------------------------------------------
# OBB dataset export for YOLOv8-OBB training
# -----------------------------------------------------------------------------

def _resolve_obb_rotation_policy(session_dir, fallback_mode="invariant"):
    """
    Resolve the session's effective OBB augmentation policy.

    OBB export intentionally reuses the shared session rotation_range semantics.
    """
    mode, _orientation_policy, augmentation_policy, profile = resolve_session_augmentation_profile(
        session_dir,
        engine="cnn",
        fallback_mode=fallback_mode,
    )
    raw_range = profile.get("rotation_range", (-15.0, 15.0))
    if isinstance(raw_range, list):
        raw_range = tuple(raw_range)
    if not isinstance(raw_range, tuple) or len(raw_range) != 2:
        raw_range = (-15.0, 15.0)
    lo = _safe_float(raw_range[0], -15.0)
    hi = _safe_float(raw_range[1], 15.0)
    if lo is None or hi is None:
        lo, hi = -15.0, 15.0
    if lo > hi:
        lo, hi = hi, lo
    return {
        "mode": mode,
        "gravity_aligned": bool(augmentation_policy.get("gravity_aligned", True)),
        "rotation_range": (float(lo), float(hi)),
    }


def _compute_base_class_id(source_class_id, orientation_schema):
    """Derive base class_id from stored source-box metadata before augmentation."""
    if orientation_schema not in {"directional", "bilateral"}:
        return 0
    try:
        return int(source_class_id)
    except Exception:
        return 0


def _apply_schema_class_transform(base_class_id, aug_info, orientation_schema):
    class_id = base_class_id
    if orientation_schema == "directional":
        if aug_info.get("mirror_axis") == "horizontal":
            class_id = 1 - class_id
    elif orientation_schema == "bilateral":
        if bool(aug_info.get("is_half_turn")):
            class_id = 1 - class_id
    elif orientation_schema in {"axial", "invariant"}:
        class_id = 0
    else:
        class_id = 0
    return class_id


def _compute_obb_from_placed_chip(aug_rgba, offset_x, offset_y, canvas_w, canvas_h, force_axis_aligned=False):
    """Return 8 normalized corner coords (x1 y1 x2 y2 x3 y3 x4 y4) for OBB label."""
    alpha = aug_rgba[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) < 4:
        return None
    pts = np.column_stack([
        xs.astype(np.float32) + offset_x,
        ys.astype(np.float32) + offset_y,
    ])
    if force_axis_aligned:
        x1 = float(np.min(pts[:, 0]))
        x2 = float(np.max(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        y2 = float(np.max(pts[:, 1]))
        corners = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    else:
        rect = cv2.minAreaRect(pts)
        corners = cv2.boxPoints(rect)
    cw_f = float(canvas_w)
    ch_f = float(canvas_h)
    result = []
    for px, py in corners:
        result.extend([
            _clamp(float(px) / cw_f, 0.0, 1.0),
            _clamp(float(py) / ch_f, 0.0, 1.0),
        ])
    return result


def _mask_outline_from_placed_chip(aug_rgba, offset_x, offset_y, max_points=128):
    """
    Derive a mask_outline polygon (in canvas coordinates) from the augmented
    chip's alpha channel.  Returns a list of [x, y] floats, or None.
    """
    if aug_rgba is None or aug_rgba.ndim < 3 or aug_rgba.shape[2] < 4:
        return None
    alpha = aug_rgba[:, :, 3]
    mask_u8 = (alpha > 10).astype(np.uint8)
    if int(mask_u8.sum()) < 20:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8:
        return None
    perimeter = cv2.arcLength(contour, True)
    eps = max(1.0, 0.003 * float(perimeter))
    approx = cv2.approxPolyDP(contour, eps, True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    return [[float(p[0] + offset_x), float(p[1] + offset_y)] for p in pts]


def _generate_synthetic_obb_images(
    session_dir,
    out_dir,
    positives,
    orientation_schema="invariant",
    head_id=None,
    tail_id=None,
    n_per_segment=15,
    max_objects=4,
    max_images=None,
    seed=0,
    split="train",
    manifest=None,
    rotation_range=(-15.0, 15.0),
):
    """
    Generate synthetic OBB train images from finalized segment crops.
    Outputs 8-point OBB polygon labels for YOLOv8-OBB format.
    Uses schema-aware class_id mapping. When gravity_aligned=True, rotation is
    clamped to ±15° to match gravity-constrained imaging setups.
    """
    # rotation_range is the resolved session policy and overrides the historical
    # gravity-aligned clamp described above.
    orientation_policy = _load_session_orientation_policy(session_dir)
    orientation_class_enabled, _class_names, _resolved_mode, _bilateral_axis = _resolve_detector_class_config(
        orientation_policy,
        fallback_mode=orientation_schema,
    )
    source_box_index = _build_source_box_index(
        positives or [],
        orientation_class_enabled=orientation_class_enabled,
        head_id=head_id,
        tail_id=tail_id,
        orientation_policy=orientation_policy,
    ) if orientation_class_enabled else {}
    finalized_imgs = frozenset(_norm_path(s["image_path"]) for s in (positives or []))
    segments, seg_stats = _collect_finalized_segments(
        session_dir,
        anchor_index=source_box_index if orientation_class_enabled else None,
        finalized_images=finalized_imgs if finalized_imgs else None,
    )
    if not segments:
        return {"num_generated": 0, "num_instances_generated": 0, **seg_stats}

    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    rng = random.Random(seed)
    n_generated = 0
    n_instances_generated = 0
    synthetic_class_hist = {}
    total_iters = max(1, int(n_per_segment)) * len(segments)

    for i in range(total_iters):
        if max_images is not None and n_generated >= int(max_images):
            break

        canvas_size = rng.choice([768, 896, 1024])
        bg = _random_canvas_background(canvas_size, canvas_size, rng)
        labels = []
        placed = []
        object_manifest = []

        if len(segments) > 1:
            target_objects = rng.randint(2, max(2, int(max_objects)))
        else:
            target_objects = 1

        seg_indices = list(range(len(segments)))
        rng.shuffle(seg_indices)
        chosen = seg_indices[: min(target_objects, len(seg_indices))]
        while len(chosen) < target_objects:
            chosen.append(rng.choice(seg_indices))

        for seg_idx in chosen:
            raw = segments[seg_idx]["rgba"]
            source_class_id = segments[seg_idx].get("source_class_id")
            if orientation_class_enabled and source_class_id is None:
                continue

            chip, head_tail_chip = _prepare_segment_chip(
                raw,
                pad_ratio=rng.uniform(0.12, 0.28),
                head_tail_fg=None,
            )

            base_class_id = _compute_base_class_id(source_class_id, orientation_schema)

            aug, head_tail_aug, aug_info = _augment_segment_chip(
                chip,
                rng,
                orientation_schema=orientation_schema,
                rot_range=rotation_range,
                head_tail_chip=head_tail_chip,
            )

            class_id = _apply_schema_class_transform(base_class_id, aug_info, orientation_schema)

            ch, cw = bg.shape[:2]
            ah, aw = aug.shape[:2]
            max_frac = 0.45
            if aw >= int(cw * max_frac) or ah >= int(ch * max_frac):
                scale = min((cw * max_frac) / max(aw, 1), (ch * max_frac) / max(ah, 1))
                nw = max(8, int(round(aw * scale)))
                nh = max(8, int(round(ah * scale)))
                aug = cv2.resize(aug, (nw, nh), interpolation=cv2.INTER_LINEAR)

            placement = _place_chip(bg, aug, placed, rng, min_gap_px=8, max_attempts=100)
            if placement is None:
                continue

            offset_x, offset_y = placement["offset"]
            obb_pts = _compute_obb_from_placed_chip(
                aug,
                offset_x,
                offset_y,
                cw,
                ch,
                force_axis_aligned=orientation_schema == "invariant",
            )
            if obb_pts is None:
                continue

            mask_outline_canvas = _mask_outline_from_placed_chip(
                aug, offset_x, offset_y
            )

            labels.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in obb_pts))
            synthetic_class_hist[int(class_id)] = synthetic_class_hist.get(int(class_id), 0) + 1
            object_manifest.append({
                "split": split,
                "image": f"__synth_obb_{i:05d}.jpg",
                "segment_id": segments[seg_idx]["id"],
                "class_id": int(class_id),
                "obb_pts": obb_pts,
                **({"mask_outline": mask_outline_canvas} if mask_outline_canvas else {}),
            })

        min_required = 2 if len(segments) > 1 else 1
        if len(labels) < min_required:
            continue
        if max_images is not None and n_generated >= int(max_images):
            break

        synth_name = f"__synth_obb_{i:05d}"
        img_out = os.path.join(img_dir, f"{synth_name}.jpg")
        lbl_out = os.path.join(lbl_dir, f"{synth_name}.txt")
        safe_imwrite(img_out, bg)
        with open(lbl_out, "w", encoding="utf-8") as f:
            f.write("\n".join(labels) + "\n")
        n_generated += 1
        n_instances_generated += len(labels)
        if isinstance(manifest, list):
            manifest.extend(object_manifest)

    return {
        "num_generated": n_generated,
        "num_instances_generated": n_instances_generated,
        "class_histogram": synthetic_class_hist,
        **seg_stats,
    }


def export_obb_dataset(
    session_dir,
    val_ratio=0.2,
    seed=42,
    generate_synthetic=True,
    orientation_schema="invariant",
    progress_callback=None,
):
    """
    Export session annotations to YOLOv8-OBB format.

    Boxes must provide valid obbCorners in canonical [LT, RT, RB, LB] order:
      class_id x1 y1 x2 y2 x3 y3 x4 y4  (all normalized)

    Args:
        generate_synthetic: When False, skip the SAM2-based synthetic augmentation
            step entirely.  Set to False when SAM2 is unavailable (CPU-only systems)
            to avoid edge-artifact poisoning from low-quality crops.

    Writes to session_dir/obb_dataset/ with dataset.yaml.
    Returns {"ok": True, "yaml_path": ..., "num_images": ..., "num_boxes": ...}
    """
    labels_dir = os.path.join(session_dir, "labels")
    images_dir = os.path.join(session_dir, "images")

    if not os.path.isdir(labels_dir):
        return {"ok": False, "error": f"Labels directory not found: {labels_dir}"}
    if not os.path.isdir(images_dir):
        return {"ok": False, "error": f"Images directory not found: {images_dir}"}

    def report_progress(message, percent, details=None):
        if not callable(progress_callback):
            return
        try:
            progress_callback(message, percent, details)
        except Exception:
            pass

    head_id, tail_id = _load_head_tail_ids(session_dir)
    orientation_policy = _load_session_orientation_policy(session_dir)
    finalized_set = _load_finalized_filenames(session_dir)

    out_dir = os.path.join(session_dir, "obb_dataset")
    _reset_output_dataset_dir(out_dir)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    # Gather all finalized samples
    samples = []
    label_files = [fname for fname in sorted(os.listdir(labels_dir)) if fname.endswith(".json")]
    total_label_files = max(1, len(label_files))
    report_progress("Scanning finalized annotations...", 5.2, {"phase": "scan_labels", "current": 0, "total": total_label_files})
    for idx, fname in enumerate(label_files, start=1):
        label_path = os.path.join(labels_dir, fname)
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        image_filename = data.get("imageFilename", "")
        if not image_filename:
            continue

        is_finalized, boxes, _ = _get_finalized_boxes(data, image_filename, finalized_set)
        if not is_finalized or not boxes:
            continue

        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            base = os.path.splitext(image_filename)[0]
            resolved = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                candidate = os.path.join(images_dir, base + ext)
                if os.path.exists(candidate):
                    resolved = candidate
                    image_filename = base + ext
                    break
            if not resolved:
                continue
            image_path = resolved

        samples.append({"image_path": image_path, "image_filename": image_filename, "boxes": boxes})
        if idx == total_label_files or idx % 25 == 0:
            report_progress(
                "Scanning finalized annotations...",
                5.2 + (1.2 * (idx / total_label_files)),
                {"phase": "scan_labels", "current": idx, "total": total_label_files},
            )

    if not samples:
        return {"ok": False, "error": "No finalized samples with OBB annotations found"}

    orientation_class_enabled, class_names, resolved_mode, _bilateral_axis = _resolve_detector_class_config(
        orientation_policy,
        fallback_mode=orientation_schema,
    )

    # Split train/val — stratified when multiple orientation classes are present so that
    # val always contains at least one sample from each class.
    val_set, _val_count, split_stats = _select_obb_val_indices(
        samples,
        val_ratio=val_ratio,
        seed=seed,
        orientation_class_enabled=orientation_class_enabled,
        head_id=head_id,
        tail_id=tail_id,
        orientation_policy=orientation_policy,
        minority_small_cutoff=20,
        minority_target_ratio=0.20,
        minority_min_ratio=0.00,
        minority_max_ratio=0.20,
    )
    warnings = []
    rotated_total = int(split_stats.get("rotated_real_images_total", 0))
    rotated_val = int(split_stats.get("rotated_real_images_val", 0))
    if rotated_total == 0:
        warnings.append(
            "No rotated real OBB samples were found; validation angle metrics and preview OBBs will be weak."
        )
    elif rotated_total < 2:
        warnings.append(
            "Very few rotated real OBB samples were found; validation angle metrics may be unstable."
        )
    elif rotated_val == 0:
        warnings.append(
            "Validation split has no rotated real OBB samples; angle validation coverage is insufficient."
        )

    num_boxes = 0
    num_images = 0
    real_class_hist = {}
    total_samples = max(1, len(samples))
    report_progress("Writing OBB dataset files...", 6.5, {"phase": "write_real", "current": 0, "total": total_samples})

    for i, sample in enumerate(samples):
        split = "val" if i in val_set else "train"

        img = safe_imread(sample["image_path"])
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        if img_w == 0 or img_h == 0:
            continue

        dest_img = os.path.join(out_dir, "images", split, sample["image_filename"])
        if not os.path.exists(dest_img):
            shutil.copy2(sample["image_path"], dest_img)

        label_basename = os.path.splitext(sample["image_filename"])[0] + ".txt"
        label_path = os.path.join(out_dir, "labels", split, label_basename)

        lines = []
        for box in sample["boxes"]:
            # Determine class_id from orientation (set by frontend toggle)
            class_id = 0
            if orientation_class_enabled:
                class_id = _resolve_obb_class_id(
                    box,
                    orientation_class_enabled=orientation_class_enabled,
                    head_id=head_id,
                    tail_id=tail_id,
                    orientation_policy=orientation_policy,
                )
                real_class_hist[int(class_id)] = real_class_hist.get(int(class_id), 0) + 1

            obb_corners = box.get("obbCorners") or box.get("obb_corners")
            if not (obb_corners and len(obb_corners) == 4):
                warnings.append(
                    f"{sample['image_filename']}: skipped box without valid OBB corners; rerun session OBB migration or reimport labels."
                )
                continue

            pts = []
            for px, py in obb_corners:
                pts.extend([
                    _clamp(float(px) / img_w, 0.0, 1.0),
                    _clamp(float(py) / img_h, 0.0, 1.0),
                ])
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in pts))
            num_boxes += 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        num_images += 1
        if (i + 1) == total_samples or (i + 1) % 20 == 0:
            report_progress(
                "Writing OBB dataset files...",
                6.5 + (1.5 * ((i + 1) / total_samples)),
                {"phase": "write_real", "current": i + 1, "total": total_samples},
            )

    # Write dataset.yaml
    nc = 2 if orientation_class_enabled else 1
    names = class_names if orientation_class_enabled else ["specimen"]
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    yaml_lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        f"nc: {nc}",
        f"names: {names}",
        "task: obb",
    ]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")
    report_progress("Finalizing OBB dataset manifest...", 8.2, {"phase": "write_yaml"})

    # --- Synthetic data augmentation ---
    resolved_policy = _resolve_obb_rotation_policy(session_dir, fallback_mode=resolved_mode)
    orientation_schema = resolved_policy["mode"]
    rotation_range = resolved_policy["rotation_range"]
    SYNTHETIC_RATIO = 1
    n_real_train = sum(1 for i in range(len(samples)) if i not in val_set)
    max_synth = max(0, int(n_real_train * SYNTHETIC_RATIO))

    synth_stats = {"num_generated": 0, "num_instances_generated": 0}
    if generate_synthetic and max_synth > 0:
        report_progress(
            "Checking synthetic augmentation inputs...",
            8.4,
            {"phase": "synthetic_prepare", "current": 0, "total": max_synth},
        )
        positives = [s for i, s in enumerate(samples) if i not in val_set]
        synth_manifest = []
        synth_stats = _generate_synthetic_obb_images(
            session_dir=session_dir,
            out_dir=out_dir,
            positives=positives,
            orientation_schema=orientation_schema,
            head_id=head_id,
            tail_id=tail_id,
            n_per_segment=15,
            max_objects=4,
            max_images=max_synth,
            seed=seed + 1,
            split="train",
            manifest=synth_manifest,
            rotation_range=rotation_range,
        )
        if orientation_class_enabled and int(synth_stats.get("segments_missing_anchors", 0)) > 0:
            warnings.append(
                "Some finalized segments were skipped for synthetic detector augmentation because they were missing reliable stored orientation metadata."
            )
        synth_hist = {
            int(k): int(v) for k, v in dict(synth_stats.get("class_histogram", {})).items()
        }
        if len([cid for cid, count in real_class_hist.items() if count > 0]) >= 2:
            if len([cid for cid, count in synth_hist.items() if count > 0]) < 2:
                warnings.append(
                    "Synthetic OBB augmentation produced incomplete class coverage relative to the real finalized dataset."
                )
        manifest_path = os.path.join(out_dir, "synth_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(synth_manifest, f, indent=2)
        report_progress(
            "Synthetic augmentation ready.",
            8.9,
            {
                "phase": "synthetic_done",
                "current": int(synth_stats.get("num_generated", 0)),
                "total": max_synth,
            },
        )
    elif not generate_synthetic:
        import logging as _logging
        _logging.getLogger(__name__).info(
            "Synthetic OBB augmentation skipped (generate_synthetic=False). "
            "Dataset will use real annotated images only."
        )
        report_progress("Synthetic augmentation skipped.", 8.9, {"phase": "synthetic_skipped"})

    return {
        "ok": True,
        "yaml_path": yaml_path,
        "num_images": num_images,
        "num_boxes": num_boxes,
        "synthetic": synth_stats,
        "minority_rule_applied": bool(split_stats.get("minority_rule_applied", False)),
        "minority_class_ids": list(split_stats.get("minority_class_ids", [])),
        "minority_total_instances": int(split_stats.get("minority_total_instances", 0)),
        "minority_val_instances": int(split_stats.get("minority_val_instances", 0)),
        "rotated_real_images_total": rotated_total,
        "rotated_real_images_val": rotated_val,
        "warnings": warnings,
        "real_class_histogram": real_class_hist,
        "synthetic_class_histogram": synth_stats.get("class_histogram", {}),
    }
