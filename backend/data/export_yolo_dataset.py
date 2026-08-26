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
import copy
import hashlib
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
from bv_utils.orientation_utils import (
    require_explicit_orientation_policy,
    resolve_session_augmentation_profile,
)


OBB_SPLIT_ASSIGNMENTS_VERSION = 2
OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION = 1
OBB_EXPORT_MANIFEST_VERSION = 4
OBB_DIRECTIONAL_VALIDATION_MIRROR_VERSION = 1
OBB_CLASS_BALANCED_TRAINING_DERIVATIVE_VERSION = 1
MAX_REAL_OBB_PADDING_RATIO = 2.0
MAX_REAL_OBB_PADDING_PIXELS = 8192
MAX_REAL_OBB_RIGHT_ANGLE_COSINE = 0.10
MAX_REAL_OBB_PARALLEL_SINE = 0.10
MAX_REAL_OBB_OPPOSITE_EDGE_REL_DELTA = 0.05
MAX_REAL_OBB_OPPOSITE_EDGE_ABS_DELTA = 2.0


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


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sample_identity(image_path, image_filename, label_data=None):
    """
    Return a stable sample id and leakage-group id.

    The sample id combines image content and filename, so annotation edits do not
    reshuffle an existing image. Exact image duplicates share a default group id
    and therefore cannot leak across train/val. Importers may provide an explicit
    source group in label metadata to keep related captures together as well.
    """
    content_sha256 = _sha256_file(image_path)
    normalized_filename = str(image_filename or "").replace("\\", "/").strip().lower()
    sample_material = f"obb-sample-v1\0{content_sha256}\0{normalized_filename}"
    sample_id = hashlib.sha256(sample_material.encode("utf-8")).hexdigest()

    explicit_group = None
    containers = [label_data]
    if isinstance(label_data, dict):
        containers.extend(
            label_data.get(key)
            for key in ("metadata", "provenance", "source")
            if isinstance(label_data.get(key), dict)
        )
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in (
            "sourceGroup",
            "source_group",
            "groupId",
            "group_id",
            "sourceImage",
            "source_image",
        ):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                explicit_group = value.replace("\\", "/").strip().lower()
                break
        if explicit_group:
            break

    if explicit_group:
        group_material = f"obb-source-group-v1\0{explicit_group}"
        group_id = hashlib.sha256(group_material.encode("utf-8")).hexdigest()
    else:
        # Exact duplicates are the minimum grouping BioVision can infer safely.
        group_id = content_sha256
    return sample_id, group_id, content_sha256


def _assign_unique_obb_export_names(samples):
    """Assign deterministic image/label names with a one-to-one YOLO stem.

    YOLO pairs an image and label by basename, ignoring the image extension.
    Consequently, exporting both ``specimen.jpg`` and ``specimen.png`` as-is
    aliases both images to ``specimen.txt``.  Preserve historical names when a
    stem is unique, but suffix every member of a colliding stem with its stable
    sample id so neither bytes nor geometry can overwrite the other sample.
    """
    stem_counts = {}
    source_parts = []
    for sample in samples:
        source_name = os.path.basename(str(sample.get("image_filename") or ""))
        stem, extension = os.path.splitext(source_name)
        if not stem or not extension:
            raise ValueError(
                f"OBB source image '{source_name}' must have a filename stem and extension."
            )
        normalized_stem = stem.casefold()
        stem_counts[normalized_stem] = stem_counts.get(normalized_stem, 0) + 1
        source_parts.append((stem, extension, normalized_stem))

    # Reserve unchanged stems first.  This also prevents a generated suffixed
    # stem from colliding with a real source that already uses that spelling.
    used_stems = {
        stem.casefold()
        for stem, _extension, normalized_stem in source_parts
        if stem_counts[normalized_stem] == 1
    }
    for sample, (stem, extension, normalized_stem) in zip(samples, source_parts):
        export_stem = stem
        if stem_counts[normalized_stem] > 1:
            sample_id = str(sample.get("sample_id") or "").strip().lower()
            if not sample_id:
                raise ValueError("OBB samples require a stable sample id before naming export files.")
            suffix_length = 12
            while True:
                export_stem = f"{stem}--{sample_id[:suffix_length]}"
                normalized_export_stem = export_stem.casefold()
                if normalized_export_stem not in used_stems:
                    break
                suffix_length += 4
                if suffix_length > len(sample_id):
                    raise ValueError(
                        f"Could not assign a unique deterministic export basename for '{sample['image_filename']}'."
                    )
            used_stems.add(normalized_export_stem)

        sample["export_image_filename"] = f"{export_stem}{extension}"
        sample["export_label_filename"] = f"{export_stem}.txt"


def _atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _validate_real_obb_corners(raw_corners, image_filename="image"):
    """Validate and return four finite corners that describe a rectangle."""
    if not isinstance(raw_corners, (list, tuple)) or len(raw_corners) != 4:
        raise ValueError(
            f"{image_filename}: an OBB must contain exactly four corners in "
            "canonical [LT, RT, RB, LB] order."
        )

    points = []
    for index, point in enumerate(raw_corners):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(
                f"{image_filename}: OBB corner {index} is not a two-coordinate point."
            )
        x = _safe_float(point[0], None)
        y = _safe_float(point[1], None)
        if x is None or y is None:
            raise ValueError(
                f"{image_filename}: OBB corner {index} contains a non-finite coordinate; "
                "repair or redraw this box before training."
            )
        points.append((float(x), float(y)))

    edge_vectors = []
    edge_lengths = []
    turns = []
    for index in range(4):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % 4]
        x3, y3 = points[(index + 2) % 4]
        edge = (x2 - x1, y2 - y1)
        edge_vectors.append(edge)
        edge_lengths.append(math.hypot(*edge))
        turns.append((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2))

    if min(edge_lengths) <= 1e-6:
        raise ValueError(
            f"{image_filename}: OBB has a zero-length edge; repair or redraw this box before training."
        )
    if any(abs(turn) <= 1e-8 for turn in turns) or not (
        all(turn > 0 for turn in turns) or all(turn < 0 for turn in turns)
    ):
        raise ValueError(
            f"{image_filename}: OBB corners are self-intersecting, concave, or out of canonical order; "
            "repair or redraw this box before training."
        )

    twice_area = abs(
        sum(
            points[index][0] * points[(index + 1) % 4][1]
            - points[(index + 1) % 4][0] * points[index][1]
            for index in range(4)
        )
    )
    if twice_area <= 1e-6:
        raise ValueError(
            f"{image_filename}: OBB area is effectively zero; repair or redraw this box before training."
        )

    # OBB labels must describe an oriented rectangle, not merely an arbitrary
    # convex quadrilateral. Allow small deviations caused by integer-rounded UI
    # coordinates while rejecting trapezoids and sheared parallelograms.
    for index in range(4):
        edge = edge_vectors[index]
        adjacent = edge_vectors[(index + 1) % 4]
        cosine = abs(edge[0] * adjacent[0] + edge[1] * adjacent[1]) / (
            edge_lengths[index] * edge_lengths[(index + 1) % 4]
        )
        if cosine > MAX_REAL_OBB_RIGHT_ANGLE_COSINE:
            raise ValueError(
                f"{image_filename}: OBB is not rectangular (adjacent edges are not perpendicular); "
                "repair or redraw this box before training."
            )

    for first, opposite in ((0, 2), (1, 3)):
        first_edge = edge_vectors[first]
        opposite_edge = edge_vectors[opposite]
        parallel_sine = abs(
            first_edge[0] * opposite_edge[1] - first_edge[1] * opposite_edge[0]
        ) / (edge_lengths[first] * edge_lengths[opposite])
        length_delta = abs(edge_lengths[first] - edge_lengths[opposite])
        length_tolerance = max(
            MAX_REAL_OBB_OPPOSITE_EDGE_ABS_DELTA,
            MAX_REAL_OBB_OPPOSITE_EDGE_REL_DELTA
            * max(edge_lengths[first], edge_lengths[opposite]),
        )
        if parallel_sine > MAX_REAL_OBB_PARALLEL_SINE or length_delta > length_tolerance:
            raise ValueError(
                f"{image_filename}: OBB is not rectangular (opposite edges do not match); "
                "repair or redraw this box before training."
            )
    return points


def _real_obb_padding(boxes, image_width, image_height, image_filename="image"):
    points = []
    for box in boxes:
        raw_corners = box.get("obbCorners") or box.get("obb_corners")
        points.extend(_validate_real_obb_corners(raw_corners, image_filename))

    if not points:
        return {"left": 0, "top": 0, "right": 0, "bottom": 0}

    min_x = min(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_x = max(point[0] for point in points)
    max_y = max(point[1] for point in points)
    padding = {
        "left": max(0, int(math.ceil(-min_x))),
        "top": max(0, int(math.ceil(-min_y))),
        "right": max(0, int(math.ceil(max_x - float(image_width)))),
        "bottom": max(0, int(math.ceil(max_y - float(image_height)))),
    }

    limits = {
        "left": max(64, int(math.ceil(image_width * MAX_REAL_OBB_PADDING_RATIO))),
        "right": max(64, int(math.ceil(image_width * MAX_REAL_OBB_PADDING_RATIO))),
        "top": max(64, int(math.ceil(image_height * MAX_REAL_OBB_PADDING_RATIO))),
        "bottom": max(64, int(math.ceil(image_height * MAX_REAL_OBB_PADDING_RATIO))),
    }
    unreasonable = [
        side
        for side, amount in padding.items()
        if amount > MAX_REAL_OBB_PADDING_PIXELS or amount > limits[side]
    ]
    if unreasonable:
        detail = ", ".join(f"{side}={padding[side]}px" for side in unreasonable)
        raise ValueError(
            f"{image_filename}: OBB coordinates require unreasonable canvas padding ({detail}). "
            "Verify that OBB corners use this image's pixel coordinate system before training."
        )
    return padding


def _translate_real_box_geometry(box, offset_x, offset_y):
    translated = copy.deepcopy(box)
    raw_corners = translated.get("obbCorners") or translated.get("obb_corners")
    if raw_corners is not None:
        translated["obbCorners"] = [
            [float(point[0]) + offset_x, float(point[1]) + offset_y]
            for point in raw_corners
        ]
        translated.pop("obb_corners", None)

    if _safe_float(translated.get("left"), None) is not None:
        translated["left"] = float(translated["left"]) + offset_x
    if _safe_float(translated.get("top"), None) is not None:
        translated["top"] = float(translated["top"]) + offset_y

    for landmark in translated.get("landmarks", []):
        if not isinstance(landmark, dict) or landmark.get("isSkipped"):
            continue
        x = _safe_float(landmark.get("x"), None)
        y = _safe_float(landmark.get("y"), None)
        if x is not None and y is not None:
            landmark["x"] = x + offset_x
            landmark["y"] = y + offset_y
    return translated


def _prepare_real_sample_for_export(image, boxes, image_filename="image"):
    """
    Pad a real image when required and translate all box/landmark geometry.

    In-bounds images are returned unchanged so their historical byte-for-byte
    copy path and normalized labels remain intact.
    """
    image_height, image_width = image.shape[:2]
    padding = _real_obb_padding(boxes, image_width, image_height, image_filename)
    offset_x = int(padding["left"])
    offset_y = int(padding["top"])
    padded = any(int(value) > 0 for value in padding.values())
    if padded:
        # Match Ultralytics' neutral letterbox fill so padding does not create a
        # unique black edge cue that appears only on repaired annotations.
        border_value = (114, 114, 114)
        export_image = cv2.copyMakeBorder(
            image,
            padding["top"],
            padding["bottom"],
            padding["left"],
            padding["right"],
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
    else:
        export_image = image

    translated_boxes = [
        _translate_real_box_geometry(box, offset_x, offset_y) for box in boxes
    ]
    export_height, export_width = export_image.shape[:2]
    transform = {
        "type": "translation_with_canvas_padding" if padded else "identity",
        "offset_x": offset_x,
        "offset_y": offset_y,
        "padding": {key: int(value) for key, value in padding.items()},
        "original_width": int(image_width),
        "original_height": int(image_height),
        "export_width": int(export_width),
        "export_height": int(export_height),
        "border_mode": "constant_114" if padded else None,
    }
    return export_image, translated_boxes, transform


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


def _resolve_bilateral_class_axis(orientation_policy):
    return "vertical_obb"


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

    if head_id is None or tail_id is None:
        return None

    head_lm = next(
        (lm for lm in landmarks if int(lm.get("id", -1)) == int(head_id)),
        None,
    )
    if head_lm is None:
        return None

    tail_lm = next(
        (lm for lm in landmarks if int(lm.get("id", -1)) == int(tail_id)),
        None,
    )
    if tail_lm is None:
        return None

    return (
        (float(head_lm["x"]), float(head_lm["y"])),
        (float(tail_lm["x"]), float(tail_lm["y"])),
    )


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
    accepted_boxes_declared = bool(
        isinstance(finalized_detection, dict)
        and "acceptedBoxes" in finalized_detection
    )
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

    if accepted_boxes_declared:
        accepted_raw = finalized_detection.get("acceptedBoxes")
        if not isinstance(accepted_raw, list):
            raise ValueError(
                f"{image_filename}: finalizedDetection.acceptedBoxes must be an array."
            )
        for accepted_index, b in enumerate(accepted_raw):
            if not isinstance(b, dict):
                raise ValueError(
                    f"{image_filename}: accepted OBB box {accepted_index + 1} is not an object."
                )
            has_explicit_corners = "obbCorners" in b or "obb_corners" in b
            if has_explicit_corners:
                raw_corners = (
                    b.get("obbCorners")
                    if "obbCorners" in b
                    else b.get("obb_corners")
                )
                _validate_real_obb_corners(
                    raw_corners,
                    f"{image_filename} accepted box {accepted_index + 1}",
                )
            nb = _normalize_box(b)
            if not nb:
                raise ValueError(
                    f"{image_filename}: accepted OBB box {accepted_index + 1} has invalid bounds."
                )
            # Backfill landmarks from draft boxes when finalized snapshot
            # was stored as geometry-only. Missing OBB geometry may be
            # backfilled for legacy snapshots, but an explicitly malformed
            # accepted OBB is rejected above rather than silently repaired.
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
                    # Backfill OBB geometry only when it was absent from the
                    # accepted legacy snapshot, never when explicitly invalid.
                    if (
                        not has_explicit_corners
                        and not nb.get("obbCorners")
                        and best.get("obbCorners")
                    ):
                        nb["obbCorners"] = list(best["obbCorners"])
                    if nb.get("angle") is None and best.get("angle") is not None:
                        nb["angle"] = best["angle"]
                    if nb.get("class_id") is None and best.get("class_id") is not None:
                        nb["class_id"] = best["class_id"]
            accepted.append(nb)

    # Backward compatibility applies only when older finalized sessions lack
    # the acceptedBoxes field. An explicitly empty array is authoritative (for
    # example, a HITL rejected_all review) and must never resurrect drafts.
    if not accepted and not accepted_boxes_declared:
        used_fallback = True
        accepted.extend([dict(b) for b in draft_boxes])

    # This is the fail-closed training boundary: every finalized box that will
    # enter the dataset must be a finite rectangle. Canonicalize the validated
    # values so the later writer cannot warn-and-drop a box.
    for accepted_index, box in enumerate(accepted):
        raw_corners = box.get("obbCorners") or box.get("obb_corners")
        validated = _validate_real_obb_corners(
            raw_corners,
            f"{image_filename} accepted box {accepted_index + 1}",
        )
        box["obbCorners"] = [[x, y] for x, y in validated]
        box.pop("obb_corners", None)

    accepted.sort(key=lambda b: (b["left"], b["top"], b["width"], b["height"]))
    return True, accepted, used_fallback


def _reset_output_dataset_dir(out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
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
    require_trusted=False,
):
    """
    Resolve class id used by OBB export.
    Priority:
      1) explicit box.class_id
      2) stored, reviewed OBB orientation metadata

    Directional/bilateral exports fail closed when neither is present. Landmark
    geometry is intentionally not a direction source: the saved OBB arrow is the
    native orientation contract.
    """
    if not orientation_class_enabled:
        return 0

    mode = str((orientation_policy or {}).get("mode", "directional")).strip().lower()
    bilateral_axis = _resolve_bilateral_class_axis(orientation_policy)
    orientation_override = str(box.get("orientation_override", "")).strip().lower()
    if require_trusted and orientation_override == "uncertain":
        raise ValueError(
            "orientation is explicitly uncertain; review it and choose a trusted "
            "direction before training"
        )

    class_id = box.get("class_id", None)
    if class_id is not None:
        try:
            resolved_class_id = int(class_id)
        except Exception as exc:
            if require_trusted:
                raise ValueError(
                    f"explicit class_id {class_id!r} is invalid; expected 0 or 1"
                ) from exc
        else:
            if not require_trusted or resolved_class_id in {0, 1}:
                if require_trusted:
                    explicit_orientation = _get_box_explicit_orientation(
                        box,
                        orientation_policy,
                    )
                    expected_orientation = (
                        ("up", "down")[resolved_class_id]
                        if mode == "bilateral" and bilateral_axis == "vertical_obb"
                        else ("left", "right")[resolved_class_id]
                    )
                    if (
                        explicit_orientation is not None
                        and explicit_orientation != expected_orientation
                    ):
                        raise ValueError(
                            f"explicit class_id {resolved_class_id} conflicts with "
                            f"orientation metadata {explicit_orientation!r}; expected "
                            f"{expected_orientation!r}"
                        )
                return resolved_class_id
        if require_trusted:
            raise ValueError(
                f"explicit class_id {resolved_class_id} is out of range; expected 0 or 1"
            )

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
        raise ValueError(
            "could not resolve a trusted up/down orientation from the saved OBB arrow; "
            "review this box before training"
        )
    raise ValueError(
        "could not resolve a trusted left/right orientation from the saved OBB arrow; "
        "review this box before training"
    )


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


def _split_profile_key(seed, val_ratio, test_ratio=None):
    resolved_test_ratio = val_ratio if test_ratio is None else test_ratio
    return (
        f"seed={int(seed)};val_ratio={float(val_ratio):.8f};"
        f"test_ratio={float(resolved_test_ratio):.8f}"
    )


def _json_sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _normalize_obb_split_name(value):
    normalized = str(value or "").strip().lower()
    if normalized == "val":
        return "validation"
    return normalized


def _obb_assignment_revision(group_assignments):
    return _json_sha256(
        {
            "format_version": OBB_SPLIT_ASSIGNMENTS_VERSION,
            "groups": {
                str(group_id): _normalize_obb_split_name(split)
                for group_id, split in sorted(group_assignments.items())
            },
        }
    )


def _migrate_obb_v1_split_assignments(payload, legacy_path):
    """Migrate v1 train/val assignments without moving a validation member.

    V1 had no independent test cohort or evaluator snapshots. Its existing
    train and validation assignments remain exactly where they were. If the old
    profile has no test benchmark, a later, newly observed non-adaptive group may
    fill that missing cohort once. Existing training data is never silently
    repurposed, which keeps before/after results scientifically comparable.
    """
    if not isinstance(payload, dict) or payload.get("version") != OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION:
        raise ValueError(
            f"Unsupported legacy OBB split assignment manifest at {legacy_path}; expected "
            f"version {OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION}."
        )
    raw_profiles = payload.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError(f"Invalid legacy OBB split assignment profiles in {legacy_path}.")

    migrated_profiles = {}
    for old_profile_key, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            raise ValueError(
                f"Invalid legacy OBB split profile '{old_profile_key}' in {legacy_path}."
            )
        try:
            seed = int(raw_profile.get("seed", 42))
            val_ratio = float(raw_profile.get("val_ratio", 0.2))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid legacy OBB split settings in profile '{old_profile_key}'."
            ) from exc
        profile_key = _split_profile_key(seed, val_ratio, val_ratio)
        if profile_key in migrated_profiles:
            raise ValueError(
                f"Legacy OBB profiles collide while migrating to '{profile_key}'. "
                "Create a new cohort version explicitly."
            )
        raw_groups = raw_profile.get("groups", {})
        raw_samples = raw_profile.get("samples", {})
        if not isinstance(raw_groups, dict) or not isinstance(raw_samples, dict):
            raise ValueError(f"Invalid legacy OBB assignments in profile '{old_profile_key}'.")
        groups = {
            str(group_id): _normalize_obb_split_name(split)
            for group_id, split in raw_groups.items()
        }
        samples = {}
        for sample_id, raw_assignment in raw_samples.items():
            if not isinstance(raw_assignment, dict):
                raise ValueError(
                    f"Invalid legacy OBB sample assignment '{sample_id}' in '{old_profile_key}'."
                )
            assignment = dict(raw_assignment)
            assignment["split"] = _normalize_obb_split_name(assignment.get("split"))
            samples[str(sample_id)] = assignment
        migrated_profiles[profile_key] = {
            **{
                key: copy.deepcopy(value)
                for key, value in raw_profile.items()
                if key not in {"groups", "samples", "seed", "val_ratio"}
            },
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": val_ratio,
            "groups": groups,
            "samples": samples,
            "migrated_from": {
                "version": OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION,
                "path": os.path.abspath(legacy_path),
                "profile_key": str(old_profile_key),
            },
        }
    return {
        "version": OBB_SPLIT_ASSIGNMENTS_VERSION,
        "migrated_from": {
            "version": OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION,
            "path": os.path.abspath(legacy_path),
        },
        "profiles": migrated_profiles,
    }


def _load_obb_split_assignments(path, legacy_path=None):
    source_path = path
    if not os.path.exists(source_path):
        if legacy_path and os.path.exists(legacy_path):
            source_path = legacy_path
        else:
            return {
                "version": OBB_SPLIT_ASSIGNMENTS_VERSION,
                "profiles": {},
            }
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"Could not read persisted OBB split assignments at {source_path}: {exc}. "
            "Repair the JSON or intentionally create a new cohort version."
        ) from exc

    if source_path == legacy_path and payload.get("version") == OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION:
        payload = _migrate_obb_v1_split_assignments(payload, source_path)
    elif not isinstance(payload, dict) or payload.get("version") != OBB_SPLIT_ASSIGNMENTS_VERSION:
        raise ValueError(
            f"Unsupported OBB split assignment manifest at {source_path}; expected version "
            f"{OBB_SPLIT_ASSIGNMENTS_VERSION}. Migrate it before training."
        )
    if not isinstance(payload.get("profiles", {}), dict):
        raise ValueError(f"Invalid OBB split assignment profiles in {source_path}.")
    payload.setdefault("profiles", {})
    return payload


def _is_confirmed_negative_review(label_data):
    """True when a reviewer finalized an image as containing no specimen.

    An explicitly empty ``acceptedBoxes`` array on a finalized image is an
    assertion, not missing data: the operator looked at the image and rejected
    every candidate box.  That is the strongest available false-positive
    evidence, so it is exported as an empty YOLO label rather than discarded.
    A finalized image that simply never declared ``acceptedBoxes`` is legacy
    data and stays excluded.
    """
    if not isinstance(label_data, dict):
        return False
    finalized_detection = label_data.get("finalizedDetection")
    if not isinstance(finalized_detection, dict):
        return False
    if not finalized_detection.get("isFinalized"):
        return False
    accepted = finalized_detection.get("acceptedBoxes")
    return isinstance(accepted, list) and not accepted


def _is_adaptive_obb_sample(sample):
    if not isinstance(sample, dict):
        return False
    records = []
    provenance = sample.get("provenance")
    if isinstance(provenance, dict):
        records.append(provenance)
    review_history = sample.get("review_history") or sample.get("reviewHistory")
    if isinstance(review_history, list):
        records.extend(item for item in review_history if isinstance(item, dict))
    adaptive_sources = {"hitl", "hitl_review", "inference_review", "model_assisted_review"}
    return any(
        str(record.get("source") or "").strip().lower() in adaptive_sources
        for record in records
    )


def _obb_group_snapshot(
    group_id,
    indices,
    samples,
    *,
    orientation_class_enabled,
    head_id,
    tail_id,
    orientation_policy,
):
    members = []
    for index in sorted(indices, key=lambda item: samples[item]["sample_id"]):
        sample = samples[index]
        boxes = []
        for box in sample.get("boxes", []):
            corners = box.get("obbCorners") or box.get("obb_corners") or []
            landmarks = sorted(
                (
                    {
                        "id": landmark.get("id"),
                        "x": _safe_float(landmark.get("x"), None),
                        "y": _safe_float(landmark.get("y"), None),
                    }
                    for landmark in box.get("landmarks", [])
                    if isinstance(landmark, dict)
                ),
                key=lambda item: (str(item.get("id")), item.get("x") or 0.0, item.get("y") or 0.0),
            )
            boxes.append(
                {
                    "obb_corners": [
                        [_safe_float(point[0], None), _safe_float(point[1], None)]
                        for point in corners
                    ],
                    "class_id": _resolve_obb_class_id(
                        box,
                        orientation_class_enabled=orientation_class_enabled,
                        head_id=head_id,
                        tail_id=tail_id,
                        orientation_policy=orientation_policy,
                    ),
                    "landmarks": landmarks,
                    "orientation_override": box.get("orientation_override"),
                    "orientation_hint": box.get("orientation_hint"),
                }
            )
        members.append(
            {
                "sample_id": sample["sample_id"],
                "content_sha256": sample.get("content_sha256"),
                "adaptive_training_sample": _is_adaptive_obb_sample(sample),
                "review_provenance_sha256": _json_sha256(
                    {
                        "provenance": sample.get("provenance") or {},
                        "review_history": sample.get("review_history") or [],
                    }
                ),
                "boxes": boxes,
            }
        )
    return _json_sha256(
        {
            "format_version": 1,
            "group_id": group_id,
            "members": members,
        }
    )


def _obb_cohort_revision(profile_key, cohort, group_assignments, group_snapshots):
    group_ids = sorted(
        group_id
        for group_id, split in group_assignments.items()
        if _normalize_obb_split_name(split) == cohort
    )
    if not group_ids:
        return None
    return _json_sha256(
        {
            "format_version": 2,
            "split_profile_key": profile_key,
            "cohort": cohort,
            "groups": [
                {
                    "group_id": group_id,
                    "snapshot_sha256": group_snapshots.get(group_id),
                }
                for group_id in group_ids
            ],
        }
    )


def _is_sha256_hex(value):
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_existing_obb_v2_manifest(payload, path):
    """Validate every persisted v2 profile without repairing it implicitly."""
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(
            f"Existing OBB split assignment manifest at {path} has no valid profiles. "
            "Restore it or intentionally create a new cohort version."
        )

    for profile_key, profile in profiles.items():
        if not isinstance(profile_key, str) or not isinstance(profile, dict):
            raise ValueError(f"Invalid OBB split profile in {path}.")
        required = {
            "seed",
            "val_ratio",
            "test_ratio",
            "groups",
            "samples",
            "assignment_revision",
            "validation_group_snapshots",
            "test_group_snapshots",
            "validation_cohort_revision",
            "test_cohort_revision",
        }
        missing_fields = sorted(required - set(profile))
        if missing_fields:
            raise ValueError(
                f"OBB split profile '{profile_key}' is missing required v2 fields "
                f"{missing_fields}. Restore the manifest or intentionally create a new cohort version."
            )
        try:
            seed = int(profile["seed"])
            val_ratio = float(profile["val_ratio"])
            test_ratio = float(profile["test_ratio"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid split settings in OBB profile '{profile_key}'.") from exc
        if profile_key != _split_profile_key(seed, val_ratio, test_ratio):
            raise ValueError(
                f"OBB split profile key '{profile_key}' does not match its persisted settings. "
                "Restore the manifest or intentionally create a new cohort version."
            )

        groups = profile["groups"]
        samples = profile["samples"]
        adaptive_groups = profile.get("adaptive_groups", {})
        if not isinstance(groups, dict) or not isinstance(samples, dict) or not isinstance(
            adaptive_groups, dict
        ):
            raise ValueError(f"Invalid assignments in OBB profile '{profile_key}'.")
        for group_id, split in groups.items():
            if not str(group_id).strip() or split not in {"train", "validation", "test"}:
                raise ValueError(
                    f"Invalid source-group assignment in OBB profile '{profile_key}'."
                )
        for group_id, adaptive in adaptive_groups.items():
            if group_id not in groups or not isinstance(adaptive, bool):
                raise ValueError(
                    f"Invalid adaptive-source history in OBB profile '{profile_key}'."
                )
        for sample_id, assignment in samples.items():
            if not str(sample_id).strip() or not isinstance(assignment, dict):
                raise ValueError(f"Invalid sample assignment in OBB profile '{profile_key}'.")
            group_id = str(assignment.get("group_id") or "").strip()
            split = assignment.get("split")
            if group_id not in groups or split != groups[group_id]:
                raise ValueError(
                    f"The OBB split assignments for profile '{profile_key}' were mutated or "
                    f"are corrupt: persisted sample '{str(sample_id)[:12]}' disagrees with "
                    "its source-group assignment."
                )

        assignment_revision = profile["assignment_revision"]
        if not _is_sha256_hex(assignment_revision) or (
            _obb_assignment_revision(groups) != str(assignment_revision).lower()
        ):
            raise ValueError(
                f"The OBB split assignments for profile '{profile_key}' are missing a valid "
                "revision or were mutated. Restore the manifest or intentionally create a new cohort version."
            )

        for cohort, snapshot_key, revision_key in (
            ("validation", "validation_group_snapshots", "validation_cohort_revision"),
            ("test", "test_group_snapshots", "test_cohort_revision"),
        ):
            snapshots = profile[snapshot_key]
            if not isinstance(snapshots, dict):
                raise ValueError(
                    f"Invalid OBB {cohort} snapshots in profile '{profile_key}'."
                )
            assigned_groups = {
                group_id for group_id, split in groups.items() if split == cohort
            }
            if set(snapshots) != assigned_groups or any(
                not _is_sha256_hex(snapshots.get(group_id)) for group_id in assigned_groups
            ):
                raise ValueError(
                    f"OBB split profile '{profile_key}' is missing an exact snapshot for every "
                    f"assigned {cohort} source group. Restore the manifest or intentionally create a new cohort version."
                )
            revision = profile[revision_key]
            if assigned_groups:
                if not _is_sha256_hex(revision) or _obb_cohort_revision(
                    profile_key,
                    cohort,
                    groups,
                    snapshots,
                ) != str(revision).lower():
                    raise ValueError(
                        f"The frozen OBB {cohort} cohort is missing a valid revision or is corrupt. "
                        "Restore the manifest or intentionally create a new cohort version."
                    )
            elif revision not in {None, ""}:
                raise ValueError(
                    f"OBB profile '{profile_key}' has a {cohort} revision without assigned groups."
                )


def _select_and_persist_obb_splits(
    samples,
    val_ratio,
    seed,
    assignments_path,
    legacy_assignments_path=None,
    test_ratio=None,
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
    Select immutable train/validation/test assignments by source-content group.

    Validation is the trainer-selection/promotion cohort. Test is an independent
    report-only benchmark. Adaptive/HITL groups are always train-only when the
    profile is first locked, and every new group is train-only after that lock.
    Exact evaluator snapshots make annotation or image mutation fail closed.
    """
    ideal_val_set, target_val_count, split_stats = _select_obb_val_indices(
        samples,
        val_ratio=val_ratio,
        seed=seed,
        orientation_class_enabled=orientation_class_enabled,
        head_id=head_id,
        tail_id=tail_id,
        orientation_policy=orientation_policy,
        minority_small_cutoff=minority_small_cutoff,
        minority_target_ratio=minority_target_ratio,
        minority_min_ratio=minority_min_ratio,
        minority_max_ratio=minority_max_ratio,
    )

    resolved_test_ratio = val_ratio if test_ratio is None else test_ratio
    resolved_test_ratio = max(0.0, min(1.0, float(resolved_test_ratio)))
    current_manifest_existed = os.path.exists(assignments_path)
    legacy_manifest_existed = bool(
        legacy_assignments_path and os.path.exists(legacy_assignments_path)
    )
    manifest = _load_obb_split_assignments(
        assignments_path,
        legacy_path=legacy_assignments_path,
    )
    if current_manifest_existed:
        _validate_existing_obb_v2_manifest(manifest, assignments_path)
    profile_key = _split_profile_key(seed, val_ratio, resolved_test_ratio)
    if profile_key not in manifest["profiles"] and (
        current_manifest_existed or legacy_manifest_existed or manifest["profiles"]
    ):
        existing_profiles = sorted(manifest["profiles"])
        raise ValueError(
            f"OBB split settings resolve to a new profile '{profile_key}', but this session "
            f"already has frozen profile(s) {existing_profiles}. Changing seed or split ratios "
            "requires an explicit future cohort-version workflow; existing training groups "
            "will not be repartitioned implicitly."
        )
    if profile_key not in manifest["profiles"]:
        manifest["profiles"][profile_key] = {
            "seed": int(seed),
            "val_ratio": float(val_ratio),
            "test_ratio": resolved_test_ratio,
            "groups": {},
            "samples": {},
        }
    profile = manifest["profiles"][profile_key]
    try:
        stored_seed = int(profile.get("seed", seed))
        stored_val_ratio = float(profile.get("val_ratio", val_ratio))
        stored_test_ratio = float(profile.get("test_ratio", resolved_test_ratio))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid split settings in OBB profile '{profile_key}'.") from exc
    if (
        stored_seed != int(seed)
        or abs(stored_val_ratio - float(val_ratio)) > 1e-12
        or abs(stored_test_ratio - resolved_test_ratio) > 1e-12
    ):
        raise ValueError(
            f"OBB split profile '{profile_key}' settings were mutated. Restore the manifest "
            "or intentionally create a new cohort version."
        )

    group_assignments = profile.setdefault("groups", {})
    sample_assignments = profile.setdefault("samples", {})
    if not isinstance(group_assignments, dict) or not isinstance(sample_assignments, dict):
        raise ValueError(f"Invalid profile '{profile_key}' in {assignments_path}.")

    group_indices = {}
    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", "")).strip()
        group_id = str(sample.get("group_id", "")).strip()
        if not sample_id or not group_id:
            raise ValueError("OBB samples must have stable sample_id and group_id values before splitting.")
        group_indices.setdefault(group_id, []).append(index)

    stored_assignment_revision = str(profile.get("assignment_revision") or "").strip()
    if stored_assignment_revision:
        computed_assignment_revision = _obb_assignment_revision(group_assignments)
        if computed_assignment_revision != stored_assignment_revision:
            raise ValueError(
                f"The OBB split assignments for profile '{profile_key}' were mutated or are "
                "corrupt. Restore the manifest or intentionally create a new cohort version."
            )

    # Recover group assignments from older sample records if necessary, while
    # refusing a conflict that would leak one source group across both splits.
    for group_id, indices in group_indices.items():
        historical_splits = {
            _normalize_obb_split_name(
                sample_assignments.get(samples[index]["sample_id"], {}).get("split", "")
            )
            for index in indices
        }
        historical_splits.discard("")
        if len(historical_splits) > 1:
            raise ValueError(
                f"Persisted OBB split assignments place source group {group_id[:12]} in "
                "multiple cohorts. Resolve the manifest before training to prevent leakage."
            )
        if (
            group_id in group_assignments
            and historical_splits
            and next(iter(historical_splits)) != group_assignments[group_id]
        ):
            raise ValueError(
                f"Persisted OBB sample and source-group assignments disagree for group "
                f"{group_id[:12]}. Resolve the manifest before training; the frozen "
                "validation cohort will not be mutated implicitly."
            )
        if group_id not in group_assignments and historical_splits:
            group_assignments[group_id] = next(iter(historical_splits))

    profile_had_assignments = bool(group_assignments)

    for group_id, split in list(group_assignments.items()):
        normalized_split = _normalize_obb_split_name(split)
        if normalized_split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Persisted OBB source group {group_id[:12]} has invalid split '{split}'."
            )
        group_assignments[group_id] = normalized_split

    persisted_adaptive_groups = profile.setdefault("adaptive_groups", {})
    if not isinstance(persisted_adaptive_groups, dict):
        raise ValueError(
            f"Invalid adaptive-source history in OBB profile '{profile_key}'. Restore the manifest."
        )
    group_is_adaptive = {}
    # Confirmed negatives carry no positive geometry, so admitting them to an
    # evaluator cohort would silently change that cohort's mAP denominator and
    # break comparability with models measured before they arrived.  Unlike
    # adaptive provenance this is re-derived every run rather than persisted:
    # if a reviewer later draws a box, the group simply stops being negative.
    group_is_negative_only = {}
    for group_id, indices in group_indices.items():
        current_adaptive = any(_is_adaptive_obb_sample(samples[index]) for index in indices)
        sticky_adaptive = bool(
            persisted_adaptive_groups.get(group_id) is True or current_adaptive
        )
        group_is_adaptive[group_id] = sticky_adaptive
        persisted_adaptive_groups[group_id] = sticky_adaptive
        group_is_negative_only[group_id] = all(
            bool(samples[index].get("is_negative")) for index in indices
        )

    def _is_train_only_group(group_id):
        return bool(
            group_is_adaptive.get(group_id, False)
            or group_is_negative_only.get(group_id, False)
        )

    contaminated_evaluators = sorted(
        group_id
        for group_id, split in group_assignments.items()
        if split in {"validation", "test"} and group_is_adaptive.get(group_id, False)
    )
    if contaminated_evaluators:
        raise ValueError(
            "A frozen OBB evaluator source was later used for HITL/model-assisted review: "
            f"{[group_id[:12] for group_id in contaminated_evaluators]}. Restore the original "
            "benchmark annotation/provenance or intentionally create a new cohort version."
        )
    current_group_snapshots = {
        group_id: _obb_group_snapshot(
            group_id,
            indices,
            samples,
            orientation_class_enabled=orientation_class_enabled,
            head_id=head_id,
            tail_id=tail_id,
            orientation_policy=orientation_policy,
        )
        for group_id, indices in group_indices.items()
    }
    persisted_validation_snapshots = profile.get("validation_group_snapshots", {})
    persisted_test_snapshots = profile.get("test_group_snapshots", {})
    if not isinstance(persisted_validation_snapshots, dict) or not isinstance(
        persisted_test_snapshots, dict
    ):
        raise ValueError(
            f"Invalid OBB evaluator snapshots in profile '{profile_key}'. Restore the manifest."
        )
    for cohort, snapshots, revision_key in (
        ("validation", persisted_validation_snapshots, "validation_cohort_revision"),
        ("test", persisted_test_snapshots, "test_cohort_revision"),
    ):
        persisted_revision = str(profile.get(revision_key) or "").strip()
        if not persisted_revision:
            continue
        computed_revision = _obb_cohort_revision(
            profile_key,
            cohort,
            group_assignments,
            snapshots,
        )
        if computed_revision != persisted_revision:
            raise ValueError(
                f"The frozen OBB {cohort} cohort revision was mutated or is corrupt. "
                "Restore the manifest or intentionally create a new cohort version."
            )

    for cohort, persisted_snapshots in (
        ("validation", persisted_validation_snapshots),
        ("test", persisted_test_snapshots),
    ):
        locked_groups = sorted(
            group_id
            for group_id, split in group_assignments.items()
            if split == cohort
        )
        missing = [group_id for group_id in locked_groups if group_id not in group_indices]
        if missing:
            raise ValueError(
                f"The frozen OBB {cohort} cohort is missing source groups "
                f"{[group_id[:12] for group_id in missing]}. Restore their images/labels or "
                "intentionally create a new cohort version."
            )
        for group_id in locked_groups:
            expected = persisted_snapshots.get(group_id)
            current = current_group_snapshots[group_id]
            if expected and str(expected) != current:
                raise ValueError(
                    f"The frozen OBB {cohort} snapshot changed for source group "
                    f"{group_id[:12]}. Restore its image and finalized annotation or "
                    "intentionally create a new cohort version."
                )

    known_groups = {group_id for group_id in group_indices if group_id in group_assignments}
    new_groups = [group_id for group_id in group_indices if group_id not in known_groups]

    orientation_mode = str((orientation_policy or {}).get("mode", "invariant")).strip().lower()
    directional_validation_mirror_enabled = bool(
        orientation_class_enabled and orientation_mode == "directional"
    )
    expected_validation_class_ids = {0, 1} if orientation_class_enabled else {0}

    def _evaluation_class_ids(real_class_ids):
        effective = set(real_class_ids)
        if directional_validation_mirror_enabled:
            effective.update(
                1 - class_id for class_id in real_class_ids if class_id in {0, 1}
            )
        return effective

    group_class_ids = {}
    for group_id, indices in group_indices.items():
        class_ids = set()
        for index in indices:
            class_ids.update(
                int(class_id)
                for class_id in _sample_class_histogram(
                    samples[index],
                    orientation_class_enabled=orientation_class_enabled,
                    head_id=head_id,
                    tail_id=tail_id,
                    orientation_policy=orientation_policy,
                )
            )
        group_class_ids[group_id] = class_ids

    def _group_rank(group_id):
        priority = 0 if any(index in ideal_val_set for index in group_indices[group_id]) else 1
        digest = hashlib.sha256(f"{int(seed)}\0{group_id}".encode("utf-8")).hexdigest()
        return priority, digest

    def _validation_readiness(assignments):
        validation_groups = {
            group_id for group_id, split in assignments.items() if split == "validation"
        }
        real_observed_classes = set()
        for group_id in validation_groups:
            real_observed_classes.update(group_class_ids.get(group_id, set()))
        observed_classes = _evaluation_class_ids(real_observed_classes)
        complete = bool(
            float(val_ratio) <= 0
            or (
                len(validation_groups) >= 2
                and expected_validation_class_ids.issubset(observed_classes)
            )
        )
        return validation_groups, observed_classes, complete

    def _choose_validation_additions(candidates, limit, existing_groups, existing_classes):
        remaining = sorted(set(candidates), key=_group_rank)
        chosen = []
        observed = set(existing_classes)
        capacity = max(0, int(limit))

        # Prefer groups that close an orientation-class gap. This lets common
        # arrivals remain training data while waiting for a late rare class.
        while remaining and len(chosen) < capacity:
            missing_classes = expected_validation_class_ids - observed
            useful = [
                group_id
                for group_id in remaining
                if group_class_ids.get(group_id, set()).intersection(missing_classes)
            ]
            if not useful:
                break
            selected = min(
                useful,
                key=lambda group_id: (
                    -len(group_class_ids.get(group_id, set()).intersection(missing_classes)),
                    _group_rank(group_id),
                ),
            )
            chosen.append(selected)
            remaining.remove(selected)
            observed.update(group_class_ids.get(selected, set()))
            observed = _evaluation_class_ids(observed)

        # Independent evidence is required even when one group contains every
        # class. Fill only to two groups during evaluator bootstrap.
        while (
            remaining
            and len(chosen) < capacity
            and len(existing_groups) + len(chosen) < 2
        ):
            selected = min(remaining, key=_group_rank)
            chosen.append(selected)
            remaining.remove(selected)
            observed.update(group_class_ids.get(selected, set()))
            observed = _evaluation_class_ids(observed)
        return chosen

    if not profile_had_assignments:
        adaptive_groups = [
            group_id for group_id in group_indices if _is_train_only_group(group_id)
        ]
        eligible_groups = [
            group_id for group_id in group_indices if not _is_train_only_group(group_id)
        ]
        for group_id in adaptive_groups:
            group_assignments[group_id] = "train"

        reserve_train = 0 if adaptive_groups else 1
        max_held_out = max(0, len(eligible_groups) - reserve_train)
        validation_groups = set()
        reserve_test = 1 if max_held_out >= 2 and resolved_test_ratio > 0 else 0
        max_validation_groups = max(0, max_held_out - reserve_test)
        if max_validation_groups > 0 and float(val_ratio) > 0:
            requested_validation_groups = max(
                2,
                int(round(len(eligible_groups) * max(0.0, min(1.0, float(val_ratio))))),
            )
            validation_count = min(requested_validation_groups, max_validation_groups)
            validation_groups = set(
                _choose_validation_additions(
                    eligible_groups,
                    validation_count,
                    set(),
                    set(),
                )
            )
            # Once readiness is satisfied, preserve ratio-requested evaluator
            # breadth for larger initial datasets.
            if len(validation_groups) < validation_count:
                remaining_validation = sorted(
                    set(eligible_groups) - validation_groups,
                    key=_group_rank,
                )
                validation_groups.update(
                    remaining_validation[: validation_count - len(validation_groups)]
                )

        test_pool = [
            group_id for group_id in eligible_groups if group_id not in validation_groups
        ]
        max_test_groups = max(0, len(test_pool) - reserve_train)
        test_groups = set()
        if max_test_groups > 0 and resolved_test_ratio > 0:
            requested_test_groups = max(1, int(round(len(eligible_groups) * resolved_test_ratio)))
            test_count = min(requested_test_groups, max_test_groups)
            ordered_test = sorted(
                test_pool,
                key=lambda group_id: hashlib.sha256(
                    f"obb-test-v2\0{int(seed)}\0{group_id}".encode("utf-8")
                ).hexdigest(),
            )
            test_groups = set(ordered_test[:test_count])

        for group_id in eligible_groups:
            if group_id in test_groups:
                group_assignments[group_id] = "test"
            elif group_id in validation_groups:
                group_assignments[group_id] = "validation"
            else:
                group_assignments[group_id] = "train"
    else:
        # A tiny first export may not contain enough independent validation
        # evidence. Existing members never move, and only newly observed clean
        # groups may close the two-group/class-coverage gap. Once readiness is
        # reached the evaluator locks exactly once and later groups are train-only.
        existing_validation_groups, existing_validation_classes, computed_ready = (
            _validation_readiness(group_assignments)
        )
        stored_ready = profile.get("validation_evaluator_complete")
        if stored_ready is not None and not isinstance(stored_ready, bool):
            raise ValueError(
                f"Invalid validation evaluator state in OBB profile '{profile_key}'."
            )
        if stored_ready is True and not computed_ready:
            raise ValueError(
                f"The locked OBB validation evaluator in profile '{profile_key}' no longer "
                "meets its persisted evidence contract. Restore the manifest/data or create "
                "a new cohort version explicitly."
            )
        validation_ready = computed_ready if stored_ready is None else stored_ready
        has_test = any(split == "test" for split in group_assignments.values())
        eligible_new_groups = [
            group_id for group_id in new_groups if not _is_train_only_group(group_id)
        ]
        adaptive_new_groups = [
            group_id for group_id in new_groups if _is_train_only_group(group_id)
        ]
        for group_id in adaptive_new_groups:
            group_assignments[group_id] = "train"

        remaining = list(eligible_new_groups)
        if not validation_ready and remaining and float(val_ratio) > 0:
            reserve_for_test = 1 if not has_test and resolved_test_ratio > 0 else 0
            validation_limit = max(0, len(remaining) - reserve_for_test)
            chosen_validation_groups = _choose_validation_additions(
                remaining,
                validation_limit,
                existing_validation_groups,
                existing_validation_classes,
            )
            for chosen_validation in chosen_validation_groups:
                group_assignments[chosen_validation] = "validation"
                remaining.remove(chosen_validation)
                existing_validation_groups.add(chosen_validation)
                existing_validation_classes.update(
                    group_class_ids.get(chosen_validation, set())
                )
            _groups, _classes, validation_ready = _validation_readiness(group_assignments)
        if not has_test and remaining and resolved_test_ratio > 0:
            chosen_test = min(
                remaining,
                key=lambda group_id: hashlib.sha256(
                    f"obb-test-bootstrap-v2\0{int(seed)}\0{group_id}".encode("utf-8")
                ).hexdigest(),
            )
            group_assignments[chosen_test] = "test"
            remaining.remove(chosen_test)
            has_test = True
        for group_id in remaining:
            group_assignments[group_id] = "train"

    val_set = {
        index
        for group_id, indices in group_indices.items()
        if group_assignments[group_id] == "validation"
        for index in indices
    }
    test_set = {
        index
        for group_id, indices in group_indices.items()
        if group_assignments[group_id] == "test"
        for index in indices
    }
    train_set = set(range(len(samples))) - val_set - test_set
    if val_set.intersection(test_set) or train_set.intersection(val_set) or train_set.intersection(test_set):
        raise ValueError("OBB train/validation/test cohorts overlap; restore the split manifest.")

    for index, sample in enumerate(samples):
        split = "validation" if index in val_set else ("test" if index in test_set else "train")
        sample_assignments[sample["sample_id"]] = {
            "split": split,
            "group_id": sample["group_id"],
            "image_filename": sample.get("image_filename", ""),
            "content_sha256": sample.get("content_sha256", ""),
        }
    profile.setdefault("bootstrap_target_val_images", int(target_val_count))
    profile.setdefault("bootstrap_val_sample_ids", sorted(
        samples[index]["sample_id"] for index in val_set
    ))
    validation_group_snapshots = dict(persisted_validation_snapshots)
    test_group_snapshots = dict(persisted_test_snapshots)
    validation_group_snapshots.update(
        {
            group_id: current_group_snapshots[group_id]
            for group_id, split in group_assignments.items()
            if split == "validation" and group_id in current_group_snapshots
        }
    )
    test_group_snapshots.update(
        {
            group_id: current_group_snapshots[group_id]
            for group_id, split in group_assignments.items()
            if split == "test" and group_id in current_group_snapshots
        }
    )
    validation_revision = _obb_cohort_revision(
        profile_key,
        "validation",
        group_assignments,
        validation_group_snapshots,
    )
    test_revision = _obb_cohort_revision(
        profile_key,
        "test",
        group_assignments,
        test_group_snapshots,
    )
    (
        final_validation_groups,
        final_validation_classes,
        validation_evaluator_complete,
    ) = _validation_readiness(group_assignments)
    final_validation_real_classes = set()
    for group_id in final_validation_groups:
        final_validation_real_classes.update(group_class_ids.get(group_id, set()))
    profile["sample_count_at_last_export"] = len(samples)
    profile["train_count_at_last_export"] = len(train_set)
    profile["val_count_at_last_export"] = len(val_set)
    profile["test_count_at_last_export"] = len(test_set)
    profile["validation_group_snapshots"] = validation_group_snapshots
    profile["test_group_snapshots"] = test_group_snapshots
    profile["validation_cohort_revision"] = validation_revision
    profile["test_cohort_revision"] = test_revision
    profile["validation_evaluator_complete"] = bool(validation_evaluator_complete)
    profile["validation_required_group_count"] = 2
    profile["validation_group_count"] = len(final_validation_groups)
    profile["validation_expected_class_ids"] = sorted(expected_validation_class_ids)
    profile["validation_observed_class_ids"] = sorted(final_validation_classes)
    profile["validation_real_observed_class_ids"] = sorted(
        final_validation_real_classes
    )
    profile["validation_derivation"] = (
        {
            "type": "horizontal_mirror",
            "version": OBB_DIRECTIONAL_VALIDATION_MIRROR_VERSION,
            "source": "frozen_real_validation",
            "class_transform": "binary_swap",
        }
        if directional_validation_mirror_enabled
        else None
    )
    profile["validation_missing_class_ids"] = sorted(
        expected_validation_class_ids - final_validation_classes
    )
    profile["assignment_revision"] = _obb_assignment_revision(group_assignments)
    profile["disjoint"] = {
        "train_validation": True,
        "train_test": True,
        "validation_test": True,
    }
    profile["new_group_policy"] = (
        "unseen_non_adaptive_to_incomplete_validation_then_train_only_after_lock"
    )
    profile["adaptive_group_policy"] = "train_only"
    profile["evaluator_lock_complete"] = bool(
        validation_evaluator_complete
        and (test_revision or resolved_test_ratio <= 0)
    )
    _atomic_write_json(assignments_path, manifest)

    rotated_flags = [_sample_has_rotated_obb(sample) for sample in samples]
    split_stats["rotated_real_images_total"] = int(sum(rotated_flags))
    split_stats["rotated_real_images_val"] = int(
        sum(1 for index in val_set if rotated_flags[index])
    )
    split_stats["rotated_real_images_test"] = int(
        sum(1 for index in test_set if rotated_flags[index])
    )
    minority_class_ids = list(split_stats.get("minority_class_ids", []))
    if minority_class_ids:
        sample_hists = [
            _sample_class_histogram(
                sample,
                orientation_class_enabled=orientation_class_enabled,
                head_id=head_id,
                tail_id=tail_id,
                orientation_policy=orientation_policy,
            )
            for sample in samples
        ]
        split_stats["minority_val_instances"] = int(
            sum(
                sample_hists[index].get(class_id, 0)
                for index in val_set
                for class_id in minority_class_ids
            )
        )
    split_stats["requested_ratio_target_val_images"] = int(target_val_count)
    split_stats["test_ratio"] = resolved_test_ratio
    split_stats["target_val_images"] = int(profile["bootstrap_target_val_images"])
    split_stats["actual_val_images"] = int(len(val_set))
    split_stats["actual_test_images"] = int(len(test_set))
    split_stats["actual_train_images"] = int(len(train_set))
    split_stats["profile_key"] = profile_key
    split_stats["validation_cohort_frozen"] = bool(validation_evaluator_complete)
    split_stats["test_cohort_frozen"] = bool(test_revision)
    split_stats["validation_cohort_revision"] = validation_revision
    split_stats["test_cohort_revision"] = test_revision
    split_stats["assignment_revision"] = profile["assignment_revision"]
    split_stats["cohort_disjoint"] = dict(profile["disjoint"])
    split_stats["adaptive_groups_train_only"] = True
    split_stats["validation_evaluator_complete"] = bool(
        validation_evaluator_complete
    )
    split_stats["validation_required_group_count"] = 2
    split_stats["validation_group_count"] = len(final_validation_groups)
    split_stats["validation_expected_class_ids"] = sorted(
        expected_validation_class_ids
    )
    split_stats["validation_observed_class_ids"] = sorted(final_validation_classes)
    split_stats["validation_real_observed_class_ids"] = sorted(
        final_validation_real_classes
    )
    split_stats["validation_derivation"] = profile["validation_derivation"]
    split_stats["validation_missing_class_ids"] = sorted(
        expected_validation_class_ids - final_validation_classes
    )
    return val_set, test_set, split_stats


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
    rotation_enabled=True,
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
    if not rotation_enabled:
        angle = 0.0
    elif orientation_schema == "bilateral":
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
    # Axial specimens are pole-invariant: their two ends do not create detector
    # classes. Only directional/bilateral contracts carry semantic class IDs.
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


def _mirror_directional_obb_label_lines(lines):
    """Mirror exported directional OBB labels without consulting landmarks.

    The saved arrow/class is the native direction contract. A horizontal mirror
    reverses that class and reverses the vertex winding back to the canonical
    [LT, RT, RB, LB] ordering expected by the existing exporter.
    """
    mirrored_lines = []
    class_histogram = {}
    for raw_line in lines:
        fields = str(raw_line or "").strip().split()
        if not fields:
            continue
        if len(fields) != 9:
            raise ValueError(
                "Directional validation mirroring requires YOLO OBB labels with "
                "one class ID and four normalized corner pairs."
            )
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Directional validation mirroring encountered a malformed OBB label."
            ) from exc
        if class_id not in {0, 1}:
            raise ValueError(
                f"Directional validation mirroring requires class IDs in {{0, 1}}, got {class_id}."
            )
        points = [
            (coordinates[index], coordinates[index + 1])
            for index in range(0, 8, 2)
        ]
        reflected = [
            (_clamp(1.0 - x, 0.0, 1.0), _clamp(y, 0.0, 1.0))
            for x, y in points
        ]
        # Reflection reverses winding. Reorder mirrored [RT, LT, LB, RB] to
        # canonical [LT, RT, RB, LB] while preserving the same physical box.
        canonical = [reflected[index] for index in (1, 0, 3, 2)]
        mirrored_class_id = 1 - class_id
        flattened = [coordinate for point in canonical for coordinate in point]
        mirrored_lines.append(
            f"{mirrored_class_id} "
            + " ".join(f"{value:.6f}" for value in flattened)
        )
        class_histogram[mirrored_class_id] = (
            class_histogram.get(mirrored_class_id, 0) + 1
        )
    return mirrored_lines, class_histogram


def _rotate_bilateral_obb_label_lines_180(lines):
    """Rotate bilateral OBB labels by 180 degrees and swap the pole class."""
    rotated_lines = []
    class_histogram = {}
    for raw_line in lines:
        fields = str(raw_line or "").strip().split()
        if not fields:
            continue
        if len(fields) != 9:
            raise ValueError(
                "Bilateral training rotation requires YOLO OBB labels with "
                "one class ID and four normalized corner pairs."
            )
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Bilateral training rotation encountered a malformed OBB label."
            ) from exc
        if class_id not in {0, 1}:
            raise ValueError(
                f"Bilateral training rotation requires class IDs in {{0, 1}}, got {class_id}."
            )
        points = [
            (coordinates[index], coordinates[index + 1])
            for index in range(0, 8, 2)
        ]
        transformed = [
            (_clamp(1.0 - x, 0.0, 1.0), _clamp(1.0 - y, 0.0, 1.0))
            for x, y in points
        ]
        # A half turn preserves winding but shifts the canonical starting corner.
        canonical = [transformed[index] for index in (2, 3, 0, 1)]
        rotated_class_id = 1 - class_id
        flattened = [coordinate for point in canonical for coordinate in point]
        rotated_lines.append(
            f"{rotated_class_id} "
            + " ".join(f"{value:.6f}" for value in flattened)
        )
        class_histogram[rotated_class_id] = (
            class_histogram.get(rotated_class_id, 0) + 1
        )
    return rotated_lines, class_histogram


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
    max_instances=None,
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
        remaining_instances = (
            None
            if max_instances is None
            else max(0, int(max_instances) - n_instances_generated)
        )
        if remaining_instances == 0:
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
        if remaining_instances is not None:
            target_objects = min(target_objects, remaining_instances)

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
                # Ultralytics supplies the bounded small-angle transform. Keep
                # offline transforms only where they encode class/pole semantics.
                rotation_enabled=orientation_schema in {"bilateral", "axial"},
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
        if remaining_instances is not None:
            min_required = min(min_required, remaining_instances)
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
        "max_images": None if max_images is None else int(max_images),
        "max_instances": None if max_instances is None else int(max_instances),
        "offline_rotation_policy": (
            "semantic_only" if orientation_schema in {"bilateral", "axial"} else "disabled"
        ),
        "class_histogram": synthetic_class_hist,
        **seg_stats,
    }


def export_obb_dataset(
    session_dir,
    val_ratio=0.2,
    test_ratio=None,
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
    require_explicit_orientation_policy(session_dir)
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

        try:
            is_finalized, boxes, _ = _get_finalized_boxes(data, image_filename, finalized_set)
        except (TypeError, ValueError) as exc:
            return {
                "ok": False,
                "error": (
                    f"Could not parse finalized OBB geometry in '{fname}': {exc}. "
                    "Repair or redraw the affected box before training."
                ),
            }
        is_negative = False
        if not is_finalized:
            continue
        if not boxes:
            # A confirmed "no specimen here" review becomes a background
            # negative; anything else with no geometry is simply not ready.
            if not _is_confirmed_negative_review(data):
                continue
            is_negative = True

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

        try:
            sample_id, group_id, content_sha256 = _stable_sample_identity(
                image_path,
                image_filename,
                label_data=data,
            )
        except OSError as exc:
            return {
                "ok": False,
                "error": f"Could not fingerprint finalized image '{image_filename}' for a stable split: {exc}",
            }

        samples.append(
            {
                "image_path": image_path,
                "image_filename": image_filename,
                "boxes": boxes,
                "is_negative": is_negative,
                "sample_id": sample_id,
                "group_id": group_id,
                "content_sha256": content_sha256,
                "provenance": (
                    dict(data.get("provenance"))
                    if isinstance(data.get("provenance"), dict)
                    else {}
                ),
                "review_history": (
                    [dict(item) for item in data.get("reviewHistory", []) if isinstance(item, dict)]
                    if isinstance(data.get("reviewHistory"), list)
                    else []
                ),
            }
        )
        if idx == total_label_files or idx % 25 == 0:
            report_progress(
                "Scanning finalized annotations...",
                5.2 + (1.2 * (idx / total_label_files)),
                {"phase": "scan_labels", "current": idx, "total": total_label_files},
            )

    if not samples:
        return {"ok": False, "error": "No finalized samples with OBB annotations found"}

    try:
        _assign_unique_obb_export_names(samples)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    orientation_class_enabled, class_names, resolved_mode, _bilateral_axis = _resolve_detector_class_config(
        orientation_policy,
        fallback_mode=orientation_schema,
    )

    # Validate every source before committing new immutable assignments. If an
    # invalid image were assigned to validation/test and only discovered during
    # writing, repairing it would immediately violate the frozen snapshot.
    for sample in samples:
        image = safe_imread(sample["image_path"])
        if image is None or image.size == 0:
            return {
                "ok": False,
                "error": (
                    f"Could not decode finalized OBB image '{sample['image_filename']}'. "
                    "Repair or replace it before locking dataset cohorts."
                ),
            }
        try:
            _prepare_real_sample_for_export(
                image,
                sample["boxes"],
                sample["image_filename"],
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if orientation_class_enabled:
            for box_index, box in enumerate(sample["boxes"]):
                try:
                    _resolve_obb_class_id(
                        box,
                        orientation_class_enabled=True,
                        head_id=head_id,
                        tail_id=tail_id,
                        orientation_policy=orientation_policy,
                        require_trusted=True,
                    )
                except ValueError as exc:
                    return {
                        "ok": False,
                        "error": (
                            f"{sample['image_filename']} accepted box {box_index + 1}: {exc}. "
                            "Directional/bilateral OBB training requires a trusted class in {0, 1}."
                        ),
                    }

    # Split train/val — stratified when multiple orientation classes are present so that
    # val always contains at least one sample from each class.
    split_assignments_path = os.path.join(
        session_dir,
        f"obb_split_assignments.v{OBB_SPLIT_ASSIGNMENTS_VERSION}.json",
    )
    legacy_split_assignments_path = os.path.join(
        session_dir,
        f"obb_split_assignments.v{OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION}.json",
    )
    try:
        val_set, test_set, split_stats = _select_and_persist_obb_splits(
            samples,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            assignments_path=split_assignments_path,
            legacy_assignments_path=legacy_split_assignments_path,
            orientation_class_enabled=orientation_class_enabled,
            head_id=head_id,
            tail_id=tail_id,
            orientation_policy=orientation_policy,
            minority_small_cutoff=20,
            minority_target_ratio=0.20,
            minority_min_ratio=0.00,
            minority_max_ratio=0.20,
        )
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}

    # Only replace the derived output after the source data and persisted cohort
    # contract have both passed validation. A rejected frozen-cohort mutation
    # therefore leaves the previous usable export intact.
    _reset_output_dataset_dir(out_dir)
    warnings = []
    rotated_total = int(split_stats.get("rotated_real_images_total", 0))
    rotated_val = int(split_stats.get("rotated_real_images_val", 0))
    rotated_test = int(split_stats.get("rotated_real_images_test", 0))
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
    if int(split_stats.get("actual_test_images", 0)) > 0 and rotated_total >= 2 and rotated_test == 0:
        warnings.append(
            "Frozen test split has no rotated real OBB samples; report-only angle coverage is insufficient."
        )

    num_boxes = 0
    num_images = 0
    real_class_hist = {}
    real_class_hist_by_split = {"train": {}, "val": {}, "test": {}}
    real_export_manifest = []
    total_samples = max(1, len(samples))
    report_progress("Writing OBB dataset files...", 6.5, {"phase": "write_real", "current": 0, "total": total_samples})

    for i, sample in enumerate(samples):
        split = "val" if i in val_set else ("test" if i in test_set else "train")

        img = safe_imread(sample["image_path"])
        if img is None:
            warnings.append(
                f"{sample['image_filename']}: skipped image that could not be decoded; repair or replace the source image."
            )
            continue
        original_img_h, original_img_w = img.shape[:2]
        if original_img_w == 0 or original_img_h == 0:
            warnings.append(
                f"{sample['image_filename']}: skipped unreadable zero-sized image; repair or replace the source image."
            )
            continue

        try:
            export_img, export_boxes, image_transform = _prepare_real_sample_for_export(
                img,
                sample["boxes"],
                sample["image_filename"],
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "warnings": warnings}
        img_h, img_w = export_img.shape[:2]

        dest_img = os.path.join(
            out_dir,
            "images",
            split,
            sample["export_image_filename"],
        )
        padded = image_transform["type"] != "identity"
        if padded:
            if not safe_imwrite(dest_img, export_img):
                return {
                    "ok": False,
                    "error": (
                        f"Could not write padded OBB training image '{dest_img}'. "
                        "Check the output path and available disk space."
                    ),
                }
            padding = image_transform["padding"]
            warnings.append(
                f"{sample['image_filename']}: padded canvas "
                f"left={padding['left']}px, top={padding['top']}px, "
                f"right={padding['right']}px, bottom={padding['bottom']}px "
                "to preserve out-of-bounds OBB geometry."
            )
        elif not os.path.exists(dest_img):
            shutil.copy2(sample["image_path"], dest_img)

        label_path = os.path.join(
            out_dir,
            "labels",
            split,
            sample["export_label_filename"],
        )

        lines = []
        for box in export_boxes:
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
            obb_corners = box.get("obbCorners") or box.get("obb_corners")

            # Geometry was translated with the image canvas as a rigid transform.
            # Do not clamp individual vertices: doing so shears rotated rectangles.
            pts = []
            for px, py in obb_corners:
                normalized_x = float(px) / img_w
                normalized_y = float(py) / img_h
                if not (-1e-9 <= normalized_x <= 1.0 + 1e-9) or not (
                    -1e-9 <= normalized_y <= 1.0 + 1e-9
                ):
                    return {
                        "ok": False,
                        "error": (
                            f"{sample['image_filename']}: translated OBB still falls outside its "
                            "export canvas; repair the annotation before training."
                        ),
                    }
                pts.extend([normalized_x, normalized_y])
            lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in pts))
            num_boxes += 1
            if orientation_class_enabled:
                real_class_hist[int(class_id)] = real_class_hist.get(int(class_id), 0) + 1
                split_hist = real_class_hist_by_split[split]
                split_hist[int(class_id)] = split_hist.get(int(class_id), 0) + 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        num_images += 1
        real_export_manifest.append(
            {
                "sample_id": sample["sample_id"],
                "group_id": sample["group_id"],
                "content_sha256": sample["content_sha256"],
                "source_image": sample["image_filename"],
                "exported_image": os.path.relpath(dest_img, out_dir).replace("\\", "/"),
                "exported_label": os.path.relpath(label_path, out_dir).replace("\\", "/"),
                "exported_image_sha256": _sha256_file(dest_img),
                "label_sha256": _sha256_file(label_path),
                "split": split,
                "boxes_exported": len(lines),
                "transform": image_transform,
            }
        )
        if (i + 1) == total_samples or (i + 1) % 20 == 0:
            report_progress(
                "Writing OBB dataset files...",
                6.5 + (1.5 * ((i + 1) / total_samples)),
                {"phase": "write_real", "current": i + 1, "total": total_samples},
            )

    # Preserve the naturally observed histogram for scientific reporting, then
    # add one deterministic train-only class counterpart per real train image.
    # This guarantees both detector classes receive equal effective exposure
    # even when a small source cohort happens to contain only one orientation.
    training_derivatives = []
    training_derivative_histogram = {}
    if orientation_class_enabled and resolved_mode in {"directional", "bilateral"}:
        natural_train_histogram = real_class_hist_by_split["train"]
        missing_natural_classes = sorted({0, 1} - set(natural_train_histogram))
        if missing_natural_classes:
            warnings.append(
                "Natural real training annotations lack orientation class IDs "
                f"{missing_natural_classes}; deterministic train-only derivatives provide "
                "model exposure, but additional natural examples are still recommended."
            )
        training_entries = sorted(
            (entry for entry in real_export_manifest if entry["split"] == "train"),
            key=lambda entry: entry["sample_id"],
        )
        for entry in training_entries:
            source_image_path = os.path.join(out_dir, entry["exported_image"])
            source_label_path = os.path.join(out_dir, entry["exported_label"])
            source_image = safe_imread(source_image_path)
            if source_image is None or source_image.size == 0:
                return {
                    "ok": False,
                    "error": (
                        "Could not decode a real training image while creating its "
                        f"class-balanced derivative: {entry['exported_image']}"
                    ),
                    "warnings": warnings,
                }
            try:
                with open(source_label_path, "r", encoding="utf-8") as handle:
                    source_lines = [line.strip() for line in handle if line.strip()]
                if resolved_mode == "directional":
                    derivative_image = cv2.flip(source_image, 1)
                    derivative_lines, derivative_histogram = _mirror_directional_obb_label_lines(
                        source_lines
                    )
                    transform = "horizontal_mirror"
                else:
                    derivative_image = cv2.rotate(source_image, cv2.ROTATE_180)
                    derivative_lines, derivative_histogram = _rotate_bilateral_obb_label_lines_180(
                        source_lines
                    )
                    transform = "rotate_180"
            except (OSError, ValueError) as exc:
                return {
                    "ok": False,
                    "error": (
                        "Could not create a deterministic class-balanced training derivative for "
                        f"'{entry['source_image']}': {exc}"
                    ),
                    "warnings": warnings,
                }

            source_name = os.path.basename(entry["exported_image"])
            source_stem, source_extension = os.path.splitext(source_name)
            derivative_stem = (
                f"__train_class_balance_v{OBB_CLASS_BALANCED_TRAINING_DERIVATIVE_VERSION}__"
                f"{source_stem}"
            )
            derivative_image_path = os.path.join(
                out_dir, "images", "train", derivative_stem + source_extension
            )
            derivative_label_path = os.path.join(
                out_dir, "labels", "train", derivative_stem + ".txt"
            )
            if not safe_imwrite(derivative_image_path, derivative_image):
                return {
                    "ok": False,
                    "error": f"Could not write training derivative '{derivative_image_path}'.",
                    "warnings": warnings,
                }
            try:
                with open(derivative_label_path, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(derivative_lines) + "\n" if derivative_lines else "")
            except OSError as exc:
                return {
                    "ok": False,
                    "error": f"Could not write training derivative labels '{derivative_label_path}': {exc}",
                    "warnings": warnings,
                }
            _merge_class_histograms(training_derivative_histogram, derivative_histogram)
            training_derivatives.append(
                {
                    "derivative_id": hashlib.sha256(
                        (
                            f"obb-class-balanced-training-v"
                            f"{OBB_CLASS_BALANCED_TRAINING_DERIVATIVE_VERSION}\0"
                            f"{entry['sample_id']}\0{resolved_mode}"
                        ).encode("utf-8")
                    ).hexdigest(),
                    "source_sample_id": entry["sample_id"],
                    "source_group_id": entry["group_id"],
                    "transform": transform,
                    "class_transform": "binary_swap",
                    "image": os.path.relpath(derivative_image_path, out_dir).replace("\\", "/"),
                    "image_sha256": _sha256_file(derivative_image_path),
                    "label": os.path.relpath(derivative_label_path, out_dir).replace("\\", "/"),
                    "label_sha256": _sha256_file(derivative_label_path),
                    "class_histogram": {
                        str(class_id): int(count)
                        for class_id, count in sorted(derivative_histogram.items())
                    },
                }
            )

    # Directional schemas use the user-selected OBB arrow/class as their native
    # direction. Mirror each frozen real validation image deterministically so
    # the evaluator measures the opposite class without requiring duplicate
    # manual annotations. These derivatives remain validation-only and retain
    # their source group identity in the cohort manifest.
    evaluator_class_hist_by_split = {
        split_name: dict(histogram)
        for split_name, histogram in real_class_hist_by_split.items()
    }
    validation_derivatives = []
    if orientation_class_enabled and resolved_mode == "directional":
        validation_entries = sorted(
            (
                entry
                for entry in real_export_manifest
                if entry["split"] == "val"
            ),
            key=lambda entry: entry["sample_id"],
        )
        for entry in validation_entries:
            source_image_path = os.path.join(out_dir, entry["exported_image"])
            source_label_path = os.path.join(out_dir, entry["exported_label"])
            source_image = safe_imread(source_image_path)
            if source_image is None or source_image.size == 0:
                return {
                    "ok": False,
                    "error": (
                        "Could not decode a frozen validation image while creating its "
                        f"directional mirror: {entry['exported_image']}"
                    ),
                    "warnings": warnings,
                }
            try:
                with open(source_label_path, "r", encoding="utf-8") as handle:
                    source_lines = [line.strip() for line in handle if line.strip()]
                mirrored_lines, mirrored_histogram = _mirror_directional_obb_label_lines(
                    source_lines
                )
            except (OSError, ValueError) as exc:
                return {
                    "ok": False,
                    "error": (
                        "Could not create a deterministic directional validation mirror for "
                        f"'{entry['source_image']}': {exc}"
                    ),
                    "warnings": warnings,
                }

            source_export_name = os.path.basename(entry["exported_image"])
            source_stem, source_extension = os.path.splitext(source_export_name)
            derivative_stem = (
                f"__eval_directional_mirror_v{OBB_DIRECTIONAL_VALIDATION_MIRROR_VERSION}__"
                f"{source_stem}"
            )
            derivative_image_path = os.path.join(
                out_dir,
                "images",
                "val",
                derivative_stem + source_extension,
            )
            derivative_label_path = os.path.join(
                out_dir,
                "labels",
                "val",
                derivative_stem + ".txt",
            )
            if not safe_imwrite(derivative_image_path, cv2.flip(source_image, 1)):
                return {
                    "ok": False,
                    "error": (
                        "Could not write deterministic directional validation mirror "
                        f"'{derivative_image_path}'."
                    ),
                    "warnings": warnings,
                }
            try:
                with open(derivative_label_path, "w", encoding="utf-8") as handle:
                    handle.write(
                        "\n".join(mirrored_lines) + "\n" if mirrored_lines else ""
                    )
            except OSError as exc:
                return {
                    "ok": False,
                    "error": (
                        "Could not write deterministic directional validation labels "
                        f"'{derivative_label_path}': {exc}"
                    ),
                    "warnings": warnings,
                }
            for class_id, count in mirrored_histogram.items():
                evaluator_histogram = evaluator_class_hist_by_split["val"]
                evaluator_histogram[int(class_id)] = (
                    evaluator_histogram.get(int(class_id), 0) + int(count)
                )
            validation_derivatives.append(
                {
                    "derivative_id": hashlib.sha256(
                        (
                            f"obb-directional-validation-mirror-v"
                            f"{OBB_DIRECTIONAL_VALIDATION_MIRROR_VERSION}\0"
                            f"{entry['sample_id']}"
                        ).encode("utf-8")
                    ).hexdigest(),
                    "source_sample_id": entry["sample_id"],
                    "source_group_id": entry["group_id"],
                    "transform": "horizontal_mirror",
                    "class_transform": "binary_swap",
                    "image": os.path.relpath(derivative_image_path, out_dir).replace("\\", "/"),
                    "image_sha256": _sha256_file(derivative_image_path),
                    "label": os.path.relpath(derivative_label_path, out_dir).replace("\\", "/"),
                    "label_sha256": _sha256_file(derivative_label_path),
                    "class_histogram": {
                        str(class_id): int(count)
                        for class_id, count in sorted(mirrored_histogram.items())
                    },
                }
            )

    # Write dataset.yaml
    nc = 2 if orientation_class_enabled else 1
    names = class_names if orientation_class_enabled else ["specimen"]
    if orientation_class_enabled:
        expected_class_ids = set(range(nc))
        for split_name, scientific_name in (("val", "validation"), ("test", "test")):
            split_hist = evaluator_class_hist_by_split[split_name]
            split_image_count = sum(
                1 for entry in real_export_manifest if entry["split"] == split_name
            )
            missing_classes = sorted(expected_class_ids - set(split_hist))
            if split_image_count > 0 and missing_classes:
                warnings.append(
                    f"Frozen {scientific_name} OBB cohort lacks orientation class IDs "
                    f"{missing_classes} after schema-aware evaluation transforms; "
                    "its class-specific metrics will be incomplete."
                )
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    val_split_path = "images/train" if not val_set else "images/val"
    if not val_set:
        warnings.append(
            "Only 1 sample available; using training images as validation (val = train). "
            "Add more annotated images for a proper train/val split."
        )
    yaml_lines = [
        f"path: {out_dir}",
        "train: images/train",
        f"val: {val_split_path}",
        "test: images/test",
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
    n_real_train = sum(
        1 for i in range(len(samples)) if i not in val_set and i not in test_set
    )
    n_real_train_instances = sum(
        int(entry.get("boxes_exported", 0))
        for entry in real_export_manifest
        if entry["split"] == "train"
    )
    max_synth = max(0, int(n_real_train * SYNTHETIC_RATIO))

    synth_stats = {"num_generated": 0, "num_instances_generated": 0}
    synth_manifest_path = None
    if generate_synthetic and max_synth > 0:
        report_progress(
            "Checking synthetic augmentation inputs...",
            8.4,
            {"phase": "synthetic_prepare", "current": 0, "total": max_synth},
        )
        positives = [
            s
            for i, s in enumerate(samples)
            if i not in val_set and i not in test_set
        ]
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
            max_instances=n_real_train_instances,
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
        synth_manifest_path = os.path.join(out_dir, "synth_manifest.json")
        _atomic_write_json(synth_manifest_path, synth_manifest)
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

    effective_training_class_histogram = dict(real_class_hist_by_split["train"])
    _merge_class_histograms(
        effective_training_class_histogram,
        training_derivative_histogram,
    )
    _merge_class_histograms(
        effective_training_class_histogram,
        {
            int(class_id): int(count)
            for class_id, count in dict(synth_stats.get("class_histogram", {})).items()
        },
    )
    if orientation_class_enabled:
        missing_effective_classes = sorted(
            {0, 1}
            - {
                int(class_id)
                for class_id, count in effective_training_class_histogram.items()
                if int(count) > 0
            }
        )
        if missing_effective_classes:
            return {
                "ok": False,
                "error": (
                    "Effective OBB training data lacks required orientation class IDs "
                    f"{missing_effective_classes}; verify the saved OBB direction classes."
                ),
                "warnings": warnings,
            }

    synthetic_files = []
    synthetic_image_dir = os.path.join(out_dir, "images", "train")
    synthetic_label_dir = os.path.join(out_dir, "labels", "train")
    for image_name in sorted(os.listdir(synthetic_image_dir)):
        if not image_name.startswith("__synth_obb_"):
            continue
        image_path = os.path.join(synthetic_image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(synthetic_label_dir, label_name)
        if not os.path.isfile(image_path) or not os.path.isfile(label_path):
            return {
                "ok": False,
                "error": (
                    f"Synthetic OBB export is incomplete for '{image_name}'; both image and label "
                    "must exist before training."
                ),
            }
        synthetic_files.append(
            {
                "image": os.path.relpath(image_path, out_dir).replace("\\", "/"),
                "image_sha256": _sha256_file(image_path),
                "label": os.path.relpath(label_path, out_dir).replace("\\", "/"),
                "label_sha256": _sha256_file(label_path),
            }
        )

    effective_dataset_material = {
        "format_version": 1,
        "task": "obb",
        "dataset_yaml_sha256": _sha256_file(yaml_path),
        "real_files": sorted(
            (
                {
                    "sample_id": entry["sample_id"],
                    "split": entry["split"],
                    "image_sha256": entry["exported_image_sha256"],
                    "label_sha256": entry["label_sha256"],
                }
                for entry in real_export_manifest
            ),
            key=lambda entry: entry["sample_id"],
        ),
        "synthetic_files": synthetic_files,
        "synthetic_manifest_sha256": (
            _sha256_file(synth_manifest_path) if synth_manifest_path else None
        ),
        "validation_derivative_files": [
            {
                "derivative_id": entry["derivative_id"],
                "source_sample_id": entry["source_sample_id"],
                "image_sha256": entry["image_sha256"],
                "label_sha256": entry["label_sha256"],
                "transform": entry["transform"],
                "class_transform": entry["class_transform"],
            }
            for entry in validation_derivatives
        ],
        "training_derivative_files": [
            {
                "derivative_id": entry["derivative_id"],
                "source_sample_id": entry["source_sample_id"],
                "image_sha256": entry["image_sha256"],
                "label_sha256": entry["label_sha256"],
                "transform": entry["transform"],
                "class_transform": entry["class_transform"],
            }
            for entry in training_derivatives
        ],
    }
    effective_dataset = {
        **effective_dataset_material,
        "revision": _json_sha256(effective_dataset_material),
    }

    def build_exported_cohort(split_name, scientific_name, revision, frozen, report_only):
        split_class_histogram = {
            str(class_id): int(real_class_hist_by_split[split_name].get(class_id, 0))
            for class_id in range(nc)
        }
        evaluator_class_histogram = {
            str(class_id): int(
                evaluator_class_hist_by_split[split_name].get(class_id, 0)
            )
            for class_id in range(nc)
        }
        members = sorted(
            (
                {
                    "sample_id": entry["sample_id"],
                    "group_id": entry["group_id"],
                    "source_content_sha256": entry["content_sha256"],
                    "exported_image_sha256": entry["exported_image_sha256"],
                    "label_sha256": entry["label_sha256"],
                }
                for entry in real_export_manifest
                if entry["split"] == split_name
            ),
            key=lambda entry: entry["sample_id"],
        )
        derivatives = (
            sorted(validation_derivatives, key=lambda entry: entry["derivative_id"])
            if split_name == "val"
            else []
        )
        derivation = (
            {
                "type": "horizontal_mirror",
                "version": OBB_DIRECTIONAL_VALIDATION_MIRROR_VERSION,
                "source": "frozen_real_validation",
                "class_transform": "binary_swap",
            }
            if split_name == "val" and resolved_mode == "directional"
            else None
        )
        format_version = 3 if derivation else 2
        material = {
            "format_version": format_version,
            "split_assignments_version": OBB_SPLIT_ASSIGNMENTS_VERSION,
            "split_profile_key": split_stats.get("profile_key"),
            "cohort": scientific_name,
            "expected_class_count": int(nc),
            "real_class_histogram": split_class_histogram,
            "evaluator_class_histogram": evaluator_class_histogram,
            "derivation": derivation,
            "derivatives": derivatives,
            "members": members,
        }
        material_sha256 = _json_sha256(material)
        normalized_revision = str(revision or "").strip() or None
        return {
            "format_version": format_version,
            "revision": (
                f"{split_stats.get('profile_key')}@{normalized_revision}"
                if normalized_revision
                else None
            ),
            "sha256": normalized_revision,
            "export_manifest_sha256": material_sha256,
            "split_profile_key": split_stats.get("profile_key"),
            "sample_count": len(members),
            "evaluation_sample_count": len(members) + len(derivatives),
            "group_count": len({entry["group_id"] for entry in members}),
            "expected_class_count": int(nc),
            "real_class_histogram": split_class_histogram,
            "evaluator_class_histogram": evaluator_class_histogram,
            "derivation": derivation,
            "frozen": bool(frozen and normalized_revision),
            "report_only": bool(report_only),
            "derivatives": derivatives,
            "members": members,
        }

    validation_cohort = build_exported_cohort(
        "val",
        "validation",
        split_stats.get("validation_cohort_revision"),
        split_stats.get("validation_cohort_frozen", False),
        False,
    )
    test_cohort = build_exported_cohort(
        "test",
        "test",
        split_stats.get("test_cohort_revision"),
        split_stats.get("test_cohort_frozen", False),
        True,
    )
    cohort_manifest_path = os.path.join(out_dir, "cohort_manifest.json")
    cohort_manifest = {
        "format_version": 3 if resolved_mode == "directional" else 2,
        "split_profile_key": split_stats.get("profile_key"),
        "assignment_revision": split_stats.get("assignment_revision"),
        "disjoint": dict(split_stats.get("cohort_disjoint", {})),
        "adaptive_groups_train_only": bool(
            split_stats.get("adaptive_groups_train_only", False)
        ),
        "validation": validation_cohort,
        "test": test_cohort,
    }
    try:
        _atomic_write_json(cohort_manifest_path, cohort_manifest)
    except OSError as exc:
        return {
            "ok": False,
            "error": f"Could not write OBB cohort manifest '{cohort_manifest_path}': {exc}",
        }

    export_manifest_path = os.path.join(out_dir, "export_manifest.json")
    export_manifest = {
        "version": OBB_EXPORT_MANIFEST_VERSION,
        "task": "obb",
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "test_ratio": float(split_stats.get("test_ratio", val_ratio)),
        "split_profile_key": split_stats.get("profile_key"),
        "split_assignments_manifest": os.path.relpath(
            split_assignments_path,
            session_dir,
        ).replace("\\", "/"),
        "target_val_images": int(split_stats.get("target_val_images", 0)),
        "requested_ratio_target_val_images": int(
            split_stats.get("requested_ratio_target_val_images", 0)
        ),
        "actual_val_images": int(split_stats.get("actual_val_images", len(val_set))),
        "actual_test_images": int(split_stats.get("actual_test_images", len(test_set))),
        "actual_train_images": int(
            split_stats.get("actual_train_images", len(samples) - len(val_set) - len(test_set))
        ),
        "assignment_revision": split_stats.get("assignment_revision"),
        "cohort_disjoint": dict(split_stats.get("cohort_disjoint", {})),
        "cohort_manifest": os.path.relpath(cohort_manifest_path, out_dir).replace("\\", "/"),
        "validation_cohort_frozen": bool(
            split_stats.get("validation_cohort_frozen", False)
        ),
        "validation_cohort": validation_cohort,
        "test_cohort_frozen": bool(split_stats.get("test_cohort_frozen", False)),
        "test_cohort": test_cohort,
        "real_images": real_export_manifest,
        "training_derivatives": training_derivatives,
        "validation_derivatives": validation_derivatives,
        "synthetic": synth_stats,
        "synthetic_manifest": (
            {
                "path": os.path.relpath(synth_manifest_path, out_dir).replace("\\", "/"),
                "sha256": _sha256_file(synth_manifest_path),
            }
            if synth_manifest_path
            else None
        ),
        "effective_dataset": effective_dataset,
        "real_class_histogram_by_split": real_class_hist_by_split,
        "training_derivative_class_histogram": training_derivative_histogram,
        "effective_training_class_histogram": effective_training_class_histogram,
        "evaluator_class_histogram_by_split": evaluator_class_hist_by_split,
    }
    try:
        _atomic_write_json(export_manifest_path, export_manifest)
    except OSError as exc:
        return {
            "ok": False,
            "error": f"Could not write OBB export manifest '{export_manifest_path}': {exc}",
        }

    return {
        "ok": True,
        "yaml_path": yaml_path,
        "export_manifest_path": export_manifest_path,
        "cohort_manifest_path": cohort_manifest_path,
        "split_assignments_path": split_assignments_path,
        "synthetic_manifest_path": synth_manifest_path,
        "split_profile_key": split_stats.get("profile_key"),
        "validation_cohort_frozen": bool(
            split_stats.get("validation_cohort_frozen", False)
        ),
        "validation_cohort": validation_cohort,
        "validation_cohort_revision": validation_cohort["revision"],
        "validation_cohort_sha256": validation_cohort["sha256"],
        "test_cohort_frozen": bool(split_stats.get("test_cohort_frozen", False)),
        "test_cohort": test_cohort,
        "test_cohort_revision": test_cohort["revision"],
        "test_cohort_sha256": test_cohort["sha256"],
        "cohort_disjoint": dict(split_stats.get("cohort_disjoint", {})),
        "num_images": num_images,
        "num_boxes": num_boxes,
        "synthetic": synth_stats,
        "effective_dataset": effective_dataset,
        "minority_rule_applied": bool(split_stats.get("minority_rule_applied", False)),
        "minority_class_ids": list(split_stats.get("minority_class_ids", [])),
        "minority_total_instances": int(split_stats.get("minority_total_instances", 0)),
        "minority_val_instances": int(split_stats.get("minority_val_instances", 0)),
        "rotated_real_images_total": rotated_total,
        "rotated_real_images_val": rotated_val,
        "rotated_real_images_test": rotated_test,
        "warnings": warnings,
        "real_class_histogram": real_class_hist,
        "real_class_histogram_by_split": real_class_hist_by_split,
        "training_derivative_class_histogram": training_derivative_histogram,
        "effective_training_class_histogram": effective_training_class_histogram,
        "synthetic_class_histogram": synth_stats.get("class_histogram", {}),
        "training_derivatives": training_derivatives,
        "validation_derivatives": validation_derivatives,
        "evaluator_class_histogram_by_split": evaluator_class_hist_by_split,
    }
