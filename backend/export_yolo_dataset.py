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

from image_utils import safe_imread, safe_imwrite


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


def _load_landmark_ids(session_dir):
    """
    Read session.json and return sorted landmark indices used for pose export.
    """
    session_path = os.path.join(session_dir, "session.json")
    if not os.path.exists(session_path):
        return []
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
    except Exception:
        return []
    template = session.get("landmarkTemplate", [])
    if not isinstance(template, list):
        return []
    ids = []
    for lm in template:
        try:
            ids.append(int(lm.get("index")))
        except Exception:
            continue
    return sorted(set(ids))


def _compute_box_keypoints(box_dict, img_w, img_h, head_id, tail_id):
    """
    Compute normalized head and tail keypoints from box landmarks.
    Returns (hx, hy, tx, ty) or None.
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
        head_lm = next((lm for lm in landmarks if lm.get("id") == head_id), None)
    if head_lm is None:
        head_lm = min(landmarks, key=lambda lm: lm.get("id", 0))

    tail_lm = None
    if tail_id is not None:
        tail_lm = next((lm for lm in landmarks if lm.get("id") == tail_id), None)
    if tail_lm is None:
        others = [lm for lm in landmarks if lm.get("id") != head_lm.get("id")]
        if not others:
            return None
        hx, hy = float(head_lm["x"]), float(head_lm["y"])
        tail_lm = max(others, key=lambda lm: math.hypot(lm["x"] - hx, lm["y"] - hy))

    hx = _clamp(float(head_lm["x"]) / img_w, 0.0, 1.0)
    hy = _clamp(float(head_lm["y"]) / img_h, 0.0, 1.0)
    tx = _clamp(float(tail_lm["x"]) / img_w, 0.0, 1.0)
    ty = _clamp(float(tail_lm["y"]) / img_h, 0.0, 1.0)
    return hx, hy, tx, ty


def _compute_box_keypoints_full(box_dict, img_w, img_h, landmark_ids):
    """
    Compute normalized keypoints for all landmark_ids in session template order.
    Returns (flat_triplets, visible_count).
    flat_triplets format: [x1, y1, v1, x2, y2, v2, ...] where v in {0,2}.
    """
    if not landmark_ids:
        return [], 0
    landmarks = [
        lm for lm in box_dict.get("landmarks", [])
        if not lm.get("isSkipped")
        and lm.get("x", -1) >= 0
        and lm.get("y", -1) >= 0
    ]
    by_id = {}
    for lm in landmarks:
        try:
            lm_id = int(lm.get("id"))
            by_id[lm_id] = lm
        except Exception:
            continue

    values = []
    visible = 0
    for lm_id in landmark_ids:
        lm = by_id.get(int(lm_id))
        if lm is None:
            values.extend([0.0, 0.0, 0])
            continue
        x = _clamp(float(lm["x"]) / img_w, 0.0, 1.0)
        y = _clamp(float(lm["y"]) / img_h, 0.0, 1.0)
        values.extend([x, y, 2])
        visible += 1
    return values, visible


def _compute_box_orientation(box_dict, head_id, tail_id):
    """
    Determine left/right specimen orientation from box landmarks.

    Returns:
      "left" | "right" | None
    """
    override = str(box_dict.get("orientation_override", "")).strip().lower()
    if override in {"left", "right"}:
        return override
    if override == "uncertain":
        return None

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
        if orientation_override_raw in {"left", "right", "uncertain"}
        else None
    )
    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        **({"orientation_override": orientation_override} if orientation_override else {}),
        "landmarks": _normalize_landmarks(box.get("landmarks", [])),
    }


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
                    if not nb.get("landmarks"):
                        sig = _box_signature(nb)
                        candidates = draft_by_signature.get(sig, []) if sig else []
                        if candidates:
                            best = max(
                                candidates,
                                key=lambda x: len(x.get("landmarks", [])),
                            )
                            if best.get("landmarks"):
                                nb["landmarks"] = [dict(lm) for lm in best["landmarks"]]
                            if not nb.get("orientation_override") and best.get("orientation_override"):
                                nb["orientation_override"] = best.get("orientation_override")
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


def _sample_orientation_classes(sample, head_id, tail_id):
    classes = set()
    for box in sample.get("boxes", []):
        orientation = _compute_box_orientation(box, head_id, tail_id)
        if orientation == "right":
            classes.add(1)
        elif orientation == "left":
            classes.add(0)
    return classes


def _select_val_indices(
    export_records,
    val_ratio,
    seed,
    orientation_class_enabled=False,
    head_id=None,
    tail_id=None,
):
    total = len(export_records)
    if total <= 0:
        return set(), 0

    val_count = max(1, int(total * val_ratio))
    val_count = min(val_count, total)
    all_indices = list(range(total))
    rng = random.Random(seed)

    if not orientation_class_enabled:
        rng.shuffle(all_indices)
        return set(all_indices[:val_count]), val_count

    positive_indices = [i for i, r in enumerate(export_records) if r.get("kind") == "positive"]
    negative_indices = [i for i, r in enumerate(export_records) if r.get("kind") != "positive"]

    target_pos = 0
    if positive_indices:
        target_pos = max(1, int(round(len(positive_indices) * val_ratio)))
        target_pos = min(target_pos, len(positive_indices), val_count)

    right_pos = []
    non_right_pos = []
    for idx in positive_indices:
        sample = export_records[idx].get("sample", {})
        classes = _sample_orientation_classes(sample, head_id, tail_id)
        if 1 in classes:
            right_pos.append(idx)
        else:
            non_right_pos.append(idx)

    selected = []
    if target_pos > 0:
        if right_pos and non_right_pos and target_pos >= 2:
            right_target = int(round(target_pos * (len(right_pos) / float(max(1, len(positive_indices))))))
            right_target = max(1, min(len(right_pos), right_target))
            non_right_target = target_pos - right_target
            if non_right_target <= 0:
                non_right_target = 1
                right_target = max(1, target_pos - 1)
            if non_right_target > len(non_right_pos):
                non_right_target = len(non_right_pos)
                right_target = min(len(right_pos), target_pos - non_right_target)

            rng.shuffle(right_pos)
            rng.shuffle(non_right_pos)
            selected.extend(right_pos[:right_target])
            selected.extend(non_right_pos[:non_right_target])
        else:
            pool = positive_indices[:]
            rng.shuffle(pool)
            selected.extend(pool[:target_pos])

    remaining = val_count - len(selected)
    if remaining > 0:
        neg_pool = [i for i in negative_indices if i not in selected]
        rng.shuffle(neg_pool)
        selected.extend(neg_pool[:remaining])

    if len(selected) < val_count:
        fallback_pool = [i for i in all_indices if i not in selected]
        rng.shuffle(fallback_pool)
        selected.extend(fallback_pool[: (val_count - len(selected))])

    return set(selected), val_count


def _count_label_class_instances(label_dir):
    class_counts = {}
    total_instances = 0
    for txt_path in glob.glob(os.path.join(label_dir, "*.txt")):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            continue
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            total_instances += 1
    return {"total_instances": total_instances, "class_counts": class_counts}


def _move_synthetic_right_to_val(out_dir, target_right_instances=8, seed=0):
    train_lbl_dir = os.path.join(out_dir, "labels", "train")
    train_img_dir = os.path.join(out_dir, "images", "train")
    val_lbl_dir = os.path.join(out_dir, "labels", "val")
    val_img_dir = os.path.join(out_dir, "images", "val")
    os.makedirs(val_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    candidates = []
    for lbl_path in glob.glob(os.path.join(train_lbl_dir, "__synth_*.txt")):
        base = os.path.splitext(os.path.basename(lbl_path))[0]
        right_count = 0
        total = 0
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            continue
        for ln in lines:
            parts = ln.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                continue
            total += 1
            if class_id == 1:
                right_count += 1
        if right_count > 0 and total > 0:
            candidates.append(
                {
                    "base": base,
                    "label_path": lbl_path,
                    "right_count": right_count,
                    "total": total,
                }
            )

    rng = random.Random(seed)
    rng.shuffle(candidates)

    moved_images = 0
    moved_right_instances = 0
    moved_basenames = []
    target = max(1, int(target_right_instances))

    for c in candidates:
        if moved_right_instances >= target:
            break
        base = c["base"]
        src_lbl = c["label_path"]
        src_img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = os.path.join(train_img_dir, base + ext)
            if os.path.exists(candidate):
                src_img = candidate
                break
        if src_img is None:
            continue

        try:
            dst_lbl = os.path.join(val_lbl_dir, os.path.basename(src_lbl))
            dst_img = os.path.join(val_img_dir, os.path.basename(src_img))
            shutil.move(src_lbl, dst_lbl)
            shutil.move(src_img, dst_img)
            moved_images += 1
            moved_right_instances += int(c["right_count"])
            moved_basenames.append(base)
        except Exception:
            continue

    return {
        "moved_images": moved_images,
        "moved_right_instances": moved_right_instances,
        "moved_basenames": moved_basenames,
    }


# -----------------------------------------------------------------------------
# Synthetic generation from finalized segment crops
# -----------------------------------------------------------------------------

def _collect_finalized_segments(session_dir, anchor_index=None):
    """
    Collect finalized SAM2 segments (RGBA).
    Only segments with accepted_by_user=true are used.

    When anchor_index is provided, attempts to attach head/tail anchor points
    (in segment crop coordinates) for orientation-aware synthetic labeling.
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
        except Exception:
            continue

        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        if fg is None or fg.ndim != 3 or fg.shape[2] != 4:
            continue
        if np.count_nonzero(fg[:, :, 3] > 10) < 20:
            continue
        seg_entry = {"id": base, "rgba": fg}

        # Optional head/tail anchors from finalized accepted boxes.
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
                head_xy = best.get("head_xy")
                tail_xy = best.get("tail_xy")
                if head_xy and tail_xy:
                    seg_entry["head_tail_fg"] = (
                        (float(head_xy[0]) - cx1, float(head_xy[1]) - cy1),
                        (float(tail_xy[0]) - cx1, float(tail_xy[1]) - cy1),
                    )
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


def _augment_segment_chip(chip_rgba, rng,
                          scale_range=(0.65, 1.35),
                          rot_range=(-60.0, 60.0),
                          flip_prob=0.5,
                          head_tail_chip=None):
    """
    Apply orientation and shape augmentations to one chip.

    Returns:
      aug_rgba, head_tail_aug
    """
    aug = chip_rgba.copy()
    points = None
    if head_tail_chip is not None:
        try:
            (hx, hy), (tx, ty) = head_tail_chip
            points = np.array([[float(hx), float(hy)], [float(tx), float(ty)]], dtype=np.float32)
        except Exception:
            points = None

    if rng.random() < flip_prob:
        h0, w0 = aug.shape[:2]
        aug = cv2.flip(aug, 1)
        if points is not None:
            points[:, 0] = (w0 - 1) - points[:, 0]

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

    return aug, head_tail_aug


def _random_canvas_background(width, height, rng):
    """
    Create synthetic background (no original-image pixels).
    """
    mode = rng.choice(["solid", "gradient", "noise"])
    if mode == "solid":
        c = rng.randint(35, 220)
        bg = np.full((height, width, 3), c, dtype=np.uint8)
        return bg

    if mode == "gradient":
        c1 = np.array([rng.randint(20, 200) for _ in range(3)], dtype=np.float32)
        c2 = np.array([rng.randint(20, 200) for _ in range(3)], dtype=np.float32)
        horizontal = rng.random() < 0.5
        if horizontal:
            t = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
            bg = c1 * (1.0 - t) + c2 * t
            bg = np.repeat(bg, height, axis=0)
        else:
            t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
            bg = c1 * (1.0 - t) + c2 * t
            bg = np.repeat(bg, width, axis=1)
        return np.clip(bg, 0, 255).astype(np.uint8)

    # noise
    base = np.full((height, width, 3), rng.randint(30, 200), dtype=np.uint8)
    noise = np.random.default_rng(rng.randint(0, 10_000_000)).integers(
        -25, 26, size=(height, width, 3), dtype=np.int16
    )
    bg = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(bg, (5, 5), 0)


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
        a = (chip_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
        blended = fg * a + roi * (1.0 - a)
        canvas[y:y + h, x:x + w] = blended.astype(np.uint8)
        placed_boxes.append(cand)
        return {"bbox": cand, "offset": (x, y)}

    return None


def _generate_synthetic_images(
    session_dir,
    out_dir,
    positives,
    head_id=None,
    tail_id=None,
    orientation_class_enabled=False,
    n_per_segment=8,
    max_objects=4,
    max_total_images=None,
    max_total_instances=None,
    seed=0,
    split="train",
    manifest=None,
):
    """
    Generate synthetic train images from finalized segment crops only.
    Uses synthetic backgrounds + strict non-overlapping multi-instance placement.
    """
    anchor_index = _build_anchor_index(positives or [], head_id, tail_id)
    segments, seg_stats = _collect_finalized_segments(session_dir, anchor_index=anchor_index)
    if not segments:
        return {
            "num_generated": 0,
            "num_left": 0,
            "num_right": 0,
            "num_ambiguous_skipped": 0,
            "num_missing_anchor_skipped": 0,
            **seg_stats,
        }

    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    rng = random.Random(seed)
    n_generated = 0
    n_instances_generated = 0
    n_left = 0
    n_right = 0
    n_ambiguous_skipped = 0
    n_missing_anchor_skipped = 0
    total_iters = max(1, int(n_per_segment)) * len(segments)

    for i in range(total_iters):
        if max_total_images is not None and n_generated >= int(max_total_images):
            break
        if max_total_instances is not None and n_instances_generated >= int(max_total_instances):
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
            head_tail_fg = segments[seg_idx].get("head_tail_fg")
            if orientation_class_enabled and head_tail_fg is None:
                n_missing_anchor_skipped += 1
                continue

            chip, head_tail_chip = _prepare_segment_chip(
                raw,
                pad_ratio=rng.uniform(0.12, 0.28),
                head_tail_fg=head_tail_fg,
            )
            aug, head_tail_aug = _augment_segment_chip(
                chip,
                rng,
                rot_range=(-60.0, 60.0),
                head_tail_chip=head_tail_chip,
            )

            class_id = 0
            if orientation_class_enabled:
                if head_tail_aug is None:
                    n_missing_anchor_skipped += 1
                    continue
                (hx, _hy), (tx, _ty) = head_tail_aug
                dx = float(hx) - float(tx)
                ambiguous_dx_px = max(6.0, 0.02 * float(canvas_size))
                if abs(dx) < ambiguous_dx_px:
                    n_ambiguous_skipped += 1
                    continue
                class_id = 0 if dx < 0.0 else 1

            # Keep chips within a useful relative size range.
            ch, cw = bg.shape[:2]
            ah, aw = aug.shape[:2]
            max_frac = 0.45
            if aw >= int(cw * max_frac) or ah >= int(ch * max_frac):
                scale = min((cw * max_frac) / max(aw, 1), (ch * max_frac) / max(ah, 1))
                nw = max(8, int(round(aw * scale)))
                nh = max(8, int(round(ah * scale)))
                aug = cv2.resize(aug, (nw, nh), interpolation=cv2.INTER_LINEAR)
                if head_tail_aug is not None:
                    sx = float(nw) / max(1.0, float(aw))
                    sy = float(nh) / max(1.0, float(ah))
                    (hx, hy), (tx, ty) = head_tail_aug
                    head_tail_aug = ((hx * sx, hy * sy), (tx * sx, ty * sy))

            placement = _place_chip(bg, aug, placed, rng, min_gap_px=8, max_attempts=100)
            if placement is None:
                continue

            bbox = placement["bbox"]
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 1 or bh <= 1:
                continue

            cx_n = _clamp((x1 + bw / 2.0) / cw, 0.0, 1.0)
            cy_n = _clamp((y1 + bh / 2.0) / ch, 0.0, 1.0)
            w_n = _clamp(bw / cw, 0.0, 1.0)
            h_n = _clamp(bh / ch, 0.0, 1.0)

            labels.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
            offset_x, offset_y = placement["offset"]
            head_abs = None
            tail_abs = None
            if head_tail_aug is not None:
                (hx, hy), (tx, ty) = head_tail_aug
                head_abs = [round(float(offset_x + hx), 2), round(float(offset_y + hy), 2)]
                tail_abs = [round(float(offset_x + tx), 2), round(float(offset_y + ty), 2)]
            object_manifest.append(
                {
                    "split": split,
                    "image": f"__synth_{i:05d}.jpg",
                    "segment_id": segments[seg_idx]["id"],
                    "class_id": int(class_id),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "head_xy": head_abs,
                    "tail_xy": tail_abs,
                }
            )
            if orientation_class_enabled:
                if class_id == 0:
                    n_left += 1
                else:
                    n_right += 1

        min_required = 2 if len(segments) > 1 else 1
        if len(labels) < min_required:
            continue
        if (
            max_total_instances is not None
            and n_instances_generated + len(labels) > int(max_total_instances)
            and n_generated > 0
        ):
            break

        synth_name = f"__synth_{i:05d}"
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
        "num_left": n_left,
        "num_right": n_right,
        "num_ambiguous_skipped": n_ambiguous_skipped,
        "num_missing_anchor_skipped": n_missing_anchor_skipped,
        **seg_stats,
    }


# -----------------------------------------------------------------------------
# Main export
# -----------------------------------------------------------------------------

def export_dataset(
    session_dir,
    class_name,
    val_ratio=0.2,
    seed=42,
    return_details=False,
    finalized_only=True,
):
    """
    Export session annotations to YOLO/YOLO-Pose format.
    """
    labels_dir = os.path.join(session_dir, "labels")
    images_dir = os.path.join(session_dir, "images")

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    head_id, tail_id = _load_head_tail_ids(session_dir)
    landmark_ids = _load_landmark_ids(session_dir)
    finalized_set = _load_finalized_filenames(session_dir) if finalized_only else set()

    positives = []
    negatives = []
    skipped_unfinalized = 0
    finalized_label_files = 0
    finalized_fallback_to_boxes = 0

    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".json"):
            continue

        label_path = os.path.join(labels_dir, fname)
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        image_filename = data.get("imageFilename", "")
        if not image_filename:
            continue

        if finalized_only:
            is_finalized, boxes, used_fallback = _get_finalized_boxes(
                data, image_filename, finalized_set
            )
            if not is_finalized:
                skipped_unfinalized += 1
                continue
            if used_fallback:
                finalized_fallback_to_boxes += 1
            finalized_label_files += 1
        else:
            boxes = []
            for b in data.get("boxes", []):
                nb = _normalize_box(b)
                if nb:
                    boxes.append(nb)

        rejected = data.get("rejectedDetections", [])
        if not boxes and not rejected:
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

        sample_base = {"image_path": image_path, "image_filename": image_filename}
        if boxes:
            positives.append({**sample_base, "boxes": boxes})
        if isinstance(rejected, list) and rejected:
            negatives.append({**sample_base, "rejected": rejected})

    if not positives and not negatives:
        if finalized_only:
            raise ValueError(
                f"No finalized detection labels found in {labels_dir}. "
                f"Finalize accepted boxes before training."
            )
        raise ValueError(f"No annotated images found in {labels_dir}")

    image_cache = {}
    for sample in positives + negatives:
        if sample["image_path"] in image_cache:
            continue
        img = safe_imread(sample["image_path"])
        if img is None:
            raise ValueError(f"Failed to read image: {sample['image_path']}")
        h, w = img.shape[:2]
        image_cache[sample["image_path"]] = {
            "img": img,
            "img_width": w,
            "img_height": h,
        }

    for sample in positives:
        meta = image_cache[sample["image_path"]]
        sample["img_width"] = meta["img_width"]
        sample["img_height"] = meta["img_height"]

    total_box_count = 0
    orientation_labeled_boxes = 0
    real_orientation_left_boxes = 0
    real_orientation_right_boxes = 0
    real_orientation_unknown_boxes = 0
    for p in positives:
        for box in p["boxes"]:
            total_box_count += 1
            orientation = _compute_box_orientation(box, head_id, tail_id)
            if orientation in ("left", "right"):
                orientation_labeled_boxes += 1
                if orientation == "left":
                    real_orientation_left_boxes += 1
                else:
                    real_orientation_right_boxes += 1
            else:
                real_orientation_unknown_boxes += 1

    orientation_labeled_ratio = (
        float(orientation_labeled_boxes) / float(total_box_count)
        if total_box_count > 0
        else 0.0
    )
    orientation_class_enabled = bool(
        head_id is not None
        and tail_id is not None
        and total_box_count > 0
        and orientation_labeled_ratio >= 0.60
    )
    num_detection_classes = 2 if orientation_class_enabled else 1
    use_pose = False
    orientation_preflight_warnings = []
    if orientation_class_enabled:
        oriented_total = real_orientation_left_boxes + real_orientation_right_boxes
        if real_orientation_right_boxes == 0:
            orientation_preflight_warnings.append(
                "No real right-facing finalized boxes found; orientation detector may collapse to left on unseen data."
            )
        elif oriented_total > 0:
            min_minor_class = max(6, int(round(oriented_total * 0.10)))
            if real_orientation_right_boxes < min_minor_class:
                orientation_preflight_warnings.append(
                    "Real right-facing boxes are underrepresented; add more real right-facing finalized images for robust orientation generalization."
                )
            imbalance_ratio = max(
                float(real_orientation_left_boxes),
                float(real_orientation_right_boxes),
            ) / max(1.0, float(min(real_orientation_left_boxes, real_orientation_right_boxes)))
            if imbalance_ratio >= 6.0:
                orientation_preflight_warnings.append(
                    "Severe left/right class imbalance in real finalized boxes; reduce imbalance before relying on orientation class predictions."
                )

    export_records = []
    for p in positives:
        export_records.append({"kind": "positive", "sample": p})
    for n in negatives:
        for i, rej in enumerate(n["rejected"]):
            export_records.append({
                "kind": "negative",
                "sample": n,
                "rejected_index": i,
                "rejected_box": rej,
            })

    if not export_records:
        raise ValueError("No exportable records after filtering.")

    val_indices, val_count = _select_val_indices(
        export_records,
        val_ratio=val_ratio,
        seed=seed,
        orientation_class_enabled=orientation_class_enabled,
        head_id=head_id,
        tail_id=tail_id,
    )

    out_dir = os.path.join(session_dir, "yolo_dataset")
    _reset_output_dataset_dir(out_dir)

    num_positive_images = 0
    num_negative_crops = 0

    for i, record in enumerate(export_records):
        split = "val" if i in val_indices else "train"
        sample = record["sample"]
        img_meta = image_cache[sample["image_path"]]
        img = img_meta["img"]
        img_h = img_meta["img_height"]
        img_w = img_meta["img_width"]

        if record["kind"] == "positive":
            dest_img_name = sample["image_filename"]
            dest_img = os.path.join(out_dir, "images", split, dest_img_name)
            if not os.path.exists(dest_img):
                shutil.copy2(sample["image_path"], dest_img)

            label_basename = os.path.splitext(dest_img_name)[0] + ".txt"
            label_path = os.path.join(out_dir, "labels", split, label_basename)

            lines = []
            for box in sample["boxes"]:
                left = box.get("left", 0)
                top = box.get("top", 0)
                width = box.get("width", 0)
                height = box.get("height", 0)
                if width <= 0 or height <= 0:
                    continue

                x_center = _clamp((left + width / 2.0) / img_w, 0.0, 1.0)
                y_center = _clamp((top + height / 2.0) / img_h, 0.0, 1.0)
                norm_w = _clamp(width / img_w, 0.0, 1.0)
                norm_h = _clamp(height / img_h, 0.0, 1.0)

                class_id = 0
                if orientation_class_enabled:
                    orientation = _compute_box_orientation(box, head_id, tail_id)
                    if orientation == "right":
                        class_id = 1
                lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                )

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n" if lines else "")
            num_positive_images += 1
            continue

        # negative crop from user-rejected detections
        rej = record["rejected_box"]
        rx = _safe_int(rej.get("left", 0))
        ry = _safe_int(rej.get("top", 0))
        rw = _safe_int(rej.get("width", 0))
        rh = _safe_int(rej.get("height", 0))
        if rw <= 1 or rh <= 1:
            continue

        cx = rx + rw // 2
        cy = ry + rh // 2
        pad_w = max(8, int(rw * 0.2))
        pad_h = max(8, int(rh * 0.2))
        x1 = _clamp(cx - rw // 2 - pad_w, 0, img_w - 1)
        y1 = _clamp(cy - rh // 2 - pad_h, 0, img_h - 1)
        x2 = _clamp(cx + rw // 2 + pad_w, 1, img_w)
        y2 = _clamp(cy + rh // 2 + pad_h, 1, img_h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        base = os.path.splitext(sample["image_filename"])[0]
        dest_img_name = f"{base}__neg_{record['rejected_index']}.jpg"
        dest_img = os.path.join(out_dir, "images", split, dest_img_name)
        safe_imwrite(dest_img, crop)

        label_basename = os.path.splitext(dest_img_name)[0] + ".txt"
        label_path = os.path.join(out_dir, "labels", split, label_basename)
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("")
        num_negative_crops += 1

    synthetic_enabled = True
    synth_manifest = []
    synthetic_mode = "standard"
    synthetic_n_per_segment = 8
    synthetic_max_images = None
    synthetic_max_instances = None
    if orientation_class_enabled:
        # Keep synthetic as a supplement for orientation classes.
        synthetic_mode = "supplement"
        real_oriented_instances = max(1, real_orientation_left_boxes + real_orientation_right_boxes)
        synthetic_n_per_segment = 2
        synthetic_max_instances = max(24, int(round(real_oriented_instances * 0.75)))
        synthetic_max_images = max(12, int(round(max(1, len(positives)) * 0.50)))
        synthetic_max_images = min(synthetic_max_images, max(12, len(positives)))
    synthetic_stats = _generate_synthetic_images(
        session_dir,
        out_dir,
        positives=positives,
        head_id=head_id,
        tail_id=tail_id,
        orientation_class_enabled=orientation_class_enabled,
        n_per_segment=synthetic_n_per_segment,
        max_objects=4,
        max_total_images=synthetic_max_images,
        max_total_instances=synthetic_max_instances,
        seed=seed,
        split="train",
        manifest=synth_manifest,
    )
    num_synthetic = int(synthetic_stats.get("num_generated", 0))

    val_counts_before_balance = _count_label_class_instances(os.path.join(out_dir, "labels", "val"))
    val_right_before = int(val_counts_before_balance.get("class_counts", {}).get(1, 0))
    synthetic_val_balance = {
        "moved_images": 0,
        "moved_right_instances": 0,
        "moved_basenames": [],
    }
    if orientation_class_enabled and val_right_before <= 0:
        target_right = max(4, int(round(max(1, val_count) * 0.25)))
        synthetic_val_balance = _move_synthetic_right_to_val(
            out_dir,
            target_right_instances=target_right,
            seed=seed + 17,
        )
        moved = set(synthetic_val_balance.get("moved_basenames", []))
        if moved:
            for m in synth_manifest:
                image_name = str(m.get("image", ""))
                image_base = os.path.splitext(image_name)[0]
                if image_base in moved:
                    m["split"] = "val"

    synth_manifest_path = None
    if synth_manifest:
        synth_manifest_path = os.path.join(out_dir, "synth_manifest.json")
        try:
            with open(synth_manifest_path, "w", encoding="utf-8") as f:
                json.dump(synth_manifest, f, indent=2)
        except Exception:
            synth_manifest_path = None

    train_counts = _count_label_class_instances(os.path.join(out_dir, "labels", "train"))
    val_counts = _count_label_class_instances(os.path.join(out_dir, "labels", "val"))

    yaml_path = os.path.join(out_dir, "dataset.yaml")
    yaml_lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        "nc: 1",
        "",
        "names:",
        f"  0: {class_name}",
    ]
    if orientation_class_enabled:
        yaml_lines = [
            f"path: {out_dir}",
            "train: images/train",
            "val: images/val",
            "nc: 2",
            "",
            "names:",
            f"  0: {class_name}_left",
            f"  1: {class_name}_right",
        ]

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    stats = {
        "yaml_path": yaml_path,
        "finalized_only": bool(finalized_only),
        "skipped_unfinalized_images": skipped_unfinalized,
        "finalized_label_files": finalized_label_files,
        "finalized_fallback_to_boxes": finalized_fallback_to_boxes,
        "total_records": len(export_records),
        "val_records": val_count,
        "train_records": max(0, len(export_records) - val_count),
        "positive_images": num_positive_images,
        "negative_crops": num_negative_crops,
        "num_synthetic": num_synthetic,
        "synthetic_enabled": synthetic_enabled,
        "synthetic_disabled_reason": None,
        "synthetic_mode": synthetic_mode,
        "synthetic_n_per_segment": synthetic_n_per_segment,
        "synthetic_max_images": synthetic_max_images,
        "synthetic_max_instances": synthetic_max_instances,
        "synthetic_instances_generated": int(synthetic_stats.get("num_instances_generated", 0)),
        "synthetic_left_instances": int(synthetic_stats.get("num_left", 0)),
        "synthetic_right_instances": int(synthetic_stats.get("num_right", 0)),
        "synthetic_ambiguous_skipped": int(synthetic_stats.get("num_ambiguous_skipped", 0)),
        "synthetic_missing_anchor_skipped": int(synthetic_stats.get("num_missing_anchor_skipped", 0)),
        "synthetic_segments_total": int(synthetic_stats.get("segments_total", 0)),
        "synthetic_segments_with_anchors": int(synthetic_stats.get("segments_with_anchors", 0)),
        "synthetic_segments_missing_anchors": int(synthetic_stats.get("segments_missing_anchors", 0)),
        "synthetic_manifest_path": synth_manifest_path,
        "synthetic_val_balance_applied": bool(
            orientation_class_enabled and int(synthetic_val_balance.get("moved_images", 0)) > 0
        ),
        "synthetic_val_moved_images": int(synthetic_val_balance.get("moved_images", 0)),
        "synthetic_val_right_instances_added": int(synthetic_val_balance.get("moved_right_instances", 0)),
        "train_class_counts": train_counts.get("class_counts", {}),
        "val_class_counts": val_counts.get("class_counts", {}),
        "num_classes": num_detection_classes,
        "orientation_class_enabled": orientation_class_enabled,
        "orientation_labeled_boxes": orientation_labeled_boxes,
        "orientation_unlabeled_boxes": max(0, total_box_count - orientation_labeled_boxes),
        "orientation_labeled_ratio": round(orientation_labeled_ratio, 6),
        "real_orientation_left_boxes": real_orientation_left_boxes,
        "real_orientation_right_boxes": real_orientation_right_boxes,
        "real_orientation_unknown_boxes": real_orientation_unknown_boxes,
        "orientation_preflight_warnings": orientation_preflight_warnings,
        "use_pose": use_pose,
        "head_id": head_id,
        "tail_id": tail_id,
        "landmark_ids": landmark_ids,
        "pose_keypoint_count": len(landmark_ids),
        "pose_boxes_with_kp": 0,
        "total_boxes": total_box_count,
    }
    if return_details:
        return stats
    return yaml_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <session_dir> <class_name>", file=sys.stderr)
        sys.exit(1)

    session_dir_arg = sys.argv[1]
    class_name_arg = sys.argv[2]
    details = export_dataset(session_dir_arg, class_name_arg, return_details=True)
    print(json.dumps({"ok": True, **details}))
