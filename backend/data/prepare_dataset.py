import json
import os
import glob
import sys
import random
import re
import hashlib
import unicodedata
import math
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import sys as _sys, os as _os
_BACKEND_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _BACKEND_ROOT not in _sys.path:
    _sys.path.insert(0, _BACKEND_ROOT)

from detection.detect_specimen import detect_specimen
from bv_utils.image_utils import load_image, safe_imread, safe_imwrite
import bv_utils.orientation_utils as ou


def _ascii_safe_base(name):
    """
    Return an ASCII-only version of a filename base for dlib XML compatibility.
    dlib (C++) opens files directly using the OS API and cannot handle non-ASCII
    paths on Windows. If the name is already ASCII it is returned unchanged.
    Otherwise non-ASCII characters are replaced with underscores and a short MD5
    hash is appended to guarantee uniqueness across different Unicode filenames.
    """
    if name.isascii():
        return name
    clean = re.sub(r'[^\x20-\x7E]+', '_', name).strip('_') or 'image'
    h = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
    return f"{clean[:60]}_{h}"


def _resolve_image_path(images_dir, img_filename):
    """
    Resolve the actual path for img_filename, handling Unicode normalization
    differences (e.g. U+202F narrow no-break space vs regular space, NFC vs NFD).
    Returns the resolved path, or raises FileNotFoundError if no match is found.
    """
    exact = os.path.join(images_dir, img_filename)
    if os.path.exists(exact):
        return exact

    # Build a normalized index of files in images_dir for fuzzy matching
    try:
        candidates = os.listdir(images_dir)
    except OSError:
        raise FileNotFoundError(f"Image not found: {exact}")

    target_nfkc = unicodedata.normalize("NFKC", img_filename)
    for candidate in candidates:
        if unicodedata.normalize("NFKC", candidate) == target_nfkc:
            resolved = os.path.join(images_dir, candidate)
            print(
                f"[WARN] Unicode normalization mismatch — label says {repr(img_filename)!r}, "
                f"matched on disk as {repr(candidate)!r}. Consider fixing the label.",
                file=sys.stderr,
            )
            return resolved

    raise FileNotFoundError(f"Image not found: {exact}")

STANDARD_SIZE = ou.STANDARD_SIZE


def _norm_path(value):
    try:
        return os.path.normcase(os.path.abspath(str(value)))
    except Exception:
        return ""


def _box_to_xyxy(box):
    try:
        left = float(box.get("left", 0))
        top = float(box.get("top", 0))
        width = float(box.get("width", 0))
        height = float(box.get("height", 0))
        right = left + width
        bottom = top + height
        if right <= left:
            right = left + 1.0
        if bottom <= top:
            bottom = top + 1.0
        return [left, top, right, bottom]
    except Exception:
        return None


def _mask_iou(a_xyxy, b_xyxy):
    if not a_xyxy or not b_xyxy:
        return 0.0
    ax1, ay1, ax2, ay2 = a_xyxy
    bx1, by1, bx2, by2 = b_xyxy
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(a_area + b_area - inter, 1e-6)
    return float(inter / union)


def _sanitize_binary_mask(mask_u8, min_area=32):
    if mask_u8 is None:
        return None
    try:
        mask = (mask_u8 > 0).astype("uint8")
        if int(mask.sum()) <= 0:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < float(min_area):
            return None
        out = np.zeros_like(mask, dtype="uint8")
        cv2.drawContours(out, [largest], -1, 1, thickness=-1)
        return out.astype("uint8")
    except Exception:
        return None


def _rough_mask_from_crop(crop_512):
    try:
        gray = cv2.cvtColor(crop_512, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if int((mask > 0).sum()) > (mask.size // 2):
            mask = cv2.bitwise_not(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype="uint8"), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype="uint8"), iterations=2)
        return _sanitize_binary_mask(mask, min_area=32)
    except Exception:
        return None


def _build_segment_index(project_root):
    seg_dir = os.path.join(project_root, "segments")
    if not os.path.isdir(seg_dir):
        return {}

    index = {}
    for fname in sorted(os.listdir(seg_dir)):
        if not fname.endswith("_meta.json"):
            continue
        meta_path = os.path.join(seg_dir, fname)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        if not bool(meta.get("accepted_by_user", False)):
            continue
        src = meta.get("source_image")
        box = meta.get("box")
        crop_origin = meta.get("crop_origin")
        if not src or not isinstance(box, dict) or not isinstance(crop_origin, (list, tuple)) or len(crop_origin) < 2:
            continue
        base = fname[:-10]  # strip _meta.json
        mask_path = os.path.join(seg_dir, f"{base}_mask.png")
        if not os.path.exists(mask_path):
            continue
        left = float(box.get("left", 0))
        top = float(box.get("top", 0))
        right = float(box.get("right", left + 1))
        bottom = float(box.get("bottom", top + 1))
        entry = {
            "source_image": str(src),
            "source_norm": _norm_path(src),
            "source_base": os.path.basename(str(src)),
            "box_xyxy": [left, top, max(right, left + 1.0), max(bottom, top + 1.0)],
            "crop_origin": [float(crop_origin[0]), float(crop_origin[1])],
            "mask_path": mask_path,
        }
        index.setdefault(entry["source_norm"], []).append(entry)
        index.setdefault(entry["source_base"], []).append(entry)
    return index


def _segment_mask_for_box(image_shape, image_path, box_xyxy, scale, segment_index, min_iou=0.30):
    if not segment_index:
        return None
    key_abs = _norm_path(image_path)
    key_base = os.path.basename(str(image_path))
    candidates = []
    if key_abs in segment_index:
        candidates.extend(segment_index[key_abs])
    if key_base in segment_index:
        candidates.extend(segment_index[key_base])
    if not candidates:
        return None

    best = None
    best_iou = 0.0
    for c in candidates:
        bx = c["box_xyxy"]
        sbx = [float(bx[0]) * scale, float(bx[1]) * scale, float(bx[2]) * scale, float(bx[3]) * scale]
        iou = _mask_iou(box_xyxy, sbx)
        if iou > best_iou:
            best_iou = iou
            best = c
    if best is None or best_iou < float(min_iou):
        return None

    try:
        mask = cv2.imread(best["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if scale != 1.0:
            new_w = max(1, int(round(mask.shape[1] * scale)))
            new_h = max(1, int(round(mask.shape[0] * scale)))
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask = _sanitize_binary_mask(mask, min_area=16)
        if mask is None:
            return None

        img_h, img_w = image_shape[:2]
        full = np.zeros((img_h, img_w), dtype="uint8")
        ox = int(round(float(best["crop_origin"][0]) * scale))
        oy = int(round(float(best["crop_origin"][1]) * scale))
        h, w = mask.shape[:2]
        x1 = max(0, ox)
        y1 = max(0, oy)
        x2 = min(img_w, ox + w)
        y2 = min(img_h, oy + h)
        if x2 <= x1 or y2 <= y1:
            return None
        mx1 = max(0, x1 - ox)
        my1 = max(0, y1 - oy)
        mx2 = mx1 + (x2 - x1)
        my2 = my1 + (y2 - y1)
        full[y1:y2, x1:x2] = (mask[my1:my2, mx1:mx2] > 0).astype("uint8")
        return _sanitize_binary_mask(full, min_area=32)
    except Exception:
        return None


def _normalize_categories(categories, fallback):
    if isinstance(categories, (list, tuple)):
        values = [str(c).strip().lower() for c in categories if str(c).strip()]
        if values:
            return values
    return [str(c).strip().lower() for c in fallback if str(c).strip()]


def _resolve_head_landmark_id(project_root, head_categories=None, fallback_to_min=True):
    """
    Resolve the landmark ID used as "head" for orientation detection.

    Preference:
    1) First landmark in session.json whose category is "head".
    2) Minimum landmark index in session.json.
    3) None (caller falls back to min ID from observed landmarks).
    """
    session_path = os.path.join(project_root, "session.json")
    if not os.path.exists(session_path):
        return None

    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
    except Exception:
        return None

    template = session.get("landmarkTemplate", [])
    if not isinstance(template, list):
        return None

    head_targets = set(_normalize_categories(head_categories, ["head"]))
    head_candidates = []
    all_indices = []
    for lm in template:
        try:
            idx = int(lm.get("index"))
        except (TypeError, ValueError):
            continue
        all_indices.append(idx)
        if str(lm.get("category", "")).strip().lower() in head_targets:
            head_candidates.append(idx)

    if head_candidates:
        return min(head_candidates)
    if fallback_to_min and all_indices:
        return min(all_indices)
    return None


def _resolve_tail_landmark_id(project_root, tail_categories=None):
    """Resolve the landmark ID used as 'tail' for orientation detection."""
    session_path = os.path.join(project_root, "session.json")
    if not os.path.exists(session_path):
        return None

    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
    except Exception:
        return None

    template = session.get("landmarkTemplate", [])
    if not isinstance(template, list):
        return None

    tail_targets = set(_normalize_categories(tail_categories, ["tail"]))
    tail_candidates = []
    for lm in template:
        try:
            idx = int(lm.get("index"))
        except (TypeError, ValueError):
            continue
        if str(lm.get("category", "")).strip().lower() in tail_targets:
            tail_candidates.append(idx)

    if tail_candidates:
        return min(tail_candidates)
    return None


def _mirror_x(x, width):
    """Mirror x in [0, width-1] across the vertical axis."""
    mirrored = (width - 1) - x
    if mirrored < 0:
        return 0.0
    max_x = float(width - 1)
    if mirrored > max_x:
        return max_x
    return mirrored


def standardize_crop(
    image,
    box,
    landmarks,
    pad_ratio=0.20,
    mirror=False,
    orientation_policy=None,
    mask_full=None,
    orientation_hint=None,
):
    """
    Crop image to bounding box + padding, resize to STANDARD_SIZE x STANDARD_SIZE,
    and remap landmarks to the new coordinate space.

    If mirror=True the cropped image is flipped horizontally and landmark x-coords
    are mirrored within the 512x512 crop space. The flip remains local to each
    specimen crop.

    When the box has obbCorners, the OBB geometry engine is used instead of the
    PCA/AABB path: extract_obb_crop() deskews the specimen, apply_obb_geometry()
    applies schema-specific canonical flips, and landmarks are remapped via the
    same affine matrix M that was applied to the image pixels.

    Returns (cropped_image, remapped_landmarks, crop_metadata).
    """
    # --- OBB fast-path (takes priority when obbCorners are present) ---
    obb_corners = box.get("obbCorners") or box.get("obb_corners")
    class_id = box.get("class_id", 0)
    if obb_corners and len(obb_corners) == 4:
        apply_leveling = ((orientation_policy or {}).get("obbLevelingMode", "on") == "on")
        crop, crop_meta = ou.extract_obb_crop(
            image, obb_corners, pad_ratio=0.15, apply_leveling=apply_leveling
        )
        crop, crop_meta, debug = ou.apply_obb_geometry(
            crop, crop_meta, int(class_id), orientation_policy or {}
        )

        # Affine-correct landmark remap:
        # Step 1 — Rotate landmark coords by M (the same warpAffine applied to pixels).
        # Guard: empty landmarks (box drawn but no points placed) → skip dot-product.
        M = np.array(crop_meta["affine_M"], dtype=np.float64)  # 2×3
        lm_pts = np.array([[lm["x"], lm["y"]] for lm in landmarks], dtype=np.float64)
        if apply_leveling and len(lm_pts):
            ones = np.ones((len(lm_pts), 1), dtype=np.float64)
            rotated_pts = (M @ np.hstack([lm_pts, ones]).T).T  # shape (N, 2)
        else:
            rotated_pts = lm_pts  # no rotation (leveling off) or empty landmark set

        # Step 2 — Shift by crop_origin, scale to STANDARD_SIZE, mirror/rotate if flipped.
        ox, oy = crop_meta["crop_origin"]
        scale = crop_meta["scale"]
        was_flipped = bool(crop_meta.get("canonical_flip_applied", False))
        was_rotated_180 = bool(debug.get("rotated_180", False))
        landmarks_512 = []
        for i, lm in enumerate(landmarks):
            x512 = (rotated_pts[i][0] - ox) * scale
            y512 = (rotated_pts[i][1] - oy) * scale
            if was_flipped:
                x512 = (STANDARD_SIZE - 1) - x512  # horizontal mirror (directional/bilateral)
            if was_rotated_180:
                # cv2.ROTATE_180 → both axes mirrored (axial mode, large OBB angle)
                x512 = (STANDARD_SIZE - 1) - x512
                y512 = (STANDARD_SIZE - 1) - y512
            x512 = float(np.clip(x512, 0, STANDARD_SIZE - 1))
            y512 = float(np.clip(y512, 0, STANDARD_SIZE - 1))
            landmarks_512.append({**lm, "x": x512, "y": y512})

        meta = {
            **crop_meta,
            "original_box": {
                "left": box.get("left", 0),
                "top": box.get("top", 0),
                "width": box.get("width", 0),
                "height": box.get("height", 0),
            },
            "mirrored": bool(was_flipped) ^ bool(mirror),
            "manual_mirror_requested": bool(mirror),
            "canonical_flip_applied": was_flipped,
            "canonicalization": debug,
            "canonical_mask_source": "obb",
        }
        return crop, landmarks_512, meta

    # --- Existing PCA/AABB path (boxes without obbCorners) ---
    bx, by = int(box["left"]), int(box["top"])
    bw, bh = int(box["width"]), int(box["height"])
    xyxy = [bx, by, bx + bw, by + bh]

    standardized, metadata = ou.base_standardize(image, xyxy, pad_ratio=pad_ratio)
    canonicalization = None
    mask_source = None
    canonical_flip_applied = False

    mode = ou.get_orientation_mode(orientation_policy or {})
    pca_mode = str((orientation_policy or {}).get("pcaLevelingMode", "off")).strip().lower()
    if pca_mode not in ("off", "on", "auto"):
        pca_mode = "off"
    should_canonicalize = mode != "invariant" and pca_mode in ("on", "auto")

    if should_canonicalize:
        mask_512 = None
        if mask_full is not None:
            try:
                ox, oy = [int(v) for v in metadata["crop_origin"]]
                cw, ch = [int(v) for v in metadata["crop_size"]]
                if cw > 0 and ch > 0:
                    mask_crop = mask_full[oy:oy + ch, ox:ox + cw]
                    if mask_crop.size > 0:
                        mask_512 = cv2.resize(
                            (mask_crop > 0).astype("uint8"),
                            (STANDARD_SIZE, STANDARD_SIZE),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        mask_source = "segments"
            except Exception:
                mask_512 = None
        if mask_512 is None:
            mask_512 = _rough_mask_from_crop(standardized)
            if mask_512 is not None:
                mask_source = "rough_otsu"

        if mask_512 is not None:
            standardized, _rot_mask, canonicalization = ou.canonicalize_with_mask(
                standardized,
                mask_512,
                policy=orientation_policy or {},
                pca_mode="auto",
                orientation_hint=orientation_hint,
            )
            canonical_flip_applied = bool((canonicalization or {}).get("canonical_flip_applied", False))

    # Keep image and landmark coordinates in the same mirrored frame.
    # canonical_flip_applied is already baked into `standardized`.
    # `mirror` here is an optional additional legacy flip.
    total_flip = bool(canonical_flip_applied) ^ bool(mirror)
    remapped = ou.remap_landmarks_to_standard(landmarks, metadata, mirror=total_flip)
    if mirror:
        standardized = cv2.flip(standardized, 1)

    metadata = {
        **metadata,
        "original_box": {"left": bx, "top": by, "width": bw, "height": bh},
        "mirrored": total_flip,
        "manual_mirror_requested": bool(mirror),
        "canonical_flip_applied": canonical_flip_applied,
        "canonicalization": canonicalization,
        "canonical_mask_source": mask_source,
    }
    return standardized, remapped, metadata


def detect_orientation(landmarks, head_id=None, tail_id=None):
    """Detect specimen orientation using the shared orientation utility."""
    return ou.detect_orientation(landmarks, head_id=head_id, tail_id=tail_id)



def _dataset_size_bucket(n_samples):
    n = int(max(0, n_samples))
    if n < 120:
        return "tiny"
    if n < 300:
        return "small"
    if n < 700:
        return "medium"
    if n < 1500:
        return "large"
    return "xlarge"


def _resolve_training_prep_box_jitter_profile(orientation_policy, orientation_mode, train_count):
    """
    Resolve schema-aware training-prep box jitter profile.

    Defaults to conservative dlib-safe jitter because this dataset feeds both
    dlib and CNN training.
    """
    raw = (orientation_policy or {}).get("trainingPrepBoxJitter", "auto")
    if raw in (False, 0, "off", "none", "disabled"):
        return {
            "enabled": False,
            "engine": "dlib",
            "copies_per_sample": 0,
            "translate_ratio": 0.0,
            "scale_range": (1.0, 1.0),
            "size_bucket": _dataset_size_bucket(train_count),
            "source": "policy_off",
        }

    engine = "dlib"
    if isinstance(raw, dict):
        engine_raw = str(raw.get("engine", "dlib")).strip().lower()
        if engine_raw in ("dlib", "cnn"):
            engine = engine_raw
    elif isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("cnn", "dlib"):
            engine = lowered

    profile = ou.get_box_jitter_profile(orientation_mode, engine=engine)
    size_bucket = _dataset_size_bucket(train_count)

    # Small datasets benefit from additional detector-noise simulation.
    if profile.get("enabled"):
        if size_bucket == "tiny":
            profile["copies_per_sample"] = int(profile.get("copies_per_sample", 0)) + 1
            profile["translate_ratio"] = float(profile.get("translate_ratio", 0.0)) + 0.005
        elif size_bucket == "small":
            profile["copies_per_sample"] = int(profile.get("copies_per_sample", 0)) + (1 if engine == "cnn" else 0)
            profile["translate_ratio"] = float(profile.get("translate_ratio", 0.0)) + (0.003 if engine == "cnn" else 0.0)

    # Optional explicit overrides.
    if isinstance(raw, dict):
        if "enabled" in raw:
            profile["enabled"] = bool(raw.get("enabled"))
        if "copies_per_sample" in raw:
            try:
                profile["copies_per_sample"] = max(0, int(raw.get("copies_per_sample", 0)))
            except Exception:
                pass
        if "translate_ratio" in raw:
            try:
                profile["translate_ratio"] = max(0.0, float(raw.get("translate_ratio", 0.0)))
            except Exception:
                pass
        if "scale_range" in raw and isinstance(raw.get("scale_range"), (list, tuple)) and len(raw.get("scale_range")) == 2:
            try:
                lo = float(raw["scale_range"][0])
                hi = float(raw["scale_range"][1])
                if lo > 0 and hi > 0 and hi >= lo:
                    profile["scale_range"] = (lo, hi)
            except Exception:
                pass

    return {
        **profile,
        "engine": engine,
        "size_bucket": size_bucket,
        "source": "policy_custom" if isinstance(raw, dict) else ("policy_engine" if isinstance(raw, str) and raw.strip().lower() in ("dlib", "cnn") else "default_auto"),
    }


def _apply_box_jitter_to_crop(crop_img, landmarks, rng, translate_ratio, scale_range):
    """
    Apply crop-space translate/scale jitter to simulate detector-box variability.
    """
    if crop_img is None or crop_img.size == 0:
        return None, None, None
    h, w = crop_img.shape[:2]
    if h != STANDARD_SIZE or w != STANDARD_SIZE:
        return None, None, None

    cx = (STANDARD_SIZE - 1) / 2.0
    cy = (STANDARD_SIZE - 1) / 2.0
    max_shift = float(max(0.0, translate_ratio)) * float(STANDARD_SIZE)
    tx = float(rng.uniform(-max_shift, max_shift))
    ty = float(rng.uniform(-max_shift, max_shift))

    scale_lo, scale_hi = float(scale_range[0]), float(scale_range[1])
    if scale_lo <= 0 or scale_hi <= 0:
        scale_lo, scale_hi = 1.0, 1.0
    if scale_hi < scale_lo:
        scale_lo, scale_hi = scale_hi, scale_lo
    scale = float(rng.uniform(scale_lo, scale_hi))

    m = np.array(
        [[scale, 0.0, (1.0 - scale) * cx + tx], [0.0, scale, (1.0 - scale) * cy + ty]],
        dtype=np.float32,
    )
    out_img = cv2.warpAffine(
        crop_img,
        m,
        (STANDARD_SIZE, STANDARD_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    out_landmarks = []
    for lm in landmarks:
        try:
            x = float(lm.get("x", -1))
            y = float(lm.get("y", -1))
            lm_id = int(lm.get("id"))
        except Exception:
            continue
        if x < 0 or y < 0:
            continue
        x2 = scale * x + (1.0 - scale) * cx + tx
        y2 = scale * y + (1.0 - scale) * cy + ty
        x2 = max(0.0, min(float(STANDARD_SIZE - 1), x2))
        y2 = max(0.0, min(float(STANDARD_SIZE - 1), y2))
        out_landmarks.append({"id": lm_id, "x": x2, "y": y2})

    if not out_landmarks:
        return None, None, None
    return out_img, out_landmarks, {"translate_x": tx, "translate_y": ty, "scale": scale}


def _augment_train_entries_with_box_jitter(train_entries, corrected_dir, profile, seed):
    """
    Augment train entries with schema-specific box jitter crops.
    """
    enabled = bool(profile.get("enabled", False))
    copies = int(max(0, profile.get("copies_per_sample", 0)))
    if (not enabled) or copies <= 0 or not train_entries:
        return [], []

    translate_ratio = float(max(0.0, profile.get("translate_ratio", 0.0)))
    scale_range = profile.get("scale_range", (1.0, 1.0))
    try:
        scale_range = (float(scale_range[0]), float(scale_range[1]))
    except Exception:
        scale_range = (1.0, 1.0)

    rng = np.random.default_rng(int(seed) + 113)
    augmented_entries = []
    jitter_log = []
    base_entries = list(train_entries)
    for idx, entry in enumerate(base_entries):
        img = safe_imread(entry["path"])
        if img is None:
            continue
        base_name = os.path.splitext(os.path.basename(entry["path"]))[0]
        for j in range(copies):
            out_img, out_landmarks, jitter_meta = _apply_box_jitter_to_crop(
                img,
                entry["landmarks"],
                rng,
                translate_ratio=translate_ratio,
                scale_range=scale_range,
            )
            if out_img is None or not out_landmarks:
                continue
            jitter_path = os.path.join(corrected_dir, f"{base_name}_jit_{idx:05d}_{j + 1:02d}.png")
            safe_imwrite(jitter_path, out_img)
            augmented_entries.append(
                {
                    "path": jitter_path,
                    "landmarks": out_landmarks,
                    "source_image": entry["source_image"],
                    "box_index": entry.get("box_index", 0),
                    "is_box_jitter_augmented": True,
                }
            )
            jitter_log.append(
                {
                    "source_crop_path": entry["path"],
                    "augmented_crop_path": jitter_path,
                    "source_image": entry["source_image"],
                    "box_index": entry.get("box_index", 0),
                    **(jitter_meta or {}),
                }
            )
    return augmented_entries, jitter_log


def json_to_dlib_xml(project_root, tag, test_split=0.2, seed=42, max_dim=1500,
                     target_orientation='left'):
    """
    Convert JSON annotations to dlib XML format for training.

    Orientation normalization is schema-driven via session orientationPolicy:
    - directional: normalize to target_orientation (left/right) by mirroring crops.
    - bilateral/axial/invariant: keep native orientation (no forced mirroring).
    """
    # Use absolute paths to avoid issues with working directory
    project_root = os.path.abspath(project_root)
    labels_dir = os.path.join(project_root, "labels")
    xml_dir = os.path.join(project_root, "xml")
    images_dir = os.path.join(project_root, "images")
    corrected_dir = os.path.join(project_root, "corrected_images")
    debug_dir = os.path.join(project_root, "debug")

    for d in [xml_dir, corrected_dir, debug_dir]:
        os.makedirs(d, exist_ok=True)

    orientation_policy = ou.load_orientation_policy(project_root)
    orientation_mode = ou.get_orientation_mode(orientation_policy)
    policy_target_orientation = str(
        orientation_policy.get("targetOrientation", target_orientation)
    ).strip().lower()
    if policy_target_orientation in ("left", "right"):
        target_orientation = policy_target_orientation
    canonical_target_orientation = target_orientation if orientation_mode == "directional" else None
    head_categories = orientation_policy.get("headCategories")
    tail_categories = orientation_policy.get("tailCategories")
    head_landmark_id = _resolve_head_landmark_id(
        project_root,
        head_categories=head_categories,
        fallback_to_min=(orientation_mode == "directional"),
    )
    tail_landmark_id = _resolve_tail_landmark_id(project_root, tail_categories=tail_categories)
    pca_mode = str(orientation_policy.get("pcaLevelingMode", "off")).strip().lower()
    if pca_mode not in ("off", "on", "auto"):
        pca_mode = "off"
    canonical_training_enabled = orientation_mode != "invariant" and pca_mode in ("on", "auto")
    segment_index = _build_segment_index(project_root) if canonical_training_enabled else {}

    debug_log, orientation_log, processed = [], [], []

    json_paths = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
    if not json_paths:
        raise RuntimeError(f"No JSON files in {labels_dir}")
    print("PROGRESS 5 loading_labels", file=sys.stderr)
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 5,
                "stage": "prepare_dataset",
                "substage": "load_labels",
                "message": f"Found {len(json_paths)} label files.",
                "labels_total": int(len(json_paths)),
            }
        ),
        file=sys.stderr,
    )

    for idx, jp in enumerate(json_paths):
        if idx == 0 or ((idx + 1) % max(1, len(json_paths) // 10) == 0) or (idx + 1 == len(json_paths)):
            frac = (idx + 1) / max(1, len(json_paths))
            pct = 5 + int(round(30 * frac))
            print(
                "PROGRESS_JSON " + json.dumps(
                    {
                        "percent": int(pct),
                        "stage": "prepare_dataset",
                        "substage": "parse_labels",
                        "message": f"Parsing labels {idx + 1}/{len(json_paths)}...",
                        "labels_done": int(idx + 1),
                        "labels_total": int(len(json_paths)),
                    }
                ),
                file=sys.stderr,
            )
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)

        img_filename = data.get("imageFilename")
        img_path = _resolve_image_path(images_dir, img_filename)

        img, w, h = load_image(img_path)
        if img is None:
            raise RuntimeError(f"Could not read: {img_path}")

        scale = 1.0
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            w, h = img.shape[1], img.shape[0]

        base = _ascii_safe_base(os.path.splitext(img_filename)[0])
        corrected_path = os.path.join(corrected_dir, f"{base}.png")
        safe_imwrite(corrected_path, img)

        detected = detect_specimen(corrected_path, margin=20)
        if not detected:
            detected = {'left': 0, 'top': 0, 'width': w, 'height': h, 'right': w, 'bottom': h}

        debug_log.append({
            "filename": img_filename,
            "corrected_path": corrected_path,
            "scale": scale,
            "box": detected,
            "original_dimensions": {"width": int(w / scale) if scale != 1.0 else w,
                                    "height": int(h / scale) if scale != 1.0 else h}
        })

        # Get boxes from JSON - support both multi-box and legacy single-box format
        json_boxes = data.get("boxes", [])

        # Multi-specimen mode: each box has its own bounding region and landmarks
        if json_boxes and any(box.get("left", 0) != 0 or box.get("top", 0) != 0 for box in json_boxes):
            # Multi-specimen: boxes have specific regions
            image_boxes = []
            for box_data in json_boxes:
                box_lm = box_data.get("landmarks", [])
                valid_lm = [lm for lm in box_lm if not lm.get("isSkipped") and lm.get("x", -1) >= 0 and lm.get("y", -1) >= 0]
                if not valid_lm:
                    continue

                # Scale landmarks and box coordinates if image was resized
                if scale != 1.0:
                    valid_lm = [{**lm, "x": lm["x"] * scale, "y": lm["y"] * scale} for lm in valid_lm]
                    box_coords = {
                        "left": int(box_data.get("left", 0) * scale),
                        "top": int(box_data.get("top", 0) * scale),
                        "width": int(box_data.get("width", w) * scale),
                        "height": int(box_data.get("height", h) * scale)
                    }
                else:
                    box_coords = {
                        "left": box_data.get("left", 0),
                        "top": box_data.get("top", 0),
                        "width": box_data.get("width", w),
                        "height": box_data.get("height", h)
                    }

                # Pass through OBB geometry fields (scale corners if image was resized)
                obb_corners_raw = box_data.get("obbCorners") or box_data.get("obb_corners")
                if obb_corners_raw and len(obb_corners_raw) == 4:
                    if scale != 1.0:
                        box_coords["obbCorners"] = [[c[0] * scale, c[1] * scale] for c in obb_corners_raw]
                    else:
                        box_coords["obbCorners"] = obb_corners_raw
                    if box_data.get("angle") is not None:
                        box_coords["angle"] = box_data["angle"]
                if box_data.get("class_id") is not None:
                    box_coords["class_id"] = box_data["class_id"]

                # Detect orientation for this individual specimen.
                # The flip is deferred to standardize_crop (mirror=was_mirrored) so that
                # only this specimen's crop is flipped — not the whole image, which would
                # corrupt the crops of adjacent specimens.
                orig_orientation = detect_orientation(
                    valid_lm,
                    head_id=head_landmark_id,
                    tail_id=tail_landmark_id,
                )
                was_mirrored = bool(
                    orientation_mode == "directional"
                    and orig_orientation is not None
                    and orig_orientation != canonical_target_orientation
                )
                if canonical_training_enabled:
                    was_mirrored = False

                image_boxes.append({
                    "box": box_coords,
                    "landmarks": valid_lm,
                    "orientation": orig_orientation,
                    "mirrored": was_mirrored
                })

            if image_boxes:
                orientation_log.append({
                    "filename": img_filename,
                    "mode": "multi-specimen",
                    "orientation_mode": orientation_mode,
                    "num_boxes": len(image_boxes),
                    "boxes": [{"orientation": b["orientation"], "mirrored": b["mirrored"]} for b in image_boxes]
                })
                processed.append({
                    "path": corrected_path,
                    "boxes": image_boxes,
                    "scale": scale,
                    "filename": img_filename,
                    "multi_specimen": True,
                    "source_image_path": img_path,
                })
        else:
            # Single-specimen mode: flatten all landmarks, use auto-detected box
            if json_boxes:
                all_lm = [lm for box in json_boxes for lm in box.get("landmarks", [])]
            else:
                all_lm = data.get("landmarks", [])
            if not all_lm:
                continue

            # Filter valid landmarks (not skipped, has valid coordinates)
            valid_lm = [lm for lm in all_lm if not lm.get("isSkipped") and lm.get("x", -1) >= 0 and lm.get("y", -1) >= 0]
            if not valid_lm:
                continue

            # Scale landmarks if image was resized
            if scale != 1.0:
                valid_lm = [{**lm, "x": lm["x"] * scale, "y": lm["y"] * scale} for lm in valid_lm]

            # Detect and normalize orientation
            orig_orientation = detect_orientation(
                valid_lm,
                head_id=head_landmark_id,
                tail_id=tail_landmark_id,
            )
            was_mirrored = bool(
                orientation_mode == "directional"
                and orig_orientation
                and orig_orientation != canonical_target_orientation
            )
            if canonical_training_enabled:
                was_mirrored = False
            # NOTE: do NOT mirror landmarks in image space here.
            # The flip is applied per-crop inside standardize_crop (mirror=was_mirrored)
            # so both the image pixels and the landmark coords stay consistent.

            orientation_log.append({
                "filename": img_filename,
                "mode": "single-specimen",
                "orientation_mode": orientation_mode,
                "original_orientation": orig_orientation,
                "target_orientation": target_orientation,
                "canonical_target_orientation": canonical_target_orientation,
                "was_mirrored": was_mirrored
            })

            processed.append({
                "path": corrected_path,
                "boxes": [{"box": detected, "landmarks": valid_lm, "orientation": orig_orientation, "mirrored": was_mirrored}],
                "scale": scale,
                "filename": img_filename,
                "multi_specimen": False,
                "source_image_path": img_path,
            })

    if not processed:
        raise RuntimeError("No valid images with landmarks found")
    print("PROGRESS 45 standardizing_crops", file=sys.stderr)
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 45,
                "stage": "prepare_dataset",
                "substage": "standardize",
                "message": f"Standardizing crops for {len(processed)} images.",
                "images_processed": int(len(processed)),
            }
        ),
        file=sys.stderr,
    )

    # Find common landmarks across all boxes in all images
    id_sets = []
    for p in processed:
        for box_data in p["boxes"]:
            id_sets.append(set(lm.get("id", 0) for lm in box_data["landmarks"]))

    common = id_sets[0]
    for s in id_sets[1:]:
        common &= s
    excluded = set().union(*id_sets) - common

    if excluded:
        print(f"Excluding landmark IDs {sorted(excluded)}, using {len(common)} common landmarks: {sorted(common)}", file=sys.stderr)
    if not common:
        raise RuntimeError("No common landmarks across all images/boxes")

    # Filter to only common landmarks in each box
    for p in processed:
        for box_data in p["boxes"]:
            box_data["landmarks"] = [lm for lm in box_data["landmarks"] if lm.get("id", 0) in common]

    # Standardize: crop each box to tight crop + padding, resize to 512x512
    # Each box becomes its own image entry in the XML
    crop_metadata_log = []
    standardized_entries = []  # list of {"path": ..., "landmarks": [...], "source": ...}

    for p in processed:
        # Re-load the corrected (full) image for cropping
        full_img = safe_imread(p["path"])
        if full_img is None:
            continue
        base = os.path.splitext(os.path.basename(p["path"]))[0]

        for bi, box_data in enumerate(p["boxes"]):
            box = box_data["box"]
            landmarks = box_data["landmarks"]
            orig_orientation = box_data.get("orientation")
            mirror = box_data.get("mirrored", False)
            if canonical_training_enabled:
                # Canonicalizer applies directional flips from orientation hint;
                # disable the legacy manual mirror to avoid double flips.
                mirror = False

            xyxy = _box_to_xyxy(box)
            mask_full = None
            if canonical_training_enabled and xyxy is not None:
                mask_full = _segment_mask_for_box(
                    full_img.shape[:2],
                    p.get("source_image_path", p["path"]),
                    xyxy,
                    p.get("scale", 1.0),
                    segment_index,
                    min_iou=0.30,
                )

            cropped_img, remapped_lm, meta = standardize_crop(
                full_img,
                box,
                landmarks,
                mirror=mirror,
                orientation_policy=orientation_policy if canonical_training_enabled else None,
                mask_full=mask_full,
                orientation_hint=orig_orientation,
            )

            # Save cropped image — one per box
            suffix = f"_box{bi}" if len(p["boxes"]) > 1 else ""
            crop_path = os.path.join(corrected_dir, f"{base}{suffix}_crop.png")
            safe_imwrite(crop_path, cropped_img)

            standardized_entries.append({
                "path": crop_path,
                "landmarks": remapped_lm,
                "source_image": p["filename"],
                "box_index": bi,
            })

            crop_metadata_log.append({
                "source_image": p["filename"],
                "box_index": bi,
                "crop_path": crop_path,
                "canonical_training_enabled": canonical_training_enabled,
                **meta,
            })

    if not standardized_entries:
        raise RuntimeError("No valid standardized crops produced")
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 72,
                "stage": "prepare_dataset",
                "substage": "split",
                "message": f"Generated {len(standardized_entries)} standardized crops. Splitting train/test.",
                "crops_total": int(len(standardized_entries)),
            }
        ),
        file=sys.stderr,
    )

    # Create ID mapping (dlib requires sequential 0-based IDs).
    # Part names are zero-padded so lexical sort order always matches numeric order.
    sorted_ids = sorted(common)
    part_name_width = max(2, len(str(max(0, len(sorted_ids) - 1))))

    def dlib_part_name(idx):
        return f"{idx:0{part_name_width}d}"

    dlib_name_to_original = {dlib_part_name(i): orig for i, orig in enumerate(sorted_ids)}
    dlib_names_sorted = sorted(dlib_name_to_original.keys())
    dlib_index_to_original = {i: dlib_name_to_original[name] for i, name in enumerate(dlib_names_sorted)}
    id_map = dlib_index_to_original
    rev_map = {orig: i for i, orig in enumerate(sorted_ids)}

    # Build a simple landmark template in standardized 512x512 coordinates.
    # This lets inference compare normal vs mirrored predictions against the training shape.
    template_acc = {}
    for entry in standardized_entries:
        for lm in entry["landmarks"]:
            lid = lm.get("id", 0)
            if lid not in template_acc:
                template_acc[lid] = {"x": [], "y": []}
            template_acc[lid]["x"].append(float(lm["x"]))
            template_acc[lid]["y"].append(float(lm["y"]))

    landmark_template = {}
    for lid, vals in template_acc.items():
        xs, ys = vals["x"], vals["y"]
        if not xs or not ys:
            continue
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        std_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / len(xs))
        std_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys) / len(ys))
        landmark_template[lid] = {
            "x_mean": mean_x,
            "y_mean": mean_y,
            "x_std": std_x,
            "y_std": std_y,
            "count": len(xs)
        }

    # Split into train/test sets grouped by source image to avoid leakage when one
    # source image yields multiple standardized crops (multi-specimen annotations).
    random.seed(seed)
    by_source = {}
    for entry in standardized_entries:
        by_source.setdefault(entry["source_image"], []).append(entry)
    sources = list(by_source.keys())
    random.shuffle(sources)

    n_total = len(standardized_entries)
    n_test_target = max(1, int(n_total * test_split))
    if n_total - n_test_target < 1:
        n_test_target = n_total - 1

    # If there is only one source image, a clean grouped split is impossible.
    # Fall back to crop-level split to preserve legacy behavior.
    if len(sources) <= 1:
        shuffled = list(standardized_entries)
        random.shuffle(shuffled)
        test_entries, train_entries = shuffled[:n_test_target], shuffled[n_test_target:]
        if len(shuffled) == 1:
            train_entries = test_entries = shuffled
        train_source_set = {e["source_image"] for e in train_entries}
        test_source_set = {e["source_image"] for e in test_entries}
    else:
        test_sources = []
        running_test = 0
        for src in sources:
            src_count = len(by_source[src])
            if running_test >= n_test_target:
                break
            # Keep at least one crop in train.
            if n_total - (running_test + src_count) < 1:
                continue
            test_sources.append(src)
            running_test += src_count

        # Ensure both train and test have sources.
        if not test_sources:
            test_sources = [sources[0]]
        if len(test_sources) == len(sources):
            test_sources = test_sources[:-1]

        test_source_set = set(test_sources)
        train_source_set = {s for s in sources if s not in test_source_set}
        test_entries = []
        train_entries = []
        for s in sources:
            if s in test_source_set:
                test_entries.extend(by_source[s])
            else:
                train_entries.extend(by_source[s])

    train_entries_pre_jitter = len(train_entries)
    prep_box_jitter_profile = _resolve_training_prep_box_jitter_profile(
        orientation_policy,
        orientation_mode,
        train_entries_pre_jitter,
    )
    cnn_recommended_box_jitter = ou.get_box_jitter_profile(orientation_mode, engine="cnn")
    jitter_augmented_entries, jitter_aug_log = _augment_train_entries_with_box_jitter(
        train_entries,
        corrected_dir,
        prep_box_jitter_profile,
        seed=seed,
    )
    if jitter_augmented_entries:
        train_entries.extend(jitter_augmented_entries)
        crop_metadata_log.extend(
            {
                "source_image": e.get("source_image"),
                "box_index": e.get("box_index", 0),
                "crop_path": e.get("path"),
                "is_box_jitter_augmented": True,
            }
            for e in jitter_augmented_entries
        )

    def write_xml(entries, path):
        root = ET.Element("dataset")
        images = ET.SubElement(root, "images")
        for entry in entries:
            img_el = ET.SubElement(images, "image", file=entry["path"])
            # Full-image box since the crop IS the specimen
            box_el = ET.SubElement(img_el, "box", top="0", left="0",
                                   width=str(STANDARD_SIZE), height=str(STANDARD_SIZE))
            for lm in sorted(entry["landmarks"], key=lambda x: x.get("id", 0)):
                ET.SubElement(box_el, "part", name=dlib_part_name(rev_map[lm["id"]]),
                              x=str(int(lm["x"])), y=str(int(lm["y"])))
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    train_path = os.path.join(xml_dir, f"train_{tag}.xml")
    test_path = os.path.join(xml_dir, f"test_{tag}.xml")
    write_xml(train_entries, train_path)
    write_xml(test_entries, test_path)
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 90,
                "stage": "prepare_dataset",
                "substage": "write_xml",
                "message": "Wrote train/test XML files.",
                "train_crops": int(len(train_entries)),
                "test_crops": int(len(test_entries)),
            }
        ),
        file=sys.stderr,
    )

    # Save debug files
    with open(os.path.join(debug_dir, f"id_mapping_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dlib_to_original": id_map,
            "dlib_index_to_original": id_map,
            "dlib_name_to_original": dlib_name_to_original,
            "original_to_dlib": rev_map,
            "original_ids": sorted_ids,
            "excluded_ids": sorted(excluded) if excluded else [],
            "num_landmarks": len(sorted_ids),
            "standard_size": STANDARD_SIZE,
            "part_name_width": part_name_width,
            "part_names_sorted": dlib_names_sorted,
            "landmark_template": landmark_template,
            "training_config": {
                "max_dim": max_dim,
                "test_split": test_split,
                "seed": seed,
                "orientation_mode": orientation_mode,
                "orientation_policy": orientation_policy,
                "target_orientation": canonical_target_orientation,
                "head_landmark_id": head_landmark_id,
                "tail_landmark_id": tail_landmark_id,
                "canonical_training_enabled": canonical_training_enabled,
                "pca_leveling_mode": pca_mode,
                "segment_index_entries": sum(len(v) for v in segment_index.values()) if segment_index else 0,
                "training_prep_box_jitter_profile": prep_box_jitter_profile,
                "cnn_recommended_box_jitter_profile": cnn_recommended_box_jitter,
                "training_prep_box_jitter_added_crops": len(jitter_augmented_entries),
            }
        }, f, indent=2)

    with open(os.path.join(debug_dir, f"crop_metadata_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(crop_metadata_log, f, indent=2)

    orientation_boxes = []
    for o in orientation_log:
        if o.get("mode") == "multi-specimen":
            for b in o.get("boxes", []):
                orientation_boxes.append({
                    "orientation": b.get("orientation"),
                    "mirrored": bool(b.get("mirrored", False))
                })
        else:
            orientation_boxes.append({
                "orientation": o.get("original_orientation"),
                "mirrored": bool(o.get("was_mirrored", False))
            })

    with open(os.path.join(debug_dir, f"orientation_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "orientation_mode": orientation_mode,
            "orientation_policy": orientation_policy,
            "target_orientation": canonical_target_orientation,
            "head_landmark_id": head_landmark_id,
            "tail_landmark_id": tail_landmark_id,
            "images": orientation_log,
            "summary": {
                "image_entries": len(orientation_log),
                "box_entries": len(orientation_boxes),
                "total": len(orientation_boxes),
                "mirrored": sum(1 for b in orientation_boxes if b.get("mirrored")),
                "left_facing": sum(1 for b in orientation_boxes if b.get("orientation") == "left"),
                "right_facing": sum(1 for b in orientation_boxes if b.get("orientation") == "right"),
                "unknown_orientation": sum(1 for b in orientation_boxes if b.get("orientation") is None)
            }
        }, f, indent=2)

    with open(os.path.join(debug_dir, f"training_boxes_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(debug_log, f, indent=2)

    with open(os.path.join(debug_dir, f"box_jitter_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "orientation_mode": orientation_mode,
                "profile_applied": prep_box_jitter_profile,
                "cnn_recommended_profile": cnn_recommended_box_jitter,
                "train_entries_before_jitter": train_entries_pre_jitter,
                "jitter_entries_added": len(jitter_augmented_entries),
                "train_entries_after_jitter": len(train_entries),
                "samples": jitter_aug_log[:5000],
            },
            f,
            indent=2,
        )

    with open(os.path.join(debug_dir, f"split_info_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_crops": len(train_entries),
            "train_crops_before_jitter": train_entries_pre_jitter,
            "train_crops_after_jitter": len(train_entries),
            "train_jitter_crops_added": len(jitter_augmented_entries),
            "test_crops": len(test_entries),
            "total_crops": len(standardized_entries),
            "train_sources": sorted(train_source_set),
            "test_sources": sorted(test_source_set),
            "train_source_count": len(train_source_set),
            "test_source_count": len(test_source_set),
            "source_overlap_count": len(set(train_source_set) & set(test_source_set)),
            "standard_size": STANDARD_SIZE,
            "training_prep_box_jitter_profile": prep_box_jitter_profile,
            "cnn_recommended_box_jitter_profile": cnn_recommended_box_jitter,
            "train_files": [e["path"] for e in train_entries],
            "test_files": [e["path"] for e in test_entries]
        }, f, indent=2)
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 100,
                "stage": "prepare_dataset",
                "substage": "done",
                "message": "Dataset preparation complete.",
                "train_xml": train_path,
                "test_xml": test_path,
            }
        ),
        file=sys.stderr,
    )

    print(train_path)
    print(test_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: prepare_dataset.py <root> <tag> [test_split] [seed] [target_orientation]")
        sys.exit(1)

    # Use keyword arguments to avoid positional mixups
    json_to_dlib_xml(
        project_root=sys.argv[1],
        tag=sys.argv[2],
        test_split=float(sys.argv[3]) if len(sys.argv) > 3 else 0.2,
        seed=int(sys.argv[4]) if len(sys.argv) > 4 else 42,
        target_orientation=sys.argv[5] if len(sys.argv) > 5 else 'left'
    )


