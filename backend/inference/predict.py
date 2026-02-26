import sys
import os
import json
import hashlib
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
import dlib
import cv2
import numpy as np

import sys as _sys, os as _os
_BACKEND_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _BACKEND_ROOT not in _sys.path:
    _sys.path.insert(0, _BACKEND_ROOT)

from detection.detect_specimen import detect_specimen
from bv_utils.image_utils import load_image
import bv_utils.orientation_utils as ou
import bv_utils.debug_io as dio

STANDARD_SIZE = ou.STANDARD_SIZE

# Optional torch / torchvision for CNN predictor
try:
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models
    from torchvision import transforms as tv_transforms
    _torch_available = True
except ImportError:
    _torch_available = False


def _load_first_part_names_from_xml(xml_path):
    """Return lexically-sorted part names from the first image/box in an XML file."""
    if not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        images = tree.getroot().find("images")
        if images is None:
            return []
        first_image = images.find("image")
        if first_image is None:
            return []
        first_box = first_image.find("box")
        if first_box is None:
            return []
        names = [part.get("name") for part in first_box.findall("part") if part.get("name") is not None]
        return sorted(names)
    except Exception:
        return []


def _resolve_dlib_index_mapping(project_root, tag, id_mapping_data):
    """
    Resolve dlib part index -> original landmark ID mapping.

    Supports:
    - New mapping with explicit dlib_index_to_original.
    - Older mapping files where dlib_to_original is keyed by part-name strings but
      dlib internally indexes parts by lexical sort of part names.
    """
    if not id_mapping_data:
        return {}

    explicit = id_mapping_data.get("dlib_index_to_original")
    if explicit:
        return {int(k): int(v) for k, v in explicit.items()}

    # Legacy field present in older runs.
    raw = id_mapping_data.get("dlib_to_original", {})
    if not raw:
        return {}

    # Prefer an explicit name->original map when available.
    name_map = id_mapping_data.get("dlib_name_to_original")
    if name_map:
        sorted_names = sorted(name_map.keys())
        return {idx: int(name_map[name]) for idx, name in enumerate(sorted_names)}

    # Backward-compatible recovery: infer lexical part-name order from train XML.
    train_xml = os.path.join(project_root, "xml", f"train_{tag}.xml")
    part_names = _load_first_part_names_from_xml(train_xml)
    if part_names:
        resolved = {}
        for idx, name in enumerate(part_names):
            value = None
            if name in raw:
                value = raw[name]
            else:
                # Older JSON can store keys as numeric strings without padding.
                try:
                    numeric_name = str(int(name))
                    value = raw.get(numeric_name)
                except ValueError:
                    value = None
            if value is not None:
                resolved[idx] = int(value)
        if resolved:
            return resolved

    # Final fallback: assume keys are already dlib indexes.
    return {int(k): int(v) for k, v in raw.items()}


def _resolve_landmark_id_by_category(project_root, category_name):
    """Return the first landmark index in session.json matching category_name."""
    session_path = os.path.join(project_root, "session.json")
    if not os.path.exists(session_path):
        return None
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
        template = session.get("landmarkTemplate", [])
    except Exception:
        return None
    if not isinstance(template, list):
        return None
    target = str(category_name or "").strip().lower()
    candidates = []
    for lm in template:
        try:
            idx = int(lm.get("index"))
        except (TypeError, ValueError):
            continue
        cat = str(lm.get("category", "")).strip().lower()
        if cat == target:
            candidates.append(idx)
    if not candidates:
        return None
    return min(candidates)


def _resolve_template_landmark_order(project_root):
    """Return landmark IDs sorted by session template index."""
    session_path = os.path.join(project_root, "session.json")
    if not os.path.exists(session_path):
        return []
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
        template = session.get("landmarkTemplate", [])
    except Exception:
        return []
    if not isinstance(template, list):
        return []
    out = []
    for lm in template:
        try:
            out.append(int(lm.get("index")))
        except Exception:
            continue
    return sorted(set(out))


def _resolve_head_landmark_id(project_root, id_mapping_data=None):
    """
    Resolve the landmark ID used as "head" for orientation checks.

    Preference:
    1) training_config.head_landmark_id from id_mapping.
    2) first category=='head' landmark in session.json.
    3) minimum index in session.json.
    4) None (orientation function falls back to minimum observed ID).
    """
    if id_mapping_data:
        try:
            cfg = id_mapping_data.get("training_config", {})
            if "head_landmark_id" in cfg and cfg["head_landmark_id"] is not None:
                return int(cfg["head_landmark_id"])
        except Exception:
            pass

    try:
        policy = ou.load_orientation_policy(project_root)
        head_id, _ = ou.resolve_head_tail_landmark_ids(project_root, policy)
        if head_id is not None:
            return int(head_id)
    except Exception:
        pass

    head_id = _resolve_landmark_id_by_category(project_root, "head")
    if head_id is not None:
        return head_id

    session_path = os.path.join(project_root, "session.json")
    if not os.path.exists(session_path):
        return None
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
        template = session.get("landmarkTemplate", [])
    except Exception:
        return None
    if not isinstance(template, list):
        return None

    all_indices = []
    for lm in template:
        try:
            all_indices.append(int(lm.get("index")))
        except (TypeError, ValueError):
            continue
    if all_indices:
        return min(all_indices)
    return None


def _resolve_tail_landmark_id(project_root, id_mapping_data=None):
    """
    Resolve the landmark ID used as "tail" for orientation checks.

    Preference:
    1) training_config.tail_landmark_id from id_mapping.
    2) first category=='tail' landmark in session.json.
    3) None.
    """
    if id_mapping_data:
        try:
            cfg = id_mapping_data.get("training_config", {})
            if "tail_landmark_id" in cfg and cfg["tail_landmark_id"] is not None:
                return int(cfg["tail_landmark_id"])
        except Exception:
            pass
    try:
        policy = ou.load_orientation_policy(project_root)
        _, tail_id = ou.resolve_head_tail_landmark_ids(project_root, policy)
        if tail_id is not None:
            return int(tail_id)
    except Exception:
        pass
    return _resolve_landmark_id_by_category(project_root, "tail")


def _extract_landmarks(shape, index_to_original):
    """Extract landmark list from dlib shape in 512x512 crop space."""
    landmarks = []
    for i in range(shape.num_parts):
        part = shape.part(i)
        orig_id = index_to_original.get(i, i)
        landmarks.append({"id": orig_id, "x": part.x, "y": part.y})
    return landmarks


def _scale_orientation_hint(hint, scale):
    """
    Scale orientation-hint points from resized inference image space back to
    original image space.
    """
    if not isinstance(hint, dict):
        return hint
    scaled = dict(hint)
    if not scale or scale == 1.0:
        return scaled
    for point_key in ("head_point", "tail_point"):
        point = scaled.get(point_key)
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                scaled[point_key] = [round(float(point[0]) / scale, 1), round(float(point[1]) / scale, 1)]
            except Exception:
                pass
    return scaled


def _resolve_orientation_hint_from_box(box, min_confidence=0.25, min_dx_ratio=0.05):
    """
    Resolve a robust left/right hint from detector output.

    Uses orientation_utils.resolve_orientation_hint with a reliability gate so
    ambiguous pose vectors do not force incorrect mirrored landmark layouts.
    """
    if not isinstance(box, dict):
        return None
    hint = box.get("orientation_hint")
    if hint is None:
        return None
    xyxy = None
    try:
        if all(k in box for k in ("left", "top", "right", "bottom")):
            xyxy = [box["left"], box["top"], box["right"], box["bottom"]]
    except Exception:
        xyxy = None
    return ou.resolve_orientation_hint(
        hint,
        box_xyxy=xyxy,
        min_confidence=min_confidence,
        min_dx_ratio=min_dx_ratio,
    )


_SAM2_MODEL = None
_SAM2_UNAVAILABLE_REASON = None


def _should_try_canonicalization(orientation_policy):
    if not isinstance(orientation_policy, dict):
        return False
    mode = str(orientation_policy.get("mode", "invariant")).strip().lower()
    pca_mode = str(orientation_policy.get("pcaLevelingMode", "off")).strip().lower()
    if pca_mode not in ("off", "on", "auto"):
        pca_mode = "off"
    return mode != "invariant" and pca_mode in ("on", "auto")


def _get_sam2_model():
    global _SAM2_MODEL, _SAM2_UNAVAILABLE_REASON
    if _SAM2_MODEL is not None:
        return _SAM2_MODEL
    if _SAM2_UNAVAILABLE_REASON is not None:
        return None
    try:
        from ultralytics import SAM

        _SAM2_MODEL = SAM("sam2_b.pt")
        return _SAM2_MODEL
    except Exception as exc:
        _SAM2_UNAVAILABLE_REASON = str(exc)
        return None


def _infer_ultralytics_device():
    try:
        if _torch_available and torch.cuda.is_available():
            return 0
        if _torch_available and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _box_to_xyxy(box):
    try:
        left = int(round(float(box.get("left", 0))))
        top = int(round(float(box.get("top", 0))))
        if "right" in box and "bottom" in box:
            right = int(round(float(box.get("right", left + 1))))
            bottom = int(round(float(box.get("bottom", top + 1))))
        else:
            right = left + int(round(float(box.get("width", 1))))
            bottom = top + int(round(float(box.get("height", 1))))
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1
        return [left, top, right, bottom]
    except Exception:
        return None


def _sanitize_binary_mask(mask, min_area=32):
    """
    Keep only the largest connected component from a binary mask.
    """
    if mask is None:
        return None
    try:
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        if int(mask_u8.sum()) <= 0:
            return None
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < float(min_area):
            return None
        out = np.zeros_like(mask_u8, dtype=np.uint8)
        cv2.drawContours(out, [largest], -1, 1, thickness=-1)
        return out
    except Exception:
        return None


def _mask_outline_from_full_mask(mask_full, scale=1.0, image_shape=None, max_points=256):
    """
    Convert a full-image binary mask into a lightweight contour outline.

    Returns a list of [x, y] points in original-image coordinates
    (inverse-scaled when inference used image downscaling).
    """
    if mask_full is None:
        return None
    try:
        mask_u8 = (np.asarray(mask_full) > 0).astype(np.uint8)
        if int(mask_u8.sum()) <= 0:
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
        if pts.shape[0] < 3:
            return None

        if pts.shape[0] > int(max_points):
            keep_idx = np.linspace(0, pts.shape[0] - 1, int(max_points)).astype(int)
            pts = pts[keep_idx]

        img_h = image_shape[0] if image_shape else None
        img_w = image_shape[1] if image_shape else None
        inv_scale = (1.0 / float(scale)) if float(scale) not in (0.0, 1.0) else 1.0

        out = []
        for x, y in pts:
            xo = float(x) * inv_scale
            yo = float(y) * inv_scale
            if img_w is not None and img_h is not None:
                xo = max(0.0, min(float(img_w - 1), xo))
                yo = max(0.0, min(float(img_h - 1), yo))
            out.append([int(round(xo)), int(round(yo))])
        return out if len(out) >= 3 else None
    except Exception:
        return None


def _sam2_mask_for_box(image_bgr, box):
    model = _get_sam2_model()
    if model is None:
        return None
    xyxy = _box_to_xyxy(box)
    if xyxy is None:
        return None
    try:
        results = model.predict(
            image_bgr,
            bboxes=[xyxy],
            device=_infer_ultralytics_device(),
            verbose=False,
        )
        if not results:
            return None
        masks = getattr(results[0], "masks", None)
        data = getattr(masks, "data", None) if masks is not None else None
        if data is None or int(data.shape[0]) == 0:
            return None
        masks_np = data.cpu().numpy().astype(np.uint8)
        best_mask = None
        best_area = -1
        for i in range(masks_np.shape[0]):
            clean = _sanitize_binary_mask(masks_np[i], min_area=32)
            if clean is None:
                continue
            area = int(clean.sum())
            if area > best_area:
                best_area = area
                best_mask = clean
        return best_mask
    except Exception:
        return None


def _rough_mask_from_crop(crop_512):
    """
    Estimate a coarse foreground mask when SAM2 is unavailable.

    This is a fallback only: threshold + largest contour in standardized crop space.
    """
    try:
        gray = cv2.cvtColor(crop_512, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        # Prefer the larger polarity (foreground can be light or dark).
        if int(np.count_nonzero(mask)) > (mask.size // 2):
            mask = cv2.bitwise_not(mask)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return _sanitize_binary_mask(mask, min_area=32)
    except Exception:
        return None


def _canonicalize_crop_with_standard_mask(
    crop_512,
    crop_meta,
    mask_512,
    orientation_policy,
    orientation_hint,
):
    if mask_512 is None:
        return crop_512, crop_meta, None
    try:
        canonical_img, _mask_512, canonical_meta = ou.canonicalize_with_mask(
            crop_512,
            mask_512,
            policy=orientation_policy,
            pca_mode="auto",
            orientation_hint=orientation_hint,
        )
        updated_meta = {
            **crop_meta,
            "rotation": float(canonical_meta.get("pca_rotation", 0.0)),
            "canonical_flip_applied": bool(canonical_meta.get("canonical_flip_applied", False)),
            "canonicalization": canonical_meta,
        }
        return canonical_img, updated_meta, canonical_meta
    except Exception:
        return crop_512, crop_meta, None


def _canonicalize_crop_with_mask(
    crop_512,
    crop_meta,
    mask_full,
    orientation_policy,
    orientation_hint,
):
    if mask_full is None:
        return crop_512, crop_meta, None
    try:
        cx1, cy1 = [int(v) for v in crop_meta["crop_origin"]]
        cw, ch = [int(v) for v in crop_meta["crop_size"]]
        if cw <= 0 or ch <= 0:
            return crop_512, crop_meta, None
        mask_crop = mask_full[cy1:cy1 + ch, cx1:cx1 + cw]
        if mask_crop.size == 0:
            return crop_512, crop_meta, None
        mask_512 = cv2.resize(
            mask_crop.astype(np.uint8),
            (STANDARD_SIZE, STANDARD_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        return _canonicalize_crop_with_standard_mask(
            crop_512,
            crop_meta,
            mask_512,
            orientation_policy,
            orientation_hint,
        )
    except Exception:
        return crop_512, crop_meta, None


def _build_inference_metadata(orientation_debug):
    """
    Build a stable metadata payload describing canonicalization + orientation choice.
    """
    debug = orientation_debug if isinstance(orientation_debug, dict) else {}
    canonical = debug.get("canonicalization")
    if not isinstance(canonical, dict):
        canonical = {}
    pca_rotation = canonical.get("pca_rotation")
    direction_conf = canonical.get("inferred_direction_confidence")
    raw_hint = debug.get("orientation_hint_raw")
    hint_orientation = None
    hint_source = None
    if isinstance(raw_hint, dict):
        try:
            hint_orientation = str(raw_hint.get("orientation", "")).strip().lower() or None
        except Exception:
            hint_orientation = None
        try:
            hint_source = str(raw_hint.get("source", "")).strip() or None
        except Exception:
            hint_source = None
    if hint_orientation not in ("left", "right"):
        hint_orientation = debug.get("orientation_hint")

    was_flipped = bool(
        debug.get("was_flipped", debug.get("used_flipped_crop", False))
    )
    return {
        "mask_source": canonical.get("mask_source"),
        "pca_rotation": pca_rotation,
        "pca_angle": pca_rotation,
        "canonical_flip_applied": bool(canonical.get("canonical_flip_applied", False)),
        "direction_source": canonical.get("direction_source"),
        "inferred_direction": canonical.get("inferred_direction"),
        "inferred_direction_confidence": direction_conf,
        "direction_confidence": direction_conf,
        "used_flipped_crop": bool(debug.get("used_flipped_crop", False)),
        "was_flipped": was_flipped,
        "selection_reason": debug.get("selection_reason"),
        "locked_from_canonicalization": bool(debug.get("locked_from_canonicalization", False)),
        "match_a_canonical": debug.get("match_a_canonical"),
        "match_b_canonical": debug.get("match_b_canonical"),
        "detector_hint_orientation": hint_orientation,
        "detector_hint_source": hint_source,
        "orientation_warning": debug.get("orientation_warning"),
    }


def _predict_with_orientation_lock(
    *,
    crop_512,
    predict_fn,
    orientation_policy,
    canonicalization_debug,
    target_orientation,
    landmark_template,
    head_landmark_id,
    tail_landmark_id,
    orientation_hint,
):
    orientation_mode = ou.get_orientation_mode(orientation_policy or {})
    if orientation_mode == "directional":
        if orientation_hint in ("left", "right"):
            primary = predict_fn(crop_512) or []
            direction_source = None
            direction_conf = None
            if isinstance(canonicalization_debug, dict):
                direction_source = canonicalization_debug.get("direction_source")
                direction_conf = canonicalization_debug.get("direction_confidence")
            return primary, False, {
                "used_flipped_crop": False,
                "selection_reason": "directional_detector_hint_only",
                "candidate_b_evaluated": False,
                "head_landmark_id": head_landmark_id,
                "tail_landmark_id": tail_landmark_id,
                "target_orientation": target_orientation,
                "locked_from_canonicalization": True,
                "lock_direction_source": direction_source,
                "lock_direction_confidence": direction_conf,
                "orientation_hint": orientation_hint,
            }

        # Legacy detector path: no orientation hint available.
        # Re-enable dual-candidate template/orientation scoring so directional
        # models do not silently mirror right-facing specimens.
        resolved_target = target_orientation
        if resolved_target not in ("left", "right"):
            resolved_target = str((orientation_policy or {}).get("targetOrientation", "")).strip().lower()
            if resolved_target not in ("left", "right"):
                resolved_target = None

        landmarks_512, was_flipped, orientation_debug = ou.select_orientation(
            crop_512,
            predict_fn,
            target_orientation=resolved_target,
            landmark_template=landmark_template,
            head_id=head_landmark_id,
            tail_id=tail_landmark_id,
            orientation_hint_original=None,
        )
        if not isinstance(orientation_debug, dict):
            orientation_debug = {}
        orientation_debug["locked_from_canonicalization"] = False
        orientation_debug["orientation_warning"] = {
            "code": "legacy_detector_no_orientation_hint",
            "message": (
                "Directional schema without detector orientation hint: "
                "using dual-candidate fallback. Retrain YOLO with orientation-aware classes."
            ),
        }
        return landmarks_512, was_flipped, orientation_debug

    lock_orientation = ou.should_lock_orientation_from_canonicalization(
        canonicalization_debug,
        policy=orientation_policy,
    )
    if lock_orientation:
        primary = predict_fn(crop_512) or []
        direction_source = None
        direction_conf = None
        if isinstance(canonicalization_debug, dict):
            direction_source = canonicalization_debug.get("direction_source")
            direction_conf = canonicalization_debug.get("direction_confidence")
        return primary, False, {
            "used_flipped_crop": False,
            "selection_reason": "locked_canonical_orientation",
            "candidate_b_evaluated": False,
            "head_landmark_id": head_landmark_id,
            "tail_landmark_id": tail_landmark_id,
            "target_orientation": target_orientation,
            "locked_from_canonicalization": True,
            "lock_direction_source": direction_source,
            "lock_direction_confidence": direction_conf,
            "orientation_hint": orientation_hint,
        }

    landmarks_512, was_flipped, orientation_debug = ou.select_orientation(
        crop_512,
        predict_fn,
        target_orientation=target_orientation,
        landmark_template=landmark_template,
        head_id=head_landmark_id,
        tail_id=tail_landmark_id,
        orientation_hint_original=orientation_hint,
    )
    if not isinstance(orientation_debug, dict):
        orientation_debug = {}
    orientation_debug["locked_from_canonicalization"] = False
    return landmarks_512, was_flipped, orientation_debug


# ── CNN Landmark Predictor (configurable backbone) ─────────────────────────
if _torch_available:
    def _resolve_cnn_variant(requested_variant):
        variant = str(requested_variant or "efficientnet_b0").strip().lower()
        aliases = {
            "simplebase": "resnet50",
            "efficientnet": "efficientnet_b0",
            "efficientnet-b0": "efficientnet_b0",
            "mobilenet": "mobilenet_v3_large",
            "mobilenetv3": "mobilenet_v3_large",
            "mobilenet-v3-large": "mobilenet_v3_large",
            "resnet": "resnet50",
            "resnet-50": "resnet50",
            "hrnet": "hrnet_w32",
            "hrnet-w32": "hrnet_w32",
            "simplebaseline": "resnet50",
        }
        return aliases.get(variant, variant)


    def _build_cnn_backbone(variant):
        v = _resolve_cnn_variant(variant)
        fallback_reason = None
        if v == "efficientnet_b0":
            backbone = tv_models.efficientnet_b0(weights=None)
            return backbone.features, 1280, v, fallback_reason
        if v == "mobilenet_v3_large":
            backbone = tv_models.mobilenet_v3_large(weights=None)
            return backbone.features, 960, v, fallback_reason
        if v == "resnet50":
            backbone = tv_models.resnet50(weights=None)
            features = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            )
            return features, 2048, v, fallback_reason
        if v == "hrnet_w32":
            ctor = getattr(tv_models, "hrnet_w32", None)
            if ctor is not None:
                try:
                    backbone = ctor(weights=None)
                    if hasattr(backbone, "features"):
                        return backbone.features, 2048, v, fallback_reason
                    fallback_reason = "torchvision_hrnet_features_unavailable_fallback_resnet50"
                except Exception as exc:
                    fallback_reason = f"hrnet_unavailable_fallback_resnet50:{exc}"
            else:
                fallback_reason = "torchvision_hrnet_missing_fallback_resnet50"
            features, feat_dim, _, _ = _build_cnn_backbone("resnet50")
            return features, feat_dim, "resnet50", fallback_reason

        fallback_reason = f"unknown_variant_{v}_fallback_efficientnet_b0"
        features, feat_dim, _, _ = _build_cnn_backbone("efficientnet_b0")
        return features, feat_dim, "efficientnet_b0", fallback_reason


    def _spatial_soft_argmax_2d(heatmaps, beta=25.0):
        b, k, h, w = heatmaps.shape
        logits = heatmaps.view(b, k, -1) * float(beta)
        probs = torch.softmax(logits, dim=-1)
        xs = torch.linspace(0.0, 1.0, steps=w, device=heatmaps.device, dtype=heatmaps.dtype)
        ys = torch.linspace(0.0, 1.0, steps=h, device=heatmaps.device, dtype=heatmaps.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gx = gx.reshape(1, 1, -1)
        gy = gy.reshape(1, 1, -1)
        exp_x = torch.sum(probs * gx, dim=-1)
        exp_y = torch.sum(probs * gy, dim=-1)
        coords = torch.stack([exp_x, exp_y], dim=-1)
        return coords.reshape(b, k * 2)


    class CNNLandmarkPredictorLegacy(nn.Module):
        """Legacy coordinate-regression head kept for backwards compatibility."""
        def __init__(self, n_landmarks, model_variant="efficientnet_b0"):
            super().__init__()
            features, feat_dim, resolved_variant, fallback_reason = _build_cnn_backbone(model_variant)
            self.features = features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, n_landmarks * 2),
                nn.Sigmoid(),
            )
            self.model_variant = resolved_variant
            self.variant_fallback_reason = fallback_reason

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            return self.head(x)


    class CNNLandmarkPredictor(nn.Module):
        """Backbone + deconvolution heatmap head + soft-argmax decoder."""
        def __init__(
            self,
            n_landmarks,
            model_variant="efficientnet_b0",
            head_type="heatmap_deconv",
            deconv_layers=3,
            deconv_filters=256,
            softargmax_beta=25.0,
        ):
            super().__init__()
            features, feat_dim, resolved_variant, fallback_reason = _build_cnn_backbone(model_variant)
            self.features = features
            self.n_landmarks = int(n_landmarks)
            self.head_type = str(head_type or "heatmap_deconv").strip().lower()
            if self.head_type not in ("heatmap_deconv", "regression"):
                self.head_type = "heatmap_deconv"
            self.softargmax_beta = float(softargmax_beta)
            self.deconv_layers = int(max(1, deconv_layers))
            self.deconv_filters = int(max(32, deconv_filters))
            self.model_variant = resolved_variant
            self.variant_fallback_reason = fallback_reason

            if self.head_type == "regression":
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, self.n_landmarks * 2),
                    nn.Sigmoid(),
                )
                self.deconv = None
                self.heatmap_head = None
            else:
                layers = []
                in_ch = feat_dim
                for _ in range(self.deconv_layers):
                    layers.extend(
                        [
                            nn.ConvTranspose2d(
                                in_ch,
                                self.deconv_filters,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(self.deconv_filters),
                            nn.ReLU(inplace=True),
                        ]
                    )
                    in_ch = self.deconv_filters
                self.deconv = nn.Sequential(*layers)
                self.heatmap_head = nn.Conv2d(in_ch, self.n_landmarks, kernel_size=1, stride=1, padding=0)
                self.pool = None
                self.head = None

        def forward(self, x):
            x = self.features(x)
            if self.head_type == "regression":
                x = self.pool(x)
                return self.head(x)
            up = self.deconv(x)
            heatmaps = self.heatmap_head(up)
            return _spatial_soft_argmax_2d(heatmaps, beta=self.softargmax_beta)

    _CNN_TRANSFORM = tv_transforms.Compose([
        tv_transforms.ToPILImage(),
        tv_transforms.Resize((STANDARD_SIZE, STANDARD_SIZE)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])


def standardize_crop(image, box, pad_ratio=0.20):
    """
    Crop image to bounding box + padding and resize to STANDARD_SIZE x STANDARD_SIZE.
    Returns (cropped_image, crop_metadata) for inverse mapping.
    """
    bx = int(box.get("left", 0))
    by = int(box.get("top", 0))
    if "right" in box and "bottom" in box:
        x2 = int(box.get("right", bx))
        y2 = int(box.get("bottom", by))
    else:
        x2 = bx + int(box.get("width", 1))
        y2 = by + int(box.get("height", 1))
    return ou.base_standardize(image, [bx, by, x2, y2], pad_ratio=pad_ratio)


def map_landmarks_to_original(
    landmarks_512,
    crop_meta,
    image_scale=1.0,
    was_flipped=False,
    image_shape=None,
):
    """
    Map landmarks from 512x512 crop space back to original image coordinates.

    If was_flipped=True the predictor ran on a horizontally-flipped crop, so we
    first un-flip the x-coordinate within the 512×512 space before applying the
    standard crop-origin + scale mapping.
    """
    mapped = ou.map_to_original(
        landmarks_512,
        crop_meta,
        was_flipped=was_flipped,
        image_scale=image_scale,
        image_shape=image_shape,
    )
    # Keep predict.py output shape stable (integer pixel coordinates).
    return [
        {"id": lm["id"], "x": int(round(lm["x"])), "y": int(round(lm["y"]))}
        for lm in mapped
    ]


def _normalize_input_boxes(input_boxes, scale=1.0, image_w=None, image_h=None):
    """
    Normalize externally-provided boxes to predictor input space.

    input_boxes are expected in original-image coordinates. They are scaled to
    the resized inference image when scale != 1.0.
    """
    if not isinstance(input_boxes, list):
        return []

    normalized = []
    for raw in input_boxes:
        if not isinstance(raw, dict):
            continue
        try:
            left = float(raw.get("left", 0.0)) * scale
            top = float(raw.get("top", 0.0)) * scale
            if "right" in raw and "bottom" in raw:
                right = float(raw.get("right", left)) * scale
                bottom = float(raw.get("bottom", top)) * scale
            else:
                width = float(raw.get("width", 0.0)) * scale
                height = float(raw.get("height", 0.0)) * scale
                right = left + width
                bottom = top + height
        except Exception:
            continue

        if image_w is not None and image_h is not None:
            left = max(0.0, min(float(image_w - 1), left))
            top = max(0.0, min(float(image_h - 1), top))
            right = max(0.0, min(float(image_w), right))
            bottom = max(0.0, min(float(image_h), bottom))

        if right <= left:
            right = left + 1.0
        if bottom <= top:
            bottom = top + 1.0

        box = {
            "left": int(round(left)),
            "top": int(round(top)),
            "right": int(round(right)),
            "bottom": int(round(bottom)),
            "width": int(round(right - left)),
            "height": int(round(bottom - top)),
        }
        if isinstance(raw.get("orientation_hint"), dict):
            box["orientation_hint"] = raw.get("orientation_hint")
        normalized.append(box)
    return normalized


def save_prediction_log(debug_dir, tag, log_entry, predictor_type="dlib"):
    """Append prediction logs to model-specific debug location (and dlib legacy path)."""
    project_root = os.path.abspath(os.path.join(debug_dir, os.pardir))
    model_key = str(predictor_type or "dlib").strip().lower()
    dio.append_model_prediction_log(project_root, model_key, tag, log_entry, max_entries=2000)

    # Preserve legacy debug file for dlib-compatible tooling.
    if model_key == "dlib":
        log_path = os.path.join(debug_dir, f"prediction_log_{tag}.json")
        dio.append_json_array(log_path, log_entry, max_entries=2000)


def _make_dlib_predict_fn(predictor, rect, index_to_original):
    def _predict(crop_512):
        shape = predictor(crop_512, rect)
        return _extract_landmarks(shape, index_to_original)

    return _predict


def _make_cnn_predict_fn(model, landmark_ids):
    def _predict(crop_512):
        img_rgb = cv2.cvtColor(crop_512, cv2.COLOR_BGR2RGB)
        img_tensor = _CNN_TRANSFORM(img_rgb).unsqueeze(0)
        with torch.no_grad():
            coords = model(img_tensor)[0].cpu().numpy()
        return _cnn_landmarks_from_coords(coords, landmark_ids, flip=False)

    return _predict


def predict_image(project_root, tag, image_path, yolo_model_path=None, input_box=None):
    """
    Predict landmarks using trained dlib shape predictor.
    Uses tight-crop standardization (crop + pad + resize to 512x512) to match training.
    Returns landmarks with original IDs from the annotation schema.

    Args:
        project_root: Session root directory (sessions/{speciesId}/ for session models)
        tag: Model tag matching predictor_{tag}.dat
        image_path: Path to the image to run inference on
        yolo_model_path: Optional path to session-specific YOLO detection model.
                         When provided, YOLO is used for specimen detection;
                         falls back to OpenCV contours if YOLO finds nothing.
        input_box: Optional pre-detected box dict. When provided with obbCorners, the
                   OBB geometry engine is used instead of detection + PCA canonicalization.
    """
    modeldir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Model not found: {predictor_path}")

    # Load ID mapping and training metadata.
    orientation_policy = {}
    try:
        orientation_policy = ou.load_orientation_policy(project_root)
    except Exception:
        orientation_policy = {}
    id_mapping = {}
    index_to_original = {}
    target_orientation = None
    landmark_template = {}
    head_landmark_id = None
    tail_landmark_id = _resolve_tail_landmark_id(project_root, None)
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
            index_to_original = _resolve_dlib_index_mapping(project_root, tag, id_mapping)
            target_orientation = id_mapping.get("training_config", {}).get("target_orientation")
            landmark_template = {
                int(k): v for k, v in id_mapping.get("landmark_template", {}).items()
            }
            head_landmark_id = _resolve_head_landmark_id(project_root, id_mapping)
            tail_landmark_id = _resolve_tail_landmark_id(project_root, id_mapping)
    else:
        head_landmark_id = _resolve_head_landmark_id(project_root, None)

    print("PROGRESS 10 loading_model", file=sys.stderr)

    # Load and resize image
    img, orig_w, orig_h = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Could not read: {image_path}")

    img_hash = hashlib.md5(open(image_path, 'rb').read(1000)).hexdigest()[:8]

    max_dim = 1500
    scale = 1.0
    w, h = orig_w, orig_h
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        w, h = int(orig_w * scale), int(orig_h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # Detect specimen bounding box (skip when caller provides input_box)
    temp_path = None
    if input_box is not None:
        detected = dict(input_box)
    else:
        if scale != 1.0:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            cv2.imwrite(temp_path, img)
            detection_path = temp_path
        else:
            detection_path = image_path

        detected = detect_specimen(detection_path, margin=20, yolo_model_path=yolo_model_path)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if detected is None:
            detected = {'left': 0, 'top': 0, 'right': w, 'bottom': h, 'width': w, 'height': h}

    print("PROGRESS 30 detecting", file=sys.stderr)

    orientation_hint = _resolve_orientation_hint_from_box(
        detected,
        min_confidence=0.25,
        min_dx_ratio=0.06,
    )

    # Standardize: crop using OBB geometry engine when obb_corners available,
    # otherwise fall back to axis-aligned crop + PCA canonicalization.
    cropped = None
    crop_meta = None
    canonicalization_debug = None
    mask_outline = None

    obb_corners = detected.get("obbCorners") or detected.get("obb_corners") if isinstance(detected, dict) else None
    obb_class_id = int(detected.get("class_id", 0)) if isinstance(detected, dict) else 0
    if obb_corners and len(obb_corners) == 4:
        apply_leveling = (orientation_policy.get("obbLevelingMode", "on") == "on")
        cropped, crop_meta = ou.extract_obb_crop(img, obb_corners, pad_ratio=0.15, apply_leveling=apply_leveling)
        cropped, crop_meta, canonicalization_debug = ou.apply_obb_geometry(
            cropped, crop_meta, obb_class_id, orientation_policy
        )
        if isinstance(canonicalization_debug, dict):
            canonicalization_debug["source"] = "obb_geometry"
    else:
        cropped, crop_meta = standardize_crop(img, detected)
        if _should_try_canonicalization(orientation_policy):
            sam_mask = _sam2_mask_for_box(img, detected)
            if sam_mask is not None:
                mask_outline = _mask_outline_from_full_mask(
                    sam_mask,
                    scale=scale,
                    image_shape=(orig_h, orig_w),
                )
                cropped, crop_meta, canonicalization_debug = _canonicalize_crop_with_mask(
                    cropped,
                    crop_meta,
                    sam_mask,
                    orientation_policy,
                    orientation_hint,
                )
                if isinstance(canonicalization_debug, dict):
                    canonicalization_debug["mask_source"] = "sam2"
            else:
                rough_mask = _rough_mask_from_crop(cropped)
                if rough_mask is not None:
                    cropped, crop_meta, canonicalization_debug = _canonicalize_crop_with_standard_mask(
                        cropped,
                        crop_meta,
                        rough_mask,
                        orientation_policy,
                        orientation_hint,
                    )
                    if isinstance(canonicalization_debug, dict):
                        canonicalization_debug["mask_source"] = "rough_otsu"

    # Run dlib on the standardized 512x512 image (full-image rect)
    rect = dlib.rectangle(0, 0, STANDARD_SIZE, STANDARD_SIZE)
    predictor = dlib.shape_predictor(predictor_path)
    predict_fn = _make_dlib_predict_fn(predictor, rect, index_to_original)
    landmarks_512, was_flipped, orientation_debug = _predict_with_orientation_lock(
        crop_512=cropped,
        predict_fn=predict_fn,
        orientation_policy=orientation_policy,
        canonicalization_debug=canonicalization_debug,
        target_orientation=target_orientation,
        landmark_template=landmark_template,
        head_landmark_id=head_landmark_id,
        tail_landmark_id=tail_landmark_id,
        orientation_hint=orientation_hint,
    )
    if isinstance(orientation_debug, dict):
        if canonicalization_debug is not None:
            orientation_debug["canonicalization"] = canonicalization_debug
        orientation_debug["was_flipped"] = bool(was_flipped)
        orientation_debug["orientation_hint"] = orientation_hint
        orientation_debug["orientation_hint_raw"] = (
            detected.get("orientation_hint") if isinstance(detected, dict) else None
        )

    print("PROGRESS 65 predicting", file=sys.stderr)

    # Map back to original image coordinates (un-flip if the crop was flipped)
    landmarks = sorted(
        map_landmarks_to_original(
            landmarks_512,
            crop_meta,
            scale,
            was_flipped,
            image_shape=(orig_h, orig_w),
        ),
        key=lambda lm: lm["id"]
    )

    print("PROGRESS 90 mapping", file=sys.stderr)

    # Scale bounding box to original size (skip non-numeric fields like detection_method)
    if scale != 1.0:
        detected = {k: int(v / scale) if isinstance(v, (int, float)) else v for k, v in detected.items()}
        if isinstance(detected.get("orientation_hint"), dict):
            detected["orientation_hint"] = _scale_orientation_hint(detected["orientation_hint"], scale)

    result = {
        "image": image_path,
        "landmarks": landmarks,
        "detected_box": detected,
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "inference_scale": scale,
        "num_landmarks": len(landmarks),
        "id_mapping": index_to_original if index_to_original else None,
        "detection_method": detected.get("detection_method", "opencv") if isinstance(detected, dict) else "opencv",
        "orientation_hint": detected.get("orientation_hint") if isinstance(detected, dict) else None,
        "inference_metadata": _build_inference_metadata(orientation_debug),
        "mask_outline": mask_outline,
    }

    # Save prediction log for debugging
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "num_landmarks": len(landmarks),
        "landmarks": landmarks,
        "detected_box": detected,
        "scale": scale,
        "debug_hash": img_hash,
        "orientation_debug": orientation_debug,
        "inference_metadata": _build_inference_metadata(orientation_debug),
    }
    save_prediction_log(debug_dir, tag, log_entry)

    return result


def predict_multi_specimen(
    project_root,
    tag,
    image_path,
    min_area_ratio=0.02,
    yolo_model_path=None,
    input_boxes=None,
):
    """
    Predict landmarks for multiple specimens using detection + dlib.
    Uses tight-crop standardization per specimen to match training.
    Returns list of {box, landmarks} for each detected specimen.

    Args:
        yolo_model_path: Optional path to session YOLO detection model; preferred over OpenCV.
    """
    from detection.detect_specimen import detect_multiple_specimens

    modeldir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Model not found: {predictor_path}")

    # Load ID mapping and training metadata.
    orientation_policy = {}
    try:
        orientation_policy = ou.load_orientation_policy(project_root)
    except Exception:
        orientation_policy = {}
    id_mapping = {}
    index_to_original = {}
    target_orientation = None
    landmark_template = {}
    head_landmark_id = None
    tail_landmark_id = _resolve_tail_landmark_id(project_root, None)
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
            index_to_original = _resolve_dlib_index_mapping(project_root, tag, id_mapping)
            target_orientation = id_mapping.get("training_config", {}).get("target_orientation")
            landmark_template = {
                int(k): v for k, v in id_mapping.get("landmark_template", {}).items()
            }
            head_landmark_id = _resolve_head_landmark_id(project_root, id_mapping)
            tail_landmark_id = _resolve_tail_landmark_id(project_root, id_mapping)
    else:
        head_landmark_id = _resolve_head_landmark_id(project_root, None)

    print("PROGRESS 10 loading_model", file=sys.stderr)

    # Load and resize image
    img, orig_w, orig_h = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Could not read: {image_path}")

    max_dim = 1500
    scale = 1.0
    w, h = orig_w, orig_h
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        w, h = int(orig_w * scale), int(orig_h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    detection_method = "provided_boxes"
    if isinstance(input_boxes, list) and input_boxes:
        detected_boxes = _normalize_input_boxes(input_boxes, scale=scale, image_w=w, image_h=h)
        detection_result = {"boxes": detected_boxes, "detection_method": detection_method}
    else:
        # Save temp image for detection if scaled
        temp_path = None
        if scale != 1.0:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            cv2.imwrite(temp_path, img)
            detection_path = temp_path
        else:
            detection_path = image_path

        # Detect multiple specimens (YOLO-first if model provided, else OpenCV)
        detection_result = detect_multiple_specimens(detection_path, min_area_ratio=min_area_ratio,
                                                     yolo_model_path=yolo_model_path)
        detected_boxes = detection_result.get("boxes", [])

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    print("PROGRESS 30 detecting", file=sys.stderr)

    if not detected_boxes:
        return {
            "image": image_path,
            "specimens": [],
            "num_specimens": 0,
            "image_dimensions": {"width": orig_w, "height": orig_h}
        }

    # Load dlib predictor
    predictor = dlib.shape_predictor(predictor_path)
    rect = dlib.rectangle(0, 0, STANDARD_SIZE, STANDARD_SIZE)
    predict_fn = _make_dlib_predict_fn(predictor, rect, index_to_original)

    specimens = []
    for box_idx, box in enumerate(detected_boxes):
        # Report incremental progress per specimen (40–85 range)
        pct = 40 + int(45 * (box_idx / max(len(detected_boxes), 1)))
        print(f"PROGRESS {pct} predicting", file=sys.stderr)
        orientation_hint = _resolve_orientation_hint_from_box(
            box,
            min_confidence=0.25,
            min_dx_ratio=0.06,
        )

        # Standardize: crop to bounding box + padding, resize to 512x512
        cropped, crop_meta = standardize_crop(img, box)
        canonicalization_debug = None
        mask_outline = None
        if _should_try_canonicalization(orientation_policy):
            sam_mask = _sam2_mask_for_box(img, box)
            if sam_mask is not None:
                mask_outline = _mask_outline_from_full_mask(
                    sam_mask,
                    scale=scale,
                    image_shape=(orig_h, orig_w),
                )
                cropped, crop_meta, canonicalization_debug = _canonicalize_crop_with_mask(
                    cropped,
                    crop_meta,
                    sam_mask,
                    orientation_policy,
                    orientation_hint,
                )
                if isinstance(canonicalization_debug, dict):
                    canonicalization_debug["mask_source"] = "sam2"
            else:
                rough_mask = _rough_mask_from_crop(cropped)
                if rough_mask is not None:
                    cropped, crop_meta, canonicalization_debug = _canonicalize_crop_with_standard_mask(
                        cropped,
                        crop_meta,
                        rough_mask,
                        orientation_policy,
                        orientation_hint,
                    )
                    if isinstance(canonicalization_debug, dict):
                        canonicalization_debug["mask_source"] = "rough_otsu"

        landmarks_512, was_flipped, orientation_debug = _predict_with_orientation_lock(
            crop_512=cropped,
            predict_fn=predict_fn,
            orientation_policy=orientation_policy,
            canonicalization_debug=canonicalization_debug,
            target_orientation=target_orientation,
            landmark_template=landmark_template,
            head_landmark_id=head_landmark_id,
            tail_landmark_id=tail_landmark_id,
            orientation_hint=orientation_hint,
        )
        if isinstance(orientation_debug, dict):
            if canonicalization_debug is not None:
                orientation_debug["canonicalization"] = canonicalization_debug
            orientation_debug["was_flipped"] = bool(was_flipped)
            orientation_debug["orientation_hint"] = orientation_hint
            orientation_debug["orientation_hint_raw"] = box.get("orientation_hint") if isinstance(box, dict) else None

        # Map back to original image coordinates (un-flip if the crop was flipped)
        landmarks = sorted(
            map_landmarks_to_original(
                landmarks_512,
                crop_meta,
                scale,
                was_flipped,
                image_shape=(orig_h, orig_w),
            ),
            key=lambda lm: lm["id"]
        )

        # Scale bounding box back to original size
        scaled_box = {}
        if scale != 1.0:
            for k, v in box.items():
                if isinstance(v, (int, float)):
                    scaled_box[k] = int(v / scale)
                elif k in ("obbCorners", "obb_corners") and isinstance(v, list):
                    scaled_box[k] = [[pt[0] / scale, pt[1] / scale] for pt in v]
                else:
                    scaled_box[k] = v
        else:
            scaled_box = dict(box)
        scaled_box["confidence"] = box.get("confidence")
        scaled_box["class_name"] = box.get("class_name")
        if isinstance(box.get("orientation_hint"), dict):
            scaled_box["orientation_hint"] = _scale_orientation_hint(box.get("orientation_hint"), scale)

        specimens.append({
            "box": scaled_box,
            "landmarks": landmarks,
            "num_landmarks": len(landmarks),
            "orientation_debug": orientation_debug,
            "inference_metadata": _build_inference_metadata(orientation_debug),
            "mask_outline": mask_outline,
        })

    print("PROGRESS 90 mapping", file=sys.stderr)

    result = {
        "image": image_path,
        "specimens": specimens,
        "num_specimens": len(specimens),
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "inference_scale": scale,
        "detection_method": detection_result.get("detection_method", "opencv"),
    }

    # Save prediction log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image": os.path.basename(image_path),
        "mode": "multi-specimen",
        "num_specimens": len(specimens),
        "specimens": specimens
    }
    save_prediction_log(debug_dir, tag, log_entry)

    return result


# ── CNN inference paths ────────────────────────────────────────────────────────

def _load_cnn_model(project_root, tag):
    """
    Load CNN model + config + orientation metadata.

    Returns:
        (model, landmark_ids, target_orientation, landmark_template, head_landmark_id, tail_landmark_id)
    """
    if not _torch_available:
        raise RuntimeError(
            "torch/torchvision not installed. Cannot use CNN predictor. "
            "Install with: pip install torch torchvision"
        )
    modeldir = os.path.join(project_root, "models")
    model_path = os.path.join(modeldir, f"cnn_{tag}.pth")
    config_path = os.path.join(modeldir, f"cnn_{tag}_config.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN model not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"CNN config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    n_landmarks = config["n_landmarks"]
    landmark_ids = config.get("landmark_ids", list(range(n_landmarks)))
    model_variant = config.get("model_variant_resolved") or config.get("model_variant_requested") or "efficientnet_b0"
    head_type = str(config.get("cnn_head_type", "regression")).strip().lower()
    deconv_layers = int(config.get("cnn_deconv_layers", 3))
    deconv_filters = int(config.get("cnn_deconv_filters", 256))
    softargmax_beta = float(config.get("cnn_softargmax_beta", 25.0))

    state = torch.load(model_path, map_location="cpu")
    model = None
    # Preferred path: new heatmap-deconv architecture.
    if head_type == "heatmap_deconv":
        try:
            model = CNNLandmarkPredictor(
                n_landmarks,
                model_variant=model_variant,
                head_type="heatmap_deconv",
                deconv_layers=deconv_layers,
                deconv_filters=deconv_filters,
                softargmax_beta=softargmax_beta,
            )
            model.load_state_dict(state, strict=True)
        except Exception:
            model = None

    # Fallback for older checkpoints.
    if model is None:
        model = CNNLandmarkPredictorLegacy(n_landmarks, model_variant=model_variant)
        model.load_state_dict(state, strict=False)
    model.eval()

    # Load orientation / template data from id_mapping (same as dlib path)
    target_orientation = None
    landmark_template = {}
    head_landmark_id = None
    tail_landmark_id = _resolve_tail_landmark_id(project_root, None)
    id_mapping_path = os.path.join(project_root, "debug", f"id_mapping_{tag}.json")
    if os.path.exists(id_mapping_path):
        try:
            with open(id_mapping_path, "r", encoding="utf-8") as f:
                id_map = json.load(f)
            target_orientation = id_map.get("training_config", {}).get("target_orientation")
            landmark_template = {
                int(k): v for k, v in id_map.get("landmark_template", {}).items()
            }
            head_landmark_id = _resolve_head_landmark_id(project_root, id_map)
            tail_landmark_id = _resolve_tail_landmark_id(project_root, id_map)
        except Exception:
            pass

    return model, landmark_ids, target_orientation, landmark_template, head_landmark_id, tail_landmark_id


def _cnn_landmarks_from_coords(coords_np, landmark_ids, flip=False):
    """Convert flat CNN output coords to landmark dicts, optionally un-flipping X."""
    denom = float(max(1, STANDARD_SIZE - 1))
    lms = []
    for i, lm_id in enumerate(landmark_ids):
        x = float(coords_np[i * 2]) * denom
        y = float(coords_np[i * 2 + 1]) * denom
        if flip:
            x = (STANDARD_SIZE - 1) - x
        x = max(0.0, min(float(STANDARD_SIZE - 1), x))
        y = max(0.0, min(float(STANDARD_SIZE - 1), y))
        lms.append({"id": lm_id, "x": x, "y": y})
    return lms


def predict_cnn_image(project_root, tag, image_path, yolo_model_path=None):
    """
    Predict landmarks using trained CNN shape predictor.
    Same detection + standardization pipeline as the dlib path.
    """
    debug_dir = os.path.join(project_root, "debug")

    orientation_policy = {}
    try:
        orientation_policy = ou.load_orientation_policy(project_root)
    except Exception:
        orientation_policy = {}

    print("PROGRESS 10 loading_model", file=sys.stderr)
    model, landmark_ids, target_orientation, landmark_template, head_landmark_id, tail_landmark_id = \
        _load_cnn_model(project_root, tag)

    # Load and resize image
    img, orig_w, orig_h = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Could not read: {image_path}")

    max_dim = 1500
    scale = 1.0
    w, h = orig_w, orig_h
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        w, h = int(orig_w * scale), int(orig_h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # Detection
    temp_path = None
    if scale != 1.0:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        cv2.imwrite(temp_path, img)
        detection_path = temp_path
    else:
        detection_path = image_path

    detected = detect_specimen(detection_path, margin=20, yolo_model_path=yolo_model_path)
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    print("PROGRESS 30 detecting", file=sys.stderr)

    if detected is None:
        detected = {'left': 0, 'top': 0, 'right': w, 'bottom': h, 'width': w, 'height': h}
    orientation_hint = _resolve_orientation_hint_from_box(
        detected,
        min_confidence=0.25,
        min_dx_ratio=0.06,
    )

    cropped, crop_meta = standardize_crop(img, detected)
    canonicalization_debug = None
    mask_outline = None
    if _should_try_canonicalization(orientation_policy):
        sam_mask = _sam2_mask_for_box(img, detected)
        if sam_mask is not None:
            mask_outline = _mask_outline_from_full_mask(
                sam_mask,
                scale=scale,
                image_shape=(orig_h, orig_w),
            )
            cropped, crop_meta, canonicalization_debug = _canonicalize_crop_with_mask(
                cropped,
                crop_meta,
                sam_mask,
                orientation_policy,
                orientation_hint,
            )
            if isinstance(canonicalization_debug, dict):
                canonicalization_debug["mask_source"] = "sam2"
        else:
            rough_mask = _rough_mask_from_crop(cropped)
            if rough_mask is not None:
                cropped, crop_meta, canonicalization_debug = _canonicalize_crop_with_standard_mask(
                    cropped,
                    crop_meta,
                    rough_mask,
                    orientation_policy,
                    orientation_hint,
                )
                if isinstance(canonicalization_debug, dict):
                    canonicalization_debug["mask_source"] = "rough_otsu"

    print("PROGRESS 65 predicting", file=sys.stderr)

    predict_fn = _make_cnn_predict_fn(model, landmark_ids)
    landmarks_512, was_flipped, orientation_debug = _predict_with_orientation_lock(
        crop_512=cropped,
        predict_fn=predict_fn,
        orientation_policy=orientation_policy,
        canonicalization_debug=canonicalization_debug,
        target_orientation=target_orientation,
        landmark_template=landmark_template,
        head_landmark_id=head_landmark_id,
        tail_landmark_id=tail_landmark_id,
        orientation_hint=orientation_hint,
    )
    if isinstance(orientation_debug, dict):
        if canonicalization_debug is not None:
            orientation_debug["canonicalization"] = canonicalization_debug
        orientation_debug["was_flipped"] = bool(was_flipped)
        orientation_debug["orientation_hint"] = orientation_hint
        orientation_debug["orientation_hint_raw"] = (
            detected.get("orientation_hint") if isinstance(detected, dict) else None
        )

    print("PROGRESS 90 mapping", file=sys.stderr)

    landmarks = sorted(
        map_landmarks_to_original(
            landmarks_512,
            crop_meta,
            scale,
            was_flipped=was_flipped,
            image_shape=(orig_h, orig_w),
        ),
        key=lambda lm: lm["id"]
    )

    if scale != 1.0:
        detected = {k: int(v / scale) if isinstance(v, (int, float)) else v for k, v in detected.items()}
        if isinstance(detected.get("orientation_hint"), dict):
            detected["orientation_hint"] = _scale_orientation_hint(detected["orientation_hint"], scale)

    detection_method = detected.get("detection_method", "opencv") if isinstance(detected, dict) else "opencv"
    fallback_reason = detected.get("fallback_reason") if isinstance(detected, dict) else None

    result = {
        "image": image_path,
        "landmarks": landmarks,
        "detected_box": detected,
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "inference_scale": scale,
        "num_landmarks": len(landmarks),
        "detection_method": detection_method,
        "fallback_reason": fallback_reason,
        "predictor_type": "cnn",
        "predictor_variant": getattr(model, "model_variant", None),
        "orientation_hint": detected.get("orientation_hint") if isinstance(detected, dict) else None,
        "orientation_debug": orientation_debug or {
            "was_flipped": bool(was_flipped),
            "orientation_hint": orientation_hint,
        },
        "inference_metadata": _build_inference_metadata(
            orientation_debug or {
                "was_flipped": bool(was_flipped),
                "orientation_hint": orientation_hint,
            }
        ),
        "mask_outline": mask_outline,
    }

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image": os.path.basename(image_path),
        "predictor_type": "cnn",
        "predictor_variant": getattr(model, "model_variant", None),
        "num_landmarks": len(landmarks),
        "landmarks": landmarks,
        "detected_box": detected,
        "detection_method": detection_method,
        "fallback_reason": fallback_reason,
        "scale": scale,
        "was_flipped": was_flipped,
        "orientation_debug": orientation_debug
        or {"was_flipped": bool(was_flipped), "orientation_hint": orientation_hint},
        "inference_metadata": _build_inference_metadata(
            orientation_debug or {
                "was_flipped": bool(was_flipped),
                "orientation_hint": orientation_hint,
            }
        ),
    }
    save_prediction_log(debug_dir, tag, log_entry, predictor_type="cnn")
    return result


def predict_cnn_multi_specimen(
    project_root,
    tag,
    image_path,
    min_area_ratio=0.02,
    yolo_model_path=None,
    input_boxes=None,
):
    """
    Predict landmarks for multiple specimens using detection + CNN.
    """
    from detection.detect_specimen import detect_multiple_specimens

    debug_dir = os.path.join(project_root, "debug")

    orientation_policy = {}
    try:
        orientation_policy = ou.load_orientation_policy(project_root)
    except Exception:
        orientation_policy = {}

    print("PROGRESS 10 loading_model", file=sys.stderr)
    model, landmark_ids, target_orientation, landmark_template, head_landmark_id, tail_landmark_id = \
        _load_cnn_model(project_root, tag)

    img, orig_w, orig_h = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Could not read: {image_path}")

    max_dim = 1500
    scale = 1.0
    w, h = orig_w, orig_h
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        w, h = int(orig_w * scale), int(orig_h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    if isinstance(input_boxes, list) and input_boxes:
        detected_boxes = _normalize_input_boxes(input_boxes, scale=scale, image_w=w, image_h=h)
        detection_result = {"boxes": detected_boxes, "detection_method": "provided_boxes"}
    else:
        temp_path = None
        if scale != 1.0:
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            cv2.imwrite(temp_path, img)
            detection_path = temp_path
        else:
            detection_path = image_path

        detection_result = detect_multiple_specimens(detection_path, min_area_ratio=min_area_ratio,
                                                     yolo_model_path=yolo_model_path)
        detected_boxes = detection_result.get("boxes", [])
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    print("PROGRESS 30 detecting", file=sys.stderr)

    detection_method = detection_result.get("detection_method", "opencv")
    fallback_reason = detection_result.get("fallback_reason")

    if not detected_boxes:
        return {
            "image": image_path,
            "specimens": [],
            "num_specimens": 0,
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "detection_method": detection_method,
            "fallback_reason": fallback_reason,
        }

    specimens = []
    predict_fn = _make_cnn_predict_fn(model, landmark_ids)
    for box_idx, box in enumerate(detected_boxes):
        pct = 40 + int(45 * (box_idx / max(len(detected_boxes), 1)))
        print(f"PROGRESS {pct} predicting", file=sys.stderr)
        orientation_hint = _resolve_orientation_hint_from_box(
            box,
            min_confidence=0.25,
            min_dx_ratio=0.06,
        )

        cropped, crop_meta = standardize_crop(img, box)
        canonicalization_debug = None
        mask_outline = None
        if _should_try_canonicalization(orientation_policy):
            sam_mask = _sam2_mask_for_box(img, box)
            if sam_mask is not None:
                mask_outline = _mask_outline_from_full_mask(
                    sam_mask,
                    scale=scale,
                    image_shape=(orig_h, orig_w),
                )
                cropped, crop_meta, canonicalization_debug = _canonicalize_crop_with_mask(
                    cropped,
                    crop_meta,
                    sam_mask,
                    orientation_policy,
                    orientation_hint,
                )
                if isinstance(canonicalization_debug, dict):
                    canonicalization_debug["mask_source"] = "sam2"
            else:
                rough_mask = _rough_mask_from_crop(cropped)
                if rough_mask is not None:
                    cropped, crop_meta, canonicalization_debug = _canonicalize_crop_with_standard_mask(
                        cropped,
                        crop_meta,
                        rough_mask,
                        orientation_policy,
                        orientation_hint,
                    )
                    if isinstance(canonicalization_debug, dict):
                        canonicalization_debug["mask_source"] = "rough_otsu"

        landmarks_512, was_flipped, orientation_debug = _predict_with_orientation_lock(
            crop_512=cropped,
            predict_fn=predict_fn,
            orientation_policy=orientation_policy,
            canonicalization_debug=canonicalization_debug,
            target_orientation=target_orientation,
            landmark_template=landmark_template,
            head_landmark_id=head_landmark_id,
            tail_landmark_id=tail_landmark_id,
            orientation_hint=orientation_hint,
        )
        if isinstance(orientation_debug, dict):
            if canonicalization_debug is not None:
                orientation_debug["canonicalization"] = canonicalization_debug
            orientation_debug["was_flipped"] = bool(was_flipped)
            orientation_debug["orientation_hint"] = orientation_hint
            orientation_debug["orientation_hint_raw"] = box.get("orientation_hint") if isinstance(box, dict) else None

        landmarks = sorted(
            map_landmarks_to_original(
                landmarks_512,
                crop_meta,
                scale,
                was_flipped=was_flipped,
                image_shape=(orig_h, orig_w),
            ),
            key=lambda lm: lm["id"]
        )

        # Scale bounding box back to original size
        scaled_box = {}
        if scale != 1.0:
            for k, v in box.items():
                if isinstance(v, (int, float)):
                    scaled_box[k] = int(v / scale)
                elif k in ("obbCorners", "obb_corners") and isinstance(v, list):
                    scaled_box[k] = [[pt[0] / scale, pt[1] / scale] for pt in v]
                else:
                    scaled_box[k] = v
        else:
            scaled_box = dict(box)
        scaled_box["confidence"] = box.get("confidence")
        scaled_box["class_name"] = box.get("class_name")
        if isinstance(box.get("orientation_hint"), dict):
            scaled_box["orientation_hint"] = _scale_orientation_hint(box.get("orientation_hint"), scale)

        specimen_orientation_debug = orientation_debug or {
            "was_flipped": bool(was_flipped),
            "orientation_hint": orientation_hint,
        }
        specimens.append({
            "box": scaled_box,
            "landmarks": landmarks,
            "num_landmarks": len(landmarks),
            "orientation_debug": specimen_orientation_debug,
            "inference_metadata": _build_inference_metadata(specimen_orientation_debug),
            "mask_outline": mask_outline,
        })

    print("PROGRESS 90 mapping", file=sys.stderr)

    result = {
        "image": image_path,
        "specimens": specimens,
        "num_specimens": len(specimens),
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "inference_scale": scale,
        "detection_method": detection_method,
        "fallback_reason": fallback_reason,
        "predictor_type": "cnn",
        "predictor_variant": getattr(model, "model_variant", None),
    }

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image": os.path.basename(image_path),
        "mode": "multi-specimen",
        "predictor_type": "cnn",
        "num_specimens": len(specimens),
        "detection_method": detection_method,
        "fallback_reason": fallback_reason,
        "specimens": specimens,
    }
    save_prediction_log(debug_dir, tag, log_entry, predictor_type="cnn")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python predict.py <project_root> <tag> <image_path> "
            "[--multi] [--yolo-model <path>] "
            "[--boxes-json <path>] [--obb-json <path>] [--predictor-type dlib|cnn]"
        )
        sys.exit(1)

    project_root = sys.argv[1]
    tag = sys.argv[2]
    image_path = sys.argv[3]
    multi_mode = "--multi" in sys.argv

    yolo_model_path = None
    if "--yolo-model" in sys.argv:
        idx = sys.argv.index("--yolo-model")
        if idx + 1 < len(sys.argv):
            yolo_model_path = sys.argv[idx + 1]

    predictor_type = "dlib"
    if "--predictor-type" in sys.argv:
        idx = sys.argv.index("--predictor-type")
        if idx + 1 < len(sys.argv):
            predictor_type = sys.argv[idx + 1]

    input_boxes = None
    if "--boxes-json" in sys.argv:
        idx = sys.argv.index("--boxes-json")
        if idx + 1 < len(sys.argv):
            boxes_path = sys.argv[idx + 1]
            if os.path.exists(boxes_path):
                try:
                    with open(boxes_path, "r", encoding="utf-8") as f:
                        raw_boxes = json.load(f)
                    if isinstance(raw_boxes, list):
                        input_boxes = raw_boxes
                except Exception:
                    input_boxes = None

    # --obb-json: path to a single OBB detection dict (for single-specimen OBB inference)
    obb_input_box = None
    if "--obb-json" in sys.argv:
        idx = sys.argv.index("--obb-json")
        if idx + 1 < len(sys.argv):
            obb_json_path = sys.argv[idx + 1]
            if os.path.exists(obb_json_path):
                try:
                    with open(obb_json_path, "r", encoding="utf-8") as f:
                        obb_input_box = json.load(f)
                except Exception:
                    obb_input_box = None

    if input_boxes is not None and not multi_mode:
        multi_mode = True

    if multi_mode:
        if predictor_type == "cnn":
            result = predict_cnn_multi_specimen(project_root, tag, image_path,
                                                yolo_model_path=yolo_model_path,
                                                input_boxes=input_boxes)
        else:
            result = predict_multi_specimen(project_root, tag, image_path,
                                            yolo_model_path=yolo_model_path,
                                            input_boxes=input_boxes)
    else:
        if predictor_type == "cnn":
            result = predict_cnn_image(project_root, tag, image_path,
                                       yolo_model_path=yolo_model_path)
        else:
            result = predict_image(project_root, tag, image_path,
                                   yolo_model_path=yolo_model_path,
                                   input_box=obb_input_box)

    print("PROGRESS 100 done", file=sys.stderr)
    print(json.dumps(result))
