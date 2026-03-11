import math
import os
import json
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np


STANDARD_SIZE = 512
STANDARDIZED_OBB_PAD_RATIO = 0.05
STANDARDIZED_OBB_INTERPOLATION = cv2.INTER_LINEAR
ORIENTATION_MODES = {"directional", "bilateral", "axial", "invariant"}


def load_augmentation_policy(session_dir: str) -> dict:
    """Load augmentationPolicy block from session.json, or {} if absent/unreadable."""
    session_path = os.path.join(session_dir, "session.json")
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return dict(meta.get("augmentationPolicy") or {})
    except Exception:
        return {}


def _safe_list_of_str(value: Any, default: Sequence[str] | None = None) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return [str(v).strip().lower() for v in (default or []) if str(v).strip()]
    out: list[str] = []
    for item in value:
        item_s = str(item).strip().lower()
        if item_s and item_s not in out:
            out.append(item_s)
    return out


def _safe_orientation_mode(mode: Any) -> str:
    mode_s = str(mode or "").strip().lower()
    if mode_s in ORIENTATION_MODES:
        return mode_s
    return "invariant"


def _safe_orientation_target(value: Any) -> str:
    target = str(value or "").strip().lower()
    if target in ("left", "right"):
        return target
    return "left"


def _safe_direction(value: Any) -> str | None:
    direction = str(value or "").strip().lower()
    if direction in ("left", "right"):
        return direction
    return None


def _safe_direction_priority(value: Any) -> str:
    priority = str(value or "auto").strip().lower()
    if priority in ("auto", "hint_first"):
        return priority
    return "auto"


def _template_has_category(landmark_template: Sequence[Mapping[str, Any]] | None, category: str) -> bool:
    if not isinstance(landmark_template, Sequence):
        return False
    target = str(category).strip().lower()
    for lm in landmark_template:
        cat = str(lm.get("category", "")).strip().lower()
        if cat == target:
            return True
    return False


def infer_orientation_policy_from_template(
    landmark_template: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """
    Infer a safe default orientation policy from schema landmark categories.

    - directional: schema has head/tail categories
    - invariant: no directional cues
    """
    has_head = _template_has_category(landmark_template, "head")
    has_tail = _template_has_category(landmark_template, "tail")
    has_caudal_tail = _template_has_category(landmark_template, "caudal-fin")
    tail_categories: list[str] = []
    if has_tail:
        tail_categories.append("tail")
    if has_caudal_tail:
        tail_categories.append("caudal-fin")
    if not tail_categories:
        tail_categories = ["tail"]
    if has_head or has_tail or has_caudal_tail:
        return {
            "mode": "directional",
            "targetOrientation": "left",
            "headCategories": ["head"],
            "tailCategories": tail_categories,
            "pcaLevelingMode": "auto",
        }
    return {
        "mode": "invariant",
        "pcaLevelingMode": "off",
    }


def sanitize_orientation_policy(
    policy: Mapping[str, Any] | None,
    landmark_template: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    inferred = infer_orientation_policy_from_template(landmark_template)
    raw = dict(policy or {})

    mode = _safe_orientation_mode(raw.get("mode", inferred.get("mode")))
    target_orientation = _safe_orientation_target(
        raw.get("targetOrientation", inferred.get("targetOrientation"))
    )
    head_categories = _safe_list_of_str(
        raw.get("headCategories"), inferred.get("headCategories", ["head"])
    )
    tail_categories = _safe_list_of_str(
        raw.get("tailCategories"), inferred.get("tailCategories", ["tail"])
    )
    pca_mode = str(raw.get("pcaLevelingMode", inferred.get("pcaLevelingMode", "off"))).strip().lower()
    if pca_mode not in ("off", "on", "auto"):
        pca_mode = "off"
    direction_priority = _safe_direction_priority(raw.get("directionPriority", "auto"))
    try:
        min_moment_conf = max(0.0, float(raw.get("minMomentDirectionConfidence", 0.03)))
    except Exception:
        min_moment_conf = 0.03
    template_direction_fallback = bool(raw.get("templateDirectionFallback", False))
    training_prep_box_jitter = raw.get("trainingPrepBoxJitter", "auto")
    if not isinstance(training_prep_box_jitter, (dict, str, bool, int, float)):
        training_prep_box_jitter = "auto"

    bilateral_pairs: list[list[int]] = []
    if isinstance(raw.get("bilateralPairs"), (list, tuple)):
        for pair in raw["bilateralPairs"]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            try:
                bilateral_pairs.append([int(pair[0]), int(pair[1])])
            except Exception:
                continue

    obb_leveling_mode = str(raw.get("obbLevelingMode", "on")).strip().lower()
    if obb_leveling_mode not in ("on", "off"):
        obb_leveling_mode = "on"

    out: dict[str, Any] = {
        "mode": mode,
        "pcaLevelingMode": pca_mode,
        "obbLevelingMode": obb_leveling_mode,
        "directionPriority": direction_priority,
        "minMomentDirectionConfidence": float(min_moment_conf),
        "templateDirectionFallback": template_direction_fallback,
        "trainingPrepBoxJitter": training_prep_box_jitter,
    }
    if mode == "directional":
        out["targetOrientation"] = target_orientation
        out["headCategories"] = head_categories or ["head"]
        out["tailCategories"] = tail_categories or ["tail"]
    if mode == "bilateral" and bilateral_pairs:
        out["bilateralPairs"] = bilateral_pairs
    return out


def load_orientation_policy(project_root: str) -> dict[str, Any]:
    session_path = os.path.join(project_root, "session.json")
    session: dict[str, Any] = {}
    if os.path.exists(session_path):
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                session = json.load(f)
        except Exception:
            session = {}
    template = session.get("landmarkTemplate")
    policy = session.get("orientationPolicy")
    return sanitize_orientation_policy(policy, template if isinstance(template, list) else [])


def load_session_landmark_template(project_root: str) -> list[dict[str, Any]]:
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
    return [lm for lm in template if isinstance(lm, Mapping)]


def resolve_landmark_id_from_categories(
    landmark_template: Sequence[Mapping[str, Any]],
    categories: Sequence[str],
    *,
    fallback_to_min: bool = False,
) -> int | None:
    cats = {str(c).strip().lower() for c in categories if str(c).strip()}
    matches: list[int] = []
    all_indices: list[int] = []
    for lm in landmark_template:
        try:
            idx = int(lm.get("index"))
        except Exception:
            continue
        all_indices.append(idx)
        lm_cat = str(lm.get("category", "")).strip().lower()
        if lm_cat in cats:
            matches.append(idx)
    if matches:
        return min(matches)
    if fallback_to_min and all_indices:
        return min(all_indices)
    return None


def resolve_head_tail_landmark_ids(
    project_root: str,
    policy: Mapping[str, Any] | None = None,
) -> tuple[int | None, int | None]:
    template = load_session_landmark_template(project_root)
    if policy is None:
        policy = load_orientation_policy(project_root)
    mode = _safe_orientation_mode(policy.get("mode"))
    head_categories = _safe_list_of_str(policy.get("headCategories"), ["head"])
    tail_categories = _safe_list_of_str(policy.get("tailCategories"), ["tail"])

    # Directional mode needs a deterministic head anchor for orientation checks.
    head_id = resolve_landmark_id_from_categories(
        template, head_categories, fallback_to_min=(mode == "directional")
    )
    tail_id = resolve_landmark_id_from_categories(template, tail_categories, fallback_to_min=False)
    return head_id, tail_id


def default_allow_flip_augmentation(mode: str | None) -> bool:
    resolved_mode = _safe_orientation_mode(mode)
    if resolved_mode == "directional":
        return False
    return True


def get_orientation_mode(mode_or_policy: Any) -> str:
    """
    Resolve orientation mode from either a raw mode string or orientationPolicy.
    """
    if isinstance(mode_or_policy, Mapping):
        return _safe_orientation_mode(mode_or_policy.get("mode"))
    return _safe_orientation_mode(mode_or_policy)


def get_bilateral_pairs(policy: Mapping[str, Any] | None) -> list[tuple[int, int]]:
    """
    Return normalized bilateral landmark ID pairs from orientation policy.
    """
    if not isinstance(policy, Mapping):
        return []
    pairs_raw = policy.get("bilateralPairs")
    if not isinstance(pairs_raw, (list, tuple)):
        return []
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair in pairs_raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            a = int(pair[0])
            b = int(pair[1])
        except Exception:
            continue
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    return pairs


def build_pair_swap_map(pairs: Sequence[Sequence[int]]) -> dict[int, int]:
    """
    Build a bidirectional integer-ID swap map from pairs.
    """
    swap: dict[int, int] = {}
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            a = int(pair[0])
            b = int(pair[1])
        except Exception:
            continue
        if a == b:
            continue
        swap[a] = b
        swap[b] = a
    return swap


def build_dlib_name_swap_map(
    id_mapping_data: Mapping[str, Any] | None,
    pair_swap_map: Mapping[int, int] | None,
) -> dict[str, str]:
    """
    Build dlib part-name swaps (e.g. "00" <-> "01") from original-ID swaps.
    """
    if not isinstance(id_mapping_data, Mapping) or not pair_swap_map:
        return {}

    original_to_name: dict[int, str] = {}
    name_to_original = id_mapping_data.get("dlib_name_to_original")
    if isinstance(name_to_original, Mapping):
        for name, orig in name_to_original.items():
            try:
                original_to_name[int(orig)] = str(name)
            except Exception:
                continue

    if not original_to_name:
        original_to_dlib = id_mapping_data.get("original_to_dlib")
        width_raw = id_mapping_data.get("part_name_width", 2)
        try:
            width = max(1, int(width_raw))
        except Exception:
            width = 2
        if isinstance(original_to_dlib, Mapping):
            for orig, dlib_idx in original_to_dlib.items():
                try:
                    oid = int(orig)
                    didx = int(dlib_idx)
                except Exception:
                    continue
                original_to_name[oid] = f"{didx:0{width}d}"

    if not original_to_name:
        return {}

    name_swap: dict[str, str] = {}
    for src_oid, dst_oid in pair_swap_map.items():
        src_name = original_to_name.get(int(src_oid))
        dst_name = original_to_name.get(int(dst_oid))
        if not src_name or not dst_name:
            continue
        name_swap[src_name] = dst_name
    return name_swap


def get_schema_augmentation_profile(
    mode: str | None, *, engine: str, augmentation_override: dict | None = None
) -> dict[str, Any]:
    """
    Return schema-driven augmentation defaults by model engine.

    engine:
      - "dlib": returns {"angles": [...], "flip": bool}
      - "cnn": returns {
            "flip_prob": float,
            "vertical_flip_prob": float,
            "rotation_range": (min_deg, max_deg),
            "rotate_180_prob": float,
            "scale_range": (min_scale, max_scale),
            "translate_ratio": float,
        }

    augmentation_override: optional dict loaded from session.json augmentationPolicy.
      Fields present in the override replace the corresponding hardcoded defaults.
    """
    resolved_mode = _safe_orientation_mode(mode)
    eng = str(engine or "").strip().lower()

    if eng == "dlib":
        # With OBB leveling, the specimen is already axis-aligned in the 512×512 crop.
        # Tight rotation angles (±6°) preserve sub-pixel spatial stability.
        # Axial schemas no longer need 180° augmentation: apply_obb_geometry handles
        # the canonical flip via class_id at both training and inference time.
        profiles = {
            "directional": {"angles": [-6.0, -3.0, 3.0, 6.0], "flip": False},
            "bilateral": {"angles": [-6.0, -3.0, 3.0, 6.0], "flip": True},
            "axial": {"angles": [-6.0, -3.0, 3.0, 6.0], "flip": True},
            # Invariant specimens (starfish, shell fragments, etc.) have no canonical
            # orientation, so they are immune to the bimodal-target trap. Keep the
            # same ±6° envelope as directional/axial — wide angles (e.g. ±30°) cause
            # mean-shape collapse in dlib because the predictor cannot distinguish a
            # rotated specimen from a different pose. flip: True and box-jitter provide
            # the extra augmentation variance that invariant schemas need. The CNN
            # handles wide angular invariance via its continuous (-180°, 180°) range.
            "invariant": {"angles": [-6.0, -3.0, 3.0, 6.0], "flip": True},
        }
        profile = dict(profiles.get(resolved_mode, profiles["invariant"]))
        if augmentation_override:
            if "angles" in augmentation_override:
                # Dlib regression trees collapse above ±6°; enforce hard cap.
                clamped = [max(-6.0, min(6.0, float(a))) for a in augmentation_override["angles"]]
                profile["angles"] = clamped or [-6.0, -3.0, 3.0, 6.0]
            if "dlib_flip" in augmentation_override:
                profile["flip"] = bool(augmentation_override["dlib_flip"])
        return profile

    if eng == "cnn":
        # With OBB leveling, OBB-oriented schemas use tighter rotation ranges (±10°)
        # to exploit specimen alignment without overriding the geometry engine's work.
        # Invariant schemas retain wide augmentation since they have no orientation concept.
        profiles = {
            "directional": {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.0,
                "rotation_range": (-12.0, 12.0),
                "rotate_180_prob": 0.0,
                "scale_range": (0.95, 1.05),
                "translate_ratio": 0.03,
            },
            "bilateral": {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.0,
                "rotation_range": (-12.0, 12.0),
                "rotate_180_prob": 0.0,
                "scale_range": (0.95, 1.05),
                "translate_ratio": 0.03,
            },
            # Axial: 180° augmentation retained for CNN (class_id-based flip is primary
            # but CNN benefits from rotate_180 augmentation for axial variation).
            "axial": {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.0,
                "rotation_range": (-10.0, 10.0),
                "rotate_180_prob": 0.5,
                "scale_range": (0.94, 1.06),
                "translate_ratio": 0.03,
            },
            # Invariant schemas rely on strong augmentation (no OBB leveling concept).
            "invariant": {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.25,
                "rotation_range": (-180.0, 180.0),
                "rotate_180_prob": 0.0,
                "scale_range": (0.88, 1.12),
                "translate_ratio": 0.06,
            },
        }
        profile = dict(profiles.get(resolved_mode, profiles["invariant"]))
        if augmentation_override:
            for k in ("flip_prob", "vertical_flip_prob", "rotate_180_prob", "translate_ratio"):
                if k in augmentation_override:
                    profile[k] = float(augmentation_override[k])
            if "rotation_range" in augmentation_override:
                rr = augmentation_override["rotation_range"]
                profile["rotation_range"] = tuple(rr) if isinstance(rr, list) else rr
            if "scale_range" in augmentation_override:
                sr = augmentation_override["scale_range"]
                profile["scale_range"] = tuple(sr) if isinstance(sr, list) else sr
        return profile

    raise ValueError(f"Unsupported engine '{engine}' for schema augmentation profile.")


def resolve_session_augmentation_profile(
    session_dir: str,
    *,
    engine: str,
    fallback_mode: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Resolve the effective schema-aware augmentation profile for a session.

    Returns:
        (mode, orientation_policy, augmentation_policy, resolved_profile)
    """
    orientation_policy = load_orientation_policy(session_dir)
    augmentation_policy = load_augmentation_policy(session_dir)
    mode = _safe_orientation_mode(
        orientation_policy.get("mode") if isinstance(orientation_policy, Mapping) else fallback_mode
    )
    if mode not in ORIENTATION_MODES:
        mode = _safe_orientation_mode(fallback_mode)
    profile = get_schema_augmentation_profile(
        mode,
        engine=engine,
        augmentation_override=augmentation_policy,
    )
    return mode, orientation_policy, augmentation_policy, profile


def get_box_jitter_profile(mode: str | None, *, engine: str) -> dict[str, Any]:
    """
    Return schema-specific box-jitter defaults.

    The jitter is intended to simulate detector crop variability.
    - dlib: scale-only source-box variance to protect mean-shape stability.
    - cnn: stronger defaults for detector-noise robustness.
    """
    resolved_mode = _safe_orientation_mode(mode)
    eng = str(engine or "").strip().lower()

    if eng == "dlib":
        profiles = {
            "directional": {
                "enabled": True,
                "copies_per_sample": 1,
                "translate_ratio": 0.0,
                "scale_range": (0.98, 1.02),
                "strategy": "scale_only_pre_standardize",
            },
            "bilateral": {
                "enabled": True,
                "copies_per_sample": 1,
                "translate_ratio": 0.0,
                "scale_range": (0.97, 1.03),
                "strategy": "scale_only_pre_standardize",
            },
            "axial": {
                "enabled": True,
                "copies_per_sample": 1,
                "translate_ratio": 0.0,
                "scale_range": (0.96, 1.04),
                "strategy": "scale_only_pre_standardize",
            },
            "invariant": {
                "enabled": True,
                "copies_per_sample": 2,
                "translate_ratio": 0.0,
                "scale_range": (0.95, 1.05),
                "strategy": "scale_only_pre_standardize",
            },
        }
        return dict(profiles.get(resolved_mode, profiles["invariant"]))

    if False and eng == "dlib":
        # Geometric lock: OBB-leveled crops occupy the canvas tightly.
        # Cap translate at ±3% and scale at ±5% max to avoid wasting model
        # capacity on a spatially wandering specimen.
        profiles = {
            "directional": {
                "enabled": True,
                "copies_per_sample": 1,
                "translate_ratio": 0.0,
                "scale_range": (0.98, 1.02),
                "strategy": "scale_only_pre_standardize",
            },
            "bilateral": {
                "enabled": True,
                "copies_per_sample": 1,
                "translate_ratio": 0.0,
                "scale_range": (0.97, 1.03),
                "strategy": "scale_only_pre_standardize",
            },
            "axial": {
                "enabled": True,
                "copies_per_sample": 1,
                "translate_ratio": 0.03,   # tightened from 0.035 (±3% max)
                "scale_range": (0.95, 1.05),
            },
            "invariant": {
                "enabled": True,
                "copies_per_sample": 2,
                "translate_ratio": 0.03,   # tightened from 0.04 (±3% max)
                "scale_range": (0.95, 1.05),  # tightened from (0.94, 1.06)
            },
        }
        return dict(profiles.get(resolved_mode, profiles["invariant"]))

    if eng == "cnn":
        profiles = {
            "directional": {
                "enabled": True,
                "copies_per_sample": 3,
                "translate_ratio": 0.08,
                "scale_range": (0.88, 1.12),
            },
            "bilateral": {
                "enabled": True,
                "copies_per_sample": 3,
                "translate_ratio": 0.07,
                "scale_range": (0.90, 1.10),
            },
            "axial": {
                "enabled": True,
                "copies_per_sample": 4,
                "translate_ratio": 0.08,
                "scale_range": (0.90, 1.12),
            },
            "invariant": {
                "enabled": True,
                "copies_per_sample": 4,
                "translate_ratio": 0.10,
                "scale_range": (0.85, 1.15),
            },
        }
        return dict(profiles.get(resolved_mode, profiles["invariant"]))

    raise ValueError(f"Unsupported engine '{engine}' for box jitter profile.")


def _mirror_x(x: float, width: int) -> float:
    mirrored = (width - 1) - float(x)
    if mirrored < 0:
        return 0.0
    max_x = float(width - 1)
    if mirrored > max_x:
        return max_x
    return mirrored


def _mirror_y(y: float, height: int) -> float:
    mirrored = (height - 1) - float(y)
    if mirrored < 0:
        return 0.0
    max_y = float(height - 1)
    if mirrored > max_y:
        return max_y
    return mirrored


def mirror_landmarks_512(landmarks_512: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": lm["id"],
            "x": _mirror_x(float(lm["x"]), STANDARD_SIZE),
            "y": float(lm["y"]),
        }
        for lm in landmarks_512
    ]


def base_standardize(
    image: np.ndarray,
    xyxy: Sequence[float],
    pad_ratio: float = 0.20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Crop with padding and resize to STANDARD_SIZE x STANDARD_SIZE.

    Returns:
        (crop_512, metadata)
    """
    img_h, img_w = image.shape[:2]
    if img_h <= 0 or img_w <= 0:
        raise ValueError("Invalid image dimensions for standardization.")

    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[:4]]
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    cx1 = max(0, int(x1 - bw * pad_ratio))
    cy1 = max(0, int(y1 - bh * pad_ratio))
    cx2 = min(img_w, int(x2 + bw * pad_ratio))
    cy2 = min(img_h, int(y2 + bh * pad_ratio))

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = image[y1:y2, x1:x2]
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2
    if crop.size == 0:
        crop = image
        cx1, cy1, cx2, cy2 = 0, 0, img_w, img_h

    crop_h, crop_w = crop.shape[:2]
    crop_w = max(1, int(crop_w))
    crop_h = max(1, int(crop_h))

    crop_512, letterbox_meta = _letterbox_to_square(
        crop,
        target_size=STANDARD_SIZE,
        fill_color=_median_perimeter_color(crop),
    )

    meta = {
        "center":      [int((x1 + x2) / 2), int((y1 + y2) / 2)],
        "crop_origin": [int(cx1), int(cy1)],
        "crop_size":   [int(crop_w), int(crop_h)],
        "rotation":    0.0,
        "scale_x":     float(letterbox_meta["scale"]),
        "scale_y":     float(letterbox_meta["scale"]),
        "scale":       float(letterbox_meta["scale"]),
        "pad_left":    float(letterbox_meta["pad_left"]),
        "pad_top":     float(letterbox_meta["pad_top"]),
        "pad_right":   float(letterbox_meta["pad_right"]),
        "pad_bottom":  float(letterbox_meta["pad_bottom"]),
        "fill_color":  _median_perimeter_color(crop),
        "fill_strategy": "median_perimeter",
        "padding_coordinate_space": "standardized",
        "obb_deskewed": False,
    }
    return crop_512, meta


def _get_standardized_padding(
    metadata: Mapping[str, Any],
    *,
    sx: float | None = None,
    sy: float | None = None,
) -> tuple[float, float, float, float]:
    scale_x = float(sx if sx is not None else metadata.get("scale_x", metadata.get("scale", 1.0))) or 1.0
    scale_y = float(sy if sy is not None else metadata.get("scale_y", metadata.get("scale", 1.0))) or 1.0
    pad_left = float(metadata.get("pad_left", 0.0))
    pad_top = float(metadata.get("pad_top", 0.0))
    pad_right = float(metadata.get("pad_right", 0.0))
    pad_bottom = float(metadata.get("pad_bottom", 0.0))

    coordinate_space = str(metadata.get("padding_coordinate_space", "")).strip().lower()
    if coordinate_space in ("", "legacy"):
        coordinate_space = "standardized" if bool(metadata.get("obb_deskewed")) else "crop"

    if coordinate_space == "crop":
        pad_left *= scale_x
        pad_top *= scale_y
        pad_right *= scale_x
        pad_bottom *= scale_y

    return pad_left, pad_top, pad_right, pad_bottom


def remap_landmarks_to_standard(
    landmarks: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    mirror: bool = False,
) -> list[dict[str, Any]]:
    """Map image-space landmarks to STANDARD_SIZE crop space."""
    ox, oy = metadata["crop_origin"]
    sx = float(metadata.get("scale_x", metadata.get("scale", 1.0)))
    sy = float(metadata.get("scale_y", metadata.get("scale", 1.0)))
    pad_left, pad_top, _pad_right, _pad_bottom = _get_standardized_padding(
        metadata,
        sx=sx,
        sy=sy,
    )

    remapped: list[dict[str, Any]] = []
    for lm in landmarks:
        x = ((float(lm["x"]) - float(ox)) * sx) + pad_left
        y = ((float(lm["y"]) - float(oy)) * sy) + pad_top
        if mirror:
            x = _mirror_x(x, STANDARD_SIZE)
        # Keep standardized coordinates within the 512x512 crop frame.
        x = max(0.0, min(float(STANDARD_SIZE - 1), float(x)))
        y = max(0.0, min(float(STANDARD_SIZE - 1), float(y)))
        remapped.append({**lm, "x": x, "y": y})
    return remapped


def _pca_angle(points_rc: np.ndarray) -> float:
    """
    Compute principal-axis angle from (row, col) points.
    Returns angle in degrees in crop coordinate frame.
    """
    try:
        from sklearn.decomposition import PCA as _PCA

        pca = _PCA(n_components=2)
        pca.fit(points_rc)
        return float(np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0])))
    except Exception:
        pass

    pts = points_rc[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
    rect = cv2.minAreaRect(pts)
    angle = float(rect[2])
    return angle + 90.0 if angle < -45.0 else angle


def pca_rotate_crop(
    crop_512: np.ndarray,
    mask_512: np.ndarray,
    min_coverage: float = 0.05,
    min_points: int = 10,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """
    Rotate a STANDARD_SIZE crop using PCA axis estimated from a STANDARD_SIZE mask.
    """
    if mask_512 is None:
        return crop_512, 0.0, {"applied": False, "reason": "no_mask"}

    mask = (mask_512 > 0).astype(np.uint8)
    points = np.column_stack(np.where(mask > 0))
    coverage = float(np.count_nonzero(mask)) / float(STANDARD_SIZE * STANDARD_SIZE)

    if coverage < float(min_coverage):
        return crop_512, 0.0, {"applied": False, "reason": "low_coverage", "coverage": coverage}
    if len(points) < int(min_points):
        return crop_512, 0.0, {"applied": False, "reason": "too_few_points", "points": int(len(points))}

    angle = _pca_angle(points)
    center = (STANDARD_SIZE / 2.0, STANDARD_SIZE / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        crop_512,
        M,
        (STANDARD_SIZE, STANDARD_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, float(angle), {"applied": True, "coverage": coverage}


def _rotate_mask_512(mask_512: np.ndarray, angle_deg: float) -> np.ndarray:
    center = (STANDARD_SIZE / 2.0, STANDARD_SIZE / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(
        (mask_512 > 0).astype(np.uint8),
        M,
        (STANDARD_SIZE, STANDARD_SIZE),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (rotated > 0).astype(np.uint8)




def canonicalize_with_mask(
    crop_512: np.ndarray,
    mask_512: np.ndarray | None,
    *,
    policy: Mapping[str, Any] | None = None,
    pca_mode: str = "auto",
    orientation_hint: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """
    Canonicalize a standardized crop using SAM2 mask geometry.

    Steps:
      1) Optional PCA leveling (mode-dependent).
      2) Optional canonical left/right flip for directional schemas.
    """
    policy_obj = dict(policy or {})
    mode = _safe_orientation_mode(policy_obj.get("mode"))
    target_orientation = _safe_orientation_target(policy_obj.get("targetOrientation", "left"))

    req_pca_mode = str(pca_mode or "auto").strip().lower()
    if req_pca_mode not in ("off", "on", "auto"):
        req_pca_mode = "auto"
    policy_pca_mode = str(policy_obj.get("pcaLevelingMode", "off")).strip().lower()
    if policy_pca_mode not in ("off", "on", "auto"):
        policy_pca_mode = "off"
    eff_pca_mode = policy_pca_mode if req_pca_mode == "auto" else req_pca_mode

    can_use_pca = bool(mask_512 is not None) and eff_pca_mode in ("on", "auto") and mode != "invariant"

    out_img = crop_512
    out_mask = (mask_512 > 0).astype(np.uint8) if mask_512 is not None else None
    pca_meta: dict[str, Any] = {"applied": False, "reason": "disabled"}
    angle = 0.0

    if can_use_pca and out_mask is not None:
        out_img, angle, pca_meta = pca_rotate_crop(out_img, out_mask)
        if pca_meta.get("applied"):
            out_mask = _rotate_mask_512(out_mask, angle)

    direction_resolved = resolve_direction_with_fallback(
        out_mask,
        orientation_hint=orientation_hint,
        template_direction=target_orientation if mode == "directional" else None,
        direction_priority=policy_obj.get("directionPriority", "auto"),
        template_fallback=bool(policy_obj.get("templateDirectionFallback", False)),
    )
    inferred_direction = direction_resolved.get("direction")
    inferred_conf = float(direction_resolved.get("confidence", 0.0))
    direction_source = str(direction_resolved.get("source", "none"))

    canonical_flip_applied = False
    if mode == "directional" and inferred_direction in ("left", "right"):
        if inferred_direction != target_orientation:
            out_img = cv2.flip(out_img, 1)
            if out_mask is not None:
                out_mask = cv2.flip(out_mask, 1)
            canonical_flip_applied = True

    return out_img, out_mask, {
        "mode": mode,
        "target_orientation": target_orientation,
        "requested_pca_mode": req_pca_mode,
        "effective_pca_mode": eff_pca_mode,
        "pca_applied": bool(pca_meta.get("applied")),
        "pca_rotation": float(angle),
        "pca_meta": pca_meta,
        "direction_source": direction_source,
        "inferred_direction": inferred_direction,
        "inferred_direction_confidence": float(inferred_conf),
        "direction_confidence": float(inferred_conf),
        "direction_debug": direction_resolved.get("debug"),
        "canonical_flip_applied": bool(canonical_flip_applied),
    }


def resolve_direction_with_fallback(
    mask_512: np.ndarray | None,
    *,
    orientation_hint: str | None = None,
    template_direction: str | None = None,
    direction_priority: str = "auto",
    template_fallback: bool = False,
) -> dict[str, Any]:
    """
    Resolve left/right direction using upstream detector orientation metadata.

      1) Detector hint from YOLO (`orientation_hint`)
      2) Optional template fallback (disabled by default)

    Returns:
      {
        "direction": "left"|"right"|None,
        "confidence": float,
        "source": str,
        "debug": dict,
      }
    """
    _ = _safe_direction_priority(direction_priority)
    hint = _safe_direction(orientation_hint)
    template = _safe_direction(template_direction)
    direction = None
    confidence = 0.0
    source = "none"
    if hint in ("left", "right"):
        direction = hint
        confidence = 1.0
        source = "detector_hint"
    elif template_fallback and template in ("left", "right"):
        direction = template
        confidence = 0.0
        source = "template_prior"

    return {
        "direction": direction,
        "confidence": float(confidence),
        "source": source,
        "debug": {
            "hint_direction": hint,
            "template_direction": template if template_fallback else None,
            "mask_available": bool(mask_512 is not None),
        },
    }


def should_lock_orientation_from_canonicalization(
    canonical_meta: Mapping[str, Any] | None,
    *,
    policy: Mapping[str, Any] | None = None,
) -> bool:
    """
    Determine whether canonical orientation is reliable enough to skip
    two-candidate flip selection and avoid double-flip failures.
    """
    if not isinstance(canonical_meta, Mapping):
        return False
    mode = _safe_orientation_mode((policy or {}).get("mode", "invariant"))
    if mode != "directional":
        return False
    source = str(canonical_meta.get("direction_source", "")).strip().lower()
    canonical_source = str(canonical_meta.get("source", "")).strip().lower()
    return source == "detector_hint" or canonical_source == "obb_geometry"


def detect_orientation(
    landmarks: Sequence[Mapping[str, Any]],
    head_id: int | None = None,
    tail_id: int | None = None,
) -> str | None:
    valid = [
        lm
        for lm in landmarks
        if float(lm.get("x", -1)) >= 0 and float(lm.get("y", -1)) >= 0
    ]
    if len(valid) < 2:
        return None

    head = None
    if head_id is not None:
        head = next((lm for lm in valid if int(lm.get("id", -99999)) == int(head_id)), None)
    if head is None:
        head = min(valid, key=lambda lm: int(lm.get("id", 0)))

    # Prefer explicit head-tail axis when available.
    if tail_id is not None:
        tail = next((lm for lm in valid if int(lm.get("id", -99999)) == int(tail_id)), None)
        if tail is not None and int(tail.get("id", -1)) != int(head.get("id", -2)):
            return "left" if float(head["x"]) < float(tail["x"]) else "right"

    others = [lm for lm in valid if int(lm.get("id", -99999)) != int(head.get("id", -99998))]
    if not others:
        return None
    centroid_x = sum(float(lm["x"]) for lm in others) / float(len(others))
    return "left" if float(head["x"]) < centroid_x else "right"


def obb_corners_to_aabb(
    obb_corners: Sequence[Sequence[float]],
    *,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, float]:
    pts = np.array(obb_corners, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        raise ValueError("obb_corners_to_aabb expects 4 OBB corners")

    left = float(np.min(pts[:, 0]))
    top = float(np.min(pts[:, 1]))
    right = float(np.max(pts[:, 0]))
    bottom = float(np.max(pts[:, 1]))

    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        left = max(0.0, min(float(img_w - 1), left))
        top = max(0.0, min(float(img_h - 1), top))
        right = max(left + 1.0, min(float(img_w), right))
        bottom = max(top + 1.0, min(float(img_h), bottom))

    width = right - left
    height = bottom - top
    if width <= 1.0 or height <= 1.0:
        raise ValueError("derived AABB envelope is too small")

    return {
        "left": float(left),
        "top": float(top),
        "right": float(right),
        "bottom": float(bottom),
        "width": float(width),
        "height": float(height),
    }


def _normalize_angle_360(angle_deg: float) -> float:
    angle = float(angle_deg) % 360.0
    if angle < 0.0:
        angle += 360.0
    return angle


def _normalize_angle_signed(angle_deg: float) -> float:
    angle = _normalize_angle_360(angle_deg)
    if angle > 180.0:
        angle -= 360.0
    if angle <= -180.0:
        angle += 360.0
    if abs(angle + 180.0) <= 1e-6:
        return 180.0
    return angle


def _angular_distance_deg(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def _snap_import_angle(angle_deg: float, *, mode: str, tolerance_deg: float = 5.0) -> float:
    resolved_mode = _safe_orientation_mode(mode)
    if resolved_mode == "invariant":
        return 0.0

    angle_360 = _normalize_angle_360(angle_deg)
    targets: list[float]
    if resolved_mode == "directional":
        targets = [0.0, 180.0]
    elif resolved_mode == "bilateral":
        targets = [90.0, 270.0]
    elif resolved_mode == "axial":
        targets = [0.0, 90.0, 180.0, 270.0]
    else:
        targets = []

    best_target = None
    best_distance = None
    for target in targets:
        distance = _angular_distance_deg(angle_360, target)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_target = target

    if best_target is not None and best_distance is not None and best_distance <= float(tolerance_deg):
        return _normalize_angle_signed(best_target)
    return _normalize_angle_signed(angle_360)


def _centroid(points: Sequence[Mapping[str, float]]) -> tuple[float, float]:
    if not points:
        raise ValueError("cannot compute centroid of empty point set")
    cx = sum(float(p["x"]) for p in points) / float(len(points))
    cy = sum(float(p["y"]) for p in points) / float(len(points))
    return float(cx), float(cy)


def _bi_centroid_axis(
    valid: Sequence[Mapping[str, float | int]],
    *,
    axis: str,
) -> tuple[float, float, float]:
    if len(valid) < 2:
        raise ValueError("need at least two landmarks for bi-centroid axis")
    key = "x" if axis == "horizontal" else "y"
    sorted_points = sorted(valid, key=lambda lm: float(lm[key]))
    bucket_size = max(1, int(math.ceil(len(sorted_points) * 0.25)))
    start_bucket = sorted_points[:bucket_size]
    end_bucket = sorted_points[-bucket_size:]
    start_x, start_y = _centroid(start_bucket)
    end_x, end_y = _centroid(end_bucket)
    dx = float(end_x - start_x)
    dy = float(end_y - start_y)
    norm = float(math.hypot(dx, dy))
    if norm <= 1e-6:
        raise ValueError("bi-centroid buckets collapse to the same centroid")
    return dx / norm, dy / norm, norm


def _resolve_landmark_axis(
    valid: Sequence[Mapping[str, float | int]],
    *,
    mode: str,
    head: Mapping[str, float | int] | None = None,
    tail: Mapping[str, float | int] | None = None,
) -> tuple[float, float, str]:
    resolved_mode = _safe_orientation_mode(mode)
    if resolved_mode == "invariant":
        return 1.0, 0.0, "invariant_axis_aligned"

    if head is not None and tail is not None:
        dx = float(tail["x"]) - float(head["x"])
        dy = float(tail["y"]) - float(head["y"])
        norm = float(math.hypot(dx, dy))
        if norm > 1e-6:
            return dx / norm, dy / norm, "head_tail"

    if resolved_mode == "directional":
        try:
            ux, uy, _ = _bi_centroid_axis(valid, axis="horizontal")
            return ux, uy, "directional_bi_centroid"
        except ValueError:
            return 1.0, 0.0, "fallback"
    if resolved_mode == "bilateral":
        try:
            ux, uy, _ = _bi_centroid_axis(valid, axis="vertical")
            return ux, uy, "bilateral_bi_centroid"
        except ValueError:
            return 1.0, 0.0, "fallback"
    if resolved_mode == "axial":
        candidates: list[tuple[float, float, float]] = []
        for axis in ("horizontal", "vertical"):
            try:
                candidates.append(_bi_centroid_axis(valid, axis=axis))
            except ValueError:
                continue
        if candidates:
            horiz_or_best = max(candidates, key=lambda item: item[2])
            return horiz_or_best[0], horiz_or_best[1], "axial_axis_snapped"
        return 1.0, 0.0, "fallback"
    return 1.0, 0.0, "fallback"


def derive_obb_from_landmarks(
    landmarks: Sequence[Mapping[str, Any]],
    *,
    image_shape: tuple[int, int] | None = None,
    head_id: int | None = None,
    tail_id: int | None = None,
    mode: str = "invariant",
    pad_ratio: float = 0.10,
    min_pad_px: float = 4.0,
) -> dict[str, Any]:
    valid: list[dict[str, float | int]] = []
    for lm in landmarks:
        try:
            if bool(lm.get("isSkipped")):
                continue
            x = float(lm.get("x", -1))
            y = float(lm.get("y", -1))
            lid = int(lm.get("id", -1))
        except Exception:
            continue
        if x < 0 or y < 0:
            continue
        valid.append({"id": lid, "x": x, "y": y})

    if len(valid) == 0:
        raise ValueError("cannot derive OBB from landmarks: no valid non-skipped landmarks")

    unique_points = {(round(float(lm["x"]), 3), round(float(lm["y"]), 3)) for lm in valid}
    if len(unique_points) < 2:
        x = float(valid[0]["x"])
        y = float(valid[0]["y"])
        width = max(2.0, float(min_pad_px) * 2.0)
        height = max(2.0, float(min_pad_px) * 2.0)
        corners = np.array(
            [
                [x - width / 2.0, y - height / 2.0],
                [x + width / 2.0, y - height / 2.0],
                [x + width / 2.0, y + height / 2.0],
                [x - width / 2.0, y + height / 2.0],
            ],
            dtype=np.float32,
        )
        if image_shape is not None:
            img_h, img_w = image_shape[:2]
            corners[:, 0] = np.clip(corners[:, 0], 0.0, float(img_w - 1))
            corners[:, 1] = np.clip(corners[:, 1], 0.0, float(img_h - 1))
        corners_list = corners.tolist()
        return {
            **obb_corners_to_aabb(corners_list, image_shape=image_shape),
            "obbCorners": corners_list,
            "angle": 0.0,
            "derivation": "fallback",
        }

    head = None
    tail = None
    if head_id is not None:
        head = next((lm for lm in valid if int(lm["id"]) == int(head_id)), None)
    if tail_id is not None:
        tail = next((lm for lm in valid if int(lm["id"]) == int(tail_id)), None)

    resolved_mode = _safe_orientation_mode(mode)
    ux, uy, derivation = _resolve_landmark_axis(
        valid,
        mode=resolved_mode,
        head=head,
        tail=tail,
    )

    vx = -uy
    vy = ux
    u_values = [float(lm["x"]) * ux + float(lm["y"]) * uy for lm in valid]
    v_values = [float(lm["x"]) * vx + float(lm["y"]) * vy for lm in valid]
    min_u = min(u_values)
    max_u = max(u_values)
    min_v = min(v_values)
    max_v = max(v_values)

    raw_width = max(2.0, max_u - min_u)
    raw_height = max(2.0, max_v - min_v)
    pad_u = max(float(min_pad_px), raw_width * float(pad_ratio))
    pad_v = max(float(min_pad_px), raw_height * float(pad_ratio))
    center_u = (min_u + max_u) / 2.0
    center_v = (min_v + max_v) / 2.0
    center_x = center_u * ux + center_v * vx
    center_y = center_u * uy + center_v * vy
    width = raw_width + pad_u * 2.0
    height = raw_height + pad_v * 2.0
    angle = float(math.degrees(math.atan2(uy, ux)))
    angle = _snap_import_angle(angle, mode=resolved_mode)

    r = math.radians(angle)
    cos_a = math.cos(r)
    sin_a = math.sin(r)
    hw = width / 2.0
    hh = height / 2.0
    corners = np.array(
        [
            [center_x + cos_a * (-hw) - sin_a * (-hh), center_y + sin_a * (-hw) + cos_a * (-hh)],
            [center_x + cos_a * ( hw) - sin_a * (-hh), center_y + sin_a * ( hw) + cos_a * (-hh)],
            [center_x + cos_a * ( hw) - sin_a * ( hh), center_y + sin_a * ( hw) + cos_a * ( hh)],
            [center_x + cos_a * (-hw) - sin_a * ( hh), center_y + sin_a * (-hw) + cos_a * ( hh)],
        ],
        dtype=np.float32,
    )
    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        corners[:, 0] = np.clip(corners[:, 0], 0.0, float(img_w - 1))
        corners[:, 1] = np.clip(corners[:, 1], 0.0, float(img_h - 1))

    corners_list = corners.tolist()
    return {
        **obb_corners_to_aabb(corners_list, image_shape=image_shape),
        "obbCorners": corners_list,
        "angle": angle,
        "derivation": derivation,
    }


def derive_class_id_from_landmarks(
    landmarks: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    head_id: int | None = None,
    tail_id: int | None = None,
) -> int:
    resolved_mode = _safe_orientation_mode(mode)
    if resolved_mode == "invariant":
        return 0

    valid = [
        lm for lm in landmarks
        if not bool(lm.get("isSkipped"))
        and float(lm.get("x", -1)) >= 0
        and float(lm.get("y", -1)) >= 0
    ]
    if len(valid) < 2:
        return 0

    head = None
    tail = None
    if head_id is not None:
        head = next((lm for lm in valid if int(lm.get("id", -99999)) == int(head_id)), None)
    if tail_id is not None:
        tail = next((lm for lm in valid if int(lm.get("id", -99999)) == int(tail_id)), None)
    if head is None or tail is None:
        return 0

    if resolved_mode == "axial":
        return 0 if float(head["y"]) < float(tail["y"]) else 1

    return 0 if float(head["x"]) < float(tail["x"]) else 1


def resolve_orientation_hint(
    orientation_hint: Any,
    *,
    box_xyxy: Sequence[float] | None = None,
    min_confidence: float = 0.25,
    min_dx_ratio: float = 0.05,
) -> str | None:
    """
    Resolve a reliable left/right hint from detector metadata.

    Accepts:
      - "left" | "right"
      - {"orientation": "...", "head_point": [x,y], "tail_point": [x,y], "confidence": ...}

    Returns None when the hint is ambiguous or low-confidence.
    """
    if isinstance(orientation_hint, str):
        v = orientation_hint.strip().lower()
        return v if v in ("left", "right") else None

    if not isinstance(orientation_hint, Mapping):
        return None

    raw_orientation = str(orientation_hint.get("orientation", "")).strip().lower()
    if raw_orientation not in ("left", "right"):
        raw_orientation = ""

    # Optional confidence gate.
    conf = orientation_hint.get("confidence")
    try:
        if conf is not None and float(conf) < float(min_confidence):
            return None
    except Exception:
        pass

    head = orientation_hint.get("head_point")
    tail = orientation_hint.get("tail_point")
    if (
        isinstance(head, (list, tuple))
        and isinstance(tail, (list, tuple))
        and len(head) >= 2
        and len(tail) >= 2
    ):
        try:
            hx, tx = float(head[0]), float(tail[0])
            dx = hx - tx

            box_w = None
            if box_xyxy is not None and len(box_xyxy) >= 4:
                try:
                    box_w = abs(float(box_xyxy[2]) - float(box_xyxy[0]))
                except Exception:
                    box_w = None
            if not box_w or box_w <= 1e-6:
                box_w = float(STANDARD_SIZE)

            if abs(dx) / box_w < float(min_dx_ratio):
                return None

            inferred = "left" if hx < tx else "right"
            if raw_orientation and raw_orientation != inferred:
                return None
            return inferred
        except Exception:
            pass

    return raw_orientation or None


def score_landmarks_against_template(
    landmarks_512: Sequence[Mapping[str, Any]],
    landmark_template: Mapping[int | str, Mapping[str, float]] | None,
) -> float | None:
    if not landmark_template:
        return None

    total = 0.0
    count = 0
    for lm in landmarks_512:
        lm_id = int(lm["id"])
        t = landmark_template.get(lm_id)
        if t is None:
            t = landmark_template.get(str(lm_id))
        if not t:
            continue
        sx = max(float(t.get("x_std", 0.0)), 1.0)
        sy = max(float(t.get("y_std", 0.0)), 1.0)
        dx = (float(lm["x"]) - float(t.get("x_mean", 0.0))) / sx
        dy = (float(lm["y"]) - float(t.get("y_mean", 0.0))) / sy
        total += math.sqrt(dx * dx + dy * dy)
        count += 1

    if count == 0:
        return None
    return total / count


def select_orientation(
    crop_512: np.ndarray,
    predict_fn: Callable[[np.ndarray], list[dict[str, Any]] | None],
    *,
    target_orientation: str | None = None,
    landmark_template: Mapping[int | str, Mapping[str, float]] | None = None,
    head_id: int | None = None,
    tail_id: int | None = None,
    orientation_hint_original: str | None = None,
    template_margin: float = 0.5,
    template_gain_ratio: float = 0.20,
    primary_bad_score: float = 1.8,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """
    Evaluate normal/flipped candidates and choose orientation with the same policy
    used in predict.py.
    """
    lms_a = predict_fn(crop_512) or []
    if not lms_a:
        return [], False, {
            "used_flipped_crop": False,
            "selection_reason": "no_primary_prediction",
            "candidate_b_evaluated": False,
            "head_landmark_id": head_id,
            "tail_landmark_id": tail_id,
        }

    score_a = score_landmarks_against_template(lms_a, landmark_template)
    ori_a = detect_orientation(lms_a, head_id=head_id, tail_id=tail_id)
    need_flipped = bool(target_orientation or landmark_template or orientation_hint_original)
    if not need_flipped:
        return lms_a, False, {
            "candidate_a_orientation": ori_a,
            "candidate_b_orientation": None,
            "candidate_b_unflipped_orientation": None,
            "candidate_a_template_score": score_a,
            "candidate_b_template_score": None,
            "used_flipped_crop": False,
            "selection_reason": "primary_only_no_orientation_or_template",
            "candidate_b_evaluated": False,
            "head_landmark_id": head_id,
            "tail_landmark_id": tail_id,
            "target_orientation": target_orientation,
        }

    lms_b = predict_fn(cv2.flip(crop_512, 1)) or []
    if not lms_b:
        return lms_a, False, {
            "candidate_a_orientation": ori_a,
            "candidate_b_orientation": None,
            "candidate_b_unflipped_orientation": None,
            "candidate_a_template_score": score_a,
            "candidate_b_template_score": None,
            "used_flipped_crop": False,
            "selection_reason": "no_flipped_prediction_keep_primary",
            "candidate_b_evaluated": True,
            "head_landmark_id": head_id,
            "tail_landmark_id": tail_id,
            "target_orientation": target_orientation,
        }

    score_b = score_landmarks_against_template(lms_b, landmark_template)
    ori_b = detect_orientation(lms_b, head_id=head_id, tail_id=tail_id)
    ori_b_unflipped = detect_orientation(mirror_landmarks_512(lms_b), head_id=head_id, tail_id=tail_id)

    use_flipped = False
    selection_reason = "keep_primary"
    template_gain = None
    template_gain_rel = None

    # Optional cue in original-image orientation space.
    if orientation_hint_original:
        match_hint_a = ori_a == orientation_hint_original
        match_hint_b = ori_b_unflipped == orientation_hint_original
        if match_hint_a != match_hint_b:
            use_flipped = match_hint_b
            selection_reason = "flipped_matches_orientation_hint" if use_flipped else "primary_matches_orientation_hint"

    match_a_canonical = bool(target_orientation and ori_a == target_orientation)
    match_b_canonical = bool(target_orientation and ori_b == target_orientation)
    if selection_reason == "keep_primary":
        # Canonical-orientation check in each candidate's native prediction space.
        # This helps when training was normalized to a target orientation (e.g., left):
        # a right-facing fish often yields better geometry on the flipped crop.
        if target_orientation:
            if match_a_canonical != match_b_canonical:
                use_flipped = match_b_canonical
                selection_reason = (
                    "flipped_matches_target_canonical_space"
                    if use_flipped
                    else "primary_matches_target_canonical_space"
                )

    if selection_reason == "keep_primary":
        # Target orientation is defined in canonical (training) space, not original-image space.
        # Original-space orientation should only come from orientation_hint_original.
        if score_a is not None and score_b is not None:
            template_gain = score_a - score_b
            template_gain_rel = template_gain / max(score_a, 1e-6)
            dual_canonical = bool(target_orientation and match_a_canonical and match_b_canonical)
            original_disagree = bool(
                ori_a is not None and ori_b_unflipped is not None and ori_a != ori_b_unflipped
            )
            if (
                dual_canonical
                and original_disagree
                and template_gain > template_margin
                and template_gain_rel > template_gain_ratio
            ):
                use_flipped = True
                selection_reason = "flipped_better_template_dual_canonical"
            else:
                strong_gain = (
                    template_gain > template_margin
                    and template_gain_rel > template_gain_ratio
                    and ((not match_a_canonical) or (score_a >= primary_bad_score))
                )
                if strong_gain:
                    use_flipped = True
                    selection_reason = "flipped_strong_template_gain"
                else:
                    if template_gain > template_margin and match_a_canonical and score_a < primary_bad_score:
                        selection_reason = "primary_good_score_guard"
                    else:
                        selection_reason = "template_gain_not_strong_enough"
        else:
            selection_reason = "scores_unavailable_keep_primary"

    chosen = lms_b if use_flipped else lms_a
    return chosen, use_flipped, {
        "candidate_a_orientation": ori_a,
        "candidate_b_orientation": ori_b,
        "candidate_b_unflipped_orientation": ori_b_unflipped,
        "candidate_a_template_score": score_a,
        "candidate_b_template_score": score_b,
        "template_gain": template_gain,
        "template_gain_relative": template_gain_rel,
        "match_a_canonical": match_a_canonical,
        "match_b_canonical": match_b_canonical,
        "dual_canonical": bool(target_orientation and match_a_canonical and match_b_canonical),
        "used_flipped_crop": use_flipped,
        "selection_reason": selection_reason,
        "head_landmark_id": head_id,
        "tail_landmark_id": tail_id,
        "target_orientation": target_orientation,
        "template_margin": template_margin,
        "template_gain_ratio_threshold": template_gain_ratio,
        "primary_bad_score_threshold": primary_bad_score,
        "candidate_b_evaluated": True,
    }


def map_to_original(
    landmarks_512: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    was_flipped: bool = False,
    image_scale: float = 1.0,
    image_shape: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Map STANDARD_SIZE landmarks back to original image coordinates.

    For OBB-deskewed crops (metadata["obb_deskewed"] == True and affine_M present):
      Transform order (inverse):
        1) un-flip in STANDARD_SIZE space (if needed)
        2) un-resize to crop space
        3) offset to rotated-image space (add crop_origin)
        4) apply exact inverse affine (cv2.invertAffineTransform) to un-rotate
        5) un-scale to original image size
        6) optional bounds clamp
      This uses the stored affine_M to invert the exact warpAffine applied during
      crop extraction, correcting for crops where the OBB center was clamped near
      the image border (center ≠ 256 in 512-space).

    For non-OBB crops (affine_M absent or obb_deskewed=False):
      Transform order (inverse):
        1) un-flip (if needed)
        2) un-rotate in STANDARD_SIZE space around center (256, 256)
        3) un-resize to crop space
        4) offset by crop origin
        5) un-scale to original image size
        6) optional bounds clamp
    """
    sx = float(metadata.get("scale_x", metadata.get("scale", 1.0))) or 1.0
    sy = float(metadata.get("scale_y", metadata.get("scale", 1.0))) or 1.0
    ox, oy = metadata["crop_origin"]
    angle = float(metadata.get("rotation", 0.0))
    pad_left, pad_top, _pad_right, _pad_bottom = _get_standardized_padding(
        metadata,
        sx=sx,
        sy=sy,
    )

    img_h = image_shape[0] if image_shape is not None else None
    img_w = image_shape[1] if image_shape is not None else None

    mapped: list[dict[str, Any]] = []
    canonical_flip_applied = bool(metadata.get("canonical_flip_applied", False))
    effective_flip = bool(was_flipped) ^ canonical_flip_applied
    rotated_180 = bool(metadata.get("rotated_180", False))

    # Use exact inverse affine when available (OBB-deskewed crops).
    affine_M_raw = metadata.get("affine_M")
    use_exact_affine = bool(metadata.get("obb_deskewed")) and affine_M_raw is not None

    if use_exact_affine:
        M = np.array(affine_M_raw, dtype=np.float64)
        M_inv = cv2.invertAffineTransform(M)  # exact inverse of the warpAffine used during extraction

        for lm in landmarks_512:
            x_s = max(0.0, min(float(STANDARD_SIZE - 1), float(lm["x"])))
            y_s = max(0.0, min(float(STANDARD_SIZE - 1), float(lm["y"])))

            if effective_flip:
                x_s = _mirror_x(x_s, STANDARD_SIZE)
            if rotated_180:
                x_s = _mirror_x(x_s, STANDARD_SIZE)
                y_s = _mirror_y(y_s, STANDARD_SIZE)

            # Un-resize: 512 → crop space
            x_crop = (x_s - pad_left) / sx
            y_crop = (y_s - pad_top) / sy

            # Offset to rotated-image space
            x_rot = x_crop + float(ox)
            y_rot = y_crop + float(oy)

            # Un-rotate using exact inverse affine
            x_orig = M_inv[0, 0] * x_rot + M_inv[0, 1] * y_rot + M_inv[0, 2]
            y_orig = M_inv[1, 0] * x_rot + M_inv[1, 1] * y_rot + M_inv[1, 2]

            if image_scale and image_scale != 1.0:
                x_orig = x_orig / float(image_scale)
                y_orig = y_orig / float(image_scale)

            if img_w is not None and img_h is not None:
                x_orig = max(0.0, min(float(img_w - 1), x_orig))
                y_orig = max(0.0, min(float(img_h - 1), y_orig))

            mapped.append({
                "id": int(lm["id"]),
                "x": round(x_orig, 1),
                "y": round(y_orig, 1),
            })
        return mapped

    # Fallback: center-based rotation in 512-space (non-OBB or legacy crops).
    angle_rad = np.radians(-angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    cx_s = STANDARD_SIZE / 2.0
    cy_s = STANDARD_SIZE / 2.0

    for lm in landmarks_512:
        x_s = max(0.0, min(float(STANDARD_SIZE - 1), float(lm["x"])))
        y_s = max(0.0, min(float(STANDARD_SIZE - 1), float(lm["y"])))

        if effective_flip:
            x_s = _mirror_x(x_s, STANDARD_SIZE)
        if rotated_180:
            x_s = _mirror_x(x_s, STANDARD_SIZE)
            y_s = _mirror_y(y_s, STANDARD_SIZE)

        if abs(angle) > 1e-9:
            x_u = cx_s + (x_s - cx_s) * cos_a - (y_s - cy_s) * sin_a
            y_u = cy_s + (x_s - cx_s) * sin_a + (y_s - cy_s) * cos_a
        else:
            x_u, y_u = x_s, y_s

        x_crop = (x_u - pad_left) / sx
        y_crop = (y_u - pad_top) / sy
        x_orig = x_crop + float(ox)
        y_orig = y_crop + float(oy)

        if image_scale and image_scale != 1.0:
            x_orig = x_orig / float(image_scale)
            y_orig = y_orig / float(image_scale)

        if img_w is not None and img_h is not None:
            x_orig = max(0.0, min(float(img_w - 1), x_orig))
            y_orig = max(0.0, min(float(img_h - 1), y_orig))

        mapped.append({
            "id": int(lm["id"]),
            "x": round(x_orig, 1),
            "y": round(y_orig, 1),
        })
    return mapped


# ===========================================================================
# OBB Universal Geometry Engine
# ===========================================================================

def _median_perimeter_color(image: np.ndarray) -> list[int]:
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        return [0, 0, 0]
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return [0, 0, 0]
    perimeter = np.concatenate(
        [
            image[0, :],
            image[h - 1, :],
            image[:, 0],
            image[:, w - 1],
        ],
        axis=0,
    )
    if perimeter.size == 0:
        return [0, 0, 0]
    return [int(v) for v in np.median(perimeter, axis=0)]


def _letterbox_to_square(
    crop: np.ndarray,
    *,
    target_size: int,
    fill_color: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    if crop is None or crop.size == 0:
        raise ValueError("cannot letterbox an empty crop")

    crop_h, crop_w = crop.shape[:2]
    crop_h = max(1, int(crop_h))
    crop_w = max(1, int(crop_w))
    longest = max(crop_w, crop_h)
    scale = float(target_size) / float(longest)

    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))
    resized = cv2.resize(
        crop,
        (resized_w, resized_h),
        interpolation=STANDARDIZED_OBB_INTERPOLATION,
    )

    pad_left = (target_size - resized_w) // 2
    pad_top = (target_size - resized_h) // 2
    pad_right = target_size - resized_w - pad_left
    pad_bottom = target_size - resized_h - pad_top
    canvas = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=fill_color,
    )
    return canvas, {
        "scale": float(scale),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
        "resized_width": int(resized_w),
        "resized_height": int(resized_h),
    }

def extract_obb_crop(
    image: np.ndarray,
    obb_corners: list,
    pad_ratio: float = STANDARDIZED_OBB_PAD_RATIO,
    target_size: int = STANDARD_SIZE,
    apply_leveling: bool = True,
) -> tuple:
    """
    Extract a tight deskewed OBB crop, then letterbox it into a square canvas.

    Args:
        image: Full-resolution BGR image.
        obb_corners: 4-corner list [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in image pixels.
        pad_ratio: Fraction of each OBB side added as padding.
        target_size: Output crop size (default 512).
        apply_leveling: If True (default), rotate the image to deskew the OBB. If False,
            pixels are untouched (AABB-like crop), but affine_M is still stored so
            downstream landmark remapping can apply or skip the same transform.

    Returns:
        (crop_512, metadata) — crop_512 is (target_size, target_size, C);
        metadata is compatible with existing map_to_original().
        metadata["affine_M"] always contains the 2×3 rotation matrix as a nested list,
        and metadata["leveling_applied"] indicates whether warpAffine was executed.
    """
    pts = np.array(obb_corners, dtype=np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(pts)          # ((cx,cy), (w,h), angle_deg)
    center, (rw, rh), angle = rect
    # minAreaRect guarantees angle ∈ (−90°, 0°].  Adding 90° is correct only when
    # it moves the angle *closer* to 0° (i.e. |angle+90°| < |angle|), which
    # simplifies to angle < −45°.  The old rw<rh heuristic fails for near-square
    # OBBs where both sides are almost equal and the wrong side wins by a pixel.
    if angle < -45.0:    # long axis is steep; add 90° to bring it near-horizontal
        angle += 90.0

    img_h, img_w = image.shape[:2]
    cx_i, cy_i = int(round(center[0])), int(round(center[1]))

    # Build the rotation matrix (always computed; used for coord transforms even when
    # leveling is disabled so downstream landmark remapping stays consistent).
    M = cv2.getRotationMatrix2D((float(cx_i), float(cy_i)), angle, 1.0)

    if apply_leveling:
        # Rotate the entire image to deskew the OBB.  Use BORDER_CONSTANT (black)
        # instead of BORDER_REFLECT_101 so that areas swept outside the original
        # image boundary are filled with neutral zeros rather than mirrored fish
        # content, which would contaminate the crop with fake landmark targets.
        rotated = cv2.warpAffine(
            image, M, (img_w, img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        # Pixels untouched; M is still stored for coordinate transforms
        rotated = image

    transformed_pts = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
    min_x = float(np.min(transformed_pts[:, 0]))
    max_x = float(np.max(transformed_pts[:, 0]))
    min_y = float(np.min(transformed_pts[:, 1]))
    max_y = float(np.max(transformed_pts[:, 1]))
    pad_x = float(max(max_x - min_x, 1.0)) * float(max(0.0, pad_ratio))
    pad_y = float(max(max_y - min_y, 1.0)) * float(max(0.0, pad_ratio))

    # Crop the exact transformed OBB envelope plus a small biological margin.
    x1 = max(0, int(math.floor(min_x - pad_x)))
    y1 = max(0, int(math.floor(min_y - pad_y)))
    x2 = min(img_w, int(math.ceil(max_x + pad_x)))
    y2 = min(img_h, int(math.ceil(max_y + pad_y)))
    crop = rotated[y1:y2, x1:x2]

    ch, cw = crop.shape[:2]
    if crop.size == 0 or cw <= 0 or ch <= 0:
        raise ValueError("extract_obb_crop produced an empty crop")

    fill_color = _median_perimeter_color(crop)
    crop_512, letterbox_meta = _letterbox_to_square(
        crop,
        target_size=target_size,
        fill_color=fill_color,
    )

    meta = {
        "center": [cx_i, cy_i],
        "crop_origin": [x1, y1],
        "crop_size": [cw, ch],
        "rotation": float(angle),
        "scale_x": float(letterbox_meta["scale"]),
        "scale_y": float(letterbox_meta["scale"]),
        "scale": float(letterbox_meta["scale"]),
        "pad_left": int(letterbox_meta["pad_left"]),
        "pad_top": int(letterbox_meta["pad_top"]),
        "pad_right": int(letterbox_meta["pad_right"]),
        "pad_bottom": int(letterbox_meta["pad_bottom"]),
        "padding_coordinate_space": "standardized",
        "fill_color": list(fill_color),
        "fill_strategy": "median_perimeter",
        "obb_deskewed": True,
        "affine_M": M.tolist(),       # 2×3 matrix for downstream landmark remapping
        "leveling_applied": bool(apply_leveling),
    }
    return crop_512, meta


def extract_standardized_obb_crop(
    image: np.ndarray,
    obb_corners: list,
    *,
    apply_leveling: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Shared public OBB crop contract used by both training prep and inference.

    Geometry defaults are intentionally fixed here so the two paths cannot drift.
    """
    return extract_obb_crop(
        image,
        obb_corners,
        pad_ratio=STANDARDIZED_OBB_PAD_RATIO,
        target_size=STANDARD_SIZE,
        apply_leveling=apply_leveling,
    )


def project_point_to_segment(
    point: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, float]:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    abx = bx - ax
    aby = by - ay
    denom = (abx * abx) + (aby * aby)
    if denom <= 1e-12:
        return ax, ay
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    return ax + (abx * t), ay + (aby * t)


def point_in_convex_quad(
    point: Sequence[float],
    corners: Sequence[Sequence[float]],
    *,
    epsilon: float = 1e-6,
) -> bool:
    if len(corners) != 4:
        return False
    px, py = float(point[0]), float(point[1])
    sign = 0
    for idx in range(4):
        ax, ay = float(corners[idx][0]), float(corners[idx][1])
        bx, by = float(corners[(idx + 1) % 4][0]), float(corners[(idx + 1) % 4][1])
        cross = ((bx - ax) * (py - ay)) - ((by - ay) * (px - ax))
        if abs(cross) <= epsilon:
            continue
        current_sign = 1 if cross > 0 else -1
        if sign == 0:
            sign = current_sign
        elif current_sign != sign:
            return False
    return True


def project_point_to_obb_perimeter(
    point: Sequence[float],
    obb_corners: Sequence[Sequence[float]],
) -> tuple[float, float]:
    if len(obb_corners) != 4:
        return float(point[0]), float(point[1])

    best_point: tuple[float, float] | None = None
    best_dist_sq = float("inf")
    px, py = float(point[0]), float(point[1])
    for idx in range(4):
        projected = project_point_to_segment(
            point,
            obb_corners[idx],
            obb_corners[(idx + 1) % 4],
        )
        dx = projected[0] - px
        dy = projected[1] - py
        dist_sq = (dx * dx) + (dy * dy)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_point = projected

    return best_point if best_point is not None else (px, py)


def apply_obb_geometry(
    crop_512: np.ndarray,
    metadata: dict,
    class_id: int,
    orientation_policy: dict,
) -> tuple:
    """
    Schema-specific routing using OBB class_id.
    Replaces PCA canonicalization for all orientation modes.

    Args:
        crop_512: Deskewed 512×512 crop from extract_obb_crop().
        metadata: Crop metadata dict (modified in-place copy on flip).
        class_id: 0 = left-facing, 1 = right-facing (from OBB detector).
        orientation_policy: Session orientation policy dict.

    Returns:
        (crop_512, metadata, debug) — debug is a dict with flip info.
    """
    if not isinstance(orientation_policy, dict):
        return crop_512, metadata, {"mode": "invariant", "flip_applied": False}

    mode = str(orientation_policy.get("mode", "invariant")).lower()
    debug = {"mode": mode, "class_id": class_id, "flip_applied": False}
    metadata = {**metadata, "rotated_180": bool(metadata.get("rotated_180", False))}

    if mode == "directional":
        target = str(orientation_policy.get("targetOrientation", "left")).lower()
        detected = "left" if class_id == 0 else "right"
        if detected != target:
            crop_512 = cv2.flip(crop_512, 1)
            metadata = {**metadata, "canonical_flip_applied": True}
            debug["flip_applied"] = True
            debug["flip_reason"] = f"class_id={class_id} contradicts target={target}"

    elif mode == "bilateral":
        # Always normalize to left-facing (class_id=0); record flip for coord mirroring
        if class_id == 1:
            crop_512 = cv2.flip(crop_512, 1)
            metadata = {**metadata, "canonical_flip_applied": True, "bilateral_flip": True}
            debug["flip_applied"] = True
            debug["flip_reason"] = "bilateral normalization"

    elif mode == "axial":
        # If OBB angle is significantly off from horizontal, rotate 180°
        obb_angle = float(metadata.get("rotation", 0.0))
        if abs(obb_angle) > 45:
            crop_512 = cv2.rotate(crop_512, cv2.ROTATE_180)
            metadata = {**metadata, "rotated_180": True}
            debug["rotated_180"] = True

    # invariant: deskew was applied by extract_obb_crop; no flip needed

    return crop_512, metadata, debug
