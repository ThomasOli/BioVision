import math
import os
import json
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np


STANDARD_SIZE = 512
ORIENTATION_MODES = {"directional", "bilateral", "axial", "invariant"}


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

    out: dict[str, Any] = {
        "mode": mode,
        "pcaLevelingMode": pca_mode,
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


def should_mirror_to_target(
    original_orientation: str | None,
    *,
    policy: Mapping[str, Any] | None = None,
    mode: str | None = None,
    target_orientation: str | None = None,
) -> bool:
    original = str(original_orientation or "").strip().lower()
    if original not in ("left", "right"):
        return False

    resolved_mode = _safe_orientation_mode(mode or (policy or {}).get("mode"))
    if resolved_mode != "directional":
        return False

    target = _safe_orientation_target(
        target_orientation if target_orientation is not None else (policy or {}).get("targetOrientation")
    )
    return original != target


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


def get_schema_augmentation_profile(mode: str | None, *, engine: str) -> dict[str, Any]:
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
    """
    resolved_mode = _safe_orientation_mode(mode)
    eng = str(engine or "").strip().lower()

    if eng == "dlib":
        profiles = {
            # Directional schemas are canonicalized to a target facing; keep
            # augmentation conservative to protect mean-shape stability.
            "directional": {"angles": [-5.0, 5.0], "flip": False},
            "bilateral": {"angles": [-15.0, -7.0, 7.0, 15.0], "flip": True},
            "axial": {"angles": [180.0], "flip": True},
            "invariant": {"angles": [], "flip": False},
        }
        return dict(profiles.get(resolved_mode, profiles["invariant"]))

    if eng == "cnn":
        profiles = {
            # Keep left/right learning for directional schemas.
            "directional": {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.0,
                "rotation_range": (-20.0, 20.0),
                "rotate_180_prob": 0.0,
                "scale_range": (0.92, 1.08),
                "translate_ratio": 0.06,
            },
            # Bilateral schemas allow mirroring but require pair swaps.
            "bilateral": {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.0,
                "rotation_range": (-25.0, 25.0),
                "rotate_180_prob": 0.0,
                "scale_range": (0.90, 1.12),
                "translate_ratio": 0.08,
            },
            # Axial schemas are safe under 180-degree turns.
            "axial": {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.0,
                "rotation_range": (-20.0, 20.0),
                "rotate_180_prob": 0.5,
                "scale_range": (0.90, 1.12),
                "translate_ratio": 0.08,
            },
            # Invariant schemas rely on strong augmentation.
            "invariant": {
                "flip_prob": 0.5,
                "vertical_flip_prob": 0.25,
                "rotation_range": (-180.0, 180.0),
                "rotate_180_prob": 0.0,
                "scale_range": (0.85, 1.15),
                "translate_ratio": 0.10,
            },
        }
        return dict(profiles.get(resolved_mode, profiles["invariant"]))

    raise ValueError(f"Unsupported engine '{engine}' for schema augmentation profile.")


def get_box_jitter_profile(mode: str | None, *, engine: str) -> dict[str, Any]:
    """
    Return schema-specific box-jitter defaults.

    The jitter is intended to simulate detector crop variability.
    - dlib: conservative defaults to protect mean-shape stability.
    - cnn: stronger defaults for detector-noise robustness.
    """
    resolved_mode = _safe_orientation_mode(mode)
    eng = str(engine or "").strip().lower()

    if eng == "dlib":
        profiles = {
            "directional": {
                "enabled": True,
                "copies_per_sample": 1,
                "translate_ratio": 0.02,
                "scale_range": (0.98, 1.02),
            },
            "bilateral": {
                "enabled": True,
                "copies_per_sample": 2,
                "translate_ratio": 0.03,
                "scale_range": (0.96, 1.04),
            },
            "axial": {
                "enabled": True,
                "copies_per_sample": 2,
                "translate_ratio": 0.035,
                "scale_range": (0.95, 1.05),
            },
            "invariant": {
                "enabled": True,
                "copies_per_sample": 2,
                "translate_ratio": 0.04,
                "scale_range": (0.94, 1.06),
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

    crop_512 = cv2.resize(crop, (STANDARD_SIZE, STANDARD_SIZE), interpolation=cv2.INTER_LINEAR)
    sx = float(STANDARD_SIZE / crop_w)
    sy = float(STANDARD_SIZE / crop_h)

    meta = {
        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
        "crop_origin": [int(cx1), int(cy1)],
        "crop_size": [int(crop_w), int(crop_h)],
        "rotation": 0.0,
        "scale_x": sx,
        "scale_y": sy,
        "scale": sx,
    }
    return crop_512, meta


def remap_landmarks_to_standard(
    landmarks: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    mirror: bool = False,
) -> list[dict[str, Any]]:
    """Map image-space landmarks to STANDARD_SIZE crop space."""
    ox, oy = metadata["crop_origin"]
    sx = float(metadata.get("scale_x", metadata.get("scale", 1.0)))
    sy = float(metadata.get("scale_y", metadata.get("scale", 1.0)))

    remapped: list[dict[str, Any]] = []
    for lm in landmarks:
        x = (float(lm["x"]) - float(ox)) * sx
        y = (float(lm["y"]) - float(oy)) * sy
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


def estimate_direction_from_moments(
    mask_512: np.ndarray | None,
    *,
    min_mass_delta_ratio: float = 0.03,
    min_center_delta_ratio: float = 0.01,
) -> tuple[str | None, float, dict[str, Any]]:
    """
    Infer left/right direction from mask mass distribution.

    Heuristic:
      - Compare left-half vs right-half mask mass.
      - Compare center-of-mass x against image center.
      - Return None when both cues are too weak.
    """
    if mask_512 is None:
        return None, 0.0, {"reason": "no_mask"}

    mask = (mask_512 > 0).astype(np.uint8)
    total = float(np.count_nonzero(mask))
    if total <= 0:
        return None, 0.0, {"reason": "empty_mask"}

    mid = STANDARD_SIZE // 2
    left_mass = float(np.count_nonzero(mask[:, :mid]))
    right_mass = float(np.count_nonzero(mask[:, mid:]))
    mass_delta = right_mass - left_mass
    mass_delta_ratio = mass_delta / max(total, 1.0)

    moments = cv2.moments(mask)
    cx = None
    center_delta_ratio = 0.0
    if moments.get("m00", 0.0):
        cx = float(moments["m10"] / moments["m00"])
        center_delta_ratio = (cx - (STANDARD_SIZE / 2.0)) / float(STANDARD_SIZE)

    direction = None
    confidence = 0.0
    if abs(mass_delta_ratio) >= float(min_mass_delta_ratio):
        # In directional schemas we treat the heavier side as the "head side".
        direction = "right" if mass_delta_ratio > 0 else "left"
        confidence = abs(float(mass_delta_ratio))
    elif abs(center_delta_ratio) >= float(min_center_delta_ratio):
        direction = "right" if center_delta_ratio > 0 else "left"
        confidence = abs(float(center_delta_ratio))

    return direction, float(confidence), {
        "left_mass": left_mass,
        "right_mass": right_mass,
        "mass_delta_ratio": float(mass_delta_ratio),
        "center_x": None if cx is None else float(cx),
        "center_delta_ratio": float(center_delta_ratio),
    }


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
    return source == "detector_hint"


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
    Transform order (inverse):
      1) un-flip (if needed)
      2) un-rotate in STANDARD_SIZE space
      3) un-resize to crop space
      4) offset by crop origin
      5) un-scale to original image size
      6) optional bounds clamp
    """
    sx = float(metadata.get("scale_x", metadata.get("scale", 1.0))) or 1.0
    sy = float(metadata.get("scale_y", metadata.get("scale", 1.0))) or 1.0
    ox, oy = metadata["crop_origin"]
    angle = float(metadata.get("rotation", 0.0))

    img_h = image_shape[0] if image_shape is not None else None
    img_w = image_shape[1] if image_shape is not None else None

    angle_rad = np.radians(-angle)
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    cx_s = STANDARD_SIZE / 2.0
    cy_s = STANDARD_SIZE / 2.0

    mapped: list[dict[str, Any]] = []
    canonical_flip_applied = bool(metadata.get("canonical_flip_applied", False))
    effective_flip = bool(was_flipped) ^ canonical_flip_applied

    for lm in landmarks_512:
        x_s = max(0.0, min(float(STANDARD_SIZE - 1), float(lm["x"])))
        y_s = max(0.0, min(float(STANDARD_SIZE - 1), float(lm["y"])))

        if effective_flip:
            x_s = _mirror_x(x_s, STANDARD_SIZE)

        if abs(angle) > 1e-9:
            x_u = cx_s + (x_s - cx_s) * cos_a - (y_s - cy_s) * sin_a
            y_u = cy_s + (x_s - cx_s) * sin_a + (y_s - cy_s) * cos_a
        else:
            x_u, y_u = x_s, y_s

        x_crop = x_u / sx
        y_crop = y_u / sy
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
