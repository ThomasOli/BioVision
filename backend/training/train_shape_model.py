# backend/train_shape_model.py
"""Train dlib shape predictor with optimized parameters."""
import os
import sys
import json
import math
import time
import xml.etree.ElementTree as ET
import cv2
import dlib
_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import bv_utils.debug_io as dio
import bv_utils.lineage as lineage
import bv_utils.orientation_utils as ou
from bv_utils.landmark_artifacts import bundle_id_mapping

STANDARD_SIZE = 512
DLIB_PROMOTION_MIN_ABSOLUTE_IMPROVEMENT = 1e-4
DLIB_PROMOTION_MIN_RELATIVE_IMPROVEMENT = 0.005
import numpy as np


def _dlib_promotion_policy() -> dict:
    return {
        "policyVersion": "dlib_normalized_error_effect_v1",
        "minimumAbsoluteImprovement": DLIB_PROMOTION_MIN_ABSOLUTE_IMPROVEMENT,
        "minimumRelativeImprovement": DLIB_PROMOTION_MIN_RELATIVE_IMPROVEMENT,
    }


def _build_dlib_validation_evaluator_protocol(id_mapping: dict) -> dict:
    """Pin the exact dlib validation metric/mapping contract used for promotion."""
    raw_indices = id_mapping.get("dlib_index_to_original", {})
    indexed_ids = sorted(
        ((int(index), int(original_id)) for index, original_id in raw_indices.items()),
        key=lambda item: item[0],
    )
    names_by_original: dict[int, list[str]] = {}
    for part_name, original_id in id_mapping.get("dlib_name_to_original", {}).items():
        try:
            names_by_original.setdefault(int(original_id), []).append(str(part_name))
        except (TypeError, ValueError):
            continue
    landmark_order = []
    for model_index, original_id in indexed_ids:
        matching_names = sorted(
            names_by_original.get(original_id, []),
            key=lambda value: (
                int(value) if value.isdigit() else float("inf"),
                value,
            ),
        )
        landmark_order.append(
            {
                "modelIndex": model_index,
                "xmlPartName": matching_names[0] if matching_names else str(model_index),
                "schemaLandmarkId": original_id,
            }
        )

    return lineage.build_validation_evaluator_protocol(
        {
            "formatVersion": 1,
            "role": "landmark_validation_model_promotion",
            "modelType": "dlib",
            "evaluator": {
                "implementation": "backend.training.train_shape_model",
                "implementationVersion": "dlib_validation_evaluator_v1",
                "nativeMeanCallable": "dlib.test_shape_predictor(dataset_filename,predictor_filename)",
                "medianCallable": "_compute_dlib_per_image_errors",
                "framework": "dlib",
                "frameworkVersion": str(getattr(dlib, "__version__", "unknown")),
                "numpyVersion": str(np.__version__),
                "opencvVersion": str(cv2.__version__),
            },
            "preprocessing": {
                "canonicalImageSize": [STANDARD_SIZE, STANDARD_SIZE],
                "imageDecoder": "cv2.imread",
                "colorConversion": "cv2.COLOR_BGR2RGB",
                "predictionRectangleInclusive": [
                    0,
                    0,
                    STANDARD_SIZE - 1,
                    STANDARD_SIZE - 1,
                ],
                "xmlBoxSelection": "first_box",
                "targetPartOrdering": "numeric_then_lexical_part_name",
                "missingOrUnreadableSamplePolicy": "skip",
                "predictionPartCount": "min(predicted_parts,target_parts)",
            },
            "landmarkOrder": landmark_order,
            "metricDefinitions": {
                "validationMedianError": {
                    "implementation": "numpy.median",
                    "perLandmark": "euclidean_l2(predicted_xy,target_xy)",
                    "perImageReduction": "mean",
                    "normalizationDenominator": float(STANDARD_SIZE),
                    "cohortReduction": "median",
                    "promotionPriority": 1,
                },
                "validationError": {
                    "implementation": "dlib.test_shape_predictor",
                    "definition": "native_mean_average_part_error",
                    "explicitScaleNormalization": False,
                    "promotionPriority": 2,
                },
                "validationP95Error": {
                    "source": "validationMedianError_per_image_values",
                    "cohortReduction": "numpy.percentile(method=linear,p=95)",
                },
                "validationMaxError": {
                    "source": "validationMedianError_per_image_values",
                    "cohortReduction": "max",
                },
            },
            "stabilityGate": {
                "implementation": "_dlib_validation_stability_gate",
                "implementationVersion": "dlib_validation_tail_gate_v1",
                "meanFailure": {"ratioGte": 4.0, "absoluteGte": 0.08},
                "p95Failure": {"ratioGte": 6.0, "absoluteGte": 0.15},
                "maxFailure": {"ratioGte": 10.0, "absoluteGte": 0.20},
                "combination": "reject_if_any_failure",
            },
        }
    )


def _resolve_xml_image_path(xml_path: str, image_reference: str) -> str:
    reference = os.path.expanduser(str(image_reference or "").strip())
    if not reference:
        raise RuntimeError(f"Training XML {xml_path} contains an image with no file path.")
    return reference if os.path.isabs(reference) else os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(xml_path)), reference)
    )


def _effective_xml_role_snapshot(xml_path: str, role: str) -> dict:
    """Fingerprint the exact XML geometry and canonical pixels consumed by dlib."""
    if not os.path.isfile(xml_path):
        raise RuntimeError(f"Effective {role} XML is missing: {xml_path}")
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        raise RuntimeError(f"Effective {role} XML is unreadable: {xml_path}: {exc}") from exc

    images_node = root.find("images")
    samples = []
    for image_node in images_node.findall("image") if images_node is not None else []:
        image_path = _resolve_xml_image_path(xml_path, image_node.get("file", ""))
        if not os.path.isfile(image_path):
            raise RuntimeError(
                f"Effective {role} XML references a missing canonical crop: {image_path}"
            )
        boxes = []
        for box_node in image_node.findall("box"):
            parts = [
                {
                    "name": str(part.get("name") or ""),
                    "x": int(part.get("x", 0)),
                    "y": int(part.get("y", 0)),
                }
                for part in box_node.findall("part")
            ]
            parts.sort(key=lambda item: (item["name"], item["x"], item["y"]))
            boxes.append(
                {
                    "left": int(box_node.get("left", 0)),
                    "top": int(box_node.get("top", 0)),
                    "width": int(box_node.get("width", 0)),
                    "height": int(box_node.get("height", 0)),
                    "parts": parts,
                }
            )
        boxes.sort(key=lineage.sha256_json)
        image_sha256 = lineage.sha256_file(image_path)
        samples.append(
            {
                "contentId": f"sha256:{image_sha256}",
                "imageSha256": image_sha256,
                "boxes": boxes,
            }
        )
    samples.sort(key=lineage.sha256_json)
    payload = {
        "formatVersion": 1,
        "role": str(role),
        "xmlSha256": lineage.sha256_file(xml_path),
        "sampleCount": len(samples),
        "boxCount": sum(len(sample["boxes"]) for sample in samples),
        "partCount": sum(
            len(box["parts"])
            for sample in samples
            for box in sample["boxes"]
        ),
        "samples": samples,
    }
    payload["revision"] = lineage.sha256_json(payload)
    return payload


def _build_effective_dlib_dataset_snapshot(role_paths: dict[str, str], artifact_dir: str) -> dict:
    """Create an artifact-local, path-independent descriptor of every effective role."""
    roles = {}
    for role, xml_path in sorted(role_paths.items()):
        if not xml_path:
            continue
        snapshot = _effective_xml_role_snapshot(xml_path, role)
        relative_path = os.path.join("datasets", f"{role}.xml").replace("\\", "/")
        artifact_xml_path = os.path.join(artifact_dir, *relative_path.split("/"))
        lineage.atomic_copy_file(xml_path, artifact_xml_path)
        if lineage.sha256_file(artifact_xml_path) != snapshot["xmlSha256"]:
            raise RuntimeError(f"Immutable {role} XML copy failed integrity verification.")
        roles[role] = {
            **snapshot,
            "artifactRelativePath": relative_path,
        }
    canonical = {"formatVersion": 1, "modelType": "dlib", "roles": roles}
    return {**canonical, "revision": lineage.sha256_json(canonical)}


def _assert_effective_dlib_dataset_unchanged(
    role_paths: dict[str, str],
    expected: dict,
) -> None:
    """Fail if any trainer/evaluator XML, geometry, or crop byte changed mid-run."""
    roles = {}
    for role, xml_path in sorted(role_paths.items()):
        if not xml_path:
            continue
        relative_path = os.path.join("datasets", f"{role}.xml").replace("\\", "/")
        roles[role] = {
            **_effective_xml_role_snapshot(xml_path, role),
            "artifactRelativePath": relative_path,
        }
    canonical = {"formatVersion": 1, "modelType": "dlib", "roles": roles}
    observed_revision = lineage.sha256_json(canonical)
    expected_revision = str(expected.get("revision") or "")
    if not expected_revision or observed_revision != expected_revision:
        raise RuntimeError(
            "The effective dlib dataset changed while training or evaluation was running. "
            "The artifact was not published; restore the prepared XML/crops and retry."
        )


def _build_dlib_training_protocol(params_log: dict, effective_dataset_revision: str) -> dict:
    excluded = {"model_id", "artifact_tag", "run_id", "augmented_xml"}
    parameters = {
        key: value
        for key, value in params_log.items()
        if key not in excluded
    }
    canonical = {
        "formatVersion": 1,
        "modelType": "dlib",
        "trainer": "dlib.train_shape_predictor",
        "effectiveDatasetRevision": str(effective_dataset_revision),
        "parameters": parameters,
        "promotionPolicy": _dlib_promotion_policy(),
    }
    return {**canonical, "revision": lineage.sha256_json(canonical)}


def _apply_photo_jitter(
    img: np.ndarray,
    rng: np.random.Generator,
    profile: dict | None = None,
) -> np.ndarray:
    """Apply dataset-aware brightness/contrast/saturation jitter for offline dlib augmentation."""
    assert img.ndim == 3 and img.shape[2] == 3, "Expected 3-channel BGR image"
    jitter_profile = dict(profile or {})
    contrast_delta = float(jitter_profile.get("photo_jitter_contrast_delta", 0.25))
    brightness_delta = float(jitter_profile.get("photo_jitter_brightness_delta", 40.0))
    saturation_range = jitter_profile.get("photo_jitter_saturation_range", (0.7, 1.3))
    if not isinstance(saturation_range, (list, tuple)) or len(saturation_range) != 2:
        saturation_range = (0.7, 1.3)
    sat_lo = float(saturation_range[0])
    sat_hi = float(saturation_range[1])
    if sat_hi < sat_lo:
        sat_lo, sat_hi = sat_hi, sat_lo

    alpha = 1.0 + rng.uniform(-contrast_delta, contrast_delta)
    beta = rng.uniform(-brightness_delta, brightness_delta)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(sat_lo, sat_hi), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _resolve_dlib_aug_angles(orientation_mode: str, aug_angles_param) -> list:
    """Augmentation Router: select rotation angles for dlib offline augmentation.

    If the caller passes an explicit list (even []), it is honoured verbatim for
    backward compatibility. When None, the router applies model-aware defaults:

    - directional / axial: ±5° strict cap. Dlib mean-shape initialization is
      unstable beyond this for polarised/axial geometries.
    - invariant / bilateral: expanded list [-30, -15, 15, 30]. No chirality to
      protect, so heavier rotation helps regularisation.
    """
    if aug_angles_param is not None:
        return list(aug_angles_param)
    profile = ou.get_schema_augmentation_profile(
        orientation_mode,
        engine="dlib",
    )
    return list(profile.get("angles", []))


def _resolve_orientation_mode(project_root, tag):
    """
    Resolve orientation mode used for this model tag.

    Priority:
    1) debug/id_mapping_{tag}.json training_config.orientation_mode
    2) session orientation policy mode
    3) invariant
    """
    id_mapping_path = os.path.join(project_root, "debug", f"id_mapping_{tag}.json")
    if os.path.exists(id_mapping_path):
        try:
            with open(id_mapping_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            mode = str(
                raw.get("training_config", {}).get("orientation_mode", "")
            ).strip().lower()
            if mode in {"directional", "bilateral", "axial", "invariant"}:
                return mode
        except Exception:
            pass
    policy = ou.load_orientation_policy(project_root)
    mode = str(policy.get("mode", "invariant")).strip().lower()
    if mode in {"directional", "bilateral", "axial", "invariant"}:
        return mode
    return "invariant"


def _rotate_point(x, y, cx, cy, angle_deg):
    """Rotate point (x, y) around (cx, cy) by angle_deg, clamped to [0, STANDARD_SIZE-1]."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    xr = cx + (x - cx) * cos_a - (y - cy) * sin_a
    yr = cy + (x - cx) * sin_a + (y - cy) * cos_a
    return (
        max(0.0, min(STANDARD_SIZE - 1, xr)),
        max(0.0, min(STANDARD_SIZE - 1, yr)),
    )


def _load_id_mapping(project_root, tag):
    id_mapping_path = os.path.join(project_root, "debug", f"id_mapping_{tag}.json")
    if not os.path.exists(id_mapping_path):
        return {}
    try:
        with open(id_mapping_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _resolve_dlib_name_swap_map(project_root, tag, orientation_mode):
    if orientation_mode != "bilateral":
        return {}
    policy = ou.load_orientation_policy(project_root)
    pairs = ou.get_bilateral_pairs(policy)
    if not pairs:
        return {}
    id_mapping = _load_id_mapping(project_root, tag)
    pair_swap = ou.build_pair_swap_map(pairs)
    return ou.build_dlib_name_swap_map(id_mapping, pair_swap)


def augment_training_data(
    train_xml_path,
    aug_dir,
    aug_angles=None,
    add_flip=False,
    name_swap_map=None,
    max_augmented_copies_per_image=None,
    photo_jitter_profile=None,
    random_seed=42,
):
    """
    Pre-augment a dlib training XML by generating rotated (and optionally flipped)
    copies of each 512×512 crop with adjusted landmark coordinates.

    Augmented crops are saved to aug_dir and the new XML (containing original +
    augmented entries) is written alongside the originals.

    Args:
        train_xml_path: Path to the original dlib training XML.
        aug_dir: Directory to save augmented crop images.
        aug_angles: List of rotation angles in degrees, e.g. [-30, -15, 15, 30].
        add_flip: If True, also add a horizontally-flipped copy with mirrored x coords.

    Returns:
        Path to the augmented training XML.
    """
    if aug_angles is None:
        aug_angles = []
    name_swap_map = dict(name_swap_map or {})

    os.makedirs(aug_dir, exist_ok=True)

    tree = ET.parse(train_xml_path)
    root = tree.getroot()
    images_el = root.find("images")
    if images_el is None:
        return train_xml_path  # nothing to augment

    cx, cy = STANDARD_SIZE / 2.0, STANDARD_SIZE / 2.0
    augmented_entries = []  # list of (file_path, parts_list) where parts_list = [(name,x,y),...]
    rng = np.random.default_rng(int(random_seed))

    for img_el in images_el.findall("image"):
        img_file = img_el.get("file")
        if not img_file or not os.path.exists(img_file):
            continue

        # Collect all parts from the first box (dlib XML has one box = full 512×512 image)
        box_el = img_el.find("box")
        if box_el is None:
            continue
        parts = [(p.get("name"), int(p.get("x", 0)), int(p.get("y", 0)))
                 for p in box_el.findall("part")]
        if not parts:
            continue

        img = cv2.imread(img_file)
        if img is None:
            continue

        base = os.path.splitext(os.path.basename(img_file))[0]

        candidate_specs = []
        for angle in aug_angles:
            angle_tag = str(angle).replace("-", "m")
            candidate_specs.append({
                "kind": "rotate",
                "angle": float(angle),
                "flip": False,
                "suffix": f"_aug_r{angle_tag}",
            })
            if add_flip:
                candidate_specs.append({
                    "kind": "rotate",
                    "angle": float(angle),
                    "flip": True,
                    "suffix": f"_aug_r{angle_tag}_flip",
                })

        if add_flip:
            candidate_specs.append({
                "kind": "flip",
                "flip": True,
                "suffix": "_aug_flip",
            })

        if not candidate_specs:
            continue

        max_copies = None if max_augmented_copies_per_image is None else max(0, int(max_augmented_copies_per_image))
        if max_copies == 0:
            continue
        if max_copies is not None and len(candidate_specs) > max_copies:
            selected_indices = sorted(
                int(i) for i in rng.choice(len(candidate_specs), size=max_copies, replace=False).tolist()
            )
            selected_specs = [candidate_specs[i] for i in selected_indices]
        else:
            selected_specs = candidate_specs

        for spec in selected_specs:
            kind = str(spec.get("kind", ""))
            if kind == "rotate":
                angle = float(spec.get("angle", 0.0))
                M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                aug_img = cv2.warpAffine(
                    img,
                    M,
                    (STANDARD_SIZE, STANDARD_SIZE),
                    borderMode=cv2.BORDER_REPLICATE,
                )
                aug_parts = [
                    (name, *_rotate_point(x, y, cx, cy, angle))
                    for name, x, y in parts
                ]
                if bool(spec.get("flip", False)):
                    aug_img = cv2.flip(aug_img, 1)
                    aug_parts = [
                        (name_swap_map.get(name, name), float(STANDARD_SIZE - 1 - rx), float(ry))
                        for name, rx, ry in aug_parts
                    ]
            elif kind == "flip":
                aug_img = cv2.flip(img, 1)
                aug_parts = [
                    (name_swap_map.get(name, name), float(STANDARD_SIZE - 1 - x), float(y))
                    for name, x, y in parts
                ]
            else:
                continue

            aug_img = _apply_photo_jitter(aug_img, rng, photo_jitter_profile)
            out_path = os.path.join(aug_dir, f"{base}{spec.get('suffix', '_aug')}.png")
            cv2.imwrite(out_path, aug_img)
            augmented_entries.append((out_path, aug_parts))

    if not augmented_entries:
        return train_xml_path  # no augmentation produced

    # Build new XML: original entries + augmented entries
    new_root = ET.Element("dataset")
    new_images = ET.SubElement(new_root, "images")

    # Copy original entries
    for img_el in images_el.findall("image"):
        new_images.append(img_el)

    # Append augmented entries
    for file_path, parts in augmented_entries:
        img_el = ET.SubElement(new_images, "image", file=file_path)
        box_el = ET.SubElement(img_el, "box", top="0", left="0",
                               width=str(STANDARD_SIZE), height=str(STANDARD_SIZE))
        for name, x, y in parts:
            ET.SubElement(box_el, "part", name=name,
                          x=str(int(round(x))), y=str(int(round(y))))

    aug_xml_path = os.path.join(
        os.path.dirname(train_xml_path),
        os.path.basename(train_xml_path).replace(".xml", "_augmented.xml"),
    )
    ET.ElementTree(new_root).write(aug_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Augmented XML written: {aug_xml_path} "
          f"(original + {len(augmented_entries)} augmented entries)", file=sys.stderr)
    return aug_xml_path


def _capacity_tier(n_raw: int) -> str:
    """3-tier capacity bucket based on unique real-world source images.

    Starvation (< 250): prevent catastrophic memorisation — shallow cascades,
        minimal oversampling, low nu so the model takes tiny generalising steps.
    Balanced (250–999): standard feature extraction.
    Deep (≥ 1000): large enough that trees learn long-tail morphological variance.
    """
    n = int(max(0, n_raw))
    if n < 250:
        return "starvation"
    if n < 1000:
        return "balanced"
    return "deep"


def _resolve_dlib_oversampling_amount(
    base_oversampling_amount: int,
    *,
    effective_num_images: int,
    effective_training_exposure_target: int,
) -> int:
    """Cap dlib oversampling so offline augmentation and oversampling do not explode together."""
    base = max(1, int(base_oversampling_amount))
    effective_images = max(1, int(effective_num_images))
    target = max(1, int(effective_training_exposure_target))
    capped = max(1, target // effective_images)
    return min(base, capped)


def get_training_options(num_images, num_landmarks, orientation_mode="invariant"):
    """Get optimized training options based on dataset size and orientation mode.

    Uses a 3-tier capacity system keyed on unique real-world source images:
      starvation  (< 250) : shallow model, minimal oversampling, low nu
      balanced    (250–999): standard extraction
      deep        (≥ 1000) : high-capacity, large feature pool

    Orientation overrides are applied on top:
      directional / bilateral : geometry is stable → conservative cap on cascade
          depth and oversampling to avoid memorising polarised features.
      axial                   : pole-ambiguity → tight translation jitter so
          trees don't flip between poles.
    """
    options = dlib.shape_predictor_training_options()
    tier = _capacity_tier(num_images)

    # ── Base parameters by capacity tier ──────────────────────────────────────
    _base = {
        "starvation": {
            "tree_depth": 3,
            "cascade_depth": 8,
            "nu": 0.05,
            "feature_pool_size": 400,
            "num_trees_per_cascade_level": 100,
            "num_test_splits": 10,
            "oversampling_amount": 10,
            "oversampling_translation_jitter": 0.0,
            "feature_pool_region_padding": 0.1,
            "lambda_param": 0.2,
        },
        "balanced": {
            "tree_depth": 4,
            "cascade_depth": 12,
            "nu": 0.08,
            "feature_pool_size": 400,
            "num_trees_per_cascade_level": 200,
            "num_test_splits": 15,
            "oversampling_amount": 30,
            "oversampling_translation_jitter": 0.0,
            "feature_pool_region_padding": 0.1,
            "lambda_param": 0.1,
        },
        "deep": {
            "tree_depth": 4,
            "cascade_depth": 15,
            "nu": 0.1,
            "feature_pool_size": 600,
            "num_trees_per_cascade_level": 400,
            "num_test_splits": 20,
            "oversampling_amount": 60,
            "oversampling_translation_jitter": 0.0,
            "feature_pool_region_padding": 0.1,
            "lambda_param": 0.1,
        },
    }
    for key, value in _base[tier].items():
        setattr(options, key, value)

    mode = str(orientation_mode or "").strip().lower()

    # ── Directional / bilateral override ──────────────────────────────────────
    # The mean shape is highly stable (e.g. side-view fish always faces left).
    # This makes the regression trees more prone to memorising polarised features,
    # so we cap cascade depth and oversampling slightly below the base tier.
    # Starvation oversampling is boosted slightly (10→12) so the trees don't snap
    # too rigidly to the small polarised feature set.
    if mode in ("directional", "bilateral"):
        _directional = {
            "starvation": {
                "tree_depth": 3,
                "cascade_depth": 8,
                "nu": 0.06,
                "feature_pool_size": 350,
                "num_trees_per_cascade_level": 120,
                "num_test_splits": 10,
                "oversampling_amount": 12,
                "oversampling_translation_jitter": 0.0,
                "feature_pool_region_padding": 0.1,
                "lambda_param": 0.12,
            },
            "balanced": {
                "tree_depth": 4,
                "cascade_depth": 12,
                "nu": 0.08,
                "feature_pool_size": 400,
                "num_trees_per_cascade_level": 220,
                "num_test_splits": 14,
                "oversampling_amount": 30,
                "oversampling_translation_jitter": 0.0,
                "feature_pool_region_padding": 0.1,
                "lambda_param": 0.1,
            },
            "deep": {
                "tree_depth": 4,
                "cascade_depth": 14,   # capped at 14 (not 15) — stable geometry
                "nu": 0.09,
                "feature_pool_size": 450,
                "num_trees_per_cascade_level": 380,
                "num_test_splits": 18,
                "oversampling_amount": 50,  # capped at 50 (not 60)
                "oversampling_translation_jitter": 0.0,
                "feature_pool_region_padding": 0.1,
                "lambda_param": 0.1,
            },
        }
        for key, value in _directional[tier].items():
            setattr(options, key, value)

    # ── Axial override ─────────────────────────────────────────────────────────
    # OBB leveling aligns the specimen along its long axis but 0° and 180° are
    # indistinguishable. Heavy translation jitter causes trees to confuse which
    # pole they are looking at → lock translation down hard.
    if mode == "axial":
        options.oversampling_translation_jitter = 0.0

    options.random_seed = "42"
    options.be_verbose = True

    import multiprocessing
    options.num_threads = multiprocessing.cpu_count()

    return options, tier


def count_landmarks_in_xml(xml_path):
    """Count images and landmarks in a dlib XML file."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    images = root.find("images")
    if images is None:
        return 0, 0

    num_images = 0
    num_landmarks = 0
    for image in images.findall("image"):
        num_images += 1
        for box in image.findall("box"):
            parts = box.findall("part")
            if parts:
                num_landmarks = max(num_landmarks, len(parts))

    return num_images, num_landmarks


def _compute_dlib_per_image_errors(xml_path: str, predictor_path: str) -> list:
    """Run the trained predictor on every image in xml_path.

    Returns a list of per-image mean L2 landmark errors normalised by STANDARD_SIZE,
    matching the normalisation convention used by dlib.test_shape_predictor().
    Used to compute median error alongside the mean.
    """
    predictor = dlib.shape_predictor(predictor_path)
    tree = ET.parse(xml_path)
    images_el = tree.getroot().find("images")
    if images_el is None:
        return []
    errors = []
    for img_el in images_el.findall("image"):
        img_file = img_el.get("file", "")
        box_el = img_el.find("box")
        if not img_file or not os.path.exists(img_file) or box_el is None:
            continue
        gt_parts = {
            p.get("name"): (int(p.get("x", 0)), int(p.get("y", 0)))
            for p in box_el.findall("part")
        }
        if not gt_parts:
            continue
        img_bgr = cv2.imread(img_file)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        det = dlib.rectangle(0, 0, STANDARD_SIZE - 1, STANDARD_SIZE - 1)
        shape = predictor(img_rgb, det)
        n = min(shape.num_parts, len(gt_parts))
        if n == 0:
            continue
        names = sorted(
            gt_parts.keys(),
            key=lambda s: (int(s) if s.isdigit() else float("inf"), s),
        )
        dist_sum = sum(
            math.sqrt(
                (shape.part(i).x - gt_parts[names[i]][0]) ** 2
                + (shape.part(i).y - gt_parts[names[i]][1]) ** 2
            )
            for i in range(n)
        )
        errors.append(dist_sum / (n * float(STANDARD_SIZE)))
    return errors


def _compute_dlib_per_image_error_details(xml_path: str, predictor_path: str) -> list[dict]:
    predictor = dlib.shape_predictor(predictor_path)
    tree = ET.parse(xml_path)
    images_el = tree.getroot().find("images")
    if images_el is None:
        return []
    details = []
    for img_el in images_el.findall("image"):
        img_file = img_el.get("file", "")
        box_el = img_el.find("box")
        if not img_file or not os.path.exists(img_file) or box_el is None:
            continue
        gt_parts = {
            p.get("name"): (int(p.get("x", 0)), int(p.get("y", 0)))
            for p in box_el.findall("part")
        }
        if not gt_parts:
            continue
        img_bgr = cv2.imread(img_file)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        det = dlib.rectangle(0, 0, STANDARD_SIZE - 1, STANDARD_SIZE - 1)
        shape = predictor(img_rgb, det)
        n = min(shape.num_parts, len(gt_parts))
        if n == 0:
            continue
        names = sorted(
            gt_parts.keys(),
            key=lambda s: (int(s) if s.isdigit() else float("inf"), s),
        )
        per_landmark = []
        for i in range(n):
            dist = math.sqrt(
                (shape.part(i).x - gt_parts[names[i]][0]) ** 2
                + (shape.part(i).y - gt_parts[names[i]][1]) ** 2
            ) / float(STANDARD_SIZE)
            per_landmark.append(float(dist))
        details.append(
            {
                "image": img_file,
                "filename": os.path.basename(img_file),
                "mean_error": float(sum(per_landmark) / max(1, len(per_landmark))),
                "median_error": float(np.median(per_landmark)),
                "per_landmark_error": per_landmark,
            }
        )
    return details


def _dlib_validation_stability_gate(per_image_errors, mean_error, median_error):
    """Reject candidates whose median improvement hides catastrophic tails."""
    finite_errors = []
    for value in per_image_errors or []:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric) and numeric >= 0.0:
            finite_errors.append(numeric)
    if not finite_errors or mean_error is None or median_error is None:
        return {
            "available": False,
            "passed": None,
            "reason": "validation_stability_unavailable",
        }
    median = float(median_error)
    denominator = max(median, 1e-12)
    p95 = float(np.percentile(finite_errors, 95))
    maximum = float(max(finite_errors))
    mean_median_ratio = float(mean_error) / denominator
    p95_median_ratio = p95 / denominator
    max_median_ratio = maximum / denominator
    # Ratio-only gates are unstable when a strong model has a near-zero median.
    # Require both extreme relative divergence and a biologically material
    # absolute normalized error before declaring a catastrophic tail.
    mean_tail_failure = mean_median_ratio >= 4.0 and float(mean_error) >= 0.08
    p95_tail_failure = p95_median_ratio >= 6.0 and p95 >= 0.15
    max_tail_failure = max_median_ratio >= 10.0 and maximum >= 0.20
    passed = not (
        mean_tail_failure
        or p95_tail_failure
        or max_tail_failure
    )
    return {
        "available": True,
        "passed": passed,
        "reason": "validation_tail_stable" if passed else "catastrophic_validation_outliers",
        "meanMedianRatio": mean_median_ratio,
        "p95MedianRatio": p95_median_ratio,
        "maxMedianRatio": max_median_ratio,
        "p95Error": p95,
        "maxError": maximum,
        "sampleCount": len(finite_errors),
        "failedChecks": {
            "meanTail": mean_tail_failure,
            "p95Tail": p95_tail_failure,
            "maxTail": max_tail_failure,
        },
        "thresholds": {
            "meanMedianRatioExclusiveMax": 4.0,
            "meanErrorMaterialMinimum": 0.08,
            "p95MedianRatioExclusiveMax": 6.0,
            "p95ErrorMaterialMinimum": 0.15,
            "maxMedianRatioExclusiveMax": 10.0,
            "maxErrorMaterialMinimum": 0.20,
        },
    }


def train_shape_model(project_root, tag, custom_options=None,
                      aug_angles=None, aug_flip=None):
    """
    Train a dlib shape predictor model.

    Args:
        project_root: Session root directory.
        tag: Model tag (e.g. "Fish_v1").
        custom_options: Dict of dlib option overrides.
        aug_angles: Rotation angles for pre-training augmentation.
                    Defaults to [-30, -15, 15, 30] when None.
        aug_flip: If True, also add horizontally-flipped copies to training data.
                  If None, defaults by orientation mode:
                  directional=False, others=True.
    """
    project_root = os.path.abspath(project_root)
    xmldir = os.path.join(project_root, "xml")
    modeldir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    os.makedirs(modeldir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    run_dir, run_id = dio.create_model_run_dir(project_root, "dlib", tag)
    model_id = lineage.build_model_id("dlib", run_id)
    artifact_tag = lineage.build_artifact_tag(tag, run_id)
    artifact_dir = lineage.create_model_artifact_dir(project_root, "dlib", run_id)
    dio.write_run_manifest(
        run_dir,
        model_type="dlib",
        tag=tag,
        project_root=project_root,
        extra={"status": "started"},
    )

    train_xml = os.path.join(xmldir, f"train_{tag}.xml")
    validation_xml = os.path.join(xmldir, f"validation_{tag}.xml")
    test_xml = os.path.join(xmldir, f"test_{tag}.xml")
    # Train directly into an immutable run directory. Compatibility aliases are
    # published only after training and evaluation complete successfully.
    predictor_path = os.path.join(artifact_dir, "predictor.dat")

    if not os.path.exists(train_xml):
        raise FileNotFoundError(f"Train XML not found at {train_xml}")
    artifact_id_mapping_path, id_mapping_descriptor, artifact_id_mapping = bundle_id_mapping(
        os.path.join(debug_dir, f"id_mapping_{tag}.json"),
        artifact_dir,
    )
    print("PROGRESS 8 loading_dataset", file=sys.stderr)
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 8,
                "stage": "training",
                "substage": "init",
                "message": "Loaded dlib training XML.",
            }
        ),
        file=sys.stderr,
    )

    # Count raw train samples before augmentation so bucket selection is not
    # biased upward by synthetic/rotated copies.
    raw_num_images, raw_num_landmarks = count_landmarks_in_xml(train_xml)
    orientation_mode = _resolve_orientation_mode(project_root, tag)
    size_bucket = ou.get_training_capacity_tier(raw_num_images)
    aug_override = ou.load_augmentation_policy(project_root)
    aug_profile = ou.get_landmark_training_augmentation_profile(
        orientation_mode,
        engine="dlib",
        dataset_size_bucket=size_bucket,
        augmentation_override=aug_override or None,
    )
    if aug_flip is None:
        aug_flip = bool(aug_profile.get("flip", ou.default_allow_flip_augmentation(orientation_mode)))

    # ── Pre-training augmentation ──────────────────────────────────────────────
    # Augmentation Router: model-aware angle selection.
    # Explicit aug_angles override → forwarded verbatim (API compat).
    # None → router selects based on orientation mode:
    #   directional/axial → ±5° (mean-shape stability)
    #   invariant/bilateral → [-30, -15, 15, 30] (no chirality to protect)
    aug_angles = _resolve_dlib_aug_angles(orientation_mode, aug_angles)
    raw_augmentation_seed = (custom_options or {}).get(
        "augmentation_seed",
        (custom_options or {}).get("random_seed", 42),
    )
    try:
        augmentation_seed = int(raw_augmentation_seed)
    except (TypeError, ValueError):
        augmentation_seed = int(
            lineage.sha256_bytes(str(raw_augmentation_seed).encode("utf-8"))[:16],
            16,
        ) % (2 ** 32)
    print("PROGRESS 15 preparing_augmentation", file=sys.stderr)

    actual_train_xml = train_xml
    name_swap_map = {}
    if aug_flip:
        name_swap_map = _resolve_dlib_name_swap_map(project_root, tag, orientation_mode)
    if aug_angles or aug_flip:
        print("PROGRESS 22 generating_augmented_samples", file=sys.stderr)
        aug_dir = os.path.join(project_root, "corrected_images", "augmented")
        actual_train_xml = augment_training_data(
            train_xml,
            aug_dir,
            aug_angles,
            aug_flip,
            name_swap_map=name_swap_map,
            max_augmented_copies_per_image=aug_profile.get("max_augmented_copies_per_image"),
            photo_jitter_profile=aug_profile,
            random_seed=augmentation_seed,
        )

    # Count effective images/landmarks after augmentation
    num_images, num_landmarks = count_landmarks_in_xml(actual_train_xml)
    print(f"Training set: {num_images} images, {num_landmarks} landmarks per image "
          f"(aug_angles={aug_angles}, aug_flip={aug_flip}, pair_swaps={len(name_swap_map)})", file=sys.stderr)

    # Get optimized options
    options, size_bucket = get_training_options(
        raw_num_images,
        raw_num_landmarks,
        orientation_mode=orientation_mode,
    )
    effective_training_exposure_target = int(aug_profile.get("effective_training_exposure_target", 16000))
    base_oversampling_amount = int(options.oversampling_amount)
    resolved_oversampling_amount = _resolve_dlib_oversampling_amount(
        base_oversampling_amount,
        effective_num_images=num_images,
        effective_training_exposure_target=effective_training_exposure_target,
    )
    options.oversampling_amount = resolved_oversampling_amount
    effective_training_exposure = int(num_images * max(1, int(options.oversampling_amount)))
    print(
        f"Resolved dlib preset bucket: {size_bucket} "
        f"(orientation_mode={orientation_mode}, raw_images={raw_num_images}, effective_images={num_images})",
        file=sys.stderr,
    )
    print(
        f"Resolved dlib augmentation budget: max_augmented_copies_per_image={int(aug_profile.get('max_augmented_copies_per_image', 0))}, "
        f"base_oversampling={base_oversampling_amount}, resolved_oversampling={resolved_oversampling_amount}, "
        f"effective_training_exposure={effective_training_exposure}",
        file=sys.stderr,
    )
    if effective_training_exposure > int(round(effective_training_exposure_target * 1.25)):
        print(
            "WARNING: dlib effective training exposure exceeds the dataset-aware target; "
            "manual overrides may be causing excessive sample multiplication.",
            file=sys.stderr,
        )

    # Apply custom options if provided
    if custom_options:
        for key, value in custom_options.items():
            if hasattr(options, key):
                setattr(options, key, value)
                print(f"Custom option: {key} = {value}", file=sys.stderr)

    # Log training parameters
    params_log = {
        "model_id": model_id,
        "artifact_tag": artifact_tag,
        "run_id": run_id,
        "raw_num_images": raw_num_images,
        "raw_num_landmarks": raw_num_landmarks,
        "source_counts": {
            "train": raw_num_images,
            "validation": count_landmarks_in_xml(validation_xml)[0] if os.path.exists(validation_xml) else 0,
            "test": count_landmarks_in_xml(test_xml)[0] if os.path.exists(test_xml) else 0,
        },
        "num_images": num_images,
        "dataset_size_bucket": size_bucket,
        "num_landmarks": num_landmarks,
        "tree_depth": options.tree_depth,
        "cascade_depth": options.cascade_depth,
        "nu": options.nu,
        "feature_pool_size": options.feature_pool_size,
        "num_trees_per_cascade_level": options.num_trees_per_cascade_level,
        "num_test_splits": options.num_test_splits,
        "base_oversampling_amount": base_oversampling_amount,
        "oversampling_amount": options.oversampling_amount,
        "oversampling_translation_jitter": options.oversampling_translation_jitter,
        "feature_pool_region_padding": options.feature_pool_region_padding,
        "lambda_param": options.lambda_param,
        "random_seed": options.random_seed,
        "augmentation_seed": augmentation_seed,
        "num_threads": options.num_threads,
        "be_verbose": options.be_verbose,
        "aug_angles": aug_angles,
        "aug_flip": aug_flip,
        "aug_profile": aug_profile,
        "effective_training_exposure_target": effective_training_exposure_target,
        "effective_training_exposure": effective_training_exposure,
        "bilateral_name_swaps": name_swap_map,
        "orientation_mode": orientation_mode,
        "augmented_xml": actual_train_xml if actual_train_xml != train_xml else None,
    }
    params_path = os.path.join(debug_dir, f"training_params_{tag}.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_log, f, indent=2)
    print(f"Training parameters saved to: {params_path}", file=sys.stderr)
    dio.write_run_json(run_dir, "train_params.json", params_log)

    role_paths = {
        "fit": actual_train_xml,
        "trainReport": train_xml,
        **({"validation": validation_xml} if os.path.isfile(validation_xml) else {}),
        **({"test": test_xml} if os.path.isfile(test_xml) else {}),
    }
    effective_dataset = _build_effective_dlib_dataset_snapshot(role_paths, artifact_dir)
    effective_dataset_path = os.path.join(artifact_dir, "effective_dataset.json")
    lineage.atomic_write_json(effective_dataset_path, effective_dataset)
    training_protocol = _build_dlib_training_protocol(
        params_log,
        effective_dataset["revision"],
    )
    training_protocol_path = os.path.join(artifact_dir, "training_parameters.json")
    lineage.atomic_write_json(training_protocol_path, training_protocol)
    validation_evaluator_protocol = _build_dlib_validation_evaluator_protocol(
        artifact_id_mapping
    )
    validation_evaluator_protocol_path = os.path.join(
        artifact_dir,
        "validation_evaluator_protocol.json",
    )
    lineage.atomic_write_json(
        validation_evaluator_protocol_path,
        validation_evaluator_protocol,
    )

    baseline_record = lineage.get_active_model_record(project_root, "dlib")
    baseline_model_id = (
        str(baseline_record.get("modelId")) if baseline_record else None
    )
    lineage_payload = lineage.build_run_lineage(
        project_root,
        split_paths=[os.path.join(debug_dir, f"split_info_{tag}.json")],
        # dlib has no warm-start API here: every run is fit from scratch.  The
        # active model is the scientific comparison baseline, not a weight
        # parent, and those identities must not be conflated in lineage.
        parent_model_id=None,
        baseline_model_id=baseline_model_id,
        training_mode="retrain_from_base" if baseline_model_id else "train_from_base",
        initialization={
            "strategy": "from_scratch",
            "framework": "dlib",
            "checkpoint": None,
        },
    )
    lineage_payload["projectStateRevision"] = (
        lineage_payload.get("dataset", {}).get("revision")
        if isinstance(lineage_payload.get("dataset"), dict)
        else None
    )
    lineage_payload["effectiveDataset"] = {
        "revision": effective_dataset["revision"],
        "descriptorSha256": lineage.sha256_file(effective_dataset_path),
    }
    lineage_payload["trainingProtocol"] = {
        "revision": training_protocol["revision"],
        "descriptorSha256": lineage.sha256_file(training_protocol_path),
    }
    lineage_payload["validationEvaluatorProtocol"] = validation_evaluator_protocol
    lineage_payload["promotionPolicy"] = _dlib_promotion_policy()
    dio.write_run_json(run_dir, "lineage.json", lineage_payload)

    # Capture the dataset/debug artifacts used to train this run.
    for name in [
        f"id_mapping_{tag}.json",
        f"crop_metadata_{tag}.json",
        f"box_scale_{tag}.json",
        f"orientation_{tag}.json",
        f"training_boxes_{tag}.json",
        f"split_info_{tag}.json",
    ]:
        dio.copy_json_if_exists(os.path.join(debug_dir, name), run_dir, name)

    # Train the model
    train_started_at = time.time()
    print("PROGRESS 45 training_dlib_predictor", file=sys.stderr)
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 45,
                "stage": "training",
                "substage": "fit",
                "message": (
                    f"Training dlib predictor (raw={raw_num_images}, augmented={num_images}, "
                    f"effective≈{effective_training_exposure}, landmarks={num_landmarks}, "
                    f"tree_depth={options.tree_depth}, cascade_depth={options.cascade_depth})"
                ),
                "raw_images": int(raw_num_images),
                "num_images": int(num_images),
                "num_landmarks": int(num_landmarks),
                "effective_training_exposure": int(effective_training_exposure),
                "tree_depth": int(options.tree_depth),
                "cascade_depth": int(options.cascade_depth),
            }
        ),
        file=sys.stderr,
    )
    print("Training shape predictor...", file=sys.stderr)
    dlib.train_shape_predictor(actual_train_xml, predictor_path, options)
    train_elapsed = time.time() - train_started_at
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 82,
                "stage": "training",
                "substage": "fit_done",
                "message": f"dlib fit complete in {int(round(train_elapsed))}s.",
                "fit_elapsed_sec": float(train_elapsed),
            }
        ),
        file=sys.stderr,
    )
    print("MODEL_PATH", predictor_path)

    # ── Evaluate mean error (dlib native) + per-image median ──────────────────
    # Validation alone gates promotion; the permanently locked test set is
    # evaluated only for the final report.
    print("PROGRESS 88 evaluating_train_validation_error", file=sys.stderr)
    train_error = dlib.test_shape_predictor(train_xml, predictor_path)
    print("TRAIN_ERROR", train_error)
    _train_per = _compute_dlib_per_image_errors(train_xml, predictor_path)
    train_error_details = _compute_dlib_per_image_error_details(train_xml, predictor_path)
    train_median_error = float(np.median(_train_per)) if _train_per else float("nan")
    print("TRAIN_MEDIAN_ERROR", round(train_median_error, 6))

    validation_error = None
    validation_mean_normalized_error = None
    validation_median_error = None
    validation_error_details: list[dict] = []
    validation_per_image: list[float] = []
    if os.path.exists(validation_xml) and count_landmarks_in_xml(validation_xml)[0] > 0:
        validation_error = dlib.test_shape_predictor(validation_xml, predictor_path)
        print("VALIDATION_ERROR", validation_error)
        validation_per_image = _compute_dlib_per_image_errors(validation_xml, predictor_path)
        validation_error_details = _compute_dlib_per_image_error_details(
            validation_xml,
            predictor_path,
        )
        validation_mean_normalized_error = (
            float(np.mean(validation_per_image)) if validation_per_image else None
        )
        validation_median_error = (
            float(np.median(validation_per_image)) if validation_per_image else float("nan")
        )
        print("VALIDATION_MEDIAN_ERROR", round(validation_median_error, 6))

    # Report-only frozen-test evaluation.
    #
    # Model selection is validation-only: `lineage` ranks candidates on
    # `validationMedianError`/`validationError` and never reads a test metric,
    # and the stability gate below is computed from validation errors alone.
    # The frozen test cohort is measured here purely so a promoted artifact
    # carries an unbiased generalization number, matching what the CNN and OBB
    # trainers already record.
    test_error = None
    test_median_error = None
    test_mean_normalized_error = None
    test_per_image: list[float] = []
    test_error_details: list[dict] = []
    test_landmark_count = count_landmarks_in_xml(test_xml)[0] if os.path.exists(test_xml) else 0
    if test_landmark_count > 0:
        test_error = dlib.test_shape_predictor(test_xml, predictor_path)
        print("TEST_ERROR", test_error)
        test_per_image = _compute_dlib_per_image_errors(test_xml, predictor_path)
        test_error_details = _compute_dlib_per_image_error_details(test_xml, predictor_path)
        test_mean_normalized_error = (
            float(np.mean(test_per_image)) if test_per_image else None
        )
        test_median_error = (
            float(np.median(test_per_image)) if test_per_image else float("nan")
        )
        print("TEST_MEDIAN_ERROR", round(test_median_error, 6))
        test_evaluation = {
            "status": "measured",
            "role": "report_only",
            "reason": "never_used_for_model_selection",
            "cohort": "frozen_test",
            "sampleCount": len(test_per_image),
            "meanNormalizedError": test_mean_normalized_error,
            "medianNormalizedError": test_median_error,
        }
    else:
        test_evaluation = {
            "status": "unavailable",
            "role": "report_only",
            "reason": (
                "missing_frozen_test_cohort"
                if not os.path.exists(test_xml)
                else "empty_frozen_test_cohort"
            ),
            "cohort": "frozen_test",
            "sampleCount": 0,
        }

    _assert_effective_dlib_dataset_unchanged(role_paths, effective_dataset)

    stability_gate = _dlib_validation_stability_gate(
        validation_per_image,
        validation_mean_normalized_error,
        validation_median_error,
    )
    instability_ratio = stability_gate.get("meanMedianRatio")
    unstable = stability_gate.get("passed") is False
    instability_warning = (
        {
            "code": "high_mean_median_divergence",
            "severity": "warning",
            "message": "Mean error diverges sharply from median error; catastrophic outliers detected.",
            "ratio": float(instability_ratio),
            "stabilityGate": stability_gate,
        }
        if unstable and instability_ratio is not None
        else None
    )
    worst_train_failures = sorted(train_error_details, key=lambda item: item["mean_error"], reverse=True)[:10]
    worst_validation_failures = sorted(
        validation_error_details,
        key=lambda item: item["mean_error"],
        reverse=True,
    )[:10]
    worst_test_failures = sorted(test_error_details, key=lambda item: item["mean_error"], reverse=True)[:10]

    # Save results
    results = {
        "model_id": model_id,
        "artifact_tag": artifact_tag,
        "run_id": run_id,
        "model_path": predictor_path,
        "train_error": train_error,
        "train_median_error": train_median_error,
        "validation_error": validation_error,
        "validation_mean_normalized_error": validation_mean_normalized_error,
        "validation_median_error": validation_median_error,
        "test_error": test_error,
        "test_mean_normalized_error": test_mean_normalized_error,
        "test_median_error": test_median_error,
        "source_counts": params_log["source_counts"],
        "instability_ratio": instability_ratio,
        "unstable": unstable,
        "stability_gate": stability_gate,
        "instability_warning": instability_warning,
        "train_error_details": train_error_details,
        "validation_error_details": validation_error_details,
        "test_error_details": test_error_details,
        "test_evaluation": test_evaluation,
        "num_images": num_images,
        "num_landmarks": num_landmarks,
    }
    results_path = os.path.join(debug_dir, f"training_results_{tag}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}", file=sys.stderr)
    dio.write_run_json(run_dir, "worst_failures.json", {
        "train": worst_train_failures,
        "validation": worst_validation_failures,
        "test": worst_test_failures,
    })
    dio.write_json(
        os.path.join(debug_dir, f"worst_failures_{tag}.json"),
        {
            "tag": tag,
            "run_id": run_id,
            "train": worst_train_failures,
            "validation": worst_validation_failures,
            "test": worst_test_failures,
        },
    )
    dio.write_run_json(
        run_dir,
        "train_results.json",
        {
            **results,
            "model_type": "dlib",
            "tag": tag,
            "run_id": run_id,
            "train_xml": train_xml,
            "validation_xml": validation_xml if os.path.exists(validation_xml) else None,
            "test_xml": test_xml if os.path.exists(test_xml) else None,
            "augmented_xml": actual_train_xml if actual_train_xml != train_xml else None,
        },
    )
    dio.write_run_manifest(
        run_dir,
        model_type="dlib",
        tag=tag,
        project_root=project_root,
        extra={
            "status": "completed",
            "model_id": model_id,
            "artifact_tag": artifact_tag,
            "model_path": predictor_path,
            "train_error": train_error,
            "train_median_error": train_median_error,
            "validation_error": validation_error,
            "validation_median_error": validation_median_error,
            "test_error": test_error,
            "test_median_error": test_median_error,
            "unstable": unstable,
            "instability_warning": instability_warning,
        },
    )

    # Publish immutable metadata, a unique backend tag for this exact run, and a
    # legacy current-name alias.  A display-name rename updates only the registry.
    artifact_manifest_path = os.path.join(artifact_dir, "manifest.json")
    artifact_manifest = {
        "formatVersion": 2,
        "modelId": model_id,
        "modelType": "dlib",
        "predictorType": "dlib",
        "displayName": tag,
        "artifactTag": artifact_tag,
        "runId": run_id,
        "createdAt": lineage.utc_now_iso(),
        "artifact": {
            "path": predictor_path,
            "relativePath": "predictor.dat",
            "sha256": lineage.sha256_file(predictor_path),
        },
        "sidecars": {
            "idMapping": id_mapping_descriptor,
            "effectiveDataset": {
                "format": "biovision.effective-landmark-dataset.v1",
                "path": effective_dataset_path,
                "relativePath": "effective_dataset.json",
                "sha256": lineage.sha256_file(effective_dataset_path),
            },
            "trainingParameters": {
                "format": "biovision.dlib-training-parameters.v1",
                "path": training_protocol_path,
                "relativePath": "training_parameters.json",
                "sha256": lineage.sha256_file(training_protocol_path),
            },
            "validationEvaluatorProtocol": {
                "format": "biovision.landmark-validation-evaluator-protocol.v1",
                "path": validation_evaluator_protocol_path,
                "relativePath": "validation_evaluator_protocol.json",
                "sha256": lineage.sha256_file(validation_evaluator_protocol_path),
                "fingerprint": validation_evaluator_protocol["fingerprint"],
            },
        },
        "metrics": {
            "trainError": train_error,
            "trainMedianError": train_median_error,
            "validationError": validation_error,
            "validationMeanNormalizedError": validation_mean_normalized_error,
            "validationMedianError": validation_median_error,
            "validationP95Error": stability_gate.get("p95Error"),
            "validationMaxError": stability_gate.get("maxError"),
            "stabilityGatePassed": stability_gate.get("passed"),
            "stabilityGate": stability_gate,
            "unstable": unstable,
        },
        "lineage": lineage_payload,
        "validationEvaluatorProtocol": validation_evaluator_protocol,
        "testEvaluation": test_evaluation,
        "promotionPolicy": _dlib_promotion_policy(),
    }
    lineage.atomic_write_json(artifact_manifest_path, artifact_manifest)

    unique_alias_path = os.path.join(modeldir, f"predictor_{artifact_tag}.dat")
    legacy_alias_path = os.path.join(modeldir, f"predictor_{tag}.dat")
    lineage.atomic_copy_file(predictor_path, unique_alias_path)
    for stem in (
        "id_mapping",
        "crop_metadata",
        "box_scale",
        "orientation",
        "training_boxes",
        "split_info",
        "training_params",
    ):
        original_sidecar = os.path.join(debug_dir, f"{stem}_{tag}.json")
        if os.path.isfile(original_sidecar):
            lineage.atomic_copy_file(
                original_sidecar,
                os.path.join(debug_dir, f"{stem}_{artifact_tag}.json"),
            )

    registry_record = lineage.publish_model_run(
        project_root,
        model_type="dlib",
        predictor_type="dlib",
        run_id=run_id,
        display_name=tag,
        artifact_tag=artifact_tag,
        artifact_path=predictor_path,
        legacy_path=unique_alias_path,
        run_manifest_path=artifact_manifest_path,
        current_alias_path=legacy_alias_path,
        metrics=artifact_manifest["metrics"],
        promotion_policy=_dlib_promotion_policy(),
        active_aliases=[(predictor_path, legacy_alias_path)],
    )
    results["registry"] = registry_record
    results["id_mapping_path"] = artifact_id_mapping_path
    print("MODEL_ID", model_id)
    print("MODEL_TAG", artifact_tag)
    print("MODEL_STATUS", registry_record.get("status"))
    print("PROMOTION_JSON", json.dumps(registry_record.get("promotion", {}), sort_keys=True))
    print(f"dlib run debug saved to: {run_dir}", file=sys.stderr)
    print("PROGRESS 99 finalize", file=sys.stderr)
    print(
        "PROGRESS_JSON " + json.dumps(
            {
                "percent": 99,
                "stage": "evaluation",
                "substage": "finalize",
                "message": "Finalizing training artifacts.",
            }
        ),
        file=sys.stderr,
    )

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_shape_model.py <project_root> <tag> [options_json]")
        print("  options_json: Optional JSON string with custom training options.")
        print("                Keys: standard dlib options + 'aug_angles' (list), 'aug_flip' (bool).")
        print("  Example: '{\"aug_angles\": [-15, 15], \"aug_flip\": false}'")
        sys.exit(1)

    project_root = sys.argv[1]
    tag = sys.argv[2]

    custom_options = None
    if len(sys.argv) > 3:
        custom_options = json.loads(sys.argv[3])

    # Pull augmentation params out of custom_options (they're not dlib options)
    aug_angles_arg = None
    aug_flip_arg = None
    if custom_options:
        aug_angles_arg = custom_options.pop("aug_angles", None)
        if "aug_flip" in custom_options:
            aug_flip_arg = bool(custom_options.pop("aug_flip"))
        if not custom_options:
            custom_options = None

    train_shape_model(project_root, tag, custom_options,
                      aug_angles=aug_angles_arg, aug_flip=aug_flip_arg)
