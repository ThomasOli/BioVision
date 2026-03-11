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
import bv_utils.orientation_utils as ou

STANDARD_SIZE = 512
import numpy as np


def _apply_photo_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply aggressive brightness/contrast/saturation jitter for offline dlib augmentation.

    Dlib rotation angles are kept tight (±5°) to preserve mean-shape stability, so the
    augmentation burden shifts to pixel intensities to prevent memorization of lighting
    and slide backgrounds.
    """
    assert img.ndim == 3 and img.shape[2] == 3, "Expected 3-channel BGR image"
    alpha = 1.0 + rng.uniform(-0.25, 0.25)   # contrast multiplier — was ±0.15
    beta  = rng.uniform(-40.0, 40.0)           # brightness offset (pixels) — was ±20
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # Saturation jitter via HSV to combat lab-lighting and slide-background memorization
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.7, 1.3), 0, 255)
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
    mode = str(orientation_mode or "").strip().lower()
    if mode in ("directional", "axial"):
        return [-5, 5]
    return [-30, -15, 15, 30]


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
    rng = np.random.default_rng()

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

        for angle in aug_angles:
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotated = cv2.warpAffine(img, M, (STANDARD_SIZE, STANDARD_SIZE),
                                     borderMode=cv2.BORDER_REPLICATE)
            rotated = _apply_photo_jitter(rotated, rng)
            rot_parts = [
                (name, *_rotate_point(x, y, cx, cy, angle))
                for name, x, y in parts
            ]
            angle_tag = str(angle).replace("-", "m")
            out_path = os.path.join(aug_dir, f"{base}_aug_r{angle_tag}.png")
            cv2.imwrite(out_path, rotated)
            augmented_entries.append((out_path, rot_parts))

            if add_flip:
                # Also include mirrored rotated variants to cover diagonal-right poses.
                rot_flip = cv2.flip(rotated, 1)
                rot_flip = _apply_photo_jitter(rot_flip, rng)
                rot_flip_parts = [
                    (name_swap_map.get(name, name), float(STANDARD_SIZE - 1 - rx), float(ry))
                    for name, rx, ry in rot_parts
                ]
                out_path_rf = os.path.join(aug_dir, f"{base}_aug_r{angle_tag}_flip.png")
                cv2.imwrite(out_path_rf, rot_flip)
                augmented_entries.append((out_path_rf, rot_flip_parts))

        if add_flip:
            flipped = cv2.flip(img, 1)
            flipped = _apply_photo_jitter(flipped, rng)
            flip_parts = [
                (name_swap_map.get(name, name), float(STANDARD_SIZE - 1 - x), float(y))
                for name, x, y in parts
            ]
            out_path = os.path.join(aug_dir, f"{base}_aug_flip.png")
            cv2.imwrite(out_path, flipped)
            augmented_entries.append((out_path, flip_parts))

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
    dio.write_run_manifest(
        run_dir,
        model_type="dlib",
        tag=tag,
        project_root=project_root,
        extra={"status": "started"},
    )

    train_xml = os.path.join(xmldir, f"train_{tag}.xml")
    test_xml = os.path.join(xmldir, f"test_{tag}.xml")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")

    if not os.path.exists(train_xml):
        raise FileNotFoundError(f"Train XML not found at {train_xml}")
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

    orientation_mode = _resolve_orientation_mode(project_root, tag)
    aug_override = ou.load_augmentation_policy(project_root)
    aug_profile = ou.get_schema_augmentation_profile(
        orientation_mode, engine="dlib", augmentation_override=aug_override or None
    )
    if aug_flip is None:
        aug_flip = bool(aug_profile.get("flip", ou.default_allow_flip_augmentation(orientation_mode)))

    # Count raw train samples before augmentation so bucket selection is not
    # biased upward by synthetic/rotated copies.
    raw_num_images, raw_num_landmarks = count_landmarks_in_xml(train_xml)

    # ── Pre-training augmentation ──────────────────────────────────────────────
    # Augmentation Router: model-aware angle selection.
    # Explicit aug_angles override → forwarded verbatim (API compat).
    # None → router selects based on orientation mode:
    #   directional/axial → ±5° (mean-shape stability)
    #   invariant/bilateral → [-30, -15, 15, 30] (no chirality to protect)
    aug_angles = _resolve_dlib_aug_angles(orientation_mode, aug_angles)
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
    print(
        f"Resolved dlib preset bucket: {size_bucket} "
        f"(orientation_mode={orientation_mode}, raw_images={raw_num_images}, effective_images={num_images})",
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
        "raw_num_images": raw_num_images,
        "raw_num_landmarks": raw_num_landmarks,
        "source_counts": {
            "train": raw_num_images,
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
        "oversampling_amount": options.oversampling_amount,
        "oversampling_translation_jitter": options.oversampling_translation_jitter,
        "feature_pool_region_padding": options.feature_pool_region_padding,
        "lambda_param": options.lambda_param,
        "aug_angles": aug_angles,
        "aug_flip": aug_flip,
        "aug_profile": aug_profile,
        "bilateral_name_swaps": name_swap_map,
        "orientation_mode": orientation_mode,
        "augmented_xml": actual_train_xml if actual_train_xml != train_xml else None,
    }
    params_path = os.path.join(debug_dir, f"training_params_{tag}.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_log, f, indent=2)
    print(f"Training parameters saved to: {params_path}", file=sys.stderr)
    dio.write_run_json(run_dir, "train_params.json", params_log)

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
                    f"Training dlib predictor (images={num_images}, landmarks={num_landmarks}, "
                    f"tree_depth={options.tree_depth}, cascade_depth={options.cascade_depth})"
                ),
                "num_images": int(num_images),
                "num_landmarks": int(num_landmarks),
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
    print("PROGRESS 88 evaluating_train_test_error", file=sys.stderr)
    train_error = dlib.test_shape_predictor(train_xml, predictor_path)
    print("TRAIN_ERROR", train_error)
    _train_per = _compute_dlib_per_image_errors(train_xml, predictor_path)
    train_error_details = _compute_dlib_per_image_error_details(train_xml, predictor_path)
    train_median_error = float(np.median(_train_per)) if _train_per else float("nan")
    print("TRAIN_MEDIAN_ERROR", round(train_median_error, 6))

    test_error = None
    test_median_error = None
    test_error_details: list[dict] = []
    if os.path.exists(test_xml):
        test_error = dlib.test_shape_predictor(test_xml, predictor_path)
        print("TEST_ERROR", test_error)
        _test_per = _compute_dlib_per_image_errors(test_xml, predictor_path)
        test_error_details = _compute_dlib_per_image_error_details(test_xml, predictor_path)
        test_median_error = float(np.median(_test_per)) if _test_per else float("nan")
        print("TEST_MEDIAN_ERROR", round(test_median_error, 6))

    instability_ratio = None
    if test_error is not None and test_median_error is not None and test_median_error > 0:
        instability_ratio = float(test_error) / float(test_median_error)
    elif train_median_error and train_median_error > 0:
        instability_ratio = float(train_error) / float(train_median_error)
    unstable = bool(instability_ratio is not None and instability_ratio >= 10.0)
    instability_warning = (
        {
            "code": "high_mean_median_divergence",
            "severity": "warning",
            "message": "Mean error diverges sharply from median error; catastrophic outliers detected.",
            "ratio": float(instability_ratio),
        }
        if unstable and instability_ratio is not None
        else None
    )
    worst_train_failures = sorted(train_error_details, key=lambda item: item["mean_error"], reverse=True)[:10]
    worst_test_failures = sorted(test_error_details, key=lambda item: item["mean_error"], reverse=True)[:10]

    # Save results
    results = {
        "model_path": predictor_path,
        "train_error": train_error,
        "train_median_error": train_median_error,
        "test_error": test_error,
        "test_median_error": test_median_error,
        "source_counts": params_log["source_counts"],
        "instability_ratio": instability_ratio,
        "unstable": unstable,
        "instability_warning": instability_warning,
        "train_error_details": train_error_details,
        "test_error_details": test_error_details,
        "num_images": num_images,
        "num_landmarks": num_landmarks,
    }
    results_path = os.path.join(debug_dir, f"training_results_{tag}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}", file=sys.stderr)
    dio.write_run_json(run_dir, "worst_failures.json", {
        "train": worst_train_failures,
        "test": worst_test_failures,
    })
    dio.write_json(
        os.path.join(debug_dir, f"worst_failures_{tag}.json"),
        {
            "tag": tag,
            "run_id": run_id,
            "train": worst_train_failures,
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
            "model_path": predictor_path,
            "train_error": train_error,
            "train_median_error": train_median_error,
            "test_error": test_error,
            "test_median_error": test_median_error,
            "unstable": unstable,
            "instability_warning": instability_warning,
        },
    )
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
