import json
import os
import glob
import sys
import random
import xml.etree.ElementTree as ET
import cv2
from detect_specimen import detect_specimen
from image_utils import load_image

STANDARD_SIZE = 512


def standardize_crop(image, box, landmarks, pad_ratio=0.20):
    """
    Crop image to bounding box + padding, resize to STANDARD_SIZE x STANDARD_SIZE,
    and remap landmarks to the new coordinate space.
    Returns (cropped_image, remapped_landmarks, crop_metadata).
    """
    h, w = image.shape[:2]
    bx, by = int(box['left']), int(box['top'])
    bw, bh = int(box['width']), int(box['height'])

    # Pad around bounding box
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    cx1 = max(0, bx - pad_x)
    cy1 = max(0, by - pad_y)
    cx2 = min(w, bx + bw + pad_x)
    cy2 = min(h, by + bh + pad_y)

    crop = image[cy1:cy2, cx1:cx2]
    crop_h, crop_w = crop.shape[:2]

    if crop_h == 0 or crop_w == 0:
        # Fallback: use full image
        crop = image
        cx1, cy1 = 0, 0
        crop_w, crop_h = w, h

    # Scale to STANDARD_SIZE
    scale_x = STANDARD_SIZE / crop_w
    scale_y = STANDARD_SIZE / crop_h
    standardized = cv2.resize(crop, (STANDARD_SIZE, STANDARD_SIZE), interpolation=cv2.INTER_LINEAR)

    # Remap landmarks to cropped+resized coordinates
    remapped = []
    for lm in landmarks:
        nx = (lm['x'] - cx1) * scale_x
        ny = (lm['y'] - cy1) * scale_y
        remapped.append({**lm, 'x': nx, 'y': ny})

    metadata = {
        "crop_origin": [cx1, cy1],
        "crop_size": [crop_w, crop_h],
        "scale_x": scale_x,
        "scale_y": scale_y,
        "original_box": {"left": bx, "top": by, "width": bw, "height": bh},
    }

    return standardized, remapped, metadata


def detect_orientation(landmarks, head_id=0):
    """
    Detect specimen orientation using head landmark position.
    Compares head landmark (id=0, snout tip) to body centroid.
    Returns 'left' if head is on left side, 'right' if on right side.
    """
    valid = [lm for lm in landmarks if lm.get('x', -1) >= 0 and lm.get('y', -1) >= 0]
    if len(valid) < 2:
        return None
    head = next((lm for lm in valid if lm.get('id') == head_id), None)
    if not head:
        return None
    others = [lm for lm in valid if lm.get('id') != head_id]
    if not others:
        return None
    centroid_x = sum(lm['x'] for lm in others) / len(others)
    return 'left' if head['x'] < centroid_x else 'right'


def mirror_landmarks(landmarks, width):
    """Mirror landmark x-coordinates horizontally."""
    return [{**lm, 'x': width - lm['x'] if lm.get('x', -1) >= 0 else lm['x']} for lm in landmarks]


def json_to_dlib_xml(project_root, tag, test_split=0.2, seed=42, max_dim=1500,
                     target_orientation='left'):
    """
    Convert JSON annotations to dlib XML format for training.

    All images are normalized to target_orientation (default 'left').
    This ensures consistent landmark positions for training.

    At inference time, input images should match the target orientation,
    or landmarks should be mirrored after prediction.
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

    debug_log, orientation_log, processed = [], [], []

    json_paths = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
    if not json_paths:
        raise RuntimeError(f"No JSON files in {labels_dir}")

    for jp in json_paths:
        with open(jp) as f:
            data = json.load(f)

        img_filename = data.get("imageFilename")
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img, w, h = load_image(img_path)
        if img is None:
            raise RuntimeError(f"Could not read: {img_path}")

        scale = 1.0
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            w, h = img.shape[1], img.shape[0]

        base = os.path.splitext(img_filename)[0]
        corrected_path = os.path.join(corrected_dir, f"{base}.png")
        cv2.imwrite(corrected_path, img)

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

                # Detect and normalize orientation per box
                orig_orientation = detect_orientation(valid_lm)
                was_mirrored = False
                if orig_orientation and orig_orientation != target_orientation:
                    valid_lm = mirror_landmarks(valid_lm, w)
                    # Also mirror the box
                    box_coords["left"] = w - box_coords["left"] - box_coords["width"]
                    was_mirrored = True

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
                    "num_boxes": len(image_boxes),
                    "boxes": [{"orientation": b["orientation"], "mirrored": b["mirrored"]} for b in image_boxes]
                })
                processed.append({
                    "path": corrected_path,
                    "boxes": image_boxes,
                    "scale": scale,
                    "filename": img_filename,
                    "multi_specimen": True
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
            orig_orientation = detect_orientation(valid_lm)
            was_mirrored = False

            if orig_orientation and orig_orientation != target_orientation:
                valid_lm = mirror_landmarks(valid_lm, w)
                was_mirrored = True

            orientation_log.append({
                "filename": img_filename,
                "mode": "single-specimen",
                "original_orientation": orig_orientation,
                "target_orientation": target_orientation,
                "was_mirrored": was_mirrored
            })

            processed.append({
                "path": corrected_path,
                "boxes": [{"box": detected, "landmarks": valid_lm, "orientation": orig_orientation, "mirrored": was_mirrored}],
                "scale": scale,
                "filename": img_filename,
                "multi_specimen": False
            })

    if not processed:
        raise RuntimeError("No valid images with landmarks found")

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
        full_img = cv2.imread(p["path"])
        if full_img is None:
            continue
        base = os.path.splitext(os.path.basename(p["path"]))[0]

        for bi, box_data in enumerate(p["boxes"]):
            box = box_data["box"]
            landmarks = box_data["landmarks"]

            cropped_img, remapped_lm, meta = standardize_crop(full_img, box, landmarks)

            # Save cropped image — one per box
            suffix = f"_box{bi}" if len(p["boxes"]) > 1 else ""
            crop_path = os.path.join(corrected_dir, f"{base}{suffix}_crop.png")
            cv2.imwrite(crop_path, cropped_img)

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
                **meta,
            })

    if not standardized_entries:
        raise RuntimeError("No valid standardized crops produced")

    # Split into train/test sets (at crop level)
    random.seed(seed)
    random.shuffle(standardized_entries)
    n_test = max(1, int(len(standardized_entries) * test_split))
    if len(standardized_entries) - n_test < 1:
        n_test = len(standardized_entries) - 1
    test_entries, train_entries = standardized_entries[:n_test], standardized_entries[n_test:]
    if len(standardized_entries) == 1:
        train_entries = test_entries = standardized_entries

    # Create ID mapping (dlib requires sequential 0-based IDs)
    sorted_ids = sorted(common)
    id_map = {i: orig for i, orig in enumerate(sorted_ids)}
    rev_map = {orig: i for i, orig in enumerate(sorted_ids)}

    def write_xml(entries, path):
        root = ET.Element("dataset")
        images = ET.SubElement(root, "images")
        for entry in entries:
            img_el = ET.SubElement(images, "image", file=entry["path"])
            # Full-image box since the crop IS the specimen
            box_el = ET.SubElement(img_el, "box", top="0", left="0",
                                   width=str(STANDARD_SIZE), height=str(STANDARD_SIZE))
            for lm in sorted(entry["landmarks"], key=lambda x: x.get("id", 0)):
                ET.SubElement(box_el, "part", name=str(rev_map[lm["id"]]),
                              x=str(int(lm["x"])), y=str(int(lm["y"])))
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    train_path = os.path.join(xml_dir, f"train_{tag}.xml")
    test_path = os.path.join(xml_dir, f"test_{tag}.xml")
    write_xml(train_entries, train_path)
    write_xml(test_entries, test_path)

    # Save debug files
    with open(os.path.join(debug_dir, f"id_mapping_{tag}.json"), "w") as f:
        json.dump({
            "dlib_to_original": id_map,
            "original_to_dlib": rev_map,
            "original_ids": sorted_ids,
            "excluded_ids": sorted(excluded) if excluded else [],
            "num_landmarks": len(sorted_ids),
            "standard_size": STANDARD_SIZE,
            "training_config": {
                "max_dim": max_dim,
                "test_split": test_split,
                "seed": seed,
                "target_orientation": target_orientation
            }
        }, f, indent=2)

    with open(os.path.join(debug_dir, f"crop_metadata_{tag}.json"), "w") as f:
        json.dump(crop_metadata_log, f, indent=2)

    with open(os.path.join(debug_dir, f"orientation_{tag}.json"), "w") as f:
        json.dump({
            "target_orientation": target_orientation,
            "images": orientation_log,
            "summary": {
                "total": len(orientation_log),
                "mirrored": sum(1 for o in orientation_log if o["was_mirrored"]),
                "left_facing": sum(1 for o in orientation_log if o["original_orientation"] == "left"),
                "right_facing": sum(1 for o in orientation_log if o["original_orientation"] == "right")
            }
        }, f, indent=2)

    with open(os.path.join(debug_dir, f"training_boxes_{tag}.json"), "w") as f:
        json.dump(debug_log, f, indent=2)

    with open(os.path.join(debug_dir, f"split_info_{tag}.json"), "w") as f:
        json.dump({
            "train_crops": len(train_entries),
            "test_crops": len(test_entries),
            "total_crops": len(standardized_entries),
            "standard_size": STANDARD_SIZE,
            "train_files": [e["path"] for e in train_entries],
            "test_files": [e["path"] for e in test_entries]
        }, f, indent=2)

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
