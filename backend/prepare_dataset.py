import json
import os
import glob
import sys
import random
import xml.etree.ElementTree as ET
import cv2
from detect_specimen import detect_specimen
from image_utils import load_image


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

        # Get landmarks from JSON
        if "boxes" in data and data["boxes"]:
            all_lm = [lm for box in data["boxes"] for lm in box.get("landmarks", [])]
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
            "original_orientation": orig_orientation,
            "target_orientation": target_orientation,
            "was_mirrored": was_mirrored
        })

        processed.append({
            "path": corrected_path,
            "box": detected,
            "landmarks": valid_lm,
            "scale": scale,
            "filename": img_filename,
            "orientation": orig_orientation,
            "mirrored": was_mirrored
        })

    if not processed:
        raise RuntimeError("No valid images with landmarks found")

    # Find common landmarks across all images
    id_sets = [set(lm.get("id", 0) for lm in p["landmarks"]) for p in processed]
    common = id_sets[0]
    for s in id_sets[1:]:
        common &= s
    excluded = set().union(*id_sets) - common

    if excluded:
        print(f"Excluding landmark IDs {sorted(excluded)}, using {len(common)} common landmarks: {sorted(common)}", file=sys.stderr)
    if not common:
        raise RuntimeError("No common landmarks across all images")

    # Filter to only common landmarks
    for p in processed:
        p["landmarks"] = [lm for lm in p["landmarks"] if lm.get("id", 0) in common]

    # Split into train/test sets
    random.seed(seed)
    random.shuffle(processed)
    n_test = max(1, int(len(processed) * test_split))
    if len(processed) - n_test < 1:
        n_test = len(processed) - 1
    test_imgs, train_imgs = processed[:n_test], processed[n_test:]
    if len(processed) == 1:
        train_imgs = test_imgs = processed

    # Create ID mapping (dlib requires sequential 0-based IDs)
    sorted_ids = sorted(common)
    id_map = {i: orig for i, orig in enumerate(sorted_ids)}
    rev_map = {orig: i for i, orig in enumerate(sorted_ids)}

    def write_xml(imgs, path):
        root = ET.Element("dataset")
        images = ET.SubElement(root, "images")
        for p in imgs:
            img_el = ET.SubElement(images, "image", file=p["path"])
            box = p["box"]
            box_el = ET.SubElement(img_el, "box", top=str(box['top']), left=str(box['left']),
                                   width=str(box['width']), height=str(box['height']))
            for lm in sorted(p["landmarks"], key=lambda x: x.get("id", 0)):
                ET.SubElement(box_el, "part", name=str(rev_map[lm["id"]]), x=str(int(lm["x"])), y=str(int(lm["y"])))
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    train_path = os.path.join(xml_dir, f"train_{tag}.xml")
    test_path = os.path.join(xml_dir, f"test_{tag}.xml")
    write_xml(train_imgs, train_path)
    write_xml(test_imgs, test_path)

    # Save debug files
    with open(os.path.join(debug_dir, f"id_mapping_{tag}.json"), "w") as f:
        json.dump({
            "dlib_to_original": id_map,
            "original_to_dlib": rev_map,
            "original_ids": sorted_ids,
            "excluded_ids": sorted(excluded) if excluded else [],
            "num_landmarks": len(sorted_ids),
            "training_config": {
                "max_dim": max_dim,
                "test_split": test_split,
                "seed": seed,
                "target_orientation": target_orientation
            }
        }, f, indent=2)

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
            "train": len(train_imgs),
            "test": len(test_imgs),
            "train_files": [p["path"] for p in train_imgs],
            "test_files": [p["path"] for p in test_imgs]
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
