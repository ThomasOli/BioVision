import json
import os
import glob
import sys
import random
import xml.etree.ElementTree as ET
import cv2
from detect_specimen import detect_specimen
from image_utils import load_image


def json_to_dlib_xml(project_root, tag, test_split=0.2, seed=42, max_image_dim=1500):
    """
    Convert JSON landmark annotations to dlib XML format with train/test split.
    Always uses auto-detected bounding boxes for consistency with inference.
    Uses EXIF-corrected image loading to match browser display orientation.
    Resizes large images to speed up training while maintaining accuracy.

    IMPORTANT: This function saves EXIF-corrected images to a 'corrected_images'
    directory so that dlib reads images with the same orientation that browsers display.

    Args:
        project_root: Root directory of the project
        tag: Model tag/name for output files
        test_split: Fraction of data to use for testing (default 0.2 = 20%)
        seed: Random seed for reproducible splits (default 42)
        max_image_dim: Maximum image dimension (default 1500px for faster training)

    Outputs:
        - train_{tag}.xml: Training data (80% by default)
        - test_{tag}.xml: Test data (20% by default)

    XML Tree format:
    <dataset>
        <images>
            <image file="corrected_images/fossil1.png">
                <box top="..." left="..." width="..." height="...">
                    <part name="0" x="123" y="456" />
                    <part name="1" x="234" y="321" />
                </box>
            </image>
        </images>
    </dataset>
    """
    labels_dir = os.path.join(project_root, "labels")
    xmldir = os.path.join(project_root, "xml")
    images_dir = os.path.join(project_root, "images")
    corrected_images_dir = os.path.join(project_root, "corrected_images")
    debug_dir = os.path.join(project_root, "debug")

    os.makedirs(xmldir, exist_ok=True)
    os.makedirs(corrected_images_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    train_xml_path = os.path.join(xmldir, f"train_{tag}.xml")
    test_xml_path = os.path.join(xmldir, f"test_{tag}.xml")

    # Debug log for bounding boxes
    debug_log = []

    # First pass: process all JSON files and collect image data
    processed_images = []

    json_paths = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
    if not json_paths:
        raise RuntimeError(f"No JSON files found in {labels_dir}")

    for jp in json_paths:
        with open(jp, "r") as f:
            data = json.load(f)

        image_filename = data.get("imageFilename")
        image_path = os.path.join(images_dir, image_filename) if image_filename else None

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found for {jp}")

        # Load image with EXIF correction
        img, img_w, img_h = load_image(image_path)
        if img is None:
            raise RuntimeError(f"Could not read image {image_path}")

        # Resize large images to speed up training
        scale = 1.0
        if max(img_w, img_h) > max_image_dim:
            scale = max_image_dim / max(img_w, img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized {image_filename}: {img_w}x{img_h} -> {new_w}x{new_h} (scale={scale:.3f})", file=sys.stderr)
            img_w, img_h = new_w, new_h

        # Save EXIF-corrected (and possibly resized) image for dlib to use
        # Use PNG format to avoid JPEG re-compression artifacts
        base_name = os.path.splitext(image_filename)[0]
        corrected_filename = f"{base_name}.png"
        corrected_path = os.path.join(corrected_images_dir, corrected_filename)
        cv2.imwrite(corrected_path, img)

        # Auto-detect specimen bounds on corrected image for consistency
        detected = detect_specimen(corrected_path, margin=20)
        if detected is None:
            detected = {'left': 0, 'top': 0, 'width': img_w, 'height': img_h, 'right': img_w, 'bottom': img_h}

        # Log debug info
        debug_log.append({
            "filename": image_filename,
            "corrected_path": corrected_path,
            "original_dimensions": {"width": img_w / scale if scale != 1.0 else img_w,
                                    "height": img_h / scale if scale != 1.0 else img_h},
            "resized_dimensions": {"width": img_w, "height": img_h},
            "scale": scale,
            "detected_box": detected
        })

        # Get landmarks from JSON (supports both old and new formats)
        if "boxes" in data and data["boxes"]:
            # New format: collect all landmarks from all boxes
            all_landmarks = []
            for box_data in data["boxes"]:
                all_landmarks.extend(box_data.get("landmarks", []))
        else:
            # Old format: flat landmarks array
            all_landmarks = data.get("landmarks", [])

        if not all_landmarks:
            continue  # Skip images with no landmarks

        # Filter out skipped landmarks AND landmarks with invalid coordinates
        valid_landmarks = [
            lm for lm in all_landmarks
            if not lm.get("isSkipped", False)
            and lm.get("x", -1) >= 0
            and lm.get("y", -1) >= 0
        ]

        if not valid_landmarks:
            continue

        # Scale landmark coordinates if image was resized
        if scale != 1.0:
            valid_landmarks = [
                {**lm, "x": lm["x"] * scale, "y": lm["y"] * scale}
                for lm in valid_landmarks
            ]

        # Store processed data for later splitting
        processed_images.append({
            "corrected_path": corrected_path,
            "box": detected,
            "landmarks": valid_landmarks,
            "scale": scale
        })

    if not processed_images:
        raise RuntimeError("No valid images with landmarks found")

    # CRITICAL: dlib requires ALL images to have the SAME landmarks
    # Find the COMMON set of landmark IDs present in ALL images
    landmark_ids_per_image = []
    for img_data in processed_images:
        img_ids = set(lm.get("id", 0) for lm in img_data["landmarks"])
        landmark_ids_per_image.append(img_ids)

    # Intersection of all landmark sets = common landmarks
    common_landmark_ids = landmark_ids_per_image[0]
    for img_ids in landmark_ids_per_image[1:]:
        common_landmark_ids = common_landmark_ids.intersection(img_ids)

    # Union of all = all possible landmarks
    all_landmark_ids = set()
    for img_ids in landmark_ids_per_image:
        all_landmark_ids = all_landmark_ids.union(img_ids)

    # Warn about excluded landmarks
    excluded_ids = all_landmark_ids - common_landmark_ids
    if excluded_ids:
        print(f"WARNING: Excluding landmarks {sorted(excluded_ids)} - not present in all images", file=sys.stderr)
        print(f"  Training will use {len(common_landmark_ids)} common landmarks: {sorted(common_landmark_ids)}", file=sys.stderr)

    if not common_landmark_ids:
        raise RuntimeError("No common landmarks found across all images!")

    # Filter each image's landmarks to only include common ones
    for img_data in processed_images:
        img_data["landmarks"] = [
            lm for lm in img_data["landmarks"]
            if lm.get("id", 0) in common_landmark_ids
        ]

    # Shuffle and split into train/test sets
    random.seed(seed)
    random.shuffle(processed_images)

    n_test = max(1, int(len(processed_images) * test_split))
    # Ensure at least 1 training image
    if len(processed_images) - n_test < 1:
        n_test = len(processed_images) - 1

    test_images = processed_images[:n_test]
    train_images = processed_images[n_test:]

    # If only 1 image, use it for both train and test
    if len(processed_images) == 1:
        train_images = processed_images
        test_images = processed_images

    print(f"Split: {len(train_images)} training, {len(test_images)} test images", file=sys.stderr)

    # Build a consistent ID mapping from common landmarks only
    # dlib requires sequential part names (0, 1, 2, ...), but we want to preserve original IDs
    # We'll create a mapping: dlib_index -> original_id
    sorted_original_ids = sorted(common_landmark_ids)
    id_mapping = {idx: orig_id for idx, orig_id in enumerate(sorted_original_ids)}
    reverse_mapping = {orig_id: idx for idx, orig_id in enumerate(sorted_original_ids)}

    print(f"Landmark ID mapping: {len(id_mapping)} landmarks", file=sys.stderr)
    print(f"  Original IDs: {sorted_original_ids}", file=sys.stderr)

    # Helper function to create XML from image list
    def create_xml(image_list, output_path):
        dataset = ET.Element("dataset")
        images_el = ET.SubElement(dataset, "images")

        for img_data in image_list:
            image_el = ET.SubElement(images_el, "image", file=img_data["corrected_path"])

            box = img_data["box"]
            box_el = ET.SubElement(image_el, "box",
                                   top=str(box['top']), left=str(box['left']),
                                   width=str(box['width']), height=str(box['height']))

            # Sort landmarks by original ID and assign sequential dlib indices
            for lm in sorted(img_data["landmarks"], key=lambda lm: lm.get("id", 0)):
                orig_id = lm.get("id", 0)
                dlib_idx = reverse_mapping[orig_id]
                ET.SubElement(box_el, "part",
                              name=str(dlib_idx),
                              x=str(int(lm["x"])),
                              y=str(int(lm["y"])))

        tree = ET.ElementTree(dataset)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    # Create train and test XML files
    create_xml(train_images, train_xml_path)
    create_xml(test_images, test_xml_path)

    # Save ID mapping and training config for inference
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")
    with open(id_mapping_path, "w") as f:
        json.dump({
            "dlib_to_original": id_mapping,
            "original_to_dlib": reverse_mapping,
            "original_ids": sorted_original_ids,
            "excluded_ids": sorted(excluded_ids) if excluded_ids else [],
            "all_ids_in_dataset": sorted(all_landmark_ids),
            "training_config": {
                "max_image_dim": max_image_dim,
                "image_scales": {img["corrected_path"]: img["scale"] for img in processed_images}
            }
        }, f, indent=2)
    print(f"ID mapping saved to: {id_mapping_path}", file=sys.stderr)

    # Save debug log
    debug_log_path = os.path.join(debug_dir, f"training_boxes_{tag}.json")
    with open(debug_log_path, "w") as f:
        json.dump(debug_log, f, indent=2)

    # Save split info
    split_info = {
        "train_count": len(train_images),
        "test_count": len(test_images),
        "test_split": test_split,
        "seed": seed,
        "train_files": [img["corrected_path"] for img in train_images],
        "test_files": [img["corrected_path"] for img in test_images]
    }
    split_info_path = os.path.join(debug_dir, f"split_info_{tag}.json")
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    # Print paths (train path first for backward compatibility)
    print(train_xml_path)
    print(test_xml_path)
    print(f"Debug log saved to: {debug_log_path}", file=sys.stderr)
    print(f"Split info saved to: {split_info_path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_dataset.py <project_root> <tag> [test_split] [seed]")
        sys.exit(1)

    project_root = sys.argv[1]
    tag = sys.argv[2]
    test_split = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    json_to_dlib_xml(project_root, tag, test_split=test_split, seed=seed)
