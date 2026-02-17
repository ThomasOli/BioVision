#!/usr/bin/env python3
"""
Export BioVision session annotations to YOLO training format.

Reads session label JSONs and converts bounding boxes to YOLO normalized format.
Creates train/val split with dataset.yaml for ultralytics training.
"""

import os
import sys
import json
import random
import shutil

import cv2


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _safe_int(value, default=0):
    try:
        return int(round(float(value)))
    except Exception:
        return default


def export_dataset(session_dir, class_name, val_ratio=0.2, seed=42, return_details=False):
    """
    Export session annotations to YOLO format.

    Args:
        session_dir: Path to session directory (e.g., sessions/<speciesId>/)
        class_name: Name of the object class (e.g., "fish")
        val_ratio: Fraction of images for validation (default 0.2)
        seed: Random seed for reproducible splits

    Returns:
        Path to the generated dataset.yaml
    """
    labels_dir = os.path.join(session_dir, "labels")
    images_dir = os.path.join(session_dir, "images")

    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Collect all label files:
    # - positive samples from accepted boxes
    # - hard-negative crops from user-rejected auto boxes
    positives = []
    negatives = []
    for fname in os.listdir(labels_dir):
        if not fname.endswith(".json"):
            continue

        label_path = os.path.join(labels_dir, fname)
        with open(label_path, "r") as f:
            data = json.load(f)

        boxes = data.get("boxes", [])
        rejected = data.get("rejectedDetections", [])
        has_boxes = bool(boxes)
        has_rejected = bool(rejected)
        if not has_boxes and not has_rejected:
            continue

        image_filename = data.get("imageFilename", "")
        if not image_filename:
            continue

        # Find corresponding image file
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            # Try common extensions
            base = os.path.splitext(image_filename)[0]
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                candidate = os.path.join(images_dir, base + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    image_filename = base + ext
                    break
            else:
                continue

        sample_base = {
            "image_path": image_path,
            "image_filename": image_filename,
        }
        if has_boxes:
            positives.append({
                **sample_base,
                "boxes": boxes,
            })
        if has_rejected:
            negatives.append({
                **sample_base,
                "rejected": rejected,
            })

    if not positives and not negatives:
        raise ValueError(f"No annotated images found in {labels_dir}")

    # Load dimensions (and image matrix for negatives)
    image_cache = {}
    for sample in positives + negatives:
        if sample["image_path"] in image_cache:
            continue
        img = cv2.imread(sample["image_path"])
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

    # Train/val split over all exported records (positives + negative crops)
    random.seed(seed)
    export_records = []
    for p in positives:
        export_records.append({"kind": "positive", "sample": p})
    for n in negatives:
        for i, rej in enumerate(n["rejected"]):
            export_records.append({"kind": "negative", "sample": n, "rejected_index": i, "rejected_box": rej})

    indices = list(range(len(export_records)))
    random.shuffle(indices)
    val_count = max(1, int(len(export_records) * val_ratio))
    val_indices = set(indices[:val_count])

    # Create output directory structure
    out_dir = os.path.join(session_dir, "yolo_dataset")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    # Export each record
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
            # Copy image to dataset
            dest_img_name = sample["image_filename"]
            dest_img = os.path.join(out_dir, "images", split, dest_img_name)
            if not os.path.exists(dest_img):
                shutil.copy2(sample["image_path"], dest_img)

            # Write YOLO label file
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

                # Convert to YOLO normalized format: class_id x_center y_center w h
                x_center = (left + width / 2.0) / img_w
                y_center = (top + height / 2.0) / img_h
                norm_w = width / img_w
                norm_h = height / img_h

                # Clamp to [0, 1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

            with open(label_path, "w") as f:
                f.write("\n".join(lines) + "\n" if lines else "")
            num_positive_images += 1
            continue

        # Hard-negative crop export: crop around rejected auto-detection and write empty label file.
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
        cv2.imwrite(dest_img, crop)

        label_basename = os.path.splitext(dest_img_name)[0] + ".txt"
        label_path = os.path.join(out_dir, "labels", split, label_basename)
        with open(label_path, "w") as f:
            f.write("")
        num_negative_crops += 1

    # Write dataset.yaml
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    yaml_content = (
        f"path: {out_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    stats = {
        "yaml_path": yaml_path,
        "total_records": len(export_records),
        "val_records": val_count,
        "train_records": max(0, len(export_records) - val_count),
        "positive_images": num_positive_images,
        "negative_crops": num_negative_crops,
    }
    if return_details:
        return stats
    return yaml_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <session_dir> <class_name>", file=sys.stderr)
        sys.exit(1)

    session_dir = sys.argv[1]
    class_name = sys.argv[2]

    details = export_dataset(session_dir, class_name, return_details=True)
    print(json.dumps({"ok": True, **details}))
