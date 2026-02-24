"""
Debug script to trace the complete landmark pipeline and identify mismatches.
Run this after training to compare annotation vs training vs inference coordinates.

Uses the same crop-standardized pipeline as predict.py (512x512 tight crop).
"""
import os
import sys
import json
import glob
import math
import dlib
import cv2
from image_utils import load_image, safe_imread, safe_imwrite
from detect_specimen import detect_specimen

STANDARD_SIZE = 512


def standardize_crop(image, box, pad_ratio=0.20):
    """Crop image to bounding box + padding, resize to STANDARD_SIZE x STANDARD_SIZE."""
    h, w = image.shape[:2]
    bx, by = int(box['left']), int(box['top'])
    bw, bh = int(box['width']), int(box['height'])

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    cx1 = max(0, bx - pad_x)
    cy1 = max(0, by - pad_y)
    cx2 = min(w, bx + bw + pad_x)
    cy2 = min(h, by + bh + pad_y)

    crop = image[cy1:cy2, cx1:cx2]
    crop_h, crop_w = crop.shape[:2]

    if crop_h == 0 or crop_w == 0:
        crop = image
        cx1, cy1 = 0, 0
        crop_w, crop_h = w, h

    scale_x = STANDARD_SIZE / crop_w
    scale_y = STANDARD_SIZE / crop_h
    standardized = cv2.resize(crop, (STANDARD_SIZE, STANDARD_SIZE), interpolation=cv2.INTER_LINEAR)
    return standardized, (cx1, cy1, scale_x, scale_y)


def map_to_original(x_512, y_512, cx1, cy1, scale_x, scale_y, image_scale=1.0):
    """Map a 512x512 prediction point back to original image coordinates."""
    x_crop = x_512 / scale_x
    y_crop = y_512 / scale_y
    x_img = x_crop + cx1
    y_img = y_crop + cy1
    if image_scale != 1.0:
        x_img /= image_scale
        y_img /= image_scale
    return int(round(x_img)), int(round(y_img))


def debug_pipeline(project_root, tag, test_image_path=None):
    """
    Debug the entire pipeline by comparing:
    1. Original annotation coordinates from JSON
    2. Training bounding box and coordinates
    3. Inference bounding box and predicted coordinates (crop-standardized)
    """
    print("=" * 60)
    print("BIOVISION PIPELINE DEBUG")
    print("=" * 60)

    labels_dir = os.path.join(project_root, "labels")
    images_dir = os.path.join(project_root, "images")
    corrected_dir = os.path.join(project_root, "corrected_images")
    models_dir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    predictor_path = os.path.join(models_dir, f"predictor_{tag}.dat")

    # Load ID mapping if available
    id_mapping = None
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, encoding='utf-8') as f:
            data = json.load(f)
            id_mapping = {int(k): v for k, v in data.get("dlib_to_original", {}).items()}

    json_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
    if not json_files:
        print("ERROR: No JSON label files found!")
        return

    print(f"\nFound {len(json_files)} label file(s)")
    if not os.path.exists(predictor_path):
        print(f"ERROR: Model not found at {predictor_path}")
        return

    predictor = dlib.shape_predictor(predictor_path)
    max_dim = 1500

    for jp in json_files:
        print("\n" + "=" * 60)
        with open(jp, "r", encoding='utf-8') as f:
            data = json.load(f)

        image_filename = data.get("imageFilename")
        print(f"Image: {image_filename}")

        # 1. ANNOTATION DATA
        print("\n[1] ANNOTATION DATA (from JSON)")
        print("-" * 40)

        if "boxes" in data and data["boxes"]:
            all_landmarks = []
            for box_data in data["boxes"]:
                all_landmarks.extend(box_data.get("landmarks", []))
        else:
            all_landmarks = data.get("landmarks", [])

        valid_landmarks = [lm for lm in all_landmarks if not lm.get("isSkipped", False)]
        print(f"  Landmark count: {len(valid_landmarks)}")
        for lm in sorted(valid_landmarks, key=lambda x: x.get("id", 0)):
            print(f"    Landmark {lm.get('id', '?')}: ({lm['x']:.1f}, {lm['y']:.1f})")

        # 2. ORIGINAL IMAGE INFO
        print("\n[2] ORIGINAL IMAGE")
        print("-" * 40)
        original_path = os.path.join(images_dir, image_filename)
        if os.path.exists(original_path):
            raw_img = safe_imread(original_path)
            if raw_img is not None:
                raw_h, raw_w = raw_img.shape[:2]
                print(f"  Raw dimensions (cv2): {raw_w} x {raw_h}")

            exif_img, exif_w, exif_h = load_image(original_path)
            if exif_img is not None:
                print(f"  EXIF-corrected dimensions: {exif_w} x {exif_h}")
                if raw_img is not None and (raw_w != exif_w or raw_h != exif_h):
                    print(f"  WARNING: EXIF ROTATION DETECTED! Dimensions changed.")
        else:
            print(f"  ERROR: Image not found at {original_path}")

        # 3. CORRECTED IMAGE (used for training)
        print("\n[3] CORRECTED IMAGE (for training)")
        print("-" * 40)
        base_name = os.path.splitext(image_filename)[0]
        corrected_path = os.path.join(corrected_dir, f"{base_name}.png")
        train_box = None
        if os.path.exists(corrected_path):
            corrected_img = safe_imread(corrected_path)
            if corrected_img is not None:
                corr_h, corr_w = corrected_img.shape[:2]
                print(f"  Corrected dimensions: {corr_w} x {corr_h}")
                train_box = detect_specimen(corrected_path, margin=20)
                print(f"  Training bounding box:")
                print(f"    left={train_box['left']}, top={train_box['top']}")
                print(f"    right={train_box['right']}, bottom={train_box['bottom']}")
                print(f"    size={train_box['width']} x {train_box['height']}")
        else:
            print(f"  WARNING: Corrected image not found. Run training first.")

        # 4. INFERENCE TEST (using same crop-standardized pipeline as predict.py)
        print("\n[4] INFERENCE TEST (crop-standardized, matching predict.py)")
        print("-" * 40)

        test_path = test_image_path or original_path
        if not os.path.exists(test_path):
            print(f"  ERROR: Test image not found: {test_path}")
            continue

        inf_img, orig_w, orig_h = load_image(test_path)
        if inf_img is None:
            print(f"  ERROR: Could not load test image")
            continue

        print(f"  Original dimensions: {orig_w} x {orig_h}")

        # Downscale if too large (same as predict.py)
        scale = 1.0
        w, h = orig_w, orig_h
        if max(orig_w, orig_h) > max_dim:
            scale = max_dim / max(orig_w, orig_h)
            w, h = int(orig_w * scale), int(orig_h * scale)
            inf_img = cv2.resize(inf_img, (w, h), interpolation=cv2.INTER_AREA)
            print(f"  Downscaled to: {w} x {h} (scale={scale:.3f})")

        inf_box = detect_specimen(test_path if scale == 1.0 else None, margin=20)
        # For scaled images, detect on the resized array directly
        if scale != 1.0:
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
            os.close(tmp_fd)
            safe_imwrite(tmp_path, inf_img)
            inf_box = detect_specimen(tmp_path, margin=20)
            os.remove(tmp_path)

        if inf_box is None:
            inf_box = {'left': 0, 'top': 0, 'right': w, 'bottom': h, 'width': w, 'height': h}

        print(f"  Detection box: left={inf_box['left']}, top={inf_box['top']}, "
              f"right={inf_box['right']}, bottom={inf_box['bottom']}")

        # Crop and standardize to 512x512 (same as predict.py)
        cropped, (cx1, cy1, sx, sy) = standardize_crop(inf_img, inf_box)
        print(f"  Crop origin: ({cx1}, {cy1}), scale: ({sx:.3f}, {sy:.3f})")

        # Run predictor on standardized 512x512 image
        rect = dlib.rectangle(0, 0, STANDARD_SIZE, STANDARD_SIZE)
        shape = predictor(cropped, rect)

        print(f"\n  Predicted landmarks (mapped back to original coordinates):")
        for i in range(shape.num_parts):
            part = shape.part(i)
            orig_id = id_mapping.get(i, i) if id_mapping else i
            x_orig, y_orig = map_to_original(part.x, part.y, cx1, cy1, sx, sy, scale)

            # Find closest annotation landmark by ID
            ann_lm = next((lm for lm in valid_landmarks if lm.get("id") == orig_id), None)
            if ann_lm:
                dist = math.sqrt((x_orig - ann_lm['x'])**2 + (y_orig - ann_lm['y'])**2)
                status = "OK" if dist < 15 else "!!"
                print(f"    [{status}] Landmark {orig_id}: "
                      f"predicted=({x_orig}, {y_orig}), "
                      f"annotated=({ann_lm['x']:.0f}, {ann_lm['y']:.0f}), "
                      f"error={dist:.1f}px")
            else:
                print(f"    [??] Landmark {orig_id}: predicted=({x_orig}, {y_orig}), no annotation")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug_pipeline.py <project_root> <model_tag> [test_image_path]")
        sys.exit(1)

    project_root = sys.argv[1]
    tag = sys.argv[2]
    test_image = sys.argv[3] if len(sys.argv) > 3 else None

    debug_pipeline(project_root, tag, test_image)
