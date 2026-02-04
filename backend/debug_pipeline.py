"""
Debug script to trace the complete landmark pipeline and identify mismatches.
Run this after training to compare annotation vs training vs inference coordinates.
"""
import os
import sys
import json
import glob
import cv2
import dlib
from image_utils import load_image
from detect_specimen import detect_specimen


def debug_pipeline(project_root, tag, test_image_path=None):
    """
    Debug the entire pipeline by comparing:
    1. Original annotation coordinates from JSON
    2. Training bounding box and coordinates
    3. Inference bounding box and predicted coordinates
    """
    print("=" * 60)
    print("BIOVISION PIPELINE DEBUG")
    print("=" * 60)

    labels_dir = os.path.join(project_root, "labels")
    images_dir = os.path.join(project_root, "images")
    corrected_dir = os.path.join(project_root, "corrected_images")
    models_dir = os.path.join(project_root, "models")
    predictor_path = os.path.join(models_dir, f"predictor_{tag}.dat")

    # Find JSON files
    json_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))
    if not json_files:
        print("ERROR: No JSON label files found!")
        return

    print(f"\nFound {len(json_files)} label file(s)")

    for jp in json_files:
        print("\n" + "=" * 60)
        with open(jp, "r") as f:
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
            # Read raw with cv2 (no EXIF correction)
            raw_img = cv2.imread(original_path)
            if raw_img is not None:
                raw_h, raw_w = raw_img.shape[:2]
                print(f"  Raw dimensions (cv2): {raw_w} x {raw_h}")

            # Read with EXIF correction
            exif_img, exif_w, exif_h = load_image(original_path)
            if exif_img is not None:
                print(f"  EXIF-corrected dimensions: {exif_w} x {exif_h}")
                if raw_w != exif_w or raw_h != exif_h:
                    print(f"  ⚠️  EXIF ROTATION DETECTED! Dimensions changed.")
        else:
            print(f"  ERROR: Image not found at {original_path}")

        # 3. CORRECTED IMAGE (used for training)
        print("\n[3] CORRECTED IMAGE (for training)")
        print("-" * 40)
        base_name = os.path.splitext(image_filename)[0]
        corrected_path = os.path.join(corrected_dir, f"{base_name}.png")
        if os.path.exists(corrected_path):
            corrected_img = cv2.imread(corrected_path)
            if corrected_img is not None:
                corr_h, corr_w = corrected_img.shape[:2]
                print(f"  Corrected dimensions: {corr_w} x {corr_h}")

                # Detection on corrected image
                train_box = detect_specimen(corrected_path, margin=20)
                print(f"  Training bounding box:")
                print(f"    left={train_box['left']}, top={train_box['top']}")
                print(f"    right={train_box['right']}, bottom={train_box['bottom']}")
                print(f"    size={train_box['width']} x {train_box['height']}")
        else:
            print(f"  WARNING: Corrected image not found. Run training first.")

        # 4. INFERENCE TEST
        print("\n[4] INFERENCE TEST")
        print("-" * 40)

        # Use either the provided test image or the original
        test_path = test_image_path or original_path

        if not os.path.exists(predictor_path):
            print(f"  ERROR: Model not found at {predictor_path}")
            continue

        # Load with EXIF correction (same as inference would)
        inf_img, inf_w, inf_h = load_image(test_path)
        if inf_img is None:
            print(f"  ERROR: Could not load test image")
            continue

        print(f"  Test image dimensions: {inf_w} x {inf_h}")

        # Detection on test image
        inf_box = detect_specimen(test_path, margin=20)
        print(f"  Inference bounding box:")
        print(f"    left={inf_box['left']}, top={inf_box['top']}")
        print(f"    right={inf_box['right']}, bottom={inf_box['bottom']}")
        print(f"    size={inf_box['width']} x {inf_box['height']}")

        # Compare boxes
        if os.path.exists(corrected_path):
            box_match = (
                train_box['left'] == inf_box['left'] and
                train_box['top'] == inf_box['top'] and
                train_box['right'] == inf_box['right'] and
                train_box['bottom'] == inf_box['bottom']
            )
            if box_match:
                print(f"  ✓ Bounding boxes MATCH")
            else:
                print(f"  ✗ Bounding boxes MISMATCH!")
                print(f"    Training: ({train_box['left']}, {train_box['top']}) -> ({train_box['right']}, {train_box['bottom']})")
                print(f"    Inference: ({inf_box['left']}, {inf_box['top']}) -> ({inf_box['right']}, {inf_box['bottom']})")

        # Run prediction
        predictor = dlib.shape_predictor(predictor_path)
        rect = dlib.rectangle(
            left=inf_box['left'],
            top=inf_box['top'],
            right=inf_box['right'],
            bottom=inf_box['bottom']
        )
        shape = predictor(inf_img, rect)

        print(f"\n  Predicted landmarks:")
        for i in range(shape.num_parts):
            part = shape.part(i)
            # Find corresponding annotation
            ann_lm = next((lm for lm in valid_landmarks if lm.get("id", -1) == i), None)
            if ann_lm:
                dist = ((part.x - ann_lm['x'])**2 + (part.y - ann_lm['y'])**2)**0.5
                status = "✓" if dist < 10 else "✗"
                print(f"    {status} Landmark {i}: predicted=({part.x}, {part.y}), annotated=({ann_lm['x']:.0f}, {ann_lm['y']:.0f}), error={dist:.1f}px")
            else:
                print(f"    ? Landmark {i}: predicted=({part.x}, {part.y}), no annotation found")

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
