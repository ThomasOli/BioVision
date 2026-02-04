import sys
import os
import json
import dlib
from detect_specimen import detect_specimen
from image_utils import load_image


def predict_image(project_root, tag, image_path):
    """
    Predict landmarks on an image using a trained dlib shape predictor.
    Always uses auto-detected bounding box for consistency with training.
    Uses EXIF-corrected image loading to match browser display orientation.
    Restores original landmark IDs using saved mapping from training.
    """
    import hashlib

    modeldir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Cannot find model at {predictor_path}")

    # Load ID mapping if available (maps dlib indices back to original IDs)
    id_mapping = None
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, 'r') as f:
            mapping_data = json.load(f)
            # Convert string keys to int (JSON keys are always strings)
            id_mapping = {int(k): v for k, v in mapping_data.get("dlib_to_original", {}).items()}
        print(f"DEBUG: Loaded ID mapping with {len(id_mapping)} landmarks", file=sys.stderr)

    # Load image with EXIF orientation correction
    img, w, h = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image {image_path}")

    # Compute image hash for debugging (verifies different images are being processed)
    with open(image_path, 'rb') as f:
        img_hash = hashlib.md5(f.read(1000)).hexdigest()[:8]

    print(f"DEBUG: Processing {os.path.basename(image_path)} (hash: {img_hash}, dims: {w}x{h})", file=sys.stderr)

    # Always auto-detect specimen bounds for consistency with training
    detected = detect_specimen(image_path, margin=20)
    if detected is None:
        # Fallback to full image if detection fails
        detected = {
            'left': 0,
            'top': 0,
            'right': w,
            'bottom': h,
            'width': w,
            'height': h
        }

    print(f"DEBUG: Detected box: left={detected['left']}, top={detected['top']}, "
          f"right={detected['right']}, bottom={detected['bottom']}", file=sys.stderr)

    rect = dlib.rectangle(
        left=detected['left'],
        top=detected['top'],
        right=detected['right'],
        bottom=detected['bottom']
    )

    predictor = dlib.shape_predictor(predictor_path)
    shape = predictor(img, rect)

    landmarks = []
    for i in range(shape.num_parts):
        part = shape.part(i)
        # Map dlib index back to original ID if mapping exists
        original_id = id_mapping.get(i, i) if id_mapping else i
        landmarks.append({"id": original_id, "x": int(part.x), "y": int(part.y)})

    # Log first 3 landmarks for debugging
    if landmarks:
        print(f"DEBUG: First 3 landmarks: {landmarks[:3]}", file=sys.stderr)

    return {
        "image": image_path,
        "landmarks": landmarks,
        "detected_box": detected,  # Include detected box for debugging/visualization
        "image_dimensions": {"width": w, "height": h},  # Original image dimensions
        "debug_hash": img_hash  # For verifying different images are being processed
    }


if __name__ == "__main__":
    project_root = sys.argv[1]
    tag = sys.argv[2]
    image_path = sys.argv[3]

    result = predict_image(project_root, tag, image_path)
    print(json.dumps(result))
