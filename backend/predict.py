import sys
import os
import json
import hashlib
import tempfile
from datetime import datetime
import dlib
import cv2
from detect_specimen import detect_specimen
from image_utils import load_image


def save_prediction_log(debug_dir, tag, log_entry):
    """Append prediction to log file."""
    log_path = os.path.join(debug_dir, f"prediction_log_{tag}.json")
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
    logs.append(log_entry)
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)


def predict_image(project_root, tag, image_path):
    """
    Predict landmarks using trained dlib shape predictor.
    Returns landmarks with original IDs from the annotation schema.
    """
    modeldir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Model not found: {predictor_path}")

    # Load ID mapping (dlib sequential IDs -> original landmark IDs)
    id_mapping = None
    num_landmarks = 0
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, 'r') as f:
            data = json.load(f)
            id_mapping = {int(k): v for k, v in data.get("dlib_to_original", {}).items()}
            num_landmarks = data.get("num_landmarks", len(id_mapping) if id_mapping else 0)

    # Load and resize image
    img, orig_w, orig_h = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Could not read: {image_path}")

    img_hash = hashlib.md5(open(image_path, 'rb').read(1000)).hexdigest()[:8]

    max_dim = 1500
    scale = 1.0
    w, h = orig_w, orig_h
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        w, h = int(orig_w * scale), int(orig_h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # Detect specimen bounding box
    temp_path = None
    if scale != 1.0:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        cv2.imwrite(temp_path, img)
        detection_path = temp_path
    else:
        detection_path = image_path

    detected = detect_specimen(detection_path, margin=20)
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    if detected is None:
        detected = {'left': 0, 'top': 0, 'right': w, 'bottom': h, 'width': w, 'height': h}

    # Run dlib shape predictor
    rect = dlib.rectangle(detected['left'], detected['top'], detected['right'], detected['bottom'])
    predictor = dlib.shape_predictor(predictor_path)
    shape = predictor(img, rect)

    # Extract landmarks with ID mapping back to original schema
    landmarks = []
    for i in range(shape.num_parts):
        part = shape.part(i)
        orig_id = id_mapping.get(i, i) if id_mapping else i
        # Scale coordinates back to original image size
        x = int(part.x / scale) if scale != 1.0 else int(part.x)
        y = int(part.y / scale) if scale != 1.0 else int(part.y)
        landmarks.append({"id": orig_id, "x": x, "y": y})

    # Scale bounding box to original size
    if scale != 1.0:
        detected = {k: int(v / scale) for k, v in detected.items()}

    result = {
        "image": image_path,
        "landmarks": landmarks,
        "detected_box": detected,
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "inference_scale": scale,
        "num_landmarks": len(landmarks),
        "id_mapping": id_mapping
    }

    # Save prediction log for debugging
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image": os.path.basename(image_path),
        "image_path": image_path,
        "num_landmarks": len(landmarks),
        "landmarks": landmarks,
        "detected_box": detected,
        "scale": scale,
        "debug_hash": img_hash
    }
    save_prediction_log(debug_dir, tag, log_entry)

    return result


def predict_multi_specimen(project_root, tag, image_path, min_area_ratio=0.02):
    """
    Predict landmarks for multiple specimens using OpenCV detection + dlib.
    Returns list of {box, landmarks} for each detected specimen.
    """
    from detect_specimen import detect_multiple_specimens

    modeldir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Model not found: {predictor_path}")

    # Load ID mapping
    id_mapping = None
    if os.path.exists(id_mapping_path):
        with open(id_mapping_path, 'r') as f:
            data = json.load(f)
            id_mapping = {int(k): v for k, v in data.get("dlib_to_original", {}).items()}

    # Load and resize image
    img, orig_w, orig_h = load_image(image_path)
    if img is None:
        raise RuntimeError(f"Could not read: {image_path}")

    max_dim = 1500
    scale = 1.0
    w, h = orig_w, orig_h
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        w, h = int(orig_w * scale), int(orig_h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # Save temp image for detection if scaled
    import tempfile
    temp_path = None
    if scale != 1.0:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
        os.close(temp_fd)
        cv2.imwrite(temp_path, img)
        detection_path = temp_path
    else:
        detection_path = image_path

    # Detect multiple specimens using OpenCV
    detection_result = detect_multiple_specimens(detection_path, min_area_ratio=min_area_ratio)
    detected_boxes = detection_result.get("boxes", [])

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)

    if not detected_boxes:
        return {
            "image": image_path,
            "specimens": [],
            "num_specimens": 0,
            "image_dimensions": {"width": orig_w, "height": orig_h}
        }

    # Load dlib predictor
    predictor = dlib.shape_predictor(predictor_path)

    specimens = []
    for box in detected_boxes:
        # Create dlib rectangle (use scaled coordinates)
        rect = dlib.rectangle(box['left'], box['top'], box['right'], box['bottom'])

        # Run shape predictor
        shape = predictor(img, rect)

        # Extract landmarks with ID mapping
        landmarks = []
        for i in range(shape.num_parts):
            part = shape.part(i)
            orig_id = id_mapping.get(i, i) if id_mapping else i
            # Scale coordinates back to original image size
            x = int(part.x / scale) if scale != 1.0 else int(part.x)
            y = int(part.y / scale) if scale != 1.0 else int(part.y)
            landmarks.append({"id": orig_id, "x": x, "y": y})

        # Scale bounding box back to original size
        scaled_box = {k: int(v / scale) if scale != 1.0 else v for k, v in box.items() if isinstance(v, (int, float))}
        scaled_box["confidence"] = box.get("confidence")
        scaled_box["class_name"] = box.get("class_name")

        specimens.append({
            "box": scaled_box,
            "landmarks": landmarks,
            "num_landmarks": len(landmarks)
        })

    result = {
        "image": image_path,
        "specimens": specimens,
        "num_specimens": len(specimens),
        "image_dimensions": {"width": orig_w, "height": orig_h},
        "inference_scale": scale,
        "detection_method": "opencv"
    }

    # Save prediction log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "image": os.path.basename(image_path),
        "mode": "multi-specimen",
        "num_specimens": len(specimens),
        "specimens": specimens
    }
    save_prediction_log(debug_dir, tag, log_entry)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python predict.py <project_root> <tag> <image_path> [--multi]")
        sys.exit(1)

    project_root = sys.argv[1]
    tag = sys.argv[2]
    image_path = sys.argv[3]
    multi_mode = "--multi" in sys.argv

    if multi_mode:
        result = predict_multi_specimen(project_root, tag, image_path)
    else:
        result = predict_image(project_root, tag, image_path)
    print(json.dumps(result))
