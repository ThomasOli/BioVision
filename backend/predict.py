import sys
import os
import json
import dlib
import cv2

def predict_image(project_root, tag, image_path, box=None):
    """
    Predict landmarks on an image using a trained dlib shape predictor.

    Args:
        project_root: Path to project directory containing models/
        tag: Model tag name (loads predictor_{tag}.dat)
        image_path: Path to image file
        box: Optional dict with 'left', 'top', 'right', 'bottom' keys.
             If None, uses the full image bounds.
             For best results, provide a bounding box similar to training data.
    """
    modeldir = os.path.join(project_root, "models")
    predictor_path = os.path.join(modeldir, f"predictor_{tag}.dat")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Cannot find model at {predictor_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image {image_path}")

    h, w = img.shape[:2]

    if box is not None:
        rect = dlib.rectangle(
            left=int(box['left']),
            top=int(box['top']),
            right=int(box['right']),
            bottom=int(box['bottom'])
        )
    else:
        # Default to full image - works best if model was trained the same way
        rect = dlib.rectangle(left=0, top=0, right=w-1, bottom=h-1)

    predictor = dlib.shape_predictor(predictor_path)
    shape = predictor(img, rect)

    landmarks = []
    for i in range(shape.num_parts):
        part = shape.part(i)
        landmarks.append({"id": i, "x": int(part.x), "y": int(part.y)})

    return {"image": image_path, "landmarks": landmarks}


if __name__ == "__main__":
    project_root = sys.argv[1]
    tag = sys.argv[2]
    image_path = sys.argv[3]

    # Optional: pass bounding box as JSON string in argv[4]
    box = None
    if len(sys.argv) > 4:
        box = json.loads(sys.argv[4])

    result = predict_image(project_root, tag, image_path, box)
    print(json.dumps(result))