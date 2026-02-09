import cv2
import numpy as np
import sys
from image_utils import load_image


def detect_specimen(image_path, margin=20):
    """Detect specimen bounding box. Returns dict with box coordinates."""
    img, w, h = load_image(image_path)
    if img is None:
        return None

    # Resize large images for faster processing
    max_dim = 1000
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img_small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = img

    sh, sw = img_small.shape[:2]

    # Area thresholds (on scaled image)
    min_area = sw * sh * 0.08
    max_area = sw * sh * 0.85

    # Center of image for scoring
    cx, cy = sw // 2, sh // 2

    def score_contour(contour):
        """Score based on area and center proximity."""
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            return -1

        x, y, bw, bh = cv2.boundingRect(contour)
        box_cx, box_cy = x + bw // 2, y + bh // 2

        # Center distance score
        max_dist = np.sqrt(cx**2 + cy**2)
        center_dist = np.sqrt((box_cx - cx)**2 + (box_cy - cy)**2)
        center_score = 1 - (center_dist / max_dist)

        # Area score
        area_score = min(area / (sw * sh * 0.4), 1.0)

        return (area_score * 0.5) + (center_score * 0.5)

    def find_best_contour(contours):
        if not contours:
            return None
        scored = [(c, score_contour(c)) for c in contours]
        valid = [(c, s) for c, s in scored if s > 0]
        if valid:
            return max(valid, key=lambda x: x[1])[0]
        return None

    best_contour = None
    best_score = -1

    # Preprocess
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Strategy 1: Edge detection (fast and effective)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = find_best_contour(contours)
    if candidate is not None:
        s = score_contour(candidate)
        if s > best_score:
            best_contour = candidate
            best_score = s

    # Strategy 2: Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = find_best_contour(contours)
    if candidate is not None:
        s = score_contour(candidate)
        if s > best_score:
            best_contour = candidate
            best_score = s

    # Strategy 3: Inverted Otsu
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate = find_best_contour(contours)
    if candidate is not None:
        s = score_contour(candidate)
        if s > best_score:
            best_contour = candidate
            best_score = s

    # Strategy 4: Saliency (if available and score still low)
    if best_score < 0.4:
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(img_small)
            if success:
                saliency_map = (saliency_map * 255).astype(np.uint8)
                _, thresh = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidate = find_best_contour(contours)
                if candidate is not None:
                    s = score_contour(candidate)
                    if s > best_score:
                        best_contour = candidate
                        best_score = s
        except:
            pass

    # If detection weak, use center 70% as fallback
    if best_contour is None or best_score < 0.25:
        # Center-weighted fallback
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        return {
            'left': margin_x,
            'top': margin_y,
            'right': w - margin_x,
            'bottom': h - margin_y,
            'width': w - 2 * margin_x,
            'height': h - 2 * margin_y
        }

    # Scale bounding box back to original size
    x, y, bw, bh = cv2.boundingRect(best_contour)
    x = int(x / scale)
    y = int(y / scale)
    bw = int(bw / scale)
    bh = int(bh / scale)

    # Add margin
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(w, x + bw + margin)
    bottom = min(h, y + bh + margin)

    return {
        'left': left,
        'top': top,
        'right': right,
        'bottom': bottom,
        'width': right - left,
        'height': bottom - top
    }


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python detect_specimen.py <image_path>")
        sys.exit(1)

    result = detect_specimen(sys.argv[1])
    print(json.dumps(result))
