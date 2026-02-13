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


def detect_multiple_specimens(image_path, margin=20, min_area_ratio=0.02, max_area_ratio=0.6,
                               iou_threshold=0.3, max_specimens=20):
    """
    Detect multiple specimen bounding boxes using contour detection with watershed separation.

    Args:
        image_path: Path to image
        margin: Pixel margin around detected boxes
        min_area_ratio: Minimum area as fraction of image (default 2%)
        max_area_ratio: Maximum area as fraction of image (default 60%)
        iou_threshold: IoU threshold for non-max suppression
        max_specimens: Maximum number of specimens to return

    Returns:
        dict with 'boxes' list and metadata
    """
    img, w, h = load_image(image_path)
    if img is None:
        return {"ok": False, "error": f"Could not load image: {image_path}", "boxes": []}

    # Resize large images for faster processing
    max_dim = 1500
    scale = 1.0
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img_small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = img

    sh, sw = img_small.shape[:2]

    # Area thresholds (on scaled image)
    min_area = sw * sh * min_area_ratio
    max_area = sw * sh * max_area_ratio

    def get_contour_box(contour):
        """Get bounding box for contour if it meets area requirements."""
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            return None

        x, y, bw, bh = cv2.boundingRect(contour)

        # Filter by aspect ratio (not too extreme)
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 10:  # Too elongated
            return None

        return {
            'x': x, 'y': y, 'w': bw, 'h': bh,
            'area': area,
            'score': area / (sw * sh)  # Use normalized area as confidence
        }

    def compute_iou(box1, box2):
        """Compute IoU between two boxes."""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
        y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1['w'] * box1['h']
        area2 = box2['w'] * box2['h']
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def non_max_suppression(boxes, iou_thresh):
        """Apply non-maximum suppression to remove overlapping boxes."""
        if not boxes:
            return []

        # Sort by score (area) descending
        boxes = sorted(boxes, key=lambda b: b['score'], reverse=True)

        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)

            # Remove boxes that overlap too much with best
            boxes = [b for b in boxes if compute_iou(best, b) < iou_thresh]

        return keep

    def watershed_separation(binary_mask):
        """Use watershed to separate touching objects."""
        # Distance transform
        dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Find sure foreground (peaks in distance transform)
        _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        # Find sure background
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=3)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Apply watershed
        img_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)

        return markers

    all_boxes = []

    # Preprocess
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use smaller kernel to avoid merging nearby objects
    kernel_small = np.ones((3, 3), np.uint8)

    # Strategy 1: Edge detection with minimal closing (to preserve separation)
    edges = cv2.Canny(blurred, 50, 150)
    # Only dilate once, don't close - this prevents merging
    dilated = cv2.dilate(edges, kernel_small, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        box = get_contour_box(c)
        if box:
            box['method'] = 'canny'
            all_boxes.append(box)

    # Strategy 2: Otsu thresholding with watershed separation
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Try watershed to separate touching objects
    try:
        markers = watershed_separation(thresh)
        for label in range(2, markers.max() + 1):  # Skip background (1) and boundaries (-1)
            mask = np.uint8(markers == label) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                box = get_contour_box(c)
                if box:
                    box['method'] = 'watershed'
                    all_boxes.append(box)
    except Exception:
        pass

    # Also try without watershed as backup
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        box = get_contour_box(c)
        if box:
            box['method'] = 'otsu_inv'
            all_boxes.append(box)

    # Strategy 3: Adaptive thresholding (no morphological closing to preserve separation)
    adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 21, 5)
    # Use opening to remove noise without connecting nearby objects
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel_small)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        box = get_contour_box(c)
        if box:
            box['method'] = 'adaptive'
            all_boxes.append(box)

    # Strategy 4: Color-based segmentation (HSV saturation) - no closing
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
    # Opening only to clean noise
    sat_thresh = cv2.morphologyEx(sat_thresh, cv2.MORPH_OPEN, kernel_small)
    contours, _ = cv2.findContours(sat_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        box = get_contour_box(c)
        if box:
            box['method'] = 'saturation'
            all_boxes.append(box)

    # Strategy 5: K-means color clustering (good for separating distinct colored specimens)
    try:
        pixels = img_small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = min(5, max(2, len(all_boxes) + 2))  # Adaptive k based on current detections
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(sh, sw)

        # Find contours in each cluster (skip background-like clusters)
        for i in range(k):
            cluster_mask = np.uint8(labels == i) * 255
            cluster_area = np.sum(cluster_mask > 0)

            # Skip if this cluster is too large (likely background) or too small
            if cluster_area > sw * sh * 0.7 or cluster_area < min_area:
                continue

            contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                box = get_contour_box(c)
                if box:
                    box['method'] = 'kmeans'
                    all_boxes.append(box)
    except Exception:
        pass

    # Apply non-maximum suppression
    filtered_boxes = non_max_suppression(all_boxes, iou_threshold)

    # Limit to max specimens
    filtered_boxes = filtered_boxes[:max_specimens]

    # Convert to output format (scale back to original size)
    result_boxes = []
    for box in filtered_boxes:
        x = int(box['x'] / scale)
        y = int(box['y'] / scale)
        bw = int(box['w'] / scale)
        bh = int(box['h'] / scale)

        # Add margin
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(w, x + bw + margin)
        bottom = min(h, y + bh + margin)

        result_boxes.append({
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom,
            'width': right - left,
            'height': bottom - top,
            'confidence': round(box['score'], 4),
            'class_id': 0,
            'class_name': 'specimen'
        })

    # Sort by position (top to bottom, left to right)
    result_boxes.sort(key=lambda b: (b['top'], b['left']))

    return {
        "ok": True,
        "boxes": result_boxes,
        "image_width": w,
        "image_height": h,
        "num_detections": len(result_boxes),
        "detection_method": "opencv_contour"
    }


def check_detection_available():
    """Check that detection is available (OpenCV is always available)."""
    return {
        "available": True,
        "primary_method": "opencv"
    }


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python detect_specimen.py <image_path> [min_area_ratio] | --check")
        sys.exit(1)

    if sys.argv[1] == "--check":
        # Check detection availability
        print(json.dumps(check_detection_available()))
    elif len(sys.argv) >= 3 and sys.argv[2] != "--multi":
        # Multi-specimen detection with min_area_ratio parameter
        min_area = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
        result = detect_multiple_specimens(sys.argv[1], min_area_ratio=min_area)
        print(json.dumps(result, indent=2))
    elif "--multi" in sys.argv:
        # Multi-specimen detection with default parameters
        result = detect_multiple_specimens(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        # Single specimen detection
        result = detect_specimen(sys.argv[1])
        print(json.dumps(result))
