import cv2
import numpy as np
import os
import sys

import sys as _sys, os as _os
_BACKEND_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _BACKEND_ROOT not in _sys.path:
    _sys.path.insert(0, _BACKEND_ROOT)

from bv_utils.image_utils import load_image



def detect_with_yolo(image_path, model_path, conf_threshold=0.25, margin=20):
    """
    Detect the primary specimen bounding box using a trained YOLO model.
    Returns the highest-confidence box in the same format as detect_specimen(),
    or None if YOLO is unavailable / detects nothing.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model(image_path, conf=conf_threshold, verbose=False)
        if not results:
            return None

        result = results[0]
        is_obb = hasattr(result, "obb") and result.obb is not None
        boxes_obj = result.obb if is_obb else getattr(result, "boxes", None)
        if boxes_obj is None or len(boxes_obj) == 0:
            return None

        best_idx = int(boxes_obj.conf.argmax())

        obb_corners = None
        if is_obb:
            corners = boxes_obj.xyxyxyxy[best_idx].cpu().numpy().tolist()
            obb_corners = corners
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            xyxy = [min(xs), min(ys), max(xs), max(ys)]
        else:
            xyxy = boxes_obj.xyxy[best_idx].tolist()

        left  = max(0, int(xyxy[0]) - margin)
        top   = max(0, int(xyxy[1]) - margin)
        right  = int(xyxy[2]) + margin
        bottom = int(xyxy[3]) + margin
        output = {
            'left': left, 'top': top, 'right': right, 'bottom': bottom,
            'width': right - left, 'height': bottom - top,
            'confidence': float(boxes_obj.conf[best_idx]),
            'detection_method': 'yolo_obb' if is_obb else 'yolo',
        }
        if obb_corners:
            output['obbCorners'] = obb_corners
        return output
    except Exception:
        return None


def detect_multiple_with_yolo(image_path, model_path, conf_threshold=0.25, margin=20,
                               max_specimens=20, nms_iou=None):
    """
    Detect multiple specimen bounding boxes using a trained YOLO model.
    Returns list of boxes in the same format as detect_multiple_specimens(),
    or None if YOLO is unavailable / detects nothing.
    """
    try:
        import json as _json_ds
        if nms_iou is None:
            _cfg = os.path.join(os.path.dirname(model_path),
                                "obb_training", "session_obb", "obb_config.json")
            if os.path.exists(_cfg):
                try:
                    with open(_cfg, "r", encoding="utf-8") as _f:
                        nms_iou = float(_json_ds.load(_f).get("nms_iou", 0.3))
                except Exception:
                    nms_iou = 0.3
        from ultralytics import YOLO
        model = YOLO(model_path)
        predict_kwargs = {"conf": conf_threshold, "verbose": False}
        if nms_iou is not None:
            predict_kwargs["iou"] = nms_iou
        results = model(image_path, **predict_kwargs)
        if not results:
            return None

        result = results[0]
        is_obb = hasattr(result, "obb") and result.obb is not None
        boxes_obj = result.obb if is_obb else getattr(result, "boxes", None)
        if boxes_obj is None or len(boxes_obj) == 0:
            return None

        img_w = result.orig_shape[1]
        img_h = result.orig_shape[0]

        boxes_out = []
        for i in range(min(len(boxes_obj), max_specimens)):
            conf   = float(boxes_obj.conf[i])
            cls_id = int(boxes_obj.cls[i]) if boxes_obj.cls is not None else 0
            cls_name = result.names.get(cls_id, 'specimen') if hasattr(result, "names") else 'specimen'

            obb_corners = None
            if is_obb:
                corners = boxes_obj.xyxyxyxy[i].cpu().numpy().tolist()
                obb_corners = corners
                xs   = [p[0] for p in corners]
                ys   = [p[1] for p in corners]
                xyxy = [min(xs), min(ys), max(xs), max(ys)]
            else:
                xyxy = boxes_obj.xyxy[i].cpu().numpy().tolist()

            left   = max(0, int(xyxy[0]) - margin)
            top    = max(0, int(xyxy[1]) - margin)
            right  = min(img_w, int(xyxy[2]) + margin)
            bottom = min(img_h, int(xyxy[3]) + margin)
            box_out = {
                'left': left, 'top': top, 'right': right, 'bottom': bottom,
                'width': right - left, 'height': bottom - top,
                'confidence': conf, 'class_id': cls_id, 'class_name': cls_name,
            }
            if obb_corners:
                box_out['obbCorners'] = obb_corners
            boxes_out.append(box_out)

        boxes_out.sort(key=lambda b: (b['top'], b['left']))
        return boxes_out
    except Exception:
        return None


def detect_specimen(image_path, margin=20, yolo_model_path=None):
    """
    Detect specimen bounding box. Returns dict with box coordinates.
    If yolo_model_path is provided, tries YOLO detection first and falls
    back to OpenCV contours only if YOLO finds nothing.
    """
    # Try YOLO first if a model path is provided
    if yolo_model_path:
        yolo_result = detect_with_yolo(image_path, yolo_model_path, margin=margin)
        if yolo_result is not None:
            return yolo_result

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
                               iou_threshold=0.3, max_specimens=20, yolo_model_path=None):
    """
    Detect multiple specimen bounding boxes using contour detection with watershed separation.
    If yolo_model_path is provided, uses YOLO and falls back to OpenCV only if YOLO returns nothing.

    Args:
        image_path: Path to image
        margin: Pixel margin around detected boxes
        min_area_ratio: Minimum area as fraction of image (default 2%)
        max_area_ratio: Maximum area as fraction of image (default 60%)
        iou_threshold: IoU threshold for non-max suppression
        max_specimens: Maximum number of specimens to return
        yolo_model_path: Optional path to a fine-tuned YOLO .pt model

    Returns:
        dict with 'boxes' list and metadata
    """
    # Try YOLO first if a model path is provided
    if yolo_model_path:
        yolo_boxes = detect_multiple_with_yolo(image_path, yolo_model_path, margin=margin,
                                               max_specimens=max_specimens)
        if yolo_boxes:
            img, w, h = load_image(image_path)
            img_w = w if img is not None else 0
            img_h = h if img is not None else 0
            return {
                "ok": True,
                "boxes": yolo_boxes,
                "image_width": img_w,
                "image_height": img_h,
                "num_detections": len(yolo_boxes),
                "detection_method": "yolo",
            }

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
        print("Usage: python detect_specimen.py <image_path> [--yolo-model <path>] [--multi] [min_area_ratio] | --check")
        sys.exit(1)

    if sys.argv[1] == "--check":
        # Check detection availability
        print(json.dumps(check_detection_available()))
        sys.exit(0)

    # Parse --yolo-model flag
    yolo_model = None
    args = sys.argv[1:]
    if "--yolo-model" in args:
        idx = args.index("--yolo-model")
        if idx + 1 < len(args):
            yolo_model = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    image_path = args[0]

    if "--multi" in args:
        min_area = 0.02
        for a in args[1:]:
            try:
                min_area = float(a)
                break
            except ValueError:
                pass
        result = detect_multiple_specimens(image_path, min_area_ratio=min_area,
                                           yolo_model_path=yolo_model)
        print(json.dumps(result, indent=2))
    elif len(args) >= 2 and not args[1].startswith("--"):
        # Legacy: min_area_ratio as positional arg → multi detection
        min_area = float(args[1])
        result = detect_multiple_specimens(image_path, min_area_ratio=min_area,
                                           yolo_model_path=yolo_model)
        print(json.dumps(result, indent=2))
    else:
        # Single specimen detection
        result = detect_specimen(image_path, yolo_model_path=yolo_model)
        print(json.dumps(result))
