import cv2
import numpy as np
import os
import sys
from image_utils import load_image


def _infer_session_dir_from_model_path(model_path):
    """Walk up from model_path and return the nearest folder containing session.json."""
    if not model_path:
        return None
    current = os.path.abspath(model_path)
    if os.path.isfile(current):
        current = os.path.dirname(current)
    for _ in range(8):
        session_path = os.path.join(current, "session.json")
        if os.path.exists(session_path):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def _load_head_tail_ids_for_model(model_path):
    """
    Resolve head/tail landmark IDs from the owning session.json (if available).
    """
    session_dir = _infer_session_dir_from_model_path(model_path)
    if not session_dir:
        return None, None
    session_path = os.path.join(session_dir, "session.json")
    if not os.path.exists(session_path):
        return None, None
    try:
        import json

        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
    except Exception:
        return None, None

    template = session.get("landmarkTemplate", [])
    if not isinstance(template, list):
        return None, None

    orientation_policy = session.get("orientationPolicy", {})
    if not isinstance(orientation_policy, dict):
        orientation_policy = {}

    def _normalize_targets(raw, fallback):
        if isinstance(raw, (list, tuple)):
            values = [str(v).strip().lower() for v in raw if str(v).strip()]
            if values:
                return set(values)
        return set(str(v).strip().lower() for v in fallback if str(v).strip())

    head_targets = _normalize_targets(orientation_policy.get("headCategories"), ["head"])
    tail_targets = _normalize_targets(orientation_policy.get("tailCategories"), ["tail"])

    head_id = None
    tail_id = None
    for lm in template:
        try:
            idx = int(lm.get("index"))
        except (TypeError, ValueError):
            continue
        cat = str(lm.get("category", "")).strip().lower()
        if cat in head_targets and head_id is None:
            head_id = idx
        elif cat in tail_targets and tail_id is None:
            tail_id = idx
    return head_id, tail_id


def _load_landmark_order_for_model(model_path):
    """
    Resolve sorted session landmark indices to map YOLO-pose keypoint slots.
    """
    session_dir = _infer_session_dir_from_model_path(model_path)
    if not session_dir:
        return []
    session_path = os.path.join(session_dir, "session.json")
    if not os.path.exists(session_path):
        return []
    try:
        import json

        with open(session_path, "r", encoding="utf-8") as f:
            session = json.load(f)
    except Exception:
        return []
    template = session.get("landmarkTemplate", [])
    if not isinstance(template, list):
        return []
    ids = []
    for lm in template:
        try:
            ids.append(int(lm.get("index")))
        except Exception:
            continue
    return sorted(set(ids))


def _extract_orientation_hint(keypoints_data, det_idx, enable_hint,
                              head_id=None, tail_id=None, landmark_order=None):
    """
    Extract left/right orientation from YOLO-pose keypoints.

    Uses explicit head/tail IDs when available; falls back to first two keypoints.
    """
    if not enable_hint or keypoints_data is None:
        return None
    try:
        if det_idx >= len(keypoints_data):
            return None
        kp_item = keypoints_data[det_idx]
        kp_xy = kp_item.xy[0].cpu().numpy()
        if kp_xy is None or kp_xy.shape[0] < 2:
            return None

        head_idx = 0
        tail_idx = 1
        if (
            landmark_order
            and head_id is not None
            and tail_id is not None
            and len(landmark_order) == kp_xy.shape[0]
            and int(head_id) in landmark_order
            and int(tail_id) in landmark_order
        ):
            head_idx = int(landmark_order.index(int(head_id)))
            tail_idx = int(landmark_order.index(int(tail_id)))

        head_pt = kp_xy[head_idx].tolist()
        tail_pt = kp_xy[tail_idx].tolist()
        if not np.isfinite(head_pt).all() or not np.isfinite(tail_pt).all():
            return None

        orientation = "left" if head_pt[0] < tail_pt[0] else "right"
        hint = {
            "orientation": orientation,
            "head_point": [float(head_pt[0]), float(head_pt[1])],
            "tail_point": [float(tail_pt[0]), float(tail_pt[1])],
            "source": "yolo_pose",
        }

        conf_tensor = getattr(kp_item, "conf", None)
        if conf_tensor is not None:
            conf_arr = conf_tensor[0].cpu().numpy()
            if conf_arr is not None and len(conf_arr) > max(head_idx, tail_idx):
                hint["head_confidence"] = float(conf_arr[head_idx])
                hint["tail_confidence"] = float(conf_arr[tail_idx])
                hint["confidence"] = float(min(conf_arr[head_idx], conf_arr[tail_idx]))
        return hint
    except Exception:
        return None


def _orientation_from_class_name(class_name):
    token = str(class_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return None
    if token.endswith("_left") or token == "left" or "_left_" in token:
        return "left"
    if token.endswith("_right") or token == "right" or "_right_" in token:
        return "right"
    return None


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
        if result.boxes is None or len(result.boxes) == 0:
            return None

        # Orientation hints only apply to session-trained pose models where
        # keypoints are [head, tail].
        head_id, tail_id = _load_head_tail_ids_for_model(model_path)
        landmark_order = _load_landmark_order_for_model(model_path)
        can_use_pose_hint = bool(head_id is not None and tail_id is not None)

        # Pick highest-confidence detection
        boxes = result.boxes
        best_idx = int(boxes.conf.argmax())
        keypoints_data = getattr(result, "keypoints", None)
        orientation_hint = _extract_orientation_hint(
            keypoints_data,
            best_idx,
            enable_hint=can_use_pose_hint,
            head_id=head_id,
            tail_id=tail_id,
            landmark_order=landmark_order,
        )

        xyxy = boxes.xyxy[best_idx].tolist()
        left = max(0, int(xyxy[0]) - margin)
        top = max(0, int(xyxy[1]) - margin)
        right = int(xyxy[2]) + margin
        bottom = int(xyxy[3]) + margin
        output = {
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom,
            'width': right - left,
            'height': bottom - top,
            'confidence': float(boxes.conf[best_idx]),
            'detection_method': 'yolo',
        }
        if orientation_hint is None:
            cls_id = int(boxes.cls[best_idx]) if boxes.cls is not None else 0
            cls_name = result.names.get(cls_id, "specimen") if hasattr(result, "names") else "specimen"
            cls_ori = _orientation_from_class_name(cls_name)
            if cls_ori is not None:
                orientation_hint = {
                    "orientation": cls_ori,
                    "confidence": float(boxes.conf[best_idx]),
                    "source": "detector_class",
                }
        if orientation_hint is not None:
            output["orientation_hint"] = orientation_hint
        return output
    except Exception:
        return None


def detect_multiple_with_yolo(image_path, model_path, conf_threshold=0.25, margin=20,
                               max_specimens=20):
    """
    Detect multiple specimen bounding boxes using a trained YOLO model.
    Returns list of boxes in the same format as detect_multiple_specimens(),
    or None if YOLO is unavailable / detects nothing.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model(image_path, conf=conf_threshold, verbose=False)
        if not results:
            return None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        img_w = result.orig_shape[1]
        img_h = result.orig_shape[0]
        keypoints_data = getattr(result, "keypoints", None)
        head_id, tail_id = _load_head_tail_ids_for_model(model_path)
        landmark_order = _load_landmark_order_for_model(model_path)
        can_use_pose_hint = bool(head_id is not None and tail_id is not None)

        boxes_out = []
        for i in range(min(len(result.boxes), max_specimens)):
            xyxy = result.boxes.xyxy[i].tolist()
            conf = float(result.boxes.conf[i])
            left = max(0, int(xyxy[0]) - margin)
            top = max(0, int(xyxy[1]) - margin)
            right = min(img_w, int(xyxy[2]) + margin)
            bottom = min(img_h, int(xyxy[3]) + margin)
            box_out = {
                'left': left,
                'top': top,
                'right': right,
                'bottom': bottom,
                'width': right - left,
                'height': bottom - top,
                'confidence': conf,
                'class_id': int(result.boxes.cls[i]) if result.boxes.cls is not None else 0,
                'class_name': result.names.get(int(result.boxes.cls[i]), 'specimen') if result.boxes.cls is not None else 'specimen',
            }
            orientation_hint = _extract_orientation_hint(
                keypoints_data,
                i,
                enable_hint=can_use_pose_hint,
                head_id=head_id,
                tail_id=tail_id,
                landmark_order=landmark_order,
            )
            if orientation_hint is None:
                cls_ori = _orientation_from_class_name(box_out.get("class_name"))
                if cls_ori is not None:
                    orientation_hint = {
                        "orientation": cls_ori,
                        "confidence": conf,
                        "source": "detector_class",
                    }
            if orientation_hint is not None:
                box_out["orientation_hint"] = orientation_hint
            boxes_out.append(box_out)

        # Sort top-to-bottom, left-to-right
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
