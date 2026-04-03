import json
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from detection_utils import normalize_orientation_payload


def _resolve_obb_detection_preset(conf_threshold=0.3, nms_iou=0.3, max_specimens=20, detection_preset="balanced", imgsz=None):
    preset = str(detection_preset or "balanced").strip().lower()
    conf = float(conf_threshold)
    iou = float(nms_iou)
    top_k = int(max_specimens)
    resolved_imgsz = int(imgsz) if imgsz is not None else 640

    if preset == "custom":
        conf = min(max(conf, 0.01), 0.99)
        iou = min(max(iou, 0.05), 0.95)
        top_k = min(max(top_k, 1), 250)
        resolved_imgsz = 1280 if resolved_imgsz >= 1280 else 960 if resolved_imgsz >= 960 else 640
    elif preset == "precision":
        conf = max(conf, 0.45)
        top_k = min(top_k, 8)
        resolved_imgsz = 640
    elif preset == "recall":
        conf = min(conf, 0.2)
        top_k = max(top_k, 30)
        resolved_imgsz = 960 if resolved_imgsz < 960 else resolved_imgsz
    elif preset == "single_object":
        conf = max(conf, 0.35)
        top_k = 1
        resolved_imgsz = 640
    else:
        preset = "balanced"
        conf = max(0.3, min(conf, 0.9))
        top_k = max(1, min(top_k, 25))
        resolved_imgsz = 640 if resolved_imgsz not in (640, 960, 1280) else resolved_imgsz

    return {
        "preset": preset,
        "conf": conf,
        "iou": iou,
        "top_k": max(1, top_k),
        "imgsz": resolved_imgsz,
    }


def _build_canonical_obb_from_xywhr(cx, cy, width, height, angle_rad):
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    return np.asarray(
        [
            [float(cx) + cos_a * (-half_w) - sin_a * (-half_h), float(cy) + sin_a * (-half_w) + cos_a * (-half_h)],
            [float(cx) + cos_a * (half_w) - sin_a * (-half_h), float(cy) + sin_a * (half_w) + cos_a * (-half_h)],
            [float(cx) + cos_a * (half_w) - sin_a * (half_h), float(cy) + sin_a * (half_w) + cos_a * (half_h)],
            [float(cx) + cos_a * (-half_w) - sin_a * (half_h), float(cy) + sin_a * (-half_w) + cos_a * (half_h)],
        ],
        dtype=np.float32,
    )


def _as_corner_array(corners):
    arr = np.asarray(corners, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError("expected 4 OBB corners")
    return arr


def _roll_corners_to_top_left(corners):
    pts = _as_corner_array(corners)
    top_left_idx = min(
        range(4),
        key=lambda idx: (float(pts[idx][1]), float(pts[idx][0])),
    )
    return np.roll(pts, -top_left_idx, axis=0)


def _is_valid_canonical_obb(corners, tolerance=1e-3):
    pts = _as_corner_array(corners)
    if len({(round(float(x), 4), round(float(y), 4)) for x, y in pts}) != 4:
        return False

    area2 = 0.0
    for idx in range(4):
        x1, y1 = pts[idx]
        x2, y2 = pts[(idx + 1) % 4]
        area2 += float(x1) * float(y2) - float(y1) * float(x2)
    if abs(area2) <= tolerance:
        return False

    top_mean_y = float(pts[0][1] + pts[1][1]) / 2.0
    bottom_mean_y = float(pts[2][1] + pts[3][1]) / 2.0
    left_mean_x = float(pts[0][0] + pts[3][0]) / 2.0
    right_mean_x = float(pts[1][0] + pts[2][0]) / 2.0
    if top_mean_y > bottom_mean_y + tolerance:
        return False
    if left_mean_x > right_mean_x + tolerance:
        return False

    edge_lengths = [
        float(np.linalg.norm(pts[(idx + 1) % 4] - pts[idx]))
        for idx in range(4)
    ]
    if min(edge_lengths) <= tolerance:
        return False

    return True


def _canonicalize_by_row_sort(corners):
    pts = _as_corner_array(corners)
    sorted_idx = sorted(
        range(4),
        key=lambda idx: (float(pts[idx][1]), float(pts[idx][0])),
    )
    top = sorted((pts[sorted_idx[0]], pts[sorted_idx[1]]), key=lambda point: float(point[0]))
    bottom = sorted((pts[sorted_idx[2]], pts[sorted_idx[3]]), key=lambda point: float(point[0]))
    return np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def _canonicalize_by_angle_sort(corners):
    pts = _as_corner_array(corners)
    center = pts.mean(axis=0)
    order = sorted(
        range(4),
        key=lambda idx: math.atan2(float(pts[idx][1] - center[1]), float(pts[idx][0] - center[0])),
    )
    ordered = pts[order]
    rolled = _roll_corners_to_top_left(ordered)
    clockwise = np.asarray([rolled[0], rolled[3], rolled[2], rolled[1]], dtype=np.float32)
    counter_clockwise = np.asarray([rolled[0], rolled[1], rolled[2], rolled[3]], dtype=np.float32)
    clockwise_valid = _is_valid_canonical_obb(clockwise)
    counter_clockwise_valid = _is_valid_canonical_obb(counter_clockwise)
    if clockwise_valid and not counter_clockwise_valid:
        return clockwise
    if counter_clockwise_valid and not clockwise_valid:
        return counter_clockwise
    if clockwise_valid:
        return clockwise
    return counter_clockwise


def _canonicalize_by_min_area_rect(corners):
    try:
        import cv2
    except Exception:
        return None

    pts = _as_corner_array(corners)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    return _canonicalize_by_angle_sort(box)


def canonicalize_detector_obb_corners(corners, xywhr=None):
    pts = _as_corner_array(corners)

    if xywhr is not None:
        xywhr_arr = np.asarray(xywhr, dtype=np.float32).reshape(-1)
        if xywhr_arr.shape[0] >= 5 and np.all(np.isfinite(xywhr_arr[:5])):
            candidate = _build_canonical_obb_from_xywhr(
                xywhr_arr[0],
                xywhr_arr[1],
                xywhr_arr[2],
                xywhr_arr[3],
                xywhr_arr[4],
            )
            if _is_valid_canonical_obb(candidate):
                return candidate.tolist()

    candidate = _canonicalize_by_row_sort(pts)
    if _is_valid_canonical_obb(candidate):
        return candidate.tolist()

    candidate = _canonicalize_by_angle_sort(pts)
    if _is_valid_canonical_obb(candidate):
        return candidate.tolist()

    candidate = _canonicalize_by_min_area_rect(pts)
    if candidate is not None and _is_valid_canonical_obb(candidate):
        return candidate.tolist()

    return pts.tolist()


def _obb_to_xyxy(corners):
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def _aabb_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _class_agnostic_dedup(boxes, iou_threshold=0.5):
    kept = []
    for box in sorted(boxes, key=lambda item: item.get("confidence", 0.0), reverse=True):
        bbox = (box["left"], box["top"], box["right"], box["bottom"])
        if any(_aabb_iou(bbox, (k["left"], k["top"], k["right"], k["bottom"])) > iou_threshold for k in kept):
            continue
        kept.append(box)
    return kept


def _build_orientation_hint(class_id, confidence, orientation_policy=None):
    payload = normalize_orientation_payload(class_id, orientation_policy)
    hint = payload.get("orientation_hint")
    if not isinstance(hint, dict):
        return None
    return {
        **hint,
        "confidence": float(confidence),
    }


def _parse_obb_boxes(result, margin=20, max_specimens=20, orientation_policy=None):
    boxes_obj = getattr(result, "obb", None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    img_h, img_w = result.orig_shape[:2]
    names = getattr(result, "names", {}) or {}
    parsed = []
    limit = min(len(boxes_obj), max(1, int(max_specimens)))
    for i in range(limit):
        raw_corners = boxes_obj.xyxyxyxy[i].cpu().numpy().tolist()
        xywhr = boxes_obj.xywhr[i].cpu().numpy().tolist()
        corners = canonicalize_detector_obb_corners(raw_corners, xywhr=xywhr)
        angle_rad = float(xywhr[4])
        class_id = int(boxes_obj.cls[i]) if boxes_obj.cls is not None else 0
        confidence = float(boxes_obj.conf[i])
        x1, y1, x2, y2 = _obb_to_xyxy(corners)
        left = max(0, int(round(x1)) - int(margin))
        top = max(0, int(round(y1)) - int(margin))
        right = min(img_w, int(round(x2)) + int(margin))
        bottom = min(img_h, int(round(y2)) + int(margin))
        orientation_hint = _build_orientation_hint(class_id, confidence, orientation_policy)
        parsed.append(
            {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": names.get(class_id, "specimen") if isinstance(names, dict) else "specimen",
                "detection_method": "yolo_obb",
                "obbCorners": corners,
                "angle": angle_rad * 180.0 / math.pi,
                **({"orientation_hint": orientation_hint} if orientation_hint else {}),
            }
        )
    return _class_agnostic_dedup(parsed)


def detect_with_yolo(
    image_path,
    model_path,
    conf_threshold=0.25,
    margin=20,
    orientation_policy=None,
    nms_iou=None,
    detection_preset="balanced",
    imgsz=None,
):
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    if not model_path or not os.path.exists(model_path):
        return None

    resolved = _resolve_obb_detection_preset(
        conf_threshold=conf_threshold,
        nms_iou=0.3 if nms_iou is None else nms_iou,
        max_specimens=1,
        detection_preset=detection_preset,
        imgsz=imgsz,
    )
    model = YOLO(model_path)
    results = model.predict(
        image_path,
        conf=float(resolved["conf"]),
        iou=float(resolved["iou"]),
        imgsz=int(resolved["imgsz"]),
        task="obb",
        verbose=False,
    )
    if not results:
        return None

    boxes = _parse_obb_boxes(
        results[0],
        margin=margin,
        max_specimens=1,
        orientation_policy=orientation_policy,
    )
    return boxes[0] if boxes else None


def detect_multiple_with_yolo(
    image_path,
    model_path,
    conf_threshold=0.25,
    margin=20,
    max_specimens=20,
    nms_iou=None,
    orientation_policy=None,
    detection_preset="balanced",
    imgsz=None,
):
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    if not model_path or not os.path.exists(model_path):
        return None

    resolved = _resolve_obb_detection_preset(
        conf_threshold=conf_threshold,
        nms_iou=0.3 if nms_iou is None else nms_iou,
        max_specimens=max_specimens,
        detection_preset=detection_preset,
        imgsz=imgsz,
    )
    predict_kwargs = {
        "conf": float(resolved["conf"]),
        "iou": float(resolved["iou"]),
        "imgsz": int(resolved["imgsz"]),
        "task": "obb",
        "verbose": False,
    }
    model = YOLO(model_path)
    results = model.predict(image_path, **predict_kwargs)
    if not results:
        return None

    boxes = _parse_obb_boxes(
        results[0],
        margin=margin,
        max_specimens=resolved["top_k"],
        orientation_policy=orientation_policy,
    )
    boxes.sort(key=lambda item: (item["top"], item["left"]))
    return boxes


def detect_specimen(
    image_path,
    margin=20,
    yolo_model_path=None,
    orientation_policy=None,
    conf_threshold=0.25,
    nms_iou=None,
    detection_preset="balanced",
    imgsz=None,
):
    return detect_with_yolo(
        image_path,
        yolo_model_path,
        conf_threshold=conf_threshold,
        margin=margin,
        orientation_policy=orientation_policy,
        nms_iou=nms_iou,
        detection_preset=detection_preset,
        imgsz=imgsz,
    )


def detect_multiple_specimens(
    image_path,
    min_area_ratio=0.02,
    yolo_model_path=None,
    margin=20,
    max_specimens=20,
    nms_iou=None,
    orientation_policy=None,
    conf_threshold=0.25,
    detection_preset="balanced",
    imgsz=None,
):
    del min_area_ratio
    boxes = detect_multiple_with_yolo(
        image_path,
        yolo_model_path,
        conf_threshold=conf_threshold,
        margin=margin,
        max_specimens=max_specimens,
        nms_iou=nms_iou,
        orientation_policy=orientation_policy,
        detection_preset=detection_preset,
        imgsz=imgsz,
    )
    if boxes is None:
        return {
            "ok": False,
            "boxes": [],
            "error": "OBB detector unavailable or produced no detections.",
            "detection_method": "yolo_obb",
        }
    return {
        "ok": True,
        "boxes": boxes,
        "num_detections": len(boxes),
        "detection_method": "yolo_obb",
        "fallback": False,
    }


def check_detection_available():
    try:
        from ultralytics import YOLO  # noqa: F401

        return {"available": True, "primary_method": "yolo_obb"}
    except Exception:
        return {"available": False, "primary_method": "yolo_obb"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_specimen.py <image_path> [--yolo-model <path>] [--multi] | --check")
        sys.exit(1)

    if sys.argv[1] == "--check":
        print(json.dumps(check_detection_available()))
        sys.exit(0)

    yolo_model = None
    conf_threshold = 0.25
    nms_iou = None
    max_specimens = 20
    detection_preset = "balanced"
    imgsz = None
    args = sys.argv[1:]
    if "--yolo-model" in args:
        idx = args.index("--yolo-model")
        if idx + 1 < len(args):
            yolo_model = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
    if "--conf" in args:
        idx = args.index("--conf")
        if idx + 1 < len(args):
            conf_threshold = float(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
    if "--nms-iou" in args:
        idx = args.index("--nms-iou")
        if idx + 1 < len(args):
            nms_iou = float(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
    if "--max-specimens" in args:
        idx = args.index("--max-specimens")
        if idx + 1 < len(args):
            max_specimens = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
    if "--detection-preset" in args:
        idx = args.index("--detection-preset")
        if idx + 1 < len(args):
            detection_preset = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
    if "--imgsz" in args:
        idx = args.index("--imgsz")
        if idx + 1 < len(args):
            imgsz = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]

    image_path = args[0]
    if "--multi" in args:
        result = detect_multiple_specimens(
            image_path,
            yolo_model_path=yolo_model,
            conf_threshold=conf_threshold,
            max_specimens=max_specimens,
            nms_iou=nms_iou,
            detection_preset=detection_preset,
            imgsz=imgsz,
        )
        print(json.dumps(result, indent=2))
    else:
        result = detect_specimen(
            image_path,
            yolo_model_path=yolo_model,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
            detection_preset=detection_preset,
            imgsz=imgsz,
        )
        print(json.dumps(result))
