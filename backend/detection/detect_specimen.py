import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from detection_utils import normalize_orientation_payload
from obb_utils import (
    canonicalize_detector_obb_corners,
    iter_ultralytics_obb,
    load_obb_confidence_threshold,
    load_obb_nms_iou,
    resolve_obb_detection_preset as _resolve_obb_detection_preset,
)


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
    img_h, img_w = result.orig_shape[:2]
    parsed = []
    for box in iter_ultralytics_obb(result, max_objects=max_specimens):
        corners = box["corners"]
        class_id = box["class_id"]
        confidence = box["confidence"]
        xs = [float(point[0]) for point in corners]
        ys = [float(point[1]) for point in corners]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
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
                "class_name": box["class_name"],
                "detection_method": "yolo_obb",
                "obbCorners": corners,
                "angle": box["angle"],
                **({"orientation_hint": orientation_hint} if orientation_hint else {}),
            }
        )
    # Ultralytics has already applied oriented NMS.  A second AABB-envelope
    # pass would incorrectly remove crossing or closely packed rotated objects.
    parsed.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)
    return parsed


def detect_with_yolo(
    image_path,
    model_path,
    conf_threshold=None,
    margin=20,
    orientation_policy=None,
    nms_iou=None,
    detection_preset=None,
    imgsz=None,
):
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    if not model_path or not os.path.exists(model_path):
        return None

    resolved = _resolve_obb_detection_preset(
        conf_threshold=(
            load_obb_confidence_threshold(model_path, default=0.3)
            if conf_threshold is None
            else conf_threshold
        ),
        nms_iou=load_obb_nms_iou(model_path) if nms_iou is None else nms_iou,
        max_objects=1,
        detection_preset=("custom" if detection_preset is None else detection_preset),
        imgsz=imgsz,
    )
    model = YOLO(model_path)
    results = model.predict(
        image_path,
        conf=float(resolved["conf"]),
        iou=float(resolved["iou"]),
        imgsz=int(resolved["imgsz"]),
        task="obb",
        agnostic_nms=True,
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
    conf_threshold=None,
    margin=20,
    max_specimens=20,
    nms_iou=None,
    orientation_policy=None,
    detection_preset=None,
    imgsz=None,
):
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    if not model_path or not os.path.exists(model_path):
        return None

    resolved = _resolve_obb_detection_preset(
        conf_threshold=(
            load_obb_confidence_threshold(model_path, default=0.3)
            if conf_threshold is None
            else conf_threshold
        ),
        nms_iou=load_obb_nms_iou(model_path) if nms_iou is None else nms_iou,
        max_objects=max_specimens,
        detection_preset=("custom" if detection_preset is None else detection_preset),
        imgsz=imgsz,
    )
    predict_kwargs = {
        "conf": float(resolved["conf"]),
        "iou": float(resolved["iou"]),
        "imgsz": int(resolved["imgsz"]),
        "task": "obb",
        "agnostic_nms": True,
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
    conf_threshold=None,
    nms_iou=None,
    detection_preset=None,
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
    conf_threshold=None,
    detection_preset=None,
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
    conf_threshold = None
    nms_iou = None
    max_specimens = 20
    detection_preset = None
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
