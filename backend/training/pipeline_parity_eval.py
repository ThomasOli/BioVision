"""
Pipeline-parity evaluator for landmark models.

Purpose:
  Evaluate the *runtime inference pipeline* (crop -> optional mask/PCA canonicalization
  -> orientation selection -> inverse mapping) on train/test XML images.

This complements train/test errors from model trainers by measuring pixel-space error
after runtime preprocessing and mapping.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

import numpy as np


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _xywh_to_xyxy(box: Mapping[str, Any]) -> tuple[float, float, float, float]:
    left = _safe_float(box.get("left"), 0.0)
    top = _safe_float(box.get("top"), 0.0)
    width = _safe_float(box.get("width"), 0.0)
    height = _safe_float(box.get("height"), 0.0)
    right = left + max(1.0, width)
    bottom = top + max(1.0, height)
    return left, top, right, bottom


def _iou_xywh(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(a_area + b_area - inter, 1e-6)
    return float(inter / union)


def _resolve_session_yolo_model(project_root: str) -> str | None:
    model_dir = os.path.join(project_root, "models")
    if not os.path.isdir(model_dir):
        return None
    candidates = glob.glob(os.path.join(model_dir, "yolo_whole_*.pt"))
    if not candidates:
        return None

    def _version_key(path: str) -> tuple[int, float]:
        name = os.path.basename(path)
        match = re.search(r"_v(\d+)\.pt$", name, flags=re.IGNORECASE)
        version = int(match.group(1)) if match else -1
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        return (version, mtime)

    candidates.sort(key=_version_key, reverse=True)
    return candidates[0]


def _load_part_name_mapping(project_root: str, tag: str) -> tuple[dict[str, int], dict[int, int]]:
    path = os.path.join(project_root, "debug", f"id_mapping_{tag}.json")
    if not os.path.exists(path):
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}, {}

    name_map: dict[str, int] = {}
    index_map: dict[int, int] = {}

    raw_name_map = data.get("dlib_name_to_original", {})
    if isinstance(raw_name_map, Mapping):
        for k, v in raw_name_map.items():
            try:
                name_map[str(k)] = int(v)
            except Exception:
                continue

    raw_index_map = data.get("dlib_index_to_original", {})
    if isinstance(raw_index_map, Mapping):
        for k, v in raw_index_map.items():
            try:
                index_map[int(k)] = int(v)
            except Exception:
                continue

    return name_map, index_map


def _load_head_tail_ids(project_root: str, tag: str) -> tuple[int | None, int | None]:
    """
    Resolve head/tail IDs for orientation hints during parity eval.
    """
    # Prefer run/debug orientation artifact.
    orientation_path = os.path.join(project_root, "debug", f"orientation_{tag}.json")
    if os.path.exists(orientation_path):
        try:
            with open(orientation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            head_id = data.get("head_landmark_id")
            tail_id = data.get("tail_landmark_id")
            head_id = int(head_id) if head_id is not None else None
            tail_id = int(tail_id) if tail_id is not None else None
            return head_id, tail_id
        except Exception:
            pass

    # Fallback: session schema categories.
    session_path = os.path.join(project_root, "session.json")
    if not os.path.exists(session_path):
        return None, None
    try:
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

    def _norm_targets(raw, fallback):
        if isinstance(raw, (list, tuple)):
            vals = [str(v).strip().lower() for v in raw if str(v).strip()]
            if vals:
                return set(vals)
        return set(str(v).strip().lower() for v in fallback if str(v).strip())

    head_targets = _norm_targets(orientation_policy.get("headCategories"), ["head"])
    tail_targets = _norm_targets(orientation_policy.get("tailCategories"), ["tail"])

    head_id = None
    tail_id = None
    for lm in template:
        try:
            idx = int(lm.get("index"))
        except Exception:
            continue
        cat = str(lm.get("category", "")).strip().lower()
        if cat in head_targets and head_id is None:
            head_id = idx
        elif cat in tail_targets and tail_id is None:
            tail_id = idx
    return head_id, tail_id


def _build_orientation_hint_from_gt(
    gt_landmarks: Sequence[Mapping[str, Any]],
    *,
    head_id: int | None,
    tail_id: int | None,
) -> dict[str, Any] | None:
    """
    Build detector-style orientation_hint from GT landmarks for parity eval.
    """
    by_id: dict[int, tuple[float, float]] = {}
    for lm in gt_landmarks:
        try:
            lm_id = int(lm.get("id"))
            x = float(lm.get("x"))
            y = float(lm.get("y"))
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        by_id[lm_id] = (x, y)

    if not by_id:
        return None

    head_pt = by_id.get(int(head_id)) if head_id is not None else None
    tail_pt = by_id.get(int(tail_id)) if tail_id is not None else None

    if head_pt is None:
        try:
            first_id = min(by_id.keys())
            head_pt = by_id[first_id]
        except Exception:
            return None
    if tail_pt is None:
        # Fallback to farthest landmark from head.
        hx, hy = head_pt
        far = None
        far_d = -1.0
        for lm_id, pt in by_id.items():
            if head_id is not None and int(lm_id) == int(head_id):
                continue
            dx = float(pt[0]) - float(hx)
            dy = float(pt[1]) - float(hy)
            d = dx * dx + dy * dy
            if d > far_d:
                far_d = d
                far = pt
        tail_pt = far

    if head_pt is None or tail_pt is None:
        return None

    hx, hy = float(head_pt[0]), float(head_pt[1])
    tx, ty = float(tail_pt[0]), float(tail_pt[1])
    orientation = "left" if hx < tx else "right"
    return {
        "orientation": orientation,
        "confidence": 1.0,
        "head_point": [hx, hy],
        "tail_point": [tx, ty],
        "source": "parity_gt_landmarks",
    }


def _map_part_name_to_id(
    part_name: str,
    name_map: Mapping[str, int],
    index_map: Mapping[int, int],
) -> int:
    if part_name in name_map:
        return int(name_map[part_name])
    try:
        idx = int(part_name)
    except Exception:
        return _safe_int(part_name, 0)
    if idx in index_map:
        return int(index_map[idx])
    return int(idx)


def _parse_xml_records(
    xml_path: str,
    *,
    part_name_map: Mapping[str, int],
    part_index_map: Mapping[int, int],
) -> list[dict[str, Any]]:
    if not xml_path or not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []

    images_el = root.find("images")
    if images_el is None:
        return []

    records: list[dict[str, Any]] = []
    for image_el in images_el.findall("image"):
        image_path = str(image_el.get("file", "")).strip()
        if not image_path or not os.path.exists(image_path):
            continue
        for box_el in image_el.findall("box"):
            left = _safe_int(box_el.get("left"), 0)
            top = _safe_int(box_el.get("top"), 0)
            width = max(1, _safe_int(box_el.get("width"), 1))
            height = max(1, _safe_int(box_el.get("height"), 1))
            gt_landmarks: list[dict[str, float | int]] = []
            for part in box_el.findall("part"):
                name = str(part.get("name", "")).strip()
                if not name:
                    continue
                x = _safe_float(part.get("x"), -1.0)
                y = _safe_float(part.get("y"), -1.0)
                if x < 0 or y < 0:
                    continue
                gt_landmarks.append(
                    {
                        "id": _map_part_name_to_id(name, part_name_map, part_index_map),
                        "x": float(x),
                        "y": float(y),
                    }
                )
            if not gt_landmarks:
                continue
            records.append(
                {
                    "image_path": image_path,
                    "box": {
                        "left": float(left),
                        "top": float(top),
                        "width": float(width),
                        "height": float(height),
                        "right": float(left + width),
                        "bottom": float(top + height),
                    },
                    "landmarks": gt_landmarks,
                }
            )
    return records


def _to_id_map(landmarks: Sequence[Mapping[str, Any]]) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for lm in landmarks:
        try:
            lm_id = int(lm.get("id"))
            x = float(lm.get("x"))
            y = float(lm.get("y"))
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        out[lm_id] = (x, y)
    return out


def _match_specimen_by_iou(
    specimens: Sequence[Mapping[str, Any]],
    gt_box: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, float]:
    best = None
    best_iou = -1.0
    for specimen in specimens:
        box = specimen.get("box")
        if not isinstance(box, Mapping):
            continue
        iou = _iou_xywh(box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best = specimen
    return best, max(0.0, best_iou)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    try:
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))
    except Exception:
        return None


def _build_metric_summary(
    *,
    image_total: int,
    image_predicted: int,
    image_missing_specimen: int,
    image_errors: int,
    landmark_total: int,
    landmark_matched: int,
    pixel_errors: Sequence[float],
    ious: Sequence[float],
    runtime_sec: float,
) -> dict[str, Any]:
    mean_err = float(np.mean(pixel_errors)) if pixel_errors else None
    med_err = _percentile(pixel_errors, 50.0)
    p90_err = _percentile(pixel_errors, 90.0)
    max_err = float(np.max(pixel_errors)) if pixel_errors else None
    min_err = float(np.min(pixel_errors)) if pixel_errors else None
    mean_iou = float(np.mean(ious)) if ious else None
    med_iou = _percentile(ious, 50.0)
    return {
        "images_total": int(image_total),
        "images_predicted": int(image_predicted),
        "images_missing_specimen": int(image_missing_specimen),
        "images_with_errors": int(image_errors),
        "landmark_total": int(landmark_total),
        "landmark_matched": int(landmark_matched),
        "landmark_coverage": float(landmark_matched / landmark_total) if landmark_total > 0 else None,
        "pixel_error_mean": mean_err,
        "pixel_error_median": med_err,
        "pixel_error_p90": p90_err,
        "pixel_error_min": min_err,
        "pixel_error_max": max_err,
        "mean_iou": mean_iou,
        "median_iou": med_iou,
        "runtime_sec": float(runtime_sec),
    }


def _inc_counter(bucket: dict[str, int], key: Any) -> None:
    k = str(key if key is not None else "none")
    bucket[k] = int(bucket.get(k, 0)) + 1


def _emit_progress_json(payload: Mapping[str, Any]) -> None:
    try:
        print("PROGRESS_JSON " + json.dumps(dict(payload)), file=sys.stderr)
    except Exception:
        pass


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "image"


def _evaluate_records(
    *,
    project_root: str,
    tag: str,
    predictor_type: str,
    records: Sequence[Mapping[str, Any]],
    yolo_model_path: str | None,
    use_gt_boxes: bool,
    split_name: str,
    eval_name: str,
    outlier_px_threshold: float = 400.0,
    max_outlier_cases: int = 200,
    debug_output_dir: str | None = None,
    head_id: int | None = None,
    tail_id: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from predict import predict_cnn_multi_specimen, predict_multi_specimen

    start_t = time.time()
    total_records = int(len(records))
    image_total = 0
    image_predicted = 0
    image_missing_specimen = 0
    image_errors = 0
    landmark_total = 0
    landmark_matched = 0
    pixel_errors: list[float] = []
    ious: list[float] = []
    warning_codes: dict[str, int] = {}
    direction_sources: dict[str, int] = {}
    selection_reasons: dict[str, int] = {}
    mask_sources: dict[str, int] = {}
    detector_hint_orientations: dict[str, int] = {}
    orientation_signal = {
        "images_with_inference_metadata": 0,
        "detector_hint_present": 0,
        "detector_hint_missing": 0,
        "canonical_flip_applied_true": 0,
        "was_flipped_true": 0,
    }
    outlier_cases: list[dict[str, Any]] = []
    outlier_threshold = float(max(0.0, outlier_px_threshold))
    outlier_dir = None
    if debug_output_dir:
        outlier_dir = os.path.join(
            os.path.abspath(debug_output_dir),
            "parity_outliers",
            split_name,
            eval_name,
        )
        os.makedirs(outlier_dir, exist_ok=True)

    report_every = max(1, min(25, max(1, total_records // 20)))
    _emit_progress_json(
        {
            "percent": 98,
            "stage": "evaluation",
            "substage": "pipeline_parity_split_start",
            "message": f"Parity {split_name}/{eval_name}: evaluating {total_records} records...",
            "split": split_name,
            "eval_mode": eval_name,
            "records_total": total_records,
            "records_done": 0,
        }
    )

    for rec_idx, rec in enumerate(records):
        if rec_idx == 0 or ((rec_idx + 1) % report_every == 0) or (rec_idx + 1 == total_records):
            _emit_progress_json(
                {
                    "percent": 98 if (rec_idx + 1) < total_records else 99,
                    "stage": "evaluation",
                    "substage": "pipeline_parity_split_progress",
                    "message": (
                        f"Parity {split_name}/{eval_name}: "
                        f"{rec_idx + 1}/{total_records} records"
                    ),
                    "split": split_name,
                    "eval_mode": eval_name,
                    "records_total": total_records,
                    "records_done": int(rec_idx + 1),
                }
            )
        image_total += 1
        image_path = str(rec.get("image_path", ""))
        gt_box = rec.get("box", {})
        gt_landmarks = rec.get("landmarks", [])
        landmark_total += len(gt_landmarks)
        if not image_path or not os.path.exists(image_path) or not isinstance(gt_box, Mapping):
            image_errors += 1
            continue

        input_boxes = None
        if use_gt_boxes:
            input_box = dict(gt_box)
            gt_hint = _build_orientation_hint_from_gt(
                gt_landmarks,
                head_id=head_id,
                tail_id=tail_id,
            )
            if gt_hint is not None:
                input_box["orientation_hint"] = gt_hint
            input_boxes = [input_box]

        try:
            if predictor_type == "cnn":
                out = predict_cnn_multi_specimen(
                    project_root,
                    tag,
                    image_path,
                    yolo_model_path=yolo_model_path,
                    input_boxes=input_boxes,
                )
            else:
                out = predict_multi_specimen(
                    project_root,
                    tag,
                    image_path,
                    yolo_model_path=yolo_model_path,
                    input_boxes=input_boxes,
                )
        except Exception:
            image_errors += 1
            continue

        specimens = out.get("specimens", []) if isinstance(out, Mapping) else []
        if not isinstance(specimens, list) or len(specimens) == 0:
            image_missing_specimen += 1
            continue

        pred_specimen, best_iou = _match_specimen_by_iou(specimens, gt_box)
        if pred_specimen is None:
            image_missing_specimen += 1
            continue

        ious.append(float(best_iou))
        pred_lms = pred_specimen.get("landmarks", []) if isinstance(pred_specimen, Mapping) else []
        gt_map = _to_id_map(gt_landmarks)
        pred_map = _to_id_map(pred_lms if isinstance(pred_lms, Sequence) else [])
        if not gt_map or not pred_map:
            image_errors += 1
            continue

        inference_meta = pred_specimen.get("inference_metadata", {}) if isinstance(pred_specimen, Mapping) else {}
        if isinstance(inference_meta, Mapping) and inference_meta:
            orientation_signal["images_with_inference_metadata"] += 1
            hint_ori = inference_meta.get("detector_hint_orientation")
            if hint_ori in ("left", "right"):
                orientation_signal["detector_hint_present"] += 1
            else:
                orientation_signal["detector_hint_missing"] += 1
            if bool(inference_meta.get("canonical_flip_applied")):
                orientation_signal["canonical_flip_applied_true"] += 1
            if bool(inference_meta.get("was_flipped")):
                orientation_signal["was_flipped_true"] += 1
            _inc_counter(direction_sources, inference_meta.get("direction_source"))
            _inc_counter(selection_reasons, inference_meta.get("selection_reason"))
            _inc_counter(mask_sources, inference_meta.get("mask_source"))
            _inc_counter(detector_hint_orientations, hint_ori)
            warn = inference_meta.get("orientation_warning")
            if isinstance(warn, Mapping):
                _inc_counter(warning_codes, warn.get("code"))
            elif isinstance(warn, str):
                _inc_counter(warning_codes, warn)

        matched_here = 0
        local_errors: list[float] = []
        for lm_id, (gx, gy) in gt_map.items():
            pred_xy = pred_map.get(int(lm_id))
            if pred_xy is None:
                continue
            px, py = pred_xy
            dist = math.hypot(float(px) - float(gx), float(py) - float(gy))
            pixel_errors.append(float(dist))
            local_errors.append(float(dist))
            matched_here += 1
        landmark_matched += matched_here
        if matched_here > 0:
            image_predicted += 1
        else:
            image_errors += 1
            continue

        if local_errors and max(local_errors) >= outlier_threshold:
            case = {
                "split": split_name,
                "eval_mode": eval_name,
                "record_index": int(rec_idx),
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "max_pixel_error": float(max(local_errors)),
                "mean_pixel_error": float(np.mean(local_errors)),
                "p90_pixel_error": float(np.percentile(np.asarray(local_errors, dtype=np.float64), 90)),
                "matched_landmarks": int(matched_here),
                "total_gt_landmarks": int(len(gt_map)),
                "best_iou": float(best_iou),
                "gt_box": dict(gt_box) if isinstance(gt_box, Mapping) else {},
                "pred_box": dict(pred_specimen.get("box", {})) if isinstance(pred_specimen, Mapping) else {},
                "inference_metadata": dict(inference_meta) if isinstance(inference_meta, Mapping) else {},
            }
            if len(outlier_cases) < max_outlier_cases:
                outlier_cases.append(case)
            if outlier_dir:
                safe_base = _safe_filename(os.path.splitext(os.path.basename(image_path))[0])
                stem = f"{rec_idx:04d}__{safe_base}"
                ext = os.path.splitext(image_path)[1] or ".png"
                try:
                    shutil.copy2(image_path, os.path.join(outlier_dir, f"{stem}{ext}"))
                except Exception:
                    pass
                try:
                    with open(os.path.join(outlier_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                        json.dump(case, f, indent=2)
                except Exception:
                    pass

    summary = _build_metric_summary(
        image_total=image_total,
        image_predicted=image_predicted,
        image_missing_specimen=image_missing_specimen,
        image_errors=image_errors,
        landmark_total=landmark_total,
        landmark_matched=landmark_matched,
        pixel_errors=pixel_errors,
        ious=ious,
        runtime_sec=time.time() - start_t,
    )
    summary[f"outlier_ge_{int(round(outlier_threshold))}px_count"] = int(len(outlier_cases))

    diagnostics = {
        "outlier_threshold_px": float(outlier_threshold),
        "orientation_signal": {
            **orientation_signal,
            "warning_code_counts": warning_codes,
            "direction_source_counts": direction_sources,
            "selection_reason_counts": selection_reasons,
            "mask_source_counts": mask_sources,
            "detector_hint_orientation_counts": detector_hint_orientations,
        },
        "outlier_cases": outlier_cases,
        "outlier_dir": outlier_dir,
    }
    _emit_progress_json(
        {
            "percent": 99,
            "stage": "evaluation",
            "substage": "pipeline_parity_split_done",
            "message": (
                f"Parity {split_name}/{eval_name} complete: "
                f"{summary.get('images_predicted', 0)}/{summary.get('images_total', 0)} predicted"
            ),
            "split": split_name,
            "eval_mode": eval_name,
            "records_total": total_records,
            "records_done": total_records,
            "runtime_sec": float(summary.get("runtime_sec") or 0.0),
        }
    )
    return summary, diagnostics


def evaluate_pipeline_parity(
    *,
    project_root: str,
    tag: str,
    predictor_type: str,
    train_xml: str,
    test_xml: str | None = None,
    debug_output_dir: str | None = None,
    outlier_px_threshold: float = 400.0,
    skip_detected_boxes: bool = False,
) -> dict[str, Any]:
    """
    Evaluate runtime inference parity for train/test splits.

    Returns split metrics for:
      - gt_boxes: provided GT boxes (isolates detector-box error)
      - detected_boxes: runtime detector boxes (full production path)
    """
    predictor = str(predictor_type or "dlib").strip().lower()
    if predictor not in ("dlib", "cnn"):
        return {"error": f"Unsupported predictor_type '{predictor}'"}

    name_map, index_map = _load_part_name_mapping(project_root, tag)
    head_id, tail_id = _load_head_tail_ids(project_root, tag)
    train_records = _parse_xml_records(
        train_xml,
        part_name_map=name_map,
        part_index_map=index_map,
    )
    test_records = _parse_xml_records(
        test_xml,
        part_name_map=name_map,
        part_index_map=index_map,
    ) if test_xml and os.path.exists(test_xml) else []

    max_records = 0
    try:
        max_records = int(os.environ.get("BV_PARITY_MAX_RECORDS", "0"))
    except Exception:
        max_records = 0
    if max_records > 0:
        train_records = train_records[:max_records]
        test_records = test_records[:max_records]

    yolo_model_path = _resolve_session_yolo_model(project_root)
    out: dict[str, Any] = {
        "predictor_type": predictor,
        "yolo_model_path": yolo_model_path,
        "train_xml": train_xml,
        "test_xml": test_xml if test_xml and os.path.exists(test_xml) else None,
        "debug_output_dir": os.path.abspath(debug_output_dir) if debug_output_dir else None,
        "outlier_px_threshold": float(max(0.0, outlier_px_threshold)),
        "skip_detected_boxes": bool(skip_detected_boxes),
        "head_landmark_id": head_id,
        "tail_landmark_id": tail_id,
        "records": {
            "train": len(train_records),
            "test": len(test_records),
            "max_records_override": max_records if max_records > 0 else None,
        },
        "splits": {},
        "orientation_signal_summary": {},
    }
    _emit_progress_json(
        {
            "percent": 98,
            "stage": "evaluation",
            "substage": "pipeline_parity_start",
            "message": (
                f"Starting parity evaluation for {predictor} "
                f"(train={len(train_records)}, test={len(test_records)})."
            ),
            "predictor_type": predictor,
            "train_records": int(len(train_records)),
            "test_records": int(len(test_records)),
        }
    )

    def _eval_split(split_name: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            return {
                "gt_boxes": _build_metric_summary(
                    image_total=0,
                    image_predicted=0,
                    image_missing_specimen=0,
                    image_errors=0,
                    landmark_total=0,
                    landmark_matched=0,
                    pixel_errors=[],
                    ious=[],
                    runtime_sec=0.0,
                ),
                "gt_boxes_diagnostics": {
                    "outlier_threshold_px": float(max(0.0, outlier_px_threshold)),
                    "orientation_signal": {
                        "images_with_inference_metadata": 0,
                        "detector_hint_present": 0,
                        "detector_hint_missing": 0,
                        "canonical_flip_applied_true": 0,
                        "was_flipped_true": 0,
                        "warning_code_counts": {},
                        "direction_source_counts": {},
                        "selection_reason_counts": {},
                        "mask_source_counts": {},
                        "detector_hint_orientation_counts": {},
                    },
                    "outlier_cases": [],
                    "outlier_dir": None,
                },
                "detected_boxes": _build_metric_summary(
                    image_total=0,
                    image_predicted=0,
                    image_missing_specimen=0,
                    image_errors=0,
                    landmark_total=0,
                    landmark_matched=0,
                    pixel_errors=[],
                    ious=[],
                    runtime_sec=0.0,
                ),
                "detected_boxes_diagnostics": {
                    "outlier_threshold_px": float(max(0.0, outlier_px_threshold)),
                    "orientation_signal": {
                        "images_with_inference_metadata": 0,
                        "detector_hint_present": 0,
                        "detector_hint_missing": 0,
                        "canonical_flip_applied_true": 0,
                        "was_flipped_true": 0,
                        "warning_code_counts": {},
                        "direction_source_counts": {},
                        "selection_reason_counts": {},
                        "mask_source_counts": {},
                        "detector_hint_orientation_counts": {},
                    },
                    "outlier_cases": [],
                    "outlier_dir": None,
                },
            }
        gt_summary, gt_diag = _evaluate_records(
            project_root=project_root,
            tag=tag,
            predictor_type=predictor,
            records=records,
            yolo_model_path=yolo_model_path,
            use_gt_boxes=True,
            split_name=split_name,
            eval_name="gt_boxes",
            outlier_px_threshold=outlier_px_threshold,
            debug_output_dir=debug_output_dir,
            head_id=head_id,
            tail_id=tail_id,
        )
        if bool(skip_detected_boxes):
            det_summary = _build_metric_summary(
                image_total=int(len(records)),
                image_predicted=0,
                image_missing_specimen=0,
                image_errors=0,
                landmark_total=0,
                landmark_matched=0,
                pixel_errors=[],
                ious=[],
                runtime_sec=0.0,
            )
            det_diag = {
                "outlier_threshold_px": float(max(0.0, outlier_px_threshold)),
                "orientation_signal": {
                    "images_with_inference_metadata": 0,
                    "detector_hint_present": 0,
                    "detector_hint_missing": 0,
                    "canonical_flip_applied_true": 0,
                    "was_flipped_true": 0,
                    "warning_code_counts": {},
                    "direction_source_counts": {},
                    "selection_reason_counts": {},
                    "mask_source_counts": {},
                    "detector_hint_orientation_counts": {},
                },
                "outlier_cases": [],
                "outlier_dir": None,
                "skipped": True,
                "reason": "skip_detected_boxes",
            }
        else:
            det_summary, det_diag = _evaluate_records(
                project_root=project_root,
                tag=tag,
                predictor_type=predictor,
                records=records,
                yolo_model_path=yolo_model_path,
                use_gt_boxes=False,
                split_name=split_name,
                eval_name="detected_boxes",
                outlier_px_threshold=outlier_px_threshold,
                debug_output_dir=debug_output_dir,
                head_id=head_id,
                tail_id=tail_id,
            )
        return {
            "gt_boxes": gt_summary,
            "gt_boxes_diagnostics": gt_diag,
            "detected_boxes": det_summary,
            "detected_boxes_diagnostics": det_diag,
        }

    out["splits"]["train"] = _eval_split("train", train_records)
    out["splits"]["test"] = _eval_split("test", test_records)

    for split_name in ("train", "test"):
        split_obj = out["splits"].get(split_name, {})
        if not isinstance(split_obj, Mapping):
            continue
        out["orientation_signal_summary"][split_name] = {
            "gt_boxes": split_obj.get("gt_boxes_diagnostics", {}).get("orientation_signal", {}),
            "detected_boxes": split_obj.get("detected_boxes_diagnostics", {}).get("orientation_signal", {}),
        }

    if debug_output_dir:
        parity_dir = os.path.join(os.path.abspath(debug_output_dir), "parity_outliers")
        try:
            os.makedirs(parity_dir, exist_ok=True)
            with open(os.path.join(parity_dir, "parity_report.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception:
            pass

    _emit_progress_json(
        {
            "percent": 99,
            "stage": "evaluation",
            "substage": "pipeline_parity_done",
            "message": "Pipeline parity evaluation complete.",
        }
    )

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate runtime inference parity on train/test XML.")
    parser.add_argument("project_root")
    parser.add_argument("tag")
    parser.add_argument("--predictor-type", choices=["dlib", "cnn"], default="dlib")
    parser.add_argument("--train-xml", default=None)
    parser.add_argument("--test-xml", default=None)
    parser.add_argument("--debug-output-dir", default=None)
    parser.add_argument("--outlier-px-threshold", type=float, default=400.0)
    parser.add_argument("--skip-detected-boxes", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.project_root)
    train_xml = args.train_xml or os.path.join(root, "xml", f"train_{args.tag}.xml")
    test_xml = args.test_xml or os.path.join(root, "xml", f"test_{args.tag}.xml")
    report = evaluate_pipeline_parity(
        project_root=root,
        tag=args.tag,
        predictor_type=args.predictor_type,
        train_xml=train_xml,
        test_xml=test_xml if os.path.exists(test_xml) else None,
        debug_output_dir=args.debug_output_dir,
        outlier_px_threshold=args.outlier_px_threshold,
        skip_detected_boxes=bool(args.skip_detected_boxes),
    )
    print(json.dumps(report, indent=2))
