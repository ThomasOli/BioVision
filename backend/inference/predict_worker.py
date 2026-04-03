#!/usr/bin/env python3
"""
Persistent landmark inference worker for batch inference UX.

Loads a dlib/CNN landmark model once and reuses it across repeated image
prediction requests. This worker only supports the current inference-page flow:
multi-specimen landmarking from provided OBB boxes.
"""

import json
import os
import sys
import traceback

import dlib

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import bv_utils.orientation_utils as ou
from inference.predict import (
    STANDARD_SIZE,
    _detect_multi_obb_boxes,
    _load_and_resize_for_inference,
    _load_cnn_model,
    _make_cnn_predict_fn,
    _make_dlib_predict_fn,
    _resolve_dlib_index_mapping,
    _resolve_head_landmark_id,
    _resolve_tail_landmark_id,
    _run_obb_inference_on_box,
)


def _send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class LandmarkPredictWorker:
    def __init__(self):
        self.loaded_key = None
        self.context = None

    def _emit_progress(self, request_id, percent, stage, **extra):
        payload = {
            "status": "progress",
            "_request_id": request_id,
            "percent": percent,
            "stage": stage,
        }
        payload.update(extra)
        _send(payload)

    def _load_dlib_context(self, project_root, tag):
        debug_dir = os.path.join(project_root, "debug")
        predictor_path = os.path.join(project_root, "models", f"predictor_{tag}.dat")
        id_mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Model not found: {predictor_path}")

        orientation_policy = {}
        try:
            orientation_policy = ou.load_orientation_policy(project_root)
        except Exception:
            orientation_policy = {}

        id_mapping = {}
        index_to_original = {}
        target_orientation = None
        landmark_template = {}
        head_landmark_id = None
        tail_landmark_id = _resolve_tail_landmark_id(project_root, None)
        if os.path.exists(id_mapping_path):
            with open(id_mapping_path, "r", encoding="utf-8") as handle:
                id_mapping = json.load(handle)
            index_to_original = _resolve_dlib_index_mapping(project_root, tag, id_mapping)
            target_orientation = id_mapping.get("training_config", {}).get("target_orientation")
            landmark_template = {
                int(k): v for k, v in id_mapping.get("landmark_template", {}).items()
            }
            head_landmark_id = _resolve_head_landmark_id(project_root, id_mapping)
            tail_landmark_id = _resolve_tail_landmark_id(project_root, id_mapping)
        else:
            head_landmark_id = _resolve_head_landmark_id(project_root, None)

        rect = dlib.rectangle(0, 0, STANDARD_SIZE, STANDARD_SIZE)
        predictor = dlib.shape_predictor(predictor_path)
        return {
            "project_root": project_root,
            "tag": tag,
            "predictor_type": "dlib",
            "orientation_policy": orientation_policy,
            "predict_fn": _make_dlib_predict_fn(predictor, rect, index_to_original),
            "target_orientation": target_orientation,
            "landmark_template": landmark_template,
            "head_landmark_id": head_landmark_id,
            "tail_landmark_id": tail_landmark_id,
        }

    def _load_cnn_context(self, project_root, tag):
        orientation_policy = {}
        try:
            orientation_policy = ou.load_orientation_policy(project_root)
        except Exception:
            orientation_policy = {}

        model, landmark_ids, target_orientation, landmark_template, head_landmark_id, tail_landmark_id = _load_cnn_model(
            project_root,
            tag,
        )
        return {
            "project_root": project_root,
            "tag": tag,
            "predictor_type": "cnn",
            "orientation_policy": orientation_policy,
            "predict_fn": _make_cnn_predict_fn(model, landmark_ids),
            "target_orientation": target_orientation,
            "landmark_template": landmark_template,
            "head_landmark_id": head_landmark_id,
            "tail_landmark_id": tail_landmark_id,
        }

    def ensure_context(self, request_id, project_root, tag, predictor_type):
        key = (os.path.abspath(project_root), str(tag), str(predictor_type))
        if self.loaded_key == key and self.context is not None:
            return False

        self._emit_progress(request_id, 10, "loading_model")
        if predictor_type == "cnn":
            self.context = self._load_cnn_context(project_root, tag)
        else:
            self.context = self._load_dlib_context(project_root, tag)
        self.loaded_key = key
        return True

    def predict(self, request_id, payload):
        project_root = payload["project_root"]
        tag = payload["tag"]
        predictor_type = payload.get("predictor_type", "dlib")
        image_path = payload["image_path"]
        input_boxes = payload.get("boxes")

        cold_start = self.ensure_context(request_id, project_root, tag, predictor_type)
        ctx = self.context
        if ctx is None:
            raise RuntimeError("Landmark worker context not initialized.")
        if not isinstance(input_boxes, list) or len(input_boxes) == 0:
            raise ValueError("predict worker requires provided OBB boxes.")

        self._emit_progress(request_id, 20, "detecting")
        img_original, img_detector, orig_w, orig_h, scale, detector_w, detector_h = _load_and_resize_for_inference(image_path)
        detection_result = _detect_multi_obb_boxes(
            image_path,
            img_detector,
            scale,
            detector_w,
            detector_h,
            input_boxes=input_boxes,
            original_w=orig_w,
            original_h=orig_h,
        )
        detected_boxes = detection_result["boxes"]

        specimens = []
        clamp_debug = []
        total = max(1, len(detected_boxes))
        for box_idx, box in enumerate(detected_boxes):
            pct = 35 + int(50 * ((box_idx + 1) / total))
            self._emit_progress(
                request_id,
                pct,
                "predicting",
                current_specimen=box_idx + 1,
                total_specimens=len(detected_boxes),
            )
            obb_prediction = _run_obb_inference_on_box(
                img_original=img_original,
                box=box,
                orig_h=orig_h,
                orig_w=orig_w,
                detector_scale=scale,
                detector_w=detector_w,
                detector_h=detector_h,
                orientation_policy=ctx["orientation_policy"],
                predict_fn=ctx["predict_fn"],
                target_orientation=ctx["target_orientation"],
                landmark_template=ctx["landmark_template"],
                head_landmark_id=ctx["head_landmark_id"],
                tail_landmark_id=ctx["tail_landmark_id"],
            )
            specimens.append({
                "box": obb_prediction["detected_box"],
                "landmarks": obb_prediction["landmarks"],
                "num_landmarks": len(obb_prediction["landmarks"]),
                "orientation_debug": obb_prediction["orientation_debug"],
                "inference_metadata": obb_prediction["inference_metadata"],
                "resolution_debug": obb_prediction["resolution_debug"],
                "mask_outline": None,
            })
            clamp_entry = {
                "box_index": box_idx,
                "clamped_landmark_ids": obb_prediction.get("clamped_landmark_ids", []),
                "clamped_landmark_count": obb_prediction.get("clamped_landmark_count", 0),
                "resolution_debug": obb_prediction["resolution_debug"],
            }
            if obb_prediction.get("pre_clamp_landmarks"):
                clamp_entry["pre_clamp_landmarks"] = obb_prediction["pre_clamp_landmarks"]
            clamp_debug.append(clamp_entry)

        self._emit_progress(request_id, 92, "mapping")
        return {
            "image": image_path,
            "specimens": specimens,
            "num_specimens": len(specimens),
            "image_dimensions": {"width": orig_w, "height": orig_h},
            "inference_scale": scale,
            "detection_method": detection_result.get("detection_method", "provided_obb_boxes"),
            "fallback_reason": detection_result.get("fallback_reason"),
            "predictor_type": predictor_type,
            "debug": {
                "cold_start": bool(cold_start),
                "clamp_debug": clamp_debug,
            },
        }


def main():
    worker = LandmarkPredictWorker()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request_id = None
        try:
            msg = json.loads(line)
            request_id = str(msg.get("_request_id") or "")
            cmd = str(msg.get("cmd") or "").strip().lower()
            if cmd == "shutdown":
                _send({"status": "result", "_request_id": request_id, "ok": True})
                return
            if cmd != "predict":
                raise ValueError(f"Unsupported command: {cmd}")
            result = worker.predict(request_id, msg)
            _send({"status": "result", "_request_id": request_id, "ok": True, "data": result})
        except Exception as exc:
            _send({
                "status": "error",
                "_request_id": request_id,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })


if __name__ == "__main__":
    main()
