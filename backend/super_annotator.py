#!/usr/bin/env python3
"""
SuperAnnotator — Persistent Python controller for BioVision.

Runs as a long-lived process, communicating via line-delimited JSON over stdin/stdout.
Combines YOLO-World (detection), SAM2 (segmentation), and Dlib (landmark prediction)
into a unified pipeline with graceful degradation.

Commands (JSON per line on stdin):
  {"cmd": "init"}
  {"cmd": "check"}
  {"cmd": "annotate", "image_path": "...", "class_name": "Fish", ...}
  {"cmd": "refine_sam", "image_path": "...", "object_index": 0, "click_point": [x,y], "click_label": 1}
  {"cmd": "shutdown"}

Responses (JSON per line on stdout):
  {"status": "ready", "mode": "...", ...}
  {"status": "progress", "message": "...", "percent": N, "stage": "..."}
  {"status": "result", "objects": [...], ...}
  {"status": "error", "error": "..."}
"""

import sys
import io
import json
import os
import hashlib
import shutil
import traceback
from datetime import datetime

import numpy as np
import cv2

# ── Encoding fix (Windows) ─────────────────────────────────────────────────
# On Windows, Python defaults stdin/stdout to the console code page (CP1252).
# Non-ASCII Unicode in JSON paths (e.g. macOS narrow no-break space U+202F in
# screenshot filenames) gets mangled on decode when CP1252 is used.  Force
# UTF-8 here as a safety net in addition to PYTHONUTF8=1 set by the parent.
if hasattr(sys.stdin, "buffer"):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# Ensure all print/logging goes to stderr so stdout is reserved for JSON protocol
import logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[SuperAnnotator] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path for image_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_utils import load_image
import orientation_utils as ou
import debug_io as dio

STANDARD_SIZE = ou.STANDARD_SIZE
_CURRENT_REQUEST_ID = None


def send(obj):
    """Send a JSON object to stdout (one line)."""
    global _CURRENT_REQUEST_ID
    if _CURRENT_REQUEST_ID and "_request_id" not in obj:
        obj = {**obj, "_request_id": _CURRENT_REQUEST_ID}
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def send_progress(message, percent, stage="processing"):
    send({"status": "progress", "message": message, "percent": percent, "stage": stage})


class SuperAnnotator:
    def __init__(self):
        self.yolo_model = None
        self.sam2_model = None
        self.dlib_predictor = None
        self.dlib_id_mapping = None
        self.dlib_model_path = None
        self.dlib_target_orientation = None
        self.dlib_landmark_template = {}
        self.dlib_head_landmark_id = None
        self.dlib_tail_landmark_id = None
        self.mode = "classic_fallback"
        self.gpu = False
        self.device = "cpu"   # resolved by check_capabilities(); passed to predict()
        self.yolo_init_attempted = False
        self.yolo_init_error = None
        self.sam2_init_attempted = False
        self.sam2_init_error = None
        self._cached_image_path = None
        self._cached_image = None
        self._cached_sam_results = None
        # Mask cache: image_path → [(xyxy, mask_uint8), ...] for the last N annotated images.
        # Populated by annotate() so save_segment_for_box can reuse masks without re-running SAM2.
        self._mask_cache: dict = {}
        self._mask_cache_max = 10  # keep at most 10 images in memory
        # Fine-tuned model cache: avoid reloading .pt on every detect call
        self._finetuned_model = None
        self._finetuned_model_path = None

    @staticmethod
    def _format_yolo_error(err):
        """Normalize common YOLO dependency/init errors into actionable text."""
        msg = str(err)
        if "No module named 'clip'" in msg or 'No module named "clip"' in msg:
            return (
                "Missing Python dependency 'clip' required by YOLO-World text prompts. "
                "Install in this venv, then restart app: "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        return msg

    @staticmethod
    def _build_class_prompts(class_name):
        """Build a small prompt set to improve YOLO-World open-vocabulary recall."""
        base = (class_name or "").strip()
        if not base:
            return ["object"]

        prompts = [base]
        lower = base.lower()
        if not lower.startswith(("a ", "an ", "the ")):
            article = "an" if lower[:1] in ("a", "e", "i", "o", "u") else "a"
            prompts.append(f"{article} {base}")
        prompts.append(f"{base} specimen")
        prompts.append(f"{base} object")

        # Common biological shorthand for the current app domain.
        if lower == "fish":
            prompts.append("whole fish")
            prompts.append("fish body")

        # Preserve order, remove duplicates.
        unique = []
        seen = set()
        for p in prompts:
            key = p.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return unique

    @staticmethod
    def _safe_class_name(class_name):
        return (class_name or "object").strip().lower().replace(" ", "_")

    @staticmethod
    def _load_yolo_registry(registry_path):
        if not os.path.exists(registry_path):
            return {"active_model": None, "training_runs": []}
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"active_model": None, "training_runs": []}
            data.setdefault("active_model", None)
            data.setdefault("training_runs", [])
            return data
        except Exception:
            return {"active_model": None, "training_runs": []}

    @staticmethod
    def _save_yolo_registry(registry_path, data):
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _to_metric_value(value):
        """Best-effort conversion of metric-like objects to finite float."""
        try:
            if value is None:
                return None
            if hasattr(value, "item"):
                value = value.item()
            value = float(value)
            if not np.isfinite(value):
                return None
            return round(value, 5)
        except Exception:
            return None

    @classmethod
    def _extract_eval_metrics(cls, val_results):
        """
        Extract detection and pose metrics from Ultralytics val() output.

        Returns:
          {
            "box_map50": float|None,
            "box_map50_95": float|None,
            "pose_map50": float|None,
            "pose_map50_95": float|None,
          }
        """
        metrics = {
            "box_map50": None,
            "box_map50_95": None,
            "pose_map50": None,
            "pose_map50_95": None,
        }

        try:
            if hasattr(val_results, "box"):
                metrics["box_map50"] = cls._to_metric_value(getattr(val_results.box, "map50", None))
                metrics["box_map50_95"] = cls._to_metric_value(getattr(val_results.box, "map", None))
            if hasattr(val_results, "pose"):
                metrics["pose_map50"] = cls._to_metric_value(getattr(val_results.pose, "map50", None))
                metrics["pose_map50_95"] = cls._to_metric_value(getattr(val_results.pose, "map", None))
        except Exception:
            pass

        rd = getattr(val_results, "results_dict", None) or {}
        if isinstance(rd, dict):
            if metrics["box_map50"] is None:
                for key in ("metrics/mAP50(B)", "metrics/mAP50(box)", "metrics/mAP50"):
                    if key in rd:
                        metrics["box_map50"] = cls._to_metric_value(rd.get(key))
                        if metrics["box_map50"] is not None:
                            break
            if metrics["box_map50_95"] is None:
                for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95(box)", "metrics/mAP50-95"):
                    if key in rd:
                        metrics["box_map50_95"] = cls._to_metric_value(rd.get(key))
                        if metrics["box_map50_95"] is not None:
                            break
            if metrics["pose_map50"] is None:
                for key in ("metrics/mAP50(P)", "metrics/mAP50(pose)", "metrics/kpt_mAP50"):
                    if key in rd:
                        metrics["pose_map50"] = cls._to_metric_value(rd.get(key))
                        if metrics["pose_map50"] is not None:
                            break
            if metrics["pose_map50_95"] is None:
                for key in ("metrics/mAP50-95(P)", "metrics/mAP50-95(pose)", "metrics/kpt_mAP50-95"):
                    if key in rd:
                        metrics["pose_map50_95"] = cls._to_metric_value(rd.get(key))
                        if metrics["pose_map50_95"] is not None:
                            break

        return metrics

    @staticmethod
    def _resolve_primary_eval_metric(eval_metrics, use_pose=False):
        """
        Choose the metric family used for promotion decisions.
        Pose mAP is preferred when pose training is active and available.
        """
        if use_pose and eval_metrics.get("pose_map50") is not None:
            return "pose", eval_metrics.get("pose_map50"), eval_metrics.get("pose_map50_95")
        return "box", eval_metrics.get("box_map50"), eval_metrics.get("box_map50_95")

    def _evaluate_detector(self, model_path, dataset_yaml):
        from ultralytics import YOLO
        eval_model = YOLO(model_path)
        val_results = eval_model.val(
            data=dataset_yaml,
            split="val",
            imgsz=640,
            batch=4,
            verbose=False,
        )
        return self._extract_eval_metrics(val_results)

    @staticmethod
    def _resolve_dataset_yaml_paths(dataset_yaml):
        """
        Parse dataset.yaml for path/train/val and resolve absolute dirs.
        """
        config = {}
        try:
            with open(dataset_yaml, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    config[k.strip()] = v.strip()
        except Exception:
            return None

        yaml_dir = os.path.dirname(os.path.abspath(dataset_yaml))
        root = config.get("path", yaml_dir)
        if not os.path.isabs(root):
            root = os.path.normpath(os.path.join(yaml_dir, root))
        val_rel = config.get("val", "images/val")
        val_img_dir = val_rel if os.path.isabs(val_rel) else os.path.normpath(os.path.join(root, val_rel))

        if os.path.basename(val_img_dir) == "val" and os.path.basename(os.path.dirname(val_img_dir)) == "images":
            val_lbl_dir = os.path.join(os.path.dirname(os.path.dirname(val_img_dir)), "labels", "val")
        else:
            val_lbl_dir = os.path.join(root, "labels", "val")

        return {
            "root": root,
            "val_img_dir": val_img_dir,
            "val_lbl_dir": val_lbl_dir,
        }

    @staticmethod
    def _yolo_line_to_xyxy(line, img_w, img_h):
        parts = line.split()
        if len(parts) < 5:
            return None
        try:
            class_id = int(float(parts[0]))
            cx = float(parts[1]) * float(img_w)
            cy = float(parts[2]) * float(img_h)
            bw = float(parts[3]) * float(img_w)
            bh = float(parts[4]) * float(img_h)
        except Exception:
            return None
        x1 = max(0.0, cx - bw / 2.0)
        y1 = max(0.0, cy - bh / 2.0)
        x2 = min(float(img_w - 1), cx + bw / 2.0)
        y2 = min(float(img_h - 1), cy + bh / 2.0)
        if x2 <= x1 or y2 <= y1:
            return None
        return class_id, (x1, y1, x2, y2)

    @staticmethod
    def _xyxy_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0:
            return 0.0
        return float(inter / denom)

    def _evaluate_orientation_on_val(self, model_path, dataset_yaml, iou_threshold=0.5, conf_threshold=0.25):
        """
        Evaluate left/right class stability on val labels for orientation-aware YOLO detectors.
        """
        summary = {
            "enabled": False,
            "gt_instances": 0,
            "gt_left_instances": 0,
            "gt_right_instances": 0,
            "matched_instances": 0,
            "matched_right_instances": 0,
            "correct_orientation": 0,
            "correct_right_orientation": 0,
            "orientation_accuracy_all": None,
            "orientation_accuracy_matched": None,
            "right_recall": None,
            "right_coverage_ok": None,
            "error": None,
        }
        try:
            from ultralytics import YOLO
        except Exception as exc:
            summary["error"] = f"ultralytics_unavailable:{exc}"
            return summary

        paths = self._resolve_dataset_yaml_paths(dataset_yaml)
        if not paths:
            summary["error"] = "dataset_yaml_parse_failed"
            return summary
        val_img_dir = paths["val_img_dir"]
        val_lbl_dir = paths["val_lbl_dir"]
        if not os.path.isdir(val_img_dir) or not os.path.isdir(val_lbl_dir):
            summary["error"] = "val_dirs_missing"
            return summary

        model = YOLO(model_path)
        label_files = [f for f in os.listdir(val_lbl_dir) if f.endswith(".txt")]
        if not label_files:
            summary["error"] = "no_val_labels"
            return summary

        for label_name in sorted(label_files):
            label_path = os.path.join(val_lbl_dir, label_name)
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
            except Exception:
                continue
            if not lines:
                continue

            image_base = os.path.splitext(label_name)[0]
            image_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
                candidate = os.path.join(val_img_dir, image_base + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if image_path is None:
                continue

            img = cv2.imread(image_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            gt_boxes = []
            for ln in lines:
                parsed = self._yolo_line_to_xyxy(ln, img_w, img_h)
                if not parsed:
                    continue
                cls_id, xyxy = parsed
                if cls_id not in (0, 1):
                    continue
                gt_boxes.append((cls_id, xyxy))
            if not gt_boxes:
                continue

            results = model.predict(
                image_path,
                conf=conf_threshold,
                iou=0.6,
                imgsz=640,
                verbose=False,
            )
            preds = []
            if results:
                r0 = results[0]
                boxes = getattr(r0, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy_arr = boxes.xyxy.cpu().numpy()
                    cls_arr = boxes.cls.cpu().numpy() if boxes.cls is not None else None
                    for i in range(len(xyxy_arr)):
                        if cls_arr is None:
                            pred_cls = 0
                        else:
                            try:
                                pred_cls = int(cls_arr[i])
                            except Exception:
                                pred_cls = 0
                        preds.append((pred_cls, tuple(float(v) for v in xyxy_arr[i].tolist())))

            for gt_cls, gt_xyxy in gt_boxes:
                summary["gt_instances"] += 1
                if gt_cls == 0:
                    summary["gt_left_instances"] += 1
                elif gt_cls == 1:
                    summary["gt_right_instances"] += 1

                best_iou = 0.0
                best_cls = None
                for pred_cls, pred_xyxy in preds:
                    iou = self._xyxy_iou(gt_xyxy, pred_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_cls = pred_cls
                if best_iou >= float(iou_threshold):
                    summary["matched_instances"] += 1
                    if gt_cls == 1:
                        summary["matched_right_instances"] += 1
                    if best_cls == gt_cls:
                        summary["correct_orientation"] += 1
                        if gt_cls == 1:
                            summary["correct_right_orientation"] += 1

        if summary["gt_instances"] <= 0:
            summary["error"] = "no_orientation_gt_instances"
            return summary

        summary["enabled"] = True
        summary["orientation_accuracy_all"] = round(
            float(summary["correct_orientation"]) / float(summary["gt_instances"]), 5
        )
        if summary["matched_instances"] > 0:
            summary["orientation_accuracy_matched"] = round(
                float(summary["correct_orientation"]) / float(summary["matched_instances"]), 5
            )
        if summary["gt_right_instances"] > 0:
            rr = float(summary["matched_right_instances"]) / float(summary["gt_right_instances"])
            summary["right_recall"] = round(rr, 5)
            summary["right_coverage_ok"] = bool(rr >= 0.50)
        else:
            summary["right_recall"] = None
            summary["right_coverage_ok"] = True

        return summary

    @staticmethod
    def _resolve_checkpoint_task(model_path):
        """Return ultralytics task string for a checkpoint path (detect|pose|...)."""
        if not model_path or not os.path.exists(model_path):
            return None
        try:
            from ultralytics import YOLO

            task = getattr(YOLO(model_path), "task", None)
            if task is None:
                return None
            task = str(task).strip().lower()
            return task or None
        except Exception:
            return None

    @staticmethod
    def _resolve_detection_preset(conf_threshold, nms_iou, max_objects, detection_preset):
        preset = (detection_preset or "balanced").strip().lower()
        conf = float(conf_threshold)
        iou = float(nms_iou)
        top_k = int(max_objects)
        imgsz = 1280
        allow_relaxed_retry = True

        if preset == "precision":
            conf = max(conf, 0.45)
            iou = min(iou, 0.55)
            top_k = min(top_k, 8)
            allow_relaxed_retry = False
        elif preset == "recall":
            conf = min(conf, 0.2)
            iou = max(iou, 0.72)
            top_k = max(top_k, 30)
            imgsz = 1536
            allow_relaxed_retry = True
        elif preset == "single_object":
            conf = max(conf, 0.35)
            iou = min(iou, 0.5)
            top_k = 1
            allow_relaxed_retry = False
        else:
            preset = "balanced"
            conf = max(0.15, min(conf, 0.9))
            iou = max(0.55, min(iou, 0.75))
            top_k = max(1, min(top_k, 25))

        return {
            "preset": preset,
            "conf": conf,
            "iou": iou,
            "top_k": max(1, top_k),
            "imgsz": imgsz,
            "allow_relaxed_retry": allow_relaxed_retry,
        }

    @staticmethod
    def _resolve_yolo_train_params(dataset_size, detection_preset, use_pose=False, epochs_override=None):
        """
        Resolve YOLO fine-tune hyperparameters from effective dataset size.

        dataset_size: positive integer count supplied by user or derived from export stats.
        detection_preset: balanced|precision|recall|single_object
        use_pose: whether training task is pose or detect
        epochs_override: explicit user override for epochs (optional)
        """
        size = max(1, int(dataset_size))
        preset = (detection_preset or "balanced").strip().lower()
        if preset not in ("balanced", "precision", "recall", "single_object"):
            preset = "balanced"

        # Size buckets keep tiny datasets conservative and larger sets faster.
        if size <= 30:
            params = {
                "size_bucket": "tiny",
                "epochs": 60,
                "batch": 2,
                "freeze": 10,
                "lr0": 0.0015,
                "mosaic": 0.30,
                "close_mosaic": 3,
                "degrees": 8.0,
                "translate": 0.05,
                "scale": 0.15,
                "fliplr": 0.5,
            }
        elif size <= 120:
            params = {
                "size_bucket": "small",
                "epochs": 45,
                "batch": 4,
                "freeze": 6,
                "lr0": 0.0025,
                "mosaic": 0.50,
                "close_mosaic": 5,
                "degrees": 10.0,
                "translate": 0.08,
                "scale": 0.20,
                "fliplr": 0.5,
            }
        elif size <= 500:
            params = {
                "size_bucket": "medium",
                "epochs": 32,
                "batch": 6,
                "freeze": 3,
                "lr0": 0.0035,
                "mosaic": 0.70,
                "close_mosaic": 8,
                "degrees": 12.0,
                "translate": 0.10,
                "scale": 0.30,
                "fliplr": 0.5,
            }
        else:
            params = {
                "size_bucket": "large",
                "epochs": 22,
                "batch": 8,
                "freeze": 0,
                "lr0": 0.0050,
                "mosaic": 0.80,
                "close_mosaic": 10,
                "degrees": 12.0,
                "translate": 0.10,
                "scale": 0.35,
                "fliplr": 0.5,
            }

        # Preset-specific nudges.
        if preset == "precision":
            params["mosaic"] = round(params["mosaic"] * 0.6, 3)
            params["degrees"] = max(2.0, params["degrees"] - 3.0)
            params["translate"] = round(params["translate"] * 0.7, 3)
            params["scale"] = round(params["scale"] * 0.75, 3)
            params["lr0"] = round(params["lr0"] * 0.85, 6)
        elif preset == "recall":
            params["mosaic"] = min(0.9, round(params["mosaic"] + 0.1, 3))
            params["degrees"] = min(16.0, params["degrees"] + 2.0)
            params["translate"] = min(0.15, round(params["translate"] + 0.02, 3))
            params["scale"] = min(0.45, round(params["scale"] + 0.05, 3))
        elif preset == "single_object":
            params["mosaic"] = 0.0
            params["close_mosaic"] = 0
            params["translate"] = min(params["translate"], 0.05)
            params["scale"] = min(params["scale"], 0.20)

        # Pose training typically benefits from slightly more adaptation.
        if use_pose:
            params["freeze"] = max(0, int(params["freeze"]) - 2)
            params["epochs"] = int(params["epochs"]) + 5

        if epochs_override is not None:
            try:
                params["epochs"] = max(1, int(round(float(epochs_override))))
            except Exception:
                pass

        params["patience"] = max(8, min(24, int(round(params["epochs"] * 0.30))))
        params["augment"] = True
        params["imgsz"] = 640
        params["dataset_size"] = size
        params["detection_preset"] = preset
        return params

    # ------------------------------------------------------------------
    # Capability check
    # ------------------------------------------------------------------
    def check_capabilities(self):
        """Detect hardware capabilities and determine best mode."""
        gpu = False
        try:
            import torch
            gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            pass

        free_ram_gb = 0
        try:
            import psutil
            free_ram_gb = round(psutil.virtual_memory().available / (1024 ** 3), 1)
        except ImportError:
            free_ram_gb = 4.0  # assume reasonable if psutil missing

        if gpu and free_ram_gb > 4:
            mode = "auto_high_performance"
        elif free_ram_gb > 1.5:
            mode = "auto_lite"
        else:
            mode = "classic_fallback"

        self.gpu = gpu
        # Resolve a device specifier for Ultralytics predict() calls.
        # Using integer 0 for CUDA (Ultralytics convention) avoids "cuda:0" string ambiguity.
        try:
            import torch
            if torch.cuda.is_available():
                self.device = 0          # cuda:0
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

        return {
            "mode": mode,
            "gpu": gpu,
            "free_ram_gb": free_ram_gb,
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def init_models(self):
        """Load models based on detected capabilities."""
        caps = self.check_capabilities()
        self.mode = caps["mode"]

        yolo_loaded = False
        sam2_loaded = False

        # Try loading YOLO-World
        if self.mode in ("auto_high_performance", "auto_lite"):
            self.yolo_init_attempted = True
            self.yolo_init_error = None
            try:
                send_progress("Loading YOLO-World model...", 10, "init")
                from ultralytics import YOLOWorld
                self.yolo_model = YOLOWorld("yolov8s-worldv2.pt")
                # Smoke-test: encode a class name so a missing CLIP dependency is caught
                # at init rather than on the first detection call.
                self.yolo_model.set_classes(["object"])
                # Move backbone to target device.
                if self.device != "cpu":
                    self.yolo_model.to(self.device)
                # Patch set_classes() to auto-migrate txt_feats on every call
                # (including internal Ultralytics warmup calls during predict()).
                self._patch_set_classes()
                # Also migrate the txt_feats from the smoke-test above.
                self._move_txt_feats_to_device()
                yolo_loaded = True
                logger.info("YOLO-World loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load YOLO-World: {e}")
                self.yolo_model = None
                self.yolo_init_error = self._format_yolo_error(e)

        # Try loading SAM2
        if self.mode == "auto_high_performance":
            self.sam2_init_attempted = True
            self.sam2_init_error = None
            try:
                send_progress("Loading SAM2 model...", 40, "init")
                from ultralytics import SAM
                self.sam2_model = SAM("sam2_b.pt")
                sam2_loaded = True
                logger.info("SAM2 loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load SAM2: {e}")
                self.sam2_model = None
                self.sam2_init_error = str(e)

        # Update mode based on what actually loaded
        if yolo_loaded and sam2_loaded:
            self.mode = "auto_high_performance"
        elif yolo_loaded:
            self.mode = "auto_lite"
        else:
            self.mode = "classic_fallback"

        send_progress("Ready", 100, "init")

        return {
            "status": "ready",
            "mode": self.mode,
            "gpu": self.gpu,
            "yolo_loaded": yolo_loaded,
            "sam2_loaded": sam2_loaded,
        }

    # ------------------------------------------------------------------
    # Check (lightweight, no model loading)
    # ------------------------------------------------------------------
    def check(self):
        """Report current state without loading models."""
        caps = self.check_capabilities()
        return {
            "status": "ready",
            "available": True,
            "mode": caps["mode"],
            "gpu": caps["gpu"],
            "yolo_ready": self.yolo_model is not None,
            "sam2_ready": self.sam2_model is not None,
            "yolo_failed": self.yolo_init_attempted and self.yolo_model is None and self.yolo_init_error is not None,
            "sam2_failed": self.sam2_init_attempted and self.sam2_model is None and self.sam2_init_error is not None,
            "yolo_error": self.yolo_init_error,
            "sam2_error": self.sam2_init_error,
        }

    # ------------------------------------------------------------------
    # Load / cache image
    # ------------------------------------------------------------------
    def _load_image(self, image_path):
        """Load image with EXIF correction, caching for repeated SAM calls."""
        if self._cached_image_path == image_path and self._cached_image is not None:
            return self._cached_image
        img, w, h = load_image(image_path)
        self._cached_image_path = image_path
        self._cached_image = img
        self._cached_sam_results = None  # invalidate SAM cache
        return img

    def _move_txt_feats_to_device(self):
        """
        Move YOLO-World's CLIP text feature tensors to self.device.

        set_classes() calls the CLIP text encoder which always outputs tensors on CPU,
        regardless of where the YOLO backbone lives.  Ultralytics stores these as plain
        tensor *attributes* (not registered buffers), so model.to(device) silently skips
        them.  We must move them manually after every set_classes() call.
        """
        if self.device == "cpu":
            return
        try:
            import torch
            for m in self.yolo_model.model.modules():
                if hasattr(m, "txt_feats") and isinstance(m.txt_feats, torch.Tensor):
                    m.txt_feats = m.txt_feats.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move txt_feats to {self.device}: {e}")

    def _patch_set_classes(self):
        """
        Monkey-patch yolo_model.set_classes() to automatically move txt_feats to
        self.device after every call — including calls made internally by Ultralytics'
        predictor warmup / streaming pipeline.

        Without this, set_classes() always leaves txt_feats on CPU (CLIP outputs CPU
        tensors regardless of target device), causing an index_select device-mismatch
        crash on CUDA systems whenever Ultralytics calls it internally during predict().
        """
        if self.device == "cpu":
            return
        import types
        original = self.yolo_model.set_classes
        device = self.device
        move_fn = self._move_txt_feats_to_device

        def _patched(classes, *args, **kwargs):
            original(classes, *args, **kwargs)
            move_fn()

        self.yolo_model.set_classes = types.MethodType(
            lambda _self, classes, *a, **kw: _patched(classes, *a, **kw),
            self.yolo_model,
        )

    # ------------------------------------------------------------------
    # Stage A: Detection
    # ------------------------------------------------------------------
    def detect_yolo(self, image, class_name, conf_threshold=0.5, nms_iou=0.6, top_k=10, imgsz=1280):
        """YOLO-World open-vocabulary detection with NMS and top-k filtering."""
        prompts = self._build_class_prompts(class_name)
        self.yolo_model.set_classes(prompts)
        # set_classes() regenerates CLIP text features on CPU every call.
        # Explicitly move txt_feats tensors to the target device before predict().
        self._move_txt_feats_to_device()
        results = self.yolo_model.predict(
            image,
            conf=conf_threshold,
            imgsz=imgsz,
            iou=min(0.85, max(float(nms_iou), 0.45)),
            max_det=max(50, int(top_k) * 6),
            device=self.device,
            verbose=False,
        )
        boxes = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0]) if hasattr(box, "cls") else 0
            cls_name = prompts[cls_idx] if 0 <= cls_idx < len(prompts) else class_name
            boxes.append({
                "xyxy": xyxy,
                "confidence": round(conf, 3),
                "class_name": cls_name,
            })

        # Apply NMS to remove overlapping duplicates
        if len(boxes) > 1:
            boxes = self._nms(boxes, iou_threshold=nms_iou)

        # Prefer whole-object boxes over tiny contained part boxes (e.g. fins).
        boxes = self._suppress_part_boxes(boxes, image.shape, class_name)

        # Sort by confidence descending and keep top-k
        boxes.sort(key=lambda b: b["confidence"], reverse=True)
        if top_k > 0 and len(boxes) > top_k:
            boxes = boxes[:top_k]

        return boxes

    @staticmethod
    def _iou(box_a, box_b):
        """Compute IoU between two boxes in xyxy format."""
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    @classmethod
    def _nms(cls, boxes, iou_threshold=0.6):
        """Non-Maximum Suppression: remove overlapping boxes, keep highest confidence."""
        sorted_boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
        keep = []
        while sorted_boxes:
            best = sorted_boxes.pop(0)
            keep.append(best)
            sorted_boxes = [
                b for b in sorted_boxes
                if cls._iou(best["xyxy"], b["xyxy"]) < iou_threshold
            ]
        return keep

    @staticmethod
    def _box_area(xyxy):
        x1, y1, x2, y2 = xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @classmethod
    def _intersection_over_small(cls, big_xyxy, small_xyxy):
        xa1, ya1, xa2, ya2 = big_xyxy
        xb1, yb1, xb2, yb2 = small_xyxy
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        small_area = cls._box_area(small_xyxy)
        return inter / small_area if small_area > 0 else 0.0

    @classmethod
    def _suppress_part_boxes(cls, boxes, image_shape, class_name):
        """Drop tiny boxes mostly contained in larger boxes to reduce part detections."""
        if not boxes:
            return boxes

        h, w = image_shape[:2]
        image_area = max(1.0, float(h * w))
        is_fish_query = "fish" in (class_name or "").lower()

        enriched = []
        for b in boxes:
            area = cls._box_area(b["xyxy"])
            if area <= 0:
                continue
            item = dict(b)
            item["_area"] = area
            enriched.append(item)

        if not enriched:
            return []

        max_area = max(b["_area"] for b in enriched)
        abs_floor = image_area * (0.0015 if is_fish_query else 0.0010)
        rel_floor = max_area * (0.05 if is_fish_query else 0.03)
        size_floor = max(abs_floor, rel_floor)

        filtered = [b for b in enriched if b["_area"] >= size_floor]
        if not filtered:
            filtered = enriched

        # Sort larger-first so enclosed small parts are removed.
        filtered.sort(key=lambda b: (b["_area"], b["confidence"]), reverse=True)
        kept = []
        for candidate in filtered:
            is_part = False
            for accepted in kept:
                ios = cls._intersection_over_small(accepted["xyxy"], candidate["xyxy"])
                if ios > 0.85 and candidate["_area"] < accepted["_area"] * 0.55:
                    is_part = True
                    break
            if not is_part:
                kept.append(candidate)

        for b in kept:
            b.pop("_area", None)
        return kept

    def detect_classic(self, image, min_area_ratio=0.02):
        """OpenCV Otsu + contour detection fallback."""
        from detect_specimen import detect_multiple_specimens
        h, w = image.shape[:2]

        # Save to temp file for detect_multiple_specimens (expects path)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cv2.imwrite(tmp_path, image)
            result = detect_multiple_specimens(tmp_path, min_area_ratio=min_area_ratio)
        finally:
            os.unlink(tmp_path)

        boxes = []
        if result.get("ok") and result.get("boxes"):
            for b in result["boxes"]:
                boxes.append({
                    "xyxy": [b["left"], b["top"], b["right"], b["bottom"]],
                    "confidence": b.get("confidence", 0.5),
                    "class_name": "object",
                })
        return boxes

    # ------------------------------------------------------------------
    # Stage A.5: SAM2 refinement
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # SAM2 mask quality scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _score_sam_mask(mask_np, box_xyxy, image_shape, min_score=0.35):
        """
        Score a SAM2 binary mask for full-object quality.

        Criteria (weighted sum, higher = better):
          - fill_ratio (0.4):  fraction of box area covered by mask
          - center_ok  (0.3):  mask centroid is inside the detection box
          - cc_ratio   (0.2):  largest connected component / total mask pixels
          - truncation (0.1, penalty): fraction of mask pixels on image border

        Returns (score, metrics_dict).  score < min_score should be rejected.
        """
        mask_bin = (mask_np > 0).astype(np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        img_h, img_w = image_shape[:2]
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)

        mask_pixels = int(mask_bin.sum())
        if mask_pixels == 0:
            return 0.0, {"fill_ratio": 0, "center_ok": 0, "cc_ratio": 0, "edge_truncation": 0}

        # 1. Fill ratio: fraction of box area that is masked
        x1c, y1c = max(x1, 0), max(y1, 0)
        x2c, y2c = min(x2, img_w), min(y2, img_h)
        mask_in_box = mask_bin[y1c:y2c, x1c:x2c]
        fill_ratio = float(mask_in_box.sum()) / (box_w * box_h)

        # 2. Center-in-box
        ys, xs = np.where(mask_bin > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        center_ok = float(x1 <= cx <= x2 and y1 <= cy <= y2)

        # 3. Largest connected component fraction
        n_labels, labels_im = cv2.connectedComponents(mask_bin)
        if n_labels > 1:
            largest_cc = max(int((labels_im == i).sum()) for i in range(1, n_labels))
        else:
            largest_cc = 0
        cc_ratio = largest_cc / max(mask_pixels, 1)

        # 4. Edge truncation penalty
        border_pixels = (
            int(mask_bin[0, :].sum()) + int(mask_bin[-1, :].sum()) +
            int(mask_bin[:, 0].sum()) + int(mask_bin[:, -1].sum())
        )
        truncation = min(border_pixels / max(mask_pixels, 1), 1.0)

        score = (
            fill_ratio * 0.4 +
            center_ok  * 0.3 +
            cc_ratio   * 0.2 -
            truncation * 0.1
        )
        metrics = {
            "fill_ratio": round(fill_ratio, 4),
            "center_ok": bool(center_ok),
            "cc_ratio": round(cc_ratio, 4),
            "edge_truncation": round(truncation, 4),
        }
        return float(score), metrics

    @staticmethod
    def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
        """Pixel-level IoU between two binary masks of the same shape."""
        inter = np.logical_and(m1, m2).sum()
        if inter == 0:
            return 0.0
        return float(inter) / max(float(np.logical_or(m1, m2).sum()), 1)

    def refine_with_sam2(self, image, boxes, min_mask_score=0.35):
        """
        Refine YOLO boxes with SAM2 masks.

        For each box, SAM2 may return multiple mask candidates.  We score all
        candidates with _score_sam_mask and keep the highest-scoring one.
        Masks that score below min_mask_score are rejected (None returned).
        """
        masks = []
        for i, box_data in enumerate(boxes):
            try:
                xyxy = box_data["xyxy"]
                results = self.sam2_model.predict(image, bboxes=[xyxy], device=self.device, verbose=False)
                # results[0].masks.data may contain multiple candidates
                candidates = results[0].masks.data.cpu().numpy().astype(np.uint8)  # shape [N, H, W]
                best_mask = None
                best_score = -1.0
                for c_idx in range(candidates.shape[0]):
                    cand = candidates[c_idx]
                    score, _ = self._score_sam_mask(cand, xyxy, image.shape)
                    if score > best_score:
                        best_score = score
                        best_mask = cand
                if best_mask is not None and best_score >= min_mask_score:
                    masks.append(best_mask)
                else:
                    logger.debug(
                        f"SAM2 mask for object {i} rejected (best score {best_score:.3f} < {min_mask_score})"
                    )
                    masks.append(None)
            except RuntimeError as e:
                # OOM or other GPU error — degrade gracefully
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    logger.warning(f"SAM2 OOM on object {i}, skipping mask refinement")
                    masks.append(None)
                else:
                    logger.warning(f"SAM2 error on object {i}: {e}")
                    masks.append(None)
            except Exception as e:
                logger.warning(f"SAM2 error on object {i}: {e}")
                masks.append(None)
        return masks

    def mask_to_outline(self, mask, max_points=100):
        """Convert binary mask to simplified polygon outline."""
        if mask is None:
            return []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        biggest = max(contours, key=cv2.contourArea)
        # Simplify polygon
        epsilon = cv2.arcLength(biggest, True) * 0.005
        approx = cv2.approxPolyDP(biggest, epsilon, True)
        # Limit points
        if len(approx) > max_points:
            step = max(1, len(approx) // max_points)
            approx = approx[::step]
        return [[int(p[0][0]), int(p[0][1])] for p in approx]

    def tight_box_from_mask(self, mask):
        """Get tight bounding box from binary mask."""
        if mask is None:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        return [x, y, x + w, y + h]

    # ------------------------------------------------------------------
    # Stage B: Normalization (The "Standardizer")
    # ------------------------------------------------------------------
    def standardize_instance(
        self,
        image,
        xyxy,
        mask=None,
        pca_mode="auto",
        orientation_policy=None,
        orientation_hint=None,
    ):
        """
        Base normalization shared with training/inference:
          crop + padding -> resize to STANDARD_SIZE.

        Optional PCA rotation is an enhancement on top of the shared base and
        only applies when explicitly enabled.
        """
        standardized, metadata = ou.base_standardize(image, xyxy, pad_ratio=0.20)

        if pca_mode in ("on", "auto") and mask is not None:
            try:
                cx1, cy1 = [int(v) for v in metadata["crop_origin"]]
                cw, ch = [int(v) for v in metadata["crop_size"]]
                if cw > 0 and ch > 0:
                    mask_crop = mask[cy1:cy1 + ch, cx1:cx1 + cw]
                    if mask_crop.size > 0:
                        mask_512 = cv2.resize(
                            mask_crop.astype(np.uint8),
                            (STANDARD_SIZE, STANDARD_SIZE),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        standardized, _mask_512, canonical_meta = ou.canonicalize_with_mask(
                            standardized,
                            mask_512,
                            policy=orientation_policy,
                            pca_mode=pca_mode,
                            orientation_hint=orientation_hint,
                        )
                        metadata["rotation"] = float(canonical_meta.get("pca_rotation", 0.0))
                        metadata["canonical_flip_applied"] = bool(
                            canonical_meta.get("canonical_flip_applied", False)
                        )
                        metadata["canonicalization"] = canonical_meta
            except Exception as e:
                logger.debug(f"PCA mode skipped due to mask processing error: {e}")

        return standardized, metadata

    # ------------------------------------------------------------------
    # Stage C: Dlib landmark prediction
    # ------------------------------------------------------------------
    def load_dlib_model(self, model_path, id_mapping_path=None):
        """Load dlib shape predictor and ID mapping."""
        if self.dlib_model_path == model_path and self.dlib_predictor is not None:
            return  # already loaded

        import dlib
        self.dlib_predictor = dlib.shape_predictor(model_path)
        self.dlib_model_path = model_path

        self.dlib_id_mapping = None
        self.dlib_target_orientation = None
        self.dlib_landmark_template = {}
        self.dlib_head_landmark_id = None
        self.dlib_tail_landmark_id = None
        if id_mapping_path and os.path.exists(id_mapping_path):
            try:
                with open(id_mapping_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                explicit = raw.get("dlib_index_to_original")
                if explicit:
                    self.dlib_id_mapping = {int(k): int(v) for k, v in explicit.items()}
                else:
                    # Backward-compatible field: {"dlib_to_original": {"0": 1, ...}}
                    mapping_dict = raw.get("dlib_to_original", raw)
                    parsed = {}
                    for k, v in (mapping_dict or {}).items():
                        try:
                            parsed[int(k)] = int(v)
                        except Exception:
                            continue
                    self.dlib_id_mapping = parsed or None

                self.dlib_target_orientation = raw.get("training_config", {}).get("target_orientation")
                self.dlib_head_landmark_id = raw.get("training_config", {}).get("head_landmark_id")
                if self.dlib_head_landmark_id is not None:
                    self.dlib_head_landmark_id = int(self.dlib_head_landmark_id)
                self.dlib_tail_landmark_id = raw.get("training_config", {}).get("tail_landmark_id")
                if self.dlib_tail_landmark_id is not None:
                    self.dlib_tail_landmark_id = int(self.dlib_tail_landmark_id)

                self.dlib_landmark_template = {
                    int(k): v for k, v in (raw.get("landmark_template", {}) or {}).items()
                }
            except Exception:
                pass

        # Resolve head/tail IDs from session orientation policy when training
        # metadata does not provide them.
        if self.dlib_head_landmark_id is None or self.dlib_tail_landmark_id is None:
            try:
                session_dir = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))
                head_id, tail_id = ou.resolve_head_tail_landmark_ids(session_dir)
                if self.dlib_head_landmark_id is None and head_id is not None:
                    self.dlib_head_landmark_id = int(head_id)
                if self.dlib_tail_landmark_id is None and tail_id is not None:
                    self.dlib_tail_landmark_id = int(tail_id)
            except Exception:
                pass

    def predict_landmarks(self, standardized_image):
        """Run dlib on a STANDARD_SIZE × STANDARD_SIZE image."""
        import dlib
        rect = dlib.rectangle(0, 0, STANDARD_SIZE, STANDARD_SIZE)

        # Convert to grayscale for dlib
        if len(standardized_image.shape) == 3:
            gray = cv2.cvtColor(standardized_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = standardized_image

        shape = self.dlib_predictor(gray, rect)
        landmarks = []
        for i in range(shape.num_parts):
            original_id = i
            if self.dlib_id_mapping and i in self.dlib_id_mapping:
                original_id = self.dlib_id_mapping[i]
            landmarks.append({
                "id": original_id,
                "x": shape.part(i).x,
                "y": shape.part(i).y,
            })
        return landmarks

    # ------------------------------------------------------------------
    # Stage D: Global mapping (inverse transform)
    # ------------------------------------------------------------------
    def map_to_original(self, landmarks_512, metadata, was_flipped=False, image_shape=None):
        """Map STANDARD_SIZE landmarks back to original image coordinates."""
        return ou.map_to_original(
            landmarks_512,
            metadata,
            was_flipped=was_flipped,
            image_shape=image_shape,
        )

    # ------------------------------------------------------------------
    # YOLOv8 fine-tuning
    # ------------------------------------------------------------------
    def preview_yolo_train_plan(
        self,
        session_dir,
        class_name,
        epochs=None,
        detection_preset="balanced",
        dataset_size=None,
        auto_tune=True,
    ):
        """Compute export stats + resolved YOLO train params without running training."""
        from export_yolo_dataset import export_dataset

        export_details = export_dataset(session_dir, class_name, return_details=True)
        # YOLO-Pose training is disabled in the default pipeline.
        use_pose = False
        orientation_class_enabled = bool(export_details.get("orientation_class_enabled")) and not bool(use_pose)

        try:
            user_dataset_size = int(round(float(dataset_size))) if dataset_size is not None else None
            if user_dataset_size is not None and user_dataset_size <= 0:
                user_dataset_size = None
        except Exception:
            user_dataset_size = None

        exported_dataset_size = int(
            export_details.get("train_records")
            or export_details.get("total_records")
            or 1
        )
        dataset_size_effective = max(1, user_dataset_size or exported_dataset_size)
        dataset_size_source = "user" if user_dataset_size is not None else "export"

        if auto_tune:
            resolved_train_params = self._resolve_yolo_train_params(
                dataset_size=dataset_size_effective,
                detection_preset=detection_preset,
                use_pose=bool(use_pose),
                epochs_override=epochs,
            )
        else:
            resolved_epochs = 25
            if epochs is not None:
                try:
                    resolved_epochs = max(1, int(round(float(epochs))))
                except Exception:
                    pass
            resolved_train_params = {
                "size_bucket": "manual",
                "epochs": resolved_epochs,
                "batch": 4,
                "freeze": 10,
                "lr0": 0.01,
                "mosaic": 1.0,
                "close_mosaic": 10,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "fliplr": 0.5,
                "patience": max(8, min(24, int(round(resolved_epochs * 0.30)))),
                "augment": True,
                "imgsz": 640,
                "dataset_size": dataset_size_effective,
                "detection_preset": (detection_preset or "balanced"),
            }

        if orientation_class_enabled and float(resolved_train_params.get("fliplr", 0.0)) != 0.0:
            resolved_train_params = {**resolved_train_params}
            resolved_train_params["fliplr"] = 0.0
            resolved_train_params["orientation_flip_guard"] = "fliplr_disabled_for_orientation_classes"

        preflight_warnings = []
        if isinstance(export_details, dict):
            preflight_warnings = list(export_details.get("orientation_preflight_warnings") or [])

        return {
            "dataset": export_details,
            "use_pose": bool(use_pose),
            "detection_preset": detection_preset,
            "auto_tune": bool(auto_tune),
            "dataset_size_effective": dataset_size_effective,
            "dataset_size_source": dataset_size_source,
            "resolved_train_params": resolved_train_params,
            "preflight_warnings": preflight_warnings,
        }

    def train_yolo(self, session_dir, class_name, epochs=None, detection_preset="balanced",
                   dataset_size=None, auto_tune=True):
        """Fine-tune session YOLO with versioning + validation-based promotion."""
        send_progress("Exporting dataset...", 5, "training")
        from export_yolo_dataset import export_dataset
        export_details = export_dataset(session_dir, class_name, return_details=True)
        # YOLO-Pose training is disabled in the default pipeline.
        use_pose = False
        orientation_class_enabled = bool(export_details.get("orientation_class_enabled")) and not bool(use_pose)
        dataset_yaml = export_details["yaml_path"]

        models_dir = os.path.join(session_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        safe_name = self._safe_class_name(class_name)
        debug_model_type = "yolo_pose" if use_pose else "yolo_detect"
        run_dir, run_id = dio.create_model_run_dir(session_dir, debug_model_type, safe_name)
        dio.write_run_manifest(
            run_dir,
            model_type=debug_model_type,
            tag=safe_name,
            project_root=session_dir,
            extra={"status": "started", "class_name": class_name},
        )
        dio.write_run_json(run_dir, "dataset_export_stats.json", export_details)
        alias_path = os.path.join(
            models_dir,
            f"{'pose' if use_pose else 'yolo'}_{safe_name}.pt",
        )
        registry_path = os.path.join(models_dir, f"yolo_{safe_name}_registry.json")
        registry = self._load_yolo_registry(registry_path)
        runs = registry.get("training_runs", [])
        next_version = len(runs) + 1

        active_path = None
        active_entry = registry.get("active_model")
        if isinstance(active_entry, dict):
            active_use_pose = bool(active_entry.get("use_pose", False))
            if active_use_pose == bool(use_pose):
                active_path = active_entry.get("path")
        if not active_path and os.path.exists(alias_path):
            active_path = alias_path

        # Guard against cross-task warm starts (detect checkpoint used for pose or vice versa).
        active_task = self._resolve_checkpoint_task(active_path) if active_path else None
        expected_task = "pose" if use_pose else "detect"
        if active_path and active_task and active_task != expected_task:
            logger.warning(
                f"Skipping incompatible warm-start checkpoint {active_path} "
                f"(task={active_task}, expected={expected_task})."
            )
            active_path = None

        send_progress("Starting YOLOv8 training...", 10, "training")
        from ultralytics import YOLO
        default_base = "yolov8s-pose.pt" if use_pose else "yolov8s.pt"
        base_weights = active_path if active_path and os.path.exists(active_path) else default_base
        model = YOLO(base_weights)

        try:
            user_dataset_size = int(round(float(dataset_size))) if dataset_size is not None else None
            if user_dataset_size is not None and user_dataset_size <= 0:
                user_dataset_size = None
        except Exception:
            user_dataset_size = None

        exported_dataset_size = int(
            export_details.get("train_records")
            or export_details.get("total_records")
            or 1
        )
        dataset_size_effective = max(1, user_dataset_size or exported_dataset_size)
        dataset_size_source = "user" if user_dataset_size is not None else "export"

        if auto_tune:
            resolved_train_params = self._resolve_yolo_train_params(
                dataset_size=dataset_size_effective,
                detection_preset=detection_preset,
                use_pose=bool(use_pose),
                epochs_override=epochs,
            )
        else:
            resolved_epochs = 25
            if epochs is not None:
                try:
                    resolved_epochs = max(1, int(round(float(epochs))))
                except Exception:
                    pass
            resolved_train_params = {
                "size_bucket": "manual",
                "epochs": resolved_epochs,
                "batch": 4,
                "freeze": 10,
                "lr0": 0.01,
                "mosaic": 1.0,
                "close_mosaic": 10,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "fliplr": 0.5,
                "patience": max(8, min(24, int(round(resolved_epochs * 0.30)))),
                "augment": True,
                "imgsz": 640,
                "dataset_size": dataset_size_effective,
                "detection_preset": (detection_preset or "balanced"),
            }

        if orientation_class_enabled and float(resolved_train_params.get("fliplr", 0.0)) != 0.0:
            resolved_train_params = {**resolved_train_params}
            resolved_train_params["fliplr"] = 0.0
            resolved_train_params["orientation_flip_guard"] = "fliplr_disabled_for_orientation_classes"

        # Training with progress callbacks
        def on_train_epoch_end(trainer):
            epoch = trainer.epoch + 1
            total = trainer.epochs
            pct = 10 + int(80 * (epoch / total))
            send_progress(f"Training epoch {epoch}/{total}...", pct, "training")

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        train_args = {
            "data": dataset_yaml,
            "epochs": resolved_train_params["epochs"],
            "imgsz": resolved_train_params["imgsz"],
            "freeze": resolved_train_params["freeze"],
            "batch": resolved_train_params["batch"],
            "lr0": resolved_train_params["lr0"],
            "augment": resolved_train_params["augment"],
            "mosaic": resolved_train_params["mosaic"],
            "close_mosaic": resolved_train_params["close_mosaic"],
            "degrees": resolved_train_params["degrees"],
            "translate": resolved_train_params["translate"],
            "scale": resolved_train_params["scale"],
            "fliplr": resolved_train_params["fliplr"],
            "patience": resolved_train_params["patience"],
            "project": os.path.join(session_dir, "yolo_train"),
            "name": f"run_v{next_version}",
            "exist_ok": True,
            "verbose": False,
        }
        dio.write_run_json(run_dir, "train_params.json", resolved_train_params)
        dio.write_run_json(
            run_dir,
            "yolo_train_args.json",
            {
                **train_args,
                "base_weights": base_weights,
                "auto_tune": bool(auto_tune),
                "detection_preset": detection_preset,
                "dataset_size_effective": dataset_size_effective,
                "dataset_size_source": dataset_size_source,
                "use_pose": bool(use_pose),
            },
        )
        model.train(**train_args)

        # Resolve trained checkpoint
        best = os.path.join(session_dir, "yolo_train", f"run_v{next_version}", "weights", "best.pt")
        if not os.path.exists(best):
            best = os.path.join(session_dir, "yolo_train", f"run_v{next_version}", "weights", "last.pt")
        if not os.path.exists(best):
            raise FileNotFoundError("YOLO training finished but no best.pt/last.pt was found.")

        candidate_path = os.path.join(
            models_dir,
            f"{'pose' if use_pose else 'yolo'}_{safe_name}_v{next_version}.pt",
        )
        shutil.copy2(best, candidate_path)

        send_progress("Evaluating detector quality...", 92, "training")
        candidate_metrics = self._evaluate_detector(candidate_path, dataset_yaml)
        eval_metric_type, candidate_map50, candidate_map = self._resolve_primary_eval_metric(
            candidate_metrics,
            use_pose=bool(use_pose),
        )

        incumbent_metrics = {
            "box_map50": None,
            "box_map50_95": None,
            "pose_map50": None,
            "pose_map50_95": None,
        }
        candidate_orientation_metrics = None
        incumbent_orientation_metrics = None
        incumbent_map50 = None
        incumbent_map = None
        if active_path and os.path.exists(active_path):
            try:
                incumbent_metrics = self._evaluate_detector(active_path, dataset_yaml)
                if eval_metric_type == "pose":
                    incumbent_map50 = incumbent_metrics.get("pose_map50")
                    incumbent_map = incumbent_metrics.get("pose_map50_95")
                else:
                    incumbent_map50 = incumbent_metrics.get("box_map50")
                    incumbent_map = incumbent_metrics.get("box_map50_95")
            except Exception as e:
                logger.warning(f"Failed to evaluate incumbent detector, proceeding with promotion: {e}")

        if orientation_class_enabled:
            try:
                candidate_orientation_metrics = self._evaluate_orientation_on_val(candidate_path, dataset_yaml)
            except Exception as e:
                logger.warning(f"Failed candidate orientation validation: {e}")
            if active_path and os.path.exists(active_path):
                try:
                    incumbent_orientation_metrics = self._evaluate_orientation_on_val(active_path, dataset_yaml)
                except Exception as e:
                    logger.warning(f"Failed incumbent orientation validation: {e}")

        should_promote = (
            incumbent_map50 is None
            or candidate_map50 is None
            or candidate_map50 >= incumbent_map50 - 1e-4
        )
        promotion_guard_reason = None
        if should_promote and orientation_class_enabled and isinstance(candidate_orientation_metrics, dict):
            if candidate_orientation_metrics.get("enabled"):
                right_ok = candidate_orientation_metrics.get("right_coverage_ok")
                if right_ok is False:
                    should_promote = False
                    promotion_guard_reason = "orientation_right_coverage_below_threshold"
            cand_acc = candidate_orientation_metrics.get("orientation_accuracy_all")
            inc_acc = None
            if isinstance(incumbent_orientation_metrics, dict) and incumbent_orientation_metrics.get("enabled"):
                inc_acc = incumbent_orientation_metrics.get("orientation_accuracy_all")
            if (
                should_promote
                and cand_acc is not None
                and inc_acc is not None
                and float(cand_acc) + 1e-4 < float(inc_acc)
            ):
                should_promote = False
                promotion_guard_reason = "orientation_accuracy_regression"
        if should_promote:
            shutil.copy2(candidate_path, alias_path)

        run_entry = {
            "version": next_version,
            "class_name": class_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "base_weights": base_weights,
            "path": candidate_path,
            "promoted": should_promote,
            "use_pose": use_pose,
            "evaluation_metric_type": eval_metric_type,
            "candidate_map50": candidate_map50,
            "candidate_map50_95": candidate_map,
            "incumbent_map50": incumbent_map50,
            "incumbent_map50_95": incumbent_map,
            "candidate_metrics": candidate_metrics,
            "incumbent_metrics": incumbent_metrics,
            "candidate_orientation_metrics": candidate_orientation_metrics,
            "incumbent_orientation_metrics": incumbent_orientation_metrics,
            "promotion_guard_reason": promotion_guard_reason,
            "dataset": export_details,
            "detection_preset": detection_preset,
            "auto_tune": bool(auto_tune),
            "dataset_size_effective": dataset_size_effective,
            "dataset_size_source": dataset_size_source,
            "resolved_train_params": resolved_train_params,
        }
        registry["training_runs"] = [*runs, run_entry]
        if should_promote:
            registry["active_model"] = {
                "version": next_version,
                "path": candidate_path,
                "map50": candidate_map50,
                "map50_95": candidate_map,
                "metric_type": eval_metric_type,
                "metrics": candidate_metrics,
                "orientation_metrics": candidate_orientation_metrics,
                "use_pose": use_pose,
                "updated_at": run_entry["created_at"],
            }
        self._save_yolo_registry(registry_path, registry)
        dio.copy_json_if_exists(registry_path, run_dir, "registry_snapshot.json")

        send_progress("Training complete", 100, "training")
        logger.info(f"YOLOv8 candidate model saved to {candidate_path} (promoted={should_promote})")
        summary = {
            "active_model_path": alias_path if os.path.exists(alias_path) else candidate_path,
            "candidate_model_path": candidate_path,
            "registry_path": registry_path,
            "debug_run_dir": run_dir,
            "version": next_version,
            "promoted": should_promote,
            "evaluation_metric_type": eval_metric_type,
            "candidate_map50": candidate_map50,
            "candidate_map50_95": candidate_map,
            "incumbent_map50": incumbent_map50,
            "incumbent_map50_95": incumbent_map,
            "candidate_metrics": candidate_metrics,
            "incumbent_metrics": incumbent_metrics,
            "candidate_orientation_metrics": candidate_orientation_metrics,
            "incumbent_orientation_metrics": incumbent_orientation_metrics,
            "promotion_guard_reason": promotion_guard_reason,
            "candidate_pose_map50": candidate_metrics.get("pose_map50"),
            "candidate_pose_map50_95": candidate_metrics.get("pose_map50_95"),
            "candidate_box_map50": candidate_metrics.get("box_map50"),
            "candidate_box_map50_95": candidate_metrics.get("box_map50_95"),
            "incumbent_pose_map50": incumbent_metrics.get("pose_map50"),
            "incumbent_pose_map50_95": incumbent_metrics.get("pose_map50_95"),
            "incumbent_box_map50": incumbent_metrics.get("box_map50"),
            "incumbent_box_map50_95": incumbent_metrics.get("box_map50_95"),
            "dataset": export_details,
            "detection_preset": detection_preset,
            "auto_tune": bool(auto_tune),
            "dataset_size_effective": dataset_size_effective,
            "dataset_size_source": dataset_size_source,
            "resolved_train_params": resolved_train_params,
            "preflight_warnings": list(export_details.get("orientation_preflight_warnings") or []),
        }
        dio.write_run_json(
            run_dir,
            "train_results.json",
            {
                **summary,
                "model_type": debug_model_type,
                "run_id": run_id,
                "tag": safe_name,
            },
        )
        dio.write_run_manifest(
            run_dir,
            model_type=debug_model_type,
            tag=safe_name,
            project_root=session_dir,
            extra={
                "status": "completed",
                "run_id": run_id,
                "active_model_path": summary.get("active_model_path"),
                "candidate_model_path": summary.get("candidate_model_path"),
                "promoted": should_promote,
                "candidate_map50": candidate_map50,
                "candidate_map50_95": candidate_map,
                "metric_type": eval_metric_type,
                "use_pose": bool(use_pose),
            },
        )
        logger.info(f"YOLO run debug saved to: {run_dir}")
        return summary

    # ------------------------------------------------------------------
    # Fine-tuned YOLOv8 detection
    # ------------------------------------------------------------------
    def detect_finetuned(self, image, finetuned_path, class_name, conf_threshold=0.5, top_k=10):
        """Run detection with a fine-tuned YOLOv8 model.

        Uses imgsz=640 to match the training resolution (trained with imgsz=640).
        Caches the loaded model to avoid ~2-5 s reload overhead on every call.
        """
        from ultralytics import YOLO

        # Load and cache the model; reload only when the path changes.
        if self._finetuned_model is None or self._finetuned_model_path != finetuned_path:
            self._finetuned_model = YOLO(finetuned_path)
            self._finetuned_model_path = finetuned_path

        # imgsz must match training (640) — using a larger size shifts the anchor
        # grid scale and causes the head to predict sub-regions of the target object.
        results = self._finetuned_model.predict(
            image, conf=conf_threshold, imgsz=640, device=self.device, verbose=False
        )

        names_map = getattr(results[0], "names", {}) or {}

        def _orientation_from_class_name(raw_name):
            token = str(raw_name or "").strip().lower().replace("-", "_").replace(" ", "_")
            if not token:
                return None
            if token.endswith("_left") or token == "left" or "_left_" in token:
                return "left"
            if token.endswith("_right") or token == "right" or "_right_" in token:
                return "right"
            return None

        boxes = []
        for idx, box in enumerate(results[0].boxes):
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else 0
            pred_class_name = names_map.get(cls_id, class_name) if hasattr(names_map, "get") else class_name
            box_dict = {
                "xyxy": xyxy,
                "confidence": round(conf, 3),
                "class_name": class_name,
                "class_id": cls_id,
            }
            if pred_class_name:
                box_dict["detector_class_name"] = pred_class_name
            hint_orientation = _orientation_from_class_name(pred_class_name)
            if hint_orientation is not None:
                box_dict["orientation_hint"] = {
                    "orientation": hint_orientation,
                    "confidence": round(conf, 4),
                    "source": "detector_class",
                }

            boxes.append(box_dict)

        boxes.sort(key=lambda b: b["confidence"], reverse=True)
        if top_k > 0 and len(boxes) > top_k:
            boxes = boxes[:top_k]

        return boxes

    # ------------------------------------------------------------------
    # SAM2 segment persistence (for synthetic YOLO augmentation)
    # ------------------------------------------------------------------
    def _save_segments(self, image, image_path, boxes, masks, session_dir):
        """
        Save SAM2 foreground segments to session_dir/segments/ for later use as
        synthetic augmentation data during YOLO fine-tuning.

        Each accepted (box, mask) pair is saved as:
          {hash}_{idx}_fg.png   — BGRA image (alpha channel = SAM2 mask)
          {hash}_{idx}_mask.png — binary mask (0 or 255)
          {hash}_{idx}_meta.json — box coords and source image path
        """
        seg_dir = os.path.join(session_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)

        img_hash = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:10]
        MIN_SEGMENT_SCORE = 0.55   # raised from 0.35 — good segments score 0.68–0.72
        MIN_CC_RATIO      = 0.88   # reject fragmented masks
        MIN_FILL_RATIO    = 0.38   # reject masks that barely cover their box
        accepted = 0
        rejected = 0

        # ── Phase 1: score, filter, and clean each mask ────────────────────
        # Collect passing candidates before saving so we can deduplicate by
        # pixel-level mask overlap (Phase 2) before writing any files (Phase 3).
        candidates = []  # (score, orig_idx, box_data, cleaned_mask_float, qc_metrics)

        for idx, (box_data, mask) in enumerate(zip(boxes, masks)):
            if mask is None:
                rejected += 1
                continue
            try:
                xyxy = box_data["xyxy"]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                img_h, img_w = image.shape[:2]

                # Reject boxes that touch ≥2 image edges — "slice" detections where
                # YOLO-World crops a partial fish against the image boundary. Even if
                # SAM2 produces a clean mask, the crop is not a complete fish silhouette.
                BOX_EDGE_TOL = 5
                box_at_edges = sum([
                    x1 <= BOX_EDGE_TOL,
                    y1 <= BOX_EDGE_TOL,
                    x2 >= img_w - BOX_EDGE_TOL,
                    y2 >= img_h - BOX_EDGE_TOL,
                ])
                if box_at_edges >= 2:
                    rejected += 1
                    continue

                # QC scoring with stricter per-metric thresholds
                score, qc_metrics = self._score_sam_mask(mask, xyxy, image.shape)
                if (score < MIN_SEGMENT_SCORE
                        or qc_metrics.get("cc_ratio", 0) < MIN_CC_RATIO
                        or qc_metrics.get("fill_ratio", 0) < MIN_FILL_RATIO):
                    logger.debug(
                        f"Segment {idx} rejected by QC (score={score:.3f}): {qc_metrics}"
                    )
                    rejected += 1
                    continue

                # ── Mask cleanup ──────────────────────────────────────────────
                # 1. Morphological closing: fill small internal holes
                mask_u8 = (mask * 255).astype(np.uint8)
                ksize = max(3, min(15, int(min(x2 - x1, y2 - y1) * 0.04)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
                # 2. Keep only the largest connected component (remove noise pixels)
                n_labels, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask_u8)
                if n_labels > 1:
                    largest_cc_idx = 1 + int(np.argmax(cc_stats[1:, cv2.CC_STAT_AREA]))
                    mask_u8 = ((cc_labels == largest_cc_idx).astype(np.uint8) * 255)
                # 3. Light erosion to avoid background bleed at mask edges
                erode_k = max(1, ksize // 4)
                erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
                mask_u8 = cv2.erode(mask_u8, erode_kernel, iterations=1)
                cleaned_mask = mask_u8.astype(np.float32) / 255.0

                candidates.append((score, idx, box_data, cleaned_mask, qc_metrics))

            except Exception as e:
                logger.warning(f"Failed to score segment {idx}: {e}")
                rejected += 1

        # ── Phase 2: greedy mask-IoU NMS ──────────────────────────────────
        # Sort by score descending, then greedily keep a segment only if its
        # pixel-level IoU with every already-kept mask is below the threshold.
        # This removes duplicates where the same fish is detected from two
        # slightly different bounding boxes.
        MASK_IOU_THRESH = 0.45
        candidates.sort(key=lambda c: c[0], reverse=True)
        kept = []
        for cand in candidates:
            score_c, idx_c, box_c, mask_c, qc_c = cand
            if any(self._mask_iou(mask_c, k[3]) >= MASK_IOU_THRESH for k in kept):
                rejected += 1
                continue
            kept.append(cand)

        # ── Phase 3: save kept segments ────────────────────────────────────
        for score, idx, box_data, mask, qc_metrics in kept:
            try:
                xyxy = box_data["xyxy"]
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                img_h, img_w = image.shape[:2]

                pad = 10
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(img_w, x2 + pad)
                cy2 = min(img_h, y2 + pad)

                fg_crop = image[cy1:cy2, cx1:cx2].copy()
                mask_crop = mask[cy1:cy2, cx1:cx2]

                if fg_crop.size == 0 or mask_crop.size == 0:
                    rejected += 1
                    continue

                # Create BGRA with mask as alpha channel
                fg_rgba = cv2.cvtColor(fg_crop, cv2.COLOR_BGR2BGRA)
                if mask_crop.shape[:2] != fg_crop.shape[:2]:
                    import cv2 as _cv
                    mask_crop = _cv.resize(mask_crop.astype(np.uint8),
                                           (fg_crop.shape[1], fg_crop.shape[0]),
                                           interpolation=_cv.INTER_NEAREST)
                fg_rgba[:, :, 3] = (mask_crop.astype(np.float32) * 255).astype(np.uint8)

                base = f"{img_hash}_{idx}"
                cv2.imwrite(os.path.join(seg_dir, f"{base}_fg.png"), fg_rgba)
                cv2.imwrite(os.path.join(seg_dir, f"{base}_mask.png"),
                            (mask_crop.astype(np.float32) * 255).astype(np.uint8))

                meta = {
                    "source_image": image_path,
                    "box": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                    "crop_origin": [cx1, cy1],
                    "qc": {"score": round(score, 4), **qc_metrics},
                }
                with open(os.path.join(seg_dir, f"{base}_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                accepted += 1

            except Exception as e:
                logger.warning(f"Failed to save segment {idx}: {e}")
                rejected += 1

        logger.info(f"Segment persistence: accepted {accepted} / rejected {rejected}")

    def save_segment_for_box(self, image: np.ndarray, image_path: str,
                              box_xyxy: list, session_dir: str) -> bool:
        """
        Run SAM2 on a single user-accepted box and persist the segment crop.

        Called when the user explicitly saves/accepts a bounding box.
        Uses a lenient QC floor (score >= 0.20) since the user has validated
        the box. Returns True if a segment was saved, False on failure.
        """
        if self.sam2_model is None:
            return False
        try:
            x1, y1, x2, y2 = [float(v) for v in box_xyxy]

            # ── Cache lookup ──────────────────────────────────────────────────
            # If this box was already segmented during annotate(), reuse that mask
            # instead of re-running SAM2 (avoids 100–500 ms per box at save time).
            best_mask, best_score = None, -1.0
            for cached_xyxy, cached_mask in self._mask_cache.get(image_path, []):
                if self._iou([x1, y1, x2, y2], cached_xyxy) >= 0.75:
                    best_mask = cached_mask  # already uint8 (0/1)
                    best_score, _ = self._score_sam_mask(best_mask, (x1, y1, x2, y2), image.shape)
                    logger.debug(f"save_segment_for_box: cache hit (iou≥0.75), skipping SAM2")
                    break

            if best_mask is None:
                # Cache miss (manually drawn box or resized beyond threshold) — run SAM2
                logger.debug(f"save_segment_for_box: cache miss, running SAM2")
                results = self.sam2_model.predict(
                    image, bboxes=[[x1, y1, x2, y2]], device=self.device, verbose=False
                )
                masks_data = results[0].masks.data.cpu().numpy().astype(np.uint8)
                for i in range(masks_data.shape[0]):
                    m = masks_data[i]
                    sc, _ = self._score_sam_mask(m, (x1, y1, x2, y2), image.shape)
                    if sc > best_score:
                        best_score, best_mask = sc, m

            # Only reject truly failed SAM2 runs
            if best_mask is None or best_score < 0.20:
                return False

            # Mask cleanup (same morphological pipeline as _save_segments)
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            ksize = max(3, min(15, int(min(ix2 - ix1, iy2 - iy1) * 0.04)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask_u8 = (best_mask * 255).astype(np.uint8)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
            n_labels, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask_u8)
            if n_labels > 1:
                largest_cc_idx = 1 + int(np.argmax(cc_stats[1:, cv2.CC_STAT_AREA]))
                mask_u8 = ((cc_labels == largest_cc_idx).astype(np.uint8) * 255)
            erode_k = max(1, ksize // 4)
            mask_u8 = cv2.erode(
                mask_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k)),
                iterations=1,
            )
            mask = mask_u8.astype(np.float32) / 255.0

            # Crop and save
            seg_dir = os.path.join(session_dir, "segments")
            os.makedirs(seg_dir, exist_ok=True)
            img_h, img_w = image.shape[:2]
            pad = 10
            cx1 = max(0, ix1 - pad)
            cy1 = max(0, iy1 - pad)
            cx2 = min(img_w, ix2 + pad)
            cy2 = min(img_h, iy2 + pad)
            fg_crop = image[cy1:cy2, cx1:cx2].copy()
            mask_crop = mask[cy1:cy2, cx1:cx2]
            if fg_crop.size == 0 or mask_crop.size == 0:
                return False

            fg_rgba = cv2.cvtColor(fg_crop, cv2.COLOR_BGR2BGRA)
            fg_rgba[:, :, 3] = (mask_crop * 255).astype(np.uint8)

            # Deterministic filenames: image-hash + box-coord-hash.
            # Re-saving the same accepted box overwrites rather than duplicates.
            img_hash = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:10]
            box_hash = hashlib.md5(
                f"{ix1},{iy1},{ix2},{iy2}".encode("utf-8")
            ).hexdigest()[:6]
            base = os.path.join(seg_dir, f"{img_hash}_{box_hash}")
            cv2.imwrite(f"{base}_fg.png", fg_rgba)
            cv2.imwrite(f"{base}_mask.png", (mask_crop * 255).astype(np.uint8))
            score_val, qc_metrics = self._score_sam_mask(
                best_mask, (x1, y1, x2, y2), image.shape
            )
            with open(f"{base}_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source_image": image_path,
                        "box": {"left": ix1, "top": iy1, "right": ix2, "bottom": iy2},
                        "crop_origin": [cx1, cy1],
                        "qc": {"score": round(score_val, 4), **qc_metrics},
                        "accepted_by_user": True,
                        "finalized_from_accepted_boxes": True,
                    },
                    f,
                )
            return True
        except Exception as e:
            logger.warning(f"save_segment_for_box failed: {e}")
            return False

    def purge_segments_for_image(self, image_path: str, session_dir: str) -> int:
        """
        Remove all persisted segment artifacts for a specific image.
        This ensures finalization writes ONLY the current accepted boxes.
        """
        seg_dir = os.path.join(session_dir, "segments")
        if not os.path.isdir(seg_dir):
            return 0
        img_hash = hashlib.md5(image_path.encode("utf-8")).hexdigest()[:10]
        removed = 0
        for fname in list(os.listdir(seg_dir)):
            if not fname.startswith(f"{img_hash}_"):
                continue
            fpath = os.path.join(seg_dir, fname)
            try:
                os.remove(fpath)
                removed += 1
            except Exception:
                pass
        return removed

    # ------------------------------------------------------------------
    # Full pipeline: annotate
    # ------------------------------------------------------------------
    def annotate(self, image_path, class_name, dlib_model=None, id_mapping_path=None, options=None):
        """Run the full SuperAnnotator pipeline on one image."""
        options = options or {}
        conf_threshold = options.get("conf_threshold", 0.5)
        sam_enabled = options.get("sam_enabled", False)
        max_objects = options.get("max_objects", 10)
        nms_iou = options.get("nms_iou", 0.6)
        detection_mode = options.get("detection_mode", "auto")  # "auto" or "manual"
        finetuned_model = options.get("finetuned_model")
        session_dir = options.get("session_dir")      # for SAM2 segment auto-save
        detection_preset = options.get("detection_preset", "balanced")
        pca_mode = options.get("pca_mode", "auto")
        use_orientation_hint = bool(options.get("use_orientation_hint", True))
        if pca_mode not in ("off", "on", "auto"):
            pca_mode = "auto"

        orientation_policy = None
        if session_dir:
            try:
                orientation_policy = ou.load_orientation_policy(session_dir)
            except Exception:
                orientation_policy = None
        resolved = self._resolve_detection_preset(
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
            max_objects=max_objects,
            detection_preset=detection_preset,
        )
        conf_threshold = resolved["conf"]
        nms_iou = resolved["iou"]
        max_objects = resolved["top_k"]
        detect_imgsz = resolved["imgsz"]
        allow_relaxed_retry = resolved["allow_relaxed_retry"]

        # Load image
        send_progress("Loading image...", 5, "detection")
        image = self._load_image(image_path)
        img_h, img_w = image.shape[:2]

        # Stage A: Detection
        # Priority: fine-tuned YOLOv8 > YOLO-World zero-shot > classic CV fallback
        send_progress("Detecting objects...", 15, "detection")
        use_finetuned = (finetuned_model and os.path.exists(finetuned_model))
        use_yolo = (detection_mode == "auto" and self.yolo_model is not None and not use_finetuned)

        if use_finetuned:
            try:
                boxes = self.detect_finetuned(image, finetuned_model, class_name,
                                               conf_threshold, top_k=max_objects)
                detection_method = "yolov8_finetuned"
            except Exception as e:
                logger.warning(f"Fine-tuned model failed, falling back: {e}")
                # Fall through to YOLO-World
                use_yolo = (detection_mode == "auto" and self.yolo_model is not None)
                use_finetuned = False

        if not use_finetuned and use_yolo:
            try:
                boxes = self.detect_yolo(image, class_name, conf_threshold,
                                         nms_iou=nms_iou, top_k=max_objects, imgsz=detect_imgsz)
                # Retry with relaxed threshold if first pass finds nothing.
                if allow_relaxed_retry and not boxes:
                    relaxed_conf = max(0.08, float(conf_threshold) * 0.5)
                    if relaxed_conf < conf_threshold:
                        send_progress(
                            f"No detections at {conf_threshold:.2f}, retrying at {relaxed_conf:.2f}...",
                            22,
                            "detection",
                        )
                        boxes = self.detect_yolo(
                            image,
                            class_name,
                            relaxed_conf,
                            nms_iou=nms_iou,
                            top_k=max_objects,
                            imgsz=max(detect_imgsz, 1536),
                        )
                detection_method = "yolo_world"
            except Exception as e:
                err_str = str(e)
                # Distinguish recoverable runtime errors (device mismatch, OOM) from
                # unrecoverable init errors (missing CLIP/ultralytics dependency).
                is_missing_dep = (
                    "No module named" in err_str
                    or "ModuleNotFoundError" in err_str
                    or "ImportError" in err_str
                )
                if is_missing_dep:
                    # Unrecoverable — permanently disable YOLO.
                    self.yolo_model = None
                    self.yolo_init_attempted = True
                    self.yolo_init_error = self._format_yolo_error(e)
                    logger.warning(f"YOLO detection failed (missing dep), disabling: {self.yolo_init_error}")
                else:
                    # Recoverable runtime error (e.g. device mismatch) — keep the model
                    # loaded so the next detection attempt can succeed after re-patching.
                    logger.warning(f"YOLO detection failed (runtime): {err_str}")
                    # Re-apply the txt_feats patch in case predictor state was reset.
                    try:
                        self._patch_set_classes()
                        self._move_txt_feats_to_device()
                    except Exception:
                        pass
                send_progress("YOLO unavailable, falling back to classic detection...", 18, "detection")
                min_area = options.get("min_area_ratio", 0.02)
                boxes = self.detect_classic(image, min_area)
                detection_method = "opencv_fallback"

        if not use_finetuned and not use_yolo:
            min_area = options.get("min_area_ratio", 0.02)
            boxes = self.detect_classic(image, min_area)
            detection_method = "opencv"

        if max_objects > 0 and len(boxes) > max_objects:
            boxes = boxes[:max_objects]

        if not boxes:
            return {
                "status": "result",
                "objects": [],
                "image_width": img_w,
                "image_height": img_h,
                "detection_method": detection_method,
                "num_detections": 0,
            }

        # Stage A.5: Optional SAM2 refinement (user must explicitly enable)
        masks = [None] * len(boxes)
        if sam_enabled and self.sam2_model is not None and (use_yolo or use_finetuned):
            send_progress("Refining with SAM2...", 35, "segmentation")
            masks = self.refine_with_sam2(image, boxes)
            if any(m is not None for m in masks):
                detection_method += "+sam2"
                # Segments are NOT auto-saved here.
                # They are saved only when the user explicitly accepts boxes
                # via the save_segments_for_boxes command.

        self._cached_sam_results = list(zip(boxes, masks))

        # Populate mask cache so save_segment_for_box can reuse masks without re-running SAM2.
        if any(m is not None for m in masks):
            self._mask_cache[image_path] = [
                (b["xyxy"], m) for b, m in zip(boxes, masks) if m is not None
            ]
            # Evict oldest entry when over capacity
            while len(self._mask_cache) > self._mask_cache_max:
                self._mask_cache.pop(next(iter(self._mask_cache)))

        # Stage B + C + D: Normalize, predict, map back
        has_dlib = False
        if dlib_model and os.path.exists(dlib_model):
            try:
                send_progress("Loading landmark model...", 50, "prediction")
                self.load_dlib_model(dlib_model, id_mapping_path)
                has_dlib = True
            except Exception as e:
                logger.warning(f"Failed to load dlib model: {e}")

        objects = []
        for i, (box_data, mask) in enumerate(zip(boxes, masks)):
            pct = 55 + int(40 * (i / len(boxes)))
            send_progress(f"Processing object {i + 1}/{len(boxes)}...", pct, "normalization")

            xyxy = box_data["xyxy"]
            x1, y1, x2, y2 = xyxy

            # Use tight box from mask if available
            tight_xyxy = self.tight_box_from_mask(mask) if mask is not None else None
            effective_xyxy = tight_xyxy if tight_xyxy else xyxy

            orientation_hint = None
            if use_orientation_hint:
                orientation_hint = ou.resolve_orientation_hint(
                    box_data.get("orientation_hint"),
                    box_xyxy=box_data.get("xyxy"),
                    min_confidence=0.25,
                    min_dx_ratio=0.06,
                )

            # Standardize (+ optional SAM2/PCA canonicalization)
            standardized, metadata = self.standardize_instance(
                image,
                effective_xyxy,
                mask,
                pca_mode=pca_mode,
                orientation_policy=orientation_policy,
                orientation_hint=orientation_hint,
            )

            # Predict landmarks with the same orientation logic as predict.py.
            landmarks = []
            orientation_debug = None
            if has_dlib:
                orientation_mode = ou.get_orientation_mode(orientation_policy or {})
                if orientation_mode == "directional":
                    if orientation_hint in ("left", "right"):
                        landmarks_512 = self.predict_landmarks(standardized) or []
                        flip_crop = False
                        canonical = metadata.get("canonicalization") if isinstance(metadata.get("canonicalization"), dict) else {}
                        orientation_debug = {
                            "used_flipped_crop": False,
                            "selection_reason": "directional_detector_hint_only",
                            "candidate_b_evaluated": False,
                            "target_orientation": self.dlib_target_orientation,
                            "locked_from_canonicalization": True,
                            "lock_direction_source": canonical.get("direction_source"),
                            "lock_direction_confidence": canonical.get("direction_confidence"),
                        }
                    else:
                        resolved_target = self.dlib_target_orientation
                        if resolved_target not in ("left", "right"):
                            resolved_target = str((orientation_policy or {}).get("targetOrientation", "")).strip().lower()
                            if resolved_target not in ("left", "right"):
                                resolved_target = None
                        landmarks_512, flip_crop, orientation_debug = ou.select_orientation(
                            standardized,
                            self.predict_landmarks,
                            target_orientation=resolved_target,
                            landmark_template=self.dlib_landmark_template,
                            head_id=self.dlib_head_landmark_id,
                            tail_id=self.dlib_tail_landmark_id,
                            orientation_hint_original=None,
                        )
                        if not isinstance(orientation_debug, dict):
                            orientation_debug = {}
                        orientation_debug["locked_from_canonicalization"] = False
                        orientation_debug["orientation_warning"] = {
                            "code": "legacy_detector_no_orientation_hint",
                            "message": (
                                "Directional schema without detector orientation hint: "
                                "using dual-candidate fallback. Retrain YOLO with orientation-aware classes."
                            ),
                        }
                else:
                    lock_orientation = ou.should_lock_orientation_from_canonicalization(
                        metadata.get("canonicalization"),
                        policy=orientation_policy,
                    )
                    if lock_orientation:
                        landmarks_512 = self.predict_landmarks(standardized) or []
                        flip_crop = False
                        canonical = metadata.get("canonicalization") if isinstance(metadata.get("canonicalization"), dict) else {}
                        orientation_debug = {
                            "used_flipped_crop": False,
                            "selection_reason": "locked_canonical_orientation",
                            "candidate_b_evaluated": False,
                            "target_orientation": self.dlib_target_orientation,
                            "locked_from_canonicalization": True,
                            "lock_direction_source": canonical.get("direction_source"),
                            "lock_direction_confidence": canonical.get("direction_confidence"),
                        }
                    else:
                        landmarks_512, flip_crop, orientation_debug = ou.select_orientation(
                            standardized,
                            self.predict_landmarks,
                            target_orientation=self.dlib_target_orientation,
                            landmark_template=self.dlib_landmark_template,
                            head_id=self.dlib_head_landmark_id,
                            tail_id=self.dlib_tail_landmark_id,
                            orientation_hint_original=orientation_hint,
                        )
                if isinstance(orientation_debug, dict) and metadata.get("canonicalization") is not None:
                    orientation_debug["canonicalization"] = metadata.get("canonicalization")
                if isinstance(orientation_debug, dict):
                    orientation_debug["orientation_hint"] = orientation_hint
                    orientation_debug["orientation_hint_raw"] = box_data.get("orientation_hint")
                landmarks = self.map_to_original(
                    landmarks_512,
                    metadata,
                    was_flipped=flip_crop,
                    image_shape=image.shape[:2],
                )

            # Mask outline
            outline = self.mask_to_outline(mask)

            obj = {
                "box": {
                    "left": int(x1),
                    "top": int(y1),
                    "right": int(x2),
                    "bottom": int(y2),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
                "mask_outline": outline,
                "landmarks": landmarks,
                "confidence": box_data["confidence"],
                "class_name": box_data["class_name"],
                "instance_metadata": metadata,
                "detection_method": detection_method,
                "orientation_hint": box_data.get("orientation_hint"),
                "orientation_debug": orientation_debug,
            }
            objects.append(obj)

        send_progress("Done", 100, "done")

        return {
            "status": "result",
            "objects": objects,
            "image_width": img_w,
            "image_height": img_h,
            "detection_method": detection_method,
            "num_detections": len(objects),
        }

    # ------------------------------------------------------------------
    # SAM2 re-prompt (interactive refinement)
    # ------------------------------------------------------------------
    def refine_sam(self, image_path, object_index, click_point, click_label=1):
        """Re-prompt SAM2 with a user click point for mask correction."""
        if self.sam2_model is None:
            return {"status": "error", "error": "SAM2 not loaded"}

        image = self._load_image(image_path)

        if self._cached_sam_results is None or object_index >= len(self._cached_sam_results):
            return {"status": "error", "error": f"No cached results for object {object_index}"}

        box_data, _ = self._cached_sam_results[object_index]
        xyxy = box_data["xyxy"]

        try:
            results = self.sam2_model.predict(
                image,
                bboxes=[xyxy],
                points=[click_point],
                labels=[click_label],
                device=self.device,
                verbose=False,
            )
            mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            outline = self.mask_to_outline(mask)

            # Update cache
            self._cached_sam_results[object_index] = (box_data, mask)

            return {
                "status": "result",
                "ok": True,
                "mask_outline": outline,
                "object_index": object_index,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ------------------------------------------------------------------
    # SAM2 re-segment with a corrected bounding box
    # ------------------------------------------------------------------
    def resegment_box(self, image_path, box_xyxy):
        """Run SAM2 with a user-provided bounding box prompt (no cache needed)."""
        if self.sam2_model is None:
            return {"status": "error", "error": "SAM2 not loaded"}

        image = self._load_image(image_path)
        if image is None:
            return {"status": "error", "error": f"Could not read image: {image_path}"}
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        # Clamp incoming box to image bounds for robustness with imported labels.
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(0.0, min(float(w), x2))
        y2 = max(0.0, min(float(h), y2))
        if x2 <= x1:
            x2 = min(float(w), x1 + 1.0)
        if y2 <= y1:
            y2 = min(float(h), y1 + 1.0)
        try:
            results = self.sam2_model.predict(
                image,
                bboxes=[[x1, y1, x2, y2]],
                device=self.device,
                verbose=False,
            )
            masks_obj = getattr(results[0], "masks", None)
            masks_data = getattr(masks_obj, "data", None)  # shape [N, H, W]
            if masks_data is None or masks_data.shape[0] == 0:
                return {"status": "error", "error": "SAM2 returned no masks"}

            # Score all candidate masks; pick the best one
            best_mask = None
            best_score = -1.0
            for i in range(masks_data.shape[0]):
                m = masks_data[i].cpu().numpy().astype(np.uint8)
                score, _ = self._score_sam_mask(m, (x1, y1, x2, y2), image.shape)
                if score > best_score:
                    best_score = score
                    best_mask = m

            if best_mask is None:
                return {"status": "error", "error": "SAM2 produced no usable mask"}

            outline = self.mask_to_outline(best_mask)
            if not outline or len(outline) < 3:
                return {"status": "error", "error": "SAM2 mask outline was empty"}
            return {
                "status": "result",
                "ok": True,
                "mask_outline": outline,
                "score": round(best_score, 4),
                "low_quality": bool(best_score < 0.10),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ======================================================================
# Main loop
# ======================================================================
def main():
    annotator = SuperAnnotator()
    logger.info("SuperAnnotator process started, waiting for commands...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            send({"status": "error", "error": f"Invalid JSON: {e}"})
            continue

        try:
            command = cmd.get("cmd", "")
            request_id = cmd.get("_request_id")
            global _CURRENT_REQUEST_ID
            _CURRENT_REQUEST_ID = request_id if isinstance(request_id, str) else None

            if command == "init":
                result = annotator.init_models()
                send(result)

            elif command == "check":
                result = annotator.check()
                send(result)

            elif command == "annotate":
                result = annotator.annotate(
                    image_path=cmd["image_path"],
                    class_name=cmd.get("class_name", "object"),
                    dlib_model=cmd.get("dlib_model"),
                    id_mapping_path=cmd.get("id_mapping_path"),
                    options=cmd.get("options"),
                )
                send(result)

            elif command == "train_yolo":
                train_summary = annotator.train_yolo(
                    session_dir=cmd["session_dir"],
                    class_name=cmd.get("class_name", "object"),
                    epochs=cmd.get("epochs"),
                    detection_preset=cmd.get("detection_preset", "balanced"),
                    dataset_size=cmd.get("dataset_size"),
                    auto_tune=cmd.get("auto_tune", True),
                )
                send({"status": "result", **train_summary})

            elif command == "preview_yolo_train_plan":
                plan = annotator.preview_yolo_train_plan(
                    session_dir=cmd["session_dir"],
                    class_name=cmd.get("class_name", "object"),
                    epochs=cmd.get("epochs"),
                    detection_preset=cmd.get("detection_preset", "balanced"),
                    dataset_size=cmd.get("dataset_size"),
                    auto_tune=cmd.get("auto_tune", True),
                )
                send({"status": "result", **plan})

            elif command == "refine_sam":
                result = annotator.refine_sam(
                    image_path=cmd["image_path"],
                    object_index=cmd.get("object_index", 0),
                    click_point=cmd["click_point"],
                    click_label=cmd.get("click_label", 1),
                )
                send(result)

            elif command == "resegment_box":
                result = annotator.resegment_box(
                    image_path=cmd["image_path"],
                    box_xyxy=cmd["box_xyxy"],
                )
                send(result)

            elif command == "save_segments_for_boxes":
                img_path = cmd["image_path"]
                boxes_xyxy = cmd.get("boxes", [])
                sess_dir = cmd["session_dir"]
                if annotator.sam2_model is None:
                    send({"status": "ok", "saved": 0, "skipped": "no_sam2"})
                else:
                    image = annotator._load_image(img_path)
                    # Finalization semantics: replace prior segments for this image
                    # with the current accepted box set.
                    removed = annotator.purge_segments_for_image(img_path, sess_dir)
                    saved = sum(
                        annotator.save_segment_for_box(image, img_path, box, sess_dir)
                        for box in boxes_xyxy
                    )
                    send({"status": "ok", "saved": saved, "total": len(boxes_xyxy), "removed": removed})

            elif command == "shutdown":
                send({"status": "ok", "message": "Shutting down"})
                logger.info("Shutdown requested, exiting")
                break

            else:
                send({"status": "error", "error": f"Unknown command: {command}"})

        except Exception as e:
            logger.error(f"Error processing command: {traceback.format_exc()}")
            send({"status": "error", "error": str(e)})
        finally:
            _CURRENT_REQUEST_ID = None

    sys.exit(0)


if __name__ == "__main__":
    main()
