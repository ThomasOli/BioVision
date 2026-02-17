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
import json
import os
import shutil
import traceback
from datetime import datetime

import numpy as np
import cv2

# Ensure all print/logging goes to stderr so stdout is reserved for JSON protocol
import logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[SuperAnnotator] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path for image_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_utils import load_image

STANDARD_SIZE = 512


def send(obj):
    """Send a JSON object to stdout (one line)."""
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
        self.mode = "classic_fallback"
        self.gpu = False
        self.yolo_init_attempted = False
        self.yolo_init_error = None
        self.sam2_init_attempted = False
        self.sam2_init_error = None
        self._cached_image_path = None
        self._cached_image = None
        self._cached_sam_results = None

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
    def _extract_map_metrics(val_results):
        map50 = None
        map_all = None
        try:
            if hasattr(val_results, "box"):
                map50 = float(getattr(val_results.box, "map50", None))
                map_all = float(getattr(val_results.box, "map", None))
            elif hasattr(val_results, "results_dict"):
                rd = val_results.results_dict or {}
                map50 = rd.get("metrics/mAP50(B)")
                map_all = rd.get("metrics/mAP50-95(B)")
        except Exception:
            map50 = None
            map_all = None
        if map50 is not None:
            map50 = round(float(map50), 5)
        if map_all is not None:
            map_all = round(float(map_all), 5)
        return map50, map_all

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
        return self._extract_map_metrics(val_results)

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
                # Smoke-test open-vocabulary text encoder so missing CLIP is caught at init,
                # not only during first detection call.
                self.yolo_model.set_classes(["object"])
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

    # ------------------------------------------------------------------
    # Stage A: Detection
    # ------------------------------------------------------------------
    def detect_yolo(self, image, class_name, conf_threshold=0.5, nms_iou=0.6, top_k=10, imgsz=1280):
        """YOLO-World open-vocabulary detection with NMS and top-k filtering."""
        prompts = self._build_class_prompts(class_name)
        self.yolo_model.set_classes(prompts)
        results = self.yolo_model.predict(
            image,
            conf=conf_threshold,
            imgsz=imgsz,
            iou=min(0.85, max(float(nms_iou), 0.65)),
            max_det=max(50, int(top_k) * 6),
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
    def refine_with_sam2(self, image, boxes):
        """Refine YOLO boxes with SAM2 masks."""
        masks = []
        for i, box_data in enumerate(boxes):
            try:
                xyxy = box_data["xyxy"]
                results = self.sam2_model.predict(image, bboxes=[xyxy], verbose=False)
                mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
                masks.append(mask)
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
    def standardize_instance(self, image, xyxy, mask=None):
        """
        Crop, rotate via PCA, and resize to STANDARD_SIZE × STANDARD_SIZE.
        Returns (standardized_image, instance_metadata).
        """
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        img_h, img_w = image.shape[:2]
        w, h = x2 - x1, y2 - y1

        # 1. Crop with 20% padding
        pad = 0.20
        crop_x1 = max(0, int(x1 - w * pad))
        crop_y1 = max(0, int(y1 - h * pad))
        crop_x2 = min(img_w, int(x2 + w * pad))
        crop_y2 = min(img_h, int(y2 + h * pad))
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop.size == 0:
            # Fallback: use original box
            crop = image[y1:y2, x1:x2]
            crop_x1, crop_y1 = x1, y1
            crop_x2, crop_y2 = x2, y2

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        # 2. PCA orientation from mask
        angle = 0.0
        if mask is not None:
            # Crop the mask to match
            mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
            points = np.column_stack(np.where(mask_crop > 0))
            if len(points) > 10:
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca.fit(points)
                    angle = float(np.degrees(
                        np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
                    ))
                except ImportError:
                    # sklearn not available, use minAreaRect fallback
                    contours, _ = cv2.findContours(
                        mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
                        angle = float(rect[2])
                        if angle < -45:
                            angle += 90

        # 3. Rotate so main axis is horizontal
        center = (crop.shape[1] // 2, crop.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]),
                                  borderMode=cv2.BORDER_REPLICATE)

        # 4. Resize to standard size
        standardized = cv2.resize(rotated, (STANDARD_SIZE, STANDARD_SIZE),
                                   interpolation=cv2.INTER_LINEAR)

        metadata = {
            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            "crop_origin": [crop_x1, crop_y1],
            "crop_size": [crop_w, crop_h],
            "rotation": angle,
            "scale": float(STANDARD_SIZE / max(crop_w, crop_h)) if max(crop_w, crop_h) > 0 else 1.0,
        }

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
        if id_mapping_path and os.path.exists(id_mapping_path):
            try:
                with open(id_mapping_path, "r") as f:
                    raw = json.load(f)
                self.dlib_id_mapping = {int(k): int(v) for k, v in raw.items()}
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
    def map_to_original(self, landmarks_512, metadata):
        """Map 512×512 landmarks back to original image coordinates."""
        scale = metadata["scale"]
        crop_w, crop_h = metadata["crop_size"]
        cx, cy = crop_w / 2.0, crop_h / 2.0
        angle_rad = np.radians(-metadata["rotation"])
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        ox, oy = metadata["crop_origin"]

        mapped = []
        for lm in landmarks_512:
            # 1. Un-resize: 512 → crop size
            x_crop = lm["x"] / scale
            y_crop = lm["y"] / scale

            # 2. Un-rotate by -θ around crop center
            x_unrot = cx + (x_crop - cx) * cos_a - (y_crop - cy) * sin_a
            y_unrot = cy + (x_crop - cx) * sin_a + (y_crop - cy) * cos_a

            # 3. Offset by crop origin
            x_orig = x_unrot + ox
            y_orig = y_unrot + oy

            mapped.append({
                "id": lm["id"],
                "x": round(x_orig, 1),
                "y": round(y_orig, 1),
            })
        return mapped

    # ------------------------------------------------------------------
    # YOLOv8 fine-tuning
    # ------------------------------------------------------------------
    def train_yolo(self, session_dir, class_name, epochs=25):
        """Fine-tune session YOLO with versioning + validation-based promotion."""
        send_progress("Exporting dataset...", 5, "training")
        from export_yolo_dataset import export_dataset
        export_details = export_dataset(session_dir, class_name, return_details=True)
        dataset_yaml = export_details["yaml_path"]

        models_dir = os.path.join(session_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        safe_name = self._safe_class_name(class_name)
        alias_path = os.path.join(models_dir, f"yolo_{safe_name}.pt")
        registry_path = os.path.join(models_dir, f"yolo_{safe_name}_registry.json")
        registry = self._load_yolo_registry(registry_path)
        runs = registry.get("training_runs", [])
        next_version = len(runs) + 1

        active_path = None
        active_entry = registry.get("active_model")
        if isinstance(active_entry, dict):
            active_path = active_entry.get("path")
        if not active_path and os.path.exists(alias_path):
            active_path = alias_path

        send_progress("Starting YOLOv8 training...", 10, "training")
        from ultralytics import YOLO
        base_weights = active_path if active_path and os.path.exists(active_path) else "yolov8s.pt"
        model = YOLO(base_weights)

        # Training with progress callbacks
        def on_train_epoch_end(trainer):
            epoch = trainer.epoch + 1
            total = trainer.epochs
            pct = 10 + int(80 * (epoch / total))
            send_progress(f"Training epoch {epoch}/{total}...", pct, "training")

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=640,
            freeze=10,
            batch=4,
            project=os.path.join(session_dir, "yolo_train"),
            name=f"run_v{next_version}",
            exist_ok=True,
            verbose=False,
        )

        # Resolve trained checkpoint
        best = os.path.join(session_dir, "yolo_train", f"run_v{next_version}", "weights", "best.pt")
        if not os.path.exists(best):
            best = os.path.join(session_dir, "yolo_train", f"run_v{next_version}", "weights", "last.pt")
        if not os.path.exists(best):
            raise FileNotFoundError("YOLO training finished but no best.pt/last.pt was found.")

        candidate_path = os.path.join(models_dir, f"yolo_{safe_name}_v{next_version}.pt")
        shutil.copy2(best, candidate_path)

        send_progress("Evaluating detector quality...", 92, "training")
        candidate_map50, candidate_map = self._evaluate_detector(candidate_path, dataset_yaml)

        incumbent_map50 = None
        incumbent_map = None
        if active_path and os.path.exists(active_path):
            try:
                incumbent_map50, incumbent_map = self._evaluate_detector(active_path, dataset_yaml)
            except Exception as e:
                logger.warning(f"Failed to evaluate incumbent detector, proceeding with promotion: {e}")

        should_promote = (
            incumbent_map50 is None
            or candidate_map50 is None
            or candidate_map50 >= incumbent_map50 - 1e-4
        )
        if should_promote:
            shutil.copy2(candidate_path, alias_path)

        run_entry = {
            "version": next_version,
            "class_name": class_name,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "base_weights": base_weights,
            "path": candidate_path,
            "promoted": should_promote,
            "candidate_map50": candidate_map50,
            "candidate_map50_95": candidate_map,
            "incumbent_map50": incumbent_map50,
            "incumbent_map50_95": incumbent_map,
            "dataset": export_details,
        }
        registry["training_runs"] = [*runs, run_entry]
        if should_promote:
            registry["active_model"] = {
                "version": next_version,
                "path": candidate_path,
                "map50": candidate_map50,
                "map50_95": candidate_map,
                "updated_at": run_entry["created_at"],
            }
        self._save_yolo_registry(registry_path, registry)

        send_progress("Training complete", 100, "training")
        logger.info(f"YOLOv8 candidate model saved to {candidate_path} (promoted={should_promote})")
        return {
            "active_model_path": alias_path if os.path.exists(alias_path) else candidate_path,
            "candidate_model_path": candidate_path,
            "registry_path": registry_path,
            "version": next_version,
            "promoted": should_promote,
            "candidate_map50": candidate_map50,
            "candidate_map50_95": candidate_map,
            "incumbent_map50": incumbent_map50,
            "incumbent_map50_95": incumbent_map,
            "dataset": export_details,
        }

    # ------------------------------------------------------------------
    # Fine-tuned YOLOv8 detection
    # ------------------------------------------------------------------
    def detect_finetuned(self, image, finetuned_path, class_name, conf_threshold=0.5, top_k=10):
        """Run detection with a fine-tuned YOLOv8 model."""
        from ultralytics import YOLO
        ft_model = YOLO(finetuned_path)
        results = ft_model.predict(image, conf=conf_threshold, imgsz=1280, verbose=False)

        boxes = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            boxes.append({
                "xyxy": xyxy,
                "confidence": round(conf, 3),
                "class_name": class_name,
            })

        boxes.sort(key=lambda b: b["confidence"], reverse=True)
        if top_k > 0 and len(boxes) > top_k:
            boxes = boxes[:top_k]

        return boxes

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
        detection_preset = options.get("detection_preset", "balanced")
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
                # If YOLO fails at runtime (e.g. missing clip), degrade gracefully to classic CV.
                self.yolo_model = None
                self.yolo_init_attempted = True
                self.yolo_init_error = self._format_yolo_error(e)
                logger.warning(f"YOLO detection failed, falling back to classic CV: {self.yolo_init_error}")
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
        if sam_enabled and self.sam2_model is not None and use_yolo:
            send_progress("Refining with SAM2...", 35, "segmentation")
            masks = self.refine_with_sam2(image, boxes)
            if any(m is not None for m in masks):
                detection_method += "+sam2"

        self._cached_sam_results = list(zip(boxes, masks))

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

            # Standardize
            standardized, metadata = self.standardize_instance(image, effective_xyxy, mask)

            # Predict landmarks
            landmarks = []
            if has_dlib:
                landmarks_512 = self.predict_landmarks(standardized)
                landmarks = self.map_to_original(landmarks_512, metadata)

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
                    epochs=cmd.get("epochs", 25),
                )
                send({"status": "result", **train_summary})

            elif command == "refine_sam":
                result = annotator.refine_sam(
                    image_path=cmd["image_path"],
                    object_index=cmd.get("object_index", 0),
                    click_point=cmd["click_point"],
                    click_label=cmd.get("click_label", 1),
                )
                send(result)

            elif command == "shutdown":
                send({"status": "ok", "message": "Shutting down"})
                logger.info("Shutdown requested, exiting")
                break

            else:
                send({"status": "error", "error": f"Unknown command: {command}"})

        except Exception as e:
            logger.error(f"Error processing command: {traceback.format_exc()}")
            send({"status": "error", "error": str(e)})

    sys.exit(0)


if __name__ == "__main__":
    main()
