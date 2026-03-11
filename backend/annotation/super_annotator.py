#!/usr/bin/env python3
"""
SuperAnnotator Ã¢â‚¬â€ Persistent Python controller for BioVision.

Runs as a long-lived process, communicating via line-delimited JSON over stdin/stdout.
Combines session OBB detection, optional SAM2 segmentation, and Dlib landmark
prediction into one schema-aware pipeline.

Commands (JSON per line on stdin):
  {"cmd": "init"}
  {"cmd": "check"}
  {"cmd": "annotate", "image_path": "...", "class_name": "Fish", ...}
  {"cmd": "refine_sam", "image_path": "...", "object_index": 0, "click_point": [x,y], "click_label": 1}
  {"cmd": "resegment_box", "image_path": "...", "box_xyxy": [x1,y1,x2,y2]}
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

import sys as _sys, os as _os
_BACKEND_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _BACKEND_ROOT not in _sys.path:
    _sys.path.insert(0, _BACKEND_ROOT)

from bv_utils.image_utils import load_image

STANDARD_SIZE = 512


def send(obj):
    """Send a JSON object to stdout (one line)."""
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# Set by the main loop for each incoming command so all outgoing messages
# (progress and final) can echo the same ID back to Electron.
_current_request_id = None


def send_progress(message, percent, stage="processing"):
    obj = {"status": "progress", "message": message, "percent": percent, "stage": stage}
    if _current_request_id:
        obj["_request_id"] = _current_request_id
    send(obj)


def send_response(result):
    """Send a final response, echoing _request_id so Electron can match it."""
    if _current_request_id:
        result = {**result, "_request_id": _current_request_id}
    send(result)


def _aabb_iou(a, b):
    """Compute IoU between two AABB tuples (x1, y1, x2, y2)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _class_agnostic_dedup(detections, iou_threshold=0.5):
    """Remove lower-confidence duplicates that overlap above threshold, ignoring class."""
    kept = []
    for det in sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True):
        corners = det.get("corners") or []
        if len(corners) < 2:
            kept.append(det)
            continue
        xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
        b = (min(xs), min(ys), max(xs), max(ys))
        overlap = False
        for k in kept:
            kc = k.get("corners") or []
            if len(kc) < 2:
                continue
            kxs = [p[0] for p in kc]; kys = [p[1] for p in kc]
            kb = (min(kxs), min(kys), max(kxs), max(kys))
            if _aabb_iou(b, kb) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(det)
    return kept


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

    def _set_yolo_classes(self, classes):
        """Call set_classes and fix CUDA/CPU device mismatch for text features.

        After the first predict(device='cuda'), PyTorch's .to('cuda') moves all
        nn.Module parameters Ã¢â‚¬â€ including CLIP's token_embedding Ã¢â‚¬â€ to CUDA.
        But the CLIP wrapper's self.device attribute is a plain Python string
        that never gets updated, so its tokenize() still sends tokens to CPU
        while token_embedding is now on CUDA.  We sync clip_model.device to
        match the actual parameter device before every set_classes() call.
        """
        try:
            clip_model = getattr(self.yolo_model.model, "clip_model", None)
            if clip_model is not None:
                actual_device = next(clip_model.model.parameters()).device
                clip_model.device = actual_device
        except Exception:
            pass
        self.yolo_model.set_classes(classes)

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
    def _resolve_detection_preset(conf_threshold, nms_iou, max_objects, detection_preset, task="generic"):
        preset = (detection_preset or "balanced").strip().lower()
        conf = float(conf_threshold)
        iou = float(nms_iou)
        top_k = int(max_objects)
        resolved_task = str(task or "generic").strip().lower()
        imgsz = 640 if resolved_task == "obb" else 1280
        allow_relaxed_retry = True

        if resolved_task == "obb":
            if preset == "precision":
                conf = max(conf, 0.45)
                top_k = min(top_k, 8)
                allow_relaxed_retry = False
            elif preset == "recall":
                conf = min(conf, 0.2)
                top_k = max(top_k, 30)
                imgsz = 960
                allow_relaxed_retry = False
            elif preset == "single_object":
                conf = max(conf, 0.35)
                top_k = 1
                allow_relaxed_retry = False
            else:
                preset = "balanced"
                conf = max(0.3, min(conf, 0.9))
                top_k = max(1, min(top_k, 25))
                allow_relaxed_retry = False
        else:
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
        elif free_ram_gb > 1.0:
            mode = "auto_lite"
        else:
            mode = "classic_fallback"

        obb_capable = gpu or free_ram_gb > 1.0
        obb_model_tier = "none"
        if gpu and free_ram_gb > 4:
            obb_model_tier = "medium"   # yolov8m-obb.pt
        elif free_ram_gb > 1.0:
            obb_model_tier = "nano"     # yolov8n-obb.pt, freeze backbone
        elif gpu:
            obb_model_tier = "small"    # yolov8s-obb.pt

        self.gpu = gpu
        return {
            "mode": mode,
            "gpu": gpu,
            "free_ram_gb": free_ram_gb,
            "obb_capable": obb_capable,
            "obb_model_tier": obb_model_tier,
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def init_models(self):
        """Load models based on detected capabilities. Idempotent Ã¢â‚¬â€ safe to call multiple times."""
        if self.yolo_init_attempted and self.mode not in (None, "unknown"):
            return {
                "status": "already_initialized",
                "yolo_ready": self.yolo_model is not None,
                "sam2_ready": self.sam2_model is not None,
                "mode": self.mode,
            }

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
                self._set_yolo_classes(["object"])
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
            "obb_capable": caps["obb_capable"],
            "obb_model_tier": caps["obb_model_tier"],
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
    # ------------------------------------------------------------------
    # Stage A.5: SAM2 refinement
    # ------------------------------------------------------------------
    def _iterative_sam2_segment(self, image, xyxy, img_w, img_h,
                                max_iter=3, edge_thresh=5, expand_ratio=0.15):
        """Run SAM2 with automatic boundary-aware box expansion (up to max_iter passes).

        After each SAM2 pass, checks if the mask touches the bounding box edge
        (within edge_thresh pixels). If so, expands that edge by expand_ratio of
        the box dimension and reruns. Stops early when the mask no longer reaches
        any edge (converged) or when image boundaries are hit.

        Returns (mask, final_xyxy).
        """
        xyxy = [int(v) for v in xyxy]
        mask = None
        for _ in range(max_iter):
            results = self.sam2_model.predict(image, bboxes=[xyxy], verbose=False)
            mask = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
            x1, y1, x2, y2 = xyxy
            crop = mask[y1:y2, x1:x2]
            bw, bh = x2 - x1, y2 - y1
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
            if crop[:edge_thresh, :].any():    ny1 = max(0,     y1 - int(bh * expand_ratio))
            if crop[-edge_thresh:, :].any():   ny2 = min(img_h, y2 + int(bh * expand_ratio))
            if crop[:, :edge_thresh].any():    nx1 = max(0,     x1 - int(bw * expand_ratio))
            if crop[:, -edge_thresh:].any():   nx2 = min(img_w, x2 + int(bw * expand_ratio))
            if [nx1, ny1, nx2, ny2] == [x1, y1, x2, y2]:
                break  # converged -- mask does not touch any edge
            xyxy = [nx1, ny1, nx2, ny2]
        return mask, xyxy

    def refine_with_sam2(self, image, boxes):
        """Refine YOLO boxes with SAM2 masks (with iterative boundary expansion)."""
        img_h, img_w = image.shape[:2]
        masks = []
        for i, box_data in enumerate(boxes):
            try:
                mask, expanded_xyxy = self._iterative_sam2_segment(
                    image, box_data["xyxy"], img_w, img_h)
                box_data["xyxy"] = expanded_xyxy
                masks.append(mask)
            except RuntimeError as e:
                # OOM or other GPU error -- degrade gracefully
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

    def mask_to_geometry(self, mask):
        """Derive AABB and OBB geometry from a binary mask."""
        if mask is None:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) <= 0:
            return None

        x, y, w, h = cv2.boundingRect(biggest)
        rect = cv2.minAreaRect(biggest)
        box_points = cv2.boxPoints(rect)
        obb_corners = [[int(round(pt[0])), int(round(pt[1]))] for pt in box_points.tolist()]
        edge_dx = float(box_points[1][0] - box_points[0][0])
        edge_dy = float(box_points[1][1] - box_points[0][1])
        angle = float(np.degrees(np.arctan2(edge_dy, edge_dx)))

        return {
            "box_xyxy": [int(x), int(y), int(x + w), int(y + h)],
            "obb_corners": obb_corners,
            "angle": angle,
        }

    def _remove_border_touching_components(self, mask):
        """Keep only connected components that do not touch the crop border."""
        if mask is None:
            return None
        binary = (mask > 0).astype(np.uint8)
        if binary.size == 0 or np.count_nonzero(binary) == 0:
            return binary
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        kept = np.zeros_like(binary)
        best_label = None
        best_area = 0
        h, w = binary.shape[:2]
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            touches_border = x <= 0 or y <= 0 or (x + width) >= w or (y + height) >= h
            if touches_border:
                continue
            if area > best_area:
                best_label = label
                best_area = area
        if best_label is not None:
            kept[labels == best_label] = 1
        return kept

    def _normalize_segment_mask_polarity(self, crop_mask):
        """Correct masks that mostly capture background instead of the object."""
        if crop_mask is None:
            return None
        binary = (crop_mask > 0).astype(np.uint8)
        if binary.size == 0:
            return binary

        total_pixels = float(binary.shape[0] * binary.shape[1])
        if total_pixels <= 0:
            return binary

        foreground_ratio = float(np.count_nonzero(binary)) / total_pixels
        border_pixels = np.concatenate([
            binary[0, :],
            binary[-1, :],
            binary[:, 0],
            binary[:, -1],
        ])
        border_occupancy = float(np.count_nonzero(border_pixels)) / float(max(1, border_pixels.size))

        if foreground_ratio <= 0.85 and border_occupancy <= 0.50:
            return binary

        inverted = (1 - binary).astype(np.uint8)
        filtered = self._remove_border_touching_components(inverted)
        filtered_area = int(np.count_nonzero(filtered))
        original_area = int(np.count_nonzero(binary))
        if filtered_area >= 20 and filtered_area < original_area:
            return filtered
        return binary


    def save_segments_for_boxes(self, image_path, boxes, session_dir, iterative=False, expand_ratio=0.10):
        """Save SAM2 mask crops to session_dir/segments/ for each accepted box.

        Called by Electron after the user finalizes accepted boxes so that
        the OBB synthetic data generator can find the segment files.
        """
        import hashlib
        import json as _json

        if not boxes:
            return {"status": "ok", "saved": 0, "requested": 0}

        image = cv2.imread(image_path)
        if image is None:
            return {"status": "error", "error": f"Could not load image: {image_path}"}
        img_h, img_w = image.shape[:2]

        path_hash = hashlib.md5(image_path.encode()).hexdigest()[:10]

        seg_dir = os.path.join(session_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        for existing_name in list(os.listdir(seg_dir)):
            if not existing_name.startswith(f"{path_hash}_"):
                continue
            try:
                os.remove(os.path.join(seg_dir, existing_name))
            except Exception:
                pass

        # Use cached SAM2 masks only when they belong to this exact image
        cached_lookup = {}
        if (self._cached_image_path == image_path
                and self._cached_sam_results is not None):
            for box_data, mask in self._cached_sam_results:
                key = tuple(int(v) for v in box_data["xyxy"])
                cached_lookup[key] = mask

        saved = 0
        for idx, box_xyxy in enumerate(boxes):
            x1 = max(0, int(box_xyxy[0]))
            y1 = max(0, int(box_xyxy[1]))
            x2 = min(img_w, int(box_xyxy[2]))
            y2 = min(img_h, int(box_xyxy[3]))
            if x2 <= x1 or y2 <= y1:
                continue

            mask = cached_lookup.get((x1, y1, x2, y2))
            save_x1, save_y1, save_x2, save_y2 = x1, y1, x2, y2

            # Fall back to fresh SAM2 inference if no cached mask.
            if mask is None and self.sam2_model is not None:
                try:
                    if iterative:
                        mask, expanded_xyxy = self._iterative_sam2_segment(
                            image,
                            [x1, y1, x2, y2],
                            img_w,
                            img_h,
                            expand_ratio=float(expand_ratio),
                        )
                        if expanded_xyxy:
                            save_x1, save_y1, save_x2, save_y2 = [int(v) for v in expanded_xyxy]
                    else:
                        results = self.sam2_model.predict(
                            image, bboxes=[[x1, y1, x2, y2]], verbose=False)
                        mask = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
                except Exception as e:
                    logger.warning(f"SAM2 failed for box {idx}: {e}")

            # Fallback: solid rectangle mask (not accepted Ã¢â‚¬â€ poisoned background)
            if mask is not None:
                prompt_crop = mask[save_y1:save_y2, save_x1:save_x2]
                normalized_prompt_crop = self._normalize_segment_mask_polarity(prompt_crop)
                if normalized_prompt_crop is not None and prompt_crop.shape == normalized_prompt_crop.shape:
                    corrected_mask = np.zeros_like(mask, dtype=np.uint8)
                    corrected_mask[save_y1:save_y2, save_x1:save_x2] = normalized_prompt_crop
                    mask = corrected_mask
            geometry = self.mask_to_geometry(mask) if mask is not None else None
            if geometry is not None:
                save_x1 = max(0, int(geometry["box_xyxy"][0]))
                save_y1 = max(0, int(geometry["box_xyxy"][1]))
                save_x2 = min(img_w, int(geometry["box_xyxy"][2]))
                save_y2 = min(img_h, int(geometry["box_xyxy"][3]))
                mask_source = "sam2_iterative" if iterative else "sam2"
            else:
                mask = None

            if mask is None:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
                mask_source = "rectangle_fallback"
                save_x1, save_y1, save_x2, save_y2 = x1, y1, x2, y2

            crop_img  = image[save_y1:save_y2, save_x1:save_x2]
            crop_mask = mask[save_y1:save_y2, save_x1:save_x2]
            if crop_img.size == 0:
                continue

            if crop_mask.shape != crop_img.shape[:2]:
                crop_mask = cv2.resize(
                    crop_mask, (crop_img.shape[1], crop_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST)

            crop_mask = self._normalize_segment_mask_polarity(crop_mask)
            if crop_mask is None or np.count_nonzero(crop_mask) < 20:
                continue

            alpha = (crop_mask * 255).astype(np.uint8)
            bgra = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha

            base = f"{path_hash}_{idx}"
            cv2.imwrite(os.path.join(seg_dir, f"{base}_fg.png"), bgra)
            cv2.imwrite(os.path.join(seg_dir, f"{base}_mask.png"), alpha)

            meta = {
                "accepted_by_user": mask_source.startswith("sam2"),
                "mask_source": mask_source,
                "source_image": image_path,
                "box": {
                    "left": save_x1, "top": save_y1, "right": save_x2, "bottom": save_y2,
                    "width": save_x2 - save_x1, "height": save_y2 - save_y1,
                },
                "crop_origin": [save_x1, save_y1],
            }
            with open(os.path.join(seg_dir, f"{base}_meta.json"),
                      "w", encoding="utf-8") as f:
                _json.dump(meta, f)

            saved += 1

        logger.info(
            f"save_segments_for_boxes: saved {saved}/{len(boxes)} segments Ã¢â€ â€™ {seg_dir}")
        return {"status": "ok", "saved": saved, "requested": len(boxes)}

    # ------------------------------------------------------------------
    # Stage B: Normalization (The "Standardizer")
    # ------------------------------------------------------------------

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
        """Run dlib on a STANDARD_SIZE Ãƒâ€” STANDARD_SIZE image."""
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
    # Fine-tuned YOLOv8 detection (OBB-aware)
    # ------------------------------------------------------------------
    def detect_finetuned(self, image, finetuned_path, class_name, conf_threshold=0.5, top_k=10, nms_iou=0.3):
        """Run detection with a fine-tuned YOLOv8 OBB model."""
        from ultralytics import YOLO
        ft_model = YOLO(finetuned_path)
        results = ft_model.predict(
            image,
            conf=conf_threshold,
            iou=float(nms_iou),
            imgsz=640,
            task="obb",
            verbose=False,
        )

        boxes = []
        r = results[0]
        if r.obb is not None and len(r.obb):
            # OBB model Ã¢â‚¬â€ extract corners in canonical order, class_id, and AABB envelope.
            # buildObbCorners order: cp0=LT(-hw,-hh), cp1=RT(+hw,-hh),
            #                        cp2=RB(+hw,+hh),  cp3=LB(-hw,+hh)
            import math as _math
            for i in range(len(r.obb)):
                xywhr = r.obb.xywhr[i].cpu().numpy().tolist()  # [cx, cy, w, h, angle_rad]
                cx, cy, w, h, angle_rad = xywhr
                cos_a, sin_a = _math.cos(angle_rad), _math.sin(angle_rad)
                hw, hh = w / 2, h / 2
                corners = [
                    [cx + cos_a*(-hw) - sin_a*(-hh), cy + sin_a*(-hw) + cos_a*(-hh)],  # cp0 LT
                    [cx + cos_a*(+hw) - sin_a*(-hh), cy + sin_a*(+hw) + cos_a*(-hh)],  # cp1 RT
                    [cx + cos_a*(+hw) - sin_a*(+hh), cy + sin_a*(+hw) + cos_a*(+hh)],  # cp2 RB
                    [cx + cos_a*(-hw) - sin_a*(+hh), cy + sin_a*(-hw) + cos_a*(+hh)],  # cp3 LB
                ]
                conf = float(r.obb.conf[i])
                class_id = int(r.obb.cls[i])
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                xyxy = [min(xs), min(ys), max(xs), max(ys)]  # AABB envelope
                angle_deg = angle_rad * 180.0 / _math.pi
                boxes.append({
                    "xyxy": xyxy,
                    "confidence": round(conf, 3),
                    "class_name": class_name,
                    "obb_corners": corners,   # 4Ãƒâ€”[x,y] in canonical order (cp0Ã¢â‚¬â€œcp3)
                    "class_id": class_id,     # 0 = left (canonical), 1 = right
                    "angle": angle_deg,       # rotation angle in degrees
                })
        else:
            raise RuntimeError(f"OBB detector returned no oriented boxes: {finetuned_path}")

        # Class-agnostic dedup: remove lower-confidence boxes that overlap an
        # already-kept box (IoU > 0.5 on AABB envelope). Fixes YOLO OBB class-aware
        # NMS emitting both class 0 (left) and class 1 (right) for the same fish.
        deduped: list[dict] = []
        for box in sorted(boxes, key=lambda b: b["confidence"], reverse=True):
            if not any(_aabb_iou(box["xyxy"], k["xyxy"]) > 0.5 for k in deduped):
                deduped.append(box)
        boxes = deduped

        boxes.sort(key=lambda b: b["confidence"], reverse=True)
        if top_k > 0 and len(boxes) > top_k:
            boxes = boxes[:top_k]
        return boxes

    def detect_zero_shot(self, image, class_name, conf_threshold=0.25, top_k=10, nms_iou=0.65, imgsz=1280):
        """Run zero-shot YOLO-World detection and wrap boxes as axis-aligned OBBs."""
        if self.yolo_model is None:
            raise RuntimeError("YOLO-World is not available for zero-shot detection.")

        prompts = self._build_class_prompts(class_name)
        self._set_yolo_classes(prompts)
        results = self.yolo_model.predict(
            image,
            conf=float(conf_threshold),
            iou=float(nms_iou),
            imgsz=int(imgsz),
            verbose=False,
        )

        boxes = []
        r = results[0]
        raw_boxes = getattr(r, "boxes", None)
        if raw_boxes is None or len(raw_boxes) == 0:
            return []

        for i in range(len(raw_boxes)):
            xyxy = raw_boxes.xyxy[i].cpu().numpy().tolist()
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            conf = float(raw_boxes.conf[i]) if raw_boxes.conf is not None else 0.0
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            corners = [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ]
            boxes.append({
                "xyxy": [left, top, right, bottom],
                "confidence": round(conf, 3),
                "class_name": class_name,
                "obb_corners": corners,
                "class_id": 0,
                "angle": 0.0,
            })

        deduped: list[dict] = []
        for box in sorted(boxes, key=lambda b: b["confidence"], reverse=True):
            if not any(_aabb_iou(box["xyxy"], k["xyxy"]) > 0.5 for k in deduped):
                deduped.append(box)
        boxes = deduped

        boxes.sort(key=lambda b: b["confidence"], reverse=True)
        if top_k > 0 and len(boxes) > top_k:
            boxes = boxes[:top_k]
        return boxes

    # ------------------------------------------------------------------
    # Full pipeline: annotate
    # ------------------------------------------------------------------
    def annotate(self, image_path, class_name, dlib_model=None, id_mapping_path=None, options=None):
        """Run the OBB-only detection and landmark pipeline on one image."""
        from bv_utils.orientation_utils import (
            apply_obb_geometry,
            extract_standardized_obb_crop,
            map_to_original as ou_map_to_original,
        )

        options = options or {}
        conf_threshold = options.get("conf_threshold", 0.3)
        sam_enabled = options.get("sam_enabled", False)
        max_objects = options.get("max_objects", 10)
        finetuned_model = options.get("finetuned_model")
        orientation_policy = options.get("orientation_policy") or {}
        orientation_schema = str(orientation_policy.get("mode", "invariant")).strip().lower()
        detection_preset = options.get("detection_preset", "balanced")
        use_obb_detector = bool(finetuned_model and os.path.exists(finetuned_model))
        resolved = self._resolve_detection_preset(
            conf_threshold=conf_threshold,
            nms_iou=0.3,
            max_objects=max_objects,
            detection_preset=detection_preset,
            task="obb" if use_obb_detector else "generic",
        )
        conf_threshold = resolved["conf"]
        max_objects = resolved["top_k"]

        # Load image
        send_progress("Loading image...", 5, "detection")
        image = self._load_image(image_path)
        img_h, img_w = image.shape[:2]

        send_progress("Detecting objects...", 15, "detection")
        if use_obb_detector:
            import json as _json_inf
            obb_nms_iou = 0.3
            obb_cfg = os.path.join(os.path.dirname(os.path.dirname(finetuned_model)), "obb_config.json")
            if os.path.exists(obb_cfg):
                try:
                    with open(obb_cfg, "r", encoding="utf-8") as handle:
                        obb_nms_iou = float(_json_inf.load(handle).get("nms_iou", 0.3))
                except Exception:
                    pass

            boxes = self.detect_finetuned(
                image,
                finetuned_model,
                class_name,
                conf_threshold,
                top_k=max_objects,
                nms_iou=obb_nms_iou,
            )
            detection_method = "yolo_obb"
        else:
            boxes = self.detect_zero_shot(
                image,
                class_name,
                conf_threshold=conf_threshold,
                top_k=max_objects,
                nms_iou=resolved["iou"],
                imgsz=resolved["imgsz"],
            )
            detection_method = "yolo_world"

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

        masks = [None] * len(boxes)
        if sam_enabled and self.sam2_model is not None:
            send_progress("Refining with SAM2...", 35, "segmentation")
            masks = self.refine_with_sam2(image, boxes)
            if any(mask is not None for mask in masks):
                detection_method += "+sam2"

        self._cached_sam_results = list(zip(boxes, masks))

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
            obb_corners = box_data.get("obb_corners")
            if not obb_corners:
                raise RuntimeError("OBB detector returned a box without obb_corners.")
            class_id = int(box_data.get("class_id", 0))

            # --- Neighbor Ghosting ---
            # Paint every other detected object pure black on a scratch copy so that
            # adjacent specimens cannot contaminate this object's deskewed crop,
            # regardless of padding size or OBB rotation angle.
            # Single-object images skip the copy entirely (no-op fast path).
            if len(boxes) > 1:
                scene_image = image.copy()
                for j, other in enumerate(boxes):
                    if j == i:
                        continue
                    ghost_pts = np.array(other["obb_corners"], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(scene_image, [ghost_pts], (0, 0, 0))
            else:
                scene_image = image
            # --- End Neighbor Ghosting ---

            if obb_corners:
                # OBB path: deskew using the detector's angle, then flip to canonical
                # orientation for dlib landmark prediction.
                # invariant: no leveling (spatial anchor Ã¢â‚¬â€ center + scale only)
                # axial: deskew but no flip (ends are biologically interchangeable)
                # directional/bilateral: deskew + flip right-facing to canonical left
                apply_leveling = (orientation_policy.get("obbLevelingMode", "on") == "on")
                standardized, metadata = extract_standardized_obb_crop(
                    scene_image,
                    obb_corners,
                    apply_leveling=apply_leveling,
                )
                standardized, metadata, _canonicalization_debug = apply_obb_geometry(
                    standardized,
                    metadata,
                    class_id,
                    orientation_policy,
                )
                metadata["was_flipped"] = False
                # invariant: leveling skipped Ã¢â€ â€™ zero rotation so map_to_original won't un-rotate
                if orientation_schema == "invariant":
                    metadata = {**metadata, "rotation": 0.0}
                metadata["was_flipped"] = False
            else:
                raise RuntimeError("OBB detector returned a box without obb_corners.")

            # Predict landmarks
            landmarks = []
            if has_dlib:
                landmarks_512 = self.predict_landmarks(standardized)
                landmarks = ou_map_to_original(
                    landmarks_512,
                    metadata,
                    was_flipped=False,
                    image_shape=(img_h, img_w),
                )

            # Mask outline
            outline = self.mask_to_outline(mask)

            xs = [float(p[0]) for p in obb_corners]
            ys = [float(p[1]) for p in obb_corners]
            obb_info = {
                "corners": [[float(x), float(y)] for x, y in obb_corners],
                "angle": float(box_data.get("angle", 0.0)),
                "center": [float(sum(xs) / 4.0), float(sum(ys) / 4.0)],
                "size": [float(max(xs) - min(xs)), float(max(ys) - min(ys))],
            }
            orientation_hint = None
            if class_id is not None and orientation_schema in ("directional", "bilateral"):
                orientation_hint = {
                    "orientation": "left" if class_id == 0 else "right",
                    "confidence": float(box_data.get("confidence", 0.0)),
                    "source": "obb_class_id",
                }

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
                "obb": obb_info,  # OBB from SAM2 mask; None when no mask
                "obbCorners": [[int(x), int(y)] for x, y in obb_corners] if obb_corners else None,
                "class_id": class_id,
                "orientation_hint": orientation_hint,
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
    # Segment pool maintenance
    # ------------------------------------------------------------------
    def _refresh_segments(self, session_dir, sam2_enabled=True):
        """Prune stale segments; populate missing segments for finalized images.

        Step 1 Ã¢â‚¬â€ Prune: delete segment triplets (_fg.png, _mask.png, _meta.json)
        whose source_image is not in the current finalized set.

        Step 2 Ã¢â‚¬â€ Populate: for each finalized image that has no segment entry yet,
        run SAM2 to generate segments (only when sam2_enabled=True).
        """
        import json as _json
        import hashlib as _hashlib
        from data.export_yolo_dataset import _load_finalized_filenames

        finalized_set = _load_finalized_filenames(session_dir)   # set of lowercase filenames
        seg_dir = os.path.join(session_dir, "segments")
        labels_dir = os.path.join(session_dir, "labels")
        images_dir = os.path.join(session_dir, "images")

        # Step 1: prune stale segments (source_image not in finalized set)
        if os.path.isdir(seg_dir):
            for fname in sorted(os.listdir(seg_dir)):
                if not fname.endswith("_meta.json"):
                    continue
                try:
                    with open(os.path.join(seg_dir, fname), "r", encoding="utf-8") as f:
                        meta = _json.load(f)
                except Exception:
                    continue
                src_base = os.path.basename(meta.get("source_image", "")).lower()
                if src_base not in finalized_set:
                    base = fname[:-10]  # strip "_meta.json"
                    for suffix in ("_meta.json", "_fg.png", "_mask.png"):
                        p = os.path.join(seg_dir, base + suffix)
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass

        # Step 2: populate missing segments for finalized images (SAM2 only)
        if not sam2_enabled or not os.path.isdir(labels_dir):
            return
        os.makedirs(seg_dir, exist_ok=True)
        for fname in sorted(os.listdir(labels_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(labels_dir, fname), "r", encoding="utf-8") as f:
                    data = _json.load(f)
            except Exception:
                continue
            image_filename = data.get("imageFilename", "")
            if not image_filename or image_filename.lower() not in finalized_set:
                continue
            image_path = os.path.join(images_dir, image_filename)
            if not os.path.exists(image_path):
                continue
            path_hash = _hashlib.md5(image_path.encode()).hexdigest()[:10]
            # If any segment already exists for this image, leave it untouched
            # No segment for this image Ã¢â‚¬â€ run SAM2
            boxes_raw = data.get("boxes", [])
            if not boxes_raw:
                continue
            box_list = [
                [b["left"], b["top"],
                 b["left"] + b["width"], b["top"] + b["height"]]
                for b in boxes_raw
                if b.get("width", 0) > 0 and b.get("height", 0) > 0
            ]
            if box_list:
                existing_segment_count = sum(
                    1 for f in os.listdir(seg_dir)
                    if f.startswith(path_hash) and f.endswith("_fg.png")
                )
                if existing_segment_count == len(box_list):
                    continue
                if existing_segment_count > 0:
                    for existing_name in list(os.listdir(seg_dir)):
                        if not existing_name.startswith(path_hash):
                            continue
                        try:
                            os.remove(os.path.join(seg_dir, existing_name))
                        except Exception:
                            pass
                try:
                    self.save_segments_for_boxes(
                        image_path,
                        box_list,
                        session_dir,
                        iterative=bool(sam2_enabled),
                        expand_ratio=0.10,
                    )
                except Exception as e:
                    logger.warning(
                        f"_refresh_segments: SAM2 failed for {image_filename}: {e}")

    # ------------------------------------------------------------------
    # OBB dataset export
    # ------------------------------------------------------------------
    def export_obb_dataset(self, session_dir, generate_synthetic=True, orientation_schema="invariant"):
        """
        Export OBB-format YOLO dataset from session annotations.
        All exported boxes must already carry valid OBB corners.

        Args:
            generate_synthetic: Pass False when SAM2 is unavailable (CPU-only) to
                skip synthetic rotational augmentation and avoid edge-artifact poisoning.
            orientation_schema: One of "directional", "bilateral", "axial", "invariant".
                Vector schemas (directional/bilateral) export 2-class OBB; others export 1-class.
        """
        import importlib, sys
        if "data.export_yolo_dataset" in sys.modules:
            importlib.reload(sys.modules["data.export_yolo_dataset"])
        from data.export_yolo_dataset import export_obb_dataset as _export_obb
        result = _export_obb(session_dir, generate_synthetic=generate_synthetic,
                             orientation_schema=orientation_schema)
        return result

    # ------------------------------------------------------------------
    # OBB detector training
    # ------------------------------------------------------------------
    def train_yolo_obb(self, session_dir, epochs=None, model_tier="nano",
                       device="cpu", sam2_enabled=True,
                       iou_loss=0.3, cls_loss=1.5, box_loss=5.0,
                       orientation_schema="invariant"):
        """
        Train a YOLOv8-OBB detector on the session's OBB dataset.
        Unloads YOLO-World and SAM2 first to free memory.

        Args:
            device: Compute device ('cpu', 'mps', 'cuda'). Controls batch size
                and epoch defaults for thermal safety and performance.
            sam2_enabled: When False, synthetic augmentation is skipped to
                prevent edge-artifact poisoning on CPU-only systems.
        """
        import gc
        from bv_utils.orientation_utils import resolve_session_augmentation_profile

        # Hardware-routed hyperparameters
        if device in ("cuda", "mps"):
            default_epochs = 100
            default_batch = 16
        else:  # cpu
            default_epochs = 30
            # Capped between 4-8 Ã¢â‚¬â€ YOLO batch=-1 autotune can be dangerous on CPU
            default_batch = 6

        resolved_epochs = epochs if (epochs is not None and epochs != 50) else default_epochs
        resolved_batch = default_batch  # Always explicit; avoid YOLO autotune on CPU

        logger.info(
            "OBB training: device=%s, epochs=%d, batch=%d, sam2=%s",
            device, resolved_epochs, resolved_batch, sam2_enabled,
        )

        # Refresh segment pool while SAM2 is still loaded: prune stale entries and
        # populate missing ones via SAM2 before releasing GPU memory for training.
        send_progress("Refreshing segment pool...", 3, "training")
        self._refresh_segments(session_dir, sam2_enabled=sam2_enabled)

        # Unload large models before training to reclaim memory
        if self.yolo_model is not None:
            self.yolo_model = None
            logger.info("Unloaded YOLO-World before OBB training")
        if self.sam2_model is not None:
            self.sam2_model = None
            logger.info("Unloaded SAM2 before OBB training")
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Export OBB dataset Ã¢â‚¬â€ pass sam2_enabled to control synthetic generation
        send_progress("Exporting OBB dataset...", 5, "training")
        export_result = self.export_obb_dataset(session_dir, generate_synthetic=sam2_enabled,
                                                orientation_schema=orientation_schema)
        if not export_result.get("ok"):
            return {"status": "error", "error": export_result.get("error", "OBB dataset export failed")}

        resolved_mode, _orientation_policy, _augmentation_policy, aug_profile = (
            resolve_session_augmentation_profile(
                session_dir,
                engine="cnn",
                fallback_mode=orientation_schema,
            )
        )
        raw_rotation_range = aug_profile.get("rotation_range", (-15.0, 15.0))
        if isinstance(raw_rotation_range, list):
            raw_rotation_range = tuple(raw_rotation_range)
        if not isinstance(raw_rotation_range, tuple) or len(raw_rotation_range) != 2:
            raw_rotation_range = (-15.0, 15.0)
        rotation_lo = float(raw_rotation_range[0])
        rotation_hi = float(raw_rotation_range[1])
        resolved_degrees = max(abs(rotation_lo), abs(rotation_hi))
        logger.info(
            "OBB training augmentation: mode=%s rotation_range=(%.1f, %.1f) degrees=%.1f",
            resolved_mode,
            rotation_lo,
            rotation_hi,
            resolved_degrees,
        )

        dataset_yaml = export_result["yaml_path"]
        models_dir = os.path.join(session_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        base_map = {
            "nano": "yolov8n-obb.pt",
            "small": "yolov8s-obb.pt",
            "medium": "yolov8m-obb.pt",
        }
        base_model = base_map.get(model_tier, "yolov8n-obb.pt")
        freeze_layers = 14 if model_tier == "nano" else 0

        send_progress(f"Starting YOLOv8-OBB training ({model_tier}, {device})...", 10, "training")
        from ultralytics import YOLO
        model = YOLO(base_model)

        def on_train_epoch_end(trainer):
            epoch = trainer.epoch + 1
            total = trainer.epochs
            pct = 10 + int(80 * (epoch / total))
            send_progress(f"OBB training epoch {epoch}/{total}...", pct, "training")

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        output_dir = os.path.join(session_dir, "models", "obb_training")
        model.train(
            data=dataset_yaml,
            epochs=resolved_epochs,
            imgsz=640,
            batch=resolved_batch,
            device=device,
            freeze=freeze_layers,
            project=output_dir,
            name="session_obb",
            exist_ok=True,
            verbose=False,
            task="obb",
            fliplr=0.0,         # horizontal flip corrupts orientation labels (left vs right class)
            flipud=0.0,         # biological specimens should not train on upside-down flips
            degrees=resolved_degrees,
            iou=float(iou_loss),   # NMS IoU threshold for validation during training
            cls=float(cls_loss),   # classification loss gain
            box=float(box_loss),   # box regression loss gain
            patience=20 if device in ("cuda", "mps") else 10,  # stop if no mAP50-95 gain for N epochs
            cos_lr=True,        # cosine LR decay avoids late-epoch plateau oscillation
            mosaic=0.0,         # disable mosaic entirely: pretrained fine-tune on small bio data
            close_mosaic=0,
        )

        # Save NMS IoU preference alongside the run so detect_finetuned() can restore it.
        import json as _json_obb
        obb_config_path = os.path.join(output_dir, "session_obb", "obb_config.json")
        try:
            with open(obb_config_path, "w", encoding="utf-8") as _f:
                _json_obb.dump({"nms_iou": float(iou_loss)}, _f)
        except Exception:
            pass

        best_pt = os.path.join(output_dir, "session_obb", "weights", "best.pt")
        if not os.path.exists(best_pt):
            best_pt = os.path.join(output_dir, "session_obb", "weights", "last.pt")
        if not os.path.exists(best_pt):
            return {"status": "error", "error": "OBB training finished but no best.pt found"}

        dest = os.path.join(models_dir, "session_obb_detector.pt")
        shutil.copy2(best_pt, dest)

        send_progress("OBB detector training complete", 100, "done")
        return {
            "status": "result",
            "ok": True,
            "model_path": dest,
            "warnings": export_result.get("warnings", []),
        }

    # ------------------------------------------------------------------
    # OBB inference
    # ------------------------------------------------------------------
    def detect_obb(self, image_path, model_path, conf=0.3, nms_iou=None,
                   detection_preset="balanced", max_objects=20):
        """
        Run the trained session OBB detector on an image.
        Returns list of detections: [{corners, angle, class_id, confidence}]
        """
        import json as _json_obb
        from ultralytics import YOLO

        # Load the NMS IoU that was saved when this model was trained.
        # The sidecar lives at {models_dir}/obb_training/session_obb/obb_config.json.
        if nms_iou is None:
            _cfg = os.path.join(os.path.dirname(model_path),
                                "obb_training", "session_obb", "obb_config.json")
            nms_iou = 0.3   # biological default
            if os.path.exists(_cfg):
                try:
                    with open(_cfg, "r", encoding="utf-8") as _f:
                        nms_iou = float(_json_obb.load(_f).get("nms_iou", 0.3))
                except Exception:
                    pass

        resolved = self._resolve_detection_preset(
            conf_threshold=conf,
            nms_iou=nms_iou,
            max_objects=max_objects,
            detection_preset=detection_preset,
            task="obb",
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
        import math as _math
        detections = []
        for r in results:
            if r.obb is None:
                continue
            for i in range(len(r.obb)):
                try:
                    xywhr = r.obb.xywhr[i].cpu().numpy().tolist()  # [cx, cy, w, h, angle_rad]
                    cx, cy, w, h, angle_rad = xywhr
                    cos_a, sin_a = _math.cos(angle_rad), _math.sin(angle_rad)
                    hw, hh = w / 2, h / 2
                    corners = [
                        [cx + cos_a*(-hw) - sin_a*(-hh), cy + sin_a*(-hw) + cos_a*(-hh)],  # cp0 LT
                        [cx + cos_a*(+hw) - sin_a*(-hh), cy + sin_a*(+hw) + cos_a*(-hh)],  # cp1 RT
                        [cx + cos_a*(+hw) - sin_a*(+hh), cy + sin_a*(+hw) + cos_a*(+hh)],  # cp2 RB
                        [cx + cos_a*(-hw) - sin_a*(+hh), cy + sin_a*(-hw) + cos_a*(+hh)],  # cp3 LB
                    ]
                    class_id = int(r.obb.cls[i])
                    confidence = float(r.obb.conf[i])
                    angle_deg = angle_rad * 180.0 / _math.pi
                    detections.append({
                        "corners": corners,
                        "angle": angle_deg,
                        "class_id": class_id,
                        "confidence": confidence,
                    })
                except Exception as e:
                    logger.warning(f"OBB detection parse error at index {i}: {e}")
        deduped = _class_agnostic_dedup(detections)
        deduped.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        return deduped[:max(1, int(resolved["top_k"]))]

    # ------------------------------------------------------------------
    # OBB class_id tagging from placed landmarks
    # ------------------------------------------------------------------
    def tag_class_ids(self, session_dir, boxes, orientation_policy=None):
        """
        Compute class_id for each box from placed head/tail landmark coordinates.

        class_id encoding:
          directional: 0=left-facing (canonical), 1=right-facing
          bilateral:   0=canonical, 1=flipped (same X-based logic as directional)
          axial:        0=up-facing (canonical), 1=down-facing (triggers 180Ã‚Â° spin)
          invariant:   always 0

        Returns list of {"id": ..., "class_id": 0|1}.
        """
        from data.export_yolo_dataset import _load_head_tail_ids, _compute_box_orientation

        mode = str((orientation_policy or {}).get("mode", "invariant")).strip().lower()

        if mode not in ("directional", "bilateral", "axial"):
            return [{"id": b.get("id"), "class_id": 0} for b in boxes]

        head_id, tail_id = _load_head_tail_ids(session_dir)
        if head_id is None or tail_id is None:
            return [{"id": b.get("id"), "class_id": 0} for b in boxes]

        result = []
        for b in boxes:
            if mode in ("directional", "bilateral"):
                orientation = _compute_box_orientation(b, head_id, tail_id)
                class_id = 1 if orientation == "right" else 0
            else:
                # Axial: determine up/down from head vs tail Y coordinate
                landmarks = [
                    lm for lm in b.get("landmarks", [])
                    if not lm.get("isSkipped")
                    and lm.get("x", -1) >= 0
                    and lm.get("y", -1) >= 0
                ]
                head_lm = next(
                    (lm for lm in landmarks if int(lm.get("id", -1)) == head_id), None
                )
                tail_lm = next(
                    (lm for lm in landmarks if int(lm.get("id", -1)) == tail_id), None
                )
                if head_lm and tail_lm:
                    # head Y > tail Y means head is below tail Ã¢â€ â€™ down-facing = class_id=1
                    class_id = 1 if float(head_lm["y"]) > float(tail_lm["y"]) else 0
                else:
                    class_id = 0
            result.append({"id": b.get("id"), "class_id": class_id})
        return result

    # ------------------------------------------------------------------
    # SAM2 direct box segmentation (no cached state required)
    # ------------------------------------------------------------------
    def resegment_box(self, image_path, box_xyxy, iterative=False, expand_ratio=0.10):
        """Run SAM2 on a single bounding box, independent of any annotation cache."""
        if self.sam2_model is None:
            return {"status": "error", "error": "SAM2 not loaded"}

        image = self._load_image(image_path)
        try:
            if iterative:
                img_h, img_w = image.shape[:2]
                mask, _expanded_xyxy = self._iterative_sam2_segment(
                    image,
                    box_xyxy,
                    img_w,
                    img_h,
                    expand_ratio=float(expand_ratio),
                )
                if mask is None:
                    return {"status": "error", "error": "SAM2 returned no mask for this box"}
                score = 1.0
            else:
                results = self.sam2_model.predict(image, bboxes=[box_xyxy], verbose=False)
                masks_data = results[0].masks
                if masks_data is None or len(masks_data.data) == 0:
                    return {"status": "error", "error": "SAM2 returned no mask for this box"}
                mask = (masks_data.data[0].cpu().numpy() > 0.5).astype(np.uint8)
                # SAM2 stores per-mask IoU quality in boxes.conf when prompting with bboxes
                try:
                    score = float(results[0].boxes.conf[0])
                except Exception:
                    score = 1.0

            outline = self.mask_to_outline(mask)
            if not outline:
                return {"status": "error", "error": "SAM2 mask produced an empty outline"}
            geometry = self.mask_to_geometry(mask)
            if geometry is None:
                return {"status": "error", "error": "SAM2 mask produced invalid geometry"}
            return {
                "status": "result",
                "ok": True,
                "mask_outline": outline,
                "box_xyxy": geometry["box_xyxy"],
                "obb_corners": geometry["obb_corners"],
                "angle": geometry["angle"],
                "score": score,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

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
            mask = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
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

        global _current_request_id
        _current_request_id = cmd.get("_request_id")

        try:
            command = cmd.get("cmd", "")

            if command == "init":
                result = annotator.init_models()
                send_response(result)

            elif command == "check":
                result = annotator.check()
                send_response(result)

            elif command == "annotate":
                result = annotator.annotate(
                    image_path=cmd["image_path"],
                    class_name=cmd.get("class_name", "object"),
                    dlib_model=cmd.get("dlib_model"),
                    id_mapping_path=cmd.get("id_mapping_path"),
                    options=cmd.get("options"),
                )
                send_response(result)

            elif command == "refine_sam":
                result = annotator.refine_sam(
                    image_path=cmd["image_path"],
                    object_index=cmd.get("object_index", 0),
                    click_point=cmd["click_point"],
                    click_label=cmd.get("click_label", 1),
                )
                send_response(result)

            elif command == "resegment_box":
                result = annotator.resegment_box(
                    image_path=cmd["image_path"],
                    box_xyxy=cmd["box_xyxy"],
                    iterative=cmd.get("iterative", False),
                    expand_ratio=cmd.get("expand_ratio", 0.10),
                )
                send_response(result)

            elif command == "export_obb_dataset":
                result = annotator.export_obb_dataset(
                    cmd["session_dir"],
                    orientation_schema=cmd.get("orientation_schema", "invariant"),
                )
                send_response({"status": "result", **result})

            elif command == "train_yolo_obb":
                result = annotator.train_yolo_obb(
                    session_dir=cmd["session_dir"],
                    epochs=cmd.get("epochs"),          # None Ã¢â€ â€™ hardware default
                    model_tier=cmd.get("model_tier", "nano"),
                    device=cmd.get("device", "cpu"),
                    sam2_enabled=cmd.get("sam2_enabled", True),
                    iou_loss=cmd.get("iou_loss", 0.3),
                    cls_loss=cmd.get("cls_loss", 1.5),
                    box_loss=cmd.get("box_loss", 5.0),
                    orientation_schema=cmd.get("orientation_schema", "invariant"),
                )
                send_response(result)

            elif command == "detect_obb":
                detections = annotator.detect_obb(
                    image_path=cmd["image_path"],
                    model_path=cmd["model_path"],
                    conf=cmd.get("conf", 0.3),
                    nms_iou=cmd.get("nms_iou"),   # None Ã¢â€ â€™ auto-load from sidecar
                    detection_preset=cmd.get("detection_preset", "balanced"),
                    max_objects=cmd.get("max_objects", 20),
                )
                send_response({"status": "result", "detections": detections})

            elif command == "tag_class_ids":
                tagged = annotator.tag_class_ids(
                    session_dir=cmd["session_dir"],
                    boxes=cmd.get("boxes", []),
                    orientation_policy=cmd.get("orientation_policy"),
                )
                send_response({"status": "result", "tagged_boxes": tagged})

            elif command == "save_segments_for_boxes":
                result = annotator.save_segments_for_boxes(
                    image_path=cmd["image_path"],
                    boxes=cmd.get("boxes", []),
                    session_dir=cmd["session_dir"],
                    iterative=cmd.get("iterative", False),
                    expand_ratio=cmd.get("expand_ratio", 0.10),
                )
                send_response(result)

            elif command == "shutdown":
                send_response({"status": "ok", "message": "Shutting down"})
                logger.info("Shutdown requested, exiting")
                break

            else:
                send_response({"status": "error", "error": f"Unknown command: {command}"})

        except Exception as e:
            logger.error(f"Error processing command: {traceback.format_exc()}")
            send_response({"status": "error", "error": str(e)})
        finally:
            _current_request_id = None

    sys.exit(0)


if __name__ == "__main__":
    main()
