#!/usr/bin/env python3
"""
Audit YOLO-pose readiness for a BioVision session.

This script mirrors export_yolo_dataset.py logic and explains why pose mode is
or is not enabled. It focuses on finalized accepted boxes because those are the
only boxes used for YOLO fine-tuning.

Usage:
  python audit_pose_readiness.py <session_dir> [--class-name fish] [--max-failures 200]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from export_yolo_dataset import (
    _compute_box_keypoints,
    _get_finalized_boxes,
    _load_finalized_filenames,
    _load_head_tail_ids,
    _normalize_box,
)
from image_utils import safe_imread
import debug_io as dio


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_class_name(value: str) -> str:
    text = (value or "").strip().lower().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text) or "object"


def _resolve_image_path(images_dir: str, image_filename: str) -> str | None:
    direct = os.path.join(images_dir, image_filename)
    if os.path.exists(direct):
        return direct
    base = os.path.splitext(image_filename)[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        candidate = os.path.join(images_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _valid_landmarks(box: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for lm in box.get("landmarks", []) or []:
        if lm.get("isSkipped"):
            continue
        try:
            x = float(lm.get("x", -1))
            y = float(lm.get("y", -1))
            lm_id = int(lm.get("id"))
        except Exception:
            continue
        if x < 0 or y < 0:
            continue
        out.append({"id": lm_id, "x": x, "y": y})
    return out


def _box_signature(box: dict[str, Any]) -> tuple[int, int, int, int]:
    nb = _normalize_box(box)
    if not nb:
        return (0, 0, 0, 0)
    return (nb["left"], nb["top"], nb["width"], nb["height"])


def _find_matching_draft_box(accepted_box: dict[str, Any], draft_boxes: list[dict[str, Any]]) -> dict[str, Any] | None:
    sig = _box_signature(accepted_box)
    for b in draft_boxes:
        if _box_signature(b) == sig:
            return b
    return None


def audit_pose_readiness(
    session_dir: str,
    class_name: str = "object",
    max_failures: int = 200,
) -> dict[str, Any]:
    session_dir = os.path.abspath(session_dir)
    labels_dir = os.path.join(session_dir, "labels")
    images_dir = os.path.join(session_dir, "images")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    head_id, tail_id = _load_head_tail_ids(session_dir)
    finalized_set = _load_finalized_filenames(session_dir)

    counters = Counter()
    failure_reasons = Counter()
    per_image_failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_examples: list[dict[str, Any]] = []

    image_size_cache: dict[str, tuple[int, int] | None] = {}

    for fname in sorted(os.listdir(labels_dir)):
        if not fname.endswith(".json"):
            continue
        counters["label_files"] += 1
        label_path = os.path.join(labels_dir, fname)
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            counters["invalid_label_json"] += 1
            failure_reasons["invalid_label_json"] += 1
            continue

        image_filename = data.get("imageFilename", "")
        if not image_filename:
            counters["missing_image_filename"] += 1
            failure_reasons["missing_image_filename"] += 1
            continue

        is_finalized, accepted_boxes, used_fallback = _get_finalized_boxes(data, image_filename, finalized_set)
        if not is_finalized:
            counters["skipped_unfinalized_images"] += 1
            continue

        counters["finalized_images"] += 1
        if used_fallback:
            counters["finalized_fallback_to_boxes"] += 1

        image_path = _resolve_image_path(images_dir, image_filename)
        if image_path not in image_size_cache:
            if image_path and os.path.exists(image_path):
                img = safe_imread(image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    image_size_cache[image_path] = (w, h)
                else:
                    image_size_cache[image_path] = None
            else:
                image_size_cache[image_path] = None

        img_size = image_size_cache.get(image_path)
        if img_size is None:
            failure_reasons["missing_or_unreadable_image"] += 1
            counters["images_missing_for_audit"] += 1
            continue
        img_w, img_h = img_size

        draft_boxes = data.get("boxes", []) if isinstance(data.get("boxes"), list) else []

        for box_idx, box in enumerate(accepted_boxes):
            counters["finalized_boxes_total"] += 1
            valid = _valid_landmarks(box)
            lm_ids = {int(lm["id"]) for lm in valid}
            has_head = head_id is not None and head_id in lm_ids
            has_tail = tail_id is not None and tail_id in lm_ids
            kp = _compute_box_keypoints(box, img_w, img_h, head_id, tail_id)
            if kp is not None:
                counters["pose_ready_boxes"] += 1
                continue

            # Diagnose failure reasons.
            reason = None
            hint = None
            if not box.get("landmarks"):
                reason = "accepted_box_has_no_landmarks"
                matched = _find_matching_draft_box(box, draft_boxes)
                if matched:
                    matched_valid = _valid_landmarks(matched)
                    matched_ids = {int(lm["id"]) for lm in matched_valid}
                    matched_head = head_id is not None and head_id in matched_ids
                    matched_tail = tail_id is not None and tail_id in matched_ids
                    if matched_head and matched_tail:
                        hint = "accepted_box_matches_draft_with_head_tail_landmarks"
                        counters["accepted_without_lm_but_draft_has_pose"] += 1
            elif len(valid) < 2:
                reason = "insufficient_valid_landmarks"
            elif head_id is None or tail_id is None:
                reason = "head_or_tail_category_not_defined_in_session"
            elif not has_head and not has_tail:
                reason = "missing_head_and_tail_landmarks"
            elif not has_head:
                reason = "missing_head_landmark"
            elif not has_tail:
                reason = "missing_tail_landmark"
            else:
                reason = "keypoint_extraction_failed"

            failure_reasons[reason] += 1
            rec = {
                "image": image_filename,
                "box_index": box_idx,
                "box": {
                    "left": box.get("left"),
                    "top": box.get("top"),
                    "width": box.get("width"),
                    "height": box.get("height"),
                },
                "reason": reason,
                "hint": hint,
                "landmark_count": len(valid),
                "head_id": head_id,
                "tail_id": tail_id,
                "has_head": has_head,
                "has_tail": has_tail,
                "landmark_ids_present": sorted(lm_ids),
            }
            per_image_failures[image_filename].append(rec)
            if len(failure_examples) < max_failures:
                failure_examples.append(rec)

    total_boxes = int(counters["finalized_boxes_total"])
    pose_ready_boxes = int(counters["pose_ready_boxes"])
    pose_ratio = (float(pose_ready_boxes) / float(total_boxes)) if total_boxes > 0 else 0.0
    use_pose_threshold = 0.50
    use_pose_predicted = bool(head_id is not None and tail_id is not None and total_boxes > 0 and pose_ratio >= use_pose_threshold)

    per_image_summary = []
    for image_name, rows in per_image_failures.items():
        reason_counts = Counter(r["reason"] for r in rows)
        hints = sorted({r["hint"] for r in rows if r.get("hint")})
        per_image_summary.append(
            {
                "image": image_name,
                "failed_boxes": len(rows),
                "reasons": dict(reason_counts),
                "hints": hints,
            }
        )
    per_image_summary.sort(key=lambda r: r["failed_boxes"], reverse=True)

    report = {
        "generated_at": _utc_now_iso(),
        "session_dir": session_dir,
        "class_name": class_name,
        "head_id": head_id,
        "tail_id": tail_id,
        "thresholds": {
            "pose_enable_ratio": use_pose_threshold,
        },
        "summary": {
            "label_files": int(counters["label_files"]),
            "finalized_images": int(counters["finalized_images"]),
            "skipped_unfinalized_images": int(counters["skipped_unfinalized_images"]),
            "finalized_boxes_total": total_boxes,
            "pose_ready_boxes": pose_ready_boxes,
            "pose_ready_ratio": round(pose_ratio, 6),
            "use_pose_predicted": use_pose_predicted,
            "images_missing_for_audit": int(counters["images_missing_for_audit"]),
            "finalized_fallback_to_boxes": int(counters["finalized_fallback_to_boxes"]),
            "accepted_without_lm_but_draft_has_pose": int(counters["accepted_without_lm_but_draft_has_pose"]),
        },
        "failure_reasons": dict(failure_reasons),
        "per_image_failures": per_image_summary[:200],
        "failure_examples": failure_examples,
    }

    # Persist under debug/models for consistency with other model debug files.
    debug_tag = _safe_class_name(class_name)
    run_dir, run_id = dio.create_model_run_dir(session_dir, "yolo_pose_audit", debug_tag)
    dio.write_run_manifest(
        run_dir,
        model_type="yolo_pose_audit",
        tag=debug_tag,
        project_root=session_dir,
        extra={
            "status": "completed",
            "run_id": run_id,
            "pose_ready_ratio": report["summary"]["pose_ready_ratio"],
            "use_pose_predicted": use_pose_predicted,
        },
    )
    dio.write_run_json(run_dir, "pose_readiness.json", report)
    # Convenience latest pointer.
    dio.write_json(os.path.join(session_dir, "debug", "pose_readiness_latest.json"), report)
    report["debug_run_dir"] = run_dir
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit YOLO-pose readiness for a session.")
    parser.add_argument("session_dir", help="Path to session directory (contains labels/, images/, session.json)")
    parser.add_argument("--class-name", default="object", help="Class name used for YOLO training context")
    parser.add_argument("--max-failures", type=int, default=200, help="Max detailed failure examples to include")
    args = parser.parse_args()

    try:
        report = audit_pose_readiness(
            session_dir=args.session_dir,
            class_name=args.class_name,
            max_failures=max(10, int(args.max_failures)),
        )
        # Use ASCII-safe stdout to avoid Windows console encoding failures.
        print(json.dumps({"ok": True, **report}, ensure_ascii=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
