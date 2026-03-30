"""
repair_vertical_obb_session.py - Rebuild imported vertical-schema OBBs from anchors.

Usage:
    python scripts/repair_vertical_obb_session.py <session_dir> [--top-id N] [--bottom-id N]

If anchor ids are omitted, the script tries to detect them from the session landmark
template using descriptions/categories such as "Top Anchor" / "Bottom Anchor".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


Point = Tuple[float, float]


def _normalize_angle_signed(angle_deg: float) -> float:
    angle = float(angle_deg) % 360.0
    if angle > 180.0:
        angle -= 360.0
    if angle <= -180.0:
        angle += 360.0
    return 180.0 if abs(angle + 180.0) <= 1e-6 else angle


def _build_corners_from_axes(
    cx: float,
    cy: float,
    width: float,
    height: float,
    hx: float,
    hy: float,
    vx: float,
    vy: float,
) -> List[List[float]]:
    hw = width / 2.0
    hh = height / 2.0
    return [
        [cx - hw * hx - hh * vx, cy - hw * hy - hh * vy],
        [cx + hw * hx - hh * vx, cy + hw * hy - hh * vy],
        [cx + hw * hx + hh * vx, cy + hw * hy + hh * vy],
        [cx - hw * hx + hh * vx, cy - hw * hy + hh * vy],
    ]


def _measure_box_geometry(box: Dict[str, Any]) -> Tuple[float, float, float, float]:
    corners = box.get("obbCorners")
    if isinstance(corners, list) and len(corners) == 4:
        pts = [(float(p[0]), float(p[1])) for p in corners]
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        width = max(2.0, math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))
        height = max(2.0, math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]))
        return cx, cy, width, height
    left = float(box.get("left", 0.0))
    top = float(box.get("top", 0.0))
    width = max(2.0, float(box.get("width", 0.0)))
    height = max(2.0, float(box.get("height", 0.0)))
    return left + width / 2.0, top + height / 2.0, width, height


def _aabb_from_corners(corners: Iterable[Iterable[float]]) -> Tuple[int, int, int, int]:
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    left = int(round(min(xs)))
    top = int(round(min(ys)))
    right = int(round(max(xs)))
    bottom = int(round(max(ys)))
    return left, top, max(1, right - left), max(1, bottom - top)


def _read_image_size(image_path: str) -> Optional[Tuple[int, int]]:
    try:
        with open(image_path, "rb") as handle:
            header = handle.read(24)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width, height = struct.unpack(">II", header[16:24])
                return int(width), int(height)
            if header[:2] != b"\xff\xd8":
                return None
            handle.seek(2)
            while True:
                marker_prefix = handle.read(1)
                if not marker_prefix:
                    return None
                if marker_prefix != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if not marker:
                    return None
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                size_bytes = handle.read(2)
                if len(size_bytes) != 2:
                    return None
                segment_size = struct.unpack(">H", size_bytes)[0]
                if segment_size < 2:
                    return None
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                    data = handle.read(5)
                    if len(data) != 5:
                        return None
                    height, width = struct.unpack(">HH", data[1:5])
                    return int(width), int(height)
                handle.seek(segment_size - 2, os.SEEK_CUR)
    except Exception:
        return None
    return None


def _fit_corners_rigidly(
    corners: List[List[float]],
    image_size: Optional[Tuple[int, int]],
) -> List[List[float]]:
    if image_size is None:
        return corners
    image_width, image_height = image_size
    max_x_bound = max(0.0, float(image_width - 1))
    max_y_bound = max(0.0, float(image_height - 1))
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    cx = sum(xs) / 4.0
    cy = sum(ys) / 4.0
    resolved = [[float(x), float(y)] for x, y in corners]

    if span_x > max_x_bound or span_y > max_y_bound:
        width_scale = max_x_bound / span_x if span_x > 1e-6 else 1.0
        height_scale = max_y_bound / span_y if span_y > 1e-6 else 1.0
        scale = min(1.0, width_scale, height_scale)
        resolved = [
            [cx + (float(x) - cx) * scale, cy + (float(y) - cy) * scale]
            for x, y in resolved
        ]

    xs = [float(p[0]) for p in resolved]
    ys = [float(p[1]) for p in resolved]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = 0.0
    dy = 0.0
    if min_x < 0:
        dx = -min_x
    if max_x + dx > max_x_bound:
        dx += max_x_bound - (max_x + dx)
    if min_y < 0:
        dy = -min_y
    if max_y + dy > max_y_bound:
        dy += max_y_bound - (max_y + dy)
    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        resolved = [[float(x) + dx, float(y) + dy] for x, y in resolved]
    return resolved


def _build_box_signature(boxes: List[Dict[str, Any]]) -> str:
    reduced = []
    for box in boxes:
        try:
            left = int(round(float(box.get("left", 0))))
            top = int(round(float(box.get("top", 0))))
            width = int(round(float(box.get("width", 0))))
            height = int(round(float(box.get("height", 0))))
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue
        reduced.append({
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        })
    reduced.sort(key=lambda item: (item["left"], item["top"], item["width"], item["height"]))
    return json.dumps(reduced, separators=(",", ":"))


def _find_landmark(landmarks: List[Dict[str, Any]], landmark_id: int) -> Optional[Point]:
    for lm in landmarks:
        if lm.get("isSkipped"):
            continue
        try:
            if int(lm.get("id")) != int(landmark_id):
                continue
            return float(lm["x"]), float(lm["y"])
        except Exception:
            continue
    return None


def _detect_anchor_ids(session_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    policy = session_data.get("orientationPolicy") or {}
    if isinstance(policy, dict):
        anterior_ids = policy.get("anteriorAnchorIds")
        posterior_ids = policy.get("posteriorAnchorIds")
        if isinstance(anterior_ids, list) and isinstance(posterior_ids, list):
          try:
              top_id = int(anterior_ids[0]) if anterior_ids else None
              bottom_id = int(posterior_ids[0]) if posterior_ids else None
          except Exception:
              top_id = None
              bottom_id = None
          if top_id is not None and bottom_id is not None:
              return top_id, bottom_id

    template = session_data.get("landmarkTemplate") or []
    explicit_top = None
    explicit_bottom = None
    category_top = None
    category_bottom = None

    for lm in template:
        try:
            idx = int(lm.get("index"))
        except Exception:
            continue
        description = str(lm.get("description", "")).strip().lower()
        category = str(lm.get("category", "")).strip().lower()
        if explicit_top is None and "top anchor" in description:
            explicit_top = idx
        if explicit_bottom is None and "bottom anchor" in description:
            explicit_bottom = idx
        if category_top is None and category == "distal pole":
            category_top = idx
        if category_bottom is None and category == "proximal pole":
            category_bottom = idx

    return explicit_top or category_top, explicit_bottom or category_bottom


def _shift_box_landmark_ids(box: Dict[str, Any], shift: int) -> bool:
    if shift == 0:
        return False
    landmarks = box.get("landmarks")
    if not isinstance(landmarks, list) or not landmarks:
        return False
    changed = False
    for landmark in landmarks:
        try:
            landmark["id"] = int(landmark.get("id")) + int(shift)
            changed = True
        except Exception:
            continue
    return changed


def _repair_box(
    box: Dict[str, Any],
    top_id: int,
    bottom_id: int,
    image_size: Optional[Tuple[int, int]],
) -> bool:
    landmarks = box.get("landmarks")
    if not isinstance(landmarks, list) or not landmarks:
        return False

    head = _find_landmark(landmarks, top_id)
    tail = _find_landmark(landmarks, bottom_id)
    if head is None or tail is None:
        return False

    dx = tail[0] - head[0]
    dy = tail[1] - head[1]
    mag = math.hypot(dx, dy)
    if mag <= 1e-6:
        return False

    down_x = dx / mag
    down_y = dy / mag
    right_x = down_y
    right_y = -down_x

    cx, cy, width, height = _measure_box_geometry(box)
    corners = _build_corners_from_axes(cx, cy, width, height, right_x, right_y, down_x, down_y)
    corners = _fit_corners_rigidly(corners, image_size)
    left, top, aabb_width, aabb_height = _aabb_from_corners(corners)
    angle = _normalize_angle_signed(math.degrees(math.atan2(right_y, right_x)))

    box["obbCorners"] = corners
    box["angle"] = angle
    box["left"] = left
    box["top"] = top
    box["width"] = aabb_width
    box["height"] = aabb_height
    box["class_id"] = 0

    orientation_hint = box.get("orientation_hint")
    if isinstance(orientation_hint, dict):
        orientation_hint["orientation"] = "up"
        orientation_hint["confidence"] = (
            float(orientation_hint.get("confidence"))
            if orientation_hint.get("confidence") is not None
            else 1.0
        )
        orientation_hint["source"] = orientation_hint.get("source") or "session_repair"
        box["orientation_hint"] = orientation_hint
    if isinstance(box.get("orientation_override"), str):
        raw_override = str(box.get("orientation_override", "")).strip().lower()
        if raw_override in {"left", "right", "up", "down"}:
            box["orientation_override"] = "up"

    return True


def _clone_finalized_box_from_live(box: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {
        "left": int(round(float(box.get("left", 0.0)))),
        "top": int(round(float(box.get("top", 0.0)))),
        "width": int(round(float(box.get("width", 0.0)))),
        "height": int(round(float(box.get("height", 0.0)))),
    }
    if isinstance(box.get("orientation_override"), str):
        cloned["orientation_override"] = str(box["orientation_override"])
    if isinstance(box.get("orientation_hint"), dict):
        hint = box["orientation_hint"]
        cloned["orientation_hint"] = {
            **({"orientation": str(hint.get("orientation"))} if hint.get("orientation") else {}),
            **({"confidence": float(hint.get("confidence"))} if hint.get("confidence") is not None else {}),
            **({"source": str(hint.get("source"))} if hint.get("source") else {}),
        }
    if isinstance(box.get("obbCorners"), list) and len(box["obbCorners"]) == 4:
        cloned["obbCorners"] = [
            [float(point[0]), float(point[1])] for point in box["obbCorners"]
        ]
    if box.get("angle") is not None:
        cloned["angle"] = float(box["angle"])
    if box.get("class_id") is not None:
        cloned["class_id"] = int(box["class_id"])
    if isinstance(box.get("landmarks"), list):
        cloned["landmarks"] = [
            {
                "id": int(lm.get("id")),
                "x": float(lm.get("x")),
                "y": float(lm.get("y")),
                **({"isSkipped": True} if lm.get("isSkipped") else {}),
            }
            for lm in box["landmarks"]
            if lm.get("id") is not None and lm.get("x") is not None and lm.get("y") is not None
        ]
    return cloned


def repair_session(
    session_dir: str,
    top_id: Optional[int],
    bottom_id: Optional[int],
    shift_landmark_ids: int = 0,
) -> int:
    session_dir = os.path.abspath(session_dir)
    session_path = os.path.join(session_dir, "session.json")
    labels_dir = os.path.join(session_dir, "labels")
    images_dir = os.path.join(session_dir, "images")
    if not os.path.isfile(session_path):
        raise FileNotFoundError(f"Missing session.json: {session_path}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    with open(session_path, "r", encoding="utf-8") as handle:
        session_data = json.load(handle)

    if top_id is None or bottom_id is None:
        detected_top, detected_bottom = _detect_anchor_ids(session_data)
        top_id = top_id if top_id is not None else detected_top
        bottom_id = bottom_id if bottom_id is not None else detected_bottom
    if top_id is None or bottom_id is None:
        raise RuntimeError("Could not detect anchor ids automatically. Provide --top-id and --bottom-id.")

    repaired_files = 0
    for name in sorted(os.listdir(labels_dir)):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(labels_dir, name)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        image_filename = data.get("imageFilename")
        image_size = (
            _read_image_size(os.path.join(images_dir, str(image_filename)))
            if isinstance(image_filename, str) and image_filename.strip()
            else None
        )

        changed = False
        boxes = data.get("boxes")
        if isinstance(boxes, list):
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                if _shift_box_landmark_ids(box, shift_landmark_ids):
                    changed = True
                if _repair_box(box, int(top_id), int(bottom_id), image_size):
                    changed = True

        finalized = data.get("finalizedDetection")
        if isinstance(finalized, dict) and isinstance(boxes, list):
            repaired_live = [
                _clone_finalized_box_from_live(box)
                for box in boxes
                if isinstance(box, dict)
            ]
            finalized["acceptedBoxes"] = repaired_live
            finalized["boxSignature"] = _build_box_signature(repaired_live)
            changed = True

        if changed:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            repaired_files += 1

    if repaired_files > 0:
        session_data["lastModified"] = __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        with open(session_path, "w", encoding="utf-8") as handle:
            json.dump(session_data, handle, indent=2)

    return repaired_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir")
    parser.add_argument("--top-id", type=int, default=None)
    parser.add_argument("--bottom-id", type=int, default=None)
    parser.add_argument("--shift-landmark-ids", type=int, default=0)
    args = parser.parse_args()

    repaired = repair_session(
        args.session_dir,
        args.top_id,
        args.bottom_id,
        shift_landmark_ids=args.shift_landmark_ids,
    )
    print(f"Repaired {repaired} label file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
