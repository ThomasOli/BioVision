"""
normalize_session_landmark_indexing.py - normalize persisted session landmark ids to template indexing.

Usage:
    python scripts/normalize_session_landmark_indexing.py <session_dir> [<session_dir> ...]

The script compares persisted label landmark ids against the session landmark template and
applies a conservative repair when the session is clearly offset from the template:
  - already_aligned: no landmark-id rewrite
  - uniform_offset: apply a single numeric shift to all landmark ids
  - ordinal_remap: map the sorted observed id set onto the sorted template id set

For vertical-OBB bilateral sessions with detected top/bottom anchors, the script also
rebuilds live/finalized OBB geometry from the repaired landmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import repair_vertical_obb_session as vertical_repair


@dataclass
class MappingDecision:
    kind: str
    mapping: Dict[int, int]
    details: str


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _iter_landmark_ids_from_box(box: Dict[str, Any]) -> Iterable[int]:
    landmarks = box.get("landmarks")
    if not isinstance(landmarks, list):
        return
    for landmark in landmarks:
        if isinstance(landmark, dict) and landmark.get("id") is not None:
            yield int(landmark["id"])


def _collect_label_id_sets(labels_dir: str) -> List[Tuple[str, Tuple[int, ...]]]:
    observed: List[Tuple[str, Tuple[int, ...]]] = []
    for name in sorted(os.listdir(labels_dir)):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(labels_dir, name)
        data = _load_json(path)
        ids = sorted(
            {
                landmark_id
                for box in data.get("boxes", [])
                if isinstance(box, dict)
                for landmark_id in _iter_landmark_ids_from_box(box)
            }
        )
        if ids:
            observed.append((name, tuple(ids)))
    return observed


def _build_template_indices(session_data: Dict[str, Any]) -> List[int]:
    indices = []
    for landmark in session_data.get("landmarkTemplate", []):
        if isinstance(landmark, dict) and landmark.get("index") is not None:
            indices.append(int(landmark["index"]))
    return sorted(set(indices))


def _build_uniform_shift_mapping(observed: Sequence[int], template: Sequence[int]) -> Optional[Dict[int, int]]:
    if len(observed) != len(template) or not observed:
        return None
    offset = template[0] - observed[0]
    shifted = [value + offset for value in observed]
    if shifted != list(template):
        return None
    return {value: value + offset for value in observed}


def _classify_mapping(template_indices: Sequence[int], observed_sets: Sequence[Tuple[str, Tuple[int, ...]]]) -> MappingDecision:
    template = sorted(set(int(value) for value in template_indices))
    if not template:
        return MappingDecision("already_aligned", {}, "session has no landmark template")
    if not observed_sets:
        return MappingDecision("already_aligned", {}, "session has no label landmarks")

    observed_union = sorted({landmark_id for _, ids in observed_sets for landmark_id in ids})
    if not observed_union:
        return MappingDecision("already_aligned", {}, "session labels have no landmark ids")
    template_set = set(template)
    observed_set = set(observed_union)

    if observed_set.issubset(template_set):
        return MappingDecision("already_aligned", {}, "observed landmark ids are already within the template index set")

    uniform_mapping = _build_uniform_shift_mapping(observed_union, template)
    if uniform_mapping is not None:
        offset = template[0] - observed_union[0]
        return MappingDecision("uniform_offset", uniform_mapping, f"applied uniform landmark-id shift of {offset:+d}")

    if len(observed_union) == len(template):
        ordinal_mapping = {observed_union[index]: template[index] for index in range(len(template))}
        return MappingDecision("ordinal_remap", ordinal_mapping, "mapped sorted observed landmark ids onto sorted template indices")

    return MappingDecision("mixed", {}, "could not derive a safe landmark-id mapping from the template")


def _remap_box_landmark_ids(box: Dict[str, Any], mapping: Dict[int, int]) -> bool:
    changed = False
    landmarks = box.get("landmarks")
    if not isinstance(landmarks, list):
        return False
    for landmark in landmarks:
        if not isinstance(landmark, dict) or landmark.get("id") is None:
            continue
        landmark_id = int(landmark["id"])
        mapped = mapping.get(landmark_id)
        if mapped is None or mapped == landmark_id:
            continue
        landmark["id"] = int(mapped)
        changed = True
    return changed


def _normalize_template_indices(session_data: Dict[str, Any]) -> bool:
    changed = False
    template = session_data.get("landmarkTemplate")
    if not isinstance(template, list):
        return False
    for index, landmark in enumerate(template, start=1):
        if not isinstance(landmark, dict):
            continue
        if landmark.get("index") != index:
            landmark["index"] = index
            changed = True
    return changed


def _cleanup_orientation_policy(session_data: Dict[str, Any]) -> bool:
    changed = False
    orientation_policy = session_data.get("orientationPolicy")
    if isinstance(orientation_policy, dict) and "pcaLevelingMode" in orientation_policy:
        del orientation_policy["pcaLevelingMode"]
        changed = True
    return changed


def _repair_vertical_geometry_if_needed(
    session_data: Dict[str, Any],
    labels_dir: str,
    images_dir: str,
    mapping_applied: bool,
) -> int:
    policy = session_data.get("orientationPolicy")
    if not isinstance(policy, dict):
        return 0
    if policy.get("mode") != "bilateral" or policy.get("bilateralClassAxis") != "vertical_obb":
        return 0
    anterior = policy.get("anteriorAnchorIds")
    posterior = policy.get("posteriorAnchorIds")
    if not isinstance(anterior, list) or not isinstance(posterior, list) or len(anterior) == 0 or len(posterior) == 0:
        return 0
    top_id = int(anterior[0])
    bottom_id = int(posterior[0])

    repaired_files = 0
    for name in sorted(os.listdir(labels_dir)):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(labels_dir, name)
        data = _load_json(path)
        image_filename = data.get("imageFilename")
        image_size = (
            vertical_repair._read_image_size(os.path.join(images_dir, str(image_filename)))
            if isinstance(image_filename, str) and image_filename.strip()
            else None
        )
        changed = False
        boxes = data.get("boxes")
        if isinstance(boxes, list):
            for box in boxes:
                if isinstance(box, dict) and vertical_repair._repair_box(box, top_id, bottom_id, image_size):
                    changed = True
        finalized = data.get("finalizedDetection")
        if isinstance(finalized, dict) and isinstance(boxes, list) and (changed or mapping_applied):
            repaired_live = [
                vertical_repair._clone_finalized_box_from_live(box)
                for box in boxes
                if isinstance(box, dict)
            ]
            finalized["acceptedBoxes"] = repaired_live
            finalized["boxSignature"] = vertical_repair._build_box_signature(repaired_live)
            changed = True
        if changed:
            _save_json(path, data)
            repaired_files += 1
    return repaired_files


def normalize_session(session_dir: str) -> MappingDecision:
    session_dir = os.path.abspath(session_dir)
    session_path = os.path.join(session_dir, "session.json")
    labels_dir = os.path.join(session_dir, "labels")
    images_dir = os.path.join(session_dir, "images")
    if not os.path.isfile(session_path):
        raise FileNotFoundError(f"Missing session.json: {session_path}")
    if not os.path.isdir(labels_dir):
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    session_data = _load_json(session_path)
    template_indices = _build_template_indices(session_data)
    observed_sets = _collect_label_id_sets(labels_dir)
    decision = _classify_mapping(template_indices, observed_sets)

    session_changed = _cleanup_orientation_policy(session_data)
    session_changed = _normalize_template_indices(session_data) or session_changed

    mapping_applied = False
    if decision.kind in {"uniform_offset", "ordinal_remap"}:
        for name in sorted(os.listdir(labels_dir)):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(labels_dir, name)
            data = _load_json(path)
            changed = False
            boxes = data.get("boxes")
            if isinstance(boxes, list):
                for box in boxes:
                    if isinstance(box, dict) and _remap_box_landmark_ids(box, decision.mapping):
                        changed = True
            finalized = data.get("finalizedDetection")
            accepted = finalized.get("acceptedBoxes") if isinstance(finalized, dict) else None
            if isinstance(accepted, list):
                for box in accepted:
                    if isinstance(box, dict) and _remap_box_landmark_ids(box, decision.mapping):
                        changed = True
            if changed:
                _save_json(path, data)
                mapping_applied = True

    repaired_vertical = _repair_vertical_geometry_if_needed(session_data, labels_dir, images_dir, mapping_applied)

    if mapping_applied or session_changed or repaired_vertical > 0:
        session_data["lastModified"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        _save_json(session_path, session_data)

    if repaired_vertical > 0:
        decision = MappingDecision(decision.kind, decision.mapping, f"{decision.details}; rebuilt vertical OBB geometry in {repaired_vertical} label file(s)")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dirs", nargs="+")
    args = parser.parse_args()

    failures = 0
    for session_dir in args.session_dirs:
        try:
            decision = normalize_session(session_dir)
            print(f"{session_dir}: {decision.kind} ({decision.details})")
        except Exception as exc:  # pragma: no cover - CLI path
            failures += 1
            print(f"{session_dir}: failed ({exc})", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
