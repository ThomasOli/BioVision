#!/usr/bin/env python3
"""Prepare and verify externally-authored dlib XML for BioVision training.

Imported XML does not carry BioVision's schema-ID mapping or frozen validation
lineage.  This module derives those artifacts only when the XML part slots can
be tied to explicit session schema IDs.  A conventional zero-based dlib slot
order may be associated with session template order, but only after the caller
records an explicit user confirmation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


STANDARD_SIZE = 512
ORIENTATION_MODES = {"directional", "bilateral", "axial", "invariant"}
MODEL_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ImportedDlibContractError(RuntimeError):
    def __init__(self, message: str, *, code: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


class _ImportFileTransaction:
    """Rollback every final artifact touched by one preparation attempt."""

    def __init__(self):
        self.snapshots: dict[str, bytes | None] = {}
        self.created_files: set[str] = set()

    def snapshot(self, path: str) -> None:
        absolute = os.path.abspath(path)
        if absolute in self.snapshots:
            return
        self.snapshots[absolute] = Path(absolute).read_bytes() if os.path.isfile(absolute) else None

    def record_created(self, path: str) -> None:
        self.created_files.add(os.path.abspath(path))

    def rollback(self) -> None:
        for path, previous in reversed(list(self.snapshots.items())):
            if previous is None:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(previous)
        for path in sorted(self.created_files, reverse=True):
            if path in self.snapshots:
                continue
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


_ACTIVE_TRANSACTION: _ImportFileTransaction | None = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(encoded.encode("utf-8"))


def _atomic_write_json(path: str, payload: Any) -> None:
    if _ACTIVE_TRANSACTION is not None:
        _ACTIVE_TRANSACTION.snapshot(path)
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(destination))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_write_xml(path: str, root: ET.Element) -> None:
    if _ACTIVE_TRANSACTION is not None:
        _ACTIVE_TRANSACTION.snapshot(path)
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(destination))
    os.close(fd)
    try:
        ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ImportedDlibContractError(f"{label} must be an integer.", code="invalid_integer")
    try:
        parsed = int(value)
        if isinstance(value, float) and not value.is_integer():
            raise ValueError
        if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
            raise ValueError
        return parsed
    except (TypeError, ValueError) as exc:
        raise ImportedDlibContractError(
            f"{label} must be an integer, got {value!r}.", code="invalid_integer"
        ) from exc


def _load_session_contract(project_root: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    session_path = os.path.join(project_root, "session.json")
    try:
        with open(session_path, "r", encoding="utf-8") as handle:
            session = json.load(handle)
    except Exception as exc:
        raise ImportedDlibContractError(
            f"Could not read session schema metadata at {session_path}: {exc}",
            code="missing_session_schema",
        ) from exc
    if not isinstance(session, dict):
        raise ImportedDlibContractError(
            "session.json must contain an object.", code="invalid_session_schema"
        )
    raw_template = session.get("landmarkTemplate")
    if not isinstance(raw_template, list) or not raw_template:
        raise ImportedDlibContractError(
            "The active session has no landmark template. Choose a schema before importing dlib XML.",
            code="missing_landmark_template",
        )

    template: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for position, raw in enumerate(raw_template):
        if not isinstance(raw, dict):
            raise ImportedDlibContractError(
                f"Session landmark #{position + 1} is not an object.", code="invalid_landmark_template"
            )
        landmark_id = _integer(raw.get("index"), f"Session landmark #{position + 1} index")
        if landmark_id <= 0:
            raise ImportedDlibContractError(
                f"Session landmark ID {landmark_id} must be a positive integer.",
                code="invalid_schema_id",
            )
        if landmark_id in seen_ids:
            raise ImportedDlibContractError(
                f"Session landmark ID {landmark_id} is duplicated; imported XML cannot be mapped unambiguously.",
                code="duplicate_schema_id",
            )
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ImportedDlibContractError(
                f"Session landmark ID {landmark_id} has no name.", code="missing_schema_name"
            )
        name_key = name.casefold()
        if name_key in seen_names:
            raise ImportedDlibContractError(
                f"Session landmark name '{name}' is duplicated; rename it before importing XML.",
                code="duplicate_schema_name",
            )
        seen_ids.add(landmark_id)
        seen_names.add(name_key)
        template.append(
            {
                "index": landmark_id,
                "name": name,
                "category": str(raw.get("category") or "").strip(),
                "required": raw.get("required") is not False,
            }
        )

    policy = session.get("orientationPolicy")
    mode = str(policy.get("mode") or "").strip().lower() if isinstance(policy, dict) else ""
    if session.get("orientationPolicyConfigured") is not True or mode not in ORIENTATION_MODES:
        raise ImportedDlibContractError(
            "Choose and save an explicit session orientation policy before importing dlib XML.",
            code="orientation_policy_not_configured",
        )
    return template, copy.deepcopy(policy), session


def _slot_from_name(raw_name: Any, *, cohort: str) -> int:
    name = str(raw_name or "").strip()
    if not re.fullmatch(r"\d+", name):
        raise ImportedDlibContractError(
            f"{cohort} XML part name '{name}' is not an explicit non-negative dlib slot. "
            "Use integer part names 0..N-1.",
            code="invalid_part_name",
        )
    return int(name)


def _parse_xml(path: str, cohort: str) -> dict[str, Any]:
    try:
        tree = ET.parse(path)
    except Exception as exc:
        raise ImportedDlibContractError(
            f"Could not parse {cohort} XML '{path}': {exc}", code="invalid_xml"
        ) from exc
    root = tree.getroot()
    images_node = root.find("images")
    if images_node is None:
        raise ImportedDlibContractError(
            f"{cohort} XML is missing its <images> node.", code="invalid_xml"
        )
    image_nodes = list(images_node.findall("image"))
    if not image_nodes:
        raise ImportedDlibContractError(
            f"{cohort} XML contains no images.", code=f"empty_{cohort}_cohort"
        )

    records: list[dict[str, Any]] = []
    expected_slots: set[int] | None = None
    name_by_slot: dict[int, str] = {}
    for image_index, image_node in enumerate(image_nodes):
        image_path = str(image_node.get("file") or "").strip()
        if not image_path or not os.path.isfile(image_path):
            raise ImportedDlibContractError(
                f"{cohort} XML image #{image_index + 1} is unavailable: {image_path or '(missing path)' }.",
                code="missing_xml_image",
            )
        source_digest = _sha256_file(image_path)
        standardized_name_match = re.fullmatch(
            r"bvimp_([0-9a-f]{64})_[0-9a-f]{20}\.png",
            os.path.basename(image_path).casefold(),
        )
        source_identity_digest = (
            standardized_name_match.group(1) if standardized_name_match else source_digest
        )
        boxes = list(image_node.findall("box"))
        if not boxes:
            raise ImportedDlibContractError(
                f"{cohort} XML image '{image_path}' has no annotated box.", code="missing_xml_box"
            )
        canonical_boxes: list[dict[str, Any]] = []
        for box_index, box in enumerate(boxes):
            left = _integer(box.get("left"), f"{cohort} box left")
            top = _integer(box.get("top"), f"{cohort} box top")
            width = _integer(box.get("width"), f"{cohort} box width")
            height = _integer(box.get("height"), f"{cohort} box height")
            if width <= 0 or height <= 0:
                raise ImportedDlibContractError(
                    f"{cohort} XML image '{image_path}' has a non-positive box size.",
                    code="invalid_xml_box",
                )
            parts: dict[int, dict[str, Any]] = {}
            for part in box.findall("part"):
                raw_name = str(part.get("name") or "").strip()
                slot = _slot_from_name(raw_name, cohort=cohort)
                if slot in parts:
                    raise ImportedDlibContractError(
                        f"{cohort} XML box #{box_index + 1} maps more than one part name to dlib slot {slot}.",
                        code="duplicate_part_slot",
                    )
                x = _integer(part.get("x"), f"{cohort} part {raw_name} x")
                y = _integer(part.get("y"), f"{cohort} part {raw_name} y")
                if x < left or x >= left + width or y < top or y >= top + height:
                    raise ImportedDlibContractError(
                        f"{cohort} XML part {raw_name} at ({x}, {y}) lies outside its annotated box "
                        f"({left}, {top}, {width}, {height}).",
                        code="part_outside_box",
                    )
                parts[slot] = {
                    "name": raw_name,
                    "x": x,
                    "y": y,
                    "x_standard": (
                        (float(x - left) / float(max(1, width - 1))) * (STANDARD_SIZE - 1)
                    ),
                    "y_standard": (
                        (float(y - top) / float(max(1, height - 1))) * (STANDARD_SIZE - 1)
                    ),
                }
                # Compare the raw spelling, not its integer value: dlib indexes
                # parts by lexical name sort, so "1" and "01" are two distinct
                # part names to dlib even though both denote slot 1.  An XML
                # mixing both conventions was assembled from inconsistent
                # sources and cannot be trusted to have one landmark contract.
                prior_name = name_by_slot.get(slot)
                if prior_name is not None and prior_name != raw_name:
                    raise ImportedDlibContractError(
                        f"{cohort} XML uses conflicting aliases '{prior_name}' and '{raw_name}' for slot {slot}. "
                        "Use one consistent part-name spelling throughout the dataset.",
                        code="ambiguous_part_slot",
                    )
                name_by_slot.setdefault(slot, raw_name)
            slots = set(parts)
            if not slots:
                raise ImportedDlibContractError(
                    f"{cohort} XML image '{image_path}' has a box with no landmarks.",
                    code="missing_xml_parts",
                )
            if expected_slots is None:
                expected_slots = slots
            elif slots != expected_slots:
                missing = sorted(expected_slots - slots)
                extra = sorted(slots - expected_slots)
                raise ImportedDlibContractError(
                    f"{cohort} XML boxes do not share one landmark contract; missing slots {missing}, extra slots {extra}.",
                    code="inconsistent_box_parts",
                )
            canonical_boxes.append(
                {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "parts": {str(slot): parts[slot] for slot in sorted(parts)},
                }
            )
        records.append(
            {
                "node": image_node,
                "imagePath": os.path.abspath(image_path),
                "sourceId": f"sha256:{source_identity_digest}",
                "imageSha256": source_digest,
                "boxes": canonical_boxes,
            }
        )

    slots = sorted(expected_slots or set())
    return {
        "path": os.path.abspath(path),
        "tree": tree,
        "root": root,
        "records": records,
        "slots": slots,
        "nameBySlot": name_by_slot,
    }


def _assert_matching_slots(cohorts: dict[str, dict[str, Any]]) -> list[int]:
    items = list(cohorts.items())
    expected = items[0][1]["slots"]
    for cohort, parsed in items[1:]:
        if parsed["slots"] != expected:
            raise ImportedDlibContractError(
                f"{cohort} XML uses dlib slots {parsed['slots']}, but train XML uses {expected}. "
                "Train, validation, and test must share the same landmark contract.",
                code="cohort_part_mismatch",
            )
    return list(expected)


def _mapping_proposal(slots: list[int], template: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dlibSlot": slot,
            "schemaId": template[position]["index"],
            "schemaName": template[position]["name"],
            "required": template[position]["required"],
        }
        for position, slot in enumerate(slots)
    ]


def _validate_policy_landmarks(
    policy: dict[str, Any], template: list[dict[str, Any]], mapped_ids: set[int]
) -> None:
    referenced: set[int] = set()
    for field in ("anteriorAnchorIds", "posteriorAnchorIds"):
        raw_values = policy.get(field, [])
        if raw_values is None:
            raw_values = []
        if not isinstance(raw_values, list):
            raise ImportedDlibContractError(
                f"Orientation policy field {field} must be a list.", code="invalid_orientation_contract"
            )
        referenced.update(_integer(value, f"Orientation policy {field} value") for value in raw_values)
    raw_pairs = policy.get("bilateralPairs", []) or []
    if not isinstance(raw_pairs, list):
        raise ImportedDlibContractError(
            "Orientation policy bilateralPairs must be a list.", code="invalid_orientation_contract"
        )
    for pair in raw_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ImportedDlibContractError(
                "Every orientation policy bilateral pair must contain exactly two schema IDs.",
                code="invalid_orientation_contract",
            )
        referenced.update(_integer(value, "Orientation bilateral-pair ID") for value in pair)

    schema_ids = {entry["index"] for entry in template}
    unknown = sorted(referenced - schema_ids)
    if unknown:
        raise ImportedDlibContractError(
            f"Orientation policy references schema IDs that do not exist: {unknown}.",
            code="invalid_orientation_contract",
        )
    omitted = sorted(referenced - mapped_ids)
    if omitted:
        raise ImportedDlibContractError(
            f"Imported XML omits orientation anchor/pair IDs {omitted}. Include those parts before training.",
            code="missing_orientation_landmarks",
        )


def _resolve_mapping(
    slots: list[int],
    template: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    confirm_template_order: bool,
) -> tuple[dict[int, int], str, list[dict[str, Any]]]:
    schema_by_id = {entry["index"]: entry for entry in template}
    slot_set = set(slots)
    required_ids = {entry["index"] for entry in template if entry["required"]}
    if slot_set.issubset(schema_by_id) and required_ids.issubset(slot_set):
        mapping = {slot: slot for slot in slots}
        mode = "explicit_schema_ids"
        proposal = [
            {
                "dlibSlot": slot,
                "schemaId": slot,
                "schemaName": schema_by_id[slot]["name"],
                "required": schema_by_id[slot]["required"],
            }
            for slot in slots
        ]
    else:
        if len(slots) != len(template):
            raise ImportedDlibContractError(
                "XML part slots are not schema IDs, and their count does not equal the complete session "
                "template. BioVision cannot infer which optional landmarks were omitted. Rename each XML "
                "part to an explicit schema ID or import a complete 0..N-1 part set.",
                code="ambiguous_schema_mapping",
                details={"xmlSlots": slots, "schemaIds": [entry["index"] for entry in template]},
            )
        proposal = _mapping_proposal(slots, template)
        if not confirm_template_order:
            raise ImportedDlibContractError(
                "XML parts are zero-based dlib slots, not explicit session schema IDs. Confirm the displayed "
                "slot-to-landmark association before import; BioVision will not invent this mapping silently.",
                code="mapping_confirmation_required",
                details={"requiresMappingConfirmation": True, "mappingProposal": proposal},
            )
        mapping = {item["dlibSlot"]: item["schemaId"] for item in proposal}
        mode = "confirmed_template_order"

    mapped_ids = set(mapping.values())
    missing_required = sorted(required_ids - mapped_ids)
    if missing_required:
        raise ImportedDlibContractError(
            f"Imported XML omits required session landmark IDs {missing_required}.",
            code="missing_required_landmarks",
        )
    if len(mapped_ids) != len(mapping):
        raise ImportedDlibContractError(
            "Multiple dlib slots map to one schema ID.", code="ambiguous_schema_mapping"
        )
    _validate_policy_landmarks(policy, template, mapped_ids)
    return mapping, mode, proposal


def _orientation_anchor_ids(
    policy: dict[str, Any], template: list[dict[str, Any]], mapped_ids: set[int]
) -> tuple[list[int], list[int]]:
    def explicit_ids(field: str) -> list[int]:
        raw = policy.get(field, []) or []
        return [_integer(value, f"Orientation policy {field} value") for value in raw]

    anterior = explicit_ids("anteriorAnchorIds")
    posterior = explicit_ids("posteriorAnchorIds")
    if anterior and posterior:
        return anterior, posterior

    head_categories = {
        str(value).strip().casefold()
        for value in (policy.get("headCategories", []) or [])
        if str(value).strip()
    }
    tail_categories = {
        str(value).strip().casefold()
        for value in (policy.get("tailCategories", []) or [])
        if str(value).strip()
    }
    if not anterior:
        anterior = [
            entry["index"]
            for entry in template
            if entry["index"] in mapped_ids
            and str(entry.get("category") or "").strip().casefold() in head_categories
        ]
    if not posterior:
        posterior = [
            entry["index"]
            for entry in template
            if entry["index"] in mapped_ids
            and str(entry.get("category") or "").strip().casefold() in tail_categories
        ]
    return anterior, posterior


def _read_image(path: str) -> np.ndarray:
    # np.fromfile/imdecode handles Windows paths that cv2.imread may not decode.
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ImportedDlibContractError(
            f"Could not decode imported XML image '{path}': {exc}", code="invalid_xml_image"
        ) from exc
    if image is None or image.ndim != 3:
        raise ImportedDlibContractError(
            f"Could not decode imported XML image '{path}'.", code="invalid_xml_image"
        )
    return image


def _write_png_atomic(path: str, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ImportedDlibContractError(
            f"Could not encode standardized imported crop '{path}'.", code="crop_write_failed"
        )
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    encoded_bytes = encoded.tobytes()
    if os.path.isfile(destination):
        if Path(destination).read_bytes() == encoded_bytes:
            return
        raise ImportedDlibContractError(
            f"Standardized crop identity collision at '{destination}'. Remove the conflicting crop or "
            "choose a new model tag.",
            code="crop_identity_collision",
        )
    if _ACTIVE_TRANSACTION is not None:
        _ACTIVE_TRANSACTION.record_created(destination)
    fd, temporary = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(destination))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _canonicalize_cohort(
    parsed: dict[str, Any],
    *,
    cohort: str,
    tag: str,
    output_dir: str,
    mapping: dict[int, int],
    template: list[dict[str, Any]],
    policy: dict[str, Any],
) -> ET.Element:
    mode = str(policy.get("mode") or "").strip().lower()
    target = str(policy.get("targetOrientation") or "").strip().lower()
    mapped_ids = set(mapping.values())
    anterior_ids, posterior_ids = _orientation_anchor_ids(policy, template, mapped_ids)
    if mode == "directional":
        if target not in {"left", "right"}:
            raise ImportedDlibContractError(
                "Directional imported XML requires an explicit targetOrientation of left or right.",
                code="invalid_orientation_contract",
            )
        if not anterior_ids or not posterior_ids:
            raise ImportedDlibContractError(
                "Directional imported XML requires mapped anterior/head and posterior/tail anchors so each "
                "crop can be normalized to the session target orientation.",
                code="missing_orientation_landmarks",
            )

    result_root = ET.Element("dataset")
    ET.SubElement(result_root, "name").text = f"BioVision imported dlib {cohort} ({tag})"
    images = ET.SubElement(result_root, "images")
    source_keys = sorted(mapping)
    canonical_slot_by_source = {
        source_key: slot for slot, source_key in enumerate(source_keys)
    }
    width = max(2, len(str(max(0, len(source_keys) - 1))))
    for image_index, record in enumerate(parsed["records"]):
        image = _read_image(record["imagePath"])
        image_height, image_width = image.shape[:2]
        for box_index, box in enumerate(record["boxes"]):
            left, top = int(box["left"]), int(box["top"])
            box_width, box_height = int(box["width"]), int(box["height"])
            if left < 0 or top < 0 or left + box_width > image_width or top + box_height > image_height:
                raise ImportedDlibContractError(
                    f"{cohort} XML box ({left}, {top}, {box_width}, {box_height}) exceeds image bounds "
                    f"{image_width}x{image_height} for '{record['imagePath']}'.",
                    code="box_outside_image",
                )
            crop = image[top : top + box_height, left : left + box_width]
            if crop.size == 0:
                raise ImportedDlibContractError(
                    f"{cohort} XML produced an empty crop for '{record['imagePath']}'.",
                    code="invalid_xml_box",
                )
            standardized = cv2.resize(crop, (STANDARD_SIZE, STANDARD_SIZE), interpolation=cv2.INTER_AREA)
            parts = {
                int(raw_slot): {
                    **part,
                    "x_standard": (float(part["x"] - left) / float(max(1, box_width - 1)))
                    * (STANDARD_SIZE - 1),
                    "y_standard": (float(part["y"] - top) / float(max(1, box_height - 1)))
                    * (STANDARD_SIZE - 1),
                }
                for raw_slot, part in box["parts"].items()
            }

            flipped = False
            if mode == "directional":
                by_schema = {mapping[slot]: part for slot, part in parts.items()}
                anterior_x = [by_schema[item]["x_standard"] for item in anterior_ids if item in by_schema]
                posterior_x = [by_schema[item]["x_standard"] for item in posterior_ids if item in by_schema]
                if not anterior_x or not posterior_x:
                    raise ImportedDlibContractError(
                        f"{cohort} XML sample '{record['imagePath']}' omits directional anchors.",
                        code="missing_orientation_landmarks",
                    )
                anterior_mean = sum(anterior_x) / len(anterior_x)
                posterior_mean = sum(posterior_x) / len(posterior_x)
                if math.isclose(anterior_mean, posterior_mean, abs_tol=1e-6):
                    raise ImportedDlibContractError(
                        f"Cannot determine left/right orientation for '{record['imagePath']}' because anterior "
                        "and posterior anchors have the same x position.",
                        code="ambiguous_sample_orientation",
                    )
                observed = "left" if anterior_mean < posterior_mean else "right"
                flipped = observed != target
                if flipped:
                    standardized = cv2.flip(standardized, 1)
                    for part in parts.values():
                        part["x_standard"] = (STANDARD_SIZE - 1) - part["x_standard"]

            sample_identity = _sha256_json(
                {
                    "source": record["sourceId"],
                    "imageIndex": image_index,
                    "boxIndex": box_index,
                    "box": [left, top, box_width, box_height],
                    "parts": {
                        str(slot): [parts[slot]["x_standard"], parts[slot]["y_standard"]]
                        for slot in sorted(parts)
                    },
                    "flipped": flipped,
                }
            )
            source_digest = record["sourceId"].removeprefix("sha256:")
            crop_name = f"bvimp_{source_digest}_{sample_identity[:20]}.png"
            crop_path = os.path.join(output_dir, crop_name)
            _write_png_atomic(crop_path, standardized)
            image_el = ET.SubElement(images, "image", file=os.path.abspath(crop_path))
            box_el = ET.SubElement(
                image_el,
                "box",
                top="0",
                left="0",
                width=str(STANDARD_SIZE),
                height=str(STANDARD_SIZE),
            )
            for slot in source_keys:
                ET.SubElement(
                    box_el,
                    "part",
                    name=f"{canonical_slot_by_source[slot]:0{width}d}",
                    x=str(int(round(parts[slot]["x_standard"]))),
                    y=str(int(round(parts[slot]["y_standard"]))),
                )
    return result_root


def _assert_canonical_xml(parsed: dict[str, Any], cohort: str) -> None:
    for record in parsed["records"]:
        if len(record["boxes"]) != 1:
            raise ImportedDlibContractError(
                f"Prepared {cohort} XML must contain one full-image box per standardized crop.",
                code="noncanonical_import_xml",
            )
        box = record["boxes"][0]
        if (box["left"], box["top"], box["width"], box["height"]) != (
            0,
            0,
            STANDARD_SIZE,
            STANDARD_SIZE,
        ):
            raise ImportedDlibContractError(
                f"Prepared {cohort} XML contains a noncanonical box. Re-import the source XML.",
                code="noncanonical_import_xml",
            )
        image = _read_image(record["imagePath"])
        if image.shape[:2] != (STANDARD_SIZE, STANDARD_SIZE):
            raise ImportedDlibContractError(
                f"Prepared {cohort} crop is not {STANDARD_SIZE}x{STANDARD_SIZE}: {record['imagePath']}.",
                code="noncanonical_import_xml",
            )


def _copy_xml_with_records(parsed: dict[str, Any], records: list[dict[str, Any]]) -> ET.Element:
    source_root = parsed["root"]
    result_root = ET.Element(source_root.tag, source_root.attrib)
    for child in list(source_root):
        if child.tag != "images":
            result_root.append(copy.deepcopy(child))
    images = ET.SubElement(result_root, "images")
    for record in records:
        images.append(copy.deepcopy(record["node"]))
    return result_root


def _record_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "imageSha256": record["imageSha256"],
        "boxes": [
            {
                "left": box["left"],
                "top": box["top"],
                "width": box["width"],
                "height": box["height"],
                "parts": {
                    slot: {"x": part["x"], "y": part["y"]}
                    for slot, part in box["parts"].items()
                },
            }
            for box in record["boxes"]
        ],
    }


def _cohort_identity(records: list[dict[str, Any]], cohort: str) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    names: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_source[record["sourceId"]].append(_record_snapshot(record))
        names[record["sourceId"]].add(os.path.basename(record["imagePath"]))
    snapshots = {
        source_id: _sha256_json(
            {
                "formatVersion": 1,
                "cohortKind": cohort,
                "sourceId": source_id,
                "samples": sorted(samples, key=_sha256_json),
            }
        )
        for source_id, samples in sorted(by_source.items())
    }
    revision = _sha256_json(
        {
            "formatVersion": 1,
            "cohortKind": cohort,
            "sources": [
                {"sourceId": source_id, "snapshotSha256": snapshots[source_id]}
                for source_id in sorted(snapshots)
            ],
        }
    ) if snapshots else None
    return {
        "sourceIds": sorted(snapshots),
        "sourceSnapshots": snapshots,
        "sourceNames": {key: sorted(value) for key, value in names.items()},
        "revision": revision,
    }


def _derive_holdout(
    train: dict[str, Any], fraction: float, seed: int, cohort: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in train["records"]:
        by_source[record["sourceId"]].append(record)
    source_ids = sorted(by_source)
    if len(source_ids) < 2:
        raise ImportedDlibContractError(
            f"Imported train XML needs at least two remaining content-distinct images to create a disjoint "
            f"{cohort} cohort. Add images or import an explicit {cohort} XML.",
            code=f"insufficient_{cohort}_sources",
        )
    count = max(1, int(round(len(source_ids) * fraction)))
    count = min(count, len(source_ids) - 1)
    ordered = sorted(
        source_ids,
        key=lambda source_id: _sha256_bytes(
            f"imported-dlib-{cohort}-v1:{seed}:{source_id}".encode("utf-8")
        ),
    )
    holdout_ids = set(ordered[:count])
    train_records = [record for record in train["records"] if record["sourceId"] not in holdout_ids]
    holdout_records = [record for record in train["records"] if record["sourceId"] in holdout_ids]
    return train_records, holdout_records


def _assert_disjoint(cohorts: dict[str, list[dict[str, Any]]]) -> None:
    source_sets = {
        cohort: {record["sourceId"] for record in records}
        for cohort, records in cohorts.items()
    }
    names = list(source_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(source_sets[left] & source_sets[right])
            if overlap:
                raise ImportedDlibContractError(
                    f"Imported {left} and {right} cohorts share {len(overlap)} image-content source(s). "
                    "Use content-disjoint train, validation, and test files.",
                    code="cohort_source_overlap",
                    details={"left": left, "right": right, "overlap": overlap},
                )


def _build_landmark_template(
    train_records: list[dict[str, Any]], mapping: dict[int, int]
) -> dict[str, dict[str, Any]]:
    values: dict[int, dict[str, list[float]]] = {
        schema_id: {"x": [], "y": []} for schema_id in mapping.values()
    }
    for record in train_records:
        for box in record["boxes"]:
            for raw_slot, part in box["parts"].items():
                schema_id = mapping[int(raw_slot)]
                values[schema_id]["x"].append(float(part["x_standard"]))
                values[schema_id]["y"].append(float(part["y_standard"]))
    result: dict[str, dict[str, Any]] = {}
    for schema_id, axes in sorted(values.items()):
        xs, ys = axes["x"], axes["y"]
        if not xs or not ys:
            raise ImportedDlibContractError(
                f"Training cohort has no coordinates for mapped schema ID {schema_id}.",
                code="missing_training_landmark",
            )
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        result[str(schema_id)] = {
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_std": math.sqrt(sum((value - x_mean) ** 2 for value in xs) / len(xs)),
            "y_std": math.sqrt(sum((value - y_mean) ** 2 for value in ys) / len(ys)),
            "count": len(xs),
        }
    return result


def _build_mapping_payload(
    *,
    mapping: dict[int, int],
    mapping_mode: str,
    proposal: list[dict[str, Any]],
    template: list[dict[str, Any]],
    policy: dict[str, Any],
    session: dict[str, Any],
    train_records: list[dict[str, Any]],
    name_by_slot: dict[int, str],
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    mapped_ids = set(mapping.values())
    reverse = {str(schema_id): slot for slot, schema_id in mapping.items()}
    anterior_ids, posterior_ids = _orientation_anchor_ids(policy, template, mapped_ids)
    orientation_mode = str(policy["mode"]).strip().lower()
    return {
        "format": "biovision.imported-dlib-id-mapping.v1",
        "source": "imported_dlib_xml",
        "mapping_mode": mapping_mode,
        "mapping_confirmation": {
            "required": mapping_mode == "confirmed_template_order",
            "confirmed": mapping_mode == "confirmed_template_order",
            "proposal": proposal,
        },
        "dlib_to_original": {str(slot): schema_id for slot, schema_id in mapping.items()},
        "dlib_index_to_original": {str(slot): schema_id for slot, schema_id in mapping.items()},
        "dlib_name_to_original": {
            str(name_by_slot.get(slot, slot)): schema_id for slot, schema_id in mapping.items()
        },
        "original_to_dlib": reverse,
        "original_ids": sorted(mapped_ids),
        "excluded_ids": sorted(entry["index"] for entry in template if entry["index"] not in mapped_ids),
        "num_landmarks": len(mapping),
        "standard_size": STANDARD_SIZE,
        "part_name_width": max(1, *(len(str(value)) for value in name_by_slot.values())),
        "part_names_sorted": [str(name_by_slot.get(slot, slot)) for slot in sorted(mapping)],
        "landmark_template": _build_landmark_template(train_records, mapping),
        "schema_contract": template,
        "schemaSemanticFingerprint": session.get("schemaSemanticFingerprint"),
        "schemaSemanticVersion": session.get("schemaSemanticVersion"),
        "training_config": {
            "seed": seed,
            "test_split": validation_fraction,
            "orientation_mode": orientation_mode,
            "orientation_policy": policy,
            "target_orientation": policy.get("targetOrientation"),
            "head_landmark_id": anterior_ids[0] if anterior_ids else None,
            "tail_landmark_id": posterior_ids[0] if posterior_ids else None,
            "canonical_training_enabled": orientation_mode == "directional",
            "canonical_training_source": (
                "imported_xml_anchor_flip" if orientation_mode == "directional" else "imported_xml_box_resize"
            ),
            "imported_dlib_xml": True,
        },
    }


def _mapping_from_existing(
    payload: Any,
    slots: list[int],
    template: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[int, int]:
    if not isinstance(payload, dict) or payload.get("source") != "imported_dlib_xml":
        raise ImportedDlibContractError(
            "Imported XML mapping sidecar is missing or was not created by the import workflow. "
            "Re-import the XML and confirm its landmark association.",
            code="missing_import_mapping",
        )
    raw_mapping = payload.get("dlib_index_to_original")
    if not isinstance(raw_mapping, dict):
        raise ImportedDlibContractError(
            "Imported XML mapping sidecar has no dlib_index_to_original contract.",
            code="invalid_import_mapping",
        )
    mapping = {
        _integer(key, "Mapping dlib index"): _integer(value, "Mapping schema ID")
        for key, value in raw_mapping.items()
    }
    if sorted(mapping) != slots:
        raise ImportedDlibContractError(
            f"Imported XML part slots changed from the saved mapping; expected {sorted(mapping)}, found {slots}. "
            "Re-import the XML instead of training with stale metadata.",
            code="stale_import_mapping",
        )
    schema_ids = {entry["index"] for entry in template}
    mapped_ids = set(mapping.values())
    if not mapped_ids.issubset(schema_ids) or len(mapped_ids) != len(mapping):
        raise ImportedDlibContractError(
            "Imported XML mapping does not uniquely reference the active session schema.",
            code="stale_import_mapping",
        )
    required = {entry["index"] for entry in template if entry["required"]}
    if not required.issubset(mapped_ids):
        raise ImportedDlibContractError(
            f"Imported XML mapping omits required schema IDs {sorted(required - mapped_ids)}.",
            code="missing_required_landmarks",
        )
    training = payload.get("training_config")
    saved_policy = training.get("orientation_policy") if isinstance(training, dict) else None
    if saved_policy != policy:
        raise ImportedDlibContractError(
            "The session orientation policy changed after XML import. Re-import to create a new explicit "
            "mapping contract before training.",
            code="stale_orientation_contract",
        )
    saved_contract = payload.get("schema_contract")
    if saved_contract != template:
        raise ImportedDlibContractError(
            "The session landmark IDs/names/required semantics changed after XML import. Re-import the XML.",
            code="stale_schema_contract",
        )
    _validate_policy_landmarks(policy, template, mapped_ids)
    return mapping


def _prepare_or_verify_impl(
    project_root: str,
    tag: str,
    *,
    mode: str,
    validation_mode: str,
    test_mode: str,
    confirm_template_order: bool,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any]:
    project_root = os.path.abspath(project_root)
    if not MODEL_TAG_RE.fullmatch(tag):
        raise ImportedDlibContractError(
            "Model tag may contain only letters, numbers, dot, underscore, or hyphen.",
            code="invalid_model_tag",
        )
    if not 0 < validation_fraction < 0.5:
        raise ImportedDlibContractError(
            "Validation fraction must be greater than zero and less than 0.5.",
            code="invalid_validation_fraction",
        )
    template, policy, session = _load_session_contract(project_root)
    xml_dir = os.path.join(project_root, "xml")
    debug_dir = os.path.join(project_root, "debug")
    train_path = os.path.join(xml_dir, f"train_{tag}.xml")
    validation_path = os.path.join(xml_dir, f"validation_{tag}.xml")
    test_path = os.path.join(xml_dir, f"test_{tag}.xml")
    mapping_path = os.path.join(debug_dir, f"id_mapping_{tag}.json")
    split_path = os.path.join(debug_dir, f"split_info_{tag}.json")
    cohort_path = os.path.join(debug_dir, "cohorts", f"imported_dlib_{tag}.json")

    train = _parse_xml(train_path, "train")
    parsed_cohorts: dict[str, dict[str, Any]] = {"train": train}
    if validation_mode == "explicit" or mode == "verify":
        if not os.path.isfile(validation_path):
            raise ImportedDlibContractError(
                f"Frozen validation XML is missing: {validation_path}. Re-import the dataset.",
                code="missing_validation_xml",
            )
        parsed_cohorts["validation"] = _parse_xml(validation_path, "validation")
    if test_mode == "explicit" or (mode == "verify" and os.path.isfile(test_path)):
        parsed_cohorts["test"] = _parse_xml(test_path, "test")
    slots = _assert_matching_slots(parsed_cohorts)

    if mode == "prepare":
        mapping, mapping_mode, proposal = _resolve_mapping(
            slots, template, policy, confirm_template_order=confirm_template_order
        )
        crop_dir = os.path.join(project_root, "corrected_images", f"imported_dlib_{tag}")
        cohort_paths = {
            "train": train_path,
            "validation": validation_path,
            "test": test_path,
        }
        # Dlib augmentation assumes a single full 512x512 box.  Normalize every
        # imported box into that representation and rewrite part names to stable,
        # zero-padded dlib slots before any cohort split or template statistics.
        for cohort, parsed in list(parsed_cohorts.items()):
            canonical_root = _canonicalize_cohort(
                parsed,
                cohort=cohort,
                tag=tag,
                output_dir=crop_dir,
                mapping=mapping,
                template=template,
                policy=policy,
            )
            _atomic_write_xml(cohort_paths[cohort], canonical_root)
            parsed_cohorts[cohort] = _parse_xml(cohort_paths[cohort], cohort)
        mapping = {
            canonical_slot: mapping[source_key]
            for canonical_slot, source_key in enumerate(sorted(mapping))
        }
        train = parsed_cohorts["train"]
        if test_mode == "derive":
            train_records, test_records = _derive_holdout(
                train, validation_fraction, seed, "test"
            )
            _atomic_write_xml(train_path, _copy_xml_with_records(train, train_records))
            _atomic_write_xml(test_path, _copy_xml_with_records(train, test_records))
            train = _parse_xml(train_path, "train")
            parsed_cohorts["train"] = train
            parsed_cohorts["test"] = _parse_xml(test_path, "test")
        if validation_mode == "derive":
            train_records, validation_records = _derive_holdout(
                train, validation_fraction, seed, "validation"
            )
            _atomic_write_xml(train_path, _copy_xml_with_records(train, train_records))
            _atomic_write_xml(validation_path, _copy_xml_with_records(train, validation_records))
            train = _parse_xml(train_path, "train")
            parsed_cohorts["train"] = train
            parsed_cohorts["validation"] = _parse_xml(validation_path, "validation")
        _assert_matching_slots(parsed_cohorts)
        for cohort, parsed in parsed_cohorts.items():
            _assert_canonical_xml(parsed, cohort)
        name_by_slot: dict[int, str] = {}
        for parsed in parsed_cohorts.values():
            name_by_slot.update(parsed["nameBySlot"])
        mapping_payload = _build_mapping_payload(
            mapping=mapping,
            mapping_mode=mapping_mode,
            proposal=proposal,
            template=template,
            policy=policy,
            session=session,
            train_records=parsed_cohorts["train"]["records"],
            name_by_slot=name_by_slot,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    else:
        for cohort, parsed in parsed_cohorts.items():
            _assert_canonical_xml(parsed, cohort)
        try:
            with open(mapping_path, "r", encoding="utf-8") as handle:
                mapping_payload = json.load(handle)
        except Exception as exc:
            raise ImportedDlibContractError(
                f"Could not read imported XML mapping sidecar '{mapping_path}': {exc}. Re-import the XML.",
                code="missing_import_mapping",
            ) from exc
        mapping = _mapping_from_existing(mapping_payload, slots, template, policy)
        expected_template = _build_landmark_template(parsed_cohorts["train"]["records"], mapping)
        if mapping_payload.get("landmark_template") != expected_template:
            raise ImportedDlibContractError(
                "Imported train XML coordinates changed after its mapping template was frozen. Re-import the XML.",
                code="stale_import_mapping",
            )

    cohort_records = {
        cohort: parsed["records"] for cohort, parsed in parsed_cohorts.items()
    }
    _assert_disjoint(cohort_records)
    identities = {
        cohort: _cohort_identity(records, cohort)
        for cohort, records in cohort_records.items()
    }
    validation_identity = identities.get("validation")
    if not validation_identity or not validation_identity["revision"]:
        raise ImportedDlibContractError(
            "A non-empty, disjoint validation cohort is required for model promotion.",
            code="missing_validation_cohort",
        )
    test_identity = identities.get("test", {"sourceIds": [], "sourceSnapshots": {}, "revision": None})
    assignments = {
        source_id: cohort
        for cohort, identity in identities.items()
        for source_id in identity["sourceIds"]
    }
    cohort_manifest = {
        "version": 1,
        "source": "imported_dlib_xml",
        "seed": seed,
        "validationFraction": validation_fraction,
        "validationMode": validation_mode if mode == "prepare" else None,
        "assignments": assignments,
        "trainSourceSnapshots": identities["train"]["sourceSnapshots"],
        "validationSourceSnapshots": validation_identity["sourceSnapshots"],
        "testSourceSnapshots": test_identity["sourceSnapshots"],
        "trainCohortRevision": identities["train"]["revision"],
        "validationCohortRevision": validation_identity["revision"],
        "testCohortRevision": test_identity["revision"],
        "assignmentRevision": _sha256_json(assignments),
        "singleSourceOverlap": False,
        "validationSourceOverlap": False,
        "mappingSha256": _sha256_json(mapping_payload),
    }
    split_info = {
        "source": "imported_dlib_xml",
        "train_crops": len(parsed_cohorts["train"]["records"]),
        "validation_crops": len(parsed_cohorts["validation"]["records"]),
        "test_crops": len(parsed_cohorts.get("test", {}).get("records", [])),
        "total_crops": sum(len(records) for records in cohort_records.values()),
        "train_source_ids": identities["train"]["sourceIds"],
        "validation_source_ids": validation_identity["sourceIds"],
        "test_source_ids": test_identity["sourceIds"],
        "train_source_count": len(identities["train"]["sourceIds"]),
        "validation_source_count": len(validation_identity["sourceIds"]),
        "test_source_count": len(test_identity["sourceIds"]),
        "source_overlap_count": 0,
        "cohort_manifest": cohort_path,
        "validationCohortRevision": validation_identity["revision"],
        "validation_cohort_revision": validation_identity["revision"],
        "testCohortRevision": test_identity["revision"],
        "test_cohort_revision": test_identity["revision"],
        "singleSourceOverlap": False,
        "validationSourceOverlap": False,
        "split_policy": "imported_dlib_content_disjoint_validation_v1",
        "standard_size": STANDARD_SIZE,
        "train_files": [record["imagePath"] for record in parsed_cohorts["train"]["records"]],
        "validation_files": [record["imagePath"] for record in parsed_cohorts["validation"]["records"]],
        "test_files": [record["imagePath"] for record in parsed_cohorts.get("test", {}).get("records", [])],
    }

    if mode == "verify":
        try:
            with open(cohort_path, "r", encoding="utf-8") as handle:
                saved_manifest = json.load(handle)
            with open(split_path, "r", encoding="utf-8") as handle:
                saved_split = json.load(handle)
        except Exception as exc:
            raise ImportedDlibContractError(
                f"Imported XML frozen split metadata is missing or unreadable: {exc}. Re-import the XML.",
                code="missing_frozen_split",
            ) from exc
        for field in (
            "assignments",
            "trainSourceSnapshots",
            "validationSourceSnapshots",
            "testSourceSnapshots",
            "trainCohortRevision",
            "validationCohortRevision",
            "testCohortRevision",
            "mappingSha256",
        ):
            if saved_manifest.get(field) != cohort_manifest.get(field):
                raise ImportedDlibContractError(
                    f"Imported XML frozen cohort field '{field}' no longer matches current XML/image content. "
                    "Re-import the XML to intentionally create a new cohort revision.",
                    code="frozen_cohort_changed",
                )
        if saved_split.get("validationCohortRevision") != split_info["validationCohortRevision"]:
            raise ImportedDlibContractError(
                "Imported XML split metadata has a stale validation revision. Re-import the XML.",
                code="frozen_cohort_changed",
            )
    else:
        _atomic_write_json(mapping_path, mapping_payload)
        _atomic_write_json(cohort_path, cohort_manifest)
        _atomic_write_json(split_path, split_info)

    return {
        "ok": True,
        "mode": mode,
        "mappingMode": mapping_payload.get("mapping_mode"),
        "mappingPath": mapping_path,
        "splitInfoPath": split_path,
        "cohortManifestPath": cohort_path,
        "validationCohortRevision": validation_identity["revision"],
        "testCohortRevision": test_identity["revision"],
        "trainImages": len(parsed_cohorts["train"]["records"]),
        "validationImages": len(parsed_cohorts["validation"]["records"]),
        "testImages": len(parsed_cohorts.get("test", {}).get("records", [])),
        "landmarkIds": [mapping[index] for index in sorted(mapping)],
    }


def prepare_or_verify(
    project_root: str,
    tag: str,
    *,
    mode: str,
    validation_mode: str,
    test_mode: str,
    confirm_template_order: bool,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any]:
    global _ACTIVE_TRANSACTION
    if mode != "prepare":
        return _prepare_or_verify_impl(
            project_root,
            tag,
            mode=mode,
            validation_mode=validation_mode,
            test_mode=test_mode,
            confirm_template_order=confirm_template_order,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    if _ACTIVE_TRANSACTION is not None:
        raise RuntimeError("Nested imported-dlib preparation transactions are not supported.")
    transaction = _ImportFileTransaction()
    _ACTIVE_TRANSACTION = transaction
    try:
        return _prepare_or_verify_impl(
            project_root,
            tag,
            mode=mode,
            validation_mode=validation_mode,
            test_mode=test_mode,
            confirm_template_order=confirm_template_order,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    except Exception:
        transaction.rollback()
        raise
    finally:
        _ACTIVE_TRANSACTION = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
    parser.add_argument("tag")
    parser.add_argument("--mode", choices=("prepare", "verify"), default="prepare")
    parser.add_argument("--validation-mode", choices=("derive", "explicit"), default="derive")
    parser.add_argument("--test-mode", choices=("none", "derive", "explicit"), default="derive")
    parser.add_argument("--confirm-template-order", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = prepare_or_verify(
            args.project_root,
            args.tag,
            mode=args.mode,
            validation_mode=args.validation_mode,
            test_mode=args.test_mode,
            confirm_template_order=args.confirm_template_order,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    except ImportedDlibContractError as exc:
        result = {"ok": False, "error": str(exc), "code": exc.code, **exc.details}
    except Exception as exc:  # Keep IPC failures actionable without a traceback-only protocol.
        result = {"ok": False, "error": f"Imported dlib preparation failed: {exc}", "code": "unexpected_error"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
