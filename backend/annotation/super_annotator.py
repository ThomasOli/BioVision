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
import time
import uuid
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
import bv_utils.lineage as lineage
from bv_utils.landmark_artifacts import (
    ImmutableLandmarkArtifactError,
    resolve_landmark_runtime,
)
from detection.obb_utils import (
    iter_ultralytics_obb,
    load_obb_confidence_threshold,
    load_obb_nms_iou,
    resolve_obb_detection_preset,
)

STANDARD_SIZE = 512
OBB_PROMOTION_MIN_ABSOLUTE_IMPROVEMENT = 0.001
OBB_PROMOTION_MIN_RELATIVE_IMPROVEMENT = 0.005
OBB_PROMOTION_MIN_VALIDATION_SAMPLES = 2
OBB_PROMOTION_MIN_VALIDATION_GROUPS = 2
_OBB_EFFECTIVE_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_OBB_TRAINING_INPUT_CONTROLS = {
    "datasetYaml": "yaml_path",
    "exportManifest": "export_manifest_path",
    "cohortManifest": "cohort_manifest_path",
    "splitAssignments": "split_assignments_path",
    "syntheticManifest": "synthetic_manifest_path",
}


def _obb_promotion_policy():
    return {
        "policyVersion": "obb_map_effect_v1",
        "minimumAbsoluteImprovement": OBB_PROMOTION_MIN_ABSOLUTE_IMPROVEMENT,
        "minimumRelativeImprovement": OBB_PROMOTION_MIN_RELATIVE_IMPROVEMENT,
        "minimumValidationSamples": OBB_PROMOTION_MIN_VALIDATION_SAMPLES,
        "minimumValidationGroups": OBB_PROMOTION_MIN_VALIDATION_GROUPS,
        "requireAllConfiguredValidationClasses": True,
    }


def _capture_obb_training_input_guard(export_result):
    """Hash the exact OBB inputs that Ultralytics is allowed to consume.

    The exporter already emits a semantic effective-dataset descriptor.  This
    guard independently attests the material files on disk, so a concurrent
    edit, replacement, addition, or deletion during fit/evaluation cannot be
    silently published under the export-time revision.
    """
    if not isinstance(export_result, dict):
        raise RuntimeError("Cannot attest OBB training inputs without an export result")

    dataset_yaml = export_result.get("yaml_path")
    if not dataset_yaml:
        raise RuntimeError("Cannot attest OBB training inputs without dataset.yaml")
    dataset_yaml = os.path.abspath(os.fspath(dataset_yaml))
    if not os.path.isfile(dataset_yaml):
        raise RuntimeError(f"Cannot attest missing OBB dataset YAML: {dataset_yaml}")

    dataset_root = os.path.dirname(dataset_yaml)
    controls = {}
    for role, result_key in _OBB_TRAINING_INPUT_CONTROLS.items():
        source_path = export_result.get(result_key)
        if not source_path:
            continue
        source_path = os.path.abspath(os.fspath(source_path))
        if not os.path.isfile(source_path):
            raise RuntimeError(
                f"Cannot attest missing OBB {role} control file: {source_path}"
            )
        controls[role] = {
            "sha256": lineage.sha256_file(source_path),
            "sizeBytes": os.path.getsize(source_path),
        }

    effective_files = []
    for tree_name in ("images", "labels"):
        tree_root = os.path.join(dataset_root, tree_name)
        if not os.path.isdir(tree_root):
            continue
        for directory, _subdirs, filenames in os.walk(tree_root):
            for filename in sorted(filenames):
                suffix = os.path.splitext(filename)[1].lower()
                if tree_name == "images":
                    if suffix not in _OBB_EFFECTIVE_IMAGE_SUFFIXES:
                        continue
                elif suffix != ".txt":
                    continue
                file_path = os.path.join(directory, filename)
                if not os.path.isfile(file_path):
                    continue
                relative_path = os.path.relpath(file_path, dataset_root).replace("\\", "/")
                effective_files.append(
                    {
                        "relativePath": relative_path,
                        "sha256": lineage.sha256_file(file_path),
                        "sizeBytes": os.path.getsize(file_path),
                    }
                )

    declared_effective_dataset = export_result.get("effective_dataset")
    material = {
        "formatVersion": 1,
        "task": "obb",
        "declaredEffectiveDatasetRevision": (
            str(declared_effective_dataset.get("revision"))
            if isinstance(declared_effective_dataset, dict)
            and declared_effective_dataset.get("revision")
            else None
        ),
        "declaredEffectiveDatasetSha256": (
            lineage.sha256_json(declared_effective_dataset)
            if isinstance(declared_effective_dataset, dict)
            else None
        ),
        "controls": controls,
        "effectiveFiles": sorted(
            effective_files,
            key=lambda entry: entry["relativePath"],
        ),
    }
    return {
        **material,
        "revision": lineage.sha256_json(material),
    }


def _assert_obb_training_inputs_unchanged(export_result, expected_guard):
    """Re-attest OBB inputs after evaluation and fail before publication."""
    if not isinstance(expected_guard, dict) or not expected_guard.get("revision"):
        raise RuntimeError("The pre-fit OBB training-input attestation is missing")
    observed_guard = _capture_obb_training_input_guard(export_result)
    if observed_guard["revision"] != expected_guard["revision"]:
        raise RuntimeError(
            "Effective OBB training inputs changed during fit/evaluation; refusing to "
            "publish the candidate "
            f"(before={expected_guard['revision']}, after={observed_guard['revision']})"
        )
    return {
        **expected_guard,
        "postFitVerified": True,
        "postFitRevision": observed_guard["revision"],
    }


def send(obj):
    """Send a JSON object to stdout (one line)."""
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# Set by the main loop for each incoming command so all outgoing messages
# (progress and final) can echo the same ID back to Electron.
_current_request_id = None


def send_progress(message, percent, stage="processing", details=None):
    obj = {"status": "progress", "message": message, "percent": percent, "stage": stage}
    if details and isinstance(details, dict):
        obj["details"] = details
    if _current_request_id:
        obj["_request_id"] = _current_request_id
    send(obj)


def send_obb_progress(message, percent, stage="training", details=None):
    obj = {"status": "progress", "message": message, "percent": percent, "stage": stage}
    if details and isinstance(details, dict):
        obj["details"] = details
    if _current_request_id:
        obj["_request_id"] = _current_request_id
    sys.stderr.write("__BV_OBB_PROGRESS__" + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def send_response(result):
    """Send a final response, echoing _request_id so Electron can match it."""
    if _current_request_id:
        result = {**result, "_request_id": _current_request_id}
    send(result)


def _as_json_value(value):
    """Convert NumPy/Torch metric values into JSON-safe Python values."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
        if not isinstance(value, list):
            return _as_json_value(value)
        return [_as_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _metric_float(value):
    value = _as_json_value(value)
    if isinstance(value, bool) or isinstance(value, (list, dict)) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


_OBB_PROMOTION_METRICS = ("map50_95", "map50")


def _first_obb_promotion_metric(metrics):
    if not isinstance(metrics, dict):
        return None, None
    for metric_name in _OBB_PROMOTION_METRICS:
        score = _metric_float(metrics.get(metric_name))
        if score is not None:
            return metric_name, score
    return None, None


def _select_common_obb_promotion_metric(candidate_metrics, baseline_metrics):
    """Select one identically named metric available on both OBB runs."""
    if not isinstance(candidate_metrics, dict) or not isinstance(baseline_metrics, dict):
        return None, None, None
    for metric_name in _OBB_PROMOTION_METRICS:
        candidate_score = _metric_float(candidate_metrics.get(metric_name))
        baseline_score = _metric_float(baseline_metrics.get(metric_name))
        if candidate_score is not None and baseline_score is not None:
            return metric_name, candidate_score, baseline_score
    return None, None, None


def _normalize_obb_cohort(value):
    """Return the auditable, stable subset of an exported cohort identity."""
    if not isinstance(value, dict):
        return None
    sha256 = str(value.get("sha256") or value.get("hash") or "").strip().lower()
    revision = str(value.get("revision") or "").strip()
    split_profile_key = value.get("split_profile_key", value.get("splitProfileKey"))
    export_manifest_sha256 = str(
        value.get("export_manifest_sha256")
        or value.get("exportManifestSha256")
        or ""
    ).strip().lower()
    if not sha256 and not revision:
        return None
    try:
        sample_count = int(value.get("sample_count", value.get("sampleCount", 0)))
    except (TypeError, ValueError):
        sample_count = 0
    try:
        group_count = int(value.get("group_count", value.get("groupCount", 0)))
    except (TypeError, ValueError):
        group_count = 0
    try:
        format_version = int(value.get("format_version", value.get("formatVersion", 1)))
    except (TypeError, ValueError):
        format_version = 1
    try:
        expected_class_count = int(
            value.get("expected_class_count", value.get("expectedClassCount", 0))
        )
    except (TypeError, ValueError):
        expected_class_count = 0
    raw_class_histogram = value.get(
        "real_class_histogram",
        value.get("realClassHistogram", {}),
    )
    parsed_class_histogram = {}
    if isinstance(raw_class_histogram, dict):
        for raw_class_id, raw_count in raw_class_histogram.items():
            try:
                class_id = int(raw_class_id)
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if class_id < 0:
                continue
            parsed_class_histogram[str(class_id)] = max(0, count)
    normalized_class_histogram = {
        str(class_id): parsed_class_histogram.get(str(class_id), 0)
        for class_id in range(max(0, expected_class_count))
    }
    return {
        "formatVersion": format_version,
        "revision": revision or None,
        "sha256": sha256 or None,
        "exportManifestSha256": export_manifest_sha256 or None,
        "splitProfileKey": str(split_profile_key) if split_profile_key is not None else None,
        "sampleCount": max(0, sample_count),
        "groupCount": max(0, group_count),
        "expectedClassCount": max(0, expected_class_count),
        "realClassHistogram": normalized_class_histogram,
        "frozen": bool(value.get("frozen", False)),
        "reportOnly": bool(value.get("report_only", value.get("reportOnly", False))),
    }


def _normalize_obb_validation_cohort(value):
    """Backward-compatible name for validation cohort normalization."""
    return _normalize_obb_cohort(value)


def _obb_validation_class_coverage_complete(cohort):
    if not isinstance(cohort, dict):
        return False
    try:
        expected_class_count = int(cohort.get("expectedClassCount", 0))
    except (TypeError, ValueError):
        expected_class_count = 0
    # V1/migrated single-class cohorts did not persist a histogram. Preserve
    # their comparability; current directional/bilateral exports always declare 2.
    if expected_class_count <= 1:
        return True
    histogram = cohort.get("realClassHistogram")
    if not isinstance(histogram, dict):
        return False
    return all(
        int(histogram.get(str(class_id), 0) or 0) > 0
        for class_id in range(expected_class_count)
    )


def _obb_validation_cohorts_match(candidate, baseline):
    if not candidate or not baseline or not candidate.get("frozen") or not baseline.get("frozen"):
        return False
    try:
        candidate_format = int(candidate.get("formatVersion", 1))
    except (TypeError, ValueError):
        candidate_format = 1
    try:
        baseline_format = int(baseline.get("formatVersion", 1))
    except (TypeError, ValueError):
        baseline_format = 1
    if candidate_format >= 2 or baseline_format >= 2:
        candidate_material_hash = candidate.get("exportManifestSha256")
        baseline_material_hash = baseline.get("exportManifestSha256")
        if not (
            candidate_material_hash
            and baseline_material_hash
            and candidate_material_hash == baseline_material_hash
        ):
            return False
    candidate_hash = candidate.get("sha256")
    baseline_hash = baseline.get("sha256")
    if candidate_hash and baseline_hash:
        return candidate_hash == baseline_hash
    candidate_revision = candidate.get("revision")
    baseline_revision = baseline.get("revision")
    return bool(candidate_revision and candidate_revision == baseline_revision)


def _build_obb_evaluator_protocol(*, trainer_args, imgsz, batch, nms_iou, amp=False):
    """Capture every setting that can change Ultralytics OBB validation metrics."""
    def arg(name, fallback):
        if isinstance(trainer_args, dict):
            return trainer_args.get(name, fallback)
        return getattr(trainer_args, name, fallback) if trainer_args is not None else fallback

    def int_arg(name, fallback):
        value = arg(name, fallback)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(fallback)

    try:
        import ultralytics as _ultralytics
        evaluator_version = str(getattr(_ultralytics, "__version__", "unknown"))
    except Exception:
        evaluator_version = "unknown"

    confidence = _metric_float(arg("conf", None))
    return {
        "formatVersion": 1,
        "evaluator": "ultralytics",
        "evaluatorVersion": evaluator_version,
        "task": "obb",
        "split": str(arg("split", "val")),
        "imageSize": int_arg("imgsz", imgsz),
        "batchSize": int_arg("batch", batch),
        "nmsIou": float(arg("iou", nms_iou)),
        "confidenceThreshold": confidence,
        "maxDetections": int_arg("max_det", 300),
        "agnosticNms": bool(arg("agnostic_nms", False)),
        "singleClass": bool(arg("single_cls", False)),
        "rectangularBatches": bool(arg("rect", False)),
        "halfPrecision": bool(arg("half", False)),
        "mixedPrecision": bool(arg("amp", amp)),
        "ttaAugment": bool(arg("augment", False)),
        "dnnBackend": bool(arg("dnn", False)),
        "classFilter": _as_json_value(arg("classes", None)),
        "seed": int_arg("seed", 0),
        "deterministic": bool(arg("deterministic", True)),
        "metricPriority": list(_OBB_PROMOTION_METRICS),
    }


def _obb_evaluator_protocol_fingerprint(protocol):
    if not isinstance(protocol, dict):
        return None
    return lineage.sha256_json(protocol)


def _obb_evaluator_protocols_match(
    candidate_protocol,
    candidate_fingerprint,
    baseline_protocol,
    baseline_fingerprint,
):
    candidate_computed = _obb_evaluator_protocol_fingerprint(candidate_protocol)
    baseline_computed = _obb_evaluator_protocol_fingerprint(baseline_protocol)
    return bool(
        candidate_computed
        and baseline_computed
        and candidate_fingerprint == candidate_computed
        and baseline_fingerprint == baseline_computed
        and candidate_computed == baseline_computed
    )


def _obb_evaluation_kwargs_from_protocol(
    protocol,
    *,
    dataset_yaml,
    split,
    workers,
    device,
):
    """Translate a pinned protocol into explicit Ultralytics validation args."""
    kwargs = {
        "data": dataset_yaml,
        "split": str(split),
        "imgsz": int(protocol["imageSize"]),
        "batch": int(protocol["batchSize"]),
        "workers": int(workers),
        "device": device,
        "iou": float(protocol["nmsIou"]),
        "max_det": int(protocol["maxDetections"]),
        "agnostic_nms": bool(protocol["agnosticNms"]),
        "single_cls": bool(protocol["singleClass"]),
        "rect": bool(protocol["rectangularBatches"]),
        "half": bool(protocol["halfPrecision"]),
        "augment": bool(protocol["ttaAugment"]),
        "dnn": bool(protocol["dnnBackend"]),
        "seed": int(protocol["seed"]),
        "deterministic": bool(protocol["deterministic"]),
        "task": "obb",
        "plots": False,
        "verbose": False,
    }
    if protocol.get("confidenceThreshold") is not None:
        kwargs["conf"] = float(protocol["confidenceThreshold"])
    if protocol.get("classFilter") is not None:
        kwargs["classes"] = protocol["classFilter"]
    return kwargs


def _publish_obb_registry_and_aliases(
    registry_path,
    registry_payload,
    *,
    promoted,
    artifact_path,
    artifact_config_path,
    model_alias_path,
    config_alias_path,
):
    """Publish an OBB registry update without leaving aliases half-updated.

    Candidate-only updates do not touch aliases. For a promotion, both current
    aliases are snapshotted and replaced before the registry is committed. Any
    copy/write exception restores the prior pair (or removes newly created
    aliases), while the old atomic registry remains authoritative.
    """
    if not promoted:
        lineage.atomic_write_json(registry_path, registry_payload)
        return

    import tempfile

    models_dir = os.path.dirname(os.path.abspath(registry_path))
    backup_dir = tempfile.mkdtemp(prefix=".obb-publish-", dir=models_dir)
    snapshots = {}
    try:
        for index, alias_path in enumerate((model_alias_path, config_alias_path)):
            if os.path.isfile(alias_path):
                snapshot_path = os.path.join(backup_dir, f"alias-{index}.bak")
                shutil.copy2(alias_path, snapshot_path)
                snapshots[alias_path] = snapshot_path
            else:
                snapshots[alias_path] = None

        try:
            lineage.atomic_copy_file(artifact_path, model_alias_path)
            lineage.atomic_copy_file(artifact_config_path, config_alias_path)
            lineage.atomic_write_json(registry_path, registry_payload)
        except Exception as publish_error:
            rollback_errors = []
            for alias_path, snapshot_path in snapshots.items():
                try:
                    if snapshot_path is None:
                        if os.path.exists(alias_path):
                            os.remove(alias_path)
                    else:
                        lineage.atomic_copy_file(snapshot_path, alias_path)
                except Exception as rollback_error:
                    rollback_errors.append(f"{alias_path}: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    "OBB publication failed and alias rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from publish_error
            raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _load_and_validate_obb_registry(session_dir, *, allowed_unregistered_run_ids=()):
    """Load the v2 OBB registry strictly, without inventing lost history."""
    models_dir = os.path.join(os.path.abspath(session_dir), "models")
    registry_path = os.path.join(models_dir, "obb_registry.json")
    allowed_run_ids = {
        str(run_id).strip().casefold()
        for run_id in allowed_unregistered_run_ids
        if str(run_id).strip()
    }

    if not os.path.exists(registry_path):
        immutable_root = os.path.join(models_dir, "runs", "obb")
        orphan_run_ids = []
        if os.path.isdir(immutable_root):
            orphan_run_ids = sorted(
                entry.name
                for entry in os.scandir(immutable_root)
                if entry.is_dir()
                and entry.name.strip().casefold() not in allowed_run_ids
            )
        if orphan_run_ids:
            raise RuntimeError(
                "OBB registry is missing while immutable OBB run directories already exist: "
                f"{orphan_run_ids}. Recover the registry explicitly; training will not reset "
                "or overwrite model history."
            )
        return {"version": 2, "models": []}

    if not os.path.isfile(registry_path):
        raise RuntimeError(f"OBB registry path is not a file: {registry_path}")
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse existing OBB registry '{registry_path}': {exc}. "
            "Repair it explicitly before training."
        ) from exc

    if not isinstance(registry, dict) or registry.get("version") != 2:
        raise RuntimeError(
            f"Invalid OBB registry '{registry_path}': expected a version 2 object."
        )
    models = registry.get("models")
    if not isinstance(models, list):
        raise RuntimeError(
            f"Invalid OBB registry '{registry_path}': models must be an array."
        )

    model_ids = set()
    run_ids = set()
    active_count = 0
    for index, record in enumerate(models):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Invalid OBB registry '{registry_path}': model record {index} is not an object."
            )
        model_id = str(record.get("modelId") or "").strip()
        run_id = str(record.get("runId") or "").strip()
        artifact_path = str(record.get("path") or "").strip()
        status = str(record.get("status") or "").strip().lower()
        if not model_id or not run_id or not artifact_path:
            raise RuntimeError(
                f"Invalid OBB registry '{registry_path}': model record {index} is missing "
                "modelId, runId, or path."
            )
        if status not in {"active", "candidate", "deprecated"}:
            raise RuntimeError(
                f"Invalid OBB registry '{registry_path}': model record {index} has "
                f"unsupported status '{status}'."
            )
        normalized_model_id = model_id.casefold()
        normalized_run_id = run_id.casefold()
        if normalized_model_id in model_ids:
            raise RuntimeError(
                f"Invalid OBB registry '{registry_path}': duplicate modelId '{model_id}'."
            )
        if normalized_run_id in run_ids:
            raise RuntimeError(
                f"Invalid OBB registry '{registry_path}': duplicate runId '{run_id}'."
            )
        model_ids.add(normalized_model_id)
        run_ids.add(normalized_run_id)
        if status == "active":
            active_count += 1
    if active_count > 1:
        raise RuntimeError(
            f"Invalid OBB registry '{registry_path}': found {active_count} active models; "
            "at most one active OBB model is allowed."
        )
    return registry


def _extract_yolo_metrics(source, class_names=None):
    """Normalize available Ultralytics OBB validation metrics.

    Ultralytics has changed metric wrapper types and key spelling between
    releases.  This adapter intentionally accepts the result of ``train()``, a
    validator metrics object, or a simple dict used by mocked/offline runs.
    """
    if source is None:
        return {}
    if isinstance(source, (list, tuple)):
        merged = {}
        for item in source:
            current = _extract_yolo_metrics(item, class_names=class_names)
            for key, value in current.items():
                if key == "raw":
                    merged.setdefault("raw", {}).update(value)
                elif key not in merged or merged[key] in (None, [], {}):
                    merged[key] = value
        return merged

    raw_source = source if isinstance(source, dict) else getattr(source, "results_dict", None)
    raw = _as_json_value(raw_source) if isinstance(raw_source, dict) else {}

    def from_raw(*aliases):
        lowered = {str(key).lower(): value for key, value in raw.items()}
        for alias in aliases:
            if alias.lower() in lowered:
                return _metric_float(lowered[alias.lower()])
        return None

    metric_box = getattr(source, "box", None)
    if metric_box is None:
        metric_box = getattr(source, "obb", None)

    def from_attr(*names):
        for name in names:
            value = getattr(metric_box, name, None) if metric_box is not None else None
            numeric = _metric_float(value)
            if numeric is not None:
                return numeric
            value = getattr(source, name, None) if not isinstance(source, dict) else None
            numeric = _metric_float(value)
            if numeric is not None:
                return numeric
        return None

    def prefer(primary, fallback):
        return primary if primary is not None else fallback

    result = {
        "map50": prefer(from_raw("metrics/mAP50(B)", "metrics/mAP50", "map50"), from_attr("map50")),
        "map50_95": prefer(from_raw("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "map"), from_attr("map", "map50_95")),
        "precision": prefer(from_raw("metrics/precision(B)", "metrics/precision", "precision"), from_attr("mp", "precision")),
        "recall": prefer(from_raw("metrics/recall(B)", "metrics/recall", "recall"), from_attr("mr", "recall")),
    }

    def metric_array(*names):
        for name in names:
            value = getattr(metric_box, name, None) if metric_box is not None else None
            plain = _as_json_value(value)
            if isinstance(plain, list):
                return plain
        return []

    per_precision = metric_array("p")
    per_recall = metric_array("r")
    per_map50 = metric_array("ap50")
    per_map = metric_array("maps", "ap")
    class_indices = metric_array("ap_class_index")
    per_class_count = max(len(per_precision), len(per_recall), len(per_map50), len(per_map))
    names = class_names if class_names is not None else getattr(source, "names", None)
    per_class = []
    for offset in range(per_class_count):
        class_id = int(class_indices[offset]) if offset < len(class_indices) else offset
        if isinstance(names, dict):
            class_name = names.get(class_id, str(class_id))
        elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            class_name = names[class_id]
        else:
            class_name = str(class_id)
        entry = {"class_id": class_id, "class_name": str(class_name)}
        for key, values in (
            ("precision", per_precision),
            ("recall", per_recall),
            ("map50", per_map50),
            ("map50_95", per_map),
        ):
            if offset < len(values):
                metric_value = values[offset]
                if key == "map50_95" and isinstance(metric_value, list):
                    finite_values = [
                        numeric
                        for numeric in (_metric_float(item) for item in metric_value)
                        if numeric is not None
                    ]
                    metric_value = sum(finite_values) / len(finite_values) if finite_values else None
                numeric = _metric_float(metric_value)
                if numeric is not None:
                    entry[key] = numeric
        per_class.append(entry)

    result = {key: value for key, value in result.items() if value is not None}
    if per_class:
        result["per_class"] = per_class
    if raw:
        result["raw"] = raw
    return result


def _resolve_obb_training_epochs(epochs, device):
    """Use hardware defaults only when epochs is omitted."""
    if epochs is None:
        return 100 if device in ("cuda", "mps") else 30
    try:
        resolved = int(epochs)
    except (TypeError, ValueError) as exc:
        raise ValueError("OBB training epochs must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError("OBB training epochs must be a positive integer")
    return resolved


class SuperAnnotator:
    def __init__(self):
        self.yolo_model = None
        self.sam2_model = None
        self.dlib_predictor = None
        self.dlib_id_mapping = None
        self.dlib_model_path = None
        self.dlib_orientation_policy = None
        self.dlib_runtime_immutable = False
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
    def _resolve_detection_preset(conf_threshold, nms_iou, max_objects, detection_preset, task="generic", imgsz=None):
        resolved_task = str(task or "generic").strip().lower()
        if resolved_task == "obb":
            return resolve_obb_detection_preset(
                conf_threshold=conf_threshold,
                nms_iou=nms_iou,
                max_objects=max_objects,
                detection_preset=detection_preset,
                imgsz=imgsz,
            )

        preset = (detection_preset or "balanced").strip().lower()
        conf = float(conf_threshold)
        iou = float(nms_iou)
        top_k = int(max_objects)
        imgsz = int(imgsz) if imgsz is not None else 1280
        allow_relaxed_retry = True

        if preset == "custom":
            conf = max(0.01, min(conf, 0.99))
            iou = max(0.05, min(iou, 0.95))
            top_k = max(1, min(top_k, 250))
            imgsz = 1280 if imgsz >= 1280 else 960 if imgsz >= 960 else 640
            allow_relaxed_retry = False
        elif preset == "precision":
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


    def save_segments_for_boxes(self, image_path, boxes, session_dir, iterative=False, expand_ratio=0.10, allow_rectangle_fallback=True):
        """Save SAM2 mask crops to session_dir/segments/ for each accepted box.

        Called by Electron after the user finalizes accepted boxes so that
        the OBB synthetic data generator can find the segment files.
        """
        import hashlib
        import json as _json

        if not boxes:
            return {"status": "ok", "saved": 0, "requested": 0, "details": []}

        image = cv2.imread(image_path)
        if image is None:
            return {
                "status": "error",
                "error": f"could_not_load_image:{image_path}",
                "saved": 0,
                "requested": len(boxes),
                "details": [],
            }
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
        details = []
        for idx, box_xyxy in enumerate(boxes):
            x1 = max(0, int(box_xyxy[0]))
            y1 = max(0, int(box_xyxy[1]))
            x2 = min(img_w, int(box_xyxy[2]))
            y2 = min(img_h, int(box_xyxy[3]))
            if x2 <= x1 or y2 <= y1:
                details.append({"index": idx, "status": "failed", "reason": "invalid_or_empty_crop"})
                continue

            mask = cached_lookup.get((x1, y1, x2, y2))
            save_x1, save_y1, save_x2, save_y2 = x1, y1, x2, y2
            mask_source = "cached_sam2" if mask is not None else None
            failure_reason = None

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
                        mask_source = "sam2_iterative"
                    else:
                        results = self.sam2_model.predict(
                            image, bboxes=[[x1, y1, x2, y2]], verbose=False)
                        mask = (results[0].masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
                        mask_source = "sam2"
                except Exception as e:
                    logger.warning(f"SAM2 failed for box {idx}: {e}")
                    failure_reason = f"sam2_inference_failed:{e}"
            elif mask is None and self.sam2_model is None:
                failure_reason = "sam2_unavailable"

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
            else:
                mask = None

            if mask is None:
                if not allow_rectangle_fallback:
                    details.append({
                        "index": idx,
                        "status": "failed",
                        "maskSource": mask_source,
                        "reason": failure_reason or "no_usable_sam_mask",
                    })
                    continue
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 1
                mask_source = "rectangle_fallback"
                save_x1, save_y1, save_x2, save_y2 = x1, y1, x2, y2

            crop_img  = image[save_y1:save_y2, save_x1:save_x2]
            crop_mask = mask[save_y1:save_y2, save_x1:save_x2]
            if crop_img.size == 0:
                details.append({
                    "index": idx,
                    "status": "failed",
                    "maskSource": mask_source,
                    "reason": "empty_crop_after_geometry",
                })
                continue

            if crop_mask.shape != crop_img.shape[:2]:
                crop_mask = cv2.resize(
                    crop_mask, (crop_img.shape[1], crop_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST)

            crop_mask = self._normalize_segment_mask_polarity(crop_mask)
            if crop_mask is None or np.count_nonzero(crop_mask) < 20:
                details.append({
                    "index": idx,
                    "status": "failed",
                    "maskSource": mask_source,
                    "reason": "mask_too_small_after_normalization",
                })
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
            details.append({"index": idx, "status": "saved", "maskSource": mask_source})

        logger.info(
            f"save_segments_for_boxes: saved {saved}/{len(boxes)} segments Ã¢â€ â€™ {seg_dir}")
        return {"status": "ok", "saved": saved, "requested": len(boxes), "details": details}

    # ------------------------------------------------------------------
    # Stage B: Normalization (The "Standardizer")
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stage C: Dlib landmark prediction
    # ------------------------------------------------------------------
    def load_dlib_model(self, model_path, id_mapping_path=None):
        """Load dlib plus its exact immutable metadata, or a legacy alias pair."""
        requested_model_path = os.path.abspath(model_path)
        basename = os.path.basename(requested_model_path)
        project_root = None
        tag = None
        if basename.startswith("predictor_") and basename.endswith(".dat"):
            project_root = os.path.dirname(os.path.dirname(requested_model_path))
            tag = basename[len("predictor_"):-len(".dat")]
        elif basename == "predictor.dat":
            artifact_dir = os.path.dirname(requested_model_path)
            run_id = os.path.basename(artifact_dir)
            type_dir = os.path.dirname(artifact_dir)
            if os.path.basename(type_dir).lower() == "dlib":
                models_dir = os.path.dirname(os.path.dirname(type_dir))
                project_root = os.path.dirname(models_dir)
                tag = f"dlib:{run_id}"

        runtime = None
        if project_root and tag:
            runtime = resolve_landmark_runtime(
                project_root,
                tag,
                "dlib",
                allow_legacy=True,
            )
        resolved_model_path = (
            runtime["model_path"] if isinstance(runtime, dict) else requested_model_path
        )
        if self.dlib_model_path == resolved_model_path and self.dlib_predictor is not None:
            return  # already loaded

        import dlib
        self.dlib_predictor = dlib.shape_predictor(resolved_model_path)
        self.dlib_model_path = resolved_model_path

        self.dlib_id_mapping = None
        self.dlib_orientation_policy = None
        self.dlib_runtime_immutable = bool(runtime and runtime.get("immutable"))
        if runtime:
            raw = runtime.get("id_mapping") or {}
            if (
                not raw
                and not runtime.get("immutable")
                and id_mapping_path
                and os.path.exists(id_mapping_path)
            ):
                try:
                    with open(id_mapping_path, "r") as f:
                        loaded = json.load(f)
                    raw = loaded if isinstance(loaded, dict) else {}
                except Exception:
                    raw = {}
            mapping = (
                raw.get("dlib_index_to_original")
                or raw.get("dlib_to_original")
                or raw
            )
            self.dlib_id_mapping = {int(k): int(v) for k, v in mapping.items()}
            if runtime.get("immutable"):
                training_config = raw.get("training_config", {})
                self.dlib_orientation_policy = dict(
                    training_config["orientation_policy"]
                )
        elif id_mapping_path and os.path.exists(id_mapping_path):
            try:
                with open(id_mapping_path, "r") as f:
                    raw = json.load(f)
                mapping = (
                    raw.get("dlib_index_to_original", raw)
                    if isinstance(raw, dict)
                    else {}
                )
                self.dlib_id_mapping = {int(k): int(v) for k, v in mapping.items()}
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
    def detect_finetuned(self, image, finetuned_path, class_name, conf_threshold=None, top_k=10, nms_iou=None, imgsz=640):
        """Run detection with a fine-tuned YOLOv8 OBB model."""
        from ultralytics import YOLO
        if nms_iou is None:
            nms_iou = load_obb_nms_iou(finetuned_path, default=0.3)
        if conf_threshold is None:
            conf_threshold = load_obb_confidence_threshold(finetuned_path, default=0.3)
        ft_model = YOLO(finetuned_path)
        results = ft_model.predict(
            image,
            conf=conf_threshold,
            iou=float(nms_iou),
            imgsz=int(imgsz),
            task="obb",
            agnostic_nms=True,
            verbose=False,
        )

        boxes = []
        r = results[0]
        if r.obb is not None and len(r.obb):
            for parsed in iter_ultralytics_obb(r):
                corners = parsed["corners"]
                xs = [p[0] for p in corners]
                ys = [p[1] for p in corners]
                boxes.append({
                    "xyxy": [min(xs), min(ys), max(xs), max(ys)],
                    "confidence": round(parsed["confidence"], 3),
                    "class_name": class_name,
                    "obb_corners": corners,
                    "class_id": parsed["class_id"],
                    "angle": parsed["angle"],
                })
        else:
            raise RuntimeError(f"OBB detector returned no oriented boxes: {finetuned_path}")

        # Ultralytics has already applied oriented NMS for this OBB result.
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
            agnostic_nms=True,
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

        # Ultralytics has already applied class-agnostic NMS. A second AABB
        # suppression pass would discard valid nearby specimens.
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
        requested_nms_iou = options.get("nms_iou")
        requested_imgsz = options.get("imgsz")
        finetuned_model = options.get("finetuned_model")
        orientation_policy = options.get("orientation_policy") or {}
        orientation_schema = str(orientation_policy.get("mode", "invariant")).strip().lower()
        use_obb_detector = bool(finetuned_model and os.path.exists(finetuned_model))
        detection_preset = options.get("detection_preset")
        if detection_preset is None:
            detection_preset = "custom" if use_obb_detector else "balanced"
        if use_obb_detector and "conf_threshold" not in options:
            conf_threshold = load_obb_confidence_threshold(
                finetuned_model,
                default=conf_threshold,
            )
        if use_obb_detector and "nms_iou" not in options:
            requested_nms_iou = load_obb_nms_iou(
                finetuned_model,
                default=0.3,
            )
        elif requested_nms_iou is None:
            requested_nms_iou = 0.3
        resolved = self._resolve_detection_preset(
            conf_threshold=conf_threshold,
            nms_iou=requested_nms_iou,
            max_objects=max_objects,
            detection_preset=detection_preset,
            task="obb" if use_obb_detector else "generic",
            imgsz=requested_imgsz,
        )
        conf_threshold = resolved["conf"]
        max_objects = resolved["top_k"]

        # Load image
        send_progress("Loading image...", 5, "detection")
        image = self._load_image(image_path)
        img_h, img_w = image.shape[:2]

        send_progress("Detecting objects...", 15, "detection")
        if use_obb_detector:
            obb_nms_iou = float(resolved["iou"])
            if options.get("nms_iou") is None:
                obb_nms_iou = load_obb_nms_iou(finetuned_model, default=obb_nms_iou)

            boxes = self.detect_finetuned(
                image,
                finetuned_model,
                class_name,
                conf_threshold,
                top_k=max_objects,
                nms_iou=obb_nms_iou,
                imgsz=resolved["imgsz"],
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
                if self.dlib_runtime_immutable:
                    # Exact training-time orientation semantics travel with the
                    # model.  Never merge immutable inference with session.json.
                    orientation_policy = dict(self.dlib_orientation_policy or {})
                    orientation_schema = str(
                        orientation_policy.get("mode", "invariant")
                    ).strip().lower()
            except ImmutableLandmarkArtifactError:
                # A registry-selected immutable run must not degrade to its
                # mutable alias/debug files after integrity verification fails.
                raise
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
                bilateral_axis = str(
                    (orientation_policy or {}).get("bilateralClassAxis", "")
                ).strip().lower()
                if orientation_schema == "bilateral" and bilateral_axis == "vertical_obb":
                    orientation = "up" if int(class_id) == 0 else "down"
                else:
                    orientation = "left" if int(class_id) == 0 else "right"
                orientation_hint = {
                    "orientation": orientation,
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
    # OBB dataset export
    # ------------------------------------------------------------------
    def export_obb_dataset(
        self,
        session_dir,
        generate_synthetic=True,
        orientation_schema="invariant",
        progress_callback=None,
        seed=42,
    ):
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
        result = _export_obb(
            session_dir,
            generate_synthetic=generate_synthetic,
            orientation_schema=orientation_schema,
            progress_callback=progress_callback,
            seed=int(seed),
        )
        return result

    # ------------------------------------------------------------------
    # OBB detector training
    # ------------------------------------------------------------------
    def train_yolo_obb(self, session_dir, epochs=None, model_tier="nano",
                       device="cpu", sam2_enabled=True,
                       batch=None, imgsz=None,
                       iou_loss=0.3, cls_loss=1.5, box_loss=5.0,
                       orientation_schema="invariant", seed=42):
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
        from bv_utils.orientation_utils import (
            require_explicit_orientation_policy,
            resolve_session_augmentation_profile,
        )

        require_explicit_orientation_policy(session_dir)
        resolved_seed = int(seed)
        models_dir = os.path.join(session_dir, "models")
        obb_registry_path = os.path.join(models_dir, "obb_registry.json")
        obb_registry = _load_and_validate_obb_registry(session_dir)
        obb_registry_revision = lineage.sha256_json(obb_registry)
        prior_runs = [dict(record) for record in obb_registry["models"]]

        # Hardware-routed hyperparameters
        if device in ("cuda", "mps"):
            default_batch = 16
        else:  # cpu
            # Capped between 4-8 Ã¢â‚¬â€ YOLO batch=-1 autotune can be dangerous on CPU
            default_batch = 6

        resolved_epochs = _resolve_obb_training_epochs(epochs, device)
        resolved_batch = int(batch) if batch is not None else default_batch
        resolved_imgsz = int(imgsz) if imgsz is not None else 640
        cpu_count = max(1, int(os.cpu_count() or 1))
        is_windows = sys.platform.startswith("win")
        resolved_amp = bool(device == "cuda" and not is_windows)
        resolved_plots = True
        if is_windows:
            if device == "cuda":
                resolved_workers = 0
            else:
                resolved_workers = 0
        elif device == "cuda":
            resolved_workers = min(8, max(2, cpu_count // 2))
        elif device == "mps":
            resolved_workers = min(4, max(1, cpu_count // 4))
        else:
            resolved_workers = 0

        logger.info(
            "OBB training profile: device=%s, platform=%s, epochs=%d, batch=%d, imgsz=%d, workers=%d, amp=%s, plots=%s, sam2=%s",
            device, sys.platform, resolved_epochs, resolved_batch, resolved_imgsz, resolved_workers, resolved_amp, resolved_plots, sam2_enabled,
        )

        if sam2_enabled:
            send_obb_progress(
                "SAM2 enabled: using existing segment pool only...",
                3,
                "training",
                {"workers": resolved_workers, "device": device, "platform": sys.platform, "amp_enabled": resolved_amp},
            )
        else:
            send_obb_progress(
                "SAM2 disabled: training without synthetic segment augmentation...",
                3,
                "training",
                {"workers": resolved_workers, "device": device, "platform": sys.platform, "amp_enabled": resolved_amp},
            )

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
        send_obb_progress("Exporting OBB dataset...", 5, "training")

        last_export_progress_at = 0.0

        def on_export_progress(message, percent, details=None):
            nonlocal last_export_progress_at
            now = time.time()
            pct = max(5, min(9, int(percent)))
            if pct < 9 and now - last_export_progress_at < 0.5:
                return
            last_export_progress_at = now
            merged_details = dict(details or {})
            merged_details.setdefault("workers", resolved_workers)
            merged_details.setdefault("device", device)
            merged_details.setdefault("platform", sys.platform)
            merged_details.setdefault("amp_enabled", resolved_amp)
            send_obb_progress(message, pct, "export", merged_details)

        export_result = self.export_obb_dataset(
            session_dir,
            generate_synthetic=sam2_enabled,
            orientation_schema=orientation_schema,
            progress_callback=on_export_progress,
            seed=resolved_seed,
        )
        if not export_result.get("ok"):
            return {"status": "error", "error": export_result.get("error", "OBB dataset export failed")}

        warnings = list(export_result.get("warnings", []))
        synthetic_stats = export_result.get("synthetic", {}) if isinstance(export_result.get("synthetic"), dict) else {}
        if sam2_enabled and int(synthetic_stats.get("segments_total", 0)) <= 0:
            warnings.append("No usable finalized SAM2 segments found; continued with finalized OBB data only.")

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
        training_input_guard = _capture_obb_training_input_guard(export_result)
        os.makedirs(models_dir, exist_ok=True)

        base_map = {
            "nano": "yolov8n-obb.pt",
            "small": "yolov8s-obb.pt",
            "medium": "yolov8m-obb.pt",
            "large": "yolov8l-obb.pt",
        }
        base_model = base_map.get(model_tier, "yolov8n-obb.pt")
        freeze_layers = 14 if model_tier == "nano" else 0
        resolved_patience = 20 if device in ("cuda", "mps") else 10

        send_obb_progress(
            f"Preparing YOLOv8-OBB training runtime ({model_tier}, {device})...",
            10,
            "pretrain_setup",
            {
                "workers": resolved_workers,
                "device": device,
                "platform": sys.platform,
                "amp_enabled": resolved_amp,
            },
        )
        from ultralytics import YOLO
        model = YOLO(base_model)
        base_checkpoint_path = None
        for candidate in (getattr(model, "ckpt_path", None), base_model):
            try:
                candidate_path = os.path.abspath(os.fspath(candidate))
            except (TypeError, ValueError):
                continue
            if os.path.isfile(candidate_path):
                base_checkpoint_path = candidate_path
                break
        base_checkpoint = {
            "requested": base_model,
            "resolvedPath": base_checkpoint_path,
            "sha256": (
                lineage.sha256_file(base_checkpoint_path)
                if base_checkpoint_path is not None
                else None
            ),
        }

        trainer_class = None
        original_plot_training_labels = None
        original_plot_training_samples = None
        try:
            trainer_class = model._smart_load("trainer")
        except Exception:
            trainer_class = None
        if trainer_class is not None:
            original_plot_training_labels = getattr(trainer_class, "plot_training_labels", None)
            original_plot_training_samples = getattr(trainer_class, "plot_training_samples", None)
            if callable(original_plot_training_labels):
                def _skip_plot_training_labels(self):
                    return None
                trainer_class.plot_training_labels = _skip_plot_training_labels
            if callable(original_plot_training_samples):
                def _skip_plot_training_samples(self, batch, ni):
                    return None
                trainer_class.plot_training_samples = _skip_plot_training_samples

        train_started_at = time.time()
        last_batch_heartbeat_at = 0.0

        def _safe_float(value):
            try:
                if value is None:
                    return None
                if hasattr(value, "item"):
                    return float(value.item())
                return float(value)
            except Exception:
                return None

        def _extract_loss(trainer):
            candidates = []
            for attr_name in ("loss", "tloss", "loss_items"):
                value = getattr(trainer, attr_name, None)
                if value is None:
                    continue
                if hasattr(value, "detach"):
                    try:
                        value = value.detach().cpu().numpy()
                    except Exception:
                        pass
                if isinstance(value, (list, tuple)):
                    values = [_safe_float(v) for v in value]
                    values = [v for v in values if v is not None]
                    if values:
                        candidates.append(sum(values))
                        continue
                if hasattr(value, "tolist"):
                    try:
                        listed = value.tolist()
                    except Exception:
                        listed = None
                    if isinstance(listed, list):
                        values = [_safe_float(v) for v in listed]
                        values = [v for v in values if v is not None]
                        if values:
                            candidates.append(sum(values))
                            continue
                numeric = _safe_float(value)
                if numeric is not None:
                    candidates.append(numeric)
            return candidates[0] if candidates else None

        def _extract_lr(trainer):
            optimizer = getattr(trainer, "optimizer", None)
            param_groups = getattr(optimizer, "param_groups", None)
            if not param_groups:
                return None
            try:
                return _safe_float(param_groups[0].get("lr"))
            except Exception:
                return None

        def _send_training_heartbeat(trainer, *, message, percent_override=None):
            epoch_zero = max(0, int(getattr(trainer, "epoch", 0)))
            total_epochs = max(1, int(getattr(trainer, "epochs", resolved_epochs) or resolved_epochs))
            batches_total = max(1, int(len(getattr(trainer, "train_loader", []) or [])))
            batch_zero = max(0, int(getattr(trainer, "batch_i", 0)))
            batch_one = min(batches_total, batch_zero + 1)
            progress_ratio = ((epoch_zero * batches_total) + batch_one) / max(1, total_epochs * batches_total)
            percent = percent_override if percent_override is not None else 10 + int(80 * max(0.0, min(1.0, progress_ratio)))
            elapsed_sec = max(0.0, time.time() - train_started_at)
            eta_sec = None
            if progress_ratio > 1e-6:
                remaining_ratio = max(0.0, 1.0 - progress_ratio)
                eta_sec = int((elapsed_sec / progress_ratio) * remaining_ratio)
            details = {
                "epoch": epoch_zero + 1,
                "epochs": total_epochs,
                "batch": batch_one,
                "batches": batches_total,
                "loss": _extract_loss(trainer),
                "lr": _extract_lr(trainer),
                "elapsed_sec": int(elapsed_sec),
                "eta_sec": eta_sec,
                "workers": resolved_workers,
                "device": device,
                "platform": sys.platform,
                "amp_enabled": resolved_amp,
            }
            send_obb_progress(message, percent, "training", details)

        def on_train_start(trainer):
            _send_training_heartbeat(
                trainer,
                message="OBB training loop entered. Waiting for the first batches...",
                percent_override=12,
            )

        def on_train_epoch_end(trainer):
            epoch = trainer.epoch + 1
            total = trainer.epochs
            pct = 10 + int(80 * (epoch / total))
            _send_training_heartbeat(
                trainer,
                message=f"OBB training epoch {epoch}/{total}...",
                percent_override=pct,
            )

        def on_train_batch_end(trainer):
            nonlocal last_batch_heartbeat_at
            now = time.time()
            if now - last_batch_heartbeat_at < 15.0:
                return
            last_batch_heartbeat_at = now
            _send_training_heartbeat(
                trainer,
                message=(
                    f"OBB training epoch {int(getattr(trainer, 'epoch', 0)) + 1}/"
                    f"{int(getattr(trainer, 'epochs', resolved_epochs) or resolved_epochs)}..."
                ),
            )

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_train_batch_end", on_train_batch_end)

        output_dir = os.path.join(session_dir, "models", "obb_training")
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8]
        run_name = f"session_obb_{run_id}"
        try:
            train_result = model.train(
                data=dataset_yaml,
                epochs=resolved_epochs,
                imgsz=resolved_imgsz,
                batch=resolved_batch,
                workers=resolved_workers,
                device=device,
                freeze=freeze_layers,
                project=output_dir,
                name=run_name,
                exist_ok=False,
                verbose=False,
                task="obb",
                fliplr=0.0,         # horizontal flip corrupts orientation labels (left vs right class)
                flipud=0.0,         # biological specimens should not train on upside-down flips
                degrees=resolved_degrees,
                iou=float(iou_loss),   # NMS IoU threshold for validation during training
                cls=float(cls_loss),   # classification loss gain
                box=float(box_loss),   # box regression loss gain
                patience=resolved_patience,  # stop if no mAP50-95 gain for N epochs
                cos_lr=True,        # cosine LR decay avoids late-epoch plateau oscillation
                mosaic=0.0,         # disable mosaic entirely: pretrained fine-tune on small bio data
                close_mosaic=0,
                amp=resolved_amp,
                plots=resolved_plots,
                seed=resolved_seed,
                deterministic=True,
            )
        finally:
            if trainer_class is not None:
                if callable(original_plot_training_labels):
                    trainer_class.plot_training_labels = original_plot_training_labels
                if callable(original_plot_training_samples):
                    trainer_class.plot_training_samples = original_plot_training_samples

        trainer = getattr(model, "trainer", None)
        validator = getattr(trainer, "validator", None) if trainer is not None else None
        evaluator_protocol = _build_obb_evaluator_protocol(
            trainer_args=getattr(trainer, "args", None) if trainer is not None else None,
            imgsz=resolved_imgsz,
            batch=resolved_batch,
            nms_iou=float(iou_loss),
            amp=resolved_amp,
        )
        # The training result is the validation result used for checkpoint
        # selection and promotion. Pin that scientific role even if a library
        # default or mock reports a different split value.
        evaluator_protocol["split"] = "val"
        evaluator_protocol_fingerprint = _obb_evaluator_protocol_fingerprint(
            evaluator_protocol
        )
        metric_sources = [
            train_result,
            getattr(model, "metrics", None),
            getattr(validator, "metrics", None) if validator is not None else None,
            getattr(trainer, "metrics", None) if trainer is not None else None,
        ]
        class_names = getattr(model, "names", None)
        training_metrics = {}
        for metric_source in metric_sources:
            extracted = _extract_yolo_metrics(metric_source, class_names=class_names)
            if not extracted:
                continue
            if isinstance(extracted.get("raw"), dict):
                training_metrics.setdefault("raw", {}).update(extracted["raw"])
            for metric_name in ("map50", "map50_95", "precision", "recall", "per_class"):
                if metric_name not in training_metrics and metric_name in extracted:
                    training_metrics[metric_name] = extracted[metric_name]

        # Save NMS IoU preference alongside the run so detect_finetuned() can restore it.
        import json as _json_obb
        trainer_run_dir = os.path.join(output_dir, run_name)
        obb_config_path = os.path.join(trainer_run_dir, "obb_config.json")
        lineage.atomic_write_json(obb_config_path, {"nms_iou": float(iou_loss)})

        metrics_path = os.path.join(trainer_run_dir, "training_metrics.json")
        try:
            with open(metrics_path, "w", encoding="utf-8") as _f:
                _json_obb.dump(training_metrics, _f, indent=2, sort_keys=True, allow_nan=False)
        except Exception as exc:
            warnings.append(f"Could not persist OBB training metrics: {exc}")
            metrics_path = None

        best_pt = os.path.join(trainer_run_dir, "weights", "best.pt")
        if not os.path.exists(best_pt):
            best_pt = os.path.join(trainer_run_dir, "weights", "last.pt")
        if not os.path.exists(best_pt):
            return {"status": "error", "error": "OBB training finished but no best.pt found"}

        model_id = lineage.build_model_id("obb", run_id)
        current_registry = _load_and_validate_obb_registry(session_dir)
        if lineage.sha256_json(current_registry) != obb_registry_revision:
            raise RuntimeError(
                "OBB registry changed during training; refusing to publish a candidate against "
                "stale model history. Retry after the concurrent publication finishes."
            )
        artifact_dir = lineage.create_model_artifact_dir(session_dir, "obb", run_id)
        artifact_path = os.path.join(artifact_dir, "model.pt")
        artifact_config_path = os.path.join(artifact_dir, "obb_config.json")
        lineage.atomic_copy_file(best_pt, artifact_path)
        lineage.atomic_copy_file(obb_config_path, artifact_config_path)
        dataset_snapshot_descriptors = {}
        dataset_snapshot_sources = {
            "datasetYaml": export_result.get("yaml_path"),
            "exportManifest": export_result.get("export_manifest_path"),
            "cohortManifest": export_result.get("cohort_manifest_path"),
            "splitAssignments": export_result.get("split_assignments_path"),
            "syntheticManifest": export_result.get("synthetic_manifest_path"),
        }
        snapshot_extension = {
            "datasetYaml": ".yaml",
            "exportManifest": ".json",
            "cohortManifest": ".json",
            "splitAssignments": ".json",
            "syntheticManifest": ".json",
        }
        for snapshot_name, source_path in dataset_snapshot_sources.items():
            if not source_path or not os.path.isfile(source_path):
                continue
            relative_path = (
                f"dataset_export/{snapshot_name}{snapshot_extension[snapshot_name]}"
            )
            artifact_snapshot_path = os.path.join(
                artifact_dir,
                *relative_path.split("/"),
            )
            lineage.atomic_copy_file(source_path, artifact_snapshot_path)
            dataset_snapshot_descriptors[snapshot_name] = {
                "path": artifact_snapshot_path,
                "relativePath": relative_path,
                "sha256": lineage.sha256_file(artifact_snapshot_path),
            }
        effective_dataset = export_result.get("effective_dataset")
        effective_dataset_path = None
        effective_dataset_descriptor = None
        if isinstance(effective_dataset, dict) and effective_dataset.get("revision"):
            effective_dataset_path = os.path.join(artifact_dir, "effective_dataset.json")
            lineage.atomic_write_json(effective_dataset_path, effective_dataset)
            effective_dataset_descriptor = {
                "format": "biovision.effective-obb-dataset.v1",
                "path": effective_dataset_path,
                "relativePath": "effective_dataset.json",
                "sha256": lineage.sha256_file(effective_dataset_path),
                "revision": str(effective_dataset.get("revision")),
            }
        validation_cohort = _normalize_obb_validation_cohort(
            export_result.get("validation_cohort")
        )
        test_cohort = _normalize_obb_cohort(export_result.get("test_cohort"))
        # Calibrate runtime confidence/NMS only on the frozen validation cohort.
        # The fixed promotion metric above remains untouched, so neither this
        # sweep nor the report-only test can opportunistically inflate promotion.
        confidence_grid = [0.30, 0.45, 0.60]
        nms_grid = sorted({0.30, 0.50, 0.70, round(float(iou_loss), 6)})
        calibration_protocol = {
            "formatVersion": 1,
            "role": "validation_only_runtime_threshold_calibration",
            "objective": "aggregate_f1",
            "confidenceGrid": confidence_grid,
            "nmsIouGrid": nms_grid,
            "tieBreak": [
                "map50",
                "precision",
                "recall",
                "closest_to_runtime_defaults",
                "lower_confidence",
                "lower_nms_iou",
            ],
            "baseEvaluatorProtocolFingerprint": evaluator_protocol_fingerprint,
            "validationCohortSha256": (
                validation_cohort.get("sha256") if validation_cohort else None
            ),
        }
        calibration_protocol_fingerprint = lineage.sha256_json(calibration_protocol)
        calibration_status = "unavailable"
        calibration_error = None
        calibration_evidence = []
        selected_confidence = 0.30
        selected_nms_iou = float(iou_loss)
        selected_validation_metrics = {}
        if (
            validation_cohort
            and validation_cohort.get("frozen")
            and int(validation_cohort.get("sampleCount") or 0) > 0
        ):
            send_obb_progress(
                "Calibrating OBB runtime thresholds on frozen validation data...",
                92,
                "validation_calibration",
            )
            try:
                calibration_model = YOLO(artifact_path)
                if not callable(getattr(calibration_model, "val", None)):
                    raise RuntimeError("the installed Ultralytics runtime has no validation API")
                for confidence in confidence_grid:
                    for nms_iou in nms_grid:
                        sweep_protocol = dict(evaluator_protocol)
                        sweep_protocol["split"] = "val"
                        sweep_protocol["confidenceThreshold"] = float(confidence)
                        sweep_protocol["nmsIou"] = float(nms_iou)
                        sweep_result = calibration_model.val(
                            **_obb_evaluation_kwargs_from_protocol(
                                sweep_protocol,
                                dataset_yaml=dataset_yaml,
                                split="val",
                                workers=resolved_workers,
                                device=device,
                            )
                        )
                        sweep_metrics = _extract_yolo_metrics(
                            sweep_result,
                            class_names=getattr(calibration_model, "names", class_names),
                        )
                        precision = _metric_float(sweep_metrics.get("precision"))
                        recall = _metric_float(sweep_metrics.get("recall"))
                        objective = (
                            (2.0 * precision * recall) / (precision + recall)
                            if precision is not None
                            and recall is not None
                            and precision + recall > 0.0
                            else None
                        )
                        calibration_evidence.append(
                            {
                                "confidenceThreshold": float(confidence),
                                "nmsIou": float(nms_iou),
                                "objective": objective,
                                "metrics": sweep_metrics,
                                "evaluatorProtocolFingerprint": (
                                    _obb_evaluator_protocol_fingerprint(sweep_protocol)
                                ),
                            }
                        )
                feasible = [
                    entry for entry in calibration_evidence if entry.get("objective") is not None
                ]
                if not feasible:
                    raise RuntimeError("validation sweep returned no finite aggregate F1 values")

                def calibration_rank(entry):
                    metrics = entry.get("metrics", {})
                    confidence = float(entry["confidenceThreshold"])
                    nms_iou = float(entry["nmsIou"])
                    map50 = _metric_float(metrics.get("map50"))
                    precision = _metric_float(metrics.get("precision"))
                    recall = _metric_float(metrics.get("recall"))
                    return (
                        float(entry["objective"]),
                        map50 if map50 is not None else float("-inf"),
                        precision if precision is not None else float("-inf"),
                        recall if recall is not None else float("-inf"),
                        -(abs(confidence - 0.30) + abs(nms_iou - float(iou_loss))),
                        -confidence,
                        -nms_iou,
                    )

                selected = max(feasible, key=calibration_rank)
                selected_confidence = float(selected["confidenceThreshold"])
                selected_nms_iou = float(selected["nmsIou"])
                selected_validation_metrics = dict(selected.get("metrics") or {})
                calibration_status = "completed"
            except Exception as exc:
                calibration_status = "failed"
                calibration_error = str(exc)
                warnings.append(
                    "Could not calibrate OBB runtime thresholds on frozen validation data; "
                    f"using requested defaults: {exc}"
                )

        calibrated_validation_protocol = dict(evaluator_protocol)
        calibrated_validation_protocol["split"] = "val"
        calibrated_validation_protocol["confidenceThreshold"] = selected_confidence
        calibrated_validation_protocol["nmsIou"] = selected_nms_iou
        runtime_config = {
            "formatVersion": 2,
            "nms_iou": selected_nms_iou,
            "confidence_threshold": selected_confidence,
            "thresholdCalibration": {
                "status": calibration_status,
                "role": "validation_only",
                "promotionInfluence": "none",
                "protocol": calibration_protocol,
                "protocolFingerprint": calibration_protocol_fingerprint,
                "selectedValidationMetrics": selected_validation_metrics,
                "evidence": calibration_evidence,
                **({"error": calibration_error} if calibration_error else {}),
            },
        }
        lineage.atomic_write_json(obb_config_path, runtime_config)
        lineage.atomic_write_json(artifact_config_path, runtime_config)

        test_evaluator_protocol = dict(calibrated_validation_protocol)
        test_evaluator_protocol["split"] = "test"
        test_evaluator_protocol_fingerprint = _obb_evaluator_protocol_fingerprint(
            test_evaluator_protocol
        )
        test_metrics = {}
        test_report_status = "not_run"
        test_report_error = None
        test_report_not_run_reason = "promotion_decision_pending"
        verified_training_input_guard = _assert_obb_training_inputs_unchanged(
            export_result,
            training_input_guard,
        )
        training_input_guard_path = os.path.join(
            artifact_dir,
            "training_input_guard.json",
        )
        lineage.atomic_write_json(
            training_input_guard_path,
            verified_training_input_guard,
        )
        training_input_guard_descriptor = {
            "format": "biovision.obb-training-input-guard.v1",
            "path": training_input_guard_path,
            "relativePath": "training_input_guard.json",
            "sha256": lineage.sha256_file(training_input_guard_path),
            "revision": verified_training_input_guard["revision"],
            "postFitVerified": True,
        }
        training_hyperparameters = {
            "baseModel": base_model,
            "baseCheckpoint": base_checkpoint,
            "epochs": resolved_epochs,
            "batch": resolved_batch,
            "imgsz": resolved_imgsz,
            "workers": resolved_workers,
            "device": device,
            "seed": resolved_seed,
            "deterministic": True,
            "freezeLayers": freeze_layers,
            "degrees": resolved_degrees,
            "rotationRange": [rotation_lo, rotation_hi],
            "horizontalFlipProbability": 0.0,
            "verticalFlipProbability": 0.0,
            "validationNmsIou": float(iou_loss),
            "classificationLossGain": float(cls_loss),
            "boxLossGain": float(box_loss),
            "patience": resolved_patience,
            "cosineLearningRate": True,
            "mosaicProbability": 0.0,
            "amp": resolved_amp,
            "plots": resolved_plots,
            "orientationSchema": orientation_schema,
            "sam2Enabled": bool(sam2_enabled),
        }
        training_protocol_material = {
            "formatVersion": 1,
            "modelType": "obb",
            "effectiveDatasetRevision": (
                effective_dataset.get("revision")
                if isinstance(effective_dataset, dict)
                else None
            ),
            "trainingInputGuardRevision": verified_training_input_guard["revision"],
            "hyperparameters": training_hyperparameters,
            "evaluatorProtocol": evaluator_protocol,
            "promotionPolicy": _obb_promotion_policy(),
        }
        training_protocol = {
            **training_protocol_material,
            "revision": lineage.sha256_json(training_protocol_material),
        }
        training_protocol_path = os.path.join(artifact_dir, "training_protocol.json")
        lineage.atomic_write_json(training_protocol_path, training_protocol)
        training_protocol_descriptor = {
            "format": "biovision.obb-training-protocol.v1",
            "path": training_protocol_path,
            "relativePath": "training_protocol.json",
            "sha256": lineage.sha256_file(training_protocol_path),
            "revision": training_protocol["revision"],
        }
        split_paths = [
            descriptor["path"]
            for snapshot_name, descriptor in dataset_snapshot_descriptors.items()
            if snapshot_name in {"exportManifest", "splitAssignments", "cohortManifest"}
        ]
        active_prior = next(
            (prior for prior in prior_runs if isinstance(prior, dict) and prior.get("status") == "active"),
            None,
        )
        baseline_model_id = active_prior.get("modelId") if active_prior else None
        lineage_payload = lineage.build_run_lineage(
            session_dir,
            split_paths=[value for value in split_paths if value],
            # YOLO(base_model) initializes from the external Ultralytics
            # checkpoint, not from the prior active BioVision detector.  The
            # latter remains the metric baseline only.
            parent_model_id=None,
            baseline_model_id=baseline_model_id,
            training_mode="retrain_from_base",
            initialization={
                "strategy": "pretrained_checkpoint",
                "framework": "ultralytics",
                "checkpoint": base_checkpoint,
            },
        )
        lineage_payload["projectStateRevision"] = (
            lineage_payload.get("dataset", {}).get("revision")
            if isinstance(lineage_payload.get("dataset"), dict)
            else None
        )
        lineage_payload["effectiveDataset"] = (
            {
                "revision": effective_dataset.get("revision"),
                "descriptorSha256": effective_dataset_descriptor.get("sha256"),
            }
            if isinstance(effective_dataset, dict) and effective_dataset_descriptor
            else None
        )
        lineage_payload["trainingProtocol"] = {
            "revision": training_protocol["revision"],
            "descriptorSha256": training_protocol_descriptor["sha256"],
        }
        lineage_payload["trainingInputGuard"] = {
            "revision": verified_training_input_guard["revision"],
            "descriptorSha256": training_input_guard_descriptor["sha256"],
            "postFitVerified": True,
        }
        lineage_payload["promotionPolicy"] = _obb_promotion_policy()
        schema_contract = (
            lineage_payload.get("schema", {})
            if isinstance(lineage_payload, dict)
            else {}
        )
        schema_semantics = (
            schema_contract.get("semantics", {})
            if isinstance(schema_contract, dict)
            else {}
        )
        orientation_contract = (
            schema_semantics.get("orientationPolicy")
            if isinstance(schema_semantics, dict)
            else None
        )
        artifact_sha256 = lineage.sha256_file(artifact_path)
        config_sha256 = lineage.sha256_file(artifact_config_path)

        baseline_metrics = active_prior.get("metrics", {}) if active_prior else {}
        candidate_metric, candidate_score = _first_obb_promotion_metric(training_metrics)
        baseline_metric, baseline_score = _first_obb_promotion_metric(baseline_metrics)
        common_metric, common_candidate_score, common_baseline_score = (
            _select_common_obb_promotion_metric(training_metrics, baseline_metrics)
        )
        baseline_cohort = _normalize_obb_validation_cohort(
            active_prior.get("validationCohort") if active_prior else None
        )
        baseline_evaluator_protocol = (
            active_prior.get("evaluatorProtocol") if active_prior else None
        )
        baseline_evaluator_protocol_fingerprint = (
            active_prior.get("evaluatorProtocolFingerprint") if active_prior else None
        )
        candidate_schema_fingerprint = schema_contract.get("semanticFingerprint")
        baseline_schema_fingerprint = (
            active_prior.get("schemaSemanticFingerprint") if active_prior else None
        )
        if common_metric is not None:
            candidate_metric = baseline_metric = common_metric
            candidate_score = common_candidate_score
            baseline_score = common_baseline_score
        numerical_tolerance = None
        required_improvement = None
        observed_improvement = None
        if not candidate_schema_fingerprint:
            promoted = False
            promotion_reason = "candidate_missing_schema_semantic_fingerprint"
        elif candidate_score is None:
            promoted = False
            promotion_reason = "candidate_missing_validation_metric"
        elif not validation_cohort or not validation_cohort.get("frozen"):
            promoted = False
            promotion_reason = "candidate_missing_frozen_validation_cohort"
        elif int(validation_cohort.get("sampleCount") or 0) <= 0:
            promoted = False
            promotion_reason = "candidate_empty_validation_cohort"
        elif int(validation_cohort.get("sampleCount") or 0) < OBB_PROMOTION_MIN_VALIDATION_SAMPLES:
            promoted = False
            promotion_reason = "candidate_insufficient_validation_samples"
        elif int(validation_cohort.get("groupCount") or 0) < OBB_PROMOTION_MIN_VALIDATION_GROUPS:
            promoted = False
            promotion_reason = "candidate_insufficient_validation_groups"
        elif not _obb_validation_class_coverage_complete(validation_cohort):
            promoted = False
            promotion_reason = "candidate_validation_class_coverage_incomplete"
        elif (
            not evaluator_protocol_fingerprint
            or evaluator_protocol_fingerprint
            != _obb_evaluator_protocol_fingerprint(evaluator_protocol)
        ):
            promoted = False
            promotion_reason = "candidate_missing_evaluator_protocol_fingerprint"
        elif active_prior is None:
            promoted = True
            promotion_reason = "first_validated_obb_model"
        elif not baseline_schema_fingerprint:
            promoted = False
            promotion_reason = "active_model_missing_schema_semantic_fingerprint"
        elif candidate_schema_fingerprint != baseline_schema_fingerprint:
            promoted = False
            promotion_reason = "schema_semantic_fingerprint_mismatch"
        elif baseline_score is None:
            promoted = False
            promotion_reason = "active_model_missing_comparable_metric"
        elif common_metric is None:
            promoted = False
            promotion_reason = "no_common_locked_cohort_metric"
        elif not baseline_cohort or not baseline_cohort.get("frozen"):
            promoted = False
            promotion_reason = "baseline_missing_frozen_validation_cohort"
        elif not _obb_validation_cohorts_match(validation_cohort, baseline_cohort):
            promoted = False
            promotion_reason = "frozen_validation_cohort_mismatch"
        elif not baseline_evaluator_protocol or not baseline_evaluator_protocol_fingerprint:
            promoted = False
            promotion_reason = "baseline_missing_evaluator_protocol_fingerprint"
        elif not _obb_evaluator_protocols_match(
            evaluator_protocol,
            evaluator_protocol_fingerprint,
            baseline_evaluator_protocol,
            baseline_evaluator_protocol_fingerprint,
        ):
            promoted = False
            promotion_reason = "evaluator_protocol_fingerprint_mismatch"
        else:
            numerical_tolerance = max(1e-12, abs(float(baseline_score)) * 1e-9)
            policy = _obb_promotion_policy()
            required_improvement = max(
                numerical_tolerance,
                float(policy["minimumAbsoluteImprovement"]),
                abs(float(baseline_score))
                * float(policy["minimumRelativeImprovement"]),
            )
            observed_improvement = float(candidate_score) - float(baseline_score)
            promoted = observed_improvement > required_improvement
            if promoted:
                promotion_reason = "locked_cohort_improved"
            elif observed_improvement > 0.0 and required_improvement > numerical_tolerance:
                promotion_reason = "locked_cohort_improvement_below_minimum"
            else:
                promotion_reason = "locked_cohort_not_improved"
        current_registry = _load_and_validate_obb_registry(
            session_dir,
            allowed_unregistered_run_ids=(run_id,),
        )
        if lineage.sha256_json(current_registry) != obb_registry_revision:
            raise RuntimeError(
                "OBB registry changed during training; refusing to publish a candidate against "
                "stale model history. Aliases and registry were left unchanged."
            )
        if promoted:
            for prior in prior_runs:
                if isinstance(prior, dict):
                    prior["status"] = "deprecated"
        promotion = {
            "promoted": bool(promoted),
            "reason": promotion_reason,
            "metricSource": "frozen_validation_only",
            "thresholdCalibrationInfluence": "none",
            "testReportInfluence": "none",
            "metric": candidate_metric,
            "candidateScore": candidate_score,
            "candidateCohort": validation_cohort,
            "baselineMetric": baseline_metric,
            "baselineScore": baseline_score,
            "baselineCohort": baseline_cohort,
            "baselineModelId": active_prior.get("modelId") if active_prior else None,
            "candidateSchemaSemanticFingerprint": candidate_schema_fingerprint,
            "baselineSchemaSemanticFingerprint": baseline_schema_fingerprint,
            "candidateEvaluatorProtocol": evaluator_protocol,
            "candidateEvaluatorProtocolFingerprint": evaluator_protocol_fingerprint,
            "baselineEvaluatorProtocol": baseline_evaluator_protocol,
            "baselineEvaluatorProtocolFingerprint": baseline_evaluator_protocol_fingerprint,
            "observedImprovement": observed_improvement,
            "requiredImprovement": required_improvement,
            "numericalTolerance": numerical_tolerance,
            "improvementPolicy": _obb_promotion_policy(),
        }

        # Preserve scientific blinding: the frozen report-only test cohort is
        # invisible until validation-only evidence has already decided
        # promotion. A rejected candidate never invokes split=test and cannot
        # expose test metrics through its artifact or result payload.
        test_cohort_ready = bool(
            test_cohort
            and test_cohort.get("frozen")
            and int(test_cohort.get("sampleCount") or 0) > 0
        )
        test_metrics = {}
        test_report_error = None
        if promoted and test_cohort_ready:
            test_report_not_run_reason = None
            send_obb_progress(
                "Evaluating the frozen report-only OBB test cohort...",
                94,
                "test_evaluation",
            )
            try:
                test_model = YOLO(artifact_path)
                test_result = test_model.val(
                    **_obb_evaluation_kwargs_from_protocol(
                        test_evaluator_protocol,
                        dataset_yaml=dataset_yaml,
                        split="test",
                        workers=resolved_workers,
                        device=device,
                    )
                )
                test_metrics = _extract_yolo_metrics(
                    test_result,
                    class_names=getattr(test_model, "names", class_names),
                )
                test_report_status = "completed"
                if not test_metrics:
                    warnings.append(
                        "Frozen OBB test evaluation completed but returned no recognized metrics."
                    )
            except Exception as exc:
                test_report_status = "failed"
                test_report_error = str(exc)
                warnings.append(
                    "Could not evaluate the frozen report-only OBB test cohort: "
                    f"{exc}"
                )
        else:
            test_report_status = "not_run"
            test_report_not_run_reason = (
                "candidate_not_promoted"
                if not promoted
                else "frozen_test_cohort_unavailable"
            )
        if promoted and test_cohort_ready:
            # Test reporting remains report-only, but it reads the same pinned
            # dataset. Reassert the pre-fit closure after that final evaluator
            # so its artifact cannot attest to bytes changed during evaluation.
            _assert_obb_training_inputs_unchanged(
                export_result,
                training_input_guard,
            )

        test_report_path = os.path.join(artifact_dir, "test_report.json")
        test_report = {
            "formatVersion": 1,
            "modelId": model_id,
            "artifactSha256": artifact_sha256,
            "role": "report_only",
            "promotionInfluence": "none",
            "status": test_report_status,
            "cohort": test_cohort,
            "metrics": test_metrics,
            "evaluatorProtocol": test_evaluator_protocol,
            "evaluatorProtocolFingerprint": test_evaluator_protocol_fingerprint,
            "promotionDecision": {
                "promoted": bool(promoted),
                "reason": promotion_reason,
            },
            **(
                {"notRunReason": test_report_not_run_reason}
                if test_report_not_run_reason
                else {}
            ),
            **({"error": test_report_error} if test_report_error else {}),
        }
        lineage.atomic_write_json(test_report_path, test_report)
        test_report_sha256 = lineage.sha256_file(test_report_path)
        artifact_manifest_path = os.path.join(artifact_dir, "manifest.json")
        lineage.atomic_write_json(
            artifact_manifest_path,
            {
                "formatVersion": 2,
                "modelId": model_id,
                "modelType": "obb",
                "runId": run_id,
                "createdAt": lineage.utc_now_iso(),
                "artifact": {
                    "path": artifact_path,
                    "sha256": artifact_sha256,
                },
                "schemaSemanticFingerprint": schema_contract.get("semanticFingerprint"),
                "schemaSemanticVersion": schema_contract.get("semanticVersion"),
                "orientationContract": orientation_contract,
                "config": {
                    "path": artifact_config_path,
                    "relativePath": "obb_config.json",
                    "sha256": config_sha256,
                },
                "datasetExport": {
                    "manifestPath": export_result.get("export_manifest_path"),
                    "cohortManifestPath": export_result.get("cohort_manifest_path"),
                    "splitAssignmentsPath": export_result.get("split_assignments_path"),
                    "splitProfileKey": export_result.get("split_profile_key"),
                    "validationCohort": validation_cohort,
                    "testCohort": test_cohort,
                    "immutableSnapshots": dataset_snapshot_descriptors,
                    "effectiveDataset": effective_dataset_descriptor,
                    "trainingInputGuard": training_input_guard_descriptor,
                },
                "trainingProtocol": training_protocol_descriptor,
                "promotionPolicy": _obb_promotion_policy(),
                "evaluatorProtocol": evaluator_protocol,
                "evaluatorProtocolFingerprint": evaluator_protocol_fingerprint,
                "thresholdCalibration": runtime_config["thresholdCalibration"],
                "calibratedValidationEvaluatorProtocol": calibrated_validation_protocol,
                "calibratedValidationEvaluatorProtocolFingerprint": (
                    _obb_evaluator_protocol_fingerprint(calibrated_validation_protocol)
                ),
                "testReport": {
                    "path": test_report_path,
                    "relativePath": "test_report.json",
                    "sha256": test_report_sha256,
                    "status": test_report_status,
                    "role": "report_only",
                    "promotionInfluence": "none",
                    **(
                        {"notRunReason": test_report_not_run_reason}
                        if test_report_not_run_reason
                        else {}
                    ),
                },
                "hyperparameters": training_hyperparameters,
                "initialization": lineage_payload.get("initialization"),
                "metrics": training_metrics,
                "lineage": lineage_payload,
            },
        )
        obb_record = {
            "modelId": model_id,
            "name": "Session OBB Detector",
            "runId": run_id,
            "path": artifact_path,
            "artifactSha256": artifact_sha256,
            "configPath": artifact_config_path,
            "config": {
                "path": artifact_config_path,
                "relativePath": "obb_config.json",
                "sha256": config_sha256,
            },
            "manifestPath": artifact_manifest_path,
            "schemaSemanticFingerprint": schema_contract.get("semanticFingerprint"),
            "schemaSemanticVersion": schema_contract.get("semanticVersion"),
            "orientationContract": orientation_contract,
            "createdAt": lineage.utc_now_iso(),
            "status": "active" if promoted else "candidate",
            "metrics": training_metrics,
            "validationCohort": validation_cohort,
            "testCohort": test_cohort,
            "effectiveDataset": effective_dataset_descriptor,
            "trainingInputGuard": training_input_guard_descriptor,
            "datasetSnapshots": dataset_snapshot_descriptors,
            "trainingProtocol": training_protocol_descriptor,
            "evaluatorProtocol": evaluator_protocol,
            "evaluatorProtocolFingerprint": evaluator_protocol_fingerprint,
            "thresholdCalibration": runtime_config["thresholdCalibration"],
            "testReport": {
                "path": test_report_path,
                "relativePath": "test_report.json",
                "sha256": test_report_sha256,
                "status": test_report_status,
                "role": "report_only",
                "promotionInfluence": "none",
                "cohort": test_cohort,
                "metrics": test_metrics,
                **(
                    {"notRunReason": test_report_not_run_reason}
                    if test_report_not_run_reason
                    else {}
                ),
            },
            "promotion": promotion,
        }
        dest = os.path.join(models_dir, "session_obb_detector.pt")
        config_alias = os.path.join(models_dir, "session_obb_detector_config.json")
        registry_payload = {
            "version": 2,
            "updatedAt": lineage.utc_now_iso(),
            "models": [*prior_runs, obb_record],
        }
        _publish_obb_registry_and_aliases(
            obb_registry_path,
            registry_payload,
            promoted=promoted,
            artifact_path=artifact_path,
            artifact_config_path=artifact_config_path,
            model_alias_path=dest,
            config_alias_path=config_alias,
        )

        send_obb_progress("OBB detector training complete", 100, "done")
        return {
            "status": "result",
            "ok": True,
            "model_path": dest if promoted else artifact_path,
            "model_id": model_id,
            "artifact_path": artifact_path,
            "config_path": artifact_config_path,
            "manifest_path": artifact_manifest_path,
            "model_status": obb_record["status"],
            "promotion": promotion,
            "warnings": warnings,
            "metrics": training_metrics,
            "metrics_path": metrics_path,
            "threshold_calibration": runtime_config["thresholdCalibration"],
            "test_cohort": test_cohort,
            "test_metrics": test_metrics,
            "test_report_status": test_report_status,
            "test_report_not_run_reason": test_report_not_run_reason,
            "test_report_path": test_report_path,
            "training_input_guard": training_input_guard_descriptor,
            "map50": training_metrics.get("map50"),
            "map50_95": training_metrics.get("map50_95"),
            "precision": training_metrics.get("precision"),
            "recall": training_metrics.get("recall"),
            "per_class": training_metrics.get("per_class", []),
        }

    # ------------------------------------------------------------------
    # OBB inference
    # ------------------------------------------------------------------
    def detect_obb(self, image_path, model_path, conf=None, nms_iou=None,
                   detection_preset=None, max_objects=20, imgsz=None,
                   orientation_policy=None):
        """
        Run the trained session OBB detector on an image.
        Returns list of detections: [{corners, angle, class_id, confidence}]
        """
        from ultralytics import YOLO
        from detection.detection_utils import normalize_orientation_payload

        if nms_iou is None:
            nms_iou = load_obb_nms_iou(model_path, default=0.3)
        if conf is None:
            conf = load_obb_confidence_threshold(model_path, default=0.3)

        resolved = self._resolve_detection_preset(
            conf_threshold=conf,
            nms_iou=nms_iou,
            max_objects=max_objects,
            detection_preset=("custom" if detection_preset is None else detection_preset),
            task="obb",
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
        detections = []
        for r in results:
            for index, parsed in enumerate(iter_ultralytics_obb(r)):
                try:
                    class_id = parsed["class_id"]
                    normalized_orientation = normalize_orientation_payload(class_id, orientation_policy)
                    detections.append({
                        "corners": parsed["corners"],
                        "angle": parsed["angle"],
                        "class_id": int(normalized_orientation.get("class_id", class_id)),
                        "confidence": parsed["confidence"],
                        **(
                            {"orientation_hint": normalized_orientation["orientation_hint"]}
                            if "orientation_hint" in normalized_orientation
                            else {}
                        ),
                    })
                except Exception as e:
                    logger.warning(f"OBB detection parse error at index {index}: {e}")
        # Ultralytics has already applied oriented NMS for each result.
        detections.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        return detections[:max(1, int(resolved["top_k"]))]

    # ------------------------------------------------------------------
    # OBB class_id tagging from placed landmarks
    # ------------------------------------------------------------------
    def tag_class_ids(self, session_dir, boxes, orientation_policy=None):
        """
        Compute class_id for each box from placed landmarks / anchor geometry.

        class_id encoding:
          directional: 0=left-facing (canonical), 1=right-facing
          bilateral:   0=up-facing (canonical), 1=down-facing
          axial:        always 0 (the two poles are interchangeable)
          invariant:   always 0

        Returns list of {"id": ..., "class_id": 0|1}.
        """
        from data.export_yolo_dataset import _load_head_tail_ids, _resolve_obb_class_id

        mode = str((orientation_policy or {}).get("mode", "invariant")).strip().lower()

        if mode not in ("directional", "bilateral"):
            return [{"id": b.get("id"), "class_id": 0} for b in boxes]

        head_id, tail_id = _load_head_tail_ids(session_dir)

        result = []
        for b in boxes:
            class_id = _resolve_obb_class_id(
                b,
                orientation_class_enabled=True,
                head_id=head_id,
                tail_id=tail_id,
                orientation_policy=orientation_policy,
            )
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
                    batch=cmd.get("batch"),
                    imgsz=cmd.get("imgsz"),
                    iou_loss=cmd.get("iou_loss", 0.3),
                    cls_loss=cmd.get("cls_loss", 1.5),
                    box_loss=cmd.get("box_loss", 5.0),
                    orientation_schema=cmd.get("orientation_schema", "invariant"),
                    seed=cmd.get("seed", 42),
                )
                send_response(result)

            elif command == "detect_obb":
                detections = annotator.detect_obb(
                    image_path=cmd["image_path"],
                    model_path=cmd["model_path"],
                    conf=cmd.get("conf"),
                    nms_iou=cmd.get("nms_iou"),   # None Ã¢â€ â€™ auto-load from sidecar
                    detection_preset=cmd.get("detection_preset"),
                    max_objects=cmd.get("max_objects", 20),
                    imgsz=cmd.get("imgsz"),
                    orientation_policy=cmd.get("orientation_policy"),
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
                    allow_rectangle_fallback=cmd.get("allow_rectangle_fallback", True),
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
