"""Immutable dataset/model lineage helpers used by BioVision trainers.

The UI historically treated a user supplied model name as both a display name and
an artifact identifier.  That made overwrites and renames destructive.  This
module keeps those concepts separate while retaining legacy aliases for older
callers.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterable


LINEAGE_FORMAT_VERSION = 2
MODEL_REGISTRY_VERSION = 2
MODEL_PROMOTION_MIN_VALIDATION_SOURCES = 2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_evaluator_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical, self-verifying landmark validation protocol.

    Frozen cohort revisions already bind evaluator image bytes and targets. This
    independent fingerprint binds how those targets are mapped, preprocessed,
    and reduced to the validation metric used for model promotion.
    """
    if not isinstance(protocol, dict):
        raise ValueError("validation evaluator protocol must be a mapping")
    canonical = json.loads(json.dumps(protocol, sort_keys=True))
    canonical.pop("fingerprint", None)
    canonical.setdefault("formatVersion", 1)
    canonical.setdefault("role", "landmark_validation_model_promotion")
    required_mappings = ("evaluator", "preprocessing", "metricDefinitions")
    for field in required_mappings:
        if not isinstance(canonical.get(field), dict) or not canonical[field]:
            raise ValueError(f"validation evaluator protocol requires non-empty {field}")
    if not isinstance(canonical.get("landmarkOrder"), list) or not canonical["landmarkOrder"]:
        raise ValueError("validation evaluator protocol requires non-empty landmarkOrder")
    if str(canonical.get("modelType") or "").strip().lower() not in {"dlib", "cnn"}:
        raise ValueError("validation evaluator protocol modelType must be dlib or cnn")
    return {**canonical, "fingerprint": sha256_json(canonical)}


def validation_evaluator_protocol_fingerprint(
    protocol: Any,
    *,
    metric_name: str | None = None,
    model_type: str | None = None,
) -> str | None:
    """Validate a stored evaluator protocol and return its exact fingerprint."""
    if not isinstance(protocol, dict):
        return None
    stored = str(protocol.get("fingerprint") or "").strip().lower()
    if not stored:
        return None
    try:
        rebuilt = build_validation_evaluator_protocol(protocol)
    except (TypeError, ValueError):
        return None
    if stored != rebuilt["fingerprint"]:
        return None
    if model_type and str(rebuilt.get("modelType") or "").strip().lower() != str(model_type).strip().lower():
        return None
    definitions = rebuilt.get("metricDefinitions", {})
    if metric_name and str(metric_name) not in definitions:
        return None
    return stored


def read_json(file_path: str, default: Any = None) -> Any:
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def read_json_strict(
    file_path: str,
    *,
    description: str = "JSON file",
    missing_default: Any = None,
) -> Any:
    """Parse JSON, distinguishing "absent" from "present but unreadable".

    ``read_json`` swallows every failure, which turns a truncated or
    permission-denied frozen cohort manifest into an empty dict.  Callers then
    treat the cohort as unlocked and silently rebuild it, destroying the
    evidence.  Only a genuinely missing file yields ``missing_default`` here; a
    zero-byte file, malformed JSON, or an IO error is fatal.
    """
    if not file_path or not os.path.exists(file_path):
        return missing_default
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"The {description} at {file_path} is unreadable or malformed. Restore it "
            "from a backup, or delete it to intentionally create a new cohort version; "
            "BioVision will not silently rebuild frozen cohort state."
        ) from exc


def atomic_write_json(file_path: str, payload: Any) -> str:
    """Replace a JSON file via a same-directory temporary file."""
    directory = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".biovision-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return file_path


def atomic_copy_file(source_path: str, destination_path: str) -> str:
    directory = os.path.dirname(os.path.abspath(destination_path))
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".biovision-", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        shutil.copy2(source_path, temp_path)
        os.replace(temp_path, destination_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return destination_path


def _safe_component(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return text.strip("._-") or "model"


def build_model_id(model_type: str, run_id: str) -> str:
    return f"{_safe_component(model_type)}:{_safe_component(run_id)}"


def build_artifact_tag(display_name: str, run_id: str) -> str:
    """A unique backend tag that remains unchanged when the display name changes."""
    return f"{_safe_component(display_name)}__{_safe_component(run_id)}"


def create_model_artifact_dir(project_root: str, model_type: str, run_id: str) -> str:
    artifact_dir = os.path.join(
        os.path.abspath(project_root),
        "models",
        "runs",
        _safe_component(model_type),
        _safe_component(run_id),
    )
    os.makedirs(artifact_dir, exist_ok=False)
    return artifact_dir


def _normalize_landmarks(raw: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return rows
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index", position + 1))
        except Exception:
            index = position + 1
        rows.append(
            {
                "index": index,
                "name": str(item.get("name") or f"Landmark {index}").strip().casefold(),
                "category": str(item.get("category") or "").strip().casefold(),
                "required": not (
                    item.get("required") is False or item.get("optional") is True
                ),
                "optional": bool(
                    item.get("required") is False or item.get("optional") is True
                ),
            }
        )
    rows.sort(key=lambda row: (row["index"], row["name"], row["category"]))
    return rows


def _normalize_orientation_policy(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not str(raw.get("mode") or "").strip():
        return None
    result: dict[str, Any] = {
        "mode": str(raw.get("mode")).strip().lower(),
        "headCategories": sorted(str(v).strip().lower() for v in raw.get("headCategories", []) if str(v).strip()),
        "tailCategories": sorted(str(v).strip().lower() for v in raw.get("tailCategories", []) if str(v).strip()),
        "anteriorAnchorIds": sorted(int(v) for v in raw.get("anteriorAnchorIds", []) if _is_int_like(v)),
        "posteriorAnchorIds": sorted(int(v) for v in raw.get("posteriorAnchorIds", []) if _is_int_like(v)),
        "bilateralPairs": sorted(
            sorted((int(pair[0]), int(pair[1])))
            for pair in raw.get("bilateralPairs", [])
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and _is_int_like(pair[0])
            and _is_int_like(pair[1])
        ),
        "bilateralClassAxis": str(raw.get("bilateralClassAxis") or "vertical_obb").strip().lower(),
        "obbLevelingMode": str(raw.get("obbLevelingMode") or "on").strip().lower(),
    }
    if result["mode"] == "directional":
        result["targetOrientation"] = str(raw.get("targetOrientation") or "left").strip().lower()
    return result


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


def build_schema_snapshot(project_root: str) -> dict[str, Any]:
    session_path = os.path.join(project_root, "session.json")
    session = read_json(session_path, default={})
    if not isinstance(session, dict):
        session = {}
    semantics = {
        "formatVersion": 2,
        "landmarks": _normalize_landmarks(session.get("landmarkTemplate")),
        "orientationPolicy": _normalize_orientation_policy(session.get("orientationPolicy")),
        "schemaKind": session.get("schemaKind"),
        "schemaSourceId": session.get("schemaSourceId"),
    }
    return {
        # Compatibility uses the v2 fingerprint produced by the shared
        # renderer/Electron canonicalizer. This independent SHA-256 is a
        # lineage-integrity hash, not a substitute compatibility identity.
        "semanticFingerprint": session.get("schemaSemanticFingerprint"),
        "semanticVersion": session.get("schemaSemanticVersion"),
        "lineageSchemaHash": sha256_json(semantics),
        "semantics": semantics,
        "orientationPolicyConfigured": bool(session.get("orientationPolicyConfigured")),
    }


def _iter_files(directory: str, suffixes: tuple[str, ...]) -> Iterable[str]:
    if not os.path.isdir(directory):
        return []
    return (
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.lower().endswith(suffixes) and os.path.isfile(os.path.join(directory, name))
    )


def build_dataset_snapshot(project_root: str, split_paths: Iterable[str] = ()) -> dict[str, Any]:
    labels_dir = os.path.join(project_root, "labels")
    images_dir = os.path.join(project_root, "images")
    label_records: list[dict[str, Any]] = []
    origins: dict[str, int] = {}

    for label_path in _iter_files(labels_dir, (".json",)):
        label = read_json(label_path, default={})
        image_name = str(label.get("imageFilename") or "") if isinstance(label, dict) else ""
        image_path = os.path.join(images_dir, os.path.basename(image_name)) if image_name else ""
        provenance = label.get("provenance") if isinstance(label, dict) else None
        origin = str(provenance.get("source") or "unspecified") if isinstance(provenance, dict) else "unspecified"
        origins[origin] = origins.get(origin, 0) + 1
        label_records.append(
            {
                "label": os.path.basename(label_path),
                "labelSha256": sha256_file(label_path),
                "image": os.path.basename(image_path) if image_path else image_name,
                "imageSha256": sha256_file(image_path) if image_path and os.path.isfile(image_path) else None,
                "origin": origin,
            }
        )

    split_records: list[dict[str, Any]] = []
    for split_path in sorted({os.path.abspath(p) for p in split_paths if p}):
        if os.path.isfile(split_path):
            split_payload = read_json(split_path, default={})
            record = {
                "name": os.path.basename(split_path),
                "sha256": sha256_file(split_path),
            }
            if isinstance(split_payload, dict):
                test_revision = (
                    split_payload.get("testCohortRevision")
                    or split_payload.get("test_cohort_revision")
                )
                validation_revision = (
                    split_payload.get("validationCohortRevision")
                    or split_payload.get("validation_cohort_revision")
                )
                if test_revision:
                    record["testCohortRevision"] = str(test_revision)
                if validation_revision:
                    record["validationCohortRevision"] = str(validation_revision)
                assignments = split_payload.get("assignments")
                validation_source_count = split_payload.get("validation_source_count")
                test_source_count = split_payload.get("test_source_count")
                if validation_source_count is None:
                    validation_source_ids = split_payload.get("validation_source_ids")
                    validation_sources = split_payload.get("validation_sources")
                    if isinstance(validation_source_ids, list):
                        validation_source_count = len(set(map(str, validation_source_ids)))
                    elif isinstance(validation_sources, list):
                        validation_source_count = len(set(map(str, validation_sources)))
                    elif isinstance(assignments, dict):
                        validation_source_count = sum(
                            1
                            for value in assignments.values()
                            if str(value).lower() in {"val", "validation"}
                        )
                if test_source_count is None:
                    test_source_ids = split_payload.get("test_source_ids")
                    test_sources = split_payload.get("test_sources")
                    if isinstance(test_source_ids, list):
                        test_source_count = len(set(map(str, test_source_ids)))
                    elif isinstance(test_sources, list):
                        test_source_count = len(set(map(str, test_sources)))
                    elif isinstance(assignments, dict):
                        test_source_count = sum(
                            1 for value in assignments.values() if str(value).lower() == "test"
                        )
                try:
                    if validation_source_count is not None:
                        record["validationSourceCount"] = max(0, int(validation_source_count))
                except (TypeError, ValueError):
                    pass
                try:
                    if test_source_count is not None:
                        record["testSourceCount"] = max(0, int(test_source_count))
                except (TypeError, ValueError):
                    pass
                if "singleSourceOverlap" in split_payload:
                    record["singleSourceOverlap"] = bool(split_payload.get("singleSourceOverlap"))
                if "validationSourceOverlap" in split_payload:
                    record["validationSourceOverlap"] = bool(
                        split_payload.get("validationSourceOverlap")
                    )
            split_records.append(record)

    identity_payload = {"labels": label_records, "splits": split_records}
    return {
        "revision": sha256_json(identity_payload),
        "labels": label_records,
        "splits": split_records,
        "originCounts": origins,
        "labelCount": len(label_records),
    }


def _code_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _frozen_code_state() -> dict[str, Any] | None:
    """Identify a packaged build, which has no source tree or git metadata.

    In a PyInstaller bundle ``__file__`` points into a temporary extraction
    directory, so the git probe below would silently record no code identity at
    all. Identify the executable itself instead. Its size and mtime are used
    rather than a content hash because the one-file bundle embeds torch and is
    multi-gigabyte; hashing it on every training run would be prohibitive.
    """
    if not getattr(sys, "frozen", False):
        return None
    executable = os.path.abspath(sys.executable)
    state: dict[str, Any] = {
        "root": os.path.dirname(executable),
        "packaged": True,
        "executable": executable,
        "commit": None,
        "dirty": False,
        "diffSha256": None,
        "trackedDiffSha256": None,
        "untrackedFiles": {},
        "dependencyLocks": {},
    }
    try:
        stat = os.stat(executable)
        state["executableSize"] = int(stat.st_size)
        state["executableMtime"] = float(stat.st_mtime)
        state["buildRevision"] = sha256_json(
            {
                "executable": os.path.basename(executable),
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
            }
        )
    except OSError:
        pass
    return state


def collect_code_state() -> dict[str, Any]:
    frozen_state = _frozen_code_state()
    if frozen_state is not None:
        return frozen_state

    root = _code_root()
    state: dict[str, Any] = {
        "root": root,
        "packaged": False,
        "commit": None,
        "dirty": None,
        "diffSha256": None,
        "trackedDiffSha256": None,
        "untrackedFiles": {},
    }
    try:
        state["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        state["dirty"] = bool(status.strip())
        tracked_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        state["trackedDiffSha256"] = sha256_bytes(tracked_diff) if tracked_diff else None
        untracked_output = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        untracked_files: dict[str, str] = {}
        for raw_path in untracked_output.split(b"\0"):
            if not raw_path:
                continue
            relative_path = raw_path.decode("utf-8", errors="surrogateescape")
            absolute_path = os.path.abspath(os.path.join(root, relative_path))
            if os.path.commonpath([root, absolute_path]) != os.path.abspath(root):
                continue
            if os.path.isfile(absolute_path):
                untracked_files[relative_path.replace("\\", "/")] = sha256_file(absolute_path)
        state["untrackedFiles"] = untracked_files
        state["dirty"] = bool(state["dirty"] or untracked_files)
        if state["dirty"]:
            state["diffSha256"] = sha256_json(
                {
                    "trackedDiffSha256": state["trackedDiffSha256"],
                    "untrackedFiles": untracked_files,
                }
            )
    except Exception:
        pass

    dependency_files = [
        os.path.join(root, "package-lock.json"),
        os.path.join(root, "requirements.txt"),
        os.path.join(root, "backend", "requirements.txt"),
        os.path.join(root, "setup_backend.py"),
    ]
    state["dependencyLocks"] = {
        os.path.relpath(file_path, root).replace("\\", "/"): sha256_file(file_path)
        for file_path in dependency_files
        if os.path.isfile(file_path)
    }
    return state


def collect_runtime_state() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("cv2", "dlib", "numpy", "torch", "torchvision", "ultralytics"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            continue
    installed_distributions: dict[str, str] = {}
    try:
        for distribution in importlib.metadata.distributions():
            package_name = str(distribution.metadata.get("Name") or "").strip().lower()
            if package_name:
                installed_distributions[package_name] = str(distribution.version)
    except Exception:
        installed_distributions = {}

    accelerator: dict[str, Any] = {
        "cudaAvailable": False,
        "cudaRuntimeVersion": None,
        "cudnnVersion": None,
        "deviceCount": 0,
        "devices": [],
        "mpsAvailable": False,
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        devices = []
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "totalMemoryBytes": int(properties.total_memory),
                }
            )
        accelerator = {
            "cudaAvailable": cuda_available,
            "cudaRuntimeVersion": getattr(torch.version, "cuda", None),
            "cudnnVersion": (
                int(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available() and torch.backends.cudnn.version()
                else None
            ),
            "deviceCount": device_count,
            "devices": devices,
            "mpsAvailable": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
        }
    except Exception:
        pass

    return {
        "python": sys.version,
        "pythonExecutable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "installedDistributions": dict(sorted(installed_distributions.items())),
        "accelerator": accelerator,
    }


def build_run_lineage(
    project_root: str,
    *,
    split_paths: Iterable[str] = (),
    parent_model_id: str | None = None,
    baseline_model_id: str | None = None,
    training_mode: str = "train_from_base",
    initialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "formatVersion": LINEAGE_FORMAT_VERSION,
        "createdAt": utc_now_iso(),
        "dataset": build_dataset_snapshot(project_root, split_paths=split_paths),
        "schema": build_schema_snapshot(project_root),
        "code": collect_code_state(),
        "runtime": collect_runtime_state(),
        "parentModelId": parent_model_id,
        "baselineModelId": baseline_model_id,
        "trainingMode": training_mode,
        "initialization": initialization,
    }


def get_active_model_record(project_root: str, predictor_type: str) -> dict[str, Any] | None:
    registry = read_json(
        os.path.join(project_root, "models", "model_registry.json"),
        default={},
    )
    models = registry.get("models", []) if isinstance(registry, dict) else []
    if not isinstance(models, list):
        return None
    return next(
        (
            dict(entry)
            for entry in models
            if isinstance(entry, dict)
            and entry.get("predictorType") == predictor_type
            and entry.get("modelKind", "landmark") == "landmark"
            and entry.get("status") == "active"
        ),
        None,
    )


def _comparison_context(run_manifest_path: str, metric_name: str | None) -> dict[str, Any]:
    manifest = read_json(run_manifest_path, default={})
    lineage_payload = manifest.get("lineage", {}) if isinstance(manifest, dict) else {}
    schema = lineage_payload.get("schema", {}) if isinstance(lineage_payload, dict) else {}
    dataset = lineage_payload.get("dataset", {}) if isinstance(lineage_payload, dict) else {}
    splits = dataset.get("splits", []) if isinstance(dataset, dict) else []
    schema_fingerprint = (
        schema.get("semanticFingerprint") if isinstance(schema, dict) else None
    )
    cohort_field = (
        "validationCohortRevision"
        if str(metric_name or "").startswith("validation")
        else "testCohortRevision"
    )
    cohort_revision = None
    cohort_disjoint = None
    cohort_source_count = None
    split_candidates = [split for split in splits if isinstance(split, dict)] if isinstance(splits, list) else []
    model_type = str(
        manifest.get("predictorType") or manifest.get("modelType") or ""
    ).strip().lower() if isinstance(manifest, dict) else ""
    evaluator_protocol = (
        manifest.get("validationEvaluatorProtocol")
        if isinstance(manifest, dict)
        else None
    )
    if not isinstance(evaluator_protocol, dict) and isinstance(lineage_payload, dict):
        evaluator_protocol = lineage_payload.get("validationEvaluatorProtocol")
    evaluator_protocol_fingerprint = validation_evaluator_protocol_fingerprint(
        evaluator_protocol,
        metric_name=metric_name,
        model_type=model_type,
    )
    if cohort_field == "validationCohortRevision" and model_type == "cnn":
        # CNN computes its metric from cnn_validation_v1.json. A shared dlib
        # split_info file may also carry a validation revision, so select the
        # evaluator that actually produced the CNN metric explicitly.
        split_candidates.sort(
            key=lambda split: (
                0 if "cnn_validation" in str(split.get("name") or "").lower() else 1,
                str(split.get("name") or ""),
            )
        )
    for split in split_candidates:
        if isinstance(split, dict) and split.get(cohort_field):
            cohort_revision = str(split[cohort_field])
            if cohort_field == "testCohortRevision" and "singleSourceOverlap" in split:
                cohort_disjoint = not bool(split.get("singleSourceOverlap"))
            elif (
                cohort_field == "validationCohortRevision"
                and "validationSourceOverlap" in split
            ):
                cohort_disjoint = not bool(split.get("validationSourceOverlap"))
            count_field = (
                "validationSourceCount"
                if cohort_field == "validationCohortRevision"
                else "testSourceCount"
            )
            try:
                if split.get(count_field) is not None:
                    cohort_source_count = max(0, int(split.get(count_field)))
            except (TypeError, ValueError):
                cohort_source_count = None
            break
    return {
        "metric": metric_name,
        "schemaSemanticFingerprint": (
            str(schema_fingerprint) if schema_fingerprint else None
        ),
        "cohortKind": "validation" if cohort_field.startswith("validation") else "test",
        "cohortRevision": cohort_revision,
        "cohortDisjoint": cohort_disjoint,
        "cohortSourceCount": cohort_source_count,
        "evaluatorProtocolFingerprint": evaluator_protocol_fingerprint,
    }


def publish_model_run(
    project_root: str,
    *,
    model_type: str,
    predictor_type: str,
    run_id: str,
    display_name: str,
    artifact_tag: str,
    artifact_path: str,
    legacy_path: str,
    run_manifest_path: str,
    current_alias_path: str | None = None,
    config_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    promotion_policy: dict[str, Any] | None = None,
    active_aliases: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Register an immutable run and promote only on a comparable improvement."""
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    registry_path = os.path.join(models_dir, "model_registry.json")
    existing = read_json(registry_path, default={})
    existing_models = existing.get("models", []) if isinstance(existing, dict) else []
    if not isinstance(existing_models, list):
        existing_models = []

    now = utc_now_iso()
    model_id = build_model_id(model_type, run_id)
    active_existing = next(
        (
            entry
            for entry in existing_models
            if isinstance(entry, dict)
            and entry.get("predictorType") == predictor_type
            and entry.get("modelKind", "landmark") == "landmark"
            and entry.get("status") == "active"
        ),
        None,
    )
    new_metrics = metrics or {}
    raw_promotion_policy = {} if promotion_policy is None else promotion_policy
    if not isinstance(raw_promotion_policy, dict):
        raise ValueError("promotion_policy must be a mapping when provided")

    def _nonnegative_policy_float(key: str) -> float:
        raw_value = raw_promotion_policy.get(key, 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"promotion_policy.{key} must be a finite non-negative number") from exc
        if value < 0.0 or value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"promotion_policy.{key} must be a finite non-negative number")
        return value

    minimum_absolute_improvement = _nonnegative_policy_float(
        "minimumAbsoluteImprovement"
    )
    minimum_relative_improvement = _nonnegative_policy_float(
        "minimumRelativeImprovement"
    )
    applied_promotion_policy = {
        "policyVersion": str(
            raw_promotion_policy.get("policyVersion") or "numeric_noise_tolerance_v1"
        ),
        "minimumAbsoluteImprovement": minimum_absolute_improvement,
        "minimumRelativeImprovement": minimum_relative_improvement,
    }

    # Landmark model selection is validation-only. Locked test metrics remain
    # in manifests for unbiased reporting and can never choose the active alias.
    metric_priority = ("validationMedianError", "validationError")

    def _scores(metric_values: Any) -> dict[str, float]:
        result: dict[str, float] = {}
        if not isinstance(metric_values, dict):
            return result
        for metric_name in metric_priority:
            try:
                value = float(metric_values.get(metric_name))
            except Exception:
                continue
            if value == value and value not in (float("inf"), float("-inf")):
                result[metric_name] = value
        return result

    new_scores = _scores(new_metrics)
    prior_scores = _scores(active_existing.get("metrics")) if active_existing else {}
    candidate_preferred_metric = next(
        (name for name in metric_priority if name in new_scores),
        None,
    )
    baseline_preferred_metric = next(
        (name for name in metric_priority if name in prior_scores),
        None,
    )
    common_metric = next(
        (name for name in metric_priority if name in new_scores and name in prior_scores),
        None,
    ) if active_existing else candidate_preferred_metric
    new_metric_name = common_metric or candidate_preferred_metric
    prior_metric_name = common_metric or baseline_preferred_metric
    new_score = new_scores.get(common_metric) if common_metric else new_scores.get(new_metric_name)
    prior_score = prior_scores.get(common_metric) if common_metric else prior_scores.get(prior_metric_name)
    candidate_comparison = _comparison_context(run_manifest_path, new_metric_name)
    prior_comparison = None
    if active_existing:
        stored_comparison = active_existing.get("comparison")
        if (
            isinstance(stored_comparison, dict)
            and stored_comparison.get("metric") == prior_metric_name
        ):
            prior_comparison = dict(stored_comparison)
        else:
            prior_comparison = _comparison_context(
                str(active_existing.get("runManifestPath") or ""),
                prior_metric_name,
            )
    candidate_stability = new_metrics.get("stabilityGatePassed")
    observed_improvement = None
    required_improvement = None
    numerical_tolerance = None
    if candidate_stability is False or bool(new_metrics.get("unstable", False)):
        promoted = False
        promotion_reason = "candidate_failed_validation_stability_gate"
    elif active_existing is None:
        # A first artifact is not automatically a production model. It must
        # satisfy the same minimum scientific contract as later candidates;
        # otherwise it remains available for an explicit manual override.
        if new_score is None:
            promoted = False
            promotion_reason = "first_model_missing_locked_cohort_metric"
        elif not candidate_comparison.get("schemaSemanticFingerprint"):
            promoted = False
            promotion_reason = "first_model_missing_schema_semantic_fingerprint"
        elif not candidate_comparison.get("cohortRevision"):
            promoted = False
            promotion_reason = "first_model_missing_frozen_cohort_revision"
        elif int(candidate_comparison.get("cohortSourceCount") or 0) < MODEL_PROMOTION_MIN_VALIDATION_SOURCES:
            promoted = False
            promotion_reason = "first_model_insufficient_validation_sources"
        elif candidate_comparison.get("cohortDisjoint") is not True:
            promoted = False
            promotion_reason = (
                "candidate_benchmark_overlaps_training"
                if candidate_comparison.get("cohortDisjoint") is False
                else "first_model_missing_cohort_disjointness_contract"
            )
        elif not candidate_comparison.get("evaluatorProtocolFingerprint"):
            promoted = False
            promotion_reason = "first_model_missing_evaluator_protocol_fingerprint"
        else:
            promoted = True
            promotion_reason = "first_validated_model_of_type"
    elif new_score is None:
        promoted = False
        promotion_reason = "candidate_missing_locked_cohort_metric"
    elif prior_score is None:
        promoted = False
        promotion_reason = "active_model_missing_comparable_metric"
    elif common_metric is None:
        promoted = False
        promotion_reason = "incomparable_metric_name"
    elif not candidate_comparison.get("schemaSemanticFingerprint") or not prior_comparison.get("schemaSemanticFingerprint"):
        promoted = False
        promotion_reason = "missing_schema_semantic_fingerprint"
    elif candidate_comparison["schemaSemanticFingerprint"] != prior_comparison["schemaSemanticFingerprint"]:
        promoted = False
        promotion_reason = "incomparable_schema_semantic_fingerprint"
    elif not candidate_comparison.get("cohortRevision") or not prior_comparison.get("cohortRevision"):
        promoted = False
        promotion_reason = "missing_frozen_cohort_revision"
    elif candidate_comparison["cohortRevision"] != prior_comparison["cohortRevision"]:
        promoted = False
        promotion_reason = "incomparable_frozen_cohort_revision"
    elif int(candidate_comparison.get("cohortSourceCount") or 0) < MODEL_PROMOTION_MIN_VALIDATION_SOURCES:
        promoted = False
        promotion_reason = "candidate_insufficient_validation_sources"
    elif int(prior_comparison.get("cohortSourceCount") or 0) < MODEL_PROMOTION_MIN_VALIDATION_SOURCES:
        promoted = False
        promotion_reason = "baseline_insufficient_validation_sources"
    elif candidate_comparison.get("cohortDisjoint") is not True:
        promoted = False
        promotion_reason = (
            "candidate_benchmark_overlaps_training"
            if candidate_comparison.get("cohortDisjoint") is False
            else "candidate_cohort_disjointness_unverified"
        )
    elif prior_comparison.get("cohortDisjoint") is not True:
        promoted = False
        promotion_reason = (
            "baseline_benchmark_overlaps_training"
            if prior_comparison.get("cohortDisjoint") is False
            else "baseline_cohort_disjointness_unverified"
        )
    elif not candidate_comparison.get("evaluatorProtocolFingerprint"):
        promoted = False
        promotion_reason = "candidate_missing_evaluator_protocol_fingerprint"
    elif not prior_comparison.get("evaluatorProtocolFingerprint"):
        promoted = False
        promotion_reason = "baseline_missing_evaluator_protocol_fingerprint"
    elif (
        candidate_comparison["evaluatorProtocolFingerprint"]
        != prior_comparison["evaluatorProtocolFingerprint"]
    ):
        promoted = False
        promotion_reason = "incomparable_evaluator_protocol_fingerprint"
    else:
        numerical_tolerance = max(1e-12, abs(float(prior_score)) * 1e-9)
        required_improvement = max(
            numerical_tolerance,
            minimum_absolute_improvement,
            abs(float(prior_score)) * minimum_relative_improvement,
        )
        observed_improvement = float(prior_score) - float(new_score)
        promoted = observed_improvement > required_improvement
        if promoted:
            promotion_reason = "locked_cohort_improved"
        elif observed_improvement > 0.0 and required_improvement > numerical_tolerance:
            promotion_reason = "locked_cohort_improvement_below_minimum"
        else:
            promotion_reason = "locked_cohort_not_improved"

    next_models: list[dict[str, Any]] = []
    for entry in existing_models:
        if not isinstance(entry, dict) or entry.get("modelId") == model_id:
            continue
        copied = dict(entry)
        if promoted and copied.get("predictorType") == predictor_type and copied.get("modelKind", "landmark") == "landmark":
            copied["status"] = "deprecated"
        next_models.append(copied)

    published_manifest = read_json(run_manifest_path, default={})
    if not isinstance(published_manifest, dict):
        published_manifest = {}

    def _registry_file_descriptor(manifest_key: str, file_path: str | None):
        if not file_path:
            return None
        manifest_descriptor = published_manifest.get(manifest_key)
        if not isinstance(manifest_descriptor, dict):
            manifest_descriptor = {}
        return {
            "path": os.path.abspath(file_path),
            "relativePath": (
                str(manifest_descriptor.get("relativePath"))
                if manifest_descriptor.get("relativePath")
                else os.path.basename(file_path)
            ),
            "sha256": str(manifest_descriptor.get("sha256") or sha256_file(file_path)),
        }

    artifact_descriptor = _registry_file_descriptor("artifact", artifact_path)
    config_descriptor = _registry_file_descriptor("config", config_path)
    manifest_sidecars = published_manifest.get("sidecars")
    registry_sidecars = (
        json.loads(json.dumps(manifest_sidecars))
        if isinstance(manifest_sidecars, dict)
        else {}
    )

    record = {
        "modelId": model_id,
        "key": model_id,
        "name": display_name,
        "displayName": display_name,
        "artifactTag": artifact_tag,
        "predictorType": predictor_type,
        "modelKind": "landmark",
        "immutableArtifact": True,
        "path": os.path.abspath(artifact_path),
        "artifact": artifact_descriptor,
        "legacyPath": os.path.abspath(legacy_path),
        "currentAliasPath": os.path.abspath(current_alias_path) if current_alias_path else None,
        "configPath": os.path.abspath(config_path) if config_path else None,
        "config": config_descriptor,
        "sidecars": registry_sidecars,
        "runId": run_id,
        "runManifestPath": os.path.abspath(run_manifest_path),
        "createdAt": now,
        "status": "active" if promoted else "candidate",
        "metrics": new_metrics,
        "comparison": candidate_comparison,
        "promotion": {
            "promoted": promoted,
            "reason": promotion_reason,
            "metric": common_metric or new_metric_name,
            "candidateMetric": new_metric_name,
            "candidateScore": new_score,
            "baselineMetric": prior_metric_name,
            "baselineScore": prior_score,
            "baselineModelId": active_existing.get("modelId") if active_existing else None,
            "observedImprovement": observed_improvement,
            "requiredImprovement": required_improvement,
            "numericalTolerance": numerical_tolerance,
            "improvementPolicy": applied_promotion_policy,
            "minimumValidationSources": MODEL_PROMOTION_MIN_VALIDATION_SOURCES,
            "candidateStabilityGatePassed": candidate_stability,
            "candidateSchemaSemanticFingerprint": candidate_comparison.get("schemaSemanticFingerprint"),
            "baselineSchemaSemanticFingerprint": (
                prior_comparison.get("schemaSemanticFingerprint")
                if prior_comparison
                else None
            ),
            "candidateCohortRevision": candidate_comparison.get("cohortRevision"),
            "baselineCohortRevision": (
                prior_comparison.get("cohortRevision") if prior_comparison else None
            ),
            "candidateEvaluatorProtocolFingerprint": candidate_comparison.get(
                "evaluatorProtocolFingerprint"
            ),
            "baselineEvaluatorProtocolFingerprint": (
                prior_comparison.get("evaluatorProtocolFingerprint")
                if prior_comparison
                else None
            ),
        },
    }
    next_models.append(record)
    payload = {
        "version": MODEL_REGISTRY_VERSION,
        "updatedAt": now,
        "models": next_models,
    }
    alias_pairs = list(active_aliases or [])
    if active_aliases is None and current_alias_path:
        alias_pairs = [(artifact_path, current_alias_path)]
    alias_states: list[dict[str, Any]] = []
    try:
        if promoted:
            seen_destinations: set[str] = set()
            for source, destination in alias_pairs:
                source_abs = os.path.abspath(source)
                destination_abs = os.path.abspath(destination)
                destination_key = os.path.normcase(destination_abs)
                if destination_key in seen_destinations:
                    raise RuntimeError(f"Duplicate active alias destination: {destination_abs}")
                seen_destinations.add(destination_key)
                if os.path.normcase(source_abs) == destination_key:
                    continue
                if not os.path.isfile(source_abs):
                    raise FileNotFoundError(f"Active alias source does not exist: {source_abs}")
                destination_dir = os.path.dirname(destination_abs)
                os.makedirs(destination_dir, exist_ok=True)
                stage_fd, stage_path = tempfile.mkstemp(
                    prefix=".biovision-alias-stage-",
                    suffix=".tmp",
                    dir=destination_dir,
                )
                os.close(stage_fd)
                backup_path = None
                try:
                    shutil.copy2(source_abs, stage_path)
                    if os.path.isfile(destination_abs):
                        backup_fd, backup_path = tempfile.mkstemp(
                            prefix=".biovision-alias-backup-",
                            suffix=".tmp",
                            dir=destination_dir,
                        )
                        os.close(backup_fd)
                        shutil.copy2(destination_abs, backup_path)
                    alias_states.append(
                        {
                            "destination": destination_abs,
                            "stage": stage_path,
                            "backup": backup_path,
                            "committed": False,
                        }
                    )
                except Exception:
                    if os.path.exists(stage_path):
                        os.unlink(stage_path)
                    if backup_path and os.path.exists(backup_path):
                        os.unlink(backup_path)
                    raise

            for state in alias_states:
                os.replace(state["stage"], state["destination"])
                state["committed"] = True
            atomic_write_json(registry_path, payload)
        else:
            atomic_write_json(registry_path, payload)
    except Exception:
        for state in reversed(alias_states):
            if not state["committed"]:
                continue
            backup_path = state["backup"]
            destination = state["destination"]
            if backup_path and os.path.exists(backup_path):
                os.replace(backup_path, destination)
                state["backup"] = None
            elif os.path.exists(destination):
                os.unlink(destination)
        raise
    finally:
        for state in alias_states:
            for key in ("stage", "backup"):
                path = state.get(key)
                if path and os.path.exists(path):
                    os.unlink(path)
    return record


def finalize_model_test_reporting(
    project_root: str,
    *,
    model_id: str,
    run_manifest_path: str,
    test_evaluation: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist report-only frozen-test results after validation promotion.

    This intentionally cannot alter validation metrics or the stored promotion
    decision. A completed test report is accepted only for the newly active
    model; candidates may persist only a non-evaluated status.
    """
    if not isinstance(test_evaluation, dict):
        raise ValueError("test_evaluation must be a mapping")
    status = str(test_evaluation.get("status") or "").strip().lower()
    if status not in {"completed", "not_run", "failed"}:
        raise ValueError("test_evaluation status must be completed, not_run, or failed")
    report_metrics = dict(metrics or {})
    if any(not str(name).startswith("test") for name in report_metrics):
        raise ValueError("post-promotion reporting may update only test* metrics")
    if status != "completed" and report_metrics:
        raise ValueError("only a completed test evaluation may persist test metrics")

    registry_path = os.path.join(os.path.abspath(project_root), "models", "model_registry.json")
    registry = read_json(registry_path, default={})
    manifest_path = os.path.abspath(run_manifest_path)
    manifest = read_json(manifest_path, default={})
    if not isinstance(registry, dict) or not isinstance(registry.get("models"), list):
        raise RuntimeError("model registry is missing while finalizing test reporting")
    if not isinstance(manifest, dict):
        raise RuntimeError("run manifest is missing while finalizing test reporting")

    matching_index = next(
        (
            index
            for index, entry in enumerate(registry["models"])
            if isinstance(entry, dict) and str(entry.get("modelId") or "") == str(model_id)
        ),
        None,
    )
    if matching_index is None:
        raise RuntimeError(f"model registry has no record for {model_id}")
    original_record = registry["models"][matching_index]
    registered_manifest_path = os.path.abspath(
        str(original_record.get("runManifestPath") or manifest_path)
    )
    if os.path.normcase(registered_manifest_path) != os.path.normcase(manifest_path):
        raise RuntimeError("test reporting manifest does not match the registered model")
    if status == "completed" and original_record.get("status") != "active":
        raise RuntimeError("frozen test evaluation is forbidden for a non-promoted candidate")

    normalized_evaluation = {
        "policyVersion": "promoted_only_blind_test_v1",
        **json.loads(json.dumps(test_evaluation, sort_keys=True)),
        "status": status,
    }
    manifest_metrics = {
        str(name): value
        for name, value in (manifest.get("metrics", {}) or {}).items()
        if not str(name).startswith("test")
    }
    record_metrics = {
        str(name): value
        for name, value in (original_record.get("metrics", {}) or {}).items()
        if not str(name).startswith("test")
    }
    if status == "completed":
        manifest_metrics.update(report_metrics)
        record_metrics.update(report_metrics)

    updated_manifest = dict(manifest)
    updated_manifest["metrics"] = manifest_metrics
    updated_manifest["testEvaluation"] = normalized_evaluation
    updated_record = dict(original_record)
    updated_record["metrics"] = record_metrics
    updated_record["testEvaluation"] = normalized_evaluation
    updated_registry = dict(registry)
    updated_models = list(registry["models"])
    updated_models[matching_index] = updated_record
    updated_registry["models"] = updated_models
    updated_registry["updatedAt"] = utc_now_iso()

    try:
        atomic_write_json(manifest_path, updated_manifest)
        atomic_write_json(registry_path, updated_registry)
    except Exception:
        try:
            atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        raise
    return updated_record
