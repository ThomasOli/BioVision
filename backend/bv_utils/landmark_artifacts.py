"""Immutable landmark-model sidecar bundling and runtime resolution."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from bv_utils import lineage
except ModuleNotFoundError:  # Package-style unit-test imports from repository root.
    from . import lineage


ID_MAPPING_FILENAME = "id_mapping.json"
ID_MAPPING_FORMAT = "biovision.landmark-id-mapping.v1"
_ORIENTATION_MODES = {"directional", "bilateral", "axial", "invariant"}


class ImmutableLandmarkArtifactError(RuntimeError):
    """Raised when an immutable landmark artifact is incomplete or altered."""


def load_and_validate_id_mapping(path: str) -> dict[str, Any]:
    """Load the required ID/orientation/template contract for a new artifact."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ValueError(f"Could not read landmark ID mapping '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Landmark ID mapping '{path}' must contain a JSON object.")

    index_mapping = payload.get("dlib_index_to_original")
    if not isinstance(index_mapping, dict) or not index_mapping:
        raise ValueError(
            f"Landmark ID mapping '{path}' is missing dlib_index_to_original."
        )
    try:
        normalized_mapping = {
            int(index): int(original_id)
            for index, original_id in index_mapping.items()
        }
        normalized_indices = sorted(normalized_mapping)
        normalized_ids = [normalized_mapping[index] for index in normalized_indices]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Landmark ID mapping '{path}' contains non-integer index/ID values."
        ) from exc
    if normalized_indices != list(range(len(normalized_indices))):
        raise ValueError(
            f"Landmark ID mapping '{path}' must contain contiguous indices starting at zero."
        )
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError(f"Landmark ID mapping '{path}' maps multiple indices to one schema ID.")

    landmark_template = payload.get("landmark_template")
    if not isinstance(landmark_template, dict) or not landmark_template:
        raise ValueError(f"Landmark ID mapping '{path}' is missing landmark_template metadata.")
    try:
        normalized_template = {
            int(landmark_id): template
            for landmark_id, template in landmark_template.items()
        }
        for landmark_id, template in normalized_template.items():
            if not isinstance(template, dict):
                raise TypeError(f"template {landmark_id} is not an object")
            float(template["x_mean"])
            float(template["y_mean"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Landmark ID mapping '{path}' contains invalid landmark_template metadata."
        ) from exc
    missing_templates = sorted(set(normalized_ids) - set(normalized_template))
    if missing_templates:
        raise ValueError(
            f"Landmark ID mapping '{path}' has no template metadata for schema IDs "
            f"{missing_templates}."
        )

    training_config = payload.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError(f"Landmark ID mapping '{path}' is missing training_config metadata.")
    orientation_mode = str(training_config.get("orientation_mode") or "").strip().lower()
    orientation_policy = training_config.get("orientation_policy")
    policy_mode = (
        str(orientation_policy.get("mode") or "").strip().lower()
        if isinstance(orientation_policy, dict)
        else ""
    )
    if orientation_mode not in _ORIENTATION_MODES or policy_mode != orientation_mode:
        raise ValueError(
            f"Landmark ID mapping '{path}' has an invalid or inconsistent orientation contract."
        )
    return payload


def bundle_id_mapping(source_path: str, artifact_dir: str):
    """Validate and copy a training mapping into an immutable artifact directory."""
    payload = load_and_validate_id_mapping(source_path)
    destination = os.path.join(os.path.abspath(artifact_dir), ID_MAPPING_FILENAME)
    lineage.atomic_copy_file(source_path, destination)
    # Re-read the artifact copy so a bad or partial publication cannot be
    # represented by a descriptor derived only from the source file.
    load_and_validate_id_mapping(destination)
    descriptor = {
        "format": ID_MAPPING_FORMAT,
        "path": destination,
        "relativePath": ID_MAPPING_FILENAME,
        "sha256": lineage.sha256_file(destination),
    }
    return destination, descriptor, payload


def _load_registry_state(project_root: str):
    registry_path = os.path.join(os.path.abspath(project_root), "models", "model_registry.json")
    if not os.path.exists(registry_path):
        return None
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
    except Exception as exc:
        raise ImmutableLandmarkArtifactError(
            f"Landmark model registry exists but is unreadable or malformed: {registry_path}"
        ) from exc
    if not isinstance(registry, dict):
        raise ImmutableLandmarkArtifactError(
            f"Landmark model registry must contain a JSON object: {registry_path}"
        )
    raw_version = registry.get("version")
    if isinstance(raw_version, bool) or str(raw_version) not in {"1", "2"}:
        raise ImmutableLandmarkArtifactError(
            f"Landmark model registry has a missing or unsupported version: {raw_version!r}"
        )
    version = int(raw_version)
    raw_records = registry.get("models")
    if not isinstance(raw_records, list):
        raise ImmutableLandmarkArtifactError(
            f"Landmark model registry v{version} has an invalid models collection."
        )
    if version >= 2 and any(not isinstance(record, dict) for record in raw_records):
        raise ImmutableLandmarkArtifactError(
            "Landmark model registry v2 contains an invalid model record."
        )
    return {
        "path": registry_path,
        "version": version,
        "records": [record for record in raw_records if isinstance(record, dict)],
    }


def _find_registry_record(project_root: str, tag: str, predictor_type: str):
    registry_state = _load_registry_state(project_root)
    if registry_state is None:
        return None, None
    records = registry_state["records"]
    records = [
        record
        for record in records
        if isinstance(record, dict) and record.get("predictorType") == predictor_type
    ]
    identifier = str(tag)
    for field in ("modelId", "key", "artifactTag"):
        exact = [record for record in records if str(record.get(field) or "") == identifier]
        if exact:
            return exact[-1], registry_state
    named = [
        record
        for record in records
        if identifier in {
            str(record.get("name") or ""),
            str(record.get("displayName") or ""),
        }
    ]
    if named:
        active = [record for record in named if record.get("status") == "active"]
        return (active or named)[-1], registry_state
    return None, registry_state


def _is_immutable_record(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("immutableArtifact") is True:
        return True
    model_id = str(record.get("modelId") or "")
    if not model_id or model_id.startswith("legacy-"):
        return False
    artifact_path = os.path.normcase(os.path.abspath(str(record.get("path") or ".")))
    runs_segment = os.path.normcase(os.path.join("models", "runs"))
    return bool(record.get("runManifestPath") and runs_segment in artifact_path)


def _is_explicit_legacy_record(record: Any, registry_state: Any) -> bool:
    if not isinstance(record, dict) or not isinstance(registry_state, dict):
        return False
    if record.get("immutableArtifact") is True:
        return False
    if int(registry_state.get("version", 0)) < 2:
        return True
    model_id = str(record.get("modelId") or "")
    return model_id.startswith("legacy-") and not record.get("runManifestPath")


def _artifact_local_path(artifact_dir: str, relative_path: Any, label: str) -> str:
    relative = str(relative_path or "").strip().replace("\\", "/")
    if not relative or os.path.isabs(relative):
        raise ImmutableLandmarkArtifactError(
            f"Immutable landmark {label} has no valid artifact-relative path."
        )
    candidate = os.path.abspath(os.path.join(artifact_dir, *relative.split("/")))
    try:
        artifact_real = os.path.realpath(os.path.abspath(artifact_dir))
        candidate_real = os.path.realpath(candidate)
        contained = os.path.commonpath([artifact_real, candidate_real]) == artifact_real
    except ValueError:
        contained = False
    if not contained:
        raise ImmutableLandmarkArtifactError(
            f"Immutable landmark {label} resolves outside its artifact directory."
        )
    return candidate


def _valid_sha256(value: Any) -> str | None:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    return digest


def _verify_descriptor(
    artifact_dir: str,
    manifest_descriptor: Any,
    registry_descriptor: Any,
    *,
    label: str,
    default_relative_path: str | None = None,
) -> str:
    if not isinstance(manifest_descriptor, dict) or not isinstance(registry_descriptor, dict):
        raise ImmutableLandmarkArtifactError(
            f"Immutable landmark {label} descriptor is missing from its manifest or registry."
        )
    manifest_hash = _valid_sha256(manifest_descriptor.get("sha256"))
    registry_hash = _valid_sha256(registry_descriptor.get("sha256"))
    if not manifest_hash or not registry_hash or manifest_hash != registry_hash:
        raise ImmutableLandmarkArtifactError(
            f"Immutable landmark {label} manifest/registry SHA-256 is missing or inconsistent."
        )
    relative_path = manifest_descriptor.get("relativePath") or default_relative_path
    registry_relative = registry_descriptor.get("relativePath") or default_relative_path
    if str(relative_path or "").replace("\\", "/") != str(registry_relative or "").replace("\\", "/"):
        raise ImmutableLandmarkArtifactError(
            f"Immutable landmark {label} manifest/registry paths are inconsistent."
        )
    path = _artifact_local_path(artifact_dir, relative_path, label)
    if not os.path.isfile(path):
        raise ImmutableLandmarkArtifactError(
            f"Immutable landmark {label} is missing: {path}"
        )
    actual_hash = lineage.sha256_file(path)
    if actual_hash != manifest_hash:
        raise ImmutableLandmarkArtifactError(
            f"Immutable landmark {label} SHA-256 mismatch: expected {manifest_hash}, got {actual_hash}."
        )
    return path


def _immutable_artifact_dir(project_root: str, record: dict[str, Any], predictor_type: str) -> str:
    raw_run_id = str(record.get("runId") or "")
    run_id = raw_run_id.strip()
    separators = {"/", "\\"}
    if (
        not run_id
        or run_id != raw_run_id
        or run_id in {".", ".."}
        or os.path.isabs(run_id)
        or bool(os.path.splitdrive(run_id)[0])
        or any(separator in run_id for separator in separators)
        or "\x00" in run_id
    ):
        raise ImmutableLandmarkArtifactError(
            "Immutable landmark registry record has a missing or unsafe run ID."
        )
    runs_root = os.path.abspath(os.path.join(
        os.path.abspath(project_root),
        "models",
        "runs",
        predictor_type,
    ))
    canonical = os.path.abspath(os.path.join(runs_root, run_id))
    try:
        runs_root_real = os.path.realpath(runs_root)
        canonical_real = os.path.realpath(canonical)
        contained = (
            os.path.commonpath([runs_root_real, canonical_real]) == runs_root_real
        )
    except ValueError:
        contained = False
    if not contained:
        raise ImmutableLandmarkArtifactError(
            "Immutable landmark artifact directory resolves outside its model run root."
        )
    # Immutable artifacts are always rooted by type/run ID.  Never follow the
    # mutable registry path elsewhere when the canonical directory is missing.
    return canonical


def resolve_landmark_runtime(
    project_root: str,
    tag: str,
    predictor_type: str,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Resolve immutable model/config/ID metadata, or an explicit legacy layout."""
    predictor_type = str(predictor_type).strip().lower()
    if predictor_type not in {"dlib", "cnn"}:
        raise ValueError(f"Unsupported landmark predictor type: {predictor_type}")
    project_root = os.path.abspath(project_root)
    record, registry_state = _find_registry_record(project_root, tag, predictor_type)
    if _is_immutable_record(record):
        artifact_dir = _immutable_artifact_dir(project_root, record, predictor_type)
        manifest_path = os.path.join(artifact_dir, "manifest.json")
        manifest = lineage.read_json(manifest_path, default=None)
        if not isinstance(manifest, dict):
            raise ImmutableLandmarkArtifactError(
                f"Immutable landmark manifest is missing or invalid: {manifest_path}"
            )
        if str(manifest.get("modelId") or "") != str(record.get("modelId") or ""):
            raise ImmutableLandmarkArtifactError(
                "Immutable landmark manifest and registry model IDs do not match."
            )
        if (
            str(manifest.get("predictorType") or "").strip().lower() != predictor_type
            or str(manifest.get("runId") or "") != str(record.get("runId") or "")
        ):
            raise ImmutableLandmarkArtifactError(
                "Immutable landmark manifest type/run identity does not match its registry record."
            )

        model_filename = "predictor.dat" if predictor_type == "dlib" else "model.pth"
        model_path = _verify_descriptor(
            artifact_dir,
            manifest.get("artifact"),
            record.get("artifact"),
            label="model",
            default_relative_path=model_filename,
        )
        config_path = None
        if predictor_type == "cnn":
            config_path = _verify_descriptor(
                artifact_dir,
                manifest.get("config"),
                record.get("config"),
                label="CNN config",
                default_relative_path="config.json",
            )

        manifest_sidecars = manifest.get("sidecars", {})
        registry_sidecars = record.get("sidecars", {})
        manifest_id_mapping = (
            manifest_sidecars.get("idMapping")
            if isinstance(manifest_sidecars, dict)
            else None
        )
        registry_id_mapping = (
            registry_sidecars.get("idMapping")
            if isinstance(registry_sidecars, dict)
            else None
        )
        if (
            not isinstance(manifest_id_mapping, dict)
            or not isinstance(registry_id_mapping, dict)
            or manifest_id_mapping.get("format") != ID_MAPPING_FORMAT
            or registry_id_mapping.get("format") != ID_MAPPING_FORMAT
        ):
            raise ImmutableLandmarkArtifactError(
                "Immutable landmark ID mapping sidecar format is missing or unsupported."
            )
        id_mapping_path = _verify_descriptor(
            artifact_dir,
            manifest_id_mapping,
            registry_id_mapping,
            label="ID mapping sidecar",
            default_relative_path=ID_MAPPING_FILENAME,
        )
        try:
            id_mapping = load_and_validate_id_mapping(id_mapping_path)
        except ValueError as exc:
            raise ImmutableLandmarkArtifactError(
                f"Immutable landmark ID mapping sidecar is invalid: {exc}"
            ) from exc
        return {
            "immutable": True,
            "record": record,
            "manifest": manifest,
            "model_path": model_path,
            "config_path": config_path,
            "id_mapping_path": id_mapping_path,
            "id_mapping": id_mapping,
        }

    if not allow_legacy:
        raise ImmutableLandmarkArtifactError(
            f"No immutable {predictor_type} landmark artifact matched '{tag}'."
        )
    if registry_state is not None:
        if record is None:
            raise ImmutableLandmarkArtifactError(
                f"Landmark model registry v{registry_state['version']} has no registered "
                f"{predictor_type} identity matching '{tag}'; refusing mutable alias fallback."
            )
        if not _is_explicit_legacy_record(record, registry_state):
            raise ImmutableLandmarkArtifactError(
                "Registered landmark identity is not an explicit legacy model; "
                "refusing mutable alias fallback."
            )

    models_dir = os.path.join(project_root, "models")
    debug_dir = os.path.join(project_root, "debug")
    legacy_tag = str(record.get("artifactTag") or tag) if record else str(tag)
    if predictor_type == "dlib":
        model_path = (
            str(record.get("legacyPath") or record.get("path") or "")
            if record
            else os.path.join(models_dir, f"predictor_{legacy_tag}.dat")
        )
        config_path = None
    else:
        model_path = (
            str(record.get("legacyPath") or record.get("path") or "")
            if record
            else os.path.join(models_dir, f"cnn_{legacy_tag}.pth")
        )
        config_path = (
            str(record.get("configPath") or "")
            if record and record.get("configPath")
            else os.path.join(models_dir, f"cnn_{legacy_tag}_config.json")
        )
    model_path = os.path.abspath(model_path)
    config_path = os.path.abspath(config_path) if config_path else None
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Legacy landmark model not found: {model_path}")
    if config_path and not os.path.isfile(config_path):
        raise FileNotFoundError(f"Legacy CNN config not found: {config_path}")
    id_mapping_path = os.path.join(debug_dir, f"id_mapping_{legacy_tag}.json")
    id_mapping = {}
    if os.path.isfile(id_mapping_path):
        try:
            with open(id_mapping_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            id_mapping = loaded if isinstance(loaded, dict) else {}
        except Exception:
            id_mapping = {}
    return {
        "immutable": False,
        "record": record,
        "manifest": None,
        "model_path": model_path,
        "config_path": config_path,
        "id_mapping_path": id_mapping_path if os.path.isfile(id_mapping_path) else None,
        "id_mapping": id_mapping,
    }
