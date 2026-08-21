"""Deterministic cross-boundary smoke harness for BioVision's four schema modes.

This script is intentionally not a replacement for model-quality evaluation.  It
uses tiny images and mocked dlib/Ultralytics trainers so it can exercise the real
dataset, registry, inference-adapter, and promotion code without a GPU or network.
The Electron test which drives it inserts a real transactional HITL commit between
the ``initial`` and ``resume`` phases.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from annotation import super_annotator  # noqa: E402
from bv_utils import lineage  # noqa: E402
from data import prepare_dataset  # noqa: E402
from training import train_shape_model  # noqa: E402


MODES = ("directional", "bilateral", "axial", "invariant")
TAG_PREFIX = "live_loop"
INITIAL_IMAGE_COUNT = 8
LANDMARK_IDS = (3, 12)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _quiet_call(description: str, callback):
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            return callback()
    except Exception as exc:
        captured = (stdout.getvalue() + "\n" + stderr.getvalue()).strip()
        tail = captured[-5000:] if captured else "(no captured pipeline output)"
        raise RuntimeError(f"{description} failed: {exc}\n--- captured tail ---\n{tail}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _orientation_policy(mode: str) -> dict:
    policy = {
        "mode": mode,
        "obbLevelingMode": "on",
        "trainingPrepBoxJitter": False,
    }
    if mode == "directional":
        policy.update(
            {
                "targetOrientation": "left",
                "anteriorAnchorIds": [3],
                "posteriorAnchorIds": [12],
            }
        )
    elif mode == "bilateral":
        policy.update(
            {
                "bilateralPairs": [[3, 12]],
                "bilateralClassAxis": "vertical_obb",
                "anteriorAnchorIds": [3],
                "posteriorAnchorIds": [12],
            }
        )
    return policy


def _orientation_for(mode: str, index: int) -> tuple[int, str | None]:
    class_id = index % 2 if mode in {"directional", "bilateral"} else 0
    if mode == "directional":
        return class_id, "right" if class_id else "left"
    if mode == "bilateral":
        return class_id, "down" if class_id else "up"
    return 0, None


def _landmarks_for(mode: str, class_id: int, index: int) -> list[dict]:
    wobble = float(index % 3)
    if mode == "bilateral":
        if class_id == 0:
            points = ((62.0 + wobble, 25.0), (65.0 - wobble, 70.0))
        else:
            points = ((65.0 - wobble, 70.0), (62.0 + wobble, 25.0))
    elif mode == "directional" and class_id == 1:
        points = ((98.0 - wobble, 43.0), (30.0 + wobble, 53.0))
    else:
        points = ((30.0 + wobble, 43.0), (98.0 - wobble, 53.0))
    return [
        {"id": 3, "x": points[0][0], "y": points[0][1], "isSkipped": False},
        {"id": 12, "x": points[1][0], "y": points[1][1], "isSkipped": False},
    ]


def _label_payload(mode: str, image_name: str, index: int, *, hitl: bool) -> dict:
    class_id, orientation = _orientation_for(mode, index)
    box = {
        "id": f"box-{mode}-{index}",
        "left": 12,
        "top": 10,
        "width": 104,
        "height": 76,
        "obbCorners": [[12, 10], [116, 10], [116, 86], [12, 86]],
        "class_id": class_id,
        "landmarks": _landmarks_for(mode, class_id, index),
        "trainingTargets": ["landmark", "obb"],
    }
    if orientation is not None:
        box["orientation_override"] = orientation
        box["orientation_hint"] = {
            "orientation": orientation,
            "confidence": 1.0,
            "source": "human_review" if hitl else "human_annotation",
        }
    payload = {
        "imageFilename": image_name,
        "boxes": [box],
        "finalizedDetection": {
            "isFinalized": True,
            "acceptedBoxes": [box],
        },
    }
    if hitl:
        # Electron replaces this draft with the complete review event at commit.
        payload["provenance"] = {"source": "hitl_review"}
    return payload


def _write_image(path: Path, index: int, mode_index: int) -> None:
    image = np.full((96, 128, 3), 35 + ((index * 19 + mode_index * 31) % 170), dtype=np.uint8)
    cv2.rectangle(image, (12, 10), (116, 86), (240, 80 + mode_index * 20, 130), 2)
    cv2.circle(image, (30 + index % 3, 43), 4, (0, 255, 0), -1)
    cv2.circle(image, (98 - index % 3, 53), 4, (0, 0, 255), -1)
    # Give every source a content identity even if a codec changes flat regions.
    image[0, index % image.shape[1], :] = (index * 17 % 255, mode_index * 53, 211)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not create harness image {path}")


def _create_session(workspace: Path, mode: str, mode_index: int) -> dict:
    session_dir = workspace / mode
    images_dir = session_dir / "images"
    labels_dir = session_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    policy = _orientation_policy(mode)
    template = [
        {"index": 3, "name": "anchor a", "category": "head", "required": True},
        {"index": 12, "name": "anchor b", "category": "tail", "required": True},
    ]
    semantic_material = json.dumps(
        {"landmarks": template, "orientationPolicy": policy},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    session = {
        "name": f"Four-mode mocked live loop: {mode}",
        "imageCount": INITIAL_IMAGE_COUNT,
        "orientationPolicyConfigured": True,
        "orientationPolicy": policy,
        "landmarkTemplate": template,
        # Session creation normally computes this in Electron.  Keep the same
        # public v2 contract shape while the harness builds sessions headlessly.
        "schemaSemanticFingerprint": f"v2-{hashlib.sha256(semantic_material).hexdigest()}",
        "schemaSemanticVersion": 2,
    }
    _write_json(session_dir / "session.json", session)

    finalized = []
    for index in range(INITIAL_IMAGE_COUNT):
        image_name = f"annotated_{mode}_{index:02d}.png"
        _write_image(images_dir / image_name, index, mode_index)
        _write_json(
            labels_dir / f"annotated_{mode}_{index:02d}.json",
            _label_payload(mode, image_name, index, hitl=False),
        )
        finalized.append(image_name)
    _write_json(session_dir / "finalized_images.json", finalized)

    review_source = workspace / "incoming" / f"review_{mode}.png"
    review_index = 101 + mode_index
    _write_image(review_source, review_index, mode_index)
    review_name = f"review_{mode}.png"
    return {
        "mode": mode,
        "sessionDir": str(session_dir),
        "tag": f"{TAG_PREFIX}_{mode}",
        "reviewSourcePath": str(review_source),
        "reviewRequestedFilename": review_name,
        "reviewLabelPayload": _label_payload(mode, review_name, review_index, hitl=True),
        "reviewEvent": {
            "eventId": f"four-mode-live-loop-{mode}",
            "commitId": f"four-mode-live-loop-{mode}",
            "source": "hitl_review",
            "reviewOutcome": "corrected",
            "reviewedAt": f"2026-08-20T12:0{mode_index}:00.000Z",
            "landmarkModelKey": f"dlib-initial-{mode}",
            "detectionModelKey": f"obb-initial-{mode}",
            "wasEdited": True,
            "isNewTrainingSample": True,
        },
    }


class _FakeObbRows:
    def __init__(self, class_id: int):
        self.xywhr = np.asarray([[64.0, 48.0, 80.0, 40.0, 0.0]], dtype=np.float32)
        self.xyxyxyxy = np.asarray(
            [[[24.0, 28.0], [104.0, 28.0], [104.0, 68.0], [24.0, 68.0]]],
            dtype=np.float32,
        )
        self.cls = np.asarray([class_id], dtype=np.float32)
        self.conf = np.asarray([0.91], dtype=np.float32)

    def __len__(self):
        return 1


class _FakeYOLO:
    """Tiny Ultralytics-compatible adapter used by the production OBB flow."""

    phase = "initial"
    mode = "invariant"
    inference_class_id = 0

    def __init__(self, checkpoint):
        self.checkpoint = os.fspath(checkpoint)
        self.ckpt_path = self.checkpoint if os.path.isfile(self.checkpoint) else None
        self.names = {0: "specimen", 1: "alternate"}
        self.metrics = None
        self.trainer = None
        self._callbacks = {}

    def _smart_load(self, _name):
        return None

    def add_callback(self, name, callback):
        self._callbacks[name] = callback

    @classmethod
    def _score(cls) -> float:
        return 0.56 if cls.phase == "initial" else 0.74

    def train(self, **kwargs):
        run_dir = Path(kwargs["project"]) / kwargs["name"]
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "best.pt").write_bytes(
            f"mock-obb:{self.mode}:{self.phase}".encode("ascii")
        )
        score = self._score()
        metrics = {
            "metrics/mAP50(B)": min(0.99, score + 0.10),
            "metrics/mAP50-95(B)": score,
            "metrics/precision(B)": min(0.99, score + 0.08),
            "metrics/recall(B)": min(0.99, score + 0.04),
        }
        args = SimpleNamespace(
            split="val",
            imgsz=kwargs["imgsz"],
            batch=kwargs["batch"],
            iou=kwargs["iou"],
            conf=None,
            max_det=300,
            agnostic_nms=False,
            single_cls=False,
            rect=False,
            half=False,
            amp=kwargs.get("amp", False),
            augment=False,
            dnn=False,
            classes=None,
            seed=0,
            deterministic=True,
        )
        self.metrics = metrics
        self.trainer = SimpleNamespace(
            args=args,
            validator=SimpleNamespace(metrics=metrics),
            metrics=metrics,
        )
        return metrics

    def val(self, **kwargs):
        score = self._score()
        if kwargs.get("split") == "test":
            score = 0.51 if self.phase == "initial" else 0.63
        confidence = float(kwargs.get("conf", 0.30))
        iou = float(kwargs.get("iou", 0.30))
        calibration_bonus = max(0.0, 0.03 - abs(confidence - 0.45) * 0.04 - abs(iou - 0.50) * 0.02)
        return {
            "metrics/mAP50(B)": min(0.99, score + 0.10 + calibration_bonus),
            "metrics/mAP50-95(B)": min(0.99, score + calibration_bonus),
            "metrics/precision(B)": min(0.99, score + 0.07 + calibration_bonus),
            "metrics/recall(B)": min(0.99, score + 0.03 + calibration_bonus),
        }

    def predict(self, *_args, **_kwargs):
        rows = _FakeObbRows(self.inference_class_id)
        return [SimpleNamespace(obb=rows, names=self.names)]


_FAKE_ULTRALYTICS = types.ModuleType("ultralytics")
_FAKE_ULTRALYTICS.YOLO = _FakeYOLO
_FAKE_ULTRALYTICS.__version__ = "0.0.mocked-live-loop"
# Install the fake for this short-lived harness process. ``patch.dict`` is not
# used here because restoring the entire sys.modules mapping also removes Torch
# submodules lazily imported by the first mode, making a second import unsafe.
sys.modules["ultralytics"] = _FAKE_ULTRALYTICS


def _stable_runtime_patches():
    return (
        patch.object(
            lineage,
            "collect_code_state",
            return_value={"commit": "mock-live-loop", "dirty": False, "dependencyLocks": {}},
        ),
        patch.object(
            lineage,
            "collect_runtime_state",
            return_value={"python": "mock-python", "platform": "mock-platform", "packages": {}},
        ),
    )


def _prepare_landmarks(entry: dict):
    session_dir = entry["sessionDir"]
    tag = entry["tag"]
    return _quiet_call(
        f"{entry['mode']} landmark preparation",
        lambda: prepare_dataset.json_to_dlib_xml(
            session_dir,
            tag,
            test_split=0.25,
            seed=73,
        ),
    )


def _xml_image_count(path: str) -> int:
    return len(ET.parse(path).getroot().findall("./images/image"))


def _train_landmarks(entry: dict, phase: str) -> dict:
    mode = entry["mode"]
    session_dir = entry["sessionDir"]
    tag = entry["tag"]
    validation_value = 0.080 if phase == "initial" else 0.040
    test_value = 0.090 if phase == "initial" else 0.052

    def metric_for(xml_path: str) -> float:
        name = os.path.basename(xml_path).lower()
        if name.startswith("validation_"):
            return validation_value
        if name.startswith("test_"):
            return test_value
        return max(0.01, validation_value * 0.75)

    def fake_train(_xml_path, predictor_path, _options):
        Path(predictor_path).write_bytes(f"mock-dlib:{mode}:{phase}".encode("ascii"))

    def fake_errors(xml_path, _predictor_path):
        value = metric_for(xml_path)
        return [value for _ in range(max(1, _xml_image_count(xml_path)))]

    def fake_details(xml_path, _predictor_path):
        value = metric_for(xml_path)
        return [
            {
                "image": xml_path,
                "filename": f"mock-{index}.png",
                "mean_error": value,
                "median_error": value,
                "per_landmark_error": [value, value],
            }
            for index in range(max(1, _xml_image_count(xml_path)))
        ]

    run_id = f"dlib_{mode}_{phase}"
    code_patch, runtime_patch = _stable_runtime_patches()
    with (
        patch.object(train_shape_model.dio, "_run_id", return_value=run_id),
        patch.object(train_shape_model.dio, "_utc_now_iso", return_value="2026-08-20T12:00:00Z"),
        patch.object(lineage, "utc_now_iso", return_value="2026-08-20T12:00:00Z"),
        code_patch,
        runtime_patch,
        patch.object(train_shape_model.dlib, "train_shape_predictor", side_effect=fake_train),
        patch.object(train_shape_model.dlib, "test_shape_predictor", side_effect=lambda xml, _p: metric_for(xml)),
        patch.object(train_shape_model, "_compute_dlib_per_image_errors", side_effect=fake_errors),
        patch.object(train_shape_model, "_compute_dlib_per_image_error_details", side_effect=fake_details),
    ):
        result = _quiet_call(
            f"{mode} mocked landmark {phase} training",
            lambda: train_shape_model.train_shape_model(
                session_dir,
                tag,
                aug_angles=[],
                aug_flip=False,
            ),
        )
    _require(result["validation_median_error"] == validation_value, f"{mode}: wrong mocked validation metric")
    return result


def _train_obb(entry: dict, phase: str) -> dict:
    mode = entry["mode"]
    _FakeYOLO.phase = phase
    _FakeYOLO.mode = mode
    code_patch, runtime_patch = _stable_runtime_patches()
    with code_patch, runtime_patch:
        annotator = super_annotator.SuperAnnotator()
        result = _quiet_call(
            f"{mode} mocked OBB {phase} training",
            lambda: annotator.train_yolo_obb(
                entry["sessionDir"],
                epochs=1,
                model_tier="nano",
                device="cpu",
                sam2_enabled=False,
                batch=2,
                imgsz=64,
                orientation_schema=mode,
            ),
        )
    _require(result.get("ok") is True, f"{mode}: OBB training failed: {result}")
    return result


def _exercise_inference(entry: dict, landmark_result: dict, obb_result: dict) -> dict:
    mode = entry["mode"]
    mapping = _read_json(Path(landmark_result["id_mapping_path"]))
    index_mapping = {
        int(key): int(value)
        for key, value in mapping["dlib_index_to_original"].items()
    }

    class FakePoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class FakeShape:
        num_parts = len(index_mapping)

        @staticmethod
        def part(index):
            return FakePoint(100 + index * 30, 180 + index * 20)

    annotator = super_annotator.SuperAnnotator()
    annotator.dlib_predictor = lambda _image, _rect: FakeShape()
    annotator.dlib_id_mapping = index_mapping
    landmark_predictions = annotator.predict_landmarks(
        np.zeros((super_annotator.STANDARD_SIZE, super_annotator.STANDARD_SIZE, 3), dtype=np.uint8)
    )
    _require(
        [point["id"] for point in landmark_predictions] == list(LANDMARK_IDS),
        f"{mode}: immutable dlib index mapping was not honored during inference",
    )

    expected_class = 1 if mode in {"directional", "bilateral"} else 0
    _FakeYOLO.inference_class_id = expected_class
    detections = annotator.detect_obb(
        entry["reviewSourcePath"],
        obb_result["artifact_path"],
        orientation_policy=_orientation_policy(mode),
    )
    _require(len(detections) == 1, f"{mode}: mocked OBB inference returned no detection")
    detection = detections[0]
    _require(detection["class_id"] == expected_class, f"{mode}: wrong OBB inference class")
    hint = (detection.get("orientation_hint") or {}).get("orientation")
    expected_hint = "right" if mode == "directional" else "down" if mode == "bilateral" else None
    _require(hint == expected_hint, f"{mode}: inference orientation hint was {hint!r}, expected {expected_hint!r}")
    return {
        "landmarkIds": [point["id"] for point in landmark_predictions],
        "obbClassId": detection["class_id"],
        "orientationHint": hint,
    }


def _assert_export_semantics(entry: dict) -> dict:
    mode = entry["mode"]
    dataset_dir = Path(entry["sessionDir"]) / "obb_dataset"
    yaml_text = (dataset_dir / "dataset.yaml").read_text(encoding="utf-8")
    expected_nc = 2 if mode in {"directional", "bilateral"} else 1
    _require(f"nc: {expected_nc}" in yaml_text, f"{mode}: dataset class count is not explicit")
    if mode == "directional":
        _require("names: ['left', 'right']" in yaml_text, "directional: wrong class names")
    elif mode == "bilateral":
        _require("names: ['up', 'down']" in yaml_text, "bilateral: wrong class names")
    else:
        _require("names: ['specimen']" in yaml_text, f"{mode}: expected one-class detector")

    classes = []
    for split in ("train", "val", "test"):
        for label_path in sorted((dataset_dir / "labels" / split).glob("*.txt")):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    classes.append(int(line.split()[0]))
    expected_classes = {0, 1} if expected_nc == 2 else {0}
    _require(set(classes) == expected_classes, f"{mode}: exported classes {set(classes)} != {expected_classes}")
    return {"nc": expected_nc, "classes": sorted(set(classes))}


def _landmark_snapshot(entry: dict) -> dict:
    session_dir = Path(entry["sessionDir"])
    tag = entry["tag"]
    split_info = _read_json(session_dir / "debug" / f"split_info_{tag}.json")
    manifest = _read_json(session_dir / "debug" / "cohorts" / "landmark_benchmark_v1.json")
    evaluator_paths = [
        session_dir / "xml" / f"validation_{tag}.xml",
        session_dir / "xml" / f"test_{tag}.xml",
        *[Path(path) for path in split_info.get("validation_files", [])],
        *[Path(path) for path in split_info.get("test_files", [])],
    ]
    hashes = {
        os.path.relpath(path, session_dir).replace("\\", "/"): _sha256_file(path)
        for path in evaluator_paths
        if path.is_file()
    }
    assignments = manifest["assignments"]
    return {
        "validationIds": sorted(key for key, value in assignments.items() if value == "validation"),
        "testIds": sorted(key for key, value in assignments.items() if value == "test"),
        "validationRevision": manifest.get("validationCohortRevision"),
        "testRevision": manifest.get("testCohortRevision"),
        "validationSnapshots": manifest.get("validationSourceSnapshots", {}),
        "testSnapshots": manifest.get("testSourceSnapshots", {}),
        "bytes": hashes,
    }


def _obb_snapshot(entry: dict) -> dict:
    dataset_dir = Path(entry["sessionDir"]) / "obb_dataset"
    cohort = _read_json(dataset_dir / "cohort_manifest.json")
    assignments = _read_json(Path(entry["sessionDir"]) / "obb_split_assignments.v2.json")
    profile_key = cohort["split_profile_key"]
    profile = assignments["profiles"][profile_key]
    groups = profile["groups"]
    return {
        "profileKey": profile_key,
        "validationGroups": sorted(key for key, value in groups.items() if value == "validation"),
        "testGroups": sorted(key for key, value in groups.items() if value == "test"),
        "validationRevision": profile.get("validation_cohort_revision"),
        "testRevision": profile.get("test_cohort_revision"),
        "validationSnapshots": profile.get("validation_group_snapshots", {}),
        "testSnapshots": profile.get("test_group_snapshots", {}),
        "validationMembers": cohort["validation"]["members"],
        "testMembers": cohort["test"]["members"],
        "bytes": {
            "images/val": _hash_tree(dataset_dir / "images" / "val"),
            "labels/val": _hash_tree(dataset_dir / "labels" / "val"),
            "images/test": _hash_tree(dataset_dir / "images" / "test"),
            "labels/test": _hash_tree(dataset_dir / "labels" / "test"),
        },
    }


def _assert_initial_promotions(mode: str, landmark_result: dict, obb_result: dict) -> None:
    _require(landmark_result["registry"]["status"] == "active", f"{mode}: validated first landmark run was not active")
    landmark_promotion = landmark_result["registry"].get("promotion", {})
    _require(landmark_promotion.get("promoted") is True, f"{mode}: first landmark promotion missing")
    _require(
        str(landmark_promotion.get("candidateMetric", "")).startswith("validation"),
        f"{mode}: landmark promotion did not use validation",
    )
    _require(obb_result["model_status"] == "active", f"{mode}: validated first OBB run was not active")
    _require(obb_result["promotion"].get("metricSource") == "frozen_validation_only", f"{mode}: OBB did not gate on validation")
    _require(obb_result["promotion"].get("testReportInfluence") == "none", f"{mode}: OBB test influenced promotion")


def run_initial(workspace: Path, state_path: Path) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    entries = []
    for mode_index, mode in enumerate(MODES):
        entry = _create_session(workspace, mode, mode_index)
        _prepare_landmarks(entry)
        landmark_result = _train_landmarks(entry, "initial")
        obb_result = _train_obb(entry, "initial")
        _assert_initial_promotions(mode, landmark_result, obb_result)
        inference = _exercise_inference(entry, landmark_result, obb_result)
        semantics = _assert_export_semantics(entry)
        entry["baseline"] = {
            "landmark": _landmark_snapshot(entry),
            "obb": _obb_snapshot(entry),
        }
        entry["initialResults"] = {
            "landmarkModelId": landmark_result["model_id"],
            "landmarkValidationMedianError": landmark_result["validation_median_error"],
            "landmarkTestMedianError": landmark_result["test_median_error"],
            "obbModelId": obb_result["model_id"],
            "obbValidationMap50_95": obb_result["metrics"]["map50_95"],
            "obbTestMap50_95": obb_result["test_metrics"].get("map50_95"),
            "inference": inference,
            "semantics": semantics,
        }
        entries.append(entry)
    state = {"version": 1, "workspace": str(workspace), "modes": entries}
    _write_json(state_path, state)
    return state


def _assert_frozen(before: dict, after: dict, mode: str, pipeline: str) -> None:
    for key in before:
        _require(
            after.get(key) == before[key],
            f"{mode}: frozen {pipeline} evaluator field {key!r} changed after HITL",
        )


def _assert_hitl_train_only(entry: dict) -> dict:
    mode = entry["mode"]
    image_name = entry["committedImageName"]
    source_hash = entry["committedSourceSha256"]
    source_id = f"sha256:{source_hash}"
    session_dir = Path(entry["sessionDir"])

    landmark_manifest = _read_json(
        session_dir / "debug" / "cohorts" / "landmark_benchmark_v1.json"
    )
    _require(landmark_manifest["assignments"].get(source_id) == "train", f"{mode}: HITL source entered landmark evaluator")
    _require(
        landmark_manifest.get("sources", {}).get(source_id, {}).get("adaptiveTrainingSample") is True,
        f"{mode}: landmark manifest lost HITL provenance",
    )

    assignments = _read_json(session_dir / "obb_split_assignments.v2.json")
    profile_key = entry["baseline"]["obb"]["profileKey"]
    profile = assignments["profiles"][profile_key]
    _require(profile["groups"].get(source_hash) == "train", f"{mode}: HITL group entered OBB evaluator")
    matching_samples = [
        value
        for value in profile.get("samples", {}).values()
        if value.get("image_filename") == image_name
    ]
    _require(len(matching_samples) == 1 and matching_samples[0].get("split") == "train", f"{mode}: reviewed OBB sample was not train-only")
    train_label = session_dir / "obb_dataset" / "labels" / "train" / f"{Path(image_name).stem}.txt"
    _require(train_label.is_file(), f"{mode}: reviewed OBB label was not exported to train")
    _require(
        not (session_dir / "obb_dataset" / "labels" / "val" / train_label.name).exists()
        and not (session_dir / "obb_dataset" / "labels" / "test" / train_label.name).exists(),
        f"{mode}: reviewed OBB label leaked into an evaluator",
    )
    return {"landmarkSourceId": source_id, "obbGroupId": source_hash, "imageName": image_name}


def _assert_retraining_promotions(entry: dict, landmark_result: dict, obb_result: dict) -> dict:
    mode = entry["mode"]
    initial = entry["initialResults"]
    landmark_before = float(initial["landmarkValidationMedianError"])
    landmark_after = float(landmark_result["validation_median_error"])
    _require(landmark_after < landmark_before, f"{mode}: mocked landmark validation accuracy did not improve")
    _require(landmark_result["registry"]["status"] == "active", f"{mode}: improved landmark run was not promoted")
    landmark_promotion = landmark_result["registry"].get("promotion", {})
    _require(landmark_promotion.get("promoted") is True, f"{mode}: improved landmark promotion missing")
    _require(
        str(landmark_promotion.get("candidateMetric", "")).startswith("validation"),
        f"{mode}: landmark retraining promotion was not validation-gated",
    )

    obb_before = float(initial["obbValidationMap50_95"])
    obb_after = float(obb_result["metrics"]["map50_95"])
    _require(obb_after > obb_before, f"{mode}: mocked OBB validation accuracy did not improve")
    _require(obb_result["model_status"] == "active", f"{mode}: improved OBB run was not promoted")
    promotion = obb_result["promotion"]
    _require(promotion.get("promoted") is True, f"{mode}: improved OBB promotion missing")
    _require(promotion.get("metricSource") == "frozen_validation_only", f"{mode}: OBB promotion source was not validation")
    _require(promotion.get("thresholdCalibrationInfluence") == "none", f"{mode}: threshold sweep influenced promotion")
    _require(promotion.get("testReportInfluence") == "none", f"{mode}: OBB test report influenced promotion")
    report = _read_json(Path(obb_result["test_report_path"]))
    _require(report.get("role") == "report_only", f"{mode}: OBB test was not namespaced report-only")
    _require(report.get("promotionInfluence") == "none", f"{mode}: OBB test report claims promotion influence")
    _require(report.get("cohort", {}).get("reportOnly") is True, f"{mode}: OBB test cohort is not marked report-only")
    return {
        "landmarkValidationMedianErrorBefore": landmark_before,
        "landmarkValidationMedianErrorAfter": landmark_after,
        "landmarkValidationErrorReductionPercent": round((1.0 - landmark_after / landmark_before) * 100.0, 3),
        "obbValidationMap50_95Before": obb_before,
        "obbValidationMap50_95After": obb_after,
        "obbValidationMap50_95Gain": round(obb_after - obb_before, 6),
        "obbTestReportRole": report["role"],
    }


def run_resume(state_path: Path) -> dict:
    state = _read_json(state_path)
    _require(state.get("version") == 1, "Unsupported four-mode harness state")
    summaries = []
    for entry in state["modes"]:
        mode = entry["mode"]
        _require(entry.get("committedImageName"), f"{mode}: Electron HITL commit receipt missing")
        _prepare_landmarks(entry)
        landmark_result = _train_landmarks(entry, "reviewed")
        obb_result = _train_obb(entry, "reviewed")
        after_landmark = _landmark_snapshot(entry)
        after_obb = _obb_snapshot(entry)
        _assert_frozen(entry["baseline"]["landmark"], after_landmark, mode, "landmark")
        _assert_frozen(entry["baseline"]["obb"], after_obb, mode, "OBB")
        hitl = _assert_hitl_train_only(entry)
        inference = _exercise_inference(entry, landmark_result, obb_result)
        semantics = _assert_export_semantics(entry)
        improvements = _assert_retraining_promotions(entry, landmark_result, obb_result)
        summaries.append(
            {
                "mode": mode,
                "hitl": hitl,
                "inference": inference,
                "semantics": semantics,
                "improvements": improvements,
                "frozenLandmarkValidationSources": len(after_landmark["validationIds"]),
                "frozenLandmarkTestSources": len(after_landmark["testIds"]),
                "frozenObbValidationGroups": len(after_obb["validationGroups"]),
                "frozenObbTestGroups": len(after_obb["testGroups"]),
            }
        )
    summary = {
        "ok": True,
        "modes": summaries,
        "boundaries": [
            "annotate/finalize",
            "landmark prepare",
            "OBB export",
            "mocked OBB train/inference",
            "mocked dlib train/inference",
            "Electron transactional HITL commit",
            "reprepare/retrain",
            "validation-gated promotion",
            "report-only test",
        ],
        "limitations": [
            "Model optimization and metrics are deterministic mocks; this harness proves orchestration and contracts, not biological generalization.",
            "It performs filesystem-real production preparation/export/registry work but does not require CUDA, a network, or Ultralytics/dlib fitting.",
        ],
    }
    state["summary"] = summary
    _write_json(state_path, state)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initial = subparsers.add_parser("initial")
    initial.add_argument("--workspace", required=True, type=Path)
    initial.add_argument("--state", required=True, type=Path)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--state", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "initial":
        state = run_initial(args.workspace.resolve(), args.state.resolve())
        print(json.dumps({"ok": True, "modes": [entry["mode"] for entry in state["modes"]]}))
    else:
        summary = run_resume(args.state.resolve())
        print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
