import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import backend.annotation.super_annotator as super_annotator_module
from backend.annotation.super_annotator import (
    SuperAnnotator,
    _assert_obb_training_inputs_unchanged,
    _build_obb_evaluator_protocol,
    _capture_obb_training_input_guard,
    _extract_yolo_metrics,
    _obb_evaluator_protocol_fingerprint,
    _obb_evaluator_protocols_match,
    _resolve_obb_training_epochs,
    _select_common_obb_promotion_metric,
)


def _locked_cohort(token="a"):
    digest = str(token) * 64
    return {
        "format_version": 1,
        "revision": f"locked-v1@{digest}",
        "sha256": digest,
        "split_profile_key": "seed=1337;val_ratio=0.20000000",
        "sample_count": 2,
        "group_count": 2,
        "frozen": True,
    }


def _locked_cohort_v2(source_token="a", material_token="m"):
    cohort = _locked_cohort(source_token)
    cohort.update(
        {
            "format_version": 2,
            "export_manifest_sha256": str(material_token) * 64,
        }
    )
    return cohort


def _locked_test_cohort(token="t"):
    return {
        **_locked_cohort(token),
        "report_only": True,
    }


class _FakeBoxMetrics:
    map50 = np.float32(0.91)
    map = np.float32(0.67)
    mp = np.float32(0.82)
    mr = np.float32(0.74)
    p = np.asarray([0.84, 0.80], dtype=np.float32)
    r = np.asarray([0.76, 0.72], dtype=np.float32)
    ap50 = np.asarray([0.93, 0.89], dtype=np.float32)
    ap = np.asarray([0.70, 0.64], dtype=np.float32)
    ap_class_index = np.asarray([0, 1], dtype=np.int64)


class _FakeMetrics:
    box = _FakeBoxMetrics()
    results_dict = {
        "metrics/precision(B)": np.float32(0.82),
        "metrics/recall(B)": np.float32(0.74),
        "metrics/mAP50(B)": np.float32(0.91),
        "metrics/mAP50-95(B)": np.float32(0.67),
        "fitness": np.float32(0.69),
    }


class _FakeYolo:
    instances = []
    metric_score = 0.67
    test_metric_score = 0.97

    def __init__(self, model_path):
        self.model_path = model_path
        self.names = {0: "canonical", 1: "reversed"}
        self.metrics = None
        self.trainer = None
        self.callbacks = {}
        self.train_kwargs = None
        self.val_kwargs = []
        self.__class__.instances.append(self)

    def _smart_load(self, _name):
        return None

    def add_callback(self, name, callback):
        self.callbacks[name] = callback

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        weights_dir = Path(kwargs["project"]) / kwargs["name"] / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        score = float(self.__class__.metric_score)
        (weights_dir / "best.pt").write_bytes(f"mock-obb-weights:{score:.3f}".encode("ascii"))
        box = types.SimpleNamespace(
            map50=np.float32(min(1.0, score + 0.24)),
            map=np.float32(score),
            mp=np.float32(0.82),
            mr=np.float32(0.74),
            p=np.asarray([0.84, 0.80], dtype=np.float32),
            r=np.asarray([0.76, 0.72], dtype=np.float32),
            ap50=np.asarray([min(1.0, score + 0.26), min(1.0, score + 0.22)], dtype=np.float32),
            ap=np.asarray([score + 0.03, max(0.0, score - 0.03)], dtype=np.float32),
            ap_class_index=np.asarray([0, 1], dtype=np.int64),
        )
        return types.SimpleNamespace(
            box=box,
            results_dict={
                "metrics/precision(B)": np.float32(0.82),
                "metrics/recall(B)": np.float32(0.74),
                "metrics/mAP50(B)": np.float32(min(1.0, score + 0.24)),
                "metrics/mAP50-95(B)": np.float32(score),
                "fitness": np.float32(score),
            },
        )

    def val(self, **kwargs):
        self.val_kwargs.append(dict(kwargs))
        if kwargs.get("split") == "test":
            score = float(self.__class__.test_metric_score)
        else:
            confidence = float(kwargs.get("conf", 0.30))
            nms_iou = float(kwargs.get("iou", 0.30))
            score = float(self.__class__.metric_score)
            score += 0.08 - abs(confidence - 0.45) * 0.20 - abs(nms_iou - 0.50) * 0.10
            score = min(0.99, max(0.0, score))
        box = types.SimpleNamespace(
            map50=np.float32(min(1.0, score + 0.20)),
            map=np.float32(score),
            mp=np.float32(min(1.0, score + 0.10)),
            mr=np.float32(max(0.0, score - 0.10)),
            p=np.asarray([min(1.0, score + 0.10), score], dtype=np.float32),
            r=np.asarray([score, max(0.0, score - 0.10)], dtype=np.float32),
            ap50=np.asarray([min(1.0, score + 0.22), min(1.0, score + 0.18)], dtype=np.float32),
            ap=np.asarray([min(1.0, score + 0.02), max(0.0, score - 0.02)], dtype=np.float32),
            ap_class_index=np.asarray([0, 1], dtype=np.int64),
        )
        return types.SimpleNamespace(
            box=box,
            results_dict={
                "metrics/precision(B)": np.float32(min(1.0, score + 0.10)),
                "metrics/recall(B)": np.float32(max(0.0, score - 0.10)),
                "metrics/mAP50(B)": np.float32(min(1.0, score + 0.20)),
                "metrics/mAP50-95(B)": np.float32(score),
            },
        )


class ObbTrainingConfigurationTests(unittest.TestCase):
    def test_explicit_fifty_epochs_is_not_treated_as_auto(self):
        self.assertEqual(_resolve_obb_training_epochs(50, "cpu"), 50)
        self.assertEqual(_resolve_obb_training_epochs(50, "cuda"), 50)
        self.assertEqual(_resolve_obb_training_epochs(None, "cpu"), 30)
        self.assertEqual(_resolve_obb_training_epochs(None, "cuda"), 100)

    def test_non_positive_epoch_count_is_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_obb_training_epochs(0, "cpu")


class ObbTrainingInputGuardTests(unittest.TestCase):
    def _make_export(self, root):
        root = Path(root)
        image_path = root / "images" / "train" / "sample.png"
        label_path = root / "labels" / "train" / "sample.txt"
        image_path.parent.mkdir(parents=True)
        label_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"stable-image")
        label_path.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")

        controls = {}
        for name in (
            "dataset.yaml",
            "export_manifest.json",
            "cohort_manifest.json",
            "split_assignments.json",
        ):
            path = root / name
            path.write_text(f"control:{name}\n", encoding="utf-8")
            controls[name] = str(path)
        return {
            "yaml_path": controls["dataset.yaml"],
            "export_manifest_path": controls["export_manifest.json"],
            "cohort_manifest_path": controls["cohort_manifest.json"],
            "split_assignments_path": controls["split_assignments.json"],
            "effective_dataset": {"format_version": 1, "revision": "declared-v1"},
        }

    def test_post_fit_guard_accepts_unchanged_effective_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            export_result = self._make_export(temp_dir)
            before = _capture_obb_training_input_guard(export_result)

            verified = _assert_obb_training_inputs_unchanged(export_result, before)

            self.assertTrue(verified["postFitVerified"])
            self.assertEqual(verified["revision"], verified["postFitRevision"])
            self.assertEqual(len(verified["effectiveFiles"]), 2)

    def test_post_fit_guard_rejects_changed_or_added_effective_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            export_result = self._make_export(temp_dir)
            before = _capture_obb_training_input_guard(export_result)
            label_path = Path(temp_dir, "labels", "train", "sample.txt")
            label_path.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "changed during fit/evaluation"):
                _assert_obb_training_inputs_unchanged(export_result, before)

            label_path.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")
            Path(temp_dir, "images", "val", "added.jpg").parent.mkdir(parents=True)
            Path(temp_dir, "images", "val", "added.jpg").write_bytes(b"late-image")
            with self.assertRaisesRegex(RuntimeError, "changed during fit/evaluation"):
                _assert_obb_training_inputs_unchanged(export_result, before)

    def test_post_fit_guard_ignores_ultralytics_cache_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            export_result = self._make_export(temp_dir)
            before = _capture_obb_training_input_guard(export_result)
            Path(temp_dir, "labels", "train.cache").write_bytes(b"runtime-cache")

            verified = _assert_obb_training_inputs_unchanged(export_result, before)

            self.assertTrue(verified["postFitVerified"])


class ObbMetricExtractionTests(unittest.TestCase):
    def test_extracts_scalar_raw_and_per_class_metrics(self):
        metrics = _extract_yolo_metrics(
            _FakeMetrics(),
            class_names={0: "canonical", 1: "reversed"},
        )

        self.assertAlmostEqual(metrics["map50"], 0.91, places=5)
        self.assertAlmostEqual(metrics["map50_95"], 0.67, places=5)
        self.assertAlmostEqual(metrics["precision"], 0.82, places=5)
        self.assertAlmostEqual(metrics["recall"], 0.74, places=5)
        self.assertEqual(metrics["per_class"][1]["class_name"], "reversed")
        self.assertAlmostEqual(metrics["per_class"][1]["map50_95"], 0.64, places=5)
        json.dumps(metrics, allow_nan=False)

    def test_promotion_metric_selection_never_compares_different_names(self):
        self.assertEqual(
            _select_common_obb_promotion_metric(
                {"map50_95": 0.61},
                {"map50": 0.92},
            ),
            (None, None, None),
        )

    def test_evaluator_protocol_fingerprint_covers_metric_affecting_settings(self):
        base = _build_obb_evaluator_protocol(
            trainer_args=None,
            imgsz=640,
            batch=8,
            nms_iou=0.30,
        )
        changed = _build_obb_evaluator_protocol(
            trainer_args=None,
            imgsz=960,
            batch=8,
            nms_iou=0.30,
        )
        changed_nms = _build_obb_evaluator_protocol(
            trainer_args=None,
            imgsz=640,
            batch=8,
            nms_iou=0.61,
        )
        base_fingerprint = _obb_evaluator_protocol_fingerprint(base)
        changed_fingerprint = _obb_evaluator_protocol_fingerprint(changed)
        self.assertNotEqual(base_fingerprint, changed_fingerprint)
        self.assertNotEqual(
            base_fingerprint,
            _obb_evaluator_protocol_fingerprint(changed_nms),
        )
        self.assertTrue(
            _obb_evaluator_protocols_match(
                base,
                base_fingerprint,
                dict(base),
                base_fingerprint,
            )
        )
        self.assertFalse(
            _obb_evaluator_protocols_match(
                base,
                base_fingerprint,
                changed,
                changed_fingerprint,
            )
        )
        self.assertEqual(
            _select_common_obb_promotion_metric(
                {"map50_95": 0.61, "map50": 0.90},
                {"map50": 0.88},
            ),
            ("map50", 0.90, 0.88),
        )


class MockedObbTrainingRunTests(unittest.TestCase):
    @staticmethod
    def _fake_torch_module():
        module = types.ModuleType("torch")
        module.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        return module

    def test_mock_training_honors_epochs_and_persists_metrics(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        _FakeYolo.instances.clear()
        _FakeYolo.metric_score = 0.67

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = os.path.join(session_dir, "mock_dataset.yaml")
            Path(dataset_yaml).write_text("path: .\n", encoding="utf-8")
            annotator = SuperAnnotator()
            export_result = {
                "ok": True,
                "yaml_path": dataset_yaml,
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort(),
            }

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(
                annotator, "export_obb_dataset", return_value=export_result
            ):
                result = annotator.train_yolo_obb(
                    session_dir,
                    epochs=50,
                    model_tier="nano",
                    device="cpu",
                    sam2_enabled=False,
                )

            self.assertTrue(result["ok"])
            self.assertEqual(_FakeYolo.instances[0].train_kwargs["epochs"], 50)
            self.assertAlmostEqual(result["map50"], 0.91, places=5)
            self.assertAlmostEqual(result["map50_95"], 0.67, places=5)
            self.assertTrue(os.path.isfile(result["model_path"]))
            self.assertTrue(os.path.isfile(result["metrics_path"]))
            self.assertEqual(
                Path(result["config_path"]).parent,
                Path(result["artifact_path"]).parent,
            )
            self.assertTrue(Path(result["config_path"]).is_file())
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["path"], result["config_path"])
            self.assertEqual(
                manifest["evaluatorProtocolFingerprint"],
                _obb_evaluator_protocol_fingerprint(manifest["evaluatorProtocol"]),
            )
            registry = json.loads(
                Path(session_dir, "models", "obb_registry.json").read_text(encoding="utf-8")
            )
            self.assertEqual(registry["models"][0]["configPath"], result["config_path"])
            self.assertEqual(registry["models"][0]["config"], manifest["config"])
            self.assertEqual(manifest["config"]["relativePath"], "obb_config.json")
            self.assertEqual(
                manifest["config"]["sha256"],
                hashlib.sha256(Path(result["config_path"]).read_bytes()).hexdigest(),
            )
            self.assertEqual(
                registry["models"][0]["evaluatorProtocolFingerprint"],
                manifest["evaluatorProtocolFingerprint"],
            )
            with open(result["metrics_path"], "r", encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertAlmostEqual(persisted["precision"], 0.82, places=5)
            self.assertEqual(len(persisted["per_class"]), 2)

    def test_mid_run_dataset_mutation_fails_before_registry_publication(self):
        class _MutatingYolo(_FakeYolo):
            dataset_yaml = None

            def train(self, **kwargs):
                result = super().train(**kwargs)
                Path(self.__class__.dataset_yaml).write_text(
                    "path: changed-during-fit\n",
                    encoding="utf-8",
                )
                return result

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _MutatingYolo
        fake_ultralytics.__version__ = "mock-obb-mutation-1.0"

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-guard-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = Path(session_dir, "mock_dataset.yaml")
            dataset_yaml.write_text("path: .\n", encoding="utf-8")
            _MutatingYolo.dataset_yaml = str(dataset_yaml)
            export_result = {
                "ok": True,
                "yaml_path": str(dataset_yaml),
                "warnings": [],
                "synthetic": {},
            }
            annotator = SuperAnnotator()

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(annotator, "export_obb_dataset", return_value=export_result):
                with self.assertRaisesRegex(RuntimeError, "changed during fit/evaluation"):
                    annotator.train_yolo_obb(
                        session_dir,
                        epochs=1,
                        device="cpu",
                        sam2_enabled=False,
                    )

            self.assertFalse(Path(session_dir, "models", "obb_registry.json").exists())

    def test_training_protocol_persists_deterministic_settings_and_descriptors(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        fake_ultralytics.__version__ = "mock-obb-protocol-1.0"
        _FakeYolo.instances.clear()
        _FakeYolo.metric_score = 0.67

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = Path(session_dir, "mock_dataset.yaml")
            dataset_yaml.write_text(
                "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n",
                encoding="utf-8",
            )
            snapshot_paths = {}
            for key, filename, payload in (
                ("export_manifest_path", "export_manifest.json", {"export": 1}),
                ("cohort_manifest_path", "cohort_manifest.json", {"cohort": 1}),
                ("split_assignments_path", "split_assignments.json", {"splits": 1}),
                ("synthetic_manifest_path", "synth_manifest.json", [{"synthetic": 1}]),
            ):
                path = Path(session_dir, filename)
                path.write_text(json.dumps(payload), encoding="utf-8")
                snapshot_paths[key] = str(path)
            effective_revision = "e" * 64
            export_result = {
                "ok": True,
                "yaml_path": str(dataset_yaml),
                **snapshot_paths,
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort("p"),
                "effective_dataset": {
                    "format_version": 1,
                    "task": "obb",
                    "real_files": [],
                    "synthetic_files": [],
                    "revision": effective_revision,
                },
            }
            annotator = SuperAnnotator()

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(annotator, "export_obb_dataset", return_value=export_result):
                result = annotator.train_yolo_obb(
                    session_dir,
                    epochs=7,
                    batch=4,
                    imgsz=960,
                    device="cpu",
                    sam2_enabled=False,
                    iou_loss=0.41,
                    cls_loss=1.7,
                    box_loss=4.2,
                    seed=12345,
                )

            self.assertTrue(result["ok"], result)
            train_kwargs = _FakeYolo.instances[0].train_kwargs
            self.assertEqual(train_kwargs["seed"], 12345)
            self.assertIs(train_kwargs["deterministic"], True)

            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            descriptor = manifest["trainingProtocol"]
            self.assertEqual(descriptor["format"], "biovision.obb-training-protocol.v1")
            self.assertEqual(descriptor["relativePath"], "training_protocol.json")
            protocol_path = Path(descriptor["path"])
            self.assertEqual(protocol_path.parent, Path(result["artifact_path"]).parent)
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            self.assertEqual(
                descriptor["sha256"],
                hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            )
            protocol_material = dict(protocol)
            protocol_revision = protocol_material.pop("revision")
            self.assertEqual(protocol_revision, super_annotator_module.lineage.sha256_json(protocol_material))
            self.assertEqual(descriptor["revision"], protocol_revision)
            self.assertEqual(protocol["effectiveDatasetRevision"], effective_revision)
            guard_descriptor = manifest["datasetExport"]["trainingInputGuard"]
            guard_path = Path(guard_descriptor["path"])
            guard = json.loads(guard_path.read_text(encoding="utf-8"))
            self.assertTrue(guard["postFitVerified"])
            self.assertEqual(guard["revision"], guard["postFitRevision"])
            self.assertEqual(protocol["trainingInputGuardRevision"], guard["revision"])
            self.assertEqual(
                guard_descriptor["sha256"],
                hashlib.sha256(guard_path.read_bytes()).hexdigest(),
            )
            hyperparameters = protocol["hyperparameters"]
            self.assertEqual(
                {
                    "epochs": hyperparameters["epochs"],
                    "batch": hyperparameters["batch"],
                    "imgsz": hyperparameters["imgsz"],
                    "workers": hyperparameters["workers"],
                    "device": hyperparameters["device"],
                    "seed": hyperparameters["seed"],
                    "deterministic": hyperparameters["deterministic"],
                },
                {
                    "epochs": 7,
                    "batch": 4,
                    "imgsz": 960,
                    "workers": 0,
                    "device": "cpu",
                    "seed": 12345,
                    "deterministic": True,
                },
            )
            self.assertEqual(hyperparameters["horizontalFlipProbability"], 0.0)
            self.assertEqual(hyperparameters["verticalFlipProbability"], 0.0)
            self.assertEqual(hyperparameters["mosaicProbability"], 0.0)
            self.assertAlmostEqual(hyperparameters["validationNmsIou"], 0.41)
            self.assertAlmostEqual(hyperparameters["classificationLossGain"], 1.7)
            self.assertAlmostEqual(hyperparameters["boxLossGain"], 4.2)
            self.assertEqual(
                protocol["evaluatorProtocol"],
                manifest["evaluatorProtocol"],
            )
            self.assertEqual(protocol["promotionPolicy"], manifest["promotionPolicy"])

            immutable = manifest["datasetExport"]["immutableSnapshots"]
            self.assertEqual(
                set(immutable),
                {
                    "datasetYaml",
                    "exportManifest",
                    "cohortManifest",
                    "splitAssignments",
                    "syntheticManifest",
                },
            )
            for snapshot in immutable.values():
                self.assertTrue(Path(snapshot["path"]).is_file())
                self.assertEqual(
                    snapshot["sha256"],
                    hashlib.sha256(Path(snapshot["path"]).read_bytes()).hexdigest(),
                )
            self.assertEqual(
                manifest["lineage"]["trainingProtocol"],
                {
                    "revision": protocol_revision,
                    "descriptorSha256": descriptor["sha256"],
                },
            )
            effective_descriptor = manifest["datasetExport"]["effectiveDataset"]
            self.assertEqual(effective_descriptor["revision"], effective_revision)
            self.assertEqual(
                manifest["lineage"]["effectiveDataset"],
                {
                    "revision": effective_revision,
                    "descriptorSha256": effective_descriptor["sha256"],
                },
            )
            self.assertEqual(
                manifest["lineage"]["trainingInputGuard"],
                {
                    "revision": guard["revision"],
                    "descriptorSha256": guard_descriptor["sha256"],
                    "postFitVerified": True,
                },
            )
            registry = json.loads(
                Path(session_dir, "models", "obb_registry.json").read_text(encoding="utf-8")
            )
            record = registry["models"][0]
            self.assertEqual(record["trainingProtocol"], descriptor)
            self.assertEqual(record["effectiveDataset"], effective_descriptor)
            self.assertEqual(record["trainingInputGuard"], guard_descriptor)
            self.assertEqual(record["datasetSnapshots"], immutable)

    def test_validation_calibration_and_report_only_test_are_auditable_and_separate(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        fake_ultralytics.__version__ = "mock-obb-2.0"
        _FakeYolo.instances.clear()
        _FakeYolo.metric_score = 0.61
        _FakeYolo.test_metric_score = 0.93

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = os.path.join(session_dir, "mock_dataset.yaml")
            Path(dataset_yaml).write_text(
                "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n",
                encoding="utf-8",
            )
            export_result = {
                "ok": True,
                "yaml_path": dataset_yaml,
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort("c"),
                "test_cohort": _locked_test_cohort("d"),
            }
            annotator = SuperAnnotator()
            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(annotator, "export_obb_dataset", return_value=export_result):
                result = annotator.train_yolo_obb(
                    session_dir,
                    epochs=1,
                    sam2_enabled=False,
                    iou_loss=0.30,
                )

            self.assertTrue(result["ok"], result)
            self.assertTrue(result["promotion"]["promoted"])
            self.assertEqual(result["promotion"]["metricSource"], "frozen_validation_only")
            self.assertEqual(result["promotion"]["testReportInfluence"], "none")
            self.assertAlmostEqual(result["promotion"]["candidateScore"], 0.61, places=5)
            self.assertAlmostEqual(result["test_metrics"]["map50_95"], 0.93, places=5)
            self.assertEqual(result["test_report_status"], "completed")

            config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
            self.assertEqual(config["thresholdCalibration"]["status"], "completed")
            self.assertEqual(config["thresholdCalibration"]["role"], "validation_only")
            self.assertEqual(config["thresholdCalibration"]["promotionInfluence"], "none")
            self.assertAlmostEqual(config["confidence_threshold"], 0.45)
            self.assertAlmostEqual(config["nms_iou"], 0.50)
            self.assertEqual(len(config["thresholdCalibration"]["evidence"]), 9)
            selected_evidence = next(
                entry
                for entry in config["thresholdCalibration"]["evidence"]
                if entry["confidenceThreshold"] == 0.45 and entry["nmsIou"] == 0.50
            )
            default_evidence = next(
                entry
                for entry in config["thresholdCalibration"]["evidence"]
                if entry["confidenceThreshold"] == 0.30 and entry["nmsIou"] == 0.30
            )
            self.assertGreater(
                selected_evidence["objective"],
                default_evidence["objective"],
                "the mocked validation sweep should select a measured F1 improvement",
            )

            validation_calls = [
                kwargs
                for instance in _FakeYolo.instances
                for kwargs in instance.val_kwargs
                if kwargs.get("split") == "val"
            ]
            test_calls = [
                kwargs
                for instance in _FakeYolo.instances
                for kwargs in instance.val_kwargs
                if kwargs.get("split") == "test"
            ]
            self.assertEqual(len(validation_calls), 9)
            self.assertEqual(len(test_calls), 1)
            self.assertAlmostEqual(test_calls[0]["conf"], config["confidence_threshold"])
            self.assertAlmostEqual(test_calls[0]["iou"], config["nms_iou"])

            report = json.loads(Path(result["test_report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["role"], "report_only")
            self.assertEqual(report["promotionInfluence"], "none")
            self.assertEqual(report["cohort"]["sha256"], "d" * 64)
            manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
            validation_protocol = manifest["calibratedValidationEvaluatorProtocol"]
            test_protocol = report["evaluatorProtocol"]
            self.assertEqual(validation_protocol["split"], "val")
            self.assertEqual(test_protocol["split"], "test")
            self.assertEqual(
                {key: value for key, value in validation_protocol.items() if key != "split"},
                {key: value for key, value in test_protocol.items() if key != "split"},
            )

    def test_rejected_candidate_never_invokes_or_exposes_frozen_test_metrics(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        fake_ultralytics.__version__ = "mock-obb-blind-test-1.0"
        _FakeYolo.instances.clear()
        _FakeYolo.test_metric_score = 0.99

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = Path(session_dir, "mock_dataset.yaml")
            dataset_yaml.write_text(
                "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n",
                encoding="utf-8",
            )
            export_result = {
                "ok": True,
                "yaml_path": str(dataset_yaml),
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort("b"),
                "test_cohort": _locked_test_cohort("z"),
            }
            annotator = SuperAnnotator()

            def test_calls():
                return [
                    kwargs
                    for instance in _FakeYolo.instances
                    for kwargs in instance.val_kwargs
                    if kwargs.get("split") == "test"
                ]

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(annotator, "export_obb_dataset", return_value=export_result):
                _FakeYolo.metric_score = 0.70
                promoted = annotator.train_yolo_obb(
                    session_dir,
                    epochs=1,
                    sam2_enabled=False,
                )
                self.assertTrue(promoted["promotion"]["promoted"])
                self.assertEqual(len(test_calls()), 1)
                self.assertEqual(promoted["test_report_status"], "completed")
                self.assertAlmostEqual(promoted["test_metrics"]["map50_95"], 0.99, places=5)

                _FakeYolo.metric_score = 0.40
                rejected = annotator.train_yolo_obb(
                    session_dir,
                    epochs=1,
                    sam2_enabled=False,
                )

            self.assertFalse(rejected["promotion"]["promoted"])
            self.assertEqual(rejected["model_status"], "candidate")
            self.assertEqual(
                rejected["promotion"]["testReportInfluence"],
                "none",
            )
            self.assertEqual(
                len(test_calls()),
                1,
                "a rejected candidate must not invoke the frozen test evaluator",
            )
            self.assertEqual(rejected["test_metrics"], {})
            self.assertEqual(rejected["test_report_status"], "not_run")
            self.assertEqual(
                rejected["test_report_not_run_reason"],
                "candidate_not_promoted",
            )

            report = json.loads(
                Path(rejected["test_report_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "not_run")
            self.assertEqual(report["notRunReason"], "candidate_not_promoted")
            self.assertEqual(report["promotionInfluence"], "none")
            self.assertEqual(report["metrics"], {})
            self.assertEqual(
                report["promotionDecision"],
                {
                    "promoted": False,
                    "reason": rejected["promotion"]["reason"],
                },
            )
            manifest = json.loads(
                Path(rejected["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["testReport"]["status"], "not_run")
            self.assertEqual(
                manifest["testReport"]["notRunReason"],
                "candidate_not_promoted",
            )
            registry = json.loads(
                Path(session_dir, "models", "obb_registry.json").read_text(encoding="utf-8")
            )
            candidate_record = next(
                record
                for record in registry["models"]
                if record["modelId"] == rejected["model_id"]
            )
            self.assertEqual(candidate_record["testReport"]["status"], "not_run")
            self.assertEqual(candidate_record["testReport"]["metrics"], {})
            self.assertEqual(
                candidate_record["testReport"]["notRunReason"],
                "candidate_not_promoted",
            )

    def test_first_model_without_frozen_validation_remains_candidate(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        _FakeYolo.instances.clear()
        _FakeYolo.metric_score = 0.72

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = os.path.join(session_dir, "mock_dataset.yaml")
            Path(dataset_yaml).write_text("path: .\n", encoding="utf-8")
            annotator = SuperAnnotator()
            export_result = {
                "ok": True,
                "yaml_path": dataset_yaml,
                "warnings": [],
                "synthetic": {},
                "validation_cohort": None,
                "test_cohort": None,
            }
            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(annotator, "export_obb_dataset", return_value=export_result):
                result = annotator.train_yolo_obb(
                    session_dir,
                    epochs=1,
                    sam2_enabled=False,
                )

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["model_status"], "candidate")
            self.assertFalse(result["promotion"]["promoted"])
            self.assertEqual(
                result["promotion"]["reason"],
                "candidate_missing_frozen_validation_cohort",
            )
            self.assertFalse(
                Path(session_dir, "models", "session_obb_detector.pt").exists()
            )

    def test_first_directional_model_without_full_validation_class_coverage_is_candidate(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        fake_ultralytics.__version__ = "mock-obb-class-coverage-1.0"
        _FakeYolo.instances.clear()
        _FakeYolo.metric_score = 0.88

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "directional"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-directional-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = Path(session_dir, "mock_dataset.yaml")
            dataset_yaml.write_text("path: .\n", encoding="utf-8")
            validation_cohort = {
                **_locked_cohort_v2("v", "c"),
                "expected_class_count": 2,
                "real_class_histogram": {"0": 2, "1": 0},
            }
            export_result = {
                "ok": True,
                "yaml_path": str(dataset_yaml),
                "warnings": [],
                "synthetic": {},
                "validation_cohort": validation_cohort,
            }
            annotator = SuperAnnotator()

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(annotator, "export_obb_dataset", return_value=export_result):
                result = annotator.train_yolo_obb(
                    session_dir,
                    epochs=1,
                    sam2_enabled=False,
                    orientation_schema="directional",
                )

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["model_status"], "candidate")
            self.assertFalse(result["promotion"]["promoted"])
            self.assertEqual(
                result["promotion"]["reason"],
                "candidate_validation_class_coverage_incomplete",
            )
            persisted_cohort = result["promotion"]["candidateCohort"]
            self.assertEqual(persisted_cohort["expectedClassCount"], 2)
            self.assertEqual(
                persisted_cohort["realClassHistogram"],
                {"0": 2, "1": 0},
            )
            policy = result["promotion"]["improvementPolicy"]
            self.assertTrue(policy["requireAllConfiguredValidationClasses"])
            self.assertEqual(policy["minimumValidationSamples"], 2)
            self.assertEqual(policy["minimumValidationGroups"], 2)
            self.assertFalse(
                Path(session_dir, "models", "session_obb_detector.pt").exists()
            )
            registry = json.loads(
                Path(session_dir, "models", "obb_registry.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                registry["models"][0]["validationCohort"],
                persisted_cohort,
            )

    def test_later_candidate_requires_two_validation_samples_and_groups(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        fake_ultralytics.__version__ = "mock-obb-min-evidence-1.0"
        _FakeYolo.instances.clear()

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = Path(session_dir, "mock_dataset.yaml")
            dataset_yaml.write_text("path: .\n", encoding="utf-8")
            baseline_cohort = {
                **_locked_cohort_v2("e", "e"),
                "expected_class_count": 1,
                "real_class_histogram": {"0": 2},
            }
            export_result = {
                "ok": True,
                "yaml_path": str(dataset_yaml),
                "warnings": [],
                "synthetic": {},
                "validation_cohort": baseline_cohort,
            }
            annotator = SuperAnnotator()

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(
                annotator, "export_obb_dataset", return_value=export_result
            ) as exporter:
                _FakeYolo.metric_score = 0.40
                baseline = annotator.train_yolo_obb(
                    session_dir, epochs=1, sam2_enabled=False
                )
                active_alias = Path(session_dir, "models", "session_obb_detector.pt")
                baseline_alias = active_alias.read_bytes()

                exporter.return_value = {
                    **export_result,
                    "validation_cohort": {
                        **baseline_cohort,
                        "sample_count": 1,
                        "group_count": 2,
                        "real_class_histogram": {"0": 1},
                    },
                }
                _FakeYolo.metric_score = 0.90
                too_few_samples = annotator.train_yolo_obb(
                    session_dir, epochs=1, sam2_enabled=False
                )

                exporter.return_value = {
                    **export_result,
                    "validation_cohort": {
                        **baseline_cohort,
                        "sample_count": 2,
                        "group_count": 1,
                    },
                }
                too_few_groups = annotator.train_yolo_obb(
                    session_dir, epochs=1, sam2_enabled=False
                )

            self.assertEqual(baseline["model_status"], "active")
            self.assertEqual(too_few_samples["model_status"], "candidate")
            self.assertEqual(
                too_few_samples["promotion"]["reason"],
                "candidate_insufficient_validation_samples",
            )
            self.assertEqual(too_few_groups["model_status"], "candidate")
            self.assertEqual(
                too_few_groups["promotion"]["reason"],
                "candidate_insufficient_validation_groups",
            )
            self.assertEqual(active_alias.read_bytes(), baseline_alias)
            registry = json.loads(
                Path(session_dir, "models", "obb_registry.json").read_text(encoding="utf-8")
            )
            active_records = [
                record for record in registry["models"] if record["status"] == "active"
            ]
            self.assertEqual([record["modelId"] for record in active_records], [baseline["model_id"]])

    def test_locked_validation_metric_gates_obb_alias_promotion(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        _FakeYolo.instances.clear()

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = os.path.join(session_dir, "mock_dataset.yaml")
            Path(dataset_yaml).write_text("path: .\n", encoding="utf-8")
            annotator = SuperAnnotator()
            export_result = {
                "ok": True,
                "yaml_path": dataset_yaml,
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort(),
            }

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(
                annotator, "export_obb_dataset", return_value=export_result
            ):
                _FakeYolo.metric_score = 0.67
                first = annotator.train_yolo_obb(session_dir, epochs=1, sam2_enabled=False)
                active_alias = Path(session_dir, "models", "session_obb_detector.pt")
                first_alias = active_alias.read_bytes()

                _FakeYolo.metric_score = 0.40
                regressed = annotator.train_yolo_obb(session_dir, epochs=1, sam2_enabled=False)
                self.assertEqual(regressed["model_status"], "candidate")
                self.assertFalse(regressed["promotion"]["promoted"])
                regressed_manifest = json.loads(
                    Path(regressed["manifest_path"]).read_text(encoding="utf-8")
                )
                self.assertIsNone(regressed_manifest["lineage"]["parentModelId"])
                self.assertEqual(
                    regressed_manifest["lineage"]["baselineModelId"],
                    first["model_id"],
                )
                self.assertEqual(
                    regressed_manifest["lineage"]["initialization"]["strategy"],
                    "pretrained_checkpoint",
                )
                self.assertEqual(
                    regressed_manifest["lineage"]["initialization"]["checkpoint"]["requested"],
                    "yolov8n-obb.pt",
                )
                self.assertEqual(regressed["promotion"]["metric"], "map50_95")
                self.assertEqual(regressed["promotion"]["baselineMetric"], "map50_95")
                self.assertEqual(
                    regressed["promotion"]["candidateCohort"]["sha256"],
                    regressed["promotion"]["baselineCohort"]["sha256"],
                )
                self.assertEqual(active_alias.read_bytes(), first_alias)

                _FakeYolo.metric_score = 0.80
                improved = annotator.train_yolo_obb(session_dir, epochs=1, sam2_enabled=False)
                self.assertEqual(improved["model_status"], "active")
                self.assertTrue(improved["promotion"]["promoted"])
                self.assertNotEqual(active_alias.read_bytes(), first_alias)
                improved_alias = active_alias.read_bytes()

                _FakeYolo.metric_score = 0.80
                tied = annotator.train_yolo_obb(session_dir, epochs=1, sam2_enabled=False)
                self.assertEqual(tied["model_status"], "candidate")
                self.assertEqual(tied["promotion"]["reason"], "locked_cohort_not_improved")
                self.assertEqual(active_alias.read_bytes(), improved_alias)

            registry = json.loads(Path(session_dir, "models", "obb_registry.json").read_text(encoding="utf-8"))
            statuses = [entry["status"] for entry in registry["models"]]
            self.assertEqual(statuses.count("active"), 1)
            self.assertEqual(statuses.count("candidate"), 1)
            self.assertEqual(statuses.count("deprecated"), 2)
            self.assertEqual(first["model_status"], "active")

    def test_tiny_locked_validation_gain_below_effect_policy_is_not_promoted(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        fake_ultralytics.__version__ = "mock-obb-effect-policy-1.0"
        _FakeYolo.instances.clear()

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = Path(session_dir, "mock_dataset.yaml")
            dataset_yaml.write_text("path: .\n", encoding="utf-8")
            export_result = {
                "ok": True,
                "yaml_path": str(dataset_yaml),
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort("g"),
            }
            annotator = SuperAnnotator()

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(annotator, "export_obb_dataset", return_value=export_result):
                _FakeYolo.metric_score = 0.60
                baseline = annotator.train_yolo_obb(
                    session_dir, epochs=1, sam2_enabled=False
                )
                active_alias = Path(session_dir, "models", "session_obb_detector.pt")
                baseline_alias = active_alias.read_bytes()

                # +0.001 is positive, but below the persisted relative floor:
                # 0.60 * 0.005 = 0.003.
                _FakeYolo.metric_score = 0.601
                candidate = annotator.train_yolo_obb(
                    session_dir, epochs=1, sam2_enabled=False
                )

            self.assertEqual(baseline["model_status"], "active")
            self.assertEqual(candidate["model_status"], "candidate")
            promotion = candidate["promotion"]
            self.assertFalse(promotion["promoted"])
            self.assertEqual(
                promotion["reason"],
                "locked_cohort_improvement_below_minimum",
            )
            self.assertAlmostEqual(promotion["observedImprovement"], 0.001, places=5)
            self.assertAlmostEqual(promotion["requiredImprovement"], 0.003, places=6)
            self.assertGreater(
                promotion["requiredImprovement"],
                promotion["observedImprovement"],
            )
            policy = promotion["improvementPolicy"]
            self.assertEqual(policy["policyVersion"], "obb_map_effect_v1")
            self.assertAlmostEqual(policy["minimumAbsoluteImprovement"], 0.001)
            self.assertAlmostEqual(policy["minimumRelativeImprovement"], 0.005)
            self.assertEqual(active_alias.read_bytes(), baseline_alias)

            manifest = json.loads(Path(candidate["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["promotionPolicy"], policy)
            self.assertEqual(manifest["trainingProtocol"]["format"], "biovision.obb-training-protocol.v1")
            registry = json.loads(
                Path(session_dir, "models", "obb_registry.json").read_text(encoding="utf-8")
            )
            active = next(record for record in registry["models"] if record["status"] == "active")
            persisted_candidate = next(
                record for record in registry["models"] if record["modelId"] == candidate["model_id"]
            )
            self.assertEqual(active["modelId"], baseline["model_id"])
            self.assertEqual(persisted_candidate["promotion"], promotion)

    def test_better_metric_from_different_validation_cohort_remains_candidate(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        _FakeYolo.instances.clear()

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = os.path.join(session_dir, "mock_dataset.yaml")
            Path(dataset_yaml).write_text("path: .\n", encoding="utf-8")
            annotator = SuperAnnotator()
            first_export = {
                "ok": True,
                "yaml_path": dataset_yaml,
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort("a"),
            }
            second_export = {
                **first_export,
                "validation_cohort": _locked_cohort("b"),
            }

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(
                annotator, "export_obb_dataset", return_value=first_export
            ) as exporter:
                _FakeYolo.metric_score = 0.40
                annotator.train_yolo_obb(session_dir, epochs=1, sam2_enabled=False)
                active_alias = Path(session_dir, "models", "session_obb_detector.pt")
                original_alias = active_alias.read_bytes()

                exporter.return_value = second_export
                _FakeYolo.metric_score = 0.90
                incomparable = annotator.train_yolo_obb(
                    session_dir, epochs=1, sam2_enabled=False
                )

            self.assertEqual(incomparable["model_status"], "candidate")
            self.assertFalse(incomparable["promotion"]["promoted"])
            self.assertEqual(
                incomparable["promotion"]["reason"],
                "frozen_validation_cohort_mismatch",
            )
            self.assertNotEqual(
                incomparable["promotion"]["candidateCohort"]["sha256"],
                incomparable["promotion"]["baselineCohort"]["sha256"],
            )
            self.assertEqual(active_alias.read_bytes(), original_alias)

    def test_changed_v2_exported_validation_material_remains_candidate(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        fake_ultralytics.__version__ = "mock-obb-material-cohort-1.0"
        _FakeYolo.instances.clear()

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = Path(session_dir, "mock_dataset.yaml")
            dataset_yaml.write_text("path: .\n", encoding="utf-8")
            baseline_export = {
                "ok": True,
                "yaml_path": str(dataset_yaml),
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort_v2("s", "a"),
            }
            changed_material_export = {
                **baseline_export,
                # Source cohort identity is unchanged; only the exact exported
                # validation image/label material closure differs.
                "validation_cohort": _locked_cohort_v2("s", "b"),
            }
            annotator = SuperAnnotator()

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(
                annotator, "export_obb_dataset", return_value=baseline_export
            ) as exporter:
                _FakeYolo.metric_score = 0.40
                baseline = annotator.train_yolo_obb(
                    session_dir, epochs=1, sam2_enabled=False
                )
                active_alias = Path(session_dir, "models", "session_obb_detector.pt")
                original_alias = active_alias.read_bytes()

                exporter.return_value = changed_material_export
                _FakeYolo.metric_score = 0.90
                candidate = annotator.train_yolo_obb(
                    session_dir, epochs=1, sam2_enabled=False
                )

            self.assertEqual(baseline["model_status"], "active")
            self.assertEqual(candidate["model_status"], "candidate")
            self.assertFalse(candidate["promotion"]["promoted"])
            self.assertEqual(
                candidate["promotion"]["reason"],
                "frozen_validation_cohort_mismatch",
            )
            candidate_cohort = candidate["promotion"]["candidateCohort"]
            baseline_cohort = candidate["promotion"]["baselineCohort"]
            self.assertEqual(candidate_cohort["formatVersion"], 2)
            self.assertEqual(candidate_cohort["sha256"], baseline_cohort["sha256"])
            self.assertEqual(candidate_cohort["revision"], baseline_cohort["revision"])
            self.assertNotEqual(
                candidate_cohort["exportManifestSha256"],
                baseline_cohort["exportManifestSha256"],
            )
            self.assertEqual(active_alias.read_bytes(), original_alias)

            registry = json.loads(
                Path(session_dir, "models", "obb_registry.json").read_text(encoding="utf-8")
            )
            persisted = {
                record["modelId"]: record["validationCohort"]
                for record in registry["models"]
            }
            self.assertEqual(
                persisted[baseline["model_id"]]["exportManifestSha256"],
                "a" * 64,
            )
            self.assertEqual(
                persisted[candidate["model_id"]]["exportManifestSha256"],
                "b" * 64,
            )

    def test_better_metric_from_different_evaluator_protocol_remains_candidate(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        fake_ultralytics.__version__ = "mock-1.0"
        _FakeYolo.instances.clear()

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = os.path.join(session_dir, "mock_dataset.yaml")
            Path(dataset_yaml).write_text("path: .\n", encoding="utf-8")
            annotator = SuperAnnotator()
            export_result = {
                "ok": True,
                "yaml_path": dataset_yaml,
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort(),
            }

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(annotator, "export_obb_dataset", return_value=export_result):
                _FakeYolo.metric_score = 0.40
                first = annotator.train_yolo_obb(
                    session_dir,
                    epochs=1,
                    imgsz=640,
                    sam2_enabled=False,
                )
                active_alias = Path(session_dir, "models", "session_obb_detector.pt")
                original_alias = active_alias.read_bytes()

                _FakeYolo.metric_score = 0.90
                incomparable = annotator.train_yolo_obb(
                    session_dir,
                    epochs=1,
                    imgsz=960,
                    sam2_enabled=False,
                )

            self.assertEqual(first["model_status"], "active")
            self.assertEqual(incomparable["model_status"], "candidate")
            self.assertEqual(
                incomparable["promotion"]["reason"],
                "evaluator_protocol_fingerprint_mismatch",
            )
            self.assertNotEqual(
                incomparable["promotion"]["candidateEvaluatorProtocolFingerprint"],
                incomparable["promotion"]["baselineEvaluatorProtocolFingerprint"],
            )
            self.assertEqual(active_alias.read_bytes(), original_alias)

    def test_alias_copy_failure_rolls_back_aliases_and_registry(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYolo
        _FakeYolo.instances.clear()

        with tempfile.TemporaryDirectory() as session_dir:
            Path(session_dir, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "schemaSemanticFingerprint": "v2-obb-test",
                        "schemaSemanticVersion": 2,
                        "landmarkTemplate": [],
                    }
                ),
                encoding="utf-8",
            )
            dataset_yaml = os.path.join(session_dir, "mock_dataset.yaml")
            Path(dataset_yaml).write_text("path: .\n", encoding="utf-8")
            annotator = SuperAnnotator()
            export_result = {
                "ok": True,
                "yaml_path": dataset_yaml,
                "warnings": [],
                "synthetic": {},
                "validation_cohort": _locked_cohort(),
            }
            models_dir = Path(session_dir, "models")
            model_alias = models_dir / "session_obb_detector.pt"
            config_alias = models_dir / "session_obb_detector_config.json"
            registry_path = models_dir / "obb_registry.json"

            with patch.dict(
                sys.modules,
                {"ultralytics": fake_ultralytics, "torch": self._fake_torch_module()},
            ), patch.object(
                annotator, "export_obb_dataset", return_value=export_result
            ):
                _FakeYolo.metric_score = 0.40
                annotator.train_yolo_obb(
                    session_dir,
                    epochs=1,
                    sam2_enabled=False,
                    iou_loss=0.30,
                )
                original_model_alias = model_alias.read_bytes()
                original_config_alias = config_alias.read_bytes()
                original_registry = registry_path.read_bytes()

                original_copy = super_annotator_module.lineage.atomic_copy_file
                injected = {"done": False}

                def fail_config_alias_once(source_path, destination_path):
                    if (
                        not injected["done"]
                        and os.path.abspath(destination_path) == os.path.abspath(config_alias)
                    ):
                        injected["done"] = True
                        raise OSError("injected OBB config alias copy failure")
                    return original_copy(source_path, destination_path)

                _FakeYolo.metric_score = 0.90
                with patch.object(
                    super_annotator_module.lineage,
                    "atomic_copy_file",
                    side_effect=fail_config_alias_once,
                ), self.assertRaisesRegex(OSError, "injected OBB config alias copy failure"):
                    annotator.train_yolo_obb(
                        session_dir,
                        epochs=1,
                        sam2_enabled=False,
                        iou_loss=0.30,
                    )

            self.assertTrue(injected["done"])
            self.assertEqual(model_alias.read_bytes(), original_model_alias)
            self.assertEqual(config_alias.read_bytes(), original_config_alias)
            self.assertEqual(registry_path.read_bytes(), original_registry)
            registry = json.loads(original_registry.decode("utf-8"))
            self.assertEqual([entry["status"] for entry in registry["models"]], ["active"])
            self.assertEqual(list(models_dir.glob(".obb-publish-*")), [])


class _PredictOnlyYolo:
    instances = []

    def __init__(self, model_path):
        self.model_path = model_path
        self.predict_kwargs = None
        self.__class__.instances.append(self)

    def predict(self, _image, **kwargs):
        self.predict_kwargs = kwargs
        return [types.SimpleNamespace(obb=None, names={})]


class ObbInferenceConfigurationTests(unittest.TestCase):
    def test_super_annotator_obb_paths_use_agnostic_nms_and_artifact_config(self):
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _PredictOnlyYolo
        _PredictOnlyYolo.instances.clear()
        annotator = SuperAnnotator()

        with tempfile.TemporaryDirectory() as artifact_dir:
            model_path = Path(artifact_dir, "model.pt")
            model_path.write_bytes(b"mock")
            Path(artifact_dir, "obb_config.json").write_text(
                json.dumps({"nms_iou": 0.63, "confidence_threshold": 0.15}), encoding="utf-8"
            )
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                with self.assertRaisesRegex(RuntimeError, "no oriented boxes"):
                    annotator.detect_finetuned(
                        np.zeros((16, 16, 3), dtype=np.uint8),
                        os.fspath(model_path),
                        "specimen",
                        nms_iou=None,
                    )
                detections = annotator.detect_obb(
                    "image.png",
                    os.fspath(model_path),
                    nms_iou=None,
                )

        self.assertEqual(detections, [])
        self.assertEqual(len(_PredictOnlyYolo.instances), 2)
        self.assertTrue(_PredictOnlyYolo.instances[0].predict_kwargs["agnostic_nms"])
        self.assertAlmostEqual(_PredictOnlyYolo.instances[0].predict_kwargs["iou"], 0.63)
        self.assertAlmostEqual(_PredictOnlyYolo.instances[0].predict_kwargs["conf"], 0.15)
        self.assertTrue(_PredictOnlyYolo.instances[1].predict_kwargs["agnostic_nms"])
        self.assertAlmostEqual(_PredictOnlyYolo.instances[1].predict_kwargs["iou"], 0.63)
        self.assertAlmostEqual(_PredictOnlyYolo.instances[1].predict_kwargs["conf"], 0.15)

    def test_default_annotate_uses_low_calibrated_artifact_threshold(self):
        annotator = SuperAnnotator()
        with tempfile.TemporaryDirectory() as artifact_dir:
            model_path = Path(artifact_dir, "model.pt")
            model_path.write_bytes(b"mock")
            Path(artifact_dir, "obb_config.json").write_text(
                json.dumps({"nms_iou": 0.62, "confidence_threshold": 0.15}),
                encoding="utf-8",
            )
            with patch.object(
                annotator,
                "_load_image",
                return_value=np.zeros((16, 16, 3), dtype=np.uint8),
            ), patch.object(annotator, "detect_finetuned", return_value=[]) as detect:
                result = annotator.annotate(
                    "image.png",
                    "specimen",
                    options={"finetuned_model": os.fspath(model_path)},
                )

        self.assertEqual(result["num_detections"], 0)
        self.assertAlmostEqual(detect.call_args.args[3], 0.15)
        self.assertAlmostEqual(detect.call_args.kwargs["nms_iou"], 0.62)

    def test_zero_shot_uses_model_nms_without_a_second_aabb_suppression(self):
        class EmptyZeroShotModel:
            def __init__(self):
                self.model = types.SimpleNamespace(clip_model=None)
                self.predict_kwargs = None
                self.classes = None

            def set_classes(self, classes):
                self.classes = classes

            def predict(self, _image, **kwargs):
                self.predict_kwargs = kwargs
                return [types.SimpleNamespace(boxes=None)]

        annotator = SuperAnnotator()
        annotator.yolo_model = EmptyZeroShotModel()
        self.assertEqual(
            annotator.detect_zero_shot(np.zeros((16, 16, 3), dtype=np.uint8), "specimen"),
            [],
        )
        self.assertTrue(annotator.yolo_model.predict_kwargs["agnostic_nms"])


if __name__ == "__main__":
    unittest.main()
