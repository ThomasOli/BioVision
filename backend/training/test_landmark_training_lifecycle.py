import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.bv_utils import lineage
from backend.data import prepare_dataset
from backend.training import train_shape_model


class MockedLandmarkTrainingLifecycleTests(unittest.TestCase):
    """Exercise preparation, HITL growth, frozen evaluation, and promotion."""

    TAG = "lifecycle"
    SCORES = {
        "run_v1": {"train": 0.18, "validation": 0.25, "test": 0.30},
        # Test deliberately regresses while validation improves: locked test is
        # report-only and must not select the active model.
        "run_v2": {"train": 0.09, "validation": 0.15, "test": 0.40},
        "run_v3": {"train": 0.08, "validation": 0.20, "test": 0.05},
    }

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.root = self._tempdir.name
        self.images_dir = os.path.join(self.root, "images")
        self.labels_dir = os.path.join(self.root, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        with open(os.path.join(self.root, "session.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schemaSemanticFingerprint": "v2-lifecycle000001",
                    "schemaSemanticVersion": 2,
                    "orientationPolicyConfigured": True,
                    "orientationPolicy": {"mode": "invariant"},
                    "landmarkTemplate": [
                        {"index": 1, "name": "anterior", "category": "tip", "required": True},
                        {"index": 2, "name": "posterior", "category": "tip", "required": True},
                    ],
                },
                handle,
            )

    def tearDown(self):
        self._tempdir.cleanup()

    def _add_sample(self, name, value, *, provenance=None, review_history=None):
        image = np.full((96, 96, 3), int(value), dtype=np.uint8)
        image[0, 0] = (value % 255, (value * 3) % 255, (value * 7) % 255)
        cv2.circle(image, (48, 48), 22, (255 - value, value // 2, value), thickness=2)
        image_path = os.path.join(self.images_dir, f"{name}.png")
        self.assertTrue(cv2.imwrite(image_path, image))

        landmarks = [
            {"id": 1, "x": 30 + value % 4, "y": 43, "isSkipped": False},
            {"id": 2, "x": 66 - value % 4, "y": 53, "isSkipped": False},
        ]
        box = {
            "left": 10,
            "top": 10,
            "width": 76,
            "height": 76,
            "obbCorners": [[10, 10], [86, 10], [86, 86], [10, 86]],
            "class_id": 0,
            "landmarks": landmarks,
        }
        label = {
            "imageFilename": f"{name}.png",
            "boxes": [box],
            "provenance": provenance or {"source": "manual_annotation"},
            "reviewHistory": list(review_history or []),
            "finalizedDetection": {
                "isFinalized": True,
                "acceptedBoxes": [box],
                "boxSignature": f"box-{name}",
            },
        }
        with open(os.path.join(self.labels_dir, f"{name}.json"), "w", encoding="utf-8") as handle:
            json.dump(label, handle, indent=2)
        return image_path, label

    def _add_hitl_sample(self, active_model_id):
        image_path, _ = self._add_sample("reviewed_hitl", 211)
        event = {
            "eventId": "commit-hitl-001",
            "commitId": "commit-hitl-001",
            "source": "hitl_review",
            "speciesId": "mock-species",
            "inferenceSessionId": "inference-session-001",
            "imageFilename": "reviewed_hitl.png",
            "sourceImageSha256": lineage.sha256_file(image_path),
            "landmarkModelKey": active_model_id,
            "landmarkPredictorType": "dlib",
            "detectionModelKey": "obb:mock-detector",
            "originalPredictionHash": "prediction-before-review",
            "reviewedPredictionHash": "prediction-after-review",
            "reviewOutcome": "corrected",
            "reviewer": "mock-reviewer",
            "reviewedAt": "2026-01-02T03:04:05Z",
            "acceptedSpecimens": 1,
            "rejectedDetections": 0,
            "detectionConfidence": {"min": 0.42, "max": 0.42, "mean": 0.42},
        }
        label_path = os.path.join(self.labels_dir, "reviewed_hitl.json")
        with open(label_path, "r", encoding="utf-8") as handle:
            label = json.load(handle)
        label["provenance"] = event
        label["reviewHistory"] = [event]
        with open(label_path, "w", encoding="utf-8") as handle:
            json.dump(label, handle, indent=2)
        lineage.atomic_write_json(os.path.join(self.root, "review_events.json"), [event])
        return event

    def _prepare(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            prepare_dataset.json_to_dlib_xml(
                self.root,
                self.TAG,
                test_split=0.34,
                seed=42,
            )
        with open(
            os.path.join(self.root, "debug", f"split_info_{self.TAG}.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)

    @staticmethod
    def _run_id_from_predictor(predictor_path):
        return Path(predictor_path).parent.name

    def _fake_train(self, _xml_path, predictor_path, _options):
        run_id = self._run_id_from_predictor(predictor_path)
        Path(predictor_path).write_bytes(f"mock-predictor:{run_id}".encode("ascii"))

    def _score(self, xml_path, predictor_path):
        run_id = self._run_id_from_predictor(predictor_path)
        basename = os.path.basename(xml_path)
        cohort = (
            "validation"
            if basename.startswith("validation_")
            else "test"
            if basename.startswith("test_")
            else "train"
        )
        return self.SCORES[run_id][cohort]

    def _fake_error_details(self, xml_path, predictor_path):
        score = self._score(xml_path, predictor_path)
        return [
            {
                "image": xml_path,
                "filename": os.path.basename(xml_path),
                "mean_error": score,
                "median_error": score,
                "per_landmark_error": [score, score],
            }
        ]

    def _train(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return train_shape_model.train_shape_model(
                self.root,
                self.TAG,
                aug_angles=[],
                aug_flip=False,
            )

    def test_validation_tail_gate_rejects_hidden_catastrophic_outlier(self):
        stable = train_shape_model._dlib_validation_stability_gate(
            [0.09, 0.10, 0.11],
            0.10,
            0.10,
        )
        unstable = train_shape_model._dlib_validation_stability_gate(
            [0.10] * 19 + [2.0],
            (0.10 * 19 + 2.0) / 20.0,
            0.10,
        )
        self.assertTrue(stable["passed"])
        self.assertFalse(unstable["passed"])
        self.assertEqual(unstable["reason"], "catastrophic_validation_outliers")

    def test_reviewed_retraining_preserves_benchmark_and_gates_promotion(self):
        for index in range(6):
            self._add_sample(f"manual_{index}", 20 + index * 17)

        with (
            patch.object(train_shape_model.dio, "_run_id", side_effect=["run_v1", "run_v2", "run_v3"]),
            patch.object(train_shape_model.dio, "_utc_now_iso", return_value="2026-01-02T03:04:05Z"),
            patch.object(train_shape_model.lineage, "utc_now_iso", return_value="2026-01-02T03:04:05Z"),
            patch.object(
                train_shape_model.lineage,
                "collect_code_state",
                return_value={"commit": "mock-commit", "dirty": False, "dependencyLocks": {}},
            ),
            patch.object(
                train_shape_model.lineage,
                "collect_runtime_state",
                return_value={"python": "mock-python", "platform": "mock-platform", "packages": {}},
            ),
            patch.object(train_shape_model.dlib, "train_shape_predictor", side_effect=self._fake_train),
            patch.object(train_shape_model.dlib, "test_shape_predictor", side_effect=self._score),
            patch.object(
                train_shape_model,
                "_compute_dlib_per_image_errors",
                side_effect=lambda xml_path, predictor_path: [self._score(xml_path, predictor_path)],
            ),
            patch.object(
                train_shape_model,
                "_compute_dlib_per_image_error_details",
                side_effect=self._fake_error_details,
            ),
        ):
            split_v1 = self._prepare()
            cohort_path = os.path.join(self.root, "debug", "cohorts", "landmark_benchmark_v1.json")
            with open(cohort_path, "r", encoding="utf-8") as handle:
                cohort_v1 = json.load(handle)
            initial_assignments = dict(cohort_v1["assignments"])
            self.assertTrue(split_v1["test_source_ids"])
            self.assertTrue(split_v1["validation_source_ids"])
            self.assertFalse(set(split_v1["train_source_ids"]) & set(split_v1["test_source_ids"]))
            self.assertFalse(
                set(split_v1["train_source_ids"]) & set(split_v1["validation_source_ids"])
            )
            self.assertFalse(
                set(split_v1["test_source_ids"]) & set(split_v1["validation_source_ids"])
            )

            run_v1 = self._train()
            self.assertEqual(run_v1["model_id"], "dlib:run_v1")
            self.assertEqual(run_v1["registry"]["status"], "active")
            predictor_v1 = Path(run_v1["model_path"])
            predictor_v1_bytes = predictor_v1.read_bytes()

            review_event = self._add_hitl_sample(run_v1["model_id"])
            split_v2 = self._prepare()
            with open(cohort_path, "r", encoding="utf-8") as handle:
                cohort_v2 = json.load(handle)
            for source_id, assignment in initial_assignments.items():
                self.assertEqual(cohort_v2["assignments"][source_id], assignment)
            adaptive_sources = [
                source_id
                for source_id, metadata in cohort_v2["sources"].items()
                if metadata.get("adaptiveTrainingSample")
            ]
            self.assertEqual(len(adaptive_sources), 1)
            hitl_source_id = adaptive_sources[0]
            self.assertEqual(cohort_v2["assignments"][hitl_source_id], "train")
            self.assertIn(hitl_source_id, split_v2["train_source_ids"])
            self.assertEqual(split_v2["test_source_ids"], split_v1["test_source_ids"])
            self.assertEqual(
                split_v2["validation_source_ids"],
                split_v1["validation_source_ids"],
            )

            run_v2 = self._train()
            self.assertEqual(run_v2["model_id"], "dlib:run_v2")
            self.assertEqual(run_v2["registry"]["status"], "active")
            self.assertTrue(run_v2["registry"]["promotion"]["promoted"])
            self.assertEqual(run_v2["registry"]["promotion"]["metric"], "validationMedianError")
            self.assertEqual(run_v2["registry"]["promotion"]["baselineModelId"], run_v1["model_id"])
            self.assertAlmostEqual(run_v2["registry"]["promotion"]["baselineScore"], 0.25)
            self.assertAlmostEqual(run_v2["registry"]["promotion"]["candidateScore"], 0.15)

            # A later fit can overfit the expanded train set. Its worse frozen-
            # cohort score must retain it as a candidate without moving aliases.
            run_v3 = self._train()
            self.assertEqual(run_v3["registry"]["status"], "candidate")
            self.assertFalse(run_v3["registry"]["promotion"]["promoted"])
            self.assertEqual(run_v3["registry"]["promotion"]["reason"], "locked_cohort_not_improved")
            self.assertEqual(run_v3["registry"]["promotion"]["baselineModelId"], run_v2["model_id"])
            self.assertAlmostEqual(run_v3["registry"]["promotion"]["baselineScore"], 0.15)
            self.assertAlmostEqual(run_v3["registry"]["promotion"]["candidateScore"], 0.20)

        manifest_v1 = lineage.read_json(os.path.join(predictor_v1.parent, "manifest.json"))
        manifest_v2 = lineage.read_json(str(Path(run_v2["model_path"]).parent / "manifest.json"))
        manifest_v3 = lineage.read_json(str(Path(run_v3["model_path"]).parent / "manifest.json"))
        self.assertTrue(predictor_v1.is_file())
        self.assertEqual(predictor_v1.read_bytes(), predictor_v1_bytes)
        self.assertNotEqual(
            manifest_v1["lineage"]["dataset"]["revision"],
            manifest_v2["lineage"]["dataset"]["revision"],
        )
        self.assertEqual(
            manifest_v2["lineage"]["dataset"]["revision"],
            manifest_v3["lineage"]["dataset"]["revision"],
        )
        self.assertEqual(manifest_v2["lineage"]["dataset"]["originCounts"]["hitl_review"], 1)
        self.assertEqual(manifest_v1["lineage"]["trainingMode"], "train_from_base")
        self.assertIsNone(manifest_v1["lineage"]["parentModelId"])
        self.assertEqual(manifest_v2["lineage"]["trainingMode"], "retrain_from_base")
        self.assertIsNone(manifest_v2["lineage"]["parentModelId"])
        self.assertEqual(manifest_v2["lineage"]["baselineModelId"], run_v1["model_id"])
        self.assertIsNone(manifest_v3["lineage"]["parentModelId"])
        self.assertEqual(manifest_v3["lineage"]["baselineModelId"], run_v2["model_id"])
        for manifest in (manifest_v1, manifest_v2, manifest_v3):
            self.assertEqual(
                manifest["lineage"]["initialization"],
                {"strategy": "from_scratch", "framework": "dlib", "checkpoint": None},
            )
        cohort_revisions = {
            manifest["lineage"]["dataset"]["splits"][0]["testCohortRevision"]
            for manifest in (manifest_v1, manifest_v2, manifest_v3)
        }
        self.assertEqual(len(cohort_revisions), 1)
        validation_cohort_revisions = {
            manifest["lineage"]["dataset"]["splits"][0]["validationCohortRevision"]
            for manifest in (manifest_v1, manifest_v2, manifest_v3)
        }
        self.assertEqual(len(validation_cohort_revisions), 1)

        current_alias = Path(self.root) / "models" / f"predictor_{self.TAG}.dat"
        self.assertEqual(current_alias.read_bytes(), Path(run_v2["model_path"]).read_bytes())
        self.assertNotEqual(current_alias.read_bytes(), Path(run_v3["model_path"]).read_bytes())

        review_events = lineage.read_json(os.path.join(self.root, "review_events.json"))
        reviewed_label = lineage.read_json(os.path.join(self.labels_dir, "reviewed_hitl.json"))
        self.assertEqual(review_events, [review_event])
        self.assertEqual(reviewed_label["provenance"], review_event)
        self.assertEqual(reviewed_label["reviewHistory"], [review_event])

        # The frozen test cohort must actually be measured, and must stay out of
        # the promotion metric block so it can never select a model.
        for manifest in (manifest_v1, manifest_v2, manifest_v3):
            evaluation = manifest["testEvaluation"]
            self.assertEqual(evaluation["status"], "measured")
            self.assertEqual(evaluation["role"], "report_only")
            self.assertGreater(evaluation["sampleCount"], 0)
            self.assertIsNotNone(evaluation["medianNormalizedError"])
            self.assertTrue(math.isfinite(evaluation["medianNormalizedError"]))
            self.assertNotIn("testMedianError", manifest["metrics"])
            self.assertNotIn("testError", manifest["metrics"])

        # run_v2 regressed on the locked test cohort (0.30 -> 0.40) while
        # improving on validation, and was still promoted; run_v3 improved on
        # test (0.05) and was still held as a candidate.  Report-only means the
        # measured number is recorded and ignored, in both directions.
        self.assertAlmostEqual(manifest_v1["testEvaluation"]["medianNormalizedError"], 0.30)
        self.assertAlmostEqual(manifest_v2["testEvaluation"]["medianNormalizedError"], 0.40)
        self.assertAlmostEqual(manifest_v3["testEvaluation"]["medianNormalizedError"], 0.05)
        self.assertEqual(run_v2["registry"]["status"], "active")
        self.assertEqual(run_v3["registry"]["status"], "candidate")

        self.assertEqual(review_event["landmarkModelKey"], run_v1["model_id"])
        self.assertEqual(review_event["reviewOutcome"], "corrected")

        registry = lineage.read_json(os.path.join(self.root, "models", "model_registry.json"))
        statuses = {record["runId"]: record["status"] for record in registry["models"]}
        self.assertEqual(statuses, {"run_v1": "deprecated", "run_v2": "active", "run_v3": "candidate"})

    def test_report_only_test_metrics_are_measured_and_never_select_a_model(self):
        """A regression guard: the frozen test cohort must not go unevaluated.

        An earlier refactor deferred this evaluation to a promotion step that
        was never written, so every run silently published a null test metric
        while still preparing, hashing and lineage-tracking the cohort.
        """
        for index in range(6):
            self._add_sample(f"manual_{index}", 20 + index * 17)

        with (
            patch.object(train_shape_model.dio, "_run_id", side_effect=["run_v1", "run_v2"]),
            patch.object(train_shape_model.dlib, "train_shape_predictor", side_effect=self._fake_train),
            patch.object(train_shape_model.dlib, "test_shape_predictor", side_effect=self._score),
            patch.object(
                train_shape_model,
                "_compute_dlib_per_image_errors",
                side_effect=lambda xml_path, predictor_path: [self._score(xml_path, predictor_path)],
            ),
            patch.object(
                train_shape_model,
                "_compute_dlib_per_image_error_details",
                side_effect=self._fake_error_details,
            ),
        ):
            self._prepare()
            run = self._train()

        self.assertIsNotNone(run["test_median_error"])
        self.assertTrue(math.isfinite(run["test_median_error"]))
        self.assertIsNotNone(run["test_error"])

        evaluation = run["test_evaluation"]
        self.assertEqual(evaluation["status"], "measured")
        self.assertEqual(evaluation["role"], "report_only")
        self.assertEqual(evaluation["cohort"], "frozen_test")
        self.assertEqual(evaluation["reason"], "never_used_for_model_selection")

        # Promotion ranks only these two names; a test metric appearing in the
        # manifest metrics block would make it selectable.
        manifest = lineage.read_json(str(Path(run["model_path"]).parent / "manifest.json"))
        self.assertIn("validationMedianError", manifest["metrics"])
        self.assertFalse(
            [key for key in manifest["metrics"] if key.lower().startswith("test")]
        )


if __name__ == "__main__":
    unittest.main()
