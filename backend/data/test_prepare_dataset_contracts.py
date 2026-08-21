import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from backend.bv_utils import lineage
from backend.data.prepare_dataset import (
    _build_landmark_template,
    _require_explicit_orientation_policy,
    _required_schema_landmark_ids,
    _resolve_landmark_training_boxes,
    _source_content_id,
    _stable_landmark_split,
    _validate_training_landmark_contract,
    json_to_dlib_xml,
)


class LandmarkContractTests(unittest.TestCase):
    def test_landmark_template_is_derived_from_training_entries_only(self):
        train_entries = [
            {"landmarks": [{"id": 3, "x": 100.0, "y": 200.0}]},
            {"landmarks": [{"id": 3, "x": 120.0, "y": 220.0}]},
        ]
        held_out_outlier = {
            "landmarks": [{"id": 3, "x": 500.0, "y": 500.0}]
        }

        template = _build_landmark_template(train_entries)

        self.assertEqual(template[3]["count"], 2)
        self.assertEqual(template[3]["x_mean"], 110.0)
        self.assertEqual(template[3]["y_mean"], 210.0)
        self.assertNotEqual(
            template[3]["x_mean"],
            _build_landmark_template([*train_entries, held_out_outlier])[3]["x_mean"],
        )

    def test_training_requires_explicit_orientation_policy(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "session.json"), "w", encoding="utf-8") as handle:
                json.dump({"orientationPolicy": {"mode": "invariant"}}, handle)
            with self.assertRaisesRegex(RuntimeError, "requires an explicit"):
                _require_explicit_orientation_policy(root)

            with open(os.path.join(root, "session.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                    },
                    handle,
                )
            self.assertEqual(_require_explicit_orientation_policy(root)["mode"], "invariant")

    def test_missing_required_landmark_blocks_training_with_location(self):
        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "sample.json")
            with open(label_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "imageFilename": "sample.png",
                        "boxes": [
                            {
                                "landmarks": [
                                    {"id": 1, "x": 10, "y": 20},
                                    {"id": 3, "x": 30, "y": 40},
                                ]
                            }
                        ],
                    },
                    handle,
                )

            with self.assertRaisesRegex(RuntimeError, r"sample\.json box 0.*\[2\]"):
                _validate_training_landmark_contract([label_path], [1, 2, 3])

    def test_contract_must_be_declared_and_explicit_optional_landmarks_are_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "sample.json")
            with open(label_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "boxes": [
                            {"landmarks": [{"id": 1, "x": 10, "y": 20}]}
                        ]
                    },
                    handle,
                )
            with open(os.path.join(root, "session.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "landmarkTemplate": [
                            {"index": 1, "name": "required"},
                            {"index": 2, "name": "optional", "optional": True},
                            {"index": 3, "name": "also optional", "required": False},
                        ]
                    },
                    handle,
                )

            required = _required_schema_landmark_ids(root)
            self.assertEqual(required, [1])
            self.assertEqual(
                _validate_training_landmark_contract([label_path], required),
                [1],
            )
            with self.assertRaisesRegex(RuntimeError, "declare at least one required"):
                _validate_training_landmark_contract([label_path], [])

    def test_explicit_detection_only_boxes_are_excluded_from_landmark_contract(self):
        with tempfile.TemporaryDirectory() as root:
            label_path = os.path.join(root, "mixed.json")
            with open(label_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "boxes": [
                            {
                                "trainingTargets": ["obb"],
                                "landmarks": [],
                            },
                            {
                                "trainingTargets": ["landmark", "obb"],
                                "landmarks": [{"id": 1, "x": 10, "y": 20}],
                            },
                        ]
                    },
                    handle,
                )
            self.assertEqual(_validate_training_landmark_contract([label_path], [1]), [1])

    def test_explicit_rejected_all_boxes_are_not_resurrected_for_landmark_training(self):
        with tempfile.TemporaryDirectory() as root:
            rejected_path = os.path.join(root, "rejected.json")
            accepted_path = os.path.join(root, "accepted.json")
            rejected_payload = {
                "boxes": [
                    {
                        "trainingTargets": ["landmark", "obb"],
                        "landmarks": [{"id": 1, "x": 10, "y": 20}],
                    }
                ],
                "landmarks": [{"id": 1, "x": 10, "y": 20}],
                "provenance": {"reviewOutcome": "rejected_all"},
                "finalizedDetection": {
                    "isFinalized": True,
                    "acceptedBoxes": [],
                },
            }
            with open(rejected_path, "w", encoding="utf-8") as handle:
                json.dump(rejected_payload, handle)
            with open(accepted_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "boxes": [
                            {
                                "landmarks": [
                                    {"id": 1, "x": 30, "y": 40},
                                    {"id": 2, "x": 50, "y": 60},
                                ]
                            }
                        ]
                    },
                    handle,
                )

            boxes, finalized_declared = _resolve_landmark_training_boxes(rejected_payload)
            self.assertTrue(finalized_declared)
            self.assertEqual(boxes, [])
            self.assertEqual(
                _validate_training_landmark_contract(
                    [rejected_path, accepted_path],
                    [1, 2],
                ),
                [1, 2],
            )

    def test_invalid_declared_accepted_boxes_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "acceptedBoxes must be an array"):
            _resolve_landmark_training_boxes(
                {"finalizedDetection": {"acceptedBoxes": None}}
            )

    def test_legacy_landmark_targets_are_merged_only_for_matching_accepted_geometry(self):
        accepted = {
            "left": 10,
            "top": 20,
            "width": 30,
            "height": 40,
        }
        resolved, finalized_declared = _resolve_landmark_training_boxes(
            {
                "boxes": [
                    {
                        **accepted,
                        "trainingTargets": ["obb"],
                        "landmarks": [{"id": 1, "x": 15, "y": 25}],
                    },
                    {
                        "left": 90,
                        "top": 90,
                        "width": 10,
                        "height": 10,
                        "trainingTargets": ["landmark"],
                        "landmarks": [{"id": 2, "x": 95, "y": 95}],
                    },
                ],
                "finalizedDetection": {"acceptedBoxes": [accepted]},
            }
        )

        self.assertTrue(finalized_declared)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["trainingTargets"], ["obb"])
        self.assertEqual(resolved[0]["landmarks"][0]["id"], 1)

    def test_stable_source_cohorts_do_not_move_when_data_grows(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")

            def entry(source_id, name, provenance=None):
                return {
                    "path": os.path.join(root, f"{name}_crop.png"),
                    "source_id": source_id,
                    "source_image": f"{name}.png",
                    "provenance": provenance or {},
                    "landmarks": [],
                }

            first = [entry(f"sha256:{value:064x}", f"source-{value}") for value in range(1, 9)]
            _, _, _, first_train, first_validation, first_test, _ = _stable_landmark_split(
                first,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            initial = {
                source_id: (
                    "test"
                    if source_id in first_test
                    else "validation"
                    if source_id in first_validation
                    else "train"
                )
                for source_id in first_train | first_validation | first_test
            }

            hitl_id = f"sha256:{99:064x}"
            grown = [entry(f"sha256:{0:064x}", "alphabetically-earlier"), *first]
            grown.append(entry(hitl_id, "reviewed", {"source": "hitl_review"}))
            _, _, _, grown_train, grown_validation, grown_test, _ = _stable_landmark_split(
                grown,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )

            for source_id, prior in initial.items():
                current = (
                    "test"
                    if source_id in grown_test
                    else "validation"
                    if source_id in grown_validation
                    else "train"
                )
                self.assertEqual(prior, current)
            self.assertEqual(grown_test, first_test)
            self.assertEqual(grown_validation, first_validation)
            self.assertIn(f"sha256:{0:064x}", grown_train)
            self.assertIn(hitl_id, grown_train)
            self.assertNotIn(hitl_id, grown_test)

    def test_source_identity_uses_content_not_filename(self):
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "a.bin")
            second = os.path.join(root, "renamed.bin")
            for file_path in (first, second):
                with open(file_path, "wb") as handle:
                    handle.write(b"same-content")
            self.assertEqual(
                _source_content_id(first, "a.bin"),
                _source_content_id(second, "renamed.bin"),
            )

    def test_same_stem_different_extension_sources_get_distinct_derived_crops(self):
        with tempfile.TemporaryDirectory() as root:
            images_dir = os.path.join(root, "images")
            labels_dir = os.path.join(root, "labels")
            os.makedirs(images_dir)
            os.makedirs(labels_dir)
            with open(os.path.join(root, "session.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "orientationPolicy": {"mode": "invariant"},
                        "orientationPolicyConfigured": True,
                        "landmarkTemplate": [
                            {"index": 1, "name": "a"},
                            {"index": 2, "name": "b"},
                        ],
                    },
                    handle,
                )

            image_specs = [
                ("fish.jpg", np.full((80, 80, 3), (20, 40, 60), dtype=np.uint8)),
                ("fish.png", np.full((80, 80, 3), (210, 160, 90), dtype=np.uint8)),
            ]
            for index, (filename, image) in enumerate(image_specs):
                self.assertTrue(cv2.imwrite(os.path.join(images_dir, filename), image))
                box = {
                    "left": 5,
                    "top": 5,
                    "width": 70,
                    "height": 70,
                    "obbCorners": [[5, 5], [75, 5], [75, 75], [5, 75]],
                    "landmarks": [
                        {"id": 1, "x": 20, "y": 30},
                        {"id": 2, "x": 60, "y": 50},
                    ],
                }
                with open(
                    os.path.join(labels_dir, f"label-{index}.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump({"imageFilename": filename, "boxes": [box]}, handle)

            json_to_dlib_xml(root, "SameStem", test_split=0.25, seed=42)

            with open(
                os.path.join(root, "debug", "crop_metadata_SameStem.json"),
                "r",
                encoding="utf-8",
            ) as handle:
                crop_metadata = json.load(handle)
            crop_paths = [
                entry["crop_path"]
                for entry in crop_metadata
                if not entry.get("is_box_scale_augmented")
            ]
            self.assertEqual(len(crop_paths), 2)
            self.assertEqual(len({os.path.normcase(path) for path in crop_paths}), 2)
            self.assertEqual(len({lineage.sha256_file(path) for path in crop_paths}), 2)
            self.assertTrue(all(os.path.isfile(path) for path in crop_paths))

    def test_frozen_test_source_rejects_later_hitl_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")

            def entry(value, provenance=None):
                return {
                    "path": os.path.join(root, f"source-{value}_crop.png"),
                    "source_id": f"sha256:{value:064x}",
                    "source_image": f"source-{value}.png",
                    "provenance": provenance or {},
                    "landmarks": [{"id": 1, "x": value, "y": value}],
                }

            entries = [entry(value) for value in range(1, 9)]
            _train, _validation, _test, _train_ids, _validation_ids, test_ids, _path = _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            locked_test = next(iter(test_ids))
            cohort_path = os.path.join(
                debug_dir,
                "cohorts",
                "landmark_benchmark_v1.json",
            )
            with open(cohort_path, "rb") as handle:
                before = handle.read()
            changed = [dict(item) for item in entries]
            for item in changed:
                if item["source_id"] == locked_test:
                    item["provenance"] = {"source": "hitl_review"}

            with self.assertRaisesRegex(
                RuntimeError,
                "evaluator sources acquired HITL/model-assisted provenance",
            ):
                _stable_landmark_split(
                    changed,
                    debug_dir=debug_dir,
                    test_split=0.9,
                    seed=999,
                )
            with open(cohort_path, "rb") as handle:
                self.assertEqual(handle.read(), before)

    def test_identical_cohort_rerun_preserves_manifest_digest_and_creation_policy(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            entries = [
                {
                    "path": os.path.join(root, f"source-{value}_crop.png"),
                    "source_id": f"sha256:{value:064x}",
                    "source_image": f"source-{value}.png",
                    "provenance": {},
                    "landmarks": [{"id": 1, "x": value, "y": value + 1}],
                }
                for value in range(1, 7)
            ]
            *_unused, cohort_path = _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            first_digest = lineage.sha256_file(cohort_path)
            first_manifest = lineage.read_json(cohort_path)

            _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.9,
                seed=999,
            )
            second_manifest = lineage.read_json(cohort_path)
            self.assertEqual(lineage.sha256_file(cohort_path), first_digest)
            self.assertEqual(second_manifest["seed"], 42)
            self.assertEqual(second_manifest["testFraction"], 0.25)
            self.assertEqual(second_manifest["createdAt"], first_manifest["createdAt"])
            self.assertEqual(
                second_manifest["testCohortRevision"],
                first_manifest["testCohortRevision"],
            )

    def test_locked_test_annotation_content_cannot_change_silently(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            entries = [
                {
                    "path": os.path.join(root, f"source-{value}_crop.png"),
                    "source_id": f"sha256:{value:064x}",
                    "source_image": f"source-{value}.png",
                    "provenance": {},
                    "box_index": 0,
                    "source_box": {"left": 0, "top": 0, "width": 20, "height": 20},
                    "source_landmarks": [{"id": 1, "x": value, "y": value}],
                    "landmarks": [{"id": 1, "x": value, "y": value}],
                }
                for value in range(1, 7)
            ]
            _train, _validation, _test, _train_ids, _validation_ids, test_ids, _path = _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            locked_test = next(iter(test_ids))
            changed = []
            for entry in entries:
                copied = {**entry, "landmarks": [dict(item) for item in entry["landmarks"]]}
                if copied["source_id"] == locked_test:
                    copied["landmarks"][0]["x"] += 1
                changed.append(copied)

            with self.assertRaisesRegex(RuntimeError, "benchmark annotation/crop content changed"):
                _stable_landmark_split(
                    changed,
                    debug_dir=debug_dir,
                    test_split=0.25,
                    seed=42,
                )

    def test_locked_test_standardized_pixels_cannot_change_silently(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            entries = []
            for value in range(1, 7):
                crop_path = os.path.join(root, f"source-{value}_crop.png")
                with open(crop_path, "wb") as handle:
                    handle.write(f"standardized-test-crop-{value}".encode("ascii"))
                entries.append(
                    {
                        "path": crop_path,
                        "source_id": f"sha256:{value:064x}",
                        "source_image": f"source-{value}.png",
                        "provenance": {},
                        "box_index": 0,
                        "source_box": {"left": 0, "top": 0, "width": 20, "height": 20},
                        "source_landmarks": [{"id": 1, "x": value, "y": value}],
                        "landmarks": [{"id": 1, "x": value, "y": value}],
                    }
                )

            *_unused, test_ids, _path = _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            locked_test = next(iter(test_ids))
            locked_entry = next(
                entry for entry in entries if entry["source_id"] == locked_test
            )
            with open(locked_entry["path"], "ab") as handle:
                handle.write(b"-mutated")

            with self.assertRaisesRegex(RuntimeError, "benchmark annotation/crop content changed"):
                _stable_landmark_split(
                    entries,
                    debug_dir=debug_dir,
                    test_split=0.25,
                    seed=42,
                )

    def test_locked_validation_content_cannot_change_silently(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            entries = [
                {
                    "path": os.path.join(root, f"source-{value}_crop.png"),
                    "source_id": f"sha256:{value:064x}",
                    "source_image": f"source-{value}.png",
                    "provenance": {},
                    "box_index": 0,
                    "source_box": {"left": 0, "top": 0, "width": 20, "height": 20},
                    "source_landmarks": [{"id": 1, "x": value, "y": value}],
                    "landmarks": [{"id": 1, "x": value, "y": value}],
                }
                for value in range(1, 7)
            ]
            for value, entry in enumerate(entries, start=1):
                with open(entry["path"], "wb") as handle:
                    handle.write(f"standardized-crop-{value}".encode("ascii"))
            (
                _train,
                _validation,
                _test,
                _train_ids,
                validation_ids,
                _test_ids,
                _path,
            ) = _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            locked_validation = next(iter(validation_ids))
            locked_entry = next(
                entry for entry in entries if entry["source_id"] == locked_validation
            )
            with open(locked_entry["path"], "ab") as handle:
                handle.write(b"-mutated")

            with self.assertRaisesRegex(RuntimeError, "validation annotation/crop content changed"):
                _stable_landmark_split(
                    entries,
                    debug_dir=debug_dir,
                    test_split=0.25,
                    seed=42,
                )

    def test_current_cohort_manifest_missing_snapshot_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            entries = [
                {
                    "path": os.path.join(root, f"source-{value}_crop.png"),
                    "source_id": f"sha256:{value:064x}",
                    "source_image": f"source-{value}.png",
                    "provenance": {},
                    "landmarks": [{"id": 1, "x": value, "y": value}],
                }
                for value in range(1, 7)
            ]
            for value, entry in enumerate(entries, start=1):
                with open(entry["path"], "wb") as handle:
                    handle.write(f"cohort-{value}".encode("ascii"))
            *_unused, cohort_path = _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            manifest = lineage.read_json(cohort_path)
            missing_source = next(iter(manifest["validationSourceSnapshots"]))
            del manifest["validationSourceSnapshots"][missing_source]
            lineage.atomic_write_json(cohort_path, manifest)

            with self.assertRaisesRegex(RuntimeError, "missing exact snapshot/revision"):
                _stable_landmark_split(
                    entries,
                    debug_dir=debug_dir,
                    test_split=0.25,
                    seed=42,
                )

    def test_malformed_existing_cohort_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            cohort_dir = os.path.join(debug_dir, "cohorts")
            os.makedirs(cohort_dir, exist_ok=True)
            cohort_path = os.path.join(cohort_dir, "landmark_benchmark_v1.json")
            with open(cohort_path, "w", encoding="utf-8") as handle:
                handle.write("{not-json")
            entry = {
                "path": os.path.join(root, "source_crop.png"),
                "source_id": f"sha256:{1:064x}",
                "source_image": "source.png",
                "provenance": {},
                "landmarks": [{"id": 1, "x": 1, "y": 2}],
            }

            with self.assertRaisesRegex(RuntimeError, "manifest is malformed"):
                _stable_landmark_split(
                    [entry],
                    debug_dir=debug_dir,
                    test_split=0.25,
                    seed=42,
                )

    def test_v2_test_manifest_migrates_without_changing_test_identity(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            entries = [
                {
                    "path": os.path.join(root, f"source-{value}_crop.png"),
                    "source_id": f"sha256:{value:064x}",
                    "source_image": f"source-{value}.png",
                    "provenance": {},
                    "landmarks": [{"id": 1, "x": value, "y": value}],
                }
                for value in range(1, 7)
            ]
            *_parts, cohort_path = _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            old = lineage.read_json(cohort_path)
            for source_id, cohort in list(old["assignments"].items()):
                if cohort == "validation":
                    old["assignments"][source_id] = "train"
            old["version"] = 2
            old.pop("validationFraction", None)
            old.pop("validationSourceSnapshots", None)
            old.pop("validationCohortRevision", None)
            old.pop("validationSourceOverlap", None)
            old["assignmentRevision"] = lineage.sha256_json(old["assignments"])
            lineage.atomic_write_json(cohort_path, old)
            locked_test_revision = old["testCohortRevision"]
            locked_test_ids = {
                source_id
                for source_id, cohort in old["assignments"].items()
                if cohort == "test"
            }

            (
                _train,
                _validation,
                _test,
                train_ids,
                validation_ids,
                test_ids,
                _path,
            ) = _stable_landmark_split(
                entries,
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            migrated = lineage.read_json(cohort_path)
            self.assertEqual(test_ids, locked_test_ids)
            self.assertEqual(migrated["testCohortRevision"], locked_test_revision)
            self.assertFalse(validation_ids)
            self.assertFalse(train_ids & validation_ids)
            self.assertFalse(test_ids & validation_ids)

            # A former training source must not be retroactively converted into
            # validation data. The next unseen source can seed validation.
            new_entry = {
                "path": os.path.join(root, "source-99_crop.png"),
                "source_id": f"sha256:{99:064x}",
                "source_image": "source-99.png",
                "provenance": {},
                "landmarks": [{"id": 1, "x": 99, "y": 99}],
            }
            with open(new_entry["path"], "wb") as handle:
                handle.write(b"new-unseen-evaluator-source")
            (
                _train,
                _validation,
                _test,
                _train_ids,
                validation_ids_after,
                test_ids_after,
                _path,
            ) = _stable_landmark_split(
                [*entries, new_entry],
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            self.assertEqual(test_ids_after, locked_test_ids)
            self.assertEqual(validation_ids_after, {new_entry["source_id"]})

    def test_one_source_growth_never_repurposes_prior_training_data_for_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")

            def entry(value):
                crop_path = os.path.join(root, f"source-{value}_crop.png")
                with open(crop_path, "wb") as handle:
                    handle.write(f"crop-{value}".encode("ascii"))
                return {
                    "path": crop_path,
                    "source_id": f"sha256:{value:064x}",
                    "source_image": f"source-{value}.png",
                    "provenance": {},
                    "landmarks": [{"id": 1, "x": value, "y": value + 1}],
                }

            first, second, third, fourth, fifth = (
                entry(1),
                entry(2),
                entry(3),
                entry(4),
                entry(5),
            )
            first_split = _stable_landmark_split(
                [first], debug_dir=debug_dir, test_split=0.25, seed=42
            )
            self.assertEqual(first_split[3], {first["source_id"]})
            self.assertEqual(first_split[4], set())
            self.assertEqual(first_split[5], set())

            second_split = _stable_landmark_split(
                [first, second], debug_dir=debug_dir, test_split=0.25, seed=42
            )
            self.assertEqual(second_split[3], {first["source_id"]})
            self.assertEqual(second_split[4], set())
            self.assertEqual(second_split[5], {second["source_id"]})

            third_split = _stable_landmark_split(
                [first, second, third], debug_dir=debug_dir, test_split=0.25, seed=42
            )
            self.assertEqual(third_split[3], {first["source_id"]})
            self.assertEqual(third_split[4], {third["source_id"]})
            self.assertEqual(third_split[5], {second["source_id"]})

            fourth_split = _stable_landmark_split(
                [first, second, third, fourth],
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            self.assertEqual(fourth_split[3], {first["source_id"]})
            self.assertEqual(
                fourth_split[4],
                {third["source_id"], fourth["source_id"]},
            )
            self.assertEqual(fourth_split[5], {second["source_id"]})
            manifest = lineage.read_json(fourth_split[6])
            self.assertTrue(manifest["validationBootstrapComplete"])

            fifth_split = _stable_landmark_split(
                [first, second, third, fourth, fifth],
                debug_dir=debug_dir,
                test_split=0.25,
                seed=42,
            )
            self.assertIn(fifth["source_id"], fifth_split[3])
            self.assertEqual(fifth_split[4], fourth_split[4])
            self.assertEqual(fifth_split[5], fourth_split[5])

    def test_single_hitl_source_never_becomes_evaluation_data(self):
        with tempfile.TemporaryDirectory() as root:
            entry = {
                "path": os.path.join(root, "reviewed_crop.png"),
                "source_id": f"sha256:{1:064x}",
                "source_image": "reviewed.png",
                "provenance": {"source": "hitl_review"},
                "landmarks": [{"id": 1, "x": 1, "y": 2}],
            }
            (
                _train,
                validation,
                test,
                train_ids,
                validation_ids,
                test_ids,
                _path,
            ) = _stable_landmark_split(
                [entry],
                debug_dir=os.path.join(root, "debug"),
                test_split=0.25,
                seed=42,
            )
            self.assertEqual(train_ids, {entry["source_id"]})
            self.assertEqual(validation, [])
            self.assertEqual(test, [])
            self.assertEqual(validation_ids, set())
            self.assertEqual(test_ids, set())
            repeated = _stable_landmark_split(
                [entry],
                debug_dir=os.path.join(root, "debug"),
                test_split=0.9,
                seed=999,
            )
            self.assertEqual(repeated[3], {entry["source_id"]})
            self.assertEqual(repeated[4], set())
            self.assertEqual(repeated[5], set())

    def test_adaptive_provenance_and_names_are_sticky_for_existing_train_source(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            source_id = f"sha256:{1:064x}"
            crop_path = os.path.join(root, "same_crop.png")
            with open(crop_path, "wb") as handle:
                handle.write(b"same-standardized-content")

            manual = {
                "path": crop_path,
                "source_id": source_id,
                "source_image": "manual-name.png",
                "provenance": {},
                "landmarks": [{"id": 1, "x": 1, "y": 2}],
            }
            *_unused, cohort_path = _stable_landmark_split(
                [manual], debug_dir=debug_dir, test_split=0.25, seed=42
            )
            first = lineage.read_json(cohort_path)
            self.assertFalse(first["sources"][source_id]["adaptiveTrainingSample"])

            reviewed = {
                **manual,
                "source_image": "reviewed-name.png",
                "provenance": {"source": "hitl_review"},
            }
            _stable_landmark_split(
                [reviewed], debug_dir=debug_dir, test_split=0.25, seed=42
            )
            second = lineage.read_json(cohort_path)
            self.assertTrue(second["sources"][source_id]["adaptiveTrainingSample"])
            self.assertEqual(
                second["sources"][source_id]["names"],
                ["manual-name.png", "reviewed-name.png"],
            )

            # A later manual prepare cannot erase the review history.
            _stable_landmark_split(
                [manual], debug_dir=debug_dir, test_split=0.25, seed=42
            )
            third = lineage.read_json(cohort_path)
            self.assertTrue(third["sources"][source_id]["adaptiveTrainingSample"])


if __name__ == "__main__":
    unittest.main()
