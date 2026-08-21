import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import backend.data.export_yolo_dataset as export_yolo_dataset_module
from backend.data.export_yolo_dataset import (
    OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION,
    OBB_SPLIT_ASSIGNMENTS_VERSION,
    _get_finalized_boxes,
    _is_confirmed_negative_review,
    _prepare_real_sample_for_export,
    _stable_sample_identity,
    export_obb_dataset,
)


class ObbDatasetExportRegressionTests(unittest.TestCase):
    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.session_dir = self._tempdir.name
        os.makedirs(os.path.join(self.session_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "labels"), exist_ok=True)
        with open(os.path.join(self.session_dir, "session.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "landmarkTemplate": [],
                    "orientationPolicy": {"mode": "invariant"},
                    "orientationPolicyConfigured": True,
                },
                f,
            )

    def tearDown(self):
        self._tempdir.cleanup()

    def _add_sample(
        self,
        name,
        corners,
        landmarks=None,
        pixel_value=64,
        source_group=None,
        provenance=None,
        label_filename=None,
        class_id=None,
    ):
        image = np.full((80, 100, 3), int(pixel_value), dtype=np.uint8)
        # Ensure every generated source has a distinct content identity.
        image[0, 0, 0] = int(pixel_value) % 255
        image[0, 1, 1] = (int(pixel_value) * 7) % 255
        image_path = os.path.join(self.session_dir, "images", name)
        self.assertTrue(cv2.imwrite(image_path, image))

        xs = [float(point[0]) for point in corners]
        ys = [float(point[1]) for point in corners]
        box = {
            "left": math.floor(min(xs)),
            "top": math.floor(min(ys)),
            "width": max(1, math.ceil(max(xs)) - math.floor(min(xs))),
            "height": max(1, math.ceil(max(ys)) - math.floor(min(ys))),
            "obbCorners": [[float(x), float(y)] for x, y in corners],
            "landmarks": list(landmarks or []),
        }
        if class_id is not None:
            box["class_id"] = int(class_id)
        label = {
            "imageFilename": name,
            "boxes": [box],
            "finalizedDetection": {
                "isFinalized": True,
                "acceptedBoxes": [box],
            },
        }
        if source_group:
            label["sourceGroup"] = source_group
        if provenance:
            label["provenance"] = dict(provenance)
        label_name = label_filename or (os.path.splitext(name)[0] + ".json")
        with open(os.path.join(self.session_dir, "labels", label_name), "w", encoding="utf-8") as f:
            json.dump(label, f)
        return image_path, box

    @staticmethod
    def _rotated_rectangle(center, width, height, degrees):
        angle = math.radians(degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        local = [
            (-width / 2.0, -height / 2.0),
            (width / 2.0, -height / 2.0),
            (width / 2.0, height / 2.0),
            (-width / 2.0, height / 2.0),
        ]
        return [
            (
                center[0] + x * cos_a - y * sin_a,
                center[1] + x * sin_a + y * cos_a,
            )
            for x, y in local
        ]

    @staticmethod
    def _edge_lengths(points):
        return [
            math.hypot(
                points[(index + 1) % 4][0] - points[index][0],
                points[(index + 1) % 4][1] - points[index][1],
            )
            for index in range(4)
        ]

    def test_in_bounds_export_keeps_image_bytes_and_historical_normalization(self):
        corners = [(10, 10), (40, 10), (40, 30), (10, 30)]
        source_path, _ = self._add_sample("in_bounds.png", corners)

        result = export_obb_dataset(self.session_dir, generate_synthetic=False)

        self.assertTrue(result["ok"], result)
        destination = os.path.join(
            self.session_dir,
            "obb_dataset",
            "images",
            "train",
            "in_bounds.png",
        )
        with open(source_path, "rb") as source, open(destination, "rb") as exported:
            self.assertEqual(source.read(), exported.read())
        label_path = os.path.join(
            self.session_dir,
            "obb_dataset",
            "labels",
            "train",
            "in_bounds.txt",
        )
        with open(label_path, "r", encoding="utf-8") as f:
            self.assertEqual(
                f.read(),
                "0 0.100000 0.125000 0.400000 0.125000 "
                "0.400000 0.375000 0.100000 0.375000\n",
            )
        with open(result["export_manifest_path"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["real_images"][0]["transform"]["type"], "identity")
        self.assertEqual(
            manifest["real_images"][0]["transform"]["padding"],
            {"left": 0, "top": 0, "right": 0, "bottom": 0},
        )

    def test_same_stem_different_extensions_get_unique_content_paired_exports(self):
        png_corners = [(10, 10), (40, 10), (40, 30), (10, 30)]
        jpg_corners = [(20, 20), (60, 20), (60, 50), (20, 50)]
        png_path, _ = self._add_sample(
            "paired.png",
            png_corners,
            pixel_value=73,
            label_filename="paired_png.json",
        )
        jpg_path, _ = self._add_sample(
            "paired.jpg",
            jpg_corners,
            pixel_value=149,
            label_filename="paired_jpg.json",
        )

        result = export_obb_dataset(
            self.session_dir,
            val_ratio=0.0,
            test_ratio=0.0,
            generate_synthetic=False,
        )

        self.assertTrue(result["ok"], result)
        manifest = json.loads(Path(result["export_manifest_path"]).read_text(encoding="utf-8"))
        entries = {entry["source_image"]: entry for entry in manifest["real_images"]}
        self.assertEqual(set(entries), {"paired.jpg", "paired.png"})
        self.assertEqual(
            len({Path(entry["exported_image"]).stem for entry in entries.values()}),
            2,
        )
        self.assertEqual(
            len({Path(entry["exported_label"]).stem for entry in entries.values()}),
            2,
        )

        source_paths = {"paired.jpg": jpg_path, "paired.png": png_path}
        expected_first_x = {"paired.jpg": 0.2, "paired.png": 0.1}
        output_root = Path(self.session_dir, "obb_dataset")
        for source_name, entry in entries.items():
            self.assertIn(f"--{entry['sample_id'][:12]}", Path(entry["exported_image"]).stem)
            self.assertEqual(
                Path(entry["exported_image"]).stem,
                Path(entry["exported_label"]).stem,
            )
            exported_image = output_root / entry["exported_image"]
            exported_label = output_root / entry["exported_label"]
            self.assertEqual(
                hashlib.sha256(exported_image.read_bytes()).hexdigest(),
                entry["exported_image_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(exported_label.read_bytes()).hexdigest(),
                entry["label_sha256"],
            )
            self.assertEqual(
                exported_image.read_bytes(),
                Path(source_paths[source_name]).read_bytes(),
            )
            values = [float(value) for value in exported_label.read_text(encoding="utf-8").split()]
            self.assertAlmostEqual(values[1], expected_first_x[source_name])

        self.assertEqual(len(list((output_root / "images" / "train").iterdir())), 2)
        self.assertEqual(len(list((output_root / "labels" / "train").iterdir())), 2)

    def test_outside_corners_pad_and_preserve_edges_angle_and_landmark_anchor(self):
        corners = self._rotated_rectangle(center=(7.0, 36.0), width=44.0, height=18.0, degrees=31.0)
        landmarks = [
            {"id": 1, "x": -3.0, "y": 34.0},
            {"id": 2, "x": 18.0, "y": 41.0},
        ]
        _, box = self._add_sample("outside.png", corners, landmarks=landmarks)
        original_image = np.zeros((80, 100, 3), dtype=np.uint8)

        _, translated_boxes, transform = _prepare_real_sample_for_export(
            original_image,
            [box],
            "outside.png",
        )
        self.assertGreater(transform["padding"]["left"], 0)
        translated = translated_boxes[0]
        dx = transform["offset_x"]
        dy = transform["offset_y"]
        for original_point, translated_point in zip(corners, translated["obbCorners"]):
            self.assertAlmostEqual(translated_point[0] - original_point[0], dx)
            self.assertAlmostEqual(translated_point[1] - original_point[1], dy)
        for original_landmark, translated_landmark in zip(landmarks, translated["landmarks"]):
            self.assertAlmostEqual(translated_landmark["x"] - original_landmark["x"], dx)
            self.assertAlmostEqual(translated_landmark["y"] - original_landmark["y"], dy)
            self.assertAlmostEqual(
                translated_landmark["x"] - translated["obbCorners"][0][0],
                original_landmark["x"] - corners[0][0],
            )
            self.assertAlmostEqual(
                translated_landmark["y"] - translated["obbCorners"][0][1],
                original_landmark["y"] - corners[0][1],
            )

        result = export_obb_dataset(self.session_dir, generate_synthetic=False)
        self.assertTrue(result["ok"], result)
        with open(result["export_manifest_path"], "r", encoding="utf-8") as f:
            manifest = json.load(f)
        entry = manifest["real_images"][0]
        exported_image_path = os.path.join(
            self.session_dir,
            "obb_dataset",
            entry["exported_image"].replace("/", os.sep),
        )
        exported_image = cv2.imread(exported_image_path)
        self.assertIsNotNone(exported_image)
        export_height, export_width = exported_image.shape[:2]
        label_path = os.path.join(
            self.session_dir,
            "obb_dataset",
            "labels",
            entry["split"],
            "outside.txt",
        )
        with open(label_path, "r", encoding="utf-8") as f:
            values = [float(value) for value in f.read().split()[1:]]
        reconstructed = [
            (values[index] * export_width, values[index + 1] * export_height)
            for index in range(0, 8, 2)
        ]
        expected = [(x + dx, y + dy) for x, y in corners]
        for actual, expected_point in zip(reconstructed, expected):
            self.assertAlmostEqual(actual[0], expected_point[0], places=4)
            self.assertAlmostEqual(actual[1], expected_point[1], places=4)
        for actual, original in zip(self._edge_lengths(reconstructed), self._edge_lengths(corners)):
            self.assertAlmostEqual(actual, original, delta=1e-4)
        original_angle = math.atan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0])
        exported_angle = math.atan2(
            reconstructed[1][1] - reconstructed[0][1],
            reconstructed[1][0] - reconstructed[0][0],
        )
        self.assertAlmostEqual(exported_angle, original_angle, places=5)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_persisted_splits_keep_existing_samples_stable_when_dataset_grows(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(5):
            self._add_sample(
                f"sample_{index:02d}.png",
                corners,
                pixel_value=20 + index,
                source_group="paired-capture" if index < 2 else None,
            )

        first = export_obb_dataset(
            self.session_dir,
            val_ratio=0.4,
            seed=19,
            generate_synthetic=False,
        )
        self.assertTrue(first["ok"], first)
        with open(first["export_manifest_path"], "r", encoding="utf-8") as f:
            first_manifest = json.load(f)
        first_splits = {
            entry["sample_id"]: entry["split"] for entry in first_manifest["real_images"]
        }
        first_val_ids = {
            sample_id for sample_id, split in first_splits.items() if split == "val"
        }
        self.assertTrue(first_val_ids, "bootstrap should create a validation cohort")
        paired_splits = {
            entry["split"]
            for entry in first_manifest["real_images"]
            if entry["source_image"] in {"sample_00.png", "sample_01.png"}
        }
        self.assertEqual(len(paired_splits), 1, "a source group leaked across train and val")

        for index in range(5, 9):
            # These names sort before the original files, proving stability is
            # identity-based rather than an artifact of list position.
            self._add_sample(f"added_{index:02d}.png", corners, pixel_value=20 + index)

        second = export_obb_dataset(
            self.session_dir,
            val_ratio=0.4,
            seed=19,
            generate_synthetic=False,
        )
        self.assertTrue(second["ok"], second)
        with open(second["export_manifest_path"], "r", encoding="utf-8") as f:
            second_manifest = json.load(f)
        second_splits = {
            entry["sample_id"]: entry["split"] for entry in second_manifest["real_images"]
        }
        self.assertEqual(
            first_splits,
            {sample_id: second_splits[sample_id] for sample_id in first_splits},
        )
        second_val_ids = {
            sample_id for sample_id, split in second_splits.items() if split == "val"
        }
        self.assertEqual(
            second_val_ids,
            first_val_ids,
            "adding samples must not expand the frozen validation cohort",
        )
        self.assertEqual(
            second["validation_cohort"]["sha256"],
            first["validation_cohort"]["sha256"],
            "train-only additions must not change the locked validation cohort identity",
        )
        added_entries = [
            entry for entry in second_manifest["real_images"]
            if entry["source_image"].startswith("added_")
        ]
        self.assertEqual(len(added_entries), 4)
        self.assertTrue(all(entry["split"] == "train" for entry in added_entries))
        self.assertTrue(os.path.exists(second["split_assignments_path"]))
        with open(second["split_assignments_path"], "r", encoding="utf-8") as f:
            assignments = json.load(f)
        self.assertEqual(assignments["version"], OBB_SPLIT_ASSIGNMENTS_VERSION)

    def test_validation_annotation_change_fails_closed(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(4):
            self._add_sample(
                f"cohort_{index:02d}.png",
                corners,
                pixel_value=100 + index,
            )

        first = export_obb_dataset(
            self.session_dir,
            val_ratio=0.5,
            seed=23,
            generate_synthetic=False,
        )
        self.assertTrue(first["ok"], first)
        with open(first["export_manifest_path"], "rb") as handle:
            locked_export_bytes = handle.read()
        with open(first["export_manifest_path"], "r", encoding="utf-8") as handle:
            first_manifest = json.load(handle)
        val_entry = next(entry for entry in first_manifest["real_images"] if entry["split"] == "val")
        label_path = os.path.join(
            self.session_dir,
            "labels",
            os.path.splitext(val_entry["source_image"])[0] + ".json",
        )
        with open(label_path, "r", encoding="utf-8") as handle:
            label = json.load(handle)
        revised_corners = [[14.0, 12.0], [44.0, 12.0], [44.0, 32.0], [14.0, 32.0]]
        label["boxes"][0]["obbCorners"] = revised_corners
        label["finalizedDetection"]["acceptedBoxes"][0]["obbCorners"] = revised_corners
        with open(label_path, "w", encoding="utf-8") as handle:
            json.dump(label, handle)

        second = export_obb_dataset(
            self.session_dir,
            val_ratio=0.5,
            seed=23,
            generate_synthetic=False,
        )
        self.assertFalse(second["ok"], second)
        self.assertIn(
            "frozen OBB validation snapshot changed",
            second["error"],
        )
        with open(first["export_manifest_path"], "rb") as handle:
            self.assertEqual(
                handle.read(),
                locked_export_bytes,
                "a rejected cohort mutation must not erase the last usable export",
            )

    def test_locked_evaluator_cannot_later_become_hitl_review_data(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(4):
            self._add_sample(
                f"review_guard_{index:02d}.png",
                corners,
                pixel_value=140 + index,
            )

        first = export_obb_dataset(
            self.session_dir,
            val_ratio=0.25,
            seed=29,
            generate_synthetic=False,
        )
        self.assertTrue(first["ok"], first)
        with open(first["export_manifest_path"], "rb") as handle:
            locked_export_bytes = handle.read()
        with open(first["export_manifest_path"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        evaluator = next(
            entry for entry in manifest["real_images"] if entry["split"] in {"val", "test"}
        )
        label_path = os.path.join(
            self.session_dir,
            "labels",
            os.path.splitext(evaluator["source_image"])[0] + ".json",
        )
        with open(label_path, "r", encoding="utf-8") as handle:
            label = json.load(handle)
        review = {
            "source": "hitl_review",
            "eventId": "reviewed-frozen-evaluator",
            "reviewOutcome": "accepted_unchanged",
        }
        label["provenance"] = review
        label["reviewHistory"] = [review]
        with open(label_path, "w", encoding="utf-8") as handle:
            json.dump(label, handle)

        second = export_obb_dataset(
            self.session_dir,
            val_ratio=0.25,
            seed=29,
            generate_synthetic=False,
        )
        self.assertFalse(second["ok"], second)
        self.assertIn("frozen OBB evaluator source", second["error"])
        with open(first["export_manifest_path"], "rb") as handle:
            self.assertEqual(handle.read(), locked_export_bytes)

    def test_one_source_growth_bootstraps_missing_evaluators_once_without_reshuffling(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        self._add_sample("original.png", corners, pixel_value=31)

        first = export_obb_dataset(self.session_dir, seed=71, generate_synthetic=False)
        self.assertTrue(first["ok"], first)
        with open(first["export_manifest_path"], "r", encoding="utf-8") as handle:
            first_manifest = json.load(handle)
        self.assertEqual(first_manifest["real_images"][0]["split"], "train")
        self.assertFalse(first["validation_cohort"]["frozen"])
        self.assertFalse(first["test_cohort"]["frozen"])

        self._add_sample("new_validation_candidate.png", corners, pixel_value=32)
        self._add_sample("new_test_candidate.png", corners, pixel_value=33)
        second = export_obb_dataset(self.session_dir, seed=71, generate_synthetic=False)
        self.assertTrue(second["ok"], second)
        with open(second["export_manifest_path"], "r", encoding="utf-8") as handle:
            second_manifest = json.load(handle)
        splits = {
            entry["source_image"]: entry["split"]
            for entry in second_manifest["real_images"]
        }
        self.assertEqual(splits["original.png"], "train")
        self.assertEqual(set(splits.values()), {"train", "val", "test"})
        self.assertFalse(second["validation_cohort"]["frozen"])
        self.assertTrue(second["test_cohort"]["frozen"])
        self.assertTrue(all(second["cohort_disjoint"].values()))
        locked_test = second["test_cohort"]["sha256"]

        self._add_sample("validation_closer.png", corners, pixel_value=34)
        third = export_obb_dataset(self.session_dir, seed=71, generate_synthetic=False)
        self.assertTrue(third["ok"], third)
        with open(third["export_manifest_path"], "r", encoding="utf-8") as handle:
            third_manifest = json.load(handle)
        closer = next(
            entry for entry in third_manifest["real_images"]
            if entry["source_image"] == "validation_closer.png"
        )
        self.assertEqual(closer["split"], "val")
        self.assertTrue(third["validation_cohort"]["frozen"])
        self.assertEqual(third["test_cohort"]["sha256"], locked_test)

        locked_validation = third["validation_cohort"]["sha256"]
        self._add_sample("later_train_only.png", corners, pixel_value=35)
        fourth = export_obb_dataset(self.session_dir, seed=71, generate_synthetic=False)
        self.assertTrue(fourth["ok"], fourth)
        fourth_manifest = json.loads(
            Path(fourth["export_manifest_path"]).read_text(encoding="utf-8")
        )
        added = next(
            entry for entry in fourth_manifest["real_images"]
            if entry["source_image"] == "later_train_only.png"
        )
        self.assertEqual(added["split"], "train")
        self.assertEqual(fourth["validation_cohort"]["sha256"], locked_validation)
        self.assertEqual(fourth["test_cohort"]["sha256"], locked_test)

    def test_three_source_start_uses_only_unseen_groups_to_finish_validation_lock(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(3):
            self._add_sample(f"initial_{index}.png", corners, pixel_value=40 + index)

        first = export_obb_dataset(self.session_dir, seed=79, generate_synthetic=False)
        self.assertTrue(first["ok"], first)
        first_manifest = json.loads(
            Path(first["export_manifest_path"]).read_text(encoding="utf-8")
        )
        first_splits = {
            entry["sample_id"]: entry["split"] for entry in first_manifest["real_images"]
        }
        self.assertEqual(list(first_splits.values()).count("val"), 1)
        self.assertFalse(first["validation_cohort"]["frozen"])

        self._add_sample("unseen_closer.png", corners, pixel_value=44)
        self._add_sample("unseen_surplus.png", corners, pixel_value=45)
        second = export_obb_dataset(self.session_dir, seed=79, generate_synthetic=False)
        self.assertTrue(second["ok"], second)
        second_manifest = json.loads(
            Path(second["export_manifest_path"]).read_text(encoding="utf-8")
        )
        second_splits = {
            entry["sample_id"]: entry["split"] for entry in second_manifest["real_images"]
        }
        self.assertEqual(
            first_splits,
            {sample_id: second_splits[sample_id] for sample_id in first_splits},
            "historically trained/evaluated groups must never be repartitioned",
        )
        new_entries = [
            entry for entry in second_manifest["real_images"]
            if entry["source_image"].startswith("unseen_")
        ]
        self.assertEqual(sorted(entry["split"] for entry in new_entries), ["train", "val"])
        self.assertEqual(second["validation_cohort"]["group_count"], 2)
        self.assertTrue(second["validation_cohort"]["frozen"])

        locked_revision = second["validation_cohort"]["sha256"]
        self._add_sample("post_lock.png", corners, pixel_value=46)
        third = export_obb_dataset(self.session_dir, seed=79, generate_synthetic=False)
        self.assertTrue(third["ok"], third)
        third_manifest = json.loads(
            Path(third["export_manifest_path"]).read_text(encoding="utf-8")
        )
        post_lock = next(
            entry for entry in third_manifest["real_images"]
            if entry["source_image"] == "post_lock.png"
        )
        self.assertEqual(post_lock["split"], "train")
        self.assertEqual(third["validation_cohort"]["sha256"], locked_revision)

    def test_late_rare_orientation_class_closes_incomplete_validation_once(self):
        session_path = Path(self.session_dir, "session.json")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["orientationPolicy"] = {"mode": "directional"}
        session_path.write_text(json.dumps(session), encoding="utf-8")
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(4):
            self._add_sample(
                f"common_{index}.png",
                corners,
                pixel_value=90 + index,
                class_id=0,
            )

        first = export_obb_dataset(self.session_dir, seed=83, generate_synthetic=False)
        self.assertTrue(first["ok"], first)
        self.assertEqual(first["validation_cohort"]["group_count"], 2)
        self.assertFalse(first["validation_cohort"]["frozen"])
        self.assertEqual(first["validation_cohort"]["real_class_histogram"]["1"], 0)

        self._add_sample("more_common.png", corners, pixel_value=95, class_id=0)
        second = export_obb_dataset(self.session_dir, seed=83, generate_synthetic=False)
        self.assertTrue(second["ok"], second)
        second_manifest = json.loads(
            Path(second["export_manifest_path"]).read_text(encoding="utf-8")
        )
        more_common = next(
            entry for entry in second_manifest["real_images"]
            if entry["source_image"] == "more_common.png"
        )
        self.assertEqual(more_common["split"], "train")
        self.assertFalse(second["validation_cohort"]["frozen"])

        self._add_sample("late_rare.png", corners, pixel_value=96, class_id=1)
        third = export_obb_dataset(self.session_dir, seed=83, generate_synthetic=False)
        self.assertTrue(third["ok"], third)
        third_manifest = json.loads(
            Path(third["export_manifest_path"]).read_text(encoding="utf-8")
        )
        late_rare = next(
            entry for entry in third_manifest["real_images"]
            if entry["source_image"] == "late_rare.png"
        )
        self.assertEqual(late_rare["split"], "val")
        self.assertTrue(third["validation_cohort"]["frozen"])
        self.assertGreater(third["validation_cohort"]["real_class_histogram"]["1"], 0)
        locked_revision = third["validation_cohort"]["sha256"]

        self._add_sample("late_rare_after_lock.png", corners, pixel_value=97, class_id=1)
        fourth = export_obb_dataset(self.session_dir, seed=83, generate_synthetic=False)
        self.assertTrue(fourth["ok"], fourth)
        fourth_manifest = json.loads(
            Path(fourth["export_manifest_path"]).read_text(encoding="utf-8")
        )
        after_lock = next(
            entry for entry in fourth_manifest["real_images"]
            if entry["source_image"] == "late_rare_after_lock.png"
        )
        self.assertEqual(after_lock["split"], "train")
        self.assertEqual(fourth["validation_cohort"]["sha256"], locked_revision)

    def test_hitl_groups_are_train_only_during_first_three_way_bootstrap(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        self._add_sample(
            "reviewed.png",
            corners,
            pixel_value=50,
            provenance={"source": "hitl_review", "commitId": "commit-1"},
        )
        for index in range(3):
            self._add_sample(f"manual_{index}.png", corners, pixel_value=51 + index)

        result = export_obb_dataset(self.session_dir, seed=17, generate_synthetic=False)
        self.assertTrue(result["ok"], result)
        with open(result["export_manifest_path"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        reviewed = next(
            entry for entry in manifest["real_images"] if entry["source_image"] == "reviewed.png"
        )
        self.assertEqual(reviewed["split"], "train")
        evaluation_names = {
            entry["source_image"]
            for entry in manifest["real_images"]
            if entry["split"] in {"val", "test"}
        }
        self.assertNotIn("reviewed.png", evaluation_names)
        self.assertEqual({entry["split"] for entry in manifest["real_images"]}, {"train", "val", "test"})

    def test_hitl_rejected_all_does_not_resurrect_nonempty_draft_boxes(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        self._add_sample("kept.png", corners, pixel_value=61)
        self._add_sample("rejected_all.png", corners, pixel_value=62)
        rejected_label_path = Path(
            self.session_dir,
            "labels",
            "rejected_all.json",
        )
        rejected_label = json.loads(rejected_label_path.read_text(encoding="utf-8"))
        self.assertTrue(rejected_label["boxes"], "the rejected sample must retain draft boxes")
        review = {
            "source": "hitl_review",
            "eventId": "review-rejected-all",
            "reviewOutcome": "rejected_all",
        }
        rejected_label["provenance"] = dict(review)
        rejected_label["reviewHistory"] = [dict(review)]
        rejected_label["finalizedDetection"]["acceptedBoxes"] = []
        rejected_label_path.write_text(json.dumps(rejected_label), encoding="utf-8")

        is_finalized, accepted, used_fallback = _get_finalized_boxes(
            rejected_label,
            "rejected_all.png",
            set(),
        )
        self.assertTrue(is_finalized)
        self.assertEqual(accepted, [])
        self.assertFalse(used_fallback)

        result = export_obb_dataset(
            self.session_dir,
            seed=181,
            generate_synthetic=False,
        )

        self.assertTrue(result["ok"], result)
        # The rejected image is exported as a background negative: it carries no
        # boxes, so it adds an image without adding geometry.
        self.assertEqual(result["num_images"], 2)
        self.assertEqual(result["num_boxes"], 1)
        manifest = json.loads(
            Path(result["export_manifest_path"]).read_text(encoding="utf-8")
        )
        exported = {entry["source_image"]: entry for entry in manifest["real_images"]}
        self.assertEqual(set(exported), {"kept.png", "rejected_all.png"})
        self.assertEqual(exported["kept.png"]["boxes_exported"], 1)
        self.assertEqual(exported["rejected_all.png"]["boxes_exported"], 0)

        # Its label file must exist and be empty - that is what makes it a
        # negative rather than an unlabeled image.
        output_root = Path(self.session_dir, "obb_dataset")
        negative_labels = list(output_root.rglob("rejected_all*.txt"))
        self.assertEqual(len(negative_labels), 1, negative_labels)
        self.assertEqual(negative_labels[0].read_text(encoding="utf-8").strip(), "")
        self.assertTrue(list(output_root.rglob("rejected_all*.png")))

        # Confirmed negatives are train-only: they must never enter an evaluator
        # cohort, where they would change the mAP denominator.
        self.assertEqual(exported["rejected_all.png"]["split"], "train")

    def test_finalized_image_without_declared_accepted_boxes_is_not_a_negative(self):
        """Legacy finalized data that never declared acceptedBoxes stays excluded."""
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        self._add_sample("kept.png", corners, pixel_value=71)
        self._add_sample("legacy_empty.png", corners, pixel_value=72)
        legacy_path = Path(self.session_dir, "labels", "legacy_empty.json")
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy["boxes"] = []
        legacy["finalizedDetection"].pop("acceptedBoxes", None)
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

        self.assertFalse(_is_confirmed_negative_review(legacy))

        result = export_obb_dataset(
            self.session_dir,
            seed=181,
            generate_synthetic=False,
        )
        self.assertTrue(result["ok"], result)
        manifest = json.loads(
            Path(result["export_manifest_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            [entry["source_image"] for entry in manifest["real_images"]],
            ["kept.png"],
        )

    def test_v1_migration_preserves_validation_and_uses_only_new_group_for_test(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        paths = []
        for index in range(3):
            path, _ = self._add_sample(
                f"legacy_{index}.png",
                corners,
                pixel_value=80 + index,
            )
            paths.append(path)
        identities = [
            _stable_sample_identity(path, os.path.basename(path))
            for path in paths
        ]
        legacy_groups = {
            identities[0][1]: "val",
            identities[1][1]: "train",
        }
        legacy_samples = {
            identities[index][0]: {
                "split": "val" if index == 0 else "train",
                "group_id": identities[index][1],
                "image_filename": os.path.basename(paths[index]),
                "content_sha256": identities[index][2],
            }
            for index in range(2)
        }
        legacy_path = os.path.join(
            self.session_dir,
            f"obb_split_assignments.v{OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION}.json",
        )
        with open(legacy_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "version": OBB_LEGACY_SPLIT_ASSIGNMENTS_VERSION,
                    "profiles": {
                        "seed=42;val_ratio=0.20000000": {
                            "seed": 42,
                            "val_ratio": 0.2,
                            "groups": legacy_groups,
                            "samples": legacy_samples,
                        }
                    },
                },
                handle,
            )

        result = export_obb_dataset(self.session_dir, seed=42, generate_synthetic=False)
        self.assertTrue(result["ok"], result)
        with open(result["export_manifest_path"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        splits = {entry["source_image"]: entry["split"] for entry in manifest["real_images"]}
        self.assertEqual(splits["legacy_0.png"], "val")
        self.assertEqual(splits["legacy_1.png"], "train")
        self.assertEqual(splits["legacy_2.png"], "test")
        with open(result["split_assignments_path"], "r", encoding="utf-8") as handle:
            migrated = json.load(handle)
        self.assertEqual(migrated["version"], OBB_SPLIT_ASSIGNMENTS_VERSION)
        self.assertEqual(migrated["migrated_from"]["version"], 1)

    def test_tampered_v2_assignment_revision_fails_closed(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(3):
            self._add_sample(f"tamper_{index}.png", corners, pixel_value=140 + index)
        first = export_obb_dataset(self.session_dir, seed=91, generate_synthetic=False)
        self.assertTrue(first["ok"], first)
        with open(first["split_assignments_path"], "r", encoding="utf-8") as handle:
            assignments = json.load(handle)
        profile = next(iter(assignments["profiles"].values()))
        group_id = next(iter(profile["groups"]))
        profile["groups"][group_id] = (
            "train" if profile["groups"][group_id] != "train" else "test"
        )
        with open(first["split_assignments_path"], "w", encoding="utf-8") as handle:
            json.dump(assignments, handle)

        second = export_obb_dataset(self.session_dir, seed=91, generate_synthetic=False)
        self.assertFalse(second["ok"], second)
        self.assertIn("were mutated or are corrupt", second["error"])

    def test_existing_v2_manifest_requires_every_evaluator_revision_and_snapshot(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(4):
            self._add_sample(f"strict_{index}.png", corners, pixel_value=155 + index)
        first = export_obb_dataset(self.session_dir, seed=97, generate_synthetic=False)
        self.assertTrue(first["ok"], first)
        assignments_path = Path(first["split_assignments_path"])
        baseline = json.loads(assignments_path.read_text(encoding="utf-8"))
        profile_key = next(iter(baseline["profiles"]))

        cases = []
        missing_revision = json.loads(json.dumps(baseline))
        del missing_revision["profiles"][profile_key]["validation_cohort_revision"]
        cases.append(("missing revision", missing_revision, "missing required v2 fields"))

        missing_snapshot = json.loads(json.dumps(baseline))
        snapshots = missing_snapshot["profiles"][profile_key]["validation_group_snapshots"]
        snapshots.pop(next(iter(snapshots)))
        cases.append(("missing snapshot", missing_snapshot, "exact snapshot"))

        for case_name, payload, expected_error in cases:
            with self.subTest(case=case_name):
                assignments_path.write_text(json.dumps(payload), encoding="utf-8")
                rejected = export_obb_dataset(
                    self.session_dir,
                    seed=97,
                    generate_synthetic=False,
                )
                self.assertFalse(rejected["ok"], rejected)
                self.assertIn(expected_error, rejected["error"])
                assignments_path.write_text(json.dumps(baseline), encoding="utf-8")

    def test_existing_empty_v2_manifest_cannot_silently_bootstrap(self):
        assignments_path = Path(
            self.session_dir,
            f"obb_split_assignments.v{OBB_SPLIT_ASSIGNMENTS_VERSION}.json",
        )
        assignments_path.write_text(
            json.dumps({"version": OBB_SPLIT_ASSIGNMENTS_VERSION, "profiles": {}}),
            encoding="utf-8",
        )
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        self._add_sample("cannot_bootstrap.png", corners, pixel_value=161)

        result = export_obb_dataset(self.session_dir, seed=42, generate_synthetic=False)

        self.assertFalse(result["ok"], result)
        self.assertIn("has no valid profiles", result["error"])

    def test_changed_seed_cannot_create_implicit_split_profile(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(4):
            self._add_sample(f"seed_guard_{index}.png", corners, pixel_value=165 + index)
        first = export_obb_dataset(self.session_dir, seed=42, generate_synthetic=False)
        self.assertTrue(first["ok"], first)
        first_export_bytes = Path(first["export_manifest_path"]).read_bytes()

        second = export_obb_dataset(self.session_dir, seed=43, generate_synthetic=False)

        self.assertFalse(second["ok"], second)
        self.assertIn("requires an explicit future cohort-version workflow", second["error"])
        self.assertEqual(Path(first["export_manifest_path"]).read_bytes(), first_export_bytes)

    def test_test_annotation_mutation_is_rejected_as_report_benchmark_tampering(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(3):
            self._add_sample(f"test_lock_{index}.png", corners, pixel_value=170 + index)
        first = export_obb_dataset(self.session_dir, seed=101, generate_synthetic=False)
        self.assertTrue(first["ok"], first)
        with open(first["export_manifest_path"], "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        test_entry = next(entry for entry in manifest["real_images"] if entry["split"] == "test")
        label_path = os.path.join(
            self.session_dir,
            "labels",
            os.path.splitext(test_entry["source_image"])[0] + ".json",
        )
        with open(label_path, "r", encoding="utf-8") as handle:
            label = json.load(handle)
        moved = [[13.0, 12.0], [43.0, 12.0], [43.0, 32.0], [13.0, 32.0]]
        label["boxes"][0]["obbCorners"] = moved
        label["finalizedDetection"]["acceptedBoxes"][0]["obbCorners"] = moved
        with open(label_path, "w", encoding="utf-8") as handle:
            json.dump(label, handle)

        second = export_obb_dataset(self.session_dir, seed=101, generate_synthetic=False)
        self.assertFalse(second["ok"], second)
        self.assertIn("frozen OBB test snapshot changed", second["error"])

    def test_synthetic_pixels_labels_and_manifest_are_in_effective_dataset_closure(self):
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(3):
            self._add_sample(
                f"effective_{index}.png",
                corners,
                pixel_value=190 + index,
            )

        baseline_label = (
            "0 0.100000 0.100000 0.500000 0.100000 "
            "0.500000 0.500000 0.100000 0.500000\n"
        )

        def synthetic_generator(*, image_value, label_text, manifest_token):
            def generate(**kwargs):
                output_dir = Path(kwargs["out_dir"])
                split = kwargs["split"]
                image_dir = output_dir / "images" / split
                label_dir = output_dir / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / "__synth_obb_00000.png"
                label_path = label_dir / "__synth_obb_00000.txt"
                image = np.full((24, 32, 3), int(image_value), dtype=np.uint8)
                if not cv2.imwrite(str(image_path), image):
                    raise AssertionError("mock synthetic image could not be written")
                label_path.write_text(label_text, encoding="utf-8")
                kwargs["manifest"].append(
                    {
                        "image": image_path.name,
                        "label": label_path.name,
                        "generatorProtocol": "focused-effective-dataset-test-v1",
                        "manifestToken": manifest_token,
                    }
                )
                return {
                    "num_generated": 1,
                    "num_instances_generated": 1,
                    "segments_total": 1,
                    "segments_missing_anchors": 0,
                    "class_histogram": {0: 1},
                }

            return generate

        def run_export(*, image_value=25, label_text=baseline_label, manifest_token="a"):
            with patch.object(
                export_yolo_dataset_module,
                "_generate_synthetic_obb_images",
                side_effect=synthetic_generator(
                    image_value=image_value,
                    label_text=label_text,
                    manifest_token=manifest_token,
                ),
            ):
                result = export_obb_dataset(
                    self.session_dir,
                    seed=211,
                    generate_synthetic=True,
                )
            self.assertTrue(result["ok"], result)
            effective = result["effective_dataset"]
            self.assertEqual(len(effective["synthetic_files"]), 1)
            synthetic_file = effective["synthetic_files"][0]
            self.assertEqual(synthetic_file["image"], "images/train/__synth_obb_00000.png")
            self.assertEqual(synthetic_file["label"], "labels/train/__synth_obb_00000.txt")
            dataset_root = Path(result["yaml_path"]).parent
            self.assertEqual(
                synthetic_file["image_sha256"],
                hashlib.sha256((dataset_root / synthetic_file["image"]).read_bytes()).hexdigest(),
            )
            self.assertEqual(
                synthetic_file["label_sha256"],
                hashlib.sha256((dataset_root / synthetic_file["label"]).read_bytes()).hexdigest(),
            )
            synthetic_manifest = Path(result["synthetic_manifest_path"])
            self.assertEqual(
                effective["synthetic_manifest_sha256"],
                hashlib.sha256(synthetic_manifest.read_bytes()).hexdigest(),
            )
            export_manifest = json.loads(
                Path(result["export_manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(export_manifest["effective_dataset"], effective)
            self.assertEqual(
                export_manifest["synthetic_manifest"]["sha256"],
                effective["synthetic_manifest_sha256"],
            )
            return effective

        baseline = run_export()
        changed_pixels = run_export(image_value=26)
        changed_label = run_export(
            label_text=(
                "0 0.125000 0.100000 0.525000 0.100000 "
                "0.525000 0.500000 0.125000 0.500000\n"
            )
        )
        changed_manifest = run_export(manifest_token="b")

        baseline_file = baseline["synthetic_files"][0]
        pixel_file = changed_pixels["synthetic_files"][0]
        label_file = changed_label["synthetic_files"][0]
        manifest_file = changed_manifest["synthetic_files"][0]
        self.assertNotEqual(pixel_file["image_sha256"], baseline_file["image_sha256"])
        self.assertEqual(pixel_file["label_sha256"], baseline_file["label_sha256"])
        self.assertNotEqual(changed_pixels["revision"], baseline["revision"])
        self.assertEqual(label_file["image_sha256"], baseline_file["image_sha256"])
        self.assertNotEqual(label_file["label_sha256"], baseline_file["label_sha256"])
        self.assertNotEqual(changed_label["revision"], baseline["revision"])
        self.assertEqual(manifest_file, baseline_file)
        self.assertNotEqual(
            changed_manifest["synthetic_manifest_sha256"],
            baseline["synthetic_manifest_sha256"],
        )
        self.assertNotEqual(changed_manifest["revision"], baseline["revision"])

    def test_malformed_accepted_box_aborts_before_replacing_last_usable_export(self):
        primary_corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        self._add_sample("mixed_geometry.png", primary_corners, pixel_value=219)
        first = export_obb_dataset(self.session_dir, generate_synthetic=False)
        self.assertTrue(first["ok"], first)

        output_root = Path(self.session_dir, "obb_dataset")

        def output_snapshot():
            return {
                path.relative_to(output_root).as_posix(): path.read_bytes()
                for path in sorted(output_root.rglob("*"))
                if path.is_file()
            }

        usable_export = output_snapshot()
        self.assertTrue(usable_export)
        label_path = Path(self.session_dir, "labels", "mixed_geometry.json")
        label = json.loads(label_path.read_text(encoding="utf-8"))
        second_valid = {
            "left": 50,
            "top": 10,
            "width": 25,
            "height": 20,
            "obbCorners": [[50, 10], [75, 10], [75, 30], [50, 30]],
            "landmarks": [],
        }
        # Keep a valid matching draft box to prove explicitly malformed accepted
        # geometry is rejected rather than backfilled, warned about, or dropped.
        label["boxes"].append(second_valid)
        accepted_primary = label["finalizedDetection"]["acceptedBoxes"][0]
        malformed_cases = (
            (
                "wrong corner count",
                [[50, 10], [75, 10], [75, 30]],
                "exactly four corners",
            ),
            (
                "non-finite coordinate",
                [[50, 10], [75, 10], [75, "not-finite"], [50, 30]],
                "non-finite coordinate",
            ),
            (
                "non-rectangular quadrilateral",
                [[50, 10], [75, 10], [72, 30], [50, 30]],
                "not rectangular",
            ),
        )

        for case_name, malformed_corners, expected_error in malformed_cases:
            with self.subTest(case=case_name):
                malformed_accepted = {
                    **second_valid,
                    "obbCorners": malformed_corners,
                }
                label["finalizedDetection"]["acceptedBoxes"] = [
                    accepted_primary,
                    malformed_accepted,
                ]
                label_path.write_text(json.dumps(label), encoding="utf-8")

                rejected = export_obb_dataset(
                    self.session_dir,
                    generate_synthetic=False,
                )

                self.assertFalse(rejected["ok"], rejected)
                self.assertIn(expected_error, rejected["error"])
                self.assertIn("accepted box 2", rejected["error"])
                self.assertEqual(
                    output_snapshot(),
                    usable_export,
                    "a rejected accepted box must not replace any prior export artifact",
                )

    def test_directional_export_requires_a_trusted_binary_class_for_every_box(self):
        session_path = Path(self.session_dir, "session.json")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["orientationPolicy"] = {"mode": "directional"}
        session_path.write_text(json.dumps(session), encoding="utf-8")
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        self._add_sample("directional_invalid.png", corners, pixel_value=224)
        label_path = Path(self.session_dir, "labels", "directional_invalid.json")
        baseline_label = json.loads(label_path.read_text(encoding="utf-8"))

        invalid_cases = (
            ("missing orientation", {}, "could not resolve a trusted left/right orientation"),
            (
                "uncertain orientation",
                {"orientation_override": "uncertain"},
                "orientation is explicitly uncertain",
            ),
            ("out-of-range class", {"class_id": 2}, "class_id 2 is out of range"),
        )
        for case_name, metadata, expected_error in invalid_cases:
            with self.subTest(case=case_name):
                label = json.loads(json.dumps(baseline_label))
                for box in (
                    label["boxes"][0],
                    label["finalizedDetection"]["acceptedBoxes"][0],
                ):
                    box.pop("class_id", None)
                    box.pop("orientation_override", None)
                    box.pop("orientation_hint", None)
                    box.update(metadata)
                label_path.write_text(json.dumps(label), encoding="utf-8")

                rejected = export_obb_dataset(
                    self.session_dir,
                    generate_synthetic=False,
                )

                self.assertFalse(rejected["ok"], rejected)
                self.assertIn("directional_invalid.png accepted box 1", rejected["error"])
                self.assertIn(expected_error, rejected["error"])
                self.assertIn("trusted class in {0, 1}", rejected["error"])
                self.assertFalse(Path(self.session_dir, "obb_dataset").exists())

    def test_directional_validation_cohort_persists_normalized_class_coverage(self):
        session_path = Path(self.session_dir, "session.json")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["orientationPolicy"] = {"mode": "directional"}
        session_path.write_text(json.dumps(session), encoding="utf-8")
        corners = [(12, 12), (42, 12), (42, 32), (12, 32)]
        for index in range(6):
            filename = f"directional_coverage_{index}.png"
            self._add_sample(filename, corners, pixel_value=230 + index)
            label_path = Path(
                self.session_dir,
                "labels",
                f"directional_coverage_{index}.json",
            )
            label = json.loads(label_path.read_text(encoding="utf-8"))
            for box in (
                label["boxes"][0],
                label["finalizedDetection"]["acceptedBoxes"][0],
            ):
                box["class_id"] = index % 2
            label_path.write_text(json.dumps(label), encoding="utf-8")

        result = export_obb_dataset(
            self.session_dir,
            val_ratio=0.5,
            test_ratio=0.15,
            seed=307,
            generate_synthetic=False,
        )

        self.assertTrue(result["ok"], result)
        cohort = result["validation_cohort"]
        self.assertEqual(cohort["expected_class_count"], 2)
        self.assertEqual(set(cohort["real_class_histogram"]), {"0", "1"})
        self.assertEqual(
            sum(cohort["real_class_histogram"].values()),
            cohort["sample_count"],
        )
        self.assertGreater(cohort["real_class_histogram"]["0"], 0)
        self.assertGreater(cohort["real_class_histogram"]["1"], 0)
        manifest = json.loads(
            Path(result["export_manifest_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["validation_cohort"], cohort)

    def test_invalid_real_obb_geometry_fails_with_repair_guidance(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        bad_box = {
            "left": 1,
            "top": 1,
            "width": 10,
            "height": 10,
            "obbCorners": [(1, 1), (1, 1), (10, 10), (1, 10)],
            "landmarks": [],
        }
        with self.assertRaisesRegex(ValueError, "zero-length edge.*repair or redraw"):
            _prepare_real_sample_for_export(image, [bad_box], "invalid.png")

    def test_convex_non_rectangular_obb_is_rejected(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        trapezoid = {
            "left": 0,
            "top": 0,
            "width": 100,
            "height": 90,
            "obbCorners": [(-5, 20), (80, 10), (90, 70), (0, 85)],
            "landmarks": [],
        }

        with self.assertRaisesRegex(ValueError, "not rectangular.*repair or redraw"):
            _prepare_real_sample_for_export(image, [trapezoid], "trapezoid.png")

    def test_small_integer_rounding_error_still_accepts_rectangular_obb(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        rounded_rectangle = {
            "left": 9,
            "top": 10,
            "width": 81,
            "height": 40,
            "obbCorners": [(10, 10), (90, 12), (89, 50), (9, 49)],
            "landmarks": [],
        }

        prepared, boxes, _padding = _prepare_real_sample_for_export(
            image, [rounded_rectangle], "rounded.png"
        )
        self.assertEqual(prepared.shape[:2], image.shape[:2])
        self.assertEqual(len(boxes), 1)


if __name__ == "__main__":
    unittest.main()
