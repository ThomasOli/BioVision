import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np

from backend.bv_utils import lineage
from backend.training import train_shape_model
from backend.training.train_shape_model import (
    augment_training_data,
    _assert_effective_dlib_dataset_unchanged,
    _build_dlib_training_protocol,
    _build_effective_dlib_dataset_snapshot,
)


class DlibEffectiveDatasetTests(unittest.TestCase):
    def _write_xml(self, path, image_path, *, x=11, y=22):
        root = ET.Element("dataset")
        images = ET.SubElement(root, "images")
        image = ET.SubElement(images, "image", file=image_path)
        box = ET.SubElement(
            image,
            "box",
            left="0",
            top="0",
            width="512",
            height="512",
        )
        ET.SubElement(box, "part", name="00", x=str(x), y=str(y))
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    def test_snapshot_tracks_only_effective_pixels_and_geometry(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "canonical.png")
            xml_path = os.path.join(root, "train.xml")
            with open(image_path, "wb") as handle:
                handle.write(b"canonical-pixels-v1")
            self._write_xml(xml_path, image_path)

            first = _build_effective_dlib_dataset_snapshot(
                {"fit": xml_path},
                os.path.join(root, "artifact-one"),
            )
            serialized = json.dumps(first, sort_keys=True)
            self.assertNotIn(root, serialized)
            self.assertEqual(first["roles"]["fit"]["sampleCount"], 1)
            self.assertTrue(
                os.path.isfile(
                    os.path.join(root, "artifact-one", "datasets", "fit.xml")
                )
            )

            # An unrelated project label must not perturb the effective-run ID.
            with open(os.path.join(root, "unused-label.json"), "w", encoding="utf-8") as handle:
                json.dump({"detectionOnly": True}, handle)
            repeated = _build_effective_dlib_dataset_snapshot(
                {"fit": xml_path},
                os.path.join(root, "artifact-two"),
            )
            self.assertEqual(repeated["revision"], first["revision"])

            with open(image_path, "wb") as handle:
                handle.write(b"canonical-pixels-v2")
            pixel_changed = _build_effective_dlib_dataset_snapshot(
                {"fit": xml_path},
                os.path.join(root, "artifact-three"),
            )
            self.assertNotEqual(pixel_changed["revision"], first["revision"])

            with open(image_path, "wb") as handle:
                handle.write(b"canonical-pixels-v1")
            self._write_xml(xml_path, image_path, x=99)
            geometry_changed = _build_effective_dlib_dataset_snapshot(
                {"fit": xml_path},
                os.path.join(root, "artifact-four"),
            )
            self.assertNotEqual(geometry_changed["revision"], first["revision"])

    def test_missing_referenced_crop_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            xml_path = os.path.join(root, "train.xml")
            self._write_xml(xml_path, os.path.join(root, "missing.png"))
            with self.assertRaisesRegex(RuntimeError, "missing canonical crop"):
                _build_effective_dlib_dataset_snapshot(
                    {"fit": xml_path},
                    os.path.join(root, "artifact"),
                )

    def test_mid_run_crop_change_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "canonical.png")
            xml_path = os.path.join(root, "train.xml")
            with open(image_path, "wb") as handle:
                handle.write(b"canonical-pixels-v1")
            self._write_xml(xml_path, image_path)
            role_paths = {"fit": xml_path}
            expected = _build_effective_dlib_dataset_snapshot(
                role_paths,
                os.path.join(root, "artifact"),
            )

            _assert_effective_dlib_dataset_unchanged(role_paths, expected)
            with open(image_path, "wb") as handle:
                handle.write(b"canonical-pixels-v2")
            with self.assertRaisesRegex(RuntimeError, "changed while training"):
                _assert_effective_dlib_dataset_unchanged(role_paths, expected)

    def test_training_protocol_revision_pins_effective_options(self):
        baseline = _build_dlib_training_protocol(
            {"tree_depth": 4, "random_seed": "42", "num_threads": 2},
            "dataset-revision",
        )
        repeated = _build_dlib_training_protocol(
            {"tree_depth": 4, "random_seed": "42", "num_threads": 2},
            "dataset-revision",
        )
        changed = _build_dlib_training_protocol(
            {"tree_depth": 5, "random_seed": "42", "num_threads": 2},
            "dataset-revision",
        )
        self.assertEqual(repeated["revision"], baseline["revision"])
        self.assertNotEqual(changed["revision"], baseline["revision"])

    def test_offline_augmentation_is_reproducible_from_recorded_seed(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "canonical.png")
            xml_path = os.path.join(root, "train.xml")
            pixels = np.zeros((512, 512, 3), dtype=np.uint8)
            pixels[:, :, 0] = np.arange(512, dtype=np.uint16)[None, :] % 256
            pixels[:, :, 1] = np.arange(512, dtype=np.uint16)[:, None] % 256
            self.assertTrue(cv2.imwrite(image_path, pixels))
            self._write_xml(xml_path, image_path)
            aug_dir = os.path.join(root, "augmented")

            def run(seed):
                output_xml = augment_training_data(
                    xml_path,
                    aug_dir,
                    aug_angles=[-5, 5],
                    add_flip=True,
                    max_augmented_copies_per_image=2,
                    photo_jitter_profile={
                        "photo_jitter_contrast_delta": 0.2,
                        "photo_jitter_brightness_delta": 20.0,
                    },
                    random_seed=seed,
                )
                image_nodes = ET.parse(output_xml).getroot().findall("./images/image")[1:]
                return [
                    (os.path.basename(node.get("file", "")), lineage.sha256_file(node.get("file")))
                    for node in image_nodes
                ]

            first = run(1729)
            repeated = run(1729)
            changed = run(1730)
            self.assertEqual(repeated, first)
            self.assertNotEqual(changed, first)

    def test_offline_rotation_keeps_landmark_on_rotated_pixels(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "canonical.png")
            xml_path = os.path.join(root, "train.xml")
            pixels = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.circle(pixels, (350, 220), 7, (255, 255, 255), thickness=-1)
            self.assertTrue(cv2.imwrite(image_path, pixels))
            self._write_xml(xml_path, image_path, x=350, y=220)

            output_xml = augment_training_data(
                xml_path,
                os.path.join(root, "augmented"),
                aug_angles=[30],
                add_flip=False,
                photo_jitter_profile={},
                random_seed=7,
            )
            augmented = ET.parse(output_xml).getroot().findall("./images/image")[1]
            part = augmented.find("./box/part")
            augmented_pixels = cv2.imread(augmented.get("file"))
            ys, xs = np.nonzero(augmented_pixels[:, :, 0] > 160)

            self.assertGreater(len(xs), 20)
            self.assertLess(abs(float(xs.mean()) - float(part.get("x"))), 1.5)
            self.assertLess(abs(float(ys.mean()) - float(part.get("y"))), 1.5)

    def test_offline_rotation_skips_copy_when_landmark_leaves_frame(self):
        with tempfile.TemporaryDirectory() as root:
            image_path = os.path.join(root, "canonical.png")
            xml_path = os.path.join(root, "train.xml")
            pixels = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.circle(pixels, (2, 2), 2, (255, 255, 255), thickness=-1)
            self.assertTrue(cv2.imwrite(image_path, pixels))
            self._write_xml(xml_path, image_path, x=2, y=2)

            output_xml = augment_training_data(
                xml_path,
                os.path.join(root, "augmented"),
                aug_angles=[45],
                add_flip=False,
                photo_jitter_profile={},
                random_seed=7,
            )
            images = ET.parse(output_xml).getroot().findall("./images/image")
            self.assertEqual(len(images), 1)

    def test_dlib_failed_run_is_finalized_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = os.path.join(root, "debug", "models", "dlib", "Fish", "run-fail")

            def fail_impl(project_root, tag, **kwargs):
                os.makedirs(run_dir, exist_ok=True)
                train_shape_model.dio.write_run_manifest(
                    run_dir,
                    model_type="dlib",
                    tag=tag,
                    project_root=project_root,
                    extra={"status": "started"},
                )
                kwargs["_run_context"].update(
                    {"run_dir": run_dir, "project_root": project_root, "tag": tag}
                )
                raise RuntimeError("synthetic dlib failure")

            with patch.object(train_shape_model, "_train_shape_model_impl", side_effect=fail_impl):
                with self.assertRaisesRegex(RuntimeError, "synthetic dlib failure"):
                    train_shape_model.train_shape_model(root, "Fish")
            manifest = train_shape_model.lineage.read_json(
                os.path.join(run_dir, "run_manifest.json")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure"]["type"], "RuntimeError")
            self.assertIn("synthetic dlib failure", manifest["failure"]["traceback"])

    def test_dlib_contract_rejects_original_source_leakage(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            os.makedirs(debug_dir)
            train_image = os.path.join(root, "train.png")
            validation_image = os.path.join(root, "validation.png")
            self.assertTrue(cv2.imwrite(train_image, np.zeros((512, 512, 3), dtype=np.uint8)))
            self.assertTrue(
                cv2.imwrite(validation_image, np.ones((512, 512, 3), dtype=np.uint8))
            )
            train_xml = os.path.join(root, "train_Fish.xml")
            validation_xml = os.path.join(root, "validation_Fish.xml")
            self._write_xml(train_xml, train_image)
            self._write_xml(validation_xml, validation_image)
            lineage.atomic_write_json(
                os.path.join(debug_dir, "split_info_Fish.json"),
                {
                    "train_file_source_ids": {train_image: "source:same"},
                    "validation_file_source_ids": {validation_image: "source:same"},
                    "test_file_source_ids": {},
                },
            )
            with self.assertRaisesRegex(RuntimeError, "overlap at the original source level"):
                train_shape_model._validate_dlib_training_contract(
                    {"train": train_xml, "validation": validation_xml},
                    debug_dir=debug_dir,
                    tag="Fish",
                    expected_parts={"00"},
                )

    def test_dlib_contract_rejects_a_universally_missing_schema_part(self):
        with tempfile.TemporaryDirectory() as root:
            debug_dir = os.path.join(root, "debug")
            os.makedirs(debug_dir)
            image_path = os.path.join(root, "train.png")
            self.assertTrue(cv2.imwrite(image_path, np.zeros((512, 512, 3), dtype=np.uint8)))
            train_xml = os.path.join(root, "train_Fish.xml")
            self._write_xml(train_xml, image_path)
            lineage.atomic_write_json(
                os.path.join(debug_dir, "split_info_Fish.json"),
                {
                    "train_file_source_ids": {image_path: "source:train"},
                    "validation_file_source_ids": {},
                    "test_file_source_ids": {},
                },
            )
            with self.assertRaisesRegex(RuntimeError, "expected.*00.*01"):
                train_shape_model._validate_dlib_training_contract(
                    {"train": train_xml},
                    debug_dir=debug_dir,
                    tag="Fish",
                    expected_parts={"00", "01"},
                )


if __name__ == "__main__":
    unittest.main()
