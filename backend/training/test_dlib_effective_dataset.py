import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from backend.bv_utils import lineage
from backend.training.train_shape_model import (
    augment_training_data,
    _assert_effective_dlib_dataset_unchanged,
    _build_dlib_training_protocol,
    _build_effective_dlib_dataset_snapshot,
)


class DlibEffectiveDatasetTests(unittest.TestCase):
    def _write_xml(self, path, image_path, *, x=11):
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
        ET.SubElement(box, "part", name="00", x=str(x), y="22")
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


if __name__ == "__main__":
    unittest.main()
