import contextlib
import io
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.data import prepare_dataset
from backend.training import train_shape_model


class OrientationModePipelineTests(unittest.TestCase):
    """Keep every explicit schema mode compatible with preparation and dlib training."""

    MODES = ("directional", "bilateral", "axial", "invariant")
    LANDMARK_IDS = (3, 12)

    @staticmethod
    def _policy(mode):
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
                }
            )
        return policy

    def _write_session(self, root, mode, *, configured=True, policy=None):
        payload = {
            "orientationPolicyConfigured": configured,
            "orientationPolicy": self._policy(mode) if policy is None else policy,
            # IDs 3 and 12 used to activate a hidden bilateral heuristic. Their
            # deliberately neutral semantics make any fallback visible here.
            "landmarkTemplate": [
                {"index": 3, "name": "arbitrary point a", "category": "marker", "required": True},
                {"index": 12, "name": "arbitrary point b", "category": "marker", "required": True},
            ],
        }
        Path(root, "session.json").write_text(json.dumps(payload), encoding="utf-8")

    def _write_samples(self, root):
        images_dir = Path(root, "images")
        labels_dir = Path(root, "labels")
        images_dir.mkdir()
        labels_dir.mkdir()
        for index, intensity in enumerate((45, 165)):
            image = np.full((96, 128, 3), intensity, dtype=np.uint8)
            cv2.rectangle(image, (12, 12), (116, 84), (255 - intensity, 80, 180), 2)
            cv2.circle(image, (30 + index, 45), 4, (0, 255, 0), -1)
            cv2.circle(image, (98 - index, 53), 4, (0, 0, 255), -1)
            image_name = f"sample_{index}.png"
            self.assertTrue(cv2.imwrite(str(images_dir / image_name), image))
            label = {
                "imageFilename": image_name,
                "boxes": [
                    {
                        "left": 12,
                        "top": 12,
                        "width": 104,
                        "height": 72,
                        "obbCorners": [[12, 12], [116, 12], [116, 84], [12, 84]],
                        "class_id": index,
                        "landmarks": [
                            {"id": 3, "x": 30 + index, "y": 45, "isSkipped": False},
                            {"id": 12, "x": 98 - index, "y": 53, "isSkipped": False},
                        ],
                    }
                ],
            }
            (labels_dir / f"sample_{index}.json").write_text(
                json.dumps(label), encoding="utf-8"
            )

    @staticmethod
    def _fake_error_details(xml_path, _predictor_path):
        return [
            {
                "image": xml_path,
                "filename": os.path.basename(xml_path),
                "mean_error": 0.1,
                "median_error": 0.1,
                "per_landmark_error": [0.1, 0.1],
            }
        ]

    def _run_mode(self, mode):
        with tempfile.TemporaryDirectory() as root:
            tag = f"orientation_{mode}"
            run_id = f"run_{mode}"
            self._write_session(root, mode)
            self._write_samples(root)

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                prepare_dataset.json_to_dlib_xml(root, tag, test_split=0.5, seed=17)

            mapping_path = Path(root, "debug", f"id_mapping_{tag}.json")
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            training_config = mapping["training_config"]
            self.assertEqual(mapping["original_ids"], list(self.LANDMARK_IDS))
            self.assertEqual(training_config["orientation_mode"], mode)
            self.assertEqual(training_config["orientation_policy"]["mode"], mode)
            self.assertEqual(
                training_config["canonical_training_enabled"], mode != "invariant"
            )
            self.assertEqual(
                training_config["target_orientation"], "left" if mode == "directional" else None
            )

            for cohort in ("train", "test"):
                xml_path = Path(root, "xml", f"{cohort}_{tag}.xml")
                images = ET.parse(xml_path).getroot().findall("./images/image")
                self.assertTrue(images, f"{mode} produced an empty {cohort} cohort")
                for image in images:
                    part_names = [part.attrib["name"] for part in image.findall("./box/part")]
                    self.assertEqual(part_names, ["00", "01"])

            def fake_train(_xml_path, predictor_path, _options):
                Path(predictor_path).write_bytes(f"mock:{mode}".encode("ascii"))

            with (
                patch.object(train_shape_model.dio, "_run_id", return_value=run_id),
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
                patch.object(
                    train_shape_model.dlib,
                    "train_shape_predictor",
                    side_effect=fake_train,
                ) as train_mock,
                patch.object(train_shape_model.dlib, "test_shape_predictor", return_value=0.1),
                patch.object(
                    train_shape_model,
                    "_compute_dlib_per_image_errors",
                    return_value=[0.1],
                ),
                patch.object(
                    train_shape_model,
                    "_compute_dlib_per_image_error_details",
                    side_effect=self._fake_error_details,
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = train_shape_model.train_shape_model(
                    root,
                    tag,
                    aug_angles=[],
                    aug_flip=False,
                )

            train_mock.assert_called_once()
            self.assertEqual(result["model_id"], f"dlib:{run_id}")
            # Two source images can form train+test but not an independent
            # validation cohort. The artifact is usable, but automatic
            # activation must wait for measured validation or a recorded manual
            # override.
            self.assertEqual(result["registry"]["status"], "candidate")
            self.assertEqual(
                result["registry"]["promotion"]["reason"],
                "first_model_missing_locked_cohort_metric",
            )
            self.assertTrue(Path(result["model_path"]).is_file())
            artifact_mapping_path = Path(result["id_mapping_path"])
            self.assertEqual(
                artifact_mapping_path,
                Path(root, "models", "runs", "dlib", run_id, "id_mapping.json"),
            )
            self.assertEqual(
                json.loads(artifact_mapping_path.read_text(encoding="utf-8")),
                mapping,
            )
            artifact_manifest = json.loads(
                Path(root, "models", "runs", "dlib", run_id, "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                artifact_manifest["promotionPolicy"],
                train_shape_model._dlib_promotion_policy(),
            )
            self.assertEqual(
                result["registry"]["promotion"]["improvementPolicy"],
                train_shape_model._dlib_promotion_policy(),
            )
            mapping_sha = train_shape_model.lineage.sha256_file(str(artifact_mapping_path))
            self.assertEqual(
                artifact_manifest["sidecars"]["idMapping"]["sha256"],
                mapping_sha,
            )
            self.assertEqual(
                result["registry"]["sidecars"]["idMapping"]["sha256"],
                mapping_sha,
            )
            for sidecar_name in ("effectiveDataset", "trainingParameters"):
                descriptor = artifact_manifest["sidecars"][sidecar_name]
                self.assertTrue(Path(descriptor["path"]).is_file())
                self.assertEqual(
                    train_shape_model.lineage.sha256_file(descriptor["path"]),
                    descriptor["sha256"],
                )
            params = json.loads(
                Path(root, "debug", f"training_params_{tag}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(params["orientation_mode"], mode)
            self.assertEqual(params["aug_angles"], [])
            self.assertFalse(params["aug_flip"])

    def test_all_explicit_orientation_modes_prepare_and_train(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                self._run_mode(mode)

    def test_ids_three_and_twelve_cannot_bypass_explicit_policy_gate(self):
        for configured, policy in (
            (False, {"mode": "invariant"}),
            (True, None),
            (True, {"mode": "auto"}),
        ):
            with self.subTest(configured=configured, policy=policy):
                with tempfile.TemporaryDirectory() as root:
                    # Passing policy={} explicitly avoids the helper's normal
                    # policy construction when testing a missing policy.
                    self._write_session(
                        root,
                        "invariant",
                        configured=configured,
                        policy={} if policy is None else policy,
                    )
                    with self.assertRaisesRegex(RuntimeError, "requires an explicit"):
                        prepare_dataset.json_to_dlib_xml(root, "blocked")

    def test_default_dlib_angles_match_safe_schema_profile_for_every_mode(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                expected = train_shape_model.ou.get_schema_augmentation_profile(
                    mode,
                    engine="dlib",
                )["angles"]
                actual = train_shape_model._resolve_dlib_aug_angles(mode, None)
                self.assertEqual(actual, expected)
                self.assertTrue(actual)
                self.assertLessEqual(max(abs(float(angle)) for angle in actual), 6.0)

        self.assertEqual(
            train_shape_model._resolve_dlib_aug_angles("bilateral", [-30, 30]),
            [-30, 30],
        )

    def test_explicit_optional_landmark_is_not_added_to_fixed_model_contract(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "session.json").write_text(
                json.dumps(
                    {
                        "orientationPolicyConfigured": True,
                        "orientationPolicy": {"mode": "invariant", "obbLevelingMode": "on"},
                        "landmarkTemplate": [
                            {"index": 1, "name": "required", "required": True},
                            {"index": 2, "name": "optional", "optional": True},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            images_dir = Path(root, "images")
            labels_dir = Path(root, "labels")
            images_dir.mkdir()
            labels_dir.mkdir()
            for index in range(2):
                image_name = f"optional_{index}.png"
                image = np.full((80, 100, 3), 60 + index * 80, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(images_dir / image_name), image))
                landmarks = [{"id": 1, "x": 30, "y": 40, "isSkipped": False}]
                if index == 1:
                    landmarks.append({"id": 2, "x": 70, "y": 45, "isSkipped": False})
                (labels_dir / f"optional_{index}.json").write_text(
                    json.dumps(
                        {
                            "imageFilename": image_name,
                            "boxes": [
                                {
                                    "left": 10,
                                    "top": 10,
                                    "width": 80,
                                    "height": 60,
                                    "obbCorners": [[10, 10], [90, 10], [90, 70], [10, 70]],
                                    "class_id": 0,
                                    "landmarks": landmarks,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                prepare_dataset.json_to_dlib_xml(root, "optional", test_split=0.5, seed=42)
            mapping = json.loads(
                Path(root, "debug", "id_mapping_optional.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mapping["original_ids"], [1])
            for cohort in ("train", "test"):
                parts = ET.parse(Path(root, "xml", f"{cohort}_optional.xml")).getroot().findall(
                    "./images/image/box/part"
                )
                self.assertTrue(parts)
                self.assertEqual({part.attrib["name"] for part in parts}, {"00"})


if __name__ == "__main__":
    unittest.main()
