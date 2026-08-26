import math
import unittest

import numpy as np

from backend.data import prepare_dataset
from backend.inference import predict


class PredictorObbOrientationContractTests(unittest.TestCase):
    def _box(self, **metadata):
        return {
            "left": 10,
            "top": 20,
            "right": 110,
            "bottom": 80,
            "width": 100,
            "height": 60,
            "obbCorners": [[10, 20], [110, 20], [110, 80], [10, 80]],
            **metadata,
        }

    def _normalized(self, **metadata):
        return predict._ensure_obb_box_geometry(self._box(**metadata))

    def test_directional_hint_without_class_drives_crop_class(self):
        box = self._normalized(
            orientation_hint={
                "orientation": "right",
                "confidence": 1.0,
                "source": "user_draw_default",
            }
        )

        class_id, hint = predict._resolve_box_geometry_orientation(
            box,
            {"mode": "directional", "targetOrientation": "left"},
        )

        self.assertFalse(box["_class_id_was_provided"])
        self.assertEqual(class_id, 1)
        self.assertEqual(hint, "right")

    def test_directional_class_without_hint_drives_predictor_lock(self):
        box = self._normalized(class_id=1)

        class_id, hint = predict._resolve_box_geometry_orientation(
            box,
            {"mode": "directional", "targetOrientation": "left"},
        )

        self.assertTrue(box["_class_id_was_provided"])
        self.assertEqual(class_id, 1)
        self.assertEqual(hint, "right")

    def test_missing_legacy_direction_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "saved detector class or direction arrow"):
            predict._resolve_box_geometry_orientation(
                self._normalized(),
                {"mode": "directional", "targetOrientation": "left"},
            )

    def test_conflicting_vector_metadata_fails_closed(self):
        box = self._normalized(
            class_id=0,
            orientation_hint={
                "orientation": "right",
                "confidence": 1.0,
                "source": "user_review",
            },
        )

        with self.assertRaisesRegex(ValueError, "conflicts"):
            predict._resolve_box_geometry_orientation(
                box,
                {"mode": "directional", "targetOrientation": "left"},
            )

    def test_bilateral_and_one_class_modes_keep_their_native_contracts(self):
        bilateral_class, bilateral_hint = predict._resolve_box_geometry_orientation(
            self._normalized(class_id=1),
            {"mode": "bilateral", "bilateralClassAxis": "vertical_obb"},
        )
        self.assertEqual((bilateral_class, bilateral_hint), (1, "down"))

        for mode in ("axial", "invariant"):
            with self.subTest(mode=mode):
                class_id, hint = predict._resolve_box_geometry_orientation(
                    self._normalized(class_id=1),
                    {"mode": mode},
                )
                self.assertEqual(class_id, 0)
                self.assertIsNone(hint)

    def test_training_crop_and_dlib_cnn_inference_round_trip_all_modes(self):
        angle = math.radians(23.0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        cx, cy, width, height = 160.0, 120.0, 210.0, 100.0
        corners = []
        for local_x, local_y in (
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
        ):
            corners.append(
                [
                    cx + cos_a * local_x - sin_a * local_y,
                    cy + sin_a * local_x + cos_a * local_y,
                ]
            )
        image = np.zeros((260, 340, 3), dtype=np.uint8)
        source_landmarks = [
            {"id": 3, "x": 125.0, "y": 100.0},
            {"id": 12, "x": 195.0, "y": 140.0},
        ]
        cases = (
            (
                "directional",
                {"mode": "directional", "targetOrientation": "left", "obbLevelingMode": "on"},
                1,
                "right",
            ),
            (
                "bilateral",
                {"mode": "bilateral", "bilateralClassAxis": "vertical_obb", "obbLevelingMode": "on"},
                1,
                "down",
            ),
            ("axial", {"mode": "axial", "obbLevelingMode": "on"}, 0, None),
            ("invariant", {"mode": "invariant", "obbLevelingMode": "on"}, 0, None),
        )

        for mode, policy, class_id, orientation in cases:
            with self.subTest(mode=mode):
                box = {
                    **self._box(),
                    "left": min(point[0] for point in corners),
                    "top": min(point[1] for point in corners),
                    "right": max(point[0] for point in corners),
                    "bottom": max(point[1] for point in corners),
                    "width": max(point[0] for point in corners) - min(point[0] for point in corners),
                    "height": max(point[1] for point in corners) - min(point[1] for point in corners),
                    "obbCorners": corners,
                    "class_id": class_id,
                }
                if orientation is not None:
                    box["orientation_hint"] = {
                        "orientation": orientation,
                        "confidence": 1.0,
                        "source": "obb_class_id",
                    }
                _crop, standardized, _meta = prepare_dataset.standardize_crop(
                    image,
                    box,
                    source_landmarks,
                    orientation_policy=policy,
                )
                normalized_box = predict._ensure_obb_box_geometry(
                    box,
                    image_shape=image.shape[:2],
                )

                predictor_outputs = {
                    "dlib": lambda _crop: [dict(landmark) for landmark in standardized],
                    "cnn": lambda _crop: predict._cnn_landmarks_from_coords(
                        np.asarray(
                            [
                                coordinate / float(predict.STANDARD_SIZE - 1)
                                for landmark in standardized
                                for coordinate in (landmark["x"], landmark["y"])
                            ],
                            dtype=np.float32,
                        ),
                        [landmark["id"] for landmark in standardized],
                    ),
                }
                for engine, predict_fn in predictor_outputs.items():
                    with self.subTest(engine=engine):
                        result = predict._run_obb_inference_on_box(
                            img_original=image,
                            box=normalized_box,
                            orig_h=image.shape[0],
                            orig_w=image.shape[1],
                            detector_scale=1.0,
                            detector_w=image.shape[1],
                            detector_h=image.shape[0],
                            orientation_policy=policy,
                            predict_fn=predict_fn,
                            target_orientation=policy.get("targetOrientation"),
                            landmark_template={},
                            head_landmark_id=None,
                            tail_landmark_id=None,
                        )
                        by_id = {landmark["id"]: landmark for landmark in result["landmarks"]}
                        for expected in source_landmarks:
                            actual = by_id[expected["id"]]
                            self.assertLessEqual(abs(actual["x"] - expected["x"]), 1.0)
                            self.assertLessEqual(abs(actual["y"] - expected["y"]), 1.0)


if __name__ == "__main__":
    unittest.main()
