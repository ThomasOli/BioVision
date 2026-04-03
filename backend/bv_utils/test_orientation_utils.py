import math
import unittest

import numpy as np

from backend.bv_utils import orientation_utils as ou
from backend.data.prepare_dataset import standardize_crop as prepare_standardize_crop


def build_obb(cx, cy, width, height, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    half_w = width / 2.0
    half_h = height / 2.0
    return [
        [cx + cos_a * (-half_w) - sin_a * (-half_h), cy + sin_a * (-half_w) + cos_a * (-half_h)],
        [cx + cos_a * (half_w) - sin_a * (-half_h), cy + sin_a * (half_w) + cos_a * (-half_h)],
        [cx + cos_a * (half_w) - sin_a * (half_h), cy + sin_a * (half_w) + cos_a * (half_h)],
        [cx + cos_a * (-half_w) - sin_a * (half_h), cy + sin_a * (-half_w) + cos_a * (half_h)],
    ]


class OrientationTransformTests(unittest.TestCase):
    def assert_round_trip_close(self, expected, actual, tolerance=0.75):
        self.assertEqual(len(expected), len(actual))
        for exp, got in zip(expected, actual):
            self.assertEqual(int(exp["id"]), int(got["id"]))
            self.assertLessEqual(abs(float(exp["x"]) - float(got["x"])), tolerance)
            self.assertLessEqual(abs(float(exp["y"]) - float(got["y"])), tolerance)

    def test_base_standardize_round_trip_uses_standardized_padding(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        _crop, meta = ou.base_standardize(image, [40, 60, 250, 180], pad_ratio=0.15)
        self.assertEqual(meta.get("padding_coordinate_space"), "standardized")

        landmarks = [
            {"id": 1, "x": 55.0, "y": 80.0},
            {"id": 2, "x": 180.0, "y": 130.0},
            {"id": 3, "x": 230.0, "y": 170.0},
        ]
        standardized = ou.remap_landmarks_to_standard(landmarks, meta)
        restored = ou.map_to_original(standardized, meta, image_shape=image.shape[:2])
        self.assert_round_trip_close(landmarks, restored)

    def test_legacy_non_obb_padding_metadata_still_round_trips(self):
        image = np.zeros((180, 260, 3), dtype=np.uint8)
        _crop, meta = ou.base_standardize(image, [30, 20, 180, 140], pad_ratio=0.10)
        legacy_meta = dict(meta)
        legacy_meta.pop("padding_coordinate_space", None)
        legacy_meta["obb_deskewed"] = False
        legacy_meta["pad_left"] = float(meta["pad_left"]) / float(meta["scale_x"])
        legacy_meta["pad_top"] = float(meta["pad_top"]) / float(meta["scale_y"])
        legacy_meta["pad_right"] = float(meta["pad_right"]) / float(meta["scale_x"])
        legacy_meta["pad_bottom"] = float(meta["pad_bottom"]) / float(meta["scale_y"])

        landmarks = [
            {"id": 1, "x": 45.0, "y": 35.0},
            {"id": 2, "x": 120.0, "y": 110.0},
            {"id": 3, "x": 175.0, "y": 135.0},
        ]
        standardized = ou.remap_landmarks_to_standard(landmarks, legacy_meta)
        restored = ou.map_to_original(standardized, legacy_meta, image_shape=image.shape[:2])
        self.assert_round_trip_close(landmarks, restored)

    def test_obb_standardize_round_trip_for_current_and_legacy_metadata(self):
        image = np.zeros((420, 620, 3), dtype=np.uint8)
        obb_corners = build_obb(310.0, 210.0, 280.0, 110.0, 23.0)
        xs = [point[0] for point in obb_corners]
        ys = [point[1] for point in obb_corners]
        box = {
            "left": int(min(xs)),
            "top": int(min(ys)),
            "width": int(max(xs) - min(xs)),
            "height": int(max(ys) - min(ys)),
            "obbCorners": obb_corners,
            "class_id": 0,
        }
        landmarks = [
            {"id": 1, "x": 235.0, "y": 180.0},
            {"id": 2, "x": 310.0, "y": 205.0},
            {"id": 3, "x": 385.0, "y": 235.0},
        ]
        _crop, standardized_landmarks, meta = prepare_standardize_crop(
            image,
            box,
            landmarks,
            orientation_policy={"mode": "directional", "targetOrientation": "left", "obbLevelingMode": "on"},
        )
        restored = ou.map_to_original(standardized_landmarks, meta, image_shape=image.shape[:2])
        self.assert_round_trip_close(landmarks, restored, tolerance=1.1)

        legacy_meta = dict(meta)
        legacy_meta.pop("padding_coordinate_space", None)
        restored_legacy = ou.map_to_original(standardized_landmarks, legacy_meta, image_shape=image.shape[:2])
        self.assert_round_trip_close(landmarks, restored_legacy, tolerance=1.1)


if __name__ == "__main__":
    unittest.main()
