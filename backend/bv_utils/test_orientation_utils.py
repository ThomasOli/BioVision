import math
import unittest

import cv2
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

    def test_manual_mirror_transforms_pixels_labels_and_inverse_mapping_together(self):
        ramp = np.tile(np.arange(620, dtype=np.uint8), (420, 1))
        image = np.dstack([ramp, ramp, ramp])
        corners = build_obb(310.0, 210.0, 280.0, 110.0, 0.0)
        box = {"obbCorners": corners, "class_id": 0}
        landmarks = [{"id": 1, "x": 250.0, "y": 200.0}]
        policy = {"mode": "invariant", "obbLevelingMode": "off"}

        base_crop, base_landmarks, _base_meta = prepare_standardize_crop(
            image, box, landmarks, mirror=False, orientation_policy=policy
        )
        mirrored_crop, mirrored_landmarks, mirrored_meta = prepare_standardize_crop(
            image, box, landmarks, mirror=True, orientation_policy=policy
        )

        self.assertTrue(np.array_equal(mirrored_crop, cv2.flip(base_crop, 1)))
        self.assertAlmostEqual(
            mirrored_landmarks[0]["x"],
            (ou.STANDARD_SIZE - 1) - base_landmarks[0]["x"],
            places=5,
        )
        self.assertTrue(mirrored_meta["manual_mirror_applied"])
        restored = ou.map_to_original(
            mirrored_landmarks,
            mirrored_meta,
            image_shape=image.shape[:2],
        )
        self.assert_round_trip_close(landmarks, restored, tolerance=0.2)

    def test_landmark_ids_three_and_twelve_do_not_imply_bilateral_schema(self):
        policy = ou.infer_orientation_policy_from_template(
            [
                {"index": 3, "name": "Arbitrary point"},
                {"index": 12, "name": "Another arbitrary point"},
            ]
        )
        self.assertEqual(policy["mode"], "invariant")

    def test_explicit_directional_policy_does_not_infer_landmark_anchors(self):
        fish_template = [
            {"index": 1, "name": "Snout Tip", "category": "head"},
            {"index": 4, "name": "Upper Caudal Peduncle", "category": "caudal-fin"},
            {"index": 5, "name": "Lower Caudal Peduncle", "category": "caudal-fin"},
        ]
        explicit = ou.sanitize_orientation_policy(
            {"mode": "directional", "targetOrientation": "left"},
            fish_template,
        )
        self.assertEqual(explicit["anteriorAnchorIds"], [])
        self.assertEqual(explicit["posteriorAnchorIds"], [])
        self.assertEqual(explicit["headCategories"], [])
        self.assertEqual(explicit["tailCategories"], [])

        legacy = ou.sanitize_orientation_policy(None, fish_template)
        self.assertEqual(legacy["anteriorAnchorIds"], [1])
        self.assertEqual(legacy["posteriorAnchorIds"], [4, 5])

    def test_directional_class_routes_to_native_canonical_facing(self):
        crop = np.arange(24 * 32 * 3, dtype=np.uint8).reshape((24, 32, 3))
        native, native_metadata, native_debug = ou.apply_obb_geometry(
            crop.copy(), {}, 0, {"mode": "directional", "targetOrientation": "left"}
        )
        mirrored, mirrored_metadata, mirrored_debug = ou.apply_obb_geometry(
            crop.copy(), {}, 1, {"mode": "directional", "targetOrientation": "left"}
        )
        self.assertTrue(np.array_equal(native, crop))
        self.assertFalse(native_metadata.get("canonical_flip_applied", False))
        self.assertFalse(native_debug["flip_applied"])
        self.assertTrue(np.array_equal(mirrored, cv2.flip(crop, 1)))
        self.assertTrue(mirrored_metadata["canonical_flip_applied"])
        self.assertTrue(mirrored_debug["flip_applied"])

    def test_landmark_derived_obb_keeps_rectangular_geometry_beyond_image_edge(self):
        derived = ou.derive_obb_from_landmarks(
            [
                {"id": 1, "x": 1.0, "y": 18.0},
                {"id": 2, "x": 32.0, "y": 42.0},
                {"id": 3, "x": 58.0, "y": 64.0},
            ],
            image_shape=(80, 90),
            head_id=1,
            tail_id=3,
            mode="directional",
            pad_ratio=0.25,
            min_pad_px=8.0,
        )

        corners = np.asarray(derived["obbCorners"], dtype=np.float64)
        self.assertTrue(
            bool(np.any(corners[:, 0] < 0.0) or np.any(corners[:, 1] < 0.0)),
            "edge-crossing padding must survive for the exporter's rigid canvas repair",
        )
        edges = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(edges, axis=1)
        self.assertGreater(float(np.min(lengths)), 0.0)
        self.assertAlmostEqual(float(lengths[0]), float(lengths[2]), places=5)
        self.assertAlmostEqual(float(lengths[1]), float(lengths[3]), places=5)
        for index in range(4):
            cosine = abs(
                float(np.dot(edges[index], edges[(index + 1) % 4]))
                / float(lengths[index] * lengths[(index + 1) % 4])
            )
            self.assertLess(cosine, 1e-5)

    def test_axial_mode_is_one_class_and_never_applies_class_driven_half_turn(self):
        landmarks = [
            {"id": 1, "x": 20.0, "y": 80.0},
            {"id": 2, "x": 20.0, "y": 10.0},
        ]
        self.assertEqual(
            ou.derive_class_id_from_landmarks(landmarks, mode="axial", head_id=1, tail_id=2),
            0,
        )
        crop = np.arange(32 * 32 * 3, dtype=np.uint8).reshape((32, 32, 3))
        routed, metadata, debug = ou.apply_obb_geometry(
            crop.copy(),
            {"rotation": 90.0},
            1,
            {"mode": "axial"},
        )
        self.assertTrue(np.array_equal(routed, crop))
        self.assertFalse(metadata.get("rotated_180", False))
        self.assertTrue(debug.get("pole_invariant"))

    def test_bilateral_class_one_rotation_keeps_pixels_and_landmarks_aligned(self):
        image = np.zeros((320, 480, 3), dtype=np.uint8)
        cv2.circle(image, (105, 105), 9, (0, 255, 0), thickness=-1)
        cv2.circle(image, (375, 215), 9, (0, 0, 255), thickness=-1)
        box = {
            "left": 40,
            "top": 60,
            "width": 400,
            "height": 200,
            "obbCorners": [[40, 60], [440, 60], [440, 260], [40, 260]],
            "class_id": 1,
        }
        original_landmarks = [
            {"id": 3, "x": 105.0, "y": 105.0},
            {"id": 12, "x": 375.0, "y": 215.0},
        ]
        crop, landmarks, metadata = prepare_standardize_crop(
            image,
            box,
            original_landmarks,
            orientation_policy={
                "mode": "bilateral",
                "bilateralClassAxis": "vertical_obb",
                "obbLevelingMode": "on",
            },
        )

        self.assertTrue(metadata["rotated_180"])
        self.assertFalse(metadata.get("canonical_flip_applied", False))
        marker_masks = {
            3: (crop[:, :, 1] > 180) & (crop[:, :, 1] > crop[:, :, 2] * 1.5),
            12: (crop[:, :, 2] > 180) & (crop[:, :, 2] > crop[:, :, 1] * 1.5),
        }
        by_id = {int(landmark["id"]): landmark for landmark in landmarks}
        for landmark_id, mask in marker_masks.items():
            ys, xs = np.nonzero(mask)
            self.assertGreater(len(xs), 20)
            self.assertLess(abs(float(xs.mean()) - float(by_id[landmark_id]["x"])), 3.0)
            self.assertLess(abs(float(ys.mean()) - float(by_id[landmark_id]["y"])), 3.0)

        restored = ou.map_to_original(landmarks, metadata, image_shape=image.shape[:2])
        self.assert_round_trip_close(original_landmarks, restored, tolerance=1.1)

    def test_bilateral_geometry_rejects_missing_legacy_axis_contract(self):
        crop = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "bilateralClassAxis"):
            ou.apply_obb_geometry(crop, {}, 1, {"mode": "bilateral"})

    def test_rotated_obb_leveling_off_round_trips_without_phantom_rotation(self):
        image = np.zeros((280, 420, 3), dtype=np.uint8)
        corners = build_obb(210.0, 140.0, 240.0, 100.0, 27.0)
        landmarks = [
            {"id": 1, "x": 175.0, "y": 120.0},
            {"id": 2, "x": 245.0, "y": 160.0},
        ]

        _crop, metadata = ou.extract_obb_crop(
            image,
            corners,
            pad_ratio=0.05,
            apply_leveling=False,
        )
        standardized = ou.remap_landmarks_to_standard(landmarks, metadata)
        restored = ou.map_to_original(
            standardized,
            metadata,
            image_shape=image.shape[:2],
        )

        self.assertFalse(metadata["obb_deskewed"])
        self.assertFalse(metadata["leveling_applied"])
        self.assertAlmostEqual(metadata["rotation"], 0.0)
        self.assertAlmostEqual(metadata["source_obb_rotation"], 27.0, places=4)
        self.assertIsNone(metadata["affine_M"])
        self.assertIsNotNone(metadata["source_affine_M"])
        self.assert_round_trip_close(landmarks, restored, tolerance=1.1)

    def test_uncertainty_metadata_survives_mapping_and_flip_disagreement_is_measured(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        _crop, meta = ou.base_standardize(image, [10, 10, 150, 110], pad_ratio=0.0)
        mapped = ou.map_to_original(
            [{"id": 1, "x": 256.0, "y": 256.0, "confidence": 0.8, "heatmap_entropy": 0.2}],
            meta,
            image_shape=image.shape[:2],
        )
        self.assertAlmostEqual(mapped[0]["confidence"], 0.8)
        self.assertAlmostEqual(mapped[0]["heatmap_entropy"], 0.2)

        asymmetric = np.zeros((512, 512, 3), dtype=np.uint8)
        asymmetric[:, :32] = 255

        def predict_fn(crop):
            left_bright = float(crop[:, :32].mean()) > float(crop[:, -32:].mean())
            x = 100.0 if left_bright else 300.0
            return [{"id": 1, "x": x, "y": 200.0}]

        _landmarks, _flipped, debug = ou.select_orientation(
            asymmetric,
            predict_fn,
            landmark_template={1: {"x": 0.5, "y": 0.5}},
        )
        self.assertIsNotNone(debug.get("model_disagreement"))
        self.assertGreater(debug["model_disagreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
