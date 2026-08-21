import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.detection.detect_specimen import (
    _parse_obb_boxes,
    canonicalize_detector_obb_corners,
    detect_multiple_with_yolo,
    detect_with_yolo,
)


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


class CanonicalizeDetectorObbCornersTests(unittest.TestCase):
    def assertSameCorners(self, actual, expected, places=4):
        self.assertEqual(len(actual), len(expected))
        for actual_point, expected_point in zip(actual, expected):
            self.assertAlmostEqual(actual_point[0], expected_point[0], places=places)
            self.assertAlmostEqual(actual_point[1], expected_point[1], places=places)

    def assertCanonical(self, corners):
        self.assertLessEqual(corners[0][1], corners[2][1])
        self.assertLessEqual(corners[1][1], corners[2][1])
        self.assertLessEqual(corners[0][0], corners[1][0])
        self.assertLessEqual(corners[3][0], corners[2][0])

    def test_canonicalizes_corner_rotations_and_winding(self):
        expected = build_obb(200.0, 120.0, 160.0, 42.0, 28.0)
        xywhr = [200.0, 120.0, 160.0, 42.0, math.radians(28.0)]
        variants = []
        for shift in range(4):
            rotated = expected[shift:] + expected[:shift]
            variants.append(rotated)
            variants.append(list(reversed(rotated)))

        for variant in variants:
            actual = canonicalize_detector_obb_corners(variant, xywhr=xywhr)
            self.assertSameCorners(actual, expected)
            self.assertCanonical(actual)

    def test_handles_near_vertical_obb(self):
        expected = build_obb(90.0, 220.0, 46.0, 180.0, 83.0)
        xywhr = [90.0, 220.0, 46.0, 180.0, math.radians(83.0)]
        variant = [expected[2], expected[3], expected[0], expected[1]]
        actual = canonicalize_detector_obb_corners(variant, xywhr=xywhr)
        self.assertSameCorners(actual, expected)
        self.assertCanonical(actual)

    def test_raw_corner_fallback_still_returns_stable_quad(self):
        expected = build_obb(240.0, 75.0, 90.0, 28.0, 14.0)
        variant = [expected[1], expected[2], expected[3], expected[0]]
        actual = canonicalize_detector_obb_corners(variant)
        self.assertEqual(len(actual), 4)
        self.assertCanonical(actual)


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value, dtype=np.float32)


class _FakeObb:
    def __init__(self, xywhr, corners, confidences, classes):
        self.xywhr = [_FakeTensor(value) for value in xywhr]
        self.xyxyxyxy = [_FakeTensor(value) for value in corners]
        self.conf = confidences
        self.cls = classes

    def __len__(self):
        return len(self.xywhr)


class _FakeResult:
    def __init__(self, xywhr, corners):
        self.obb = _FakeObb(xywhr, corners, [0.91, 0.89], [0, 0])
        self.orig_shape = (300, 300)
        self.names = {0: "specimen"}


class ParseDetectorObbTests(unittest.TestCase):
    def test_crossing_slender_obbs_are_not_suppressed_by_aabb_envelopes(self):
        first_xywhr = [150.0, 150.0, 220.0, 18.0, math.radians(45.0)]
        second_xywhr = [150.0, 150.0, 220.0, 18.0, math.radians(-45.0)]
        first = build_obb(*first_xywhr[:4], 45.0)
        second = build_obb(*second_xywhr[:4], -45.0)
        result = _FakeResult([first_xywhr, second_xywhr], [first, second])

        parsed = _parse_obb_boxes(result, margin=0, max_specimens=20)

        self.assertEqual(len(parsed), 2)
        self.assertEqual([round(item["confidence"], 2) for item in parsed], [0.91, 0.89])


class _PredictingYolo:
    instances = []
    result = None

    def __init__(self, model_path):
        self.model_path = model_path
        self.predict_kwargs = None
        self.__class__.instances.append(self)

    def predict(self, _image_path, **kwargs):
        self.predict_kwargs = kwargs
        return [self.__class__.result]


class ObbPredictConfigurationTests(unittest.TestCase):
    def test_all_detector_predict_paths_use_agnostic_obb_nms_and_artifact_config(self):
        first_xywhr = [150.0, 150.0, 220.0, 18.0, math.radians(45.0)]
        second_xywhr = [150.0, 150.0, 220.0, 18.0, math.radians(-45.0)]
        _PredictingYolo.result = _FakeResult(
            [first_xywhr, second_xywhr],
            [build_obb(*first_xywhr[:4], 45.0), build_obb(*second_xywhr[:4], -45.0)],
        )
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _PredictingYolo
        _PredictingYolo.instances.clear()

        with tempfile.TemporaryDirectory() as artifact_dir:
            model_path = Path(artifact_dir, "model.pt")
            model_path.write_bytes(b"mock")
            Path(artifact_dir, "obb_config.json").write_text(
                '{"confidence_threshold": 0.15, "nms_iou": 0.57}', encoding="utf-8"
            )
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                single = detect_with_yolo("image.png", os.fspath(model_path))
                multiple = detect_multiple_with_yolo("image.png", os.fspath(model_path))

        self.assertIsNotNone(single)
        self.assertEqual(len(multiple), 2)
        self.assertEqual(len(_PredictingYolo.instances), 2)
        for instance in _PredictingYolo.instances:
            self.assertTrue(instance.predict_kwargs["agnostic_nms"])
            self.assertAlmostEqual(instance.predict_kwargs["conf"], 0.15)
            self.assertAlmostEqual(instance.predict_kwargs["iou"], 0.57)

    def test_explicit_balanced_preset_retains_its_confidence_floor(self):
        xywhr = [100.0, 100.0, 80.0, 20.0, 0.0]
        _PredictingYolo.result = _FakeResult([xywhr], [build_obb(100, 100, 80, 20, 0)])
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _PredictingYolo
        _PredictingYolo.instances.clear()

        with tempfile.TemporaryDirectory() as artifact_dir:
            model_path = Path(artifact_dir, "model.pt")
            model_path.write_bytes(b"mock")
            Path(artifact_dir, "obb_config.json").write_text(
                '{"confidence_threshold": 0.15, "nms_iou": 0.57}', encoding="utf-8"
            )
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                detect_with_yolo(
                    "image.png",
                    os.fspath(model_path),
                    detection_preset="balanced",
                )

        self.assertAlmostEqual(_PredictingYolo.instances[0].predict_kwargs["conf"], 0.30)

if __name__ == "__main__":
    unittest.main()
