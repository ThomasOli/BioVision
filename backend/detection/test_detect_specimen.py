import math
import unittest

from backend.detection.detect_specimen import canonicalize_detector_obb_corners


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


if __name__ == "__main__":
    unittest.main()
