"""工服检测的分辨率阈值、反光条标签和 mask 降级回归测试。"""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AIDETECTION_LOG_DISABLE_FILE", "1")

from analyze.vest import VestDetector
from utils.obj import Box


class VestDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = VestDetector(enable_backlight_filter=False)
        self.person = Box("person", 0.95, [100, 100, 200, 300], source="SAM3")
        self.clothes = Box("upper garment", 0.9, [110, 120, 190, 200])

    def detect(self, boxes, width=1280, height=720):
        return self.detector.detect(boxes, image_width=width, image_height=height)

    def mask(self, bounds):
        mask = np.zeros((720, 1280), dtype=np.uint8)
        x1, y1, x2, y2 = bounds
        mask[y1:y2, x1:x2] = 1
        return mask

    def test_low_resolution_accepts_person_and_clothes_at_area_limits(self):
        person = Box("person", 0.95, [100, 100, 150, 180], source="SAM3")
        clothes = Box("upper garment", 0.9, [105, 110, 145, 160])
        self.assertEqual(self.detect([person, clothes]), [person])
        self.assertEqual(self.detector.person_min_area, 10000)
        self.assertEqual(self.detector.clothes_min_area, 5000)

    def test_low_resolution_rejects_person_below_area_limit(self):
        person = Box("person", 0.95, [100, 100, 149, 180], source="SAM3")
        clothes = Box("upper garment", 0.9, [105, 110, 145, 160])
        self.assertEqual(self.detect([person, clothes]), [])

    def test_low_resolution_thresholds_do_not_leak_into_large_frames(self):
        clothes = Box("upper garment", 0.9, [110, 130, 160, 180])
        for width, height, expected in [
            (1280, 720, [self.person]), (1920, 1080, []),
            (1280, 720, [self.person]), (1920, 1080, []),
        ]:
            with self.subTest(width=width, height=height):
                self.assertEqual(self.detect([self.person, clothes], width, height), expected)
        self.assertEqual(self.detector.person_min_area, 10000)
        self.assertEqual(self.detector.clothes_min_area, 5000)

    def test_high_resolution_keeps_person_area_limit(self):
        person = Box("person", 0.95, [100, 100, 180, 200], source="SAM3")
        clothes = Box("upper garment", 0.9, [105, 105, 175, 185])
        self.assertEqual(self.detect([person, clothes], 1920, 1080), [])

    def test_lower_custom_thresholds_are_preserved(self):
        self.detector = VestDetector(person_min_area=1000, clothes_min_area=500)
        person = Box("person", 0.95, [100, 100, 130, 150], source="SAM3")
        clothes = Box("upper garment", 0.9, [105, 110, 125, 140])
        self.assertEqual(self.detect([person, clothes]), [person])

    def test_ascii_reflective_labels_exempt_person(self):
        for label in ["high-visibility stripe", "high-vis stripe", "reflective stripe"]:
            with self.subTest(label=label):
                strip = Box(label, 0.9, [120, 150, 170, 160])
                self.assertEqual(self.detect([self.person, self.clothes, strip]), [])

    def test_missing_mask_on_either_target_falls_back_to_bbox(self):
        for person_has_mask, strip_has_mask in [(True, False), (False, True), (False, False)]:
            with self.subTest(person_mask=person_has_mask, strip_mask=strip_has_mask):
                person = Box(
                    "person", 0.95, self.person.box,
                    mask_array=self.mask(self.person.box) if person_has_mask else None,
                )
                strip = Box(
                    "reflective stripe", 0.9, [120, 150, 170, 160],
                    mask_array=self.mask([120, 150, 170, 160]) if strip_has_mask else None,
                )
                self.assertEqual(self.detect([person, self.clothes, strip]), [])

    def test_both_masks_use_pixel_overlap(self):
        person = Box("person", 0.95, self.person.box, mask_array=self.mask(self.person.box))
        strip = Box(
            "reflective stripe", 0.9, [120, 150, 170, 160],
            mask_array=self.mask([120, 150, 170, 160]),
        )
        self.assertEqual(self.detect([person, self.clothes, strip]), [])

    def test_disjoint_masks_do_not_exempt_even_when_boxes_overlap(self):
        person = Box("person", 0.95, self.person.box, mask_array=self.mask([100, 100, 200, 140]))
        strip = Box(
            "reflective stripe", 0.9, [120, 150, 170, 160],
            mask_array=self.mask([120, 150, 170, 160]),
        )
        self.assertEqual(self.detect([person, self.clothes, strip]), [person])

    def test_empty_mask_does_not_fall_back_to_bbox(self):
        person = Box("person", 0.95, self.person.box, mask_array=self.mask(self.person.box))
        strip = Box("reflective stripe", 0.9, [120, 150, 170, 160], mask_array=self.mask([0, 0, 0, 0]))
        self.assertEqual(self.detect([person, self.clothes, strip]), [person])

    def test_low_confidence_or_unrelated_strip_does_not_exempt(self):
        for strip in [Box("reflective stripe", 0.3, [120, 150, 170, 160]),
                      Box("reflective stripe", 0.9, [400, 150, 450, 160])]:
            with self.subTest(strip=strip):
                self.assertEqual(self.detect([self.person, self.clothes, strip]), [self.person])


if __name__ == "__main__":
    unittest.main()
