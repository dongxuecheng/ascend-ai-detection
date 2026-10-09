"""火焰识别的标签、严格置信度阈值和启用配置测试。"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analyze.fire import FireDetector
from config.config import config
from utils.obj import Box


class FireDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = FireDetector()

    def test_confidence_must_be_strictly_above_threshold(self):
        below = Box("fire", 0.5999, [10, 10, 20, 20])
        boundary = Box("fire", 0.6, [10, 10, 20, 20])
        above = Box("fire", 0.6001, [10, 10, 20, 20])
        high = Box("fire", 0.95, [30, 30, 50, 50])
        self.assertEqual(self.detector.detect([below, boundary, above, high]), [above, high])

    def test_only_fire_label_is_accepted(self):
        boxes = [Box(label, 0.99, [10, 10, 20, 20]) for label in ["flame", "person", "extinguisher"]]
        self.assertEqual(self.detector.detect(boxes), [])

    def test_empty_input_returns_no_violations(self):
        self.assertEqual(self.detector.detect([]), [])

    def test_fire_needs_no_person_area_or_extinguisher_condition(self):
        fire = Box("fire", 0.9, [10, 10, 11, 11])
        extinguisher = Box("extinguisher", 0.99, [10, 10, 20, 20])
        self.assertEqual(self.detector.detect([fire]), [fire])
        self.assertEqual(self.detector.detect([fire, extinguisher]), [fire])
        self.assertEqual(self.detector.detect([fire], device_id="camera"), [fire])
        self.assertEqual(self.detector.detect([fire], device_id="camera"), [fire])

    def test_algorithm_is_enabled_and_uses_regular_sam3(self):
        self.assertIn("16", config.ALGORITHM_CODES)
        self.assertEqual(config.ALGORITHM_DETECTORS["16"], ["FireDetector"])
        self.assertEqual(config.ALGORITHM_INTERVALS["16"], 30.0)
        self.assertEqual(config.ALGORITHM_SAM3_PROMPT["16"], ["fire"])
        self.assertFalse(config.ALGORITHM_SAM3_RETURN_MASK["16"])
        self.assertEqual(config.ALGORITHM_SAM3_URL["16"], config.SAM3_URL)
        self.assertNotIn("16", config.ALGM_PRE_YOLO_MODEL_DETECT_CLASSES)
        self.assertFalse(config.ALGORITHM_VL_CONFIG.get("16", {}).get("enabled", False))
        self.assertFalse(config.ALERT_DEDUP_CONFIG.get("16", {}).get("enabled", False))


if __name__ == "__main__":
    unittest.main()
