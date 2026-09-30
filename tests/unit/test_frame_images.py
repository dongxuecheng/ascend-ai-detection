"""NV12 视图、格式回退和单帧转换缓存测试。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from stream import frame as module
from stream.remote_capture import PIXEL_BGR, PIXEL_NV12, PIXEL_I420, PIXEL_YUYV422
from utils.image_formats import bgr_to_nv12, nv12_image_shape


class FrameImagesTests(unittest.TestCase):
    def test_native_nv12_preserves_bytes_and_lazy_bgr_is_cached(self):
        raw = np.full((6, 8), 128, np.uint8)
        images = module.FrameImages(raw, PIXEL_NV12, {"w": 8, "h": 4})
        self.assertIs(images.nv12(), raw)
        self.assertEqual((images.height, images.width), (4, 8))
        with patch.object(module, "frame_to_bgr", wraps=module.frame_to_bgr) as convert:
            bgr = images.bgr()
            self.assertIs(images.bgr(), bgr)
        convert.assert_called_once()
        self.assertEqual(bgr.shape, (4, 8, 3))

    def test_bgr_fallback_keeps_original_and_caches_nv12(self):
        bgr = np.full((4, 8, 3), 70, np.uint8)
        images = module.FrameImages(bgr, PIXEL_BGR)
        self.assertIs(images.bgr(), bgr)
        with patch.object(module, "bgr_to_nv12", wraps=bgr_to_nv12) as convert:
            raw = images.nv12()
            self.assertIs(images.nv12(), raw)
        convert.assert_called_once()
        np.testing.assert_allclose(cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_NV12), bgr, atol=2)

    def test_other_formats_remain_distinct_from_nv12(self):
        for pixel_format, shape in ((PIXEL_I420, (6, 8)), (PIXEL_YUYV422, (4, 16))):
            with self.subTest(pixel_format=pixel_format):
                raw = np.full(shape, 128, np.uint8)
                images = module.FrameImages(raw, pixel_format)
                self.assertEqual((images.height, images.width), (4, 8))
                self.assertEqual(images.bgr().shape, (4, 8, 3))
                self.assertEqual(images.nv12().shape, (6, 8))
                self.assertIsNot(images.nv12(), raw)

    def test_metadata_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            module.FrameImages(np.zeros((6, 8), np.uint8), PIXEL_NV12, {"h": 6})

    def test_conversion_interleaves_u_then_v(self):
        bgr = np.full((4, 8, 3), (15, 100, 200), np.uint8)
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).ravel()
        nv12 = bgr_to_nv12(bgr).ravel()
        np.testing.assert_array_equal(nv12[:32], i420[:32])
        np.testing.assert_array_equal(nv12[32::2], i420[32:40])
        np.testing.assert_array_equal(nv12[33::2], i420[40:])

    def test_odd_bgr_dimensions_are_not_silently_resized(self):
        for shape in ((3, 4, 3), (4, 3, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                bgr_to_nv12(np.zeros(shape, np.uint8))

    def test_empty_or_malformed_nv12_rejected(self):
        for shape in ((0, 4), (3, 0), (4, 4), (3, 3), (3, 4, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                nv12_image_shape(np.zeros(shape, np.uint8))


if __name__ == "__main__":
    unittest.main()
