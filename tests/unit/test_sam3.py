"""ascend-sam3 请求契约和局部 mask 转换回归测试，无需外部推理服务。"""

import base64
import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AIDETECTION_LOG_DISABLE_FILE", "1")

from detect.sam3 import call_sam3, parse_sam3_response
from utils.rle import binary_mask_to_rle


def ascend_result(mask=None):
    item = {"label": "person", "score": 0.9, "box": [2.2, 3.4, 6.9, 5.8]}
    if mask is not None:
        item.update(mask=mask, mask_width=4, mask_height=2)
    return {"results": [item]}


class Sam3RequestTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((8, 12, 3), (20, 80, 180), dtype=np.uint8)
        self.post_patch = patch("detect.sam3.requests.post")
        self.post = self.post_patch.start()
        self.addCleanup(self.post_patch.stop)
        self.post.return_value = Mock()
        self.post.return_value.json.return_value = ascend_result([1, 2, 4, 2, 8, 1])

    def test_latest_json_contract_and_jpeg(self):
        boxes = call_sam3(
            self.frame, ["person", "hard hat"], confidence_threshold=0.65,
            return_mask=True, url="http://sam3.example:18000/predict",
        )
        args, kwargs = self.post.call_args
        self.assertEqual(args, ("http://sam3.example:18000/predict",))
        payload = kwargs["json"]
        self.assertEqual(set(payload), {"image", "class_names", "confidence", "return_mask"})
        self.assertEqual(payload["class_names"], ["person", "hard hat"])
        self.assertEqual(payload["confidence"], 0.65)
        self.assertIs(payload["return_mask"], True)
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(payload["image"]), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, self.frame.shape)
        np.testing.assert_allclose(decoded[0, 0], self.frame[0, 0], atol=3)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].source, "SAM3")
        self.post.return_value.raise_for_status.assert_called_once()

    def test_configured_default_url_and_timeout(self):
        with patch("detect.sam3.config") as config:
            config.SAM3_URL_OBJ = "http://gateway:18000/predict"
            config.SAM3_TIMEOUT_SECONDS = 90
            call_sam3(self.frame, ["phone", "person", "hand"])
        self.assertEqual(self.post.call_args.args[0], "http://gateway:18000/predict")
        self.assertEqual(self.post.call_args.kwargs["timeout"], 90)
        self.assertIs(self.post.call_args.kwargs["json"]["return_mask"], False)

    def test_empty_prompts_do_not_send_request(self):
        self.assertEqual(call_sam3(self.frame, []), [])
        self.post.assert_not_called()

    def test_encoding_failure_does_not_send_request(self):
        with patch("detect.sam3.cv2.imencode", return_value=(False, None)):
            self.assertEqual(call_sam3(self.frame, ["person"]), [])
        self.post.assert_not_called()

    def test_timeout_returns_empty_results(self):
        self.post.side_effect = requests.exceptions.Timeout("test timeout")
        self.assertEqual(call_sam3(self.frame, ["person"]), [])

    def test_http_failure_does_not_parse_response(self):
        self.post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        self.assertEqual(call_sam3(self.frame, ["person"]), [])
        self.post.return_value.json.assert_not_called()

    def test_invalid_json_returns_empty_results(self):
        self.post.return_value.json.side_effect = ValueError("invalid JSON")
        self.assertEqual(call_sam3(self.frame, ["person"]), [])


class Sam3MaskTests(unittest.TestCase):
    def test_row_major_start_length_rle_is_not_coco_counts(self):
        box = parse_sam3_response(ascend_result([1, 2, 4, 2, 8, 1]))[0]
        expected = np.array([[1, 1, 0, 1], [1, 0, 0, 1]], dtype=np.uint8)
        self.assertEqual(box.mask, binary_mask_to_rle(expected))
        np.testing.assert_array_equal(box.rle_to_mask(), expected)
        self.assertEqual(box.box, [2.2, 3.4, 6.9, 5.8])

    def test_mask_is_placed_at_integer_box_origin(self):
        box = parse_sam3_response(ascend_result([1, 2, 4, 2, 8, 1]))[0]
        expected = np.zeros((8, 10), dtype=np.uint8)
        expected[3:5, 2:6] = [[1, 1, 0, 1], [1, 0, 0, 1]]
        np.testing.assert_array_equal(box.compute_mask_array(10, 8), expected)

    def test_all_background_mask_is_not_missing(self):
        box = parse_sam3_response(ascend_result([]))[0]
        self.assertIsNotNone(box.mask)
        np.testing.assert_array_equal(box.rle_to_mask(), np.zeros((2, 4), dtype=np.uint8))

    def test_all_foreground_mask(self):
        box = parse_sam3_response(ascend_result([1, 8]))[0]
        np.testing.assert_array_equal(box.rle_to_mask(), np.ones((2, 4), dtype=np.uint8))

    def test_random_masks_match_upstream_encoding(self):
        rng = np.random.default_rng(7)
        for height, width in [(1, 1), (1, 7), (6, 1), (3, 5), (17, 29)]:
            for _ in range(10):
                expected = rng.integers(0, 2, size=(height, width), dtype=np.uint8)
                # 对照上游 service/main.py 的 _png_to_rle，默认 ravel 为行优先。
                flat = np.concatenate([[0], expected.ravel(), [0]])
                runs = np.where(flat[1:] != flat[:-1])[0] + 1
                runs[1::2] -= runs[0::2]
                data = {"results": [{"label": "coal", "score": 0.8,
                                    "box": [0, 0, width, height], "mask": runs.tolist(),
                                    "mask_width": width, "mask_height": height}]}
                np.testing.assert_array_equal(parse_sam3_response(data)[0].rle_to_mask(), expected)

    def test_absent_or_null_mask_preserves_detection(self):
        self.assertIsNone(parse_sam3_response(ascend_result())[0].mask)
        data = ascend_result()
        data["results"][0].update(mask=None, mask_width=0, mask_height=0)
        self.assertIsNone(parse_sam3_response(data)[0].mask)

    def test_bad_masks_preserve_box_without_mask(self):
        bad_masks = [[1], [0, 1], [1, -1], [1, 0], [8, 2], [1.5, 2],
                     [1, 3, 2, 2], [5, 1, 1, 1], "invalid"]
        for mask in bad_masks:
            with self.subTest(mask=mask):
                boxes = parse_sam3_response(ascend_result(mask))
                self.assertEqual(len(boxes), 1)
                self.assertIsNone(boxes[0].mask)

    def test_missing_invalid_or_inconsistent_dimensions(self):
        for width in [None, 0, -1, 4.5, "4", 5]:
            with self.subTest(width=width):
                data = ascend_result([1, 2])
                if width is None:
                    del data["results"][0]["mask_width"]
                else:
                    data["results"][0]["mask_width"] = width
                self.assertIsNone(parse_sam3_response(data)[0].mask)

    def test_existing_coco_mask_is_preserved(self):
        mask = binary_mask_to_rle(np.array([[0, 1], [1, 0]], dtype=np.uint8))
        data = {"results": [{"label": "person", "score": 0.8, "box": [0, 0, 2, 2], "mask": mask}]}
        self.assertIs(parse_sam3_response(data)[0].mask, mask)


class Sam3ResponseTests(unittest.TestCase):
    def test_legacy_envelopes_still_work(self):
        items = ascend_result()["results"]
        for response in [items, {"code": 0, "data": items}, {"predictions": items}, {"output": items}]:
            self.assertEqual(len(parse_sam3_response(response)), 1)

    def test_empty_and_error_responses(self):
        for response in [{"results": []}, {"code": 1, "msg": "error"},
                         {"success": False, "error": "failed"}, {}, None,
                         {"results": {}}, {"results": [None, "bad"]}]:
            self.assertEqual(parse_sam3_response(response), [])

    def test_invalid_boxes_are_skipped_without_losing_valid_items(self):
        for coords in [None, [0, 0, 1], [0, 0, 0, 1], [2, 0, 1, 1], [0, 0, float("nan"), 1]]:
            data = ascend_result()
            data["results"].insert(0, {"label": "person", "score": 0.8, "box": coords})
            self.assertEqual(len(parse_sam3_response(data)), 1)


class Sam3ConfigTests(unittest.TestCase):
    def load_config(self, **env):
        with patch.dict(os.environ, env), patch("dotenv.load_dotenv"):
            if "SAM3_URL_OBJ" not in env:
                os.environ.pop("SAM3_URL_OBJ", None)
            return runpy.run_path(str(ROOT / "src/config/config.py"))["config"]

    def test_url_obj_defaults_to_main_endpoint(self):
        config = self.load_config(SAM3_URL="http://gateway:18000/predict")
        self.assertEqual(config.SAM3_URL_OBJ, config.SAM3_URL)
        self.assertEqual(config.ALGORITHM_SAM3_URL["8"], config.SAM3_URL)

    def test_explicit_secondary_endpoint_and_timeout(self):
        config = self.load_config(SAM3_URL="http://gateway:18000/predict",
                                  SAM3_URL_OBJ="http://secondary:18000/predict",
                                  SAM3_TIMEOUT_SECONDS="90")
        self.assertEqual(config.SAM3_URL_OBJ, "http://secondary:18000/predict")
        self.assertEqual(config.SAM3_TIMEOUT_SECONDS, 90)


if __name__ == "__main__":
    unittest.main()
