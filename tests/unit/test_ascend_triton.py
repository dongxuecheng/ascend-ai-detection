"""Ascend Triton 39511c9 检测契约测试，无需 Triton SDK 或 NPU。"""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AIDETECTION_LOG_DISABLE_FILE", "1")
from config.config import BaseConfig, config
from detect import triton_client_fast as yolo
from utils.image_formats import bgr_to_nv12


def detections():
    return {
        "NUM_DETS": np.array([2], np.int32),
        "DETECTION_BOXES": np.array([[100, 50, 900, 450], [20, 30, 80, 90]], np.float32),
        "DETECTION_SCORES": np.array([0.9, 0.8], np.float32),
        "DETECTION_CLASSES": np.array([0, 1], np.int32),
    }


class AscendTritonTests(unittest.TestCase):
    def setUp(self):
        self.transport = Mock()
        self.transport.infer.side_effect = lambda **kwargs: detections()
        factory = patch.object(yolo, "_get_shared_triton_client", return_value=self.transport)
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        self.frame = np.full((500, 1000, 3), (10, 30, 70), np.uint8)

    def client(self, **kwargs):
        return yolo.YOLOTritonFast(warmup=False, **kwargs)

    def test_bgr_hwc_input_and_original_coordinates(self):
        client = self.client()
        boxes = client.predict(self.frame, classes=[0])
        request = self.transport.infer.call_args.kwargs
        self.assertEqual(request["model_name"], "YOLO26_DET_PRE_ENSEMBLE")
        self.assertEqual(set(request["inputs"]), {"IMAGE"})
        np.testing.assert_array_equal(request["inputs"]["IMAGE"], self.frame)
        self.assertEqual(request["outputs"], yolo._ASCEND_OUTPUT_NAMES)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].box, [100, 50, 900, 450])
        self.assertEqual(boxes[0].label, "person")
        self.assertEqual(boxes[0].source, "YOLO")

    def test_non_contiguous_input_becomes_contiguous_without_color_swap(self):
        frame = self.frame[:, ::2]
        prepared = self.client()._prepare_input(frame)
        self.assertTrue(prepared.flags.c_contiguous)
        np.testing.assert_array_equal(prepared, frame)

    def test_rgb_ensemble_still_receives_bgr(self):
        client = self.client(model_name="YOLO26_DET_PRE_RGB_ENSEMBLE")
        client.predict(self.frame)
        np.testing.assert_array_equal(self.transport.infer.call_args.kwargs["inputs"]["IMAGE"], self.frame)

    def test_empty_results_and_empty_class_filter(self):
        client = self.client()
        self.assertEqual(client.predict(self.frame, classes=[]), [])
        self.transport.infer.side_effect = None
        self.transport.infer.return_value = {
            "NUM_DETS": np.array([0], np.int32),
            "DETECTION_BOXES": np.empty((0, 4), np.float32),
            "DETECTION_SCORES": np.empty(0, np.float32),
            "DETECTION_CLASSES": np.empty(0, np.int32),
        }
        self.assertEqual(client.predict(self.frame), [])

    def test_count_limits_shm_tail(self):
        result = detections()
        result["NUM_DETS"][0] = 1
        self.transport.infer.side_effect = None
        self.transport.infer.return_value = result
        self.assertEqual(len(self.client().predict(self.frame)), 1)

    def test_invalid_output_contract_raises(self):
        client = self.client()
        self.transport.infer.side_effect = None
        for changes in (
            {"NUM_DETS": np.array([-1], np.int32)},
            {"NUM_DETS": np.array([3], np.int32)},
            {"NUM_DETS": np.array([1.5], np.float32)},
            {"DETECTION_BOXES": np.zeros((1, 2, 4), np.float32)},
        ):
            with self.subTest(changes=list(changes)):
                self.transport.infer.return_value = detections() | changes
                with self.assertRaises(ValueError):
                    client.predict(self.frame)

    def test_clipping_and_nonfinite_filter(self):
        result = detections()
        result["DETECTION_BOXES"][0] = [-10, -20, 2000, 1000]
        result["DETECTION_SCORES"][1] = np.nan
        self.transport.infer.side_effect = None
        self.transport.infer.return_value = result
        boxes = self.client().predict(self.frame)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].box, [0, 0, 1000, 500])

    def test_invalid_images_rejected(self):
        client = self.client()
        for frame in (np.zeros((2, 3), np.uint8), np.zeros((0, 3, 3), np.uint8), self.frame.astype(np.float32)):
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                client.predict(frame)
        self.transport.infer.assert_not_called()

    def test_shm_specs_and_same_http_endpoint_fallback(self):
        client = self.client(url="example:54245", protocol="shm", max_detections=500)
        self.transport.infer.side_effect = [RuntimeError("SHM unavailable"), detections()]
        self.assertEqual(len(client.predict(self.frame)), 2)
        first = self.transport.infer.call_args_list[0].kwargs
        self.assertEqual(first["output_specs"]["DETECTION_BOXES"][0], (500, 4))
        self.assertEqual(first["output_specs"]["NUM_DETS"][0], (1,))
        self.assertNotIn("output_specs", self.transport.infer.call_args.kwargs)
        self.factory.assert_called_with("example:54245", "http")

    def test_no_shm_flag_honored(self):
        client = self.client(protocol="shm", use_shared_memory=False, url="example:54245")
        self.assertEqual(client._protocol, "http")
        self.factory.assert_called_with("example:54245", "http")

    def test_default_ports_follow_protocol(self):
        self.assertEqual(self.client(protocol="grpc").url, "localhost:54246")
        self.assertEqual(self.client(protocol="http").url, "localhost:54245")
        self.assertEqual(self.client(protocol="shm").url, "localhost:54245")

    def test_explicit_legacy_contract_still_available(self):
        client = self.client(backend="legacy", model_name="legacy_detector")
        prepared = client._prepare_input(self.frame)
        self.assertEqual(prepared.shape, (1, 500, 1000, 3))
        np.testing.assert_array_equal(prepared[0, 0, 0], [70, 30, 10])
        self.assertEqual(client.input_name, "raw_image")
        self.assertIn("transform_metadata", client.output_names)

    def test_reject_old_input_name_in_ascend_mode(self):
        with self.assertRaises(ValueError):
            self.client(input_name="raw_image")

    def test_nv12_is_forwarded_without_color_conversion_or_copy(self):
        client = self.client(model_name="YOLO26_DET_PRE_YUV_ENSEMBLE")
        raw = np.full((750, 1000), 128, np.uint8)
        with patch.object(yolo, "bgr_to_nv12", side_effect=AssertionError("must not convert")):
            boxes = client.predict_nv12(raw)
        request = self.transport.infer.call_args.kwargs
        self.assertEqual(set(request["inputs"]), {"YUV"})
        self.assertEqual(request["inputs"]["YUV"].shape, (750, 1000, 1))
        self.assertTrue(np.shares_memory(request["inputs"]["YUV"], raw))
        self.assertEqual(boxes[0].box, [100, 50, 900, 450])

    def test_nv12_output_clipped_to_image_height_not_buffer_height(self):
        client = self.client(model_name="YOLO26_DET_PRE_YUV_ENSEMBLE")
        result = detections()
        result["DETECTION_BOXES"][0, 3] = 700
        self.transport.infer.side_effect = None
        self.transport.infer.return_value = result
        boxes = client.predict_nv12(np.full((750, 1000, 1), 128, np.uint8))
        self.assertEqual(boxes[0].box[3], 500)

    def test_bgr_picture_entry_converts_to_nv12_for_yuv_model(self):
        client = self.client(model_name="YOLO26_DET_PRE_YUV_ENSEMBLE")
        client.predict(self.frame)
        np.testing.assert_array_equal(self.transport.infer.call_args.kwargs["inputs"]["YUV"],
                                      bgr_to_nv12(self.frame)[..., None])

    def test_invalid_nv12_and_wrong_model_rejected(self):
        client = self.client(model_name="YOLO26_DET_PRE_YUV_ENSEMBLE")
        for raw in (np.zeros((4, 4), np.uint8), np.zeros((3, 3), np.uint8),
                    np.zeros((3, 4, 3), np.uint8), np.zeros((3, 4), np.float32)):
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                client.predict_nv12(raw)
        with self.assertRaises(ValueError):
            self.client().predict_nv12(np.zeros((3, 4), np.uint8))

    def test_yuv_warmup_uses_nv12(self):
        yolo.YOLOTritonFast(model_name="YOLO26_DET_PRE_YUV_ENSEMBLE", warmup=True)
        self.assertEqual(self.transport.infer.call_args.kwargs["inputs"]["YUV"].shape, (960, 640, 1))


class AscendConfigTests(unittest.TestCase):
    def test_protocol_endpoint_selection(self):
        cfg = BaseConfig(TRITON_YOLO_URL="host:54245", TRITON_YOLO_GRPC_URL="host:54246", TRITON_PROTOCOL="grpc")
        self.assertEqual(cfg.triton_endpoint(), "host:54246")
        self.assertEqual(cfg.triton_endpoint("http"), "host:54245")
        self.assertEqual(cfg.triton_endpoint("shm"), "host:54245")
        with self.assertRaises(ValueError):
            cfg.triton_endpoint("invalid")

    def test_model_references_resolve_and_unavailable_models_disabled(self):
        self.assertEqual(config.CLASSIFICATION_MODEL_CONFIGS, {})
        self.assertEqual(config.ALGORITHM_CLASSIFIERS, {})
        self.assertNotIn("yolov5_ensemble", config.YOLO_MODEL_CONFIGS)
        for models in config.ALGM_PRE_YOLO_MODEL_DETECT_CLASSES.values():
            for name in models:
                self.assertIn(name, config.YOLO_MODEL_CONFIGS)
                self.assertEqual(name, "YOLO26_DET_PRE_YUV_ENSEMBLE")
                self.assertEqual(config.YOLO_MODEL_CONFIGS[name]["input_name"], "YUV")


if __name__ == "__main__":
    unittest.main()
