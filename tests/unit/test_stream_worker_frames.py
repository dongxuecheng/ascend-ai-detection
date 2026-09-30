"""隔离远端任务、推理和上传依赖，验证生产 worker 的读帧入口。"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from stream import remote_capture as rc
from stream import frame as frame_module


class WorkerFrameTests(unittest.TestCase):
    def setUp(self):
        dependencies = {
            "config.config": {"config": SimpleNamespace(FULL_FRAME_ALGORITHMS=[])},
            "utils.logger": {"setup_logger": Mock(return_value=Mock())},
            "utils.obj": {"Box": type("Box", (), {})},
            "task.upload": {"EventUploader": Mock},
            "task.getFence": {"TaskFence": Mock},
            "detect.sam3": {"call_sam3": Mock()},
            "detect.triton_client_fast": {"YOLOTritonFast": type("YOLO", (), {})},
            "core.analyzer": {"analyze_for_task": Mock(), "merge_prompts_for_tasks": Mock()},
            "core.classifier": {"classify_for_task": Mock()},
            "llm.vl_analyzer": {"vl_analyze_for_task": Mock()},
            "utils.osd": {"render_alert_frame": Mock()},
            "utils.alert_dedup": {"AlertDedup": Mock},
        }
        modules = {}
        for name, attributes in dependencies.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        spec = importlib.util.spec_from_file_location("worker_under_test", ROOT / "src/core/stream_worker.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.module)

    def run_one_frame(self, worker, frame, corrupted, meta=None):
        capture = Mock()
        capture.start_stream.return_value = "s"
        capture.get_last_frame_meta.return_value = meta

        def read(*args, **kwargs):
            worker.stopped = True
            return 1234, frame, corrupted

        capture.read_ex.side_effect = read
        with patch.object(self.module, "RTSPClient", return_value=capture):
            worker.run()
        capture.read_ex.assert_called_once_with("s", blocking=True, timeout_ms=5000)
        capture.disconnect.assert_called_once()
        return capture

    def test_corrupted_frame_does_not_reach_inference_or_task_schedule(self):
        worker = self.module.StreamWorker("rtsp://example/video", [SimpleNamespace(id="task", algorithmCode="8")])
        with patch.object(frame_module, "frame_to_bgr") as convert:
            capture = self.run_one_frame(worker, np.zeros((2, 4, 3), np.uint8), True)
        convert.assert_not_called()
        capture.get_last_frame_meta.assert_not_called()
        self.module.call_sam3.assert_not_called()
        self.module.analyze_for_task.assert_not_called()
        self.assertEqual(worker.task_last_run, {"task": 0.0})
        self.assertGreater(worker._last_frame_at, 0)
        self.assertEqual(worker._consecutive_read_failures, 0)

    def test_nv12_frame_without_ready_tasks_is_not_converted(self):
        worker = self.module.StreamWorker("rtsp://example/video", [])
        frame = np.full((3, 4), 128, np.uint8)
        with patch.object(frame_module, "frame_to_bgr", wraps=rc.frame_to_bgr) as convert:
            capture = self.run_one_frame(worker, frame, False, {"pix_fmt": rc.PIXEL_NV12})
        self.assertEqual(capture.start_stream.call_args.kwargs["pixel_format"], rc.PIXEL_NV12)
        convert.assert_not_called()

    def configure_task(self, prompts=None):
        self.module.config.ALGORITHM_INTERVALS = {}
        self.module.config.DEFAULT_ALGORITHM_INTERVAL = 0
        self.module.config.ALGM_PRE_YOLO_MODEL_DETECT_CLASSES = {"8": {"yuv_model": ["person"]}}
        self.module.config.ALGORITHM_SAM3_PROMPT = {"8": prompts or []}
        self.module.config.ALGORITHM_SAM3_URL = {}
        self.module.config.SAM3_URL_OBJ = "http://sam3/predict"
        self.module.config.FENCE_ALGORITHMS = []
        self.module.config.USE_YOLO_BOXES = []
        self.module.config.ALGORITHM_CLASSIFIERS = {}
        self.module.config.VL_ENABLED = False
        self.module.config.ALERT_DEDUP_CONFIG = {}
        self.module.analyze_for_task.return_value = []
        self.module.merge_prompts_for_tasks.return_value = (prompts or [], False)
        self.module.call_sam3.return_value = []
        return SimpleNamespace(id="task", algorithmCode="8", electricFence=[], deviceId="camera",
                               deviceAlgorithmIp="example", deviceChannel="1", deviceName="camera",
                               algorithmName="test")

    def test_empty_yolo_result_skips_bgr_and_sam3_but_updates_rules(self):
        task = self.configure_task(["person"])
        worker = self.module.StreamWorker("rtsp://example/video", [task])
        raw = np.full((3, 4), 128, np.uint8)
        client = Mock(input_name="YUV", label_map={0: "person"})
        client.predict_nv12.return_value = []
        with patch.object(self.module, "_get_yolo_client", return_value=client), patch.object(
            frame_module, "frame_to_bgr", wraps=rc.frame_to_bgr
        ) as convert:
            self.run_one_frame(worker, raw, False, {"pix_fmt": rc.PIXEL_NV12, "w": 4, "h": 2})
        client.predict_nv12.assert_called_once_with(raw, classes=[0])
        client.predict.assert_not_called()
        convert.assert_not_called()
        self.module.call_sam3.assert_not_called()
        self.module.analyze_for_task.assert_called_once()
        self.assertEqual(self.module.analyze_for_task.call_args.kwargs["image_height"], 2)
        self.assertGreater(worker.task_last_run[task.id], 0)

    def test_sam3_rules_and_upload_share_one_bgr_conversion(self):
        task = self.configure_task(["person"])
        worker = self.module.StreamWorker("rtsp://example/video", [task])
        raw = np.full((3, 4), 128, np.uint8)
        person = SimpleNamespace(label="person")
        client = Mock(input_name="YUV", label_map={0: "person"})
        client.predict_nv12.return_value = [person]
        self.module.call_sam3.return_value = [person]
        used_frames = []

        def analyze(*args, **kwargs):
            used_frames.append(kwargs["frame_provider"]())
            return [person]

        self.module.analyze_for_task.side_effect = analyze
        with patch.object(self.module, "_get_yolo_client", return_value=client), patch.object(
            frame_module, "frame_to_bgr", wraps=rc.frame_to_bgr
        ) as convert:
            self.run_one_frame(worker, raw, False, {"pix_fmt": rc.PIXEL_NV12})
        convert.assert_called_once()
        bgr = self.module.call_sam3.call_args.args[0]
        self.assertEqual(bgr.shape, (2, 4, 3))
        self.assertIs(used_frames[0], bgr)
        self.assertIs(self.module.render_alert_frame.call_args.kwargs["frame"], bgr)
        self.assertIs(self.module.uploader.add_alert.call_args.args[1], bgr)

    def test_existing_bgr_stream_is_converted_once_for_yuv_model(self):
        task = self.configure_task()
        worker = self.module.StreamWorker("rtsp://example/video", [task])
        raw = np.full((2, 4, 3), 128, np.uint8)
        client = Mock(input_name="YUV", label_map={0: "person"})
        client.predict_nv12.return_value = []
        with patch.object(self.module, "_get_yolo_client", return_value=client), patch.object(
            frame_module, "bgr_to_nv12", wraps=frame_module.bgr_to_nv12
        ) as convert:
            self.run_one_frame(worker, raw, False, {"pix_fmt": rc.PIXEL_BGR})
        convert.assert_called_once_with(raw)
        self.assertEqual(client.predict_nv12.call_args.args[0].shape, (3, 4))

    def test_pure_yolo_alert_converts_only_after_rules_and_reuses_for_review(self):
        task = self.configure_task()
        self.module.config.ALGORITHM_CLASSIFIERS = {"8": ["classifier"]}
        self.module.config.VL_ENABLED = True
        self.module.config.ALGORITHM_VL_CONFIG = {"8": {"enabled": True}}
        worker = self.module.StreamWorker("rtsp://example/video", [task])
        raw = np.full((3, 4), 128, np.uint8)
        person = SimpleNamespace(label="person")
        client = Mock(input_name="YUV", label_map={0: "person"})
        client.predict_nv12.return_value = [person]
        self.module.classify_for_task.return_value = [person]
        self.module.vl_analyze_for_task.return_value = [person]
        with patch.object(self.module, "_get_yolo_client", return_value=client), patch.object(
            frame_module, "frame_to_bgr", wraps=rc.frame_to_bgr
        ) as convert:
            def analyze(*args, **kwargs):
                convert.assert_not_called()
                return [person]

            self.module.analyze_for_task.side_effect = analyze
            self.run_one_frame(worker, raw, False, {"pix_fmt": rc.PIXEL_NV12})
        convert.assert_called_once()
        self.module.call_sam3.assert_not_called()
        bgr = self.module.classify_for_task.call_args.args[0]
        self.assertIs(self.module.vl_analyze_for_task.call_args.args[0], bgr)
        self.assertIs(self.module.render_alert_frame.call_args.kwargs["frame"], bgr)
        self.assertEqual(bgr.shape, (2, 4, 3))

    def test_nv12_stream_with_bgr_model_uses_compatible_picture_entry(self):
        task = self.configure_task()
        worker = self.module.StreamWorker("rtsp://example/video", [task])
        client = Mock(input_name="IMAGE", label_map={0: "person"})
        client.predict.return_value = []
        raw = np.full((3, 4), 128, np.uint8)
        with patch.object(self.module, "_get_yolo_client", return_value=client), patch.object(
            frame_module, "frame_to_bgr", wraps=rc.frame_to_bgr
        ) as convert:
            self.run_one_frame(worker, raw, False, {"pix_fmt": rc.PIXEL_NV12})
        convert.assert_called_once()
        client.predict_nv12.assert_not_called()
        self.assertEqual(client.predict.call_args.args[0].shape, (2, 4, 3))

    def test_shm_presence_uses_restarted_stream_id(self):
        worker = self.module.StreamWorker("rtsp://example/video", [])
        worker._stream_id = "old"
        worker.capture = Mock()
        worker.capture.get_active_stream_id.return_value = "new"
        with patch.object(self.module.os.path, "exists", return_value=True) as exists:
            self.assertTrue(worker._shm_file_exists())
        exists.assert_called_once_with("/dev/shm/new")

    def test_yolo_factory_uses_configured_protocol_endpoint_and_ascend_contract(self):
        self.module.config.TRITON_PROTOCOL = "grpc"
        self.module.config.triton_endpoint = Mock(return_value="ascend:8001")
        self.module.config.YOLO_MODEL_CONFIGS = {"YOLO26_DET_PRE_ENSEMBLE": {
            "backend": "ascend", "input_name": "IMAGE", "max_detections": 400,
        }}
        with patch.object(self.module, "YOLOTritonFast") as factory:
            self.module._get_yolo_client("YOLO26_DET_PRE_ENSEMBLE")
        self.module.config.triton_endpoint.assert_called_once_with("grpc")
        self.assertEqual(factory.call_args.kwargs["url"], "ascend:8001")
        self.assertEqual(factory.call_args.kwargs["input_name"], "IMAGE")
        self.assertEqual(factory.call_args.kwargs["backend"], "ascend")
        self.assertEqual(factory.call_args.kwargs["max_detections"], 400)

    def test_transient_shm_missing_waits_but_persistent_missing_recovers(self):
        for elapsed, should_recover in ((0, False), (15, True)):
            with self.subTest(elapsed=elapsed):
                worker = self.module.StreamWorker("rtsp://example/video", [])
                worker._query_stream_status = Mock(return_value=rc.STATUS_CONNECTED)
                worker._shm_file_exists = Mock(return_value=False)
                worker._recover_stream = Mock(return_value=True)
                with patch.object(self.module.time, "time", side_effect=[100, 100 + elapsed]), patch.object(
                    self.module.time, "sleep"
                ):
                    self.run_one_frame(worker, None, False)
                self.assertEqual(worker._recover_stream.called, should_recover)


if __name__ == "__main__":
    unittest.main()
