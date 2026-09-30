"""只隔离检测器依赖，执行真实分析分发器的惰性取帧逻辑。"""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


class AnalyzerLazyFrameTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "src/core/analyzer.py"
        modules = {}
        # 测试分发器，不初始化跟踪器、推理客户端等外部依赖。
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ImportFrom) and node.module.startswith("analyze."):
                module = ModuleType(node.module)
                for item in node.names:
                    setattr(module, item.name, type(item.name, (), {}))
                modules[node.module] = module
        cfg = ModuleType("config.config")
        cfg.config = SimpleNamespace(THREAD_LOCAL_DETECTOR_CLASSES=[], ALGORITHM_DETECTORS={"8": ["test"]})
        modules["config.config"] = cfg
        spec = importlib.util.spec_from_file_location("analyzer_under_test", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.module)
        self.task = SimpleNamespace(algorithmCode="8", deviceId="camera")

    def test_box_only_rule_never_requests_image_even_for_empty_input(self):
        class BoxRule:
            def detect(self, boxes, fences, device_id, image_width, image_height):
                self.dimensions = (image_width, image_height)
                return []

        rule = BoxRule()
        provider = Mock(side_effect=AssertionError("must stay NV12"))
        with patch.object(self.module, "_get_detector_instance", return_value=rule):
            self.module.analyze_for_task([], self.task, image_width=8, image_height=4, frame_provider=provider)
        provider.assert_not_called()
        self.assertEqual(rule.dimensions, (8, 4))

    def test_image_rules_receive_same_bgr_once(self):
        received = []

        class ImageRule:
            def detect(self, boxes, frame=None, **kwargs):
                received.append(frame)
                return []

        self.module._detector_class_names["8"] = ["one", "two"]
        bgr = np.zeros((4, 8, 3), np.uint8)
        provider = Mock(return_value=bgr)
        with patch.object(self.module, "_get_detector_instance", return_value=ImageRule()):
            self.module.analyze_for_task([], self.task, frame_provider=provider)
        provider.assert_called_once()
        self.assertEqual(len(received), 2)
        self.assertTrue(all(item is bgr for item in received))

    def test_existing_explicit_bgr_call_still_supported(self):
        received = []

        class ImageRule:
            def detect(self, boxes, frame=None, **kwargs):
                received.append(frame)
                return []

        bgr = np.zeros((4, 8, 3), np.uint8)
        provider = Mock(side_effect=AssertionError("already have BGR"))
        with patch.object(self.module, "_get_detector_instance", return_value=ImageRule()):
            self.module.analyze_for_task([], self.task, frame=bgr, frame_provider=provider)
        provider.assert_not_called()
        self.assertIs(received[0], bgr)


if __name__ == "__main__":
    unittest.main()
