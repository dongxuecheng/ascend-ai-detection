"""SAM3 环境变量地址与算法分组的配置回归测试。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config.config import _resolve_sam3_url_groups


class Sam3UrlGroupTests(unittest.TestCase):
    def test_environment_reference_and_literal_url(self):
        routes = _resolve_sam3_url_groups(
            [{"url_env": "SAM3_URL", "codes": ["8", 32]},
             {"url_env": "SAM3_URL_REFINE", "codes": ["0", "14"]},
             {"url": "http://legacy/predict", "codes": ["99"]}],
            {"SAM3_URL": "http://regular/predict",
             "SAM3_URL_REFINE": "http://refine/predict-obj-refine"},
        )
        self.assertEqual(routes, {
            "8": "http://regular/predict", "32": "http://regular/predict",
            "0": "http://refine/predict-obj-refine", "14": "http://refine/predict-obj-refine",
            "99": "http://legacy/predict",
        })

    def test_unknown_reference_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "SAM3_URL_TYPO"):
            _resolve_sam3_url_groups([{"url_env": "SAM3_URL_TYPO", "codes": ["8"]}], {})

    def test_empty_referenced_address_fails_at_startup(self):
        for value in ["", " "]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "SAM3_URL"):
                _resolve_sam3_url_groups(
                    [{"url_env": "SAM3_URL", "codes": ["8"]}], {"SAM3_URL": value},
                )

    def test_empty_groups_leave_default_routing_to_caller(self):
        self.assertEqual(_resolve_sam3_url_groups([], {}), {})

    def load_config(self, refine_url):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["SAM3_URL"] = "http://regular.example:18000/predict"
        env["SAM3_URL_OBJ"] = "http://fallback.example:18000/predict"
        if refine_url is None:
            env.pop("SAM3_URL_REFINE", None)
        else:
            env["SAM3_URL_REFINE"] = refine_url
        # 独立进程重新加载配置，禁用本机 .env，避免污染其他测试的全局配置。
        result = subprocess.run(
            [sys.executable, "-c",
             "import dotenv; dotenv.load_dotenv = lambda: None; "
             "import json; from config.config import config; "
             "print(json.dumps({'routes': config.ALGORITHM_SAM3_URL, "
             "'default': config.SAM3_URL_OBJ}))"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def test_real_yaml_routes_follow_environment_overrides(self):
        loaded = self.load_config("http://refine.example:19000/predict-obj-refine")
        expected = dict.fromkeys(
            ["8", "32", "33", "34", "52", "53", "38", "56", "49", "59", "50"],
            "http://regular.example:18000/predict",
        )
        expected.update(dict.fromkeys(
            ["0", "14", "58", "10"], "http://refine.example:19000/predict-obj-refine",
        ))
        self.assertEqual(loaded["routes"], expected)
        self.assertEqual(loaded["default"], "http://fallback.example:18000/predict")

    def test_missing_refine_variable_uses_same_server(self):
        loaded = self.load_config(None)
        self.assertEqual(loaded["routes"]["0"], "http://regular.example:18000/predict-obj-refine")


if __name__ == "__main__":
    unittest.main()
