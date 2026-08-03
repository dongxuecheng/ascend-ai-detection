"""
Triton 图像分类客户端（基于 triton_client 封装，调用 classifier_ensemble）

用于对规则引擎检出的违规目标裁剪区域进行二次分类确认，
例如对吸烟检测后的目标调用 classifier_ensemble 判断是/否吸烟。

classifier_ensemble 配置：
  - 输入：raw_image  [1, H, W, 3]  uint8（RGB）
  - 输出：classes    [num_classes]  int32
          scores     [num_classes]  float32（已归一化概率）
          transform_metadata [6]   float32

底层通信统一使用 src/triton_client.TritonClient，支持 HTTP 与共享内存（SHM），
SHM 初始化失败时自动回退到 HTTP。
"""

import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from triton_client import TritonClient, TritonClientError
from utils.logger import setup_logger

logger = setup_logger("classification_client")


class TritonClassificationClient:
    """
    基于 triton_client.TritonClient 的图像分类客户端。

    输入：BGR 格式 numpy 图像（已由业务层裁剪好的目标区域）
    输出：(预测类别 ID, 置信度, 各类别概率字典)
    """

    # classifier_ensemble 固定输入/输出名
    _DEFAULT_INPUT_NAME = "raw_image"
    _DEFAULT_OUTPUT_NAMES = ["classes", "scores", "transform_metadata"]

    def __init__(
        self,
        url: str,
        model_name: str,
        input_name: str = "images",
        output_name: str = "output0",
        input_size: int = 224,
        labels: Optional[List[str]] = None,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        use_shared_memory: bool = True,
        protocol: str = "shm",
    ):
        self.url = url
        self.model_name = model_name
        self.input_size = input_size
        self.labels = list(labels) if labels else []
        self.num_classes = len(self.labels)

        # mean/std 在此保留以保持接口兼容；实际预处理已由 ensemble 完成，不再使用
        self.mean = mean
        self.std = std

        # 兼容旧配置中的 images/input/output0：实际使用 classifier_ensemble 默认名
        self.input_name = input_name if input_name not in ("images", "input") else self._DEFAULT_INPUT_NAME
        self.output_names = list(self._DEFAULT_OUTPUT_NAMES)

        self._client: Optional[TritonClient] = None
        self._protocol = protocol.lower()
        self._output_specs: Optional[Dict[str, Tuple[Tuple[int, ...], np.dtype]]] = None
        self._closed = False

        if self._protocol not in ("grpc", "http", "shm"):
            raise ValueError(f"不支持的 protocol: {protocol}")

        # 创建 TritonClient：gRPC 在 TritonClient 内部已按 URL 进程级共享
        self._client = TritonClient(url=url, protocol=self._protocol, verbose=False)
        if not self._client.is_server_live():
            raise RuntimeError(f"Triton 分类服务未存活: {url}")
        if not self._client.is_model_ready(model_name):
            raise RuntimeError(f"分类模型 {model_name} 未就绪")

        # 仅在 HTTP 下尝试 SHM；gRPC 不走 SHM
        if self._protocol == "http" and use_shared_memory and self.num_classes > 0:
            try:
                self._setup_shared_memory()
                self._protocol = "shm"
                logger.info(f"Triton 分类客户端将使用共享内存协议: {url}")
            except Exception as e:
                logger.warning(f"共享内存初始化失败，回退到 HTTP: {e}")
                self._fallback_to_http()

        logger.info(
            f"Triton 分类客户端初始化成功 | protocol={self._protocol} | "
            f"model={model_name} | input_size={input_size} | labels={self.labels}"
        )

    # ---------- 共享内存 / 协议回退 ----------
    def _setup_shared_memory(self) -> None:
        """创建 SHM 客户端并预置输出缓冲区规格。"""
        if self._client is None:
            raise TritonClientError("客户端已关闭")
        self._client.close()
        self._client = TritonClient(url=self.url, protocol="shm", verbose=False)
        self._output_specs = {
            "classes": ((1, self.num_classes), np.int32),
            "scores": ((1, self.num_classes), np.float32),
            "transform_metadata": ((1, 6), np.float32),
        }

    def _fallback_to_http(self) -> None:
        """关闭当前客户端并回退到 HTTP 协议。"""
        self._protocol = "http"
        self._output_specs = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = TritonClient(url=self.url, protocol="http", verbose=False)

    # ---------- 生命周期 ----------
    def close(self) -> None:
        """关闭 Triton 客户端连接（幂等）。"""
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.debug(f"关闭分类客户端异常: {e}")
            self._client = None

    def __del__(self):
        if sys.is_finalizing() or self._closed:
            return
        self.close()

    # ---------- 输入构造 ----------
    @staticmethod
    def _prepare_input(image: np.ndarray) -> np.ndarray:
        """将 OpenCV BGR 图像转成 Triton ensemble 期望的 RGB [1,H,W,3] uint8。"""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"输入必须是 HWC BGR 图像，当前 shape={image.shape}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb[np.newaxis, ...].astype(np.uint8, copy=False)

    # ---------- 推理 ----------
    def _infer(self, input_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """调用 TritonClient 完成推理，SHM 失败时一次性回退 HTTP。"""
        if self._client is None or self._closed:
            raise RuntimeError("Triton 分类客户端已关闭")

        try:
            return self._client.infer(
                model_name=self.model_name,
                inputs={self.input_name: input_arr},
                outputs=self.output_names,
                output_specs=self._output_specs if self._protocol == "shm" else None,
            )
        except Exception as e:
            if self._protocol == "shm":
                logger.warning(f"SHM 推理失败，尝试回退到 HTTP: {e}")
                self._fallback_to_http()
                return self._client.infer(
                    model_name=self.model_name,
                    inputs={self.input_name: input_arr},
                    outputs=self.output_names,
                )
            raise

    # ---------- 后处理 ----------
    def predict(self, image: np.ndarray) -> Tuple[int, float, Dict[str, float]]:
        """
        对单张图像进行分类。

        :param image: BGR 格式 numpy 图像
        :return: (预测类别 ID, 置信度, 各类别概率字典)
        """
        input_arr = self._prepare_input(image)
        result = self._infer(input_arr)

        scores = result["scores"]
        # 兼容 [batch, num_classes] 和 [num_classes] 两种输出
        if scores.ndim == 2:
            scores = scores[0]

        probs = scores.astype(np.float64)

        # scores 已是 classifier_postprocess 输出的归一化概率，
        # 这里只做简单校验，若意外返回 logits 则兜底 softmax
        if not (np.all(probs >= 0) and abs(probs.sum() - 1.0) < 0.01):
            e_x = np.exp(probs - np.max(probs))
            probs = e_x / e_x.sum()

        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        prob_dict = {
            label: float(probs[i])
            for i, label in enumerate(self.labels)
            if i < len(probs)
        }

        return class_id, confidence, prob_dict
