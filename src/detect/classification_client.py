"""
Triton 图像分类客户端。

用于对规则引擎检出的违规目标裁剪区域进行二次分类确认，
例如对吸烟检测后的目标调用 resnet_smoke 模型判断是/否吸烟。
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tritonclient.http as httpclient

from utils.logger import setup_logger

logger = setup_logger("classification_client")


class TritonClassificationClient:
    """
    基于 Triton HTTP 的图像分类客户端。
    输入：BGR 图像（numpy array）
    输出：预测类别 ID、置信度、各类别概率
    """

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
    ):
        self.url = url
        self.model_name = model_name
        self.input_name = input_name
        self.output_name = output_name
        self.input_size = input_size
        self.labels = list(labels) if labels else []
        self.mean = mean
        self.std = std

        self.client = httpclient.InferenceServerClient(url=url, verbose=False, concurrency=1)
        if not self.client.is_server_live():
            raise RuntimeError(f"Triton 分类服务未存活: {url}")
        if not self.client.is_model_ready(model_name):
            raise RuntimeError(f"分类模型 {model_name} 未就绪")

        logger.info(
            f"Triton 分类客户端初始化成功 | model={model_name} | "
            f"url={url} | input_size={input_size} | labels={self.labels}"
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """数值稳定的 softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        将 OpenCV BGR 图像预处理为 NCHW FP32 tensor，与 PyTorch 预处理对齐：
          transforms.Resize((224,224)) -> transforms.ToTensor() -> transforms.Normalize(mean, std)
        """
        # 1. Resize 到模型输入尺寸（ bilinear，与 torchvision Resize 一致）
        img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)

        # 2. BGR -> RGB（OpenCV 默认 BGR，PyTorch 训练数据为 RGB）
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. ToTensor：像素值从 [0, 255] 缩放到 [0, 1]
        img = img / 255.0

        # 4. Normalize：(x - mean) / std
        if self.mean is not None and self.std is not None:
            mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
            img = (img - mean) / std

        # 5. HWC -> CHW -> NCHW
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def predict(self, image: np.ndarray) -> Tuple[int, float, Dict[str, float]]:
        """
        对单张图像进行分类。

        :param image: BGR 格式 numpy 图像
        :return: (预测类别 ID, 置信度, 各类别概率字典)
        """
        input_tensor = self._preprocess(image)

        inputs = [httpclient.InferInput(self.input_name, input_tensor.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_tensor)
        outputs = [httpclient.InferRequestedOutput(self.output_name)]

        results = self.client.infer(self.model_name, inputs=inputs, outputs=outputs)
        output = results.as_numpy(self.output_name)

        # 兼容 [batch, num_classes] 和 [num_classes] 两种输出
        if output.ndim == 2:
            output = output[0]

        probs = self._softmax(output)
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        prob_dict = {
            label: float(probs[i])
            for i, label in enumerate(self.labels)
            if i < len(probs)
        }

        return class_id, confidence, prob_dict

    def close(self):
        """关闭 Triton 客户端连接"""
        try:
            self.client.close()
        except Exception as e:
            logger.debug(f"关闭分类客户端异常: {e}")
