"""
YOLO Triton 推理客户端（基于 triton_client 统一封装）

适配 triton-inference-server-ascend：BGR [H,W,3] 输入 IMAGE，或 NV12 [H*3/2,W,1] 输入 YUV，
大写 DETECTION_* 输出已经是原图坐标。backend="legacy" 显式保留旧服务契约。

底层通信统一使用 src/triton_client.TritonClient。为降低多线程/多模型场景下的
连接与文件描述符开销，同一线程、同一 URL+协议 的 TritonClient 实例会被复用。
"""

import sys
import threading
import time
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

from triton_client import TritonClient, TritonClientError
from utils.obj import Box
from utils.logger import setup_logger
from utils.image_formats import bgr_to_nv12, nv12_image_shape

logger = setup_logger("triton_client_fast")


_DEFAULT_COCO_LABEL_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
    71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

# Triton ensemble 后处理输出的默认 tensor 名
_DEFAULT_INPUT_NAME = "raw_image"
_DEFAULT_OUTPUT_NAMES = [
    "num_dets",
    "detection_boxes",
    "detection_scores",
    "detection_classes",
    "transform_metadata",
]

# SHM 输出缓冲区规格（max_dets=300，与 ../triton/model_repository/*/config.pbtxt 对齐）
_DEFAULT_OUTPUT_SPECS: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {
    "num_dets": ((1, 1), np.int32),
    "detection_boxes": ((1, 300, 4), np.float32),
    "detection_scores": ((1, 300), np.float32),
    "detection_classes": ((1, 300), np.int32),
    "transform_metadata": ((1, 6), np.float32),
}

_ASCEND_OUTPUT_NAMES = ["NUM_DETS", "DETECTION_BOXES", "DETECTION_SCORES", "DETECTION_CLASSES"]

# 线程本地 TritonClient 缓存：减少 HTTP/SHM 多模型场景下的连接开销。
# gRPC 客户端在 TritonClient 内部已按 URL 进程级共享，无需此处再缓存。
_thread_local = threading.local()


def _get_shared_triton_client(url: str, protocol: str) -> TritonClient:
    """获取同线程内复用的 HTTP/SHM TritonClient 实例。"""
    if protocol == "grpc":
        return TritonClient(url=url, protocol=protocol, verbose=False)
    if not hasattr(_thread_local, "clients"):
        _thread_local.clients = {}
    key = (url, protocol)
    client = _thread_local.clients.get(key)
    if client is None or getattr(client, "_closed", False):
        client = TritonClient(url=url, protocol=protocol, verbose=False)
        _thread_local.clients[key] = client
    return client


class YOLOTritonFast:
    """YOLO Triton 客户端：基于 triton_client.TritonClient 封装。"""

    def __init__(
        self,
        url: Optional[str] = None,
        model_name: str = "YOLO26_DET_PRE_ENSEMBLE",
        input_size: int = 640,
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.45,
        warmup: bool = True,
        label_map: Optional[Dict[int, str]] = None,
        input_name: Optional[str] = None,
        output_name: str = "output0",
        output_format: str = "yolo_v8_v11",
        use_shared_memory: bool = True,
        protocol: str = "grpc",
        backend: str = "ascend",
        max_detections: int = 300,
    ):
        url = url or ("localhost:54246" if protocol.lower() == "grpc" else "localhost:54245")
        self.url = url
        self.model_name = model_name
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        # iou_thresh / output_format 在此封装中不再使用，保留参数以兼容旧接口
        self.iou_thresh = iou_thresh
        self.output_format = output_format
        self.backend = backend.lower()
        if self.backend not in ("ascend", "legacy"):
            raise ValueError(f"不支持的 backend: {backend}")
        if max_detections <= 0:
            raise ValueError("max_detections 必须大于 0")
        self.max_detections = max_detections
        self.protocol = protocol.lower()
        if self.protocol == "shm" and not use_shared_memory:
            self.protocol = "http"
        if self.protocol not in ("grpc", "http", "shm"):
            raise ValueError(f"不支持的 protocol: {protocol}")

        # 兼容旧配置中传入的 images / output0：实际使用 ensemble 默认名
        if self.backend == "ascend":
            self.input_name = input_name or ("YUV" if model_name.endswith("_YUV_ENSEMBLE") else "IMAGE")
            if self.input_name not in ("IMAGE", "YUV"):
                raise ValueError("Ascend ensemble 输入必须为 IMAGE 或 YUV")
            self.output_names = list(_ASCEND_OUTPUT_NAMES)
        else:
            self.input_name = input_name if input_name not in (None, "images") else _DEFAULT_INPUT_NAME
            self.output_names = list(_DEFAULT_OUTPUT_NAMES)

        if label_map is None:
            self.label_map = dict(_DEFAULT_COCO_LABEL_MAP)
        else:
            self.label_map = {int(k): v for k, v in label_map.items()}
        self.num_classes = len(self.label_map)

        self._client: Optional[TritonClient] = None
        self._protocol = "http"
        self._output_specs: Optional[Dict[str, Tuple[Tuple[int, ...], np.dtype]]] = None
        self._closed = False

        # 复用同线程 HTTP/SHM 客户端，gRPC 在 TritonClient 内部已进程级共享
        self._client = _get_shared_triton_client(url, self.protocol)
        if not self._client.is_server_live():
            raise RuntimeError(f"Triton 服务未存活: {url}")
        if not self._client.is_model_ready(model_name):
            raise RuntimeError(f"模型 {model_name} 未就绪")

        self._protocol = self.protocol
        if self.protocol == "shm":
            try:
                self._setup_shared_memory()
                logger.info(f"YOLO Triton 客户端将使用共享内存协议: {url}")
            except Exception as e:
                logger.warning(f"共享内存初始化失败，回退到 HTTP: {e}")
                self._fallback_to_http()
        elif self.protocol == "http":
            logger.info(f"YOLO Triton 客户端将使用 HTTP 协议: {url}")
        elif self.protocol == "grpc":
            logger.info(f"YOLO Triton 客户端将使用 gRPC 协议: {url}")

        if warmup:
            try:
                self._warmup()
            except Exception as e:
                if self._protocol == "shm":
                    logger.warning(f"共享内存 Warmup 失败，回退到 HTTP: {e}")
                    self._fallback_to_http()
                    self._warmup()
                else:
                    raise

        logger.info(
            f"✅ YOLO Triton 客户端连接成功 | protocol={self._protocol}, "
            f"model={model_name}, input_size={input_size}"
        )

    # ---------- 共享内存 / 协议回退 ----------
    def _setup_shared_memory(self) -> None:
        """复用同线程 SHM 客户端并预置输出缓冲区规格。"""
        self._client = _get_shared_triton_client(self.url, "shm")
        if self.backend == "ascend":
            self._output_specs = {
                "NUM_DETS": ((1,), np.int32),
                "DETECTION_BOXES": ((self.max_detections, 4), np.float32),
                "DETECTION_SCORES": ((self.max_detections,), np.float32),
                "DETECTION_CLASSES": ((self.max_detections,), np.int32),
            }
        else:
            self._output_specs = dict(_DEFAULT_OUTPUT_SPECS)

    def _fallback_to_http(self) -> None:
        """回退到 HTTP 协议（复用同线程共享客户端）。"""
        self._protocol = "http"
        self._output_specs = None
        self._client = _get_shared_triton_client(self.url, "http")

    def _warmup(self) -> None:
        """使用一张与模型输入尺寸一致的黑图预热推理。"""
        logger.info("Warmup 中...")
        if self.input_name == "YUV":
            dummy = np.full((self.input_size * 3 // 2, self.input_size), 128, dtype=np.uint8)
            dummy[:self.input_size] = 16
            self.predict_nv12(dummy)
        else:
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.predict(dummy, classes=None)
        logger.info("Warmup 完成")

    # ---------- 生命周期 ----------
    def close(self) -> None:
        """
        标记当前 YOLO 客户端为已关闭。

        注意：底层 TritonClient 在同一线程内被多个模型实例复用，因此此处不关闭
        底层连接，避免影响同线程其他模型。线程退出时由 Python 垃圾回收自动释放。
        """
        self._closed = True

    def __del__(self):
        # 不关闭共享客户端，避免影响同线程其他实例
        pass

    # ---------- 公共流程 ----------
    def predict(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Box]:
        """BGR 图片入口；YUV 模型仅在此兼容入口转换为 NV12。"""
        t0 = time.time()
        input_arr = self._prepare_input(frame)
        return self._predict_prepared(input_arr, frame.shape[:2], classes, t0)

    def predict_nv12(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Box]:
        """原始 NV12 直通；只补通道维度，不转 BGR、不做颜色变换。"""
        if self.backend != "ascend" or self.input_name != "YUV":
            raise ValueError("predict_nv12 需要 Ascend YUV ensemble")
        t0 = time.time()
        image_shape = nv12_image_shape(frame)
        packed = frame[..., None] if frame.ndim == 2 else frame
        return self._predict_prepared(np.ascontiguousarray(packed), image_shape, classes, t0)

    def _predict_prepared(self, input_arr, frame_shape, classes, t0) -> List[Box]:
        t1 = time.time()

        result = self._infer(input_arr)
        t2 = time.time()

        if self.backend == "ascend":
            boxes = self._parse_ascend_outputs(result, frame_shape, classes)
        else:
            transform = self._extract_transform(result.get("transform_metadata"))
            boxes = self._parse_outputs(result, frame_shape, classes, transform)
        t3 = time.time()

        filter_info = f" 类别过滤={classes}" if classes else ""
        logger.debug(
            f"耗时 | 预处理={(t1 - t0) * 1000:.1f}ms "
            f"推理={(t2 - t1) * 1000:.1f}ms "
            f"后处理={(t3 - t2) * 1000:.1f}ms | 框数={len(boxes)}{filter_info}"
        )
        return boxes

    # ---------- 输入构造 ----------
    def _prepare_input(self, frame: np.ndarray) -> np.ndarray:
        """Ascend 直接传 BGR HWC；旧服务保持 RGB NHWC。"""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"输入必须是 HWC BGR 图像，当前 shape={frame.shape}")
        if frame.dtype != np.uint8 or not frame.size:
            raise ValueError("输入必须是非空 uint8 BGR 图像")
        if self.backend == "ascend":
            if self.input_name == "YUV":
                return bgr_to_nv12(frame)[..., None]
            return np.ascontiguousarray(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb[np.newaxis, ...].astype(np.uint8, copy=False)

    # ---------- 推理 ----------
    def _infer(self, input_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """调用 TritonClient 完成推理，SHM 失败时一次性回退 HTTP。"""
        if self._client is None or self._closed:
            raise RuntimeError("Triton 客户端已关闭")

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

    # ---------- 输出解析与坐标映射 ----------
    def _parse_ascend_outputs(self, result, frame_shape, classes) -> List[Box]:
        """新 ensemble 已还原坐标，不请求或解释 TRANSFORM_METADATA。"""
        count_array = np.asarray(result["NUM_DETS"])
        if count_array.shape != (1,) or not np.issubdtype(count_array.dtype, np.integer):
            raise ValueError("NUM_DETS 必须为整数 [1]")
        count = int(count_array[0])
        boxes = np.asarray(result["DETECTION_BOXES"])
        scores = np.asarray(result["DETECTION_SCORES"])
        ids = np.asarray(result["DETECTION_CLASSES"])
        if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1 or ids.ndim != 1:
            raise ValueError("Ascend 检测输出必须为 [N,4] / [N] / [N]，不含 batch 维度")
        if count < 0 or count > min(len(boxes), len(scores), len(ids)):
            raise ValueError("NUM_DETS 超出输出张量范围")
        boxes = boxes[:count].astype(np.float32, copy=True)
        scores, ids = scores[:count], ids[:count]
        valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & np.isfinite(ids)
        valid &= (scores >= self.conf_thresh) & (ids >= 0) & (ids == np.floor(ids))
        if classes is not None:
            valid &= np.isin(ids, classes)
        height, width = frame_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return self._to_boxes(boxes[valid], scores[valid], ids[valid])

    @staticmethod
    def _extract_transform(transform_metadata: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """将 Triton 返回的 [1,6] transform_metadata 转成 2x3 仿射矩阵。"""
        if transform_metadata is None:
            return None
        try:
            flat = np.asarray(transform_metadata).reshape(-1)
            if flat.shape[0] != 6:
                logger.warning(f"transform_metadata 长度不是 6: {flat.shape}，将使用自计算映射")
                return None
            return flat.reshape(2, 3).astype(np.float64)
        except Exception as e:
            logger.warning(f"解析 transform_metadata 失败: {e}，将使用自计算映射")
            return None

    def _parse_outputs(
        self,
        result: Dict[str, np.ndarray],
        frame_shape: Tuple[int, int],
        classes: Optional[List[int]],
        transform: Optional[np.ndarray],
    ) -> List[Box]:
        """从 ensemble 输出中提取检测结果并映射回原图坐标。"""
        num_dets = int(result["num_dets"].flat[0])
        if num_dets <= 0:
            return []

        boxes_xyxy = result["detection_boxes"][0, :num_dets]  # [N, 4]
        scores = result["detection_scores"][0, :num_dets]     # [N]
        cls_ids = result["detection_classes"][0, :num_dets]   # [N]

        # 置信度过滤（后端可能已过滤，这里再做一次应用层过滤）
        conf_mask = scores >= self.conf_thresh
        if not conf_mask.any():
            return []
        boxes_xyxy = boxes_xyxy[conf_mask]
        scores = scores[conf_mask]
        cls_ids = cls_ids[conf_mask]

        # 类别过滤
        if classes:
            cls_mask = np.isin(cls_ids, list(classes))
            if not cls_mask.any():
                return []
            boxes_xyxy = boxes_xyxy[cls_mask]
            scores = scores[cls_mask]
            cls_ids = cls_ids[cls_mask]

        # 坐标映射：优先使用 Triton 返回的仿射矩阵
        boxes_xyxy = self._map_boxes_to_original(boxes_xyxy, frame_shape, transform)
        return self._to_boxes(boxes_xyxy, scores, cls_ids)

    def _map_boxes_to_original(
        self,
        boxes_xyxy: np.ndarray,
        frame_shape: Tuple[int, int],
        transform: Optional[np.ndarray],
    ) -> np.ndarray:
        """将检测框从模型输入空间映射回原图。优先使用 Triton 返回的 2x3 仿射矩阵。"""
        orig_h, orig_w = frame_shape
        if orig_h == 0 or orig_w == 0:
            return boxes_xyxy

        if transform is not None:
            # 对 xyxy 的四个角点应用仿射矩阵
            N = boxes_xyxy.shape[0]
            corners = boxes_xyxy.reshape(-1, 2)  # [N*4, 2]
            mapped = (transform[:, :2] @ corners.T).T + transform[:, 2]
            mapped = mapped.reshape(N, 4)
        else:
            # 无 transform_metadata 时回退到自计算 letterbox 映射
            scale = self.input_size / max(orig_h, orig_w)
            new_w, new_h = orig_w * scale, orig_h * scale
            pad_x = (self.input_size - new_w) / 2.0
            pad_y = (self.input_size - new_h) / 2.0

            mapped = boxes_xyxy.copy()
            mapped[:, [0, 2]] = (mapped[:, [0, 2]] - pad_x) / scale
            mapped[:, [1, 3]] = (mapped[:, [1, 3]] - pad_y) / scale

        # 裁剪到原图边界
        mapped[:, 0] = np.clip(mapped[:, 0], 0, orig_w)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, orig_h)
        mapped[:, 2] = np.clip(mapped[:, 2], 0, orig_w)
        mapped[:, 3] = np.clip(mapped[:, 3], 0, orig_h)
        return mapped

    # ---------- 构造 Box 列表 ----------
    def _to_boxes(self, boxes_xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray) -> List[Box]:
        result = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            label = self.label_map.get(int(cls_ids[i]), f"class_{int(cls_ids[i])}")
            result.append(Box(
                label=label,
                score=float(scores[i]),
                box=[max(0.0, x1), max(0.0, y1), max(0.0, x2), max(0.0, y2)],
                mask=None,
                source="YOLO",
            ))
        return result
