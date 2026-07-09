"""
YOLO Triton 高性能推理客户端（仿射变换版）

支持 System Shared Memory 上传输入/下载输出：当 Python 客户端与 Triton Server 部署在同一台机器时，
通过 POSIX 共享内存传输输入/输出 tensor，避免在 HTTP body 中重复拷贝大图像数据。
如果共享内存初始化失败，会自动回退到普通的 HTTP numpy 传输。
"""

import os
import sys
import re
import mmap
import atexit
import socket
import threading
import time
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import tritonclient.http as httpclient

try:
    import posix_ipc
    _HAS_POSIX_IPC = True
except ImportError:
    _HAS_POSIX_IPC = False

from utils.obj import Box
from utils.logger import setup_logger

logger = setup_logger("triton_client_fast")

# 共享内存文件保留的最长时间（秒）：超过此时间且无法确认归属进程存活的文件会被清理
_SHM_MAX_AGE_SEC = 600
# 后台清理线程执行间隔（秒）
_SHM_CLEANUP_INTERVAL_SEC = 300


def _get_hostname_short() -> str:
    """获取当前主机/容器标识（Docker 默认 hostname 即容器 ID 前 12 位）"""
    hostname = socket.gethostname()
    sanitized = re.sub(r"[^a-zA-Z0-9]", "", hostname).lower()
    return sanitized[:12] if sanitized else "unknown"


def _parse_shm_filename(name: str, prefix: str):
    """
    解析共享内存文件名，返回 (hostname, pid) 或 (None, None)。
    支持新格式：aidet_model_input_hostname_pid_tid
    兼容旧格式：aidet_model_input_pid_tid

    对于不符合上述格式或 pid 非数字的遗留文件，返回 (None, None)，由调用方按
    跨容器/旧格式策略兜底清理，避免因格式异常导致客户端创建失败。
    """
    rest = name[len(prefix):]
    parts = rest.split("_")
    try:
        if len(parts) == 3:
            return parts[0], int(parts[1])
        elif len(parts) == 2:
            return None, int(parts[0])
    except ValueError:
        pass
    return None, None


def _is_pid_alive(pid: int) -> bool:
    """检查当前 PID 命名空间中进程是否存活"""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _cleanup_all_aidet_shm(max_age_sec: int = _SHM_MAX_AGE_SEC):
    """
    清理 /dev/shm 下所有 aidet_* 共享内存文件。
    策略：
      - 同 hostname 且 PID 不存在的文件删除；
      - 不同 hostname 或旧格式文件，按文件修改时间超过 max_age_sec 删除。
    """
    if not _HAS_POSIX_IPC:
        return
    shm_dir = "/dev/shm"
    if not os.path.isdir(shm_dir):
        return

    my_hostname = _get_hostname_short()
    now = time.time()
    prefixes = ("aidet_",)
    cleaned = 0

    for name in os.listdir(shm_dir):
        if not any(name.startswith(p) for p in prefixes):
            continue
        if "_input_" not in name and "_output_" not in name:
            continue

        # 定位前缀，例如 aidet_yolo11_plan_input_
        for sep in ("_input_", "_output_"):
            idx = name.find(sep)
            if idx == -1:
                continue
            prefix = name[: idx + len(sep)]
            file_hostname, pid = _parse_shm_filename(name, prefix)
            break
        else:
            continue

        should_delete = False
        if file_hostname is None or file_hostname != my_hostname:
            # 跨容器/旧格式：按文件年龄清理
            try:
                st = os.stat(os.path.join(shm_dir, name))
                if now - st.st_mtime > max_age_sec:
                    should_delete = True
            except Exception:
                continue
        else:
            # 同容器：按 PID 存活判断
            if not _is_pid_alive(pid):
                should_delete = True

        if should_delete:
            try:
                posix_ipc.unlink_shared_memory(f"/{name}")
                cleaned += 1
                logger.info(f"清理遗留 Triton 共享内存: /dev/shm/{name}")
            except posix_ipc.ExistentialError:
                pass
            except Exception as e:
                logger.debug(f"清理遗留共享内存失败 {name}: {e}")

    if cleaned > 0:
        logger.info(f"共清理 {cleaned} 个遗留 Triton 共享内存文件")


def _periodic_shm_cleanup():
    """后台守护线程：定期扫描并清理遗留共享内存"""
    while True:
        time.sleep(_SHM_CLEANUP_INTERVAL_SEC)
        try:
            _cleanup_all_aidet_shm()
        except Exception as e:
            logger.debug(f"后台共享内存清理异常: {e}")


# 模块加载时启动后台清理线程（仅当 posix_ipc 可用时）
if _HAS_POSIX_IPC:
    _shm_cleanup_thread = threading.Thread(target=_periodic_shm_cleanup, daemon=True)
    _shm_cleanup_thread.start()


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

# Triton 数据类型 -> numpy 数据类型
_TRITON_DTYPE_MAP = {
    "FP32": np.float32,
    "FP16": np.float16,
    "FP64": np.float64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "BOOL": np.bool_,
}


class YOLOTritonFast:
    """高性能 YOLO Triton 客户端（仿射变换 + 可选共享内存输入/输出）"""

    def __init__(
        self,
        url: str = "localhost:38000",
        model_name: str = "yolo11_plan",
        input_size: int = 640,
        conf_thresh: float = 0.3,
        iou_thresh: float = 0.45,
        warmup: bool = True,
        label_map: Optional[Dict[int, str]] = None,
        input_name: str = "images",
        output_name: str = "output0",
        output_format: str = "yolo_v8_v11",
        use_shared_memory: bool = True,
    ):
        self.url = url
        self.model_name = model_name
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_name = input_name
        self.output_name = output_name
        self.output_format = output_format
        self.use_shared_memory = use_shared_memory

        if label_map is None:
            self.label_map = dict(_DEFAULT_COCO_LABEL_MAP)
        else:
            self.label_map = {int(k): v for k, v in label_map.items()}
        self.num_classes = len(self.label_map)

        # 输入共享内存相关状态
        self._shm_input_region_name: Optional[str] = None
        self._shm_input_size: int = 0
        self._shm_input_mmap: Optional[mmap.mmap] = None
        self._shm_input_registered: bool = False

        # 输出共享内存相关状态
        self._shm_output_region_name: Optional[str] = None
        self._shm_output_size: int = 0
        self._shm_output_mmap: Optional[mmap.mmap] = None
        self._shm_output_registered: bool = False
        self._shm_output_shape: Optional[Tuple[int, ...]] = None
        self._shm_output_dtype: Optional[np.dtype] = None

        # 关闭状态标志，防止 close() 被重复调用/重复打印日志
        self._closed = False

        self.client = httpclient.InferenceServerClient(url=url, verbose=False, concurrency=1)
        if not self.client.is_server_live():
            raise RuntimeError(f"Triton 服务未存活: {url}")
        if not self.client.is_model_ready(model_name):
            raise RuntimeError(f"模型 {model_name} 未就绪")

        # 启动时清理本模型遗留的共享内存（异常退出、kill -9 等场景）
        if self.use_shared_memory:
            self._cleanup_stale_shared_memory()

        # 初始化共享内存（失败则回退 HTTP）
        if self.use_shared_memory:
            self._setup_input_shared_memory()
            self._setup_output_shared_memory()
        else:
            logger.info("共享内存已显式关闭，将使用 HTTP numpy 传输")

        # 注册进程退出时的兜底清理，避免正常停止/重启后共享内存残留
        self._atexit_handle = atexit.register(self.close)

        if warmup:
            logger.info("Warmup 中...")
            dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
            for _ in range(3):
                self._infer_raw(dummy)
            logger.info("✅ Triton 高性能客户端连接成功")

    # ---------- 共享内存管理 ----------
    def _cleanup_stale_shared_memory(self):
        """
        启动时清理遗留的 Triton 共享内存文件。
        由模块级 _cleanup_all_aidet_shm 统一处理，支持跨容器/按年龄清理。
        """
        _cleanup_all_aidet_shm()

    def _setup_input_shared_memory(self):
        """创建 POSIX 共享内存并在 Triton Server 注册输入区域"""
        if not _HAS_POSIX_IPC:
            logger.warning("未安装 posix_ipc，无法使用输入共享内存，回退到 HTTP")
            return

        input_shape = (1, 3, self.input_size, self.input_size)
        byte_size = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize

        region_name = f"aidet_{self.model_name}_input_{_get_hostname_short()}_{os.getpid()}_{threading.current_thread().ident}"
        shm_name = f"/{region_name}"

        try:
            try:
                posix_ipc.unlink_shared_memory(shm_name)
            except posix_ipc.ExistentialError:
                pass

            shm = posix_ipc.SharedMemory(shm_name, posix_ipc.O_CREAT, size=byte_size)
            fd = shm.fd
            mm = mmap.mmap(fd, byte_size)
            os.close(fd)

            self.client.register_system_shared_memory(region_name, region_name, byte_size)

            self._shm_input_region_name = region_name
            self._shm_input_size = byte_size
            self._shm_input_mmap = mm
            self._shm_input_registered = True

            logger.info(
                f"输入共享内存已创建并注册 | region={region_name}, "
                f"size={byte_size / 1024 / 1024:.2f}MB, shape={input_shape}"
            )
        except Exception as e:
            logger.warning(f"输入共享内存初始化失败，回退到 HTTP: {e}")
            self._release_input_shared_memory()

    def _setup_output_shared_memory(self):
        """查询模型元数据/配置，创建并注册输出共享内存区域"""
        if not _HAS_POSIX_IPC:
            logger.warning("未安装 posix_ipc，无法使用输出共享内存，回退到 HTTP")
            return

        def _triton_dtype_to_np(dtype_str: str) -> Optional[np.dtype]:
            # 兼容 "FP32" 和 "TYPE_FP32" 两种形式
            return _TRITON_DTYPE_MAP.get(dtype_str) or _TRITON_DTYPE_MAP.get(dtype_str.replace("TYPE_", "", 1))

        try:
            metadata = self.client.get_model_metadata(self.model_name)
            output_spec = None
            for out in metadata.get("outputs", []):
                if out.get("name") == self.output_name:
                    output_spec = out
                    break

            if output_spec is None:
                logger.warning(f"模型元数据中未找到输出 {self.output_name}，输出共享内存不可用")
                return

            shape = tuple(int(d) for d in output_spec.get("shape", []))
            dtype_str = output_spec.get("datatype", "FP32")

            # metadata 里 batch 维度可能为 -1，此时从 model config 取真实 dims + max_batch_size
            if not shape or any(d <= 0 for d in shape):
                try:
                    model_cfg = self.client.get_model_config(self.model_name)
                    max_batch = int(model_cfg.get("max_batch_size", 0))
                    for out in model_cfg.get("output", []):
                        if out.get("name") == self.output_name:
                            dims = [int(d) for d in out.get("dims", [])]
                            shape = tuple(dims)
                            dtype_str = out.get("data_type", dtype_str)
                            break
                except Exception as e:
                    logger.warning(f"从 model config 获取输出 shape 失败: {e}")

            if not shape or any(d <= 0 for d in shape):
                logger.warning(f"输出 shape 非法: {shape}，输出共享内存不可用")
                return

            dtype = _triton_dtype_to_np(dtype_str)
            if dtype is None:
                logger.warning(f"不支持的输出数据类型: {dtype_str}，输出共享内存不可用")
                return
            byte_size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            region_name = f"aidet_{self.model_name}_output_{_get_hostname_short()}_{os.getpid()}_{threading.current_thread().ident}"
            shm_name = f"/{region_name}"

            try:
                posix_ipc.unlink_shared_memory(shm_name)
            except posix_ipc.ExistentialError:
                pass

            shm = posix_ipc.SharedMemory(shm_name, posix_ipc.O_CREAT, size=byte_size)
            fd = shm.fd
            mm = mmap.mmap(fd, byte_size)
            os.close(fd)

            self.client.register_system_shared_memory(region_name, region_name, byte_size)

            self._shm_output_region_name = region_name
            self._shm_output_size = byte_size
            self._shm_output_mmap = mm
            self._shm_output_shape = shape
            self._shm_output_dtype = dtype
            self._shm_output_registered = True

            logger.info(
                f"输出共享内存已创建并注册 | region={region_name}, "
                f"size={byte_size / 1024:.2f}KB, shape={shape}, dtype={dtype_str}"
            )
        except Exception as e:
            logger.warning(f"输出共享内存初始化失败，回退到 HTTP: {e}")
            self._release_output_shared_memory()

    def _release_input_shared_memory(self):
        """释放输入共享内存资源（幂等）"""
        if self._shm_input_registered and self._shm_input_region_name:
            try:
                self.client.unregister_system_shared_memory(self._shm_input_region_name)
            except Exception:
                pass
            self._shm_input_registered = False

        if self._shm_input_mmap is not None:
            try:
                self._shm_input_mmap.close()
            except Exception:
                pass
            self._shm_input_mmap = None

        if self._shm_input_region_name:
            try:
                posix_ipc.unlink_shared_memory(f"/{self._shm_input_region_name}")
            except Exception:
                pass
            self._shm_input_region_name = None
        self._shm_input_size = 0

    def _release_output_shared_memory(self):
        """释放输出共享内存资源（幂等）"""
        if self._shm_output_registered and self._shm_output_region_name:
            try:
                self.client.unregister_system_shared_memory(self._shm_output_region_name)
            except Exception:
                pass
            self._shm_output_registered = False

        if self._shm_output_mmap is not None:
            try:
                self._shm_output_mmap.close()
            except Exception:
                pass
            self._shm_output_mmap = None

        if self._shm_output_region_name:
            try:
                posix_ipc.unlink_shared_memory(f"/{self._shm_output_region_name}")
            except Exception:
                pass
            self._shm_output_region_name = None

        self._shm_output_size = 0
        self._shm_output_shape = None
        self._shm_output_dtype = None

    def close(self):
        """释放客户端占用的资源，包括共享内存（幂等）"""
        if self._closed:
            return
        self._closed = True

        # 注销进程退出兜底，避免 close() 后被重复执行
        if hasattr(self, "_atexit_handle"):
            try:
                atexit.unregister(self._atexit_handle)
            except Exception:
                pass
            delattr(self, "_atexit_handle")

        self._release_input_shared_memory()
        self._release_output_shared_memory()

        if self.client is not None:
            try:
                self.client.close()
            except Exception as e:
                logger.debug(f"关闭 Triton 客户端时异常: {e}")
            self.client = None

        logger.info("YOLOTritonFast 客户端已关闭")

    def __del__(self):
        # 解释器关闭期间，self.client 等依赖可能已被部分回收，
        # 此时做 HTTP/共享内存清理容易引发 segfault，直接跳过。
        if sys.is_finalizing() or self._closed:
            return
        self.close()

    # ---------- 公共流程 ----------
    def predict(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Box]:
        """推理单帧：预处理 → 推理 → 解码 → 仿射逆变换 → 构造 Box"""
        t0 = time.time()
        input_tensor, M_inv = self._preprocess(frame)
        t1 = time.time()

        output = self._infer_raw(input_tensor)
        t2 = time.time()

        # 解码：从模型输出提取 xyxy（input_size 空间）+ scores + cls_ids
        if self.output_format == "yolo_v12_end2end":
            boxes_xyxy, scores, cls_ids = self._decode_v12(output, classes)
        elif self.output_format == "yolo_v5":
            boxes_xyxy, scores, cls_ids = self._decode_v5(output, classes)
        else:
            boxes_xyxy, scores, cls_ids = self._decode_v11(output, classes)

        # 仿射逆变换：input_size 空间 → 原图
        boxes_xyxy = self._affine_inv_boxes(boxes_xyxy, M_inv)
        boxes = self._to_boxes(boxes_xyxy, scores, cls_ids)
        t3 = time.time()

        filter_info = f" 类别过滤={classes}" if classes else ""
        logger.debug(
            f"耗时 | 预处理={(t1 - t0) * 1000:.1f}ms "
            f"推理={(t2 - t1) * 1000:.1f}ms "
            f"后处理={(t3 - t2) * 1000:.1f}ms | 框数={len(boxes)}{filter_info}"
        )
        return boxes

    # ---------- 预处理：仿射变换 ----------
    def _get_affine_matrix(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算原图 → input_size×input_size 的仿射矩阵及其逆矩阵。
        采用标准 letterbox（等比缩放 + 居中填充）。
        """
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        new_w, new_h = w * scale, h * scale
        dx = (self.input_size - new_w) / 2
        dy = (self.input_size - new_h) / 2
        M = np.array([[scale, 0, dx], [0, scale, dy]], dtype=np.float64)
        M_inv = cv2.invertAffineTransform(M)
        return M, M_inv

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """仿射变换预处理，返回 (tensor, M_inv)"""
        M, M_inv = self._get_affine_matrix(img)
        warped = cv2.warpAffine(
            img, M, (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )
        warped = warped[:, :, ::-1].astype(np.float32, copy=False) / 255.0
        chw = np.ascontiguousarray(np.transpose(warped, (2, 0, 1)))
        return np.expand_dims(chw, axis=0), M_inv

    # ---------- 推理 ----------
    def _infer_raw(self, input_tensor: np.ndarray) -> np.ndarray:
        inputs = [httpclient.InferInput(self.input_name, input_tensor.shape, "FP32")]

        # 输入共享内存
        if self._shm_input_registered and self._shm_input_mmap is not None:
            self._shm_input_mmap.seek(0)
            self._shm_input_mmap.write(input_tensor.tobytes())
            self._shm_input_mmap.flush()
            inputs[0].set_shared_memory(self._shm_input_region_name, self._shm_input_size)
        else:
            inputs[0].set_data_from_numpy(input_tensor)

        # 输出共享内存
        outputs = [httpclient.InferRequestedOutput(self.output_name)]
        if self._shm_output_registered:
            outputs[0].set_shared_memory(self._shm_output_region_name, self._shm_output_size)

        results = self.client.infer(self.model_name, inputs=inputs, outputs=outputs)

        # 从共享内存读取输出，或从 HTTP 响应读取
        if self._shm_output_registered and self._shm_output_mmap is not None:
            total_elements = int(np.prod(self._shm_output_shape))
            output = np.frombuffer(
                self._shm_output_mmap,
                dtype=self._shm_output_dtype,
                count=total_elements,
            ).reshape(self._shm_output_shape)
            # 去掉 batch 维度，保持与原来 as_numpy()[0] 一致
            if output.shape[0] == 1:
                output = output[0]
            return output
        else:
            return results.as_numpy(self.output_name)[0]

    # ---------- 解码：yolo_v5 ----------
    def _decode_v5(self, output: np.ndarray, classes: Optional[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        YOLOv5 输出解码。
        输出形状: [num_anchors, 5 + num_classes]
          - [:4]   框中心 xywh
          - [4]    objectness
          - [5:]   各类别条件概率
        最终分数 = objectness * max(class_score)
        """
        preds = output                                            # [25200, 9]
        boxes_xywh = preds[:, :4]
        obj_scores = preds[:, 4]
        cls_scores = preds[:, 5:5 + self.num_classes]

        scores = obj_scores * cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)

        mask = scores >= self.conf_thresh
        if not mask.any():
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
        boxes_xywh, scores, cls_ids = boxes_xywh[mask], scores[mask], cls_ids[mask]

        if classes:
            cls_mask = np.isin(cls_ids, list(classes))
            if not cls_mask.any():
                return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
            boxes_xywh, scores, cls_ids = boxes_xywh[cls_mask], scores[cls_mask], cls_ids[cls_mask]

        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        boxes_xyxy = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)

        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(),
                                   self.conf_thresh, self.iou_thresh)
        if len(indices) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
        indices = indices.flatten()
        return boxes_xyxy[indices], scores[indices], cls_ids[indices]

    # ---------- 解码：yolo_v8_v11 ----------
    def _decode_v11(self, output: np.ndarray, classes: Optional[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = np.transpose(output)                       # [8400, N+4]
        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4:4 + self.num_classes]
        scores = cls_scores.max(axis=1)
        cls_ids = cls_scores.argmax(axis=1)

        mask = scores >= self.conf_thresh
        if not mask.any():
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
        boxes_xywh, scores, cls_ids = boxes_xywh[mask], scores[mask], cls_ids[mask]

        if classes:
            cls_mask = np.isin(cls_ids, list(classes))
            if not cls_mask.any():
                return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
            boxes_xywh, scores, cls_ids = boxes_xywh[cls_mask], scores[cls_mask], cls_ids[cls_mask]

        x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        boxes_xyxy = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)

        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(),
                                   self.conf_thresh, self.iou_thresh)
        if len(indices) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
        indices = indices.flatten()
        return boxes_xyxy[indices], scores[indices], cls_ids[indices]

    # ---------- 解码：yolo_v12_end2end ----------
    def _decode_v12(self, output: np.ndarray, classes: Optional[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds = output[0] if output.ndim == 3 else output   # [300, 6]
        scores = preds[:, 4]
        mask = scores >= self.conf_thresh
        if not mask.any():
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)

        boxes_xyxy, scores, cls_ids = preds[mask, :4], scores[mask], preds[mask, 5].astype(np.int32)

        if classes:
            cls_mask = np.isin(cls_ids, list(classes))
            if not cls_mask.any():
                return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=np.int32)
            boxes_xyxy, scores, cls_ids = boxes_xyxy[cls_mask], scores[cls_mask], cls_ids[cls_mask]

        return boxes_xyxy, scores, cls_ids

    # ---------- 坐标映射：仿射逆变换 ----------
    @staticmethod
    def _affine_inv_boxes(boxes_xyxy: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
        """将 [N,4] xyxy 框从 input_size 空间映射回原图"""
        N = len(boxes_xyxy)
        pts = boxes_xyxy.reshape(-1, 2)          # [N*2, 2]
        pts = np.hstack([pts, np.ones((N * 2, 1))])  # [N*2, 3]
        mapped = (M_inv @ pts.T).T               # [N*2, 2]
        return mapped.reshape(N, 4)

    # ---------- 构造 Box 列表 ----------
    def _to_boxes(self, boxes_xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray) -> List[Box]:
        result = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            label = self.label_map.get(int(cls_ids[i]), f"class_{int(cls_ids[i])}")
            result.append(Box(
                label=label, score=float(scores[i]),
                box=[max(0, x1), max(0, y1), max(0, x2), max(0, y2)],
                mask=None,
            ))
        return result
