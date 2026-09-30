"""紧凑 NV12 图像的校验与转换，不包含网络或设备依赖。"""

import cv2
import numpy as np


def nv12_image_shape(frame: np.ndarray) -> tuple[int, int]:
    """返回真实图像 (H, W)，而非 NV12 缓冲区的 (3H/2, W)。"""
    if frame.dtype != np.uint8 or not frame.size:
        raise ValueError("NV12 必须为非空 uint8 数组")
    if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[2] != 1):
        raise ValueError("NV12 必须为 [H*3/2,W] 或 [H*3/2,W,1]")
    rows, width = frame.shape[:2]
    if rows % 3 or width % 2:
        raise ValueError("NV12 图像宽高必须为偶数，缓冲区行数必须是 3 的倍数")
    return rows * 2 // 3, width


def bgr_to_nv12(frame: np.ndarray) -> np.ndarray:
    """BGR 图片兼容入口；生产 NV12 直通路径不调用此函数。"""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("输入必须为非空 uint8 BGR 图像")
    height, width = frame.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("转换 NV12 要求偶数宽高，请使用偶数尺寸图片或 BGR ensemble")
    i420 = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y_size = height * width
    chroma_size = y_size // 4
    nv12 = np.empty((height * 3 // 2, width), dtype=np.uint8)
    flat = nv12.reshape(-1)
    flat[:y_size] = i420[:y_size]
    flat[y_size::2] = i420[y_size:y_size + chroma_size]
    flat[y_size + 1::2] = i420[y_size + chroma_size:]
    return nv12
