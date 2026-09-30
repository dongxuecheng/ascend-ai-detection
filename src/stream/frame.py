"""单帧格式视图：按需转换并缓存，不跨帧保留图像。"""

import numpy as np

from stream.remote_capture import PIXEL_BGR, PIXEL_NV12, PIXEL_I420, PIXEL_YUYV422, frame_to_bgr
from utils.image_formats import bgr_to_nv12, nv12_image_shape


class FrameImages:
    def __init__(self, raw: np.ndarray, pixel_format: int, meta: dict | None = None):
        self.raw = raw
        self.pixel_format = pixel_format
        if pixel_format in (PIXEL_NV12, PIXEL_I420):
            self.height, self.width = nv12_image_shape(raw)
        elif pixel_format == PIXEL_BGR:
            if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3 or not raw.size:
                raise ValueError("无效 BGR 帧")
            self.height, self.width = raw.shape[:2]
        elif pixel_format == PIXEL_YUYV422:
            if raw.dtype != np.uint8 or raw.ndim != 2 or not raw.size or raw.shape[1] % 4:
                raise ValueError("无效 YUYV422 帧")
            self.height, self.width = raw.shape[0], raw.shape[1] // 2
        else:
            raise ValueError(f"未知像素格式: {pixel_format}")
        if meta and (("w" in meta and meta["w"] != self.width) or
                     ("h" in meta and meta["h"] != self.height)):
            raise ValueError("帧缓冲区尺寸与 SHM 元数据不一致")
        self._bgr = raw if pixel_format == PIXEL_BGR else None
        self._nv12 = raw if pixel_format == PIXEL_NV12 else None

    def bgr(self) -> np.ndarray:
        if self._bgr is None:
            self._bgr = frame_to_bgr(self.raw, self.pixel_format)
        return self._bgr

    def nv12(self) -> np.ndarray:
        if self._nv12 is None:
            self._nv12 = bgr_to_nv12(self.bgr())
        return self._nv12
