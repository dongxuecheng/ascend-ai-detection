# -*- coding: utf-8 -*-
"""
统一跟踪器接口

支持通过配置切换不同的跟踪算法（ByteTrack / OCSort）。
对外提供统一的 Track 数据结构和 TrackerBase.update() 接口，
底层自动适配各跟踪器的差异。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Track:
    """统一跟踪结果，兼容 counting.counter 中的使用方式。"""
    track_id: int
    tlwh: np.ndarray       # [top_left_x, top_left_y, width, height]
    score: float = 0.0
    is_activated: bool = True


class TrackerBase:
    """跟踪器统一接口基类"""

    def update(self, detections: np.ndarray, frame_shape: Optional[Tuple[int, ...]] = None) -> List[Track]:
        """
        根据检测结果更新跟踪状态。

        :param detections: numpy array, shape (N, 6) 为 [[x1,y1,x2,y2,score,class_id], ...]
                           或 shape (N, 5) 为 [[x1,y1,x2,y2,score], ...]
        :param frame_shape: 当前帧的形状，如 (H, W, 3)。OCSort 等需要原始图像尺寸做坐标变换。
        :return: list[Track]
        """
        raise NotImplementedError

    def reset(self):
        """重置跟踪器状态（可选，默认不做任何事）。"""
        pass


class ByteTrackerWrapper(TrackerBase):
    """ByteTrack 包装器"""

    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 30):
        from obj_track.byte_track.byte_tracker import BYTETracker

        class _Args:
            def __init__(self):
                self.track_thresh = track_thresh
                self.track_buffer = track_buffer
                self.match_thresh = match_thresh
                self.mot20 = False
                self.aspect_ratio_thresh = 1.6
                self.min_box_area = 10

        self._tracker = BYTETracker(_Args(), frame_rate=frame_rate)

    def update(self, detections: np.ndarray, frame_shape: Optional[Tuple[int, ...]] = None) -> List[Track]:
        stracks = self._tracker.update(detections)
        return [
            Track(
                track_id=int(t.track_id),
                tlwh=np.asarray(t.tlwh, dtype=np.float32),
                score=float(getattr(t, "score", 0.0)),
                is_activated=bool(getattr(t, "is_activated", True)),
            )
            for t in stracks
        ]


class OCSortWrapper(TrackerBase):
    """OCSort 包装器"""

    def __init__(self, det_thresh: float = 0.5, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3, delta_t: int = 3, asso_func: str = "iou",
                 inertia: float = 0.2, use_byte: bool = False):
        import sys
        import os
        # 保证 oc_sort 子目录在 sys.path 中，使其内部相对导入正常工作
        oc_sort_path = os.path.join(os.path.dirname(__file__), "oc_sort")
        if oc_sort_path not in sys.path:
            sys.path.insert(0, oc_sort_path)

        from obj_track.oc_sort.ocsort import OCSort
        self._tracker = OCSort(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte,
        )

    def update(self, detections: np.ndarray, frame_shape: Optional[Tuple[int, ...]] = None) -> List[Track]:
        if frame_shape is None:
            raise ValueError("OCSortWrapper.update() requires frame_shape (e.g., (H, W, 3))")

        img_h, img_w = frame_shape[:2]

        # OCSort 要求每帧都调用，空帧传 np.empty((0, 5))
        arr = np.asarray(detections) if detections is not None else np.empty((0,))
        if arr.size == 0:
            dets = np.empty((0, 5))
        else:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # 只取前 5 列 [x1, y1, x2, y2, score]，避免触发 ocsort 里 .cpu().numpy() 分支
            dets = arr[:, :5].astype(np.float32)

        # 检测器已输出原始图像坐标，因此 img_size = img_info 使 scale=1，不做额外缩放
        results = self._tracker.update(dets, (img_h, img_w), (img_h, img_w))

        tracks: List[Track] = []
        if results is not None and len(results) > 0:
            for row in results:
                x1, y1, x2, y2, tid = row
                tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
                tracks.append(Track(track_id=int(tid), tlwh=tlwh, score=0.0, is_activated=True))
        return tracks


# ------------------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------------------

def create_tracker(
    tracker_type: str = "bytetrack",
    track_thresh: float = 0.5,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    frame_rate: int = 30,
    det_thresh: float = 0.5,
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    delta_t: int = 3,
    asso_func: str = "iou",
    inertia: float = 0.2,
    use_byte: bool = False,
) -> TrackerBase:
    """
    根据 tracker_type 创建对应的跟踪器实例。

    :param tracker_type: "bytetrack" 或 "ocsort"（大小写不敏感）
    """
    t = tracker_type.lower().strip()
    if t == "bytetrack":
        return ByteTrackerWrapper(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )
    elif t == "ocsort":
        return OCSortWrapper(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte,
        )
    else:
        raise ValueError(f"不支持的跟踪器类型: {tracker_type}，可选: bytetrack, ocsort")
