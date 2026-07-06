### 单人作业/单人滞留检测
import time
import threading
from typing import List, Dict, Optional

from shapely.geometry import Polygon

from utils.obj import Box
from utils.filter import label_filter, area_filter, score_filter, nms_filter
from utils.logger import setup_logger

logger = setup_logger("single")


class SingleDetector:
    """
    单人检测器：在指定时间窗口内，如果连续帧都只检测到 1 个人，则判定为违章。

    典型场景：
      - 高危作业区域要求“双人作业”，若只有 1 人则告警；
      - 受限空间/值班岗位单人滞留检测。

    状态按 device_id 隔离，支持多路视频并发调用。
    """

    def __init__(
        self,
        duration_seconds: float = 300,
        min_score: float = 0.8,
        min_area: float = 1000.0,
        nms_iou: float = 0.5,
        latch: bool = True,
    ):
        """
        :param duration_seconds: 连续只检测到 1 人的持续时间阈值（秒），达到后触发违章
        :param min_score: person 框最低置信度
        :param min_area: person 框最小面积（像素）
        :param nms_iou: 对重叠 person 框做 NMS 的 IoU 阈值
        :param latch: 是否对同一次单人状态只告警一次；True=只返回一次，状态重置前不再返回
        """
        self.duration_seconds = duration_seconds
        self.min_score = min_score
        self.min_area = min_area
        self.nms_iou = nms_iou
        self.latch = latch

        # device_id -> 状态字典
        # 状态字段：start_time（计时开始时间）、triggered（是否已触发）
        self._states: Dict[str, dict] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _to_polygon_coords(fence) -> List[tuple]:
        """把围栏坐标统一转成 shapely.Polygon 可接受的 (x, y) 列表"""
        if hasattr(fence, "tolist"):
            fence = fence.tolist()
        coords = []
        for pt in fence:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append((float(pt[0]), float(pt[1])))
        return coords

    def _filter_persons(self, predictions: List[Box], fences=None) -> List[Box]:
        """
        过滤并去重得到有效的人员框；如果传入 fences，则只保留在围栏内的人员。
        """
        persons = label_filter(predictions, ["person"])
        persons = score_filter(persons, self.min_score)
        persons = area_filter(persons, self.min_area)
        persons = nms_filter(persons, self.nms_iou)

        if not fences:
            return persons

        # 只保留落在任一围栏多边形内的人员
        polygons = []
        for fence in fences:
            coords = self._to_polygon_coords(fence)
            if len(coords) < 3:
                continue
            try:
                polygons.append(Polygon(coords))
            except Exception as e:
                logger.warning(f"创建 Polygon 失败: {e}, coords={coords}")
                continue

        if not polygons:
            return []

        result = []
        for person in persons:
            for poly in polygons:
                if person.point_in_fence("bottom_center", poly):
                    result.append(person)
                    break
        return result

    def detect(
        self,
        predictions: List[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> List[Box]:
        """
        单人检测入口

        :param predictions: 当前帧的所有检测框
        :param fences: 可选的电子围栏列表，只在指定区域内计数
        :param device_id: 设备 ID，用于隔离不同设备/通道的状态
        :param image_width: 预留参数，保持接口统一
        :param image_height: 预留参数，保持接口统一
        :return: 违章目标 Box 列表，命中时返回一个 label 为 "single_person" 的 Box
        """
        now = time.monotonic()
        persons = self._filter_persons(predictions, fences)
        count = len(persons)
        key = device_id or "_global"

        with self._lock:
            state = self._states.get(key)

            # 只要当前帧人数不是 1，就重置计时/触发状态。
            # 即使 YOLO/SAM3 当前帧没有检测到人（count == 0），也会进入此分支重置状态，
            # 确保时间累计类逻辑在空输入时仍然被正确执行一遍。
            if count != 1:
                self._states[key] = {"start_time": None, "triggered": False}
                return []

            # count == 1：开始计时或延续计时
            if state is None or state.get("start_time") is None:
                self._states[key] = {"start_time": now, "triggered": False}
                return []

            elapsed = now - state["start_time"]
            if elapsed < self.duration_seconds:
                # 持续时间还不够
                return []

            if self.latch and state.get("triggered"):
                # 已经告警过，等待状态重置（人数发生变化）后再恢复检测
                return []

            # 达到阈值，触发违章
            state["triggered"] = True
            person = persons[0]
            violation = Box(
                label="single_person",
                score=person.score,
                box=list(person.box),
            )
            logger.warning(
                f"[device={device_id}] 单人违章触发 | 持续时间={elapsed:.1f}s, "
                f"box={violation.box}, score={violation.score:.3f}"
            )
            return [violation]
