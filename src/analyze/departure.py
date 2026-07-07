### 离岗/脱岗检测
import time
import threading
from typing import List, Dict

from shapely.geometry import Polygon

from utils.obj import Box
from utils.filter import label_filter, area_filter, score_filter, nms_filter
from utils.logger import setup_logger

logger = setup_logger("departure")


class DepartureDetector:
    """
    离岗检测器：在指定时间窗口内，如果监控区域内连续帧都未检测到人员，则判定为离岗违章。

    典型场景：
      - 值班岗位、操作台、关键工位要求持续有人值守；
      - 人员在岗状态监测，超过设定时间无人即告警。

    状态按 device_id 隔离，支持多路视频并发调用。
    """

    def __init__(
        self,
        duration_seconds: float = 900,
        min_score: float = 0.8,
        min_area: float = 1000.0,
        nms_iou: float = 0.5,
        latch: bool = True,
    ):
        """
        :param duration_seconds: 监控区域内连续无人的持续时间阈值（秒），达到后触发违章
        :param min_score: person 框最低置信度
        :param min_area: person 框最小面积（像素）
        :param nms_iou: 对重叠 person 框做 NMS 的 IoU 阈值
        :param latch: 是否对同一次离岗状态只告警一次；True=只返回一次，状态重置前不再返回
        """
        self.duration_seconds = duration_seconds
        self.min_score = min_score
        self.min_area = min_area
        self.nms_iou = nms_iou
        self.latch = latch

        # device_id -> 状态字典
        # 状态字段：last_seen_time（最后一次检测到人员的时间）、triggered（是否已触发）
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

    @staticmethod
    def _fence_bbox(fences) -> List[float]:
        """
        计算所有围栏的最小外接矩形 [x1, y1, x2, y2]。
        用于触发离岗告警时返回一个代表监控区域的 Box。
        """
        xs = []
        ys = []
        for fence in fences:
            coords = fence.tolist() if hasattr(fence, "tolist") else fence
            for pt in coords:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    xs.append(float(pt[0]))
                    ys.append(float(pt[1]))
        if not xs or not ys:
            return [0.0, 0.0, 0.0, 0.0]
        return [min(xs), min(ys), max(xs), max(ys)]

    def detect(
        self,
        predictions: List[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> List[Box]:
        """
        离岗检测入口

        :param predictions: 当前帧的所有检测框
        :param fences: 可选的电子围栏列表，只在指定区域内判断人员是否在岗
        :param device_id: 设备 ID，用于隔离不同设备/通道的状态
        :param image_width: 原图宽度，无围栏时用于构造全帧告警框
        :param image_height: 原图高度，无围栏时用于构造全帧告警框
        :return: 违章目标 Box 列表，命中时返回一个 label 为 "departure" 的 Box
        """
        now = time.monotonic()
        persons = self._filter_persons(predictions, fences)
        has_person = len(persons) > 0
        key = device_id or "_global"

        with self._lock:
            state = self._states.get(key)

            # 当前帧检测到人员，更新最后出现时间并解除触发状态
            if has_person:
                self._states[key] = {"last_seen_time": now, "triggered": False}
                return []

            # 当前帧无人员：开始计时或延续计时
            if state is None or state.get("last_seen_time") is None:
                self._states[key] = {"last_seen_time": now, "triggered": False}
                return []

            elapsed = now - state["last_seen_time"]
            if elapsed < self.duration_seconds:
                # 持续时间还不够
                return []

            if self.latch and state.get("triggered"):
                # 已经告警过，等待状态重置（重新检测到人员）后再恢复检测
                return []

            # 达到阈值，触发离岗违章
            state["triggered"] = True

            # 构造代表监控区域的告警框
            if fences:
                box = self._fence_bbox(fences)
            elif image_width > 0 and image_height > 0:
                box = [0.0, 0.0, float(image_width), float(image_height)]
            else:
                logger.warning(
                    f"[device={device_id}] 离岗触发但无法确定告警框（无围栏且图像尺寸未知）"
                )
                return []

            violation = Box(label="departure", score=1.0, box=box)
            logger.warning(
                f"[device={device_id}] 离岗违章触发 | 无人持续时间={elapsed:.1f}s, "
                f"box={violation.box}"
            )
            return [violation]
