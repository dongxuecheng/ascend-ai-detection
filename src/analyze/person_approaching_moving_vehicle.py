### 人员靠近移动车辆检测
import math
import threading
import time
from typing import Dict, List, Optional, Tuple

from utils.obj import Box
from utils.filter import score_filter, label_filter, area_filter
from utils.logger import setup_logger

logger = setup_logger("person_approaching_moving_vehicle")

# 人员标签：SAM3 / YOLO 的常见表述
DEFAULT_PERSON_LABELS = ["person"]
# 司机标签：与人员框做 IoU 匹配，命中说明该人员是司机（不参与靠近判定）
DEFAULT_DRIVER_LABELS = ["driver"]
# 车辆标签：不同模型/SAM3 可能返回 vehicle / car / truck 等多种表述
DEFAULT_VEHICLE_LABELS = ["truck"]


class PersonApproachingMovingVehicleDetector(object):
    """
    人员靠近移动车辆检测器。

    判定流程：
      1. 过滤出 person 与车辆（vehicle / car / truck / bus 等）框，并做置信度、面积的基础过滤。
      2. 用上一帧的车辆框与当前帧车辆框做匹配（优先 IoU，无重叠时回退到中心点距离），
         匹配成功但 IoU 低于 vehicle_move_iou 的车辆，判定为「移动中」。
      3. 对移动中的车辆框按 vehicle_expend_ratio 向四周扩展（1.2 表示四边各扩展 20%），
         扩展区域即「靠近」的判定范围。
      4. 人员框落在扩展区域内的比例（交集面积 / 人员框面积）达到 approach_iom_thresh 时，
         判定为「人员靠近移动车辆」。

    注意：
      - 车辆是否移动依赖前后两帧对比，因此首次出现的车辆不会当帧报警（仅记录状态），
        需要至少连续两帧观测到同一辆车才可能触发。
      - driver 框会与人员框做 IoU 匹配，命中（IoU >= driver_match_iou）说明该人员就是司机，
        直接跳过，不再做该人员的「靠近」判定，避免司机本人误报。
      - 状态按 device_id 隔离，支持多路视频并发调用；同时被多路线程调用时用锁保护。
      - 本检测器不使用电子围栏，detect 的 fences 参数仅保留以保持接口统一。
    """

    def __init__(
        self,
        vehicle_move_iou: float = 0.6,
        vehicle_expend_ratio: float = 1.1,
        person_score_thresh: float = 0.5,
        person_min_area: float = 1000.0,
        vehicle_score_thresh: float = 0.5,
        vehicle_min_area: float = 2000.0,
        approach_iom_thresh: float = 0.05,
        vehicle_match_distance_ratio: float = 1.0,
        history_ttl_seconds: float = 60.0,
        driver_match_iou: float = 0.5,
        person_labels: Optional[List[str]] = None,
        driver_labels: Optional[List[str]] = None,
        vehicle_labels: Optional[List[str]] = None,
        alert_label: str = "person-approaching-moving-vehicle",
        max_devices: int = 200,
    ):
        """
        :param vehicle_move_iou: 车辆移动判定阈值；同一车辆前后两帧 IoU 低于该值即认为车辆在移动
        :param vehicle_expend_ratio: 车辆框扩展比例，1.2 表示四边各扩展 20%（<=1.0 时按原值作为扩展比例）
        :param person_score_thresh: person 最低置信度
        :param person_min_area: person 最小面积（像素）
        :param vehicle_score_thresh: 车辆框最低置信度
        :param vehicle_min_area: 车辆框最小面积（像素）
        :param approach_iom_thresh: 人员框落入车辆扩展区域的最小比例，达到该值判定为靠近
        :param vehicle_match_distance_ratio: 前后两帧车辆无重叠时，允许的最大中心点位移
                                            （相对车辆框对角线的一半），用于兜底匹配同一辆车
        :param history_ttl_seconds: 车辆历史状态的过期时间（秒），超过后不再参与移动判定，
                                   避免视频流断开/重启后拿旧状态做对比
        :param driver_match_iou: 司机框与人员框匹配的 IoU 阈值，命中即认为该人员是司机，跳过靠近判定
        :param person_labels: 人员标签列表，默认 ["person"]
        :param driver_labels: 司机标签列表，默认 ["driver"]
        :param vehicle_labels: 车辆标签列表，默认 ["vehicle", "car", "truck", "bus", ...]
        :param alert_label: 报警输出标签，默认 "person-approaching-moving-vehicle"
        :param max_devices: 最多保留多少路流的状态，超出后淘汰最早写入的设备状态
        """
        self.vehicle_move_iou = vehicle_move_iou
        self.vehicle_expend_ratio = vehicle_expend_ratio

        self.person_score_thresh = person_score_thresh
        self.person_min_area = person_min_area
        self.vehicle_score_thresh = vehicle_score_thresh
        self.vehicle_min_area = vehicle_min_area

        self.approach_iom_thresh = approach_iom_thresh
        self.vehicle_match_distance_ratio = vehicle_match_distance_ratio
        self.history_ttl_seconds = history_ttl_seconds

        self.driver_match_iou = driver_match_iou
        self.person_labels = list(person_labels) if person_labels else list(DEFAULT_PERSON_LABELS)
        self.driver_labels = list(driver_labels) if driver_labels else list(DEFAULT_DRIVER_LABELS)
        self.vehicle_labels = list(vehicle_labels) if vehicle_labels else list(DEFAULT_VEHICLE_LABELS)
        self.alert_label = alert_label
        self.max_devices = max_devices

        # device_id -> [{"box": Box, "ts": float}] 上一帧车辆框，用于判断车辆是否移动
        self._vehicle_history: Dict[str, List[dict]] = {}
        self._lock = threading.Lock()

    # ---------- 过滤与几何工具 ----------
    def _filter_persons(self, predictions: List[Box]) -> List[Box]:
        """过滤出有效的人员框"""
        persons = label_filter(predictions, self.person_labels)
        persons = score_filter(persons, self.person_score_thresh)
        persons = area_filter(persons, self.person_min_area)
        return persons

    def _filter_drivers(self, predictions: List[Box]) -> List[Box]:
        """过滤出有效的司机框（用于与人员框做 IoU 匹配）"""
        drivers = label_filter(predictions, self.driver_labels)
        drivers = score_filter(drivers, self.person_score_thresh)
        drivers = area_filter(drivers, self.person_min_area)
        return drivers

    def _match_driver(self, person: Box, drivers: List[Box]) -> Optional[Box]:
        """在司机框中查找与该人员框 IoU 达到阈值的目标，命中说明该人员就是司机"""
        for driver in drivers:
            if person.iou(driver) >= self.driver_match_iou:
                return driver
        return None

    def _filter_vehicles(self, predictions: List[Box]) -> List[Box]:
        """过滤出有效的车辆框"""
        vehicles = label_filter(predictions, self.vehicle_labels)
        vehicles = score_filter(vehicles, self.vehicle_score_thresh)
        vehicles = area_filter(vehicles, self.vehicle_min_area)
        return vehicles

    def _expand_vehicle(self, vehicle: Box) -> Box:
        """
        按 vehicle_expend_ratio 向四周扩展车辆框，用于定义「靠近」的判定范围。

        ratio > 1 时取 (ratio - 1) 作为四边扩展比例（1.2 即四边各扩展 20%）；
        ratio 在 (0, 1] 之间时直接作为扩展比例使用；<= 0 时不扩展。
        """
        ratio = self.vehicle_expend_ratio
        expand_ratio = ratio - 1.0 if ratio > 1.0 else ratio
        if expand_ratio <= 0:
            return vehicle
        return vehicle.expand_box("all", expand_ratio)

    @staticmethod
    def _overlap_ratio(person: Box, target: Box) -> float:
        """人员框落在目标区域内的比例（交集面积 / 人员框面积）"""
        area = person.area()
        if area <= 0:
            return 0.0
        return person.intersection(target) / area

    @staticmethod
    def _box_center(box: Box) -> Tuple[float, float]:
        """框的中心点坐标（Box 未暴露 center 属性，这里按坐标计算）"""
        x1, y1, x2, y2 = box.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _box_diagonal(box: Box) -> float:
        """框的对角线长度"""
        x1, y1, x2, y2 = box.box
        return math.hypot(x2 - x1, y2 - y1)

    def _match_history(self, vehicle: Box, history: List[Box]) -> Tuple[Optional[Box], float, float]:
        """
        在当前车辆与历史车辆中寻找同一辆车。

        匹配策略：
          1. 优先选择 IoU 最大的历史车辆（IoU > 0 即认为是同一辆车移动后的位置）；
          2. 若与所有历史车辆都没有重叠，则退化为中心点距离匹配：
             中心点位移不超过车辆框对角线一半的 vehicle_match_distance_ratio 倍时，视为同一辆车。

        :return: (匹配到的历史车辆, 该车辆对应的 IoU, 归一化中心点位移)；未匹配到时返回 (None, 0.0, 0.0)
        """
        if not history:
            return None, 0.0, 0.0

        cx, cy = self._box_center(vehicle)
        half_diagonal = self._box_diagonal(vehicle) / 2.0

        candidates: List[Tuple[Box, float, float]] = []
        for hist in history:
            iou = vehicle.iou(hist)
            hx, hy = self._box_center(hist)
            dist = math.hypot(cx - hx, cy - hy)
            dist_norm = dist / half_diagonal if half_diagonal > 0 else 0.0
            candidates.append((hist, iou, dist_norm))

        # 1. 优先按 IoU 匹配
        best = max(candidates, key=lambda c: c[1])
        if best[1] > 0.0:
            return best[0], best[1], best[2]

        # 2. 无重叠时退化为中心点距离匹配
        best = min(candidates, key=lambda c: c[2])
        if best[2] <= self.vehicle_match_distance_ratio:
            return best[0], best[1], best[2]

        return None, 0.0, 0.0

    def _store_history(self, key: str, vehicles: List[Box], now: float) -> None:
        """写入当前帧车辆状态，并淘汰超出 max_devices 的最早设备状态"""
        with self._lock:
            self._vehicle_history[key] = [{"box": v, "ts": now} for v in vehicles]
            for old_key in list(self._vehicle_history.keys()):
                if len(self._vehicle_history) <= self.max_devices:
                    break
                if old_key != key:
                    self._vehicle_history.pop(old_key, None)

    def _load_history(self, key: str, now: float) -> List[Box]:
        """读取未过期的历史车辆框"""
        with self._lock:
            items = self._vehicle_history.get(key, [])
            return [item["box"] for item in items if now - item["ts"] <= self.history_ttl_seconds]

    def reset(self, device_id: Optional[str] = None) -> None:
        """
        清空车辆历史状态，便于测试或视频流重启后复位。

        :param device_id: 指定设备ID；为 None 时清空所有设备状态
        """
        with self._lock:
            if device_id is None:
                self._vehicle_history.clear()
            else:
                self._vehicle_history.pop(device_id or "_global", None)

    # ---------- 检测入口 ----------
    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> list[Box]:
        """
        检测人员是否靠近移动中的车辆。

        :param predictions: 当前帧所有检测框（SAM3 返回的 person / vehicle / car / truck 等）
        :param fences: 电子围栏坐标列表；本检测器不使用围栏，保留参数以保持接口统一
        :param device_id: 设备ID，用于隔离不同视频流的车辆历史状态
        :param image_width: 原图宽度（当前基于框坐标计算，保留以保持接口统一）
        :param image_height: 原图高度（当前基于框坐标计算，保留以保持接口统一）
        :return: 靠近移动车辆的违规人员 Box 列表（label 为 alert_label）；
                 与 driver 框 IoU 匹配上的人员不会出现在结果中
        """
        key = device_id or "_global"
        now = time.monotonic()

        vehicles = self._filter_vehicles(predictions)
        if not vehicles:
            # 当前帧没有车辆：清空历史，避免流中断后拿旧状态做移动判定
            self.reset(device_id)
            return []

        # 先取上一帧状态，再写入当前帧状态，保证移动判定用的是前一帧
        history = self._load_history(key, now)
        self._store_history(key, vehicles, now)

        persons = self._filter_persons(predictions)
        if not persons or not history:
            return []

        # 司机框：用于与人员框做 IoU 匹配，命中的人员直接跳过靠近判定
        drivers = self._filter_drivers(predictions)

        # 找出移动中的车辆（匹配到历史车辆，但位置变化超出阈值）
        moving_vehicles: List[Tuple[Box, float, float]] = []
        for vehicle in vehicles:
            matched, best_iou, dist_norm = self._match_history(vehicle, history)
            if matched is None:
                # 首次出现的车辆，缺乏移动证据，仅记录状态
                continue
            if best_iou < self.vehicle_move_iou:
                moving_vehicles.append((vehicle, best_iou, dist_norm))

        if not moving_vehicles:
            return []

        # 逐个人员判断是否靠近任一移动车辆
        violations: List[Box] = []
        seen_person_ids = set()
        for vehicle, best_iou, dist_norm in moving_vehicles:
            expanded = self._expand_vehicle(vehicle)
            for person in persons:
                if id(person) in seen_person_ids:
                    continue
                # 司机与人员做 IoU 匹配：命中说明该人员就是司机，直接跳过，不做靠近判定
                matched_driver = self._match_driver(person, drivers) if drivers else None
                if matched_driver is not None:
                    seen_person_ids.add(id(person))
                    logger.debug(
                        f"[device={device_id}] 人员与司机框匹配，跳过靠近判定 | person={person.box}, "
                        f"driver={matched_driver.box}, IoU={person.iou(matched_driver):.3f}"
                    )
                    continue
                ratio = self._overlap_ratio(person, expanded)
                if ratio < self.approach_iom_thresh:
                    continue

                seen_person_ids.add(id(person))
                logger.info(
                    f"[device={device_id}] 人员靠近移动车辆 | vehicle={vehicle.box}, person={person.box}, "
                    f"重叠比例={ratio:.3f}, 车辆移动IoU={best_iou:.3f}, 中心位移={dist_norm:.3f}"
                )
                violations.append(
                    Box(
                        self.alert_label,
                        person.score,
                        list(person.box),
                        mask=person.mask,
                        mask_array=person.mask_array,
                        source=person.source,
                    )
                )

        return violations


