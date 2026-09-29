# 灭火器缺失检测
import threading
import time
from collections import deque
from typing import Dict, List

from utils.obj import Box
from utils.filter import score_filter, label_filter, nms_filter
from utils.logger import setup_logger

logger = setup_logger("extinguisher")


class ExtinguisherDetector:
    """
    灭火器缺失检测器：当现场出现人员和火情，但未检测到灭火器时上报违规。

    支持火焰时序分析（火焰框跳动检测）：
      真实火焰的检测框会随火苗跳动而位置/大小不稳定；
      静止的红色物体（如灭火器箱、红色设备）虽然可能被误检为火焰，
      但其检测框在多帧间保持稳定。
    因此通过对比相邻帧火焰框的 IoU 区分“跳动”的真实火焰与“静止”的误检：
      - 火焰框连续多帧跳动（与历史帧匹配不上）→ 判定为真实火焰，保留违规；
      - 火焰框连续多帧静止 → 判定为静止物体误检，过滤违规。

    静止位置保护（enable_same_position_skip）：
      若当前帧火焰框与上一帧的火焰框位置一样（一一配对后每个框的 IoU 均 > same_iou_thresh，
      默认 0.9），说明画面/检测结果几乎没变（静止红色物体误检、卡帧、结果复用），
      不作为火情证据上报，避免同一处静止目标被反复告警。
    """

    def __init__(
        self,
        history_len: int = 8,
        min_frames: int = 4,
        stable_iou_thresh: float = 0.7,
        stable_ratio: float = 0.8,
        enable_temporal_filter: bool = True,
        enable_same_position_skip: bool = True,
        same_iou_thresh: float = 0.9,
        max_devices: int = 64,
        idle_timeout: float = 3600.0,
        reset_grace: int = 3,
    ):
        """
        :param history_len: 每个设备保留的火焰历史帧数
        :param min_frames: 需要积累的最少历史帧数；少于该帧数时无法判断跳动，先不告警
        :param stable_iou_thresh: 判定“两帧火焰框重叠稳定”的 IoU 阈值
        :param stable_ratio: 最近一帧中达到稳定重叠的比例阈值，超过则判定为静止误检
        :param enable_temporal_filter: 是否启用火焰时序过滤（关闭则保持原逻辑）
        :param enable_same_position_skip: 是否启用“与上一帧位置一样则不告警”的静止位置保护
        :param same_iou_thresh: 判定位置一样的 IoU 阈值；前后两帧火焰框一一配对后
                                每个框的 IoU 都大于该值即视为位置一样（默认 0.9）
        :param max_devices: 同时跟踪的最大设备数，超出后按最近活动时间淘汰最旧设备
        :param idle_timeout: 设备空闲超时（秒），超过后自动清除该设备历史缓存
        :param reset_grace: 连续无火焰多少帧后才清空历史缓存
        """
        self.history_len = history_len
        self.min_frames = min_frames
        self.enable_same_position_skip = enable_same_position_skip
        self.same_iou_thresh = same_iou_thresh
        self.stable_iou_thresh = stable_iou_thresh
        self.stable_ratio = stable_ratio
        self.enable_temporal_filter = enable_temporal_filter
        self.max_devices = max_devices
        self.idle_timeout = idle_timeout
        self.reset_grace = reset_grace

        # 按设备ID保存火焰框历史（deque 的每个元素是一帧的火焰框列表）
        self._history: Dict[str, deque] = {}
        # 设备最后活动时间，用于空闲清理与设备数淘汰
        self._last_active: Dict[str, float] = {}
        # 设备连续缺席计数，用于容忍偶发漏检、避免频繁清空历史
        self._miss_count: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ---------- 历史缓存清理 ----------
    def _reset_device(self, device_id: str) -> None:
        """清空并移除某设备的历史缓存（须在持锁状态下调用）"""
        self._history.pop(device_id, None)
        self._last_active.pop(device_id, None)
        self._miss_count.pop(device_id, None)

    def _prune(self, now: float) -> None:
        """
        清理历史缓存，避免 device_id 不断变化导致 dict 无限增长（内存溢出）。
        1. 删除空闲超时（idle_timeout）的设备缓存；
        2. 超过设备数上限（max_devices）时，按最后活动时间淘汰最旧设备。
        （须在持锁状态下调用）
        """
        if self._last_active:
            idle_ids = [
                did for did, ts in self._last_active.items()
                if now - ts > self.idle_timeout
            ]
            for did in idle_ids:
                self._reset_device(did)

        if len(self._history) > self.max_devices:
            sorted_ids = sorted(self._last_active, key=self._last_active.get)
            for did in sorted_ids[: len(self._history) - self.max_devices]:
                self._reset_device(did)

    def _note_miss(self, device_id: str) -> None:
        """
        记录一次“无火焰”帧。

        只有连续无火焰超过 reset_grace 次时才真正清空历史缓存，
        避免 SAM3/YOLO 偶发漏检导致历史被频繁清空，进而反复进入预热期产生间歇误报。
        （须在持锁状态下调用）
        """
        if device_id not in self._history:
            return
        self._miss_count[device_id] = self._miss_count.get(device_id, 0) + 1
        if self._miss_count[device_id] >= self.reset_grace:
            self._reset_device(device_id)

    # ---------- 火焰时序分析 ----------
    @staticmethod
    def _snapshot(boxes: List[Box]) -> List[Box]:
        """复制火焰框（仅坐标/分数），避免历史引用被后续流程修改"""
        return [Box(b.label, b.score, list(b.box)) for b in boxes]

    def _best_iou(self, box: Box, past_boxes: List[Box]) -> float:
        """返回 box 与某历史帧所有火焰框的最大 IoU（该帧无火焰框时为 0）"""
        if not past_boxes:
            return 0.0
        return max(box.iou(pb) for pb in past_boxes)

    @staticmethod
    def _merge_fire_boxes(
        boxes: List[Box],
        iou_thresh: float = 0.3,
        iom_thresh: float = 0.5,
    ) -> List[Box]:
        """
        合并高度重叠/嵌套的火焰框。

        SAM3 可能把一团火识别成多个相互重叠的小框（如 fire / flame 拆分），
        这些小框逐帧拆分方式不稳定，会导致后续时序分析跨帧匹配不上。
        这里将相互重叠（IoU >= iou_thresh）或嵌套（IoM >= iom_thresh）的框合并为并集框，
        使每一团火只有一个代表框，跨帧更稳定。
        """
        if not boxes:
            return []
        result = [Box(b.label, b.score, list(b.box)) for b in boxes]

        changed = True
        while changed:
            changed = False
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    a, b = result[i], result[j]
                    if a.iou(b) >= iou_thresh or a.iom(b) >= iom_thresh:
                        x1 = min(a.box[0], b.box[0])
                        y1 = min(a.box[1], b.box[1])
                        x2 = max(a.box[2], b.box[2])
                        y2 = max(a.box[3], b.box[3])
                        a.box = [x1, y1, x2, y2]
                        a.score = max(a.score, b.score)
                        result.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return result

    # ---------- 静止位置判断 ----------
    def _boxes_same(self, current: List[Box], previous: List[Box]) -> bool:
        """
        判断两组火焰框的位置是否“一样”：数量相同，且每个当前框都能在上一帧中
        找到 IoU > same_iou_thresh 的框（一一配对，不重复使用）。
        与顺序无关。
        """
        if len(current) != len(previous):
            return False

        remaining = list(previous)
        for c in current:
            best_i, best_iou = -1, 0.0
            for i, p in enumerate(remaining):
                iou = c.iou(p)
                if iou > best_iou:
                    best_i, best_iou = i, iou
            if best_i < 0 or best_iou <= self.same_iou_thresh:
                return False
            remaining.pop(best_i)
        return True

    def _is_same_position(self, device_id: str, fire_boxes: List[Box]) -> bool:
        """
        判断当前火焰框与上一帧火焰框的位置是否一样（逐框 IoU > same_iou_thresh）。

        注意：必须在写入本帧历史之前调用，此时 history[-1] 仍是上一帧的火焰框。
        :return: True=与上一帧位置一样（静止/画面未更新，应跳过告警）
        """
        if not self.enable_same_position_skip or not fire_boxes:
            return False

        with self._lock:
            history = self._history.get(device_id)
            if not history:
                return False
            last_boxes = history[-1]

        return self._boxes_same(fire_boxes, last_boxes)

    def _is_flame_jumping(self, device_id: str, fire_boxes: List[Box]) -> tuple:
        """
        更新火焰框历史并判断当前火焰是否“跳动”。
        :return: (is_jumping: bool, stable_count: int, total: int)
        """
        with self._lock:
            now = time.time()

            history = self._history.get(device_id)
            if history is None:
                history = deque(maxlen=self.history_len)
                self._history[device_id] = history
            self._last_active[device_id] = now
            # 命中（人+火且无灭火器），清零缺席计数
            self._miss_count[device_id] = 0

            # 新设备插入后再执行清理：空闲超时 + 设备数上限淘汰
            self._prune(now)

            # 关闭时序过滤：保持原始逻辑，直接放行
            if not self.enable_temporal_filter:
                history.append(self._snapshot(fire_boxes))
                return True, 0, len(fire_boxes)

            # 历史帧不足：无法判断是否跳动，先记录但不告警（预热期静默，
            # 避免静止误检在预热期被反复上报）
            if len(history) < self.min_frames:
                history.append(self._snapshot(fire_boxes))
                return False, 0, len(fire_boxes)

            # 只与最近一帧的火焰框比较，判断“相邻帧是否跳动”。
            # 若与整个历史窗口摊平比较，在两位置间来回跳动的火焰会被误判为稳定。
            recent_boxes = history[-1]

            stable_count = 0
            for fb in fire_boxes:
                if self._best_iou(fb, recent_boxes) >= self.stable_iou_thresh:
                    stable_count += 1

            history.append(self._snapshot(fire_boxes))

            total = len(fire_boxes)
            ratio = stable_count / total if total else 0.0
            # 大部分当前框都能与最近一帧对上 => 位置稳定 => 疑似静止物体误检
            is_jumping = ratio < self.stable_ratio
            return is_jumping, stable_count, total

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0
    ) -> list[Box]:
        """
        检测人员+火情场景下是否缺失灭火器。

        逻辑：
        1. 分别过滤出 person、fire/flame、extinguisher 三类目标。
        2. 当同时存在 person 和 fire/flame，且未检测到 extinguisher 时，判定为违规。
        3. 火焰时序分析：真实火焰的框会跳动，静止红色物体（误检）的框多帧稳定，
           稳定帧占比过高时过滤该违规。
        4. 静止位置保护：若当前帧火焰框与上一帧位置完全一致（疑似静止物体/画面未更新），
           直接跳过告警。
        5. 返回火情目标 Box 列表作为违规位置。

        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，当前灭火器检测不使用，预留以保持接口统一
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :param image_width: 原图宽度（预留）
        :param image_height: 原图高度（预留）
        :return: 违规目标 Box 列表（火情目标）
        """
        # 过滤各类目标
        person_boxes = label_filter(predictions, ['person'])
        person_boxes = score_filter(person_boxes, 0.8)

        fire_boxes = label_filter(predictions, ['fire', 'flame'])
        fire_boxes = score_filter(fire_boxes, 0.8)
        fire_boxes = nms_filter(fire_boxes, 0.5)
        # 合并 SAM3 拆分的重叠/嵌套火焰框，避免多小框导致跨帧匹配不稳定
        fire_boxes = self._merge_fire_boxes(fire_boxes)

        extinguisher_boxes = label_filter(
            predictions,
            ['extinguisher', 'fire extinguisher', 'fire extinguisher cabinet']
        )
        extinguisher_boxes = score_filter(extinguisher_boxes, 0.5)

        # 只要检测到火焰，就持续更新火焰框历史（用于下次相邻帧稳定性对比），
        # 与是否检测到人员/灭火器无关：火焰框是否“跳动”是火焰自身的属性。
        same_position = False
        if fire_boxes:
            # 先与上一帧比较位置是否一样，再更新历史（此时 history[-1] 仍是上一帧）
            same_position = self._is_same_position(device_id, fire_boxes)
            # is_jumping, stable_count, total = self._is_flame_jumping(device_id, fire_boxes)
        else:
            # 无火焰：记录缺席，连续缺席多次才清空历史缓存
            with self._lock:
                self._note_miss(device_id)
            return []

        # 不存在人员：暂不告警，但火焰历史已更新，供下次出现人员时对比
        if not person_boxes:
            return []

        # 检测到灭火器：视为合规，不告警（火焰历史已更新）
        if extinguisher_boxes:
            logger.info(f"[device={device_id}] 现场存在灭火器，跳过告警")
            return []

        # 静止位置保护：火焰框与上一帧位置一样（IoU > same_iou_thresh），说明画面/检测结果没变化
        # （静止红色物体误检、卡帧、结果复用等），不作为火情证据上报
        if same_position:
            logger.info(
                f"[device={device_id}] 火焰框与上一帧位置一样（IoU>{self.same_iou_thresh}，静止/画面未更新），跳过告警"
            )
            return []

        # 人员+火情且无灭火器：根据火焰跳动判定决定是否告警
        # logger.info(
        #     f"[device={device_id}] 火焰时序分析 | 火焰框={total} | 稳定框={stable_count} | "
        #     f"判定={'跳动(真实火焰)' if is_jumping else '静止或证据不足(跳过告警)'}"
        # )

        # 火焰框静止（疑似误检）或历史不足：过滤违规
        # if not is_jumping:
        #     logger.info(f"[device={device_id}] 火焰框稳定或历史不足，判定为误检，跳过告警")
        #     return []

        logger.info(
            f"[device={device_id}] 发现违规：存在人员({len(person_boxes)})和火情"
            f"({len(fire_boxes)})，但未检测到灭火器"
        )

        # 返回火情目标作为违规位置
        return fire_boxes


