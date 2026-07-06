### 报警去重：对同一位置连续重复的报警进行过滤
import time
import threading
from typing import List, Dict, Tuple, Optional

from utils.obj import Box
from utils.logger import setup_logger

logger = setup_logger("alert_dedup")


class AlertDedup:
    """
    报警去重器。

    作用：如果同一个位置在短时间内反复触发报警，只保留第一次，后续重复的报警被过滤掉。
    常用于避免视频流中同一目标/同一事件被连续多帧重复上传。

    去重维度：
      - 默认按 (device_id, algorithm_code, label) 分组去重；
      - 也可关闭 per_label，只按位置和算法去重。

    判定“同一位置”：两个 Box 的 IoU >= iou_thresh 即认为是同一个位置。
    """

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        iou_thresh: float = 0.5,
        per_label: bool = True,
        max_records: int = 200,
    ):
        """
        :param cooldown_seconds: 同一位置报警的默认冷却时间（秒）
        :param iou_thresh: 判定为“同一位置”的默认 IoU 阈值
        :param per_label: 是否按标签分别去重；True=不同标签即使位置重叠也不会互相过滤
        :param max_records: 每个 key 最多保留多少条历史记录，防止内存无限增长
        """
        self.cooldown_seconds = cooldown_seconds
        self.iou_thresh = iou_thresh
        self.per_label = per_label
        self.max_records = max_records

        # key -> List[AlertRecord]
        self._records: Dict[Tuple, List["AlertRecord"]] = {}
        self._lock = threading.Lock()

    def _make_key(self, device_id: str, algorithm_code: str, label: str) -> Tuple:
        """根据 device_id、算法码、标签生成状态 key"""
        base = (device_id or "_global", algorithm_code or "_all")
        if self.per_label:
            return base + (label,)
        return base

    @staticmethod
    def _cleanup(records: List["AlertRecord"], now: float) -> List["AlertRecord"]:
        """移除每条记录各自的冷却期外的历史记录"""
        return [r for r in records if (now - r.timestamp) < r.cooldown]

    def filter(
        self,
        violations: List[Box],
        device_id: str = "",
        algorithm_code: str = "",
        timestamp: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> List[Box]:
        """
        过滤重复报警。

        :param violations: 当前帧识别到的违章 Box 列表
        :param device_id: 设备/通道标识，用于隔离不同视频流的状态
        :param algorithm_code: 算法代码，用于隔离不同算法的状态
        :param timestamp: 当前时间戳（秒），默认使用 time.monotonic()
        :param cooldown_seconds: 本次去重使用的冷却时间（秒），默认使用实例初始化时的值
        :param iou_thresh: 本次去重使用的 IoU 阈值，默认使用实例初始化时的值
        :return: 需要去重后保留的违章 Box 列表
        """
        if not violations:
            return []

        cooldown = cooldown_seconds if cooldown_seconds is not None else self.cooldown_seconds
        iou = iou_thresh if iou_thresh is not None else self.iou_thresh

        now = timestamp if timestamp is not None else time.monotonic()
        result: List[Box] = []
        suppressed = 0

        with self._lock:
            for v in violations:
                key = self._make_key(device_id, algorithm_code, v.label)
                records = self._cleanup(self._records.get(key, []), now)

                # 判断是否与该 key 下最近冷却期内的某个报警位置重复
                is_dup = False
                for r in records:
                    if v.iou(r.box) >= iou:
                        is_dup = True
                        break

                if is_dup:
                    suppressed += 1
                    continue

                # 新报警，保留并记录（带上本次冷却时间）
                result.append(v)
                records.append(AlertRecord(v, now, cooldown))
                # 限制单 key 历史记录数量
                if len(records) > self.max_records:
                    records = records[-self.max_records :]
                self._records[key] = records

        if suppressed:
            logger.info(
                f"[device={device_id}, algo={algorithm_code}] 报警去重 | "
                f"原始={len(violations)}, 保留={len(result)}, 过滤={suppressed}, "
                f"cooldown={cooldown:.1f}s, iou={iou:.2f}"
            )
        return result

    def reset(self, device_id: str = "", algorithm_code: str = ""):
        """
        清空指定设备/算法的历史记录；如果都不传，则清空全部。
        """
        with self._lock:
            if not device_id and not algorithm_code:
                self._records.clear()
                logger.info("报警去重状态已清空")
                return

            keys_to_remove = []
            for key in self._records.keys():
                if self._key_match(key, device_id, algorithm_code):
                    keys_to_remove.append(key)
            for k in keys_to_remove:
                del self._records[k]
            if keys_to_remove:
                logger.info(f"清空去重状态 | device={device_id}, algo={algorithm_code}, keys={len(keys_to_remove)}")

    def _key_match(self, key: Tuple, device_id: str, algorithm_code: str) -> bool:
        """判断 key 是否匹配给定的 device_id / algorithm_code"""
        key_device, key_algo = key[0], key[1]
        match_device = not device_id or key_device == device_id or key_device == "_global"
        match_algo = not algorithm_code or key_algo == algorithm_code or key_algo == "_all"
        return match_device and match_algo


class AlertRecord:
    """单条报警记录，用于位置去重"""

    __slots__ = ("box", "timestamp", "cooldown")

    def __init__(self, box: Box, timestamp: float, cooldown: float):
        self.box = box
        self.timestamp = timestamp
        self.cooldown = cooldown
