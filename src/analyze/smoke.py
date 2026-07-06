### 抽烟检测
import threading
from typing import List, Dict

from utils.obj import Box
from utils.filter import label_filter, score_filter, area_filter, nms_filter
from utils.logger import setup_logger

logger = setup_logger("smoke")


class SmokingDetector:
    """
    抽烟检测器。

    检测目标：cigarette、person、hand、head。
    逻辑：
      1. 过滤出有效的人（person）。
      2. 找出与人（person）有空间重叠的 hand / head（排除孤立的手/头）。
      3. 如果香烟（cigarette）与上述有效 hand 或 head 相交，则判定为抽烟违章。
         注意：由于 person 框通常很大，不直接用“香烟与人相交”作为告警条件。
      4. 返回触发违章的香烟 Box 列表。
    """

    def __init__(
        self,
        person_min_score: float = 0.5,
        person_min_area: float = 1000.0,
        hand_min_score: float = 0.5,
        head_min_score: float = 0.5,
        cigarette_min_score: float = 0.5,
        min_iom: float = 0.0,
        alarm_on_head: bool = True,
    ):
        """
        :param person_min_score: person 最低置信度
        :param person_min_area: person 最小面积（像素）
        :param hand_min_score: hand 最低置信度
        :param head_min_score: head 最低置信度
        :param cigarette_min_score: cigarette 最低置信度
        :param min_iom: 判定两个框“相交”的最小 IoM（0 表示只要接触就相交）
        :param alarm_on_head: 是否也允许“香烟与 head 相交”触发告警；默认 True，即“手或头”都触发
        """
        self.person_min_score = person_min_score
        self.person_min_area = person_min_area
        self.hand_min_score = hand_min_score
        self.head_min_score = head_min_score
        self.cigarette_min_score = cigarette_min_score
        self.min_iom = min_iom
        self.alarm_on_head = alarm_on_head

        # 按 device_id 保存历史结果
        self._history: Dict[str, List[Box]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _any_overlap(src: Box, targets: List[Box], min_iom: float) -> bool:
        """判断 src 是否与 targets 中任意一个框的 IoM 达到阈值"""
        for target in targets:
            if src.iom(target) > min_iom:
                return True
        return False

    @staticmethod
    def _find_overlapped(src_list: List[Box], target_list: List[Box], min_iom: float) -> List[Box]:
        """返回 src_list 中与 target_list 任意框相交的 src"""
        result = []
        for src in src_list:
            for target in target_list:
                if src.iom(target) > min_iom:
                    result.append(src)
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
        抽烟检测入口。

        :param predictions: 当前帧的所有检测框
        :param fences: 当前未使用，预留以保持接口统一
        :param device_id: 设备 ID，用于隔离不同设备的历史状态
        :param image_width: 预留参数，保持接口统一
        :param image_height: 预留参数，保持接口统一
        :return: 触发抽烟违章的 cigarette Box 列表
        """
        # 1. 按标签过滤
        person_boxes = label_filter(predictions, ["person"])
        person_boxes = score_filter(person_boxes, self.person_min_score)
        person_boxes = area_filter(person_boxes, self.person_min_area)
        person_boxes = nms_filter(person_boxes, 0.5)

        hand_boxes = label_filter(predictions, ["hand"])
        hand_boxes = score_filter(hand_boxes, self.hand_min_score)

        head_boxes = label_filter(predictions, ["head"])
        head_boxes = score_filter(head_boxes, self.head_min_score)

        cigarette_boxes = label_filter(predictions, ["cigarette"])
        cigarette_boxes = score_filter(cigarette_boxes, self.cigarette_min_score)
        cigarette_boxes = nms_filter(cigarette_boxes, 0.5)

        # 2. 找出与人相交的 hand / head（先排除孤立的手/头）
        valid_hands = self._find_overlapped(hand_boxes, person_boxes, self.min_iom)
        valid_heads = self._find_overlapped(head_boxes, person_boxes, self.min_iom)

        # 3. 判断香烟是否与有效 hand 或 person 相交
        result = []
        seen_ids = set()
        for cigarette in cigarette_boxes:
            cid = id(cigarette)
            if cid in seen_ids:
                continue

            alarm = False

            # 3.1 香烟与“有效手”相交
            if self._any_overlap(cigarette, valid_hands, self.min_iom):
                alarm = True

            # 3.2 （可选）香烟与“有效头”相交
            if not alarm and self.alarm_on_head and self._any_overlap(cigarette, valid_heads, self.min_iom):
                alarm = True

            if alarm:
                result.append(cigarette)
                seen_ids.add(cid)

        # 保存历史结果
        if device_id:
            with self._lock:
                self._history[device_id] = result

        if result:
            logger.warning(
                f"[device={device_id}] 抽烟违章触发 | 有效手={len(valid_hands)}, "
                f"有效头={len(valid_heads)}, 违章香烟数={len(result)}"
            )
        return result
