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

    检测目标：cigarette、person、face。
    逻辑：
      1. 过滤出有效的人（person）。
      2. 找出与人（person）有空间重叠的 head（排除孤立的头）。
      3. 只有当香烟（cigarette）与上述有效 head 相交时，才判定为抽烟违章；
         与 hand 相交都不算。
      4. 返回触发违章的 face（头）Box 列表。
    """

    def __init__(
        self,
        person_min_score: float = 0.5,
        person_min_area: float = 1000.0,
        head_min_score: float = 0.6,
        cigarette_min_score: float = 0.6,
        min_iom: float = 0.3,
        alarm_on_head: bool = True,
    ):
        """
        :param person_min_score: person 最低置信度
        :param person_min_area: person 最小面积（像素）
        :param head_min_score: head 最低置信度
        :param cigarette_min_score: cigarette 最低置信度
        :param min_iom: 判定两个框“相交”的最小 IoM（0 表示只要接触就相交）
        :param alarm_on_head: 是否允许“香烟与 head 相交”触发告警；默认 True
        """
        self.person_min_score = person_min_score
        self.person_min_area = person_min_area
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
        :return: 触发抽烟违章的 head（头）Box 列表
        """
        # 1. 按标签过滤
        person_boxes = label_filter(predictions, ["person"])
        person_boxes = score_filter(person_boxes, self.person_min_score)
        person_boxes = area_filter(person_boxes, self.person_min_area)
        person_boxes = nms_filter(person_boxes, 0.5)

        head_boxes = label_filter(predictions, ["face"])
        head_boxes = score_filter(head_boxes, self.head_min_score)

        cigarette_boxes = label_filter(predictions, ["cigarette"])
        cigarette_boxes = area_filter(cigarette_boxes, 10 * 10)
        cigarette_boxes = score_filter(cigarette_boxes, self.cigarette_min_score)
        cigarette_boxes = nms_filter(cigarette_boxes, 0.5)

        # 2. 找出与人相交的 head（先排除孤立的头）
        valid_heads = self._find_overlapped(head_boxes, person_boxes, self.min_iom)

        # 3. 判断香烟是否与有效 head 相交（与 hand 相交都不算），
        #    命中时返回该 head（头）的框
        result = []
        seen_ids = set()
        for head in valid_heads:
            hid = id(head)
            if hid in seen_ids:
                continue

            # 3.1 存在任一香烟与“有效头”相交才判定为抽烟
            if self.alarm_on_head and self._any_overlap(head, cigarette_boxes, 0.0001):
                result.append(head)
                seen_ids.add(hid)

        if result:
            logger.warning(
                f"[device={device_id}] 抽烟违章触发 | 有效头={len(valid_heads)}, "
                f"违章头部数={len(result)}"
            )
        return result
