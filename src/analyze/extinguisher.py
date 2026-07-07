# 灭火器缺失检测
from utils.obj import Box
from utils.filter import score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("extinguisher")


class ExtinguisherDetector:
    """
    灭火器缺失检测器：当现场出现人员和火情，但未检测到灭火器时上报违规。
    """

    def __init__(self):
        # 按设备ID保存历史帧的检测结果，便于后续做时序分析
        self._history = {}

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
        3. 返回火情目标 Box 列表作为违规位置（前端可据此画框告警）。

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

        extinguisher_boxes = label_filter(
            predictions,
            ['extinguisher', 'fire extinguisher', 'fire extinguisher cabinet']
        )
        extinguisher_boxes = score_filter(extinguisher_boxes, 0.5)

        # 不存在人员或火情，不触发判定
        if not person_boxes or not fire_boxes:
            return []

        # 检测到灭火器，视为合规
        if extinguisher_boxes:
            logger.info(f"[device={device_id}] 现场存在灭火器，跳过告警")
            return []

        logger.info(
            f"[device={device_id}] 发现违规：存在人员({len(person_boxes)})和火情"
            f"({len(fire_boxes)})，但未检测到灭火器"
        )

        # 保存历史状态
        # if device_id:
        #     self._history[device_id] = fire_boxes

        # 返回火情目标作为违规位置
        return fire_boxes
