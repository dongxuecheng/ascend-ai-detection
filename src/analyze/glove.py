### 判断人员是否佩戴手套
from utils.obj import Box
from utils.filter import score_filter, label_filter


class GloveDetector:
    """
    手套佩戴检测器，按设备维度保存历史状态
    """

    def __init__(self):
        # 按设备ID保存历史帧的检测结果，便于后续做时序分析
        self._history = {}

    def detect(self, predictions: list[Box], fences=None, device_id: str = "", image_width: int = 0, image_height: int = 0) -> list[Box]:
        """
        判断人员是否佩戴手套
        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，当前手套检测不使用，预留以保持接口统一
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :return: 未佩戴手套的手部 Box 列表
        """
        person_boxes = label_filter(predictions, ['person'])

        hand_boxes = label_filter(predictions, ['hand'])
        hand_boxes = score_filter(hand_boxes, 0.7)

        glove_boxes = label_filter(predictions, ['glove', 'industrial glove'])
    
        # Step 1: 过滤掉孤立的手（不与任何 person 相交）
        valid_hands = []
        for hand in hand_boxes:
            for person in person_boxes:
                if hand.iom(person) > 0:
                    valid_hands.append(hand)
                    break

        # Step 2: 过滤掉孤立的手套（不与任何 person 相交）
        valid_gloves = []
        for glove in glove_boxes:
            for person in person_boxes:
                if glove.iom(person) > 0:
                    valid_gloves.append(glove)
                    break

        # Step 3: 对每个有效的手，检查是否有手套与之重合度很高
        result = []
        used_gloves = set()

        for hand in valid_hands:
            has_glove = False
            for glove in valid_gloves:
                if id(glove) in used_gloves:
                    continue
                if hand.iom(glove) > 0.5:
                    has_glove = True
                    used_gloves.add(id(glove))
                    break

            if not has_glove:
                result.append(hand)

        # 只有指定了设备ID才保存历史状态，便于后续做时序分析（如连续多帧确认、火焰跳动检测等）
        # if device_id:
        #     self._history[device_id] = result
        return result
