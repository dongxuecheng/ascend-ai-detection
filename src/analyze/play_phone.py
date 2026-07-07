### 玩手机检测
from utils.obj import Box
from utils.filter import score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("play_phone")


class PlayPhoneDetector:
    """
    玩手机检测器：检测到手机与手存在明显重叠时，判定为玩手机违规。
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
        检测是否玩手机。

        逻辑：
        1. 分别过滤出 person、hand、phone/mobile phone。
        2. 过滤掉孤立的手和手机（不与任何 person 相交），减少误报。
        3. 对每个有效的手，检查是否与手机存在较高重叠。
        4. 返回与之相交的手机 Box 列表作为违规目标。

        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，当前玩手机检测不使用，预留以保持接口统一
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :param image_width: 原图宽度（预留）
        :param image_height: 原图高度（预留）
        :return: 违规目标 Box 列表（手机）
        """
        person_boxes = label_filter(predictions, ['person'])

        hand_boxes = label_filter(predictions, ['hand'])
        hand_boxes = score_filter(hand_boxes, 0.5)

        phone_boxes = label_filter(predictions, ['mobile phone', 'phone', 'cell phone'])
        phone_boxes = score_filter(phone_boxes, 0.5)

        # Step 1: 过滤掉孤立的手（不与任何 person 相交）
        valid_hands = []
        for hand in hand_boxes:
            for person in person_boxes:
                if hand.iom(person) > 0:
                    valid_hands.append(hand)
                    break

        # Step 2: 过滤掉孤立的手机（不与任何 person 相交）
        valid_phones = []
        for phone in phone_boxes:
            for person in person_boxes:
                if phone.iom(person) > 0:
                    valid_phones.append(phone)
                    break

        # Step 3: 对每个有效的手，检查是否有手机与之重合度很高
        result = []
        used_phones = set()

        for hand in valid_hands:
            for phone in valid_phones:
                if id(phone) in used_phones:
                    continue
                # 手和手机的 IoM 较高，认为在握持/操作手机
                if hand.iom(phone) > 0.3:
                    result.append(phone)
                    used_phones.add(id(phone))
                    logger.info(
                        f"[device={device_id}] 发现玩手机：hand 与 phone 重叠 "
                        f"iom={hand.iom(phone):.2%}"
                    )
                    break

        # 保存历史状态
        # if device_id:
        #     self._history[device_id] = result

        return result
