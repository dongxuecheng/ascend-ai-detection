from utils.obj import Box
from utils.filter import area_filter, score_filter, label_filter
import numpy as np

EXPOSED_MIN_MASK_PIXELS = 200

class ExposedArmLegDetector:
    """
    是否裸露小臂、小腿等；判断是否穿手套检测器，按设备维度保存历史状态
    """

    def __init__(self):
        # 按设备ID保存历史帧的检测结果，便于后续做时序分析
        self._history = {}

    def _remove_duplicates(self, boxes: list[Box], containment_threshold: float = 0.95) -> list[Box]:
        """
        去除被另一个高分框完全包含的重复框。
        按分数降序遍历，若当前框已有 >= containment_threshold 的面积被某个已保留框覆盖，则丢弃。
        """
        if len(boxes) <= 1:
            return boxes
        sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
        kept = []
        for box in sorted_boxes:
            is_dup = False
            for kept_box in kept:
                box_iom = box.iom(kept_box)
                if box_iom >= containment_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(box)
        return kept

    def remove_duplicate_arm(self, arm_boxes: list[Box]) -> list[Box]:
        return self._remove_duplicates(arm_boxes)

    def remove_duplicate_person(self, person_boxes: list[Box]) -> list[Box]:
        return self._remove_duplicates(person_boxes)

    def detect_exposed_arm(self, predictions: list[Box], fences=None, image_width: int = 0, image_height: int = 0) -> list[Box]:
        skin_boxes = label_filter(predictions, ['skin'])
        if not skin_boxes:
            return []
        
        person_boxes = label_filter(predictions, ['person'])
        if not person_boxes:
            return []
        
        arm_boxes = label_filter(predictions, ['arm'])
        if not arm_boxes:
            return []
        
        arm_boxes = self.remove_duplicate_arm(arm_boxes)
        
        hand_boxes = label_filter(predictions, ['hand'])
        wrist_boxes = label_filter(predictions, ['wrist'])

        # 去除掉和人没有相交的标签
        person_arm_boxes = [arm for person in person_boxes for arm in arm_boxes if person.iom(arm) > 0.5]
        person_hand_boxes = [hand for person in person_boxes for hand in hand_boxes if person.iom(hand) > 0.5]
        person_skin_boxes = [skin for person in person_boxes for skin in skin_boxes if person.iom(skin) > 0.5]
        person_wrist_boxes = [wrist for person in person_boxes for wrist in wrist_boxes if person.iom(wrist) > 0.5]

        exposed_arm_boxes = []
        for arm_box in person_arm_boxes:
            # 1) 计算 皮肤 ∩ 手臂（手臂上裸露的皮肤区域）
            exposed_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            for skin_box in person_skin_boxes:
                inter = arm_box.mask_intersection(skin_box, 'exposed', image_width, image_height)
                if inter is not None and inter.mask_array is not None:
                    exposed_mask = np.logical_or(exposed_mask, inter.mask_array)

            if not exposed_mask.any():
                continue

            exposed_box = Box('exposed', 1.0, [0, 0, 1, 1], mask=None, mask_array=exposed_mask)

            # 2) 减去手部（手本来就是裸露的，不算违规）
            for hand_box in person_hand_boxes:
                exposed_box = exposed_box.mask_complement(hand_box, 'exposed', image_width, image_height)
                if exposed_box is None:
                    break

            if exposed_box is None:
                continue

            # 3) 减去手腕（手腕本来就是裸露的，不算违规）
            for wrist_box in person_wrist_boxes:
                exposed_box = exposed_box.mask_complement(wrist_box, 'exposed', image_width, image_height)
                if exposed_box is None:
                    break

            if exposed_box is None:
                continue

            mask_pixels = int(exposed_box.mask_array.sum())
            if mask_pixels >= EXPOSED_MIN_MASK_PIXELS:
                ys, xs = np.where(exposed_box.mask_array)
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                exposed_arm_boxes.append(Box(label='exposed_arm', score=1.0, box=[x1, y1, x2, y2], mask=None, mask_array=exposed_box.mask_array))
        
        return exposed_arm_boxes

    def detect_exposed_leg(self, predictions: list[Box], fences=None, image_width: int = 0, image_height: int = 0) -> list[Box]:
        skin_boxes = label_filter(predictions, ['skin'])
        if not skin_boxes:
            return []
        
        person_boxes = label_filter(predictions, ['person'])
        person_boxes = self.remove_duplicate_person(person_boxes)
        if not person_boxes:
            return []
        
        leg_boxes = label_filter(predictions, ['person leg'])
        if not leg_boxes:
            return []

        # 去除掉和人没有相交的标签
        person_leg_boxes = []
        person_skin_boxes = []
        for person in person_boxes:
            for leg in leg_boxes:
                if person.iom(leg) > 0.5:
                    person_leg_boxes.append(leg)

            for skin in skin_boxes:
                if person.iom(skin) > 0.5:
                    person_skin_boxes.append(skin)
        
        exposed_leg_boxes = []
        for leg_box in person_leg_boxes:
            # 计算 皮肤 ∩ 小腿（小腿上裸露的皮肤区域）
            exposed_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            for skin_box in person_skin_boxes:
                inter = leg_box.mask_intersection(skin_box, 'exposed', image_width, image_height)
                if inter is not None and inter.mask_array is not None:
                    exposed_mask = np.logical_or(exposed_mask, inter.mask_array)

            if not exposed_mask.any():
                continue

            mask_pixels = int(exposed_mask.sum())
            if mask_pixels >= EXPOSED_MIN_MASK_PIXELS:
                ys, xs = np.where(exposed_mask)
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(xs.max())
                exposed_leg_boxes.append(Box(label='exposed_leg', score=1.0, box=[x1, y1, x2, y2], mask=None, mask_array=exposed_mask))
        
        return exposed_leg_boxes

    def detect(self, predictions: list[Box], fences=None, device_id: str = "", image_width: int = 0, image_height: int = 0) -> list[Box]:
        """
        判断人员是否佩戴安全帽、是否裸露小臂、小腿等；
        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，当前安全帽检测不使用，预留以保持接口统一
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :return: 违章目标 Box 列表，label 可能是 'head'（未戴安全帽）、'exposed_arm'（裸露小臂）、'exposed_leg'（裸露小腿）等
        """
        result = []
        
        if device_id:
            if device_id not in self._history:
                self._history[device_id] = []
            self._history[device_id].append(predictions)

        result.extend(self.detect_exposed_arm(predictions, fences=fences, image_width=image_width, image_height=image_height))
        result.extend(self.detect_exposed_leg(predictions, fences=fences, image_width=image_width, image_height=image_height))

        return result
