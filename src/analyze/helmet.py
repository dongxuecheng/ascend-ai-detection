### 判断安全帽是否佩戴
import numpy as np

from utils.obj import Box
from utils.filter import area_filter, score_filter, label_filter


# 判断安全帽是否佩戴、是否裸露小臂、小腿等；判断是否穿手套
class HelmetDetector:
    """
    安全帽是否佩戴、是否裸露小臂、小腿等；判断是否穿手套检测器，按设备维度保存历史状态
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

    # 去除一个安全帽完全包含在另一个安全帽内的重复框
    def remove_duplicate_helmets(self, helmet_boxes: list[Box]) -> list[Box]:
        return self._remove_duplicates(helmet_boxes)

    # 去除一个头完全包含在另一个头内的重复框
    def remove_duplicate_heads(self, head_boxes: list[Box]) -> list[Box]:
        return self._remove_duplicates(head_boxes)

    def remove_duplicate_person(self, person_boxes: list[Box]) -> list[Box]:
        return self._remove_duplicates(person_boxes)

    def detect_without_helmet(self, predictions: list[Box], fences=None, image_width: int = 0, image_height: int = 0) -> list[Box]:
        # print(predictions)
        result = []
        person_boxes = label_filter(predictions, ['person'])
        # person_boxes = area_filter(person_boxes, 1000.0)
        person_boxes = score_filter(person_boxes, 0.7)
        person_boxes = self.remove_duplicate_person(person_boxes)

        helmet_boxes = label_filter(predictions, ['helmet', 'hard hat', 'hat'])
        helmet_boxes = self.remove_duplicate_helmets(helmet_boxes)

        head_boxes = label_filter(predictions, ['head'])
        # head_boxes = area_filter(head_boxes, 500.0)
        # head_boxes = score_filter(head_boxes, 0.6)
        head_boxes = self.remove_duplicate_heads(head_boxes)

        head_boxes = self.remove_duplicate_heads(head_boxes)
        helmet_boxes = self.remove_duplicate_helmets(helmet_boxes)

        used_heads = set()
        for person in person_boxes:
            # 根据人员宽高比决定裁剪比例
            if person.whration() < 0.5:
                person_top = person.cut_box('top', 0.5)
            else:
                person_top = person.cut_box('top', 0)

            # Step 1: 收集所有与这个人上半身有足够重叠的 head（允许多个）
            matched_heads = []
            for head in head_boxes:
                if id(head) in used_heads:
                    continue
                if person_top.iom(head) > 0.8:
                    matched_heads.append(head)

            if not matched_heads:
                continue

            # Step 2: 这些 head 中，只要任意一个能匹配到 helmet 就算佩戴
            has_helmet = False
            used_helmets = set()
            for head in matched_heads:
                for helmet in helmet_boxes:
                    if id(helmet) in used_helmets:
                        continue
                    # 同时检查：helmet 与原始 head 框、扩展后 head 框、person 上半部分的重叠
                    if head.iom(helmet) > 0.5:
                        has_helmet = True
                        used_helmets.add(id(helmet))
                        break
                    expand_head = head.expand_box('top', 0.5)
                    if expand_head.iom(helmet) > 0.5:
                        has_helmet = True
                        used_helmets.add(id(helmet))
                        break
                    if person_top.iom(helmet) > 0.5:
                        has_helmet = True
                        used_helmets.add(id(helmet))
                        break
                if has_helmet:
                    break

            for head in matched_heads:
                used_heads.add(id(head))

            if not has_helmet:
                # 报违规：取面积最大的 matched_head 作为代表
                result.append(max(matched_heads, key=lambda h: h.area()))
        return result

    def detect(self, predictions: list[Box], fences=None, device_id: str = "", image_width: int = 0, image_height: int = 0) -> list[Box]:
        """
        判断人员是否佩戴安全帽；
        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，当前安全帽检测不使用，预留以保持接口统一
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :return: 违章目标 Box 列表，label 可能是 'head'（未戴安全帽）、'exposed_arm'（裸露小臂）、'exposed_leg'（裸露小腿）等
        """
        result = []
         
        # if device_id:
        #     if device_id not in self._history:
        #         self._history[device_id] = []
        #     self._history[device_id].append(predictions)
        result.extend(self.detect_without_helmet(predictions, fences=fences))

        return result