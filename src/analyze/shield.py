### 判断安全帽是否佩戴
import numpy as np

from utils.obj import Box
from utils.filter import area_filter, score_filter, label_filter


class ShieldDetector:
    """
    是否正确佩戴面罩
    """

    def __init__(self):
        # 按设备ID保存历史帧的检测结果，便于后续做时序分析
        self._history = {}

    def detect_correct_wear_shield(self, predictions: list[Box], fences=None, image_width: int = 0, image_height: int = 0) -> list[Box]:
        result = []
        person_boxes = label_filter(predictions, ['person'])
        person_boxes = area_filter(person_boxes, 1000.0)
        person_boxes = score_filter(person_boxes, 0.7)

        face_boxes = label_filter(predictions, ['face'])
        face_boxes = area_filter(face_boxes, 500.0)
        face_boxes = score_filter(face_boxes, 0.7)

        shield_boxes = label_filter(predictions, ['face shield'])
        shield_boxes = area_filter(shield_boxes, 500.0)
        shield_boxes = score_filter(shield_boxes, 0.7)
        used_shields = set()
        # 1. 为每个人匹配最佳的脸（通过与人的 iom）
        # 2. 判断脸是否和面罩相交来判断是否正确佩戴面罩
        for person in person_boxes:
            best_face = None
            best_face_iom = 0.0
            for face in face_boxes:
                iom_val = person.iom(face)
                if iom_val > best_face_iom:
                    best_face_iom = iom_val
                    best_face = face

            if best_face is not None and best_face_iom > 0.8:
                has_shield = False
                for shield in shield_boxes:
                    if id(shield) in used_shields:
                        continue
                    if best_face.iom(shield) > 0.8:
                        has_shield = True
                        used_shields.add(id(shield))
                        break

                if not has_shield:
                    # 认为这个人没有正确佩戴面罩，返回脸的位置作为违规目标
                    result.append(Box(label='wrong_shield', score=best_face.score, box=best_face.box))
        return result

    def detect(self, predictions: list[Box], fences=None, device_id: str = "", image_width: int = 0, image_height: int = 0) -> list[Box]:
        """
        判断人员是否正确佩戴面罩
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

        result.extend(self.detect_correct_wear_shield(predictions, fences=fences))

        return result
