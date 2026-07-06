import numpy as np

from utils.obj import Box
from utils.filter import area_filter, score_filter, label_filter


class SafetyDetector:
    def __init__(self):
        pass
    
    def detect_person_not_safe(self, predictions: list[Box], fences=None, image_width: int = 0, image_height: int = 0) -> list[Box]:
        platform_boxes = label_filter(predictions, ['elevating work platform'])
        
        person_boxes = label_filter(predictions, ['person'])
        person_boxes = area_filter(person_boxes, 10000.0)
        person_boxes = score_filter(person_boxes, 0.7)

        if len(person_boxes) == 0:
            return []

        if len(platform_boxes) == 0:
            return []

        # platform_persons = []
        # for platform in platform_boxes:
        #     for person in person_boxes:
        #         on_platform = False
        #         if image_width > 0 and image_height > 0:
        #             inter = platform.mask_intersection(person, 'inter', image_width, image_height)
        #             if inter is not None and inter.mask_array is not None:
        #                 person_mask = person.compute_mask_array(image_width, image_height)
        #                 if person_mask is not None:
        #                     inter_pixels = int(inter.mask_array.sum())
        #                     person_pixels = int(person_mask.sum())
        #                     if person_pixels > 0 and inter_pixels / person_pixels > 0.5:
        #                         on_platform = True
        #         if not on_platform and platform.iom(person) > 0.9:
        #             on_platform = True
        #         if on_platform:
        #             platform_persons.append(person)
        platform_persons = person_boxes  # 先不做人员与平台的关联了，后续如果需要可以单独做一个专门的人员平台关联模型
        if len(platform_persons) == 0:
            return []

        belt_boxes = label_filter(predictions, ['harness'])
        # belt_boxes = area_filter(belt_boxes, 500.0)

        safety_rope_boxes = label_filter(predictions, ['safety rope', 'rolled safety rope'])
        # safety_rope_boxes = area_filter(safety_rope_boxes, 500.0)



        """
        安全绳检测困难且误报较多，先不做安全绳检测了，后续如果需要可以单独做一个专门的安全绳检测模型
        直接在结果增加 no_safety_rope，用于后续大模型二次判断
        """
        result = []
        for person in platform_persons:
            has_belt = False
            has_safety_rope = False
            for belt in belt_boxes:
                if person.iom(belt) > 0.001:
                    has_belt = True
                    break
            # for rope in safety_rope_boxes:
            #     expand_person_box = person.expand_box("all", 0.3)  # 扩大人员框以适应安全绳可能的位置
            #     if expand_person_box.iom(rope) > 0.001:
            #         has_safety_rope = True
            #         break
            if not has_belt:
                result.append(Box(label='no_belt_and_no_rope', score=person.score, box=person.box))
            else:
                result.append(Box(label='no_rope', score=person.score, box=person.box))
        return result

    

    def detect(self, predictions: list[Box], fences=None, device_id: str = "", image_width: int = 0, image_height: int = 0) -> list[Box]:
        """
        综合分析人员是否存在安全隐患（如未系安全带、未使用安全绳等）
        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，当前安全检测不使用，预留以保持接口统一
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        """
        # 这里可以调用 detect_person_not_safe 等方法，并合并结果返回
        return self.detect_person_not_safe(predictions, fences, image_width, image_height)