### 判断安全帽是否佩戴
import numpy as np
import cv2

from utils.obj import Box
from utils.filter import area_filter, score_filter, label_filter, edge_filter
from utils.logger import setup_logger

logger = setup_logger("helmet")


# 判断安全帽是否佩戴、是否裸露小臂、小腿等；判断是否穿手套
class HelmetDetector:
    """
    安全帽是否佩戴、是否裸露小臂、小腿等；判断是否穿手套检测器，按设备维度保存历史状态
    """

    def __init__(
        self,
        dark_thresh: int = 60,
        backlit_dark_ratio: float = 0.8,
        bright_thresh: int = 200,
        backlit_bright_ratio: float = 0.8,
        enable_backlight_filter: bool = False,
    ):
        """
        :param dark_thresh: 判定“暗像素”的灰度阈值（0-255），低于该值的像素视为暗像素
        :param backlit_dark_ratio: 判定区域内暗像素占比达到该值即判定为逆光（过暗）
        :param bright_thresh: 判定“亮像素”的灰度阈值（0-255），高于该值的像素视为亮像素
        :param backlit_bright_ratio: 判定区域内亮像素占比达到该值即判定为过曝（过亮）
        :param enable_backlight_filter: 是否启用光照异常区域排除（关闭则保持原逻辑）
        """
        # 按设备ID保存历史帧的检测结果，便于后续做时序分析
        self._history = {}

        # 光照异常排除参数（传统图像算法）
        self.dark_thresh = dark_thresh
        self.backlit_dark_ratio = backlit_dark_ratio
        self.bright_thresh = bright_thresh
        self.backlit_bright_ratio = backlit_bright_ratio
        self.enable_backlight_filter = enable_backlight_filter

    # ---------- 光照异常区域判断（传统图像算法） ----------
    def _is_backlit_region(self, frame: np.ndarray, box: Box) -> bool:
        """
        用传统图像算法判断某目标区域是否光照异常（过暗或过亮）。
        思路：把目标区域（直接使用目标框，不外扩）转灰度，统计暗像素占比与亮像素占比；
        暗像素占比过高说明该区域整体过暗（逆光/阴影），
        亮像素占比过高说明该区域整体过亮（过曝），
        这两种情况下检测结果不可靠，应排除。
        """
        if frame is None:
            return False
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box.box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False

        region = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        dark_ratio = float((gray < self.dark_thresh).mean())
        bright_ratio = float((gray > self.bright_thresh).mean())
        if dark_ratio >= self.backlit_dark_ratio:
            logger.info(
                f"[安全帽] 逆光(过暗)区域 | box={box.box} | 暗像素占比={dark_ratio:.3f} "
                f">= {self.backlit_dark_ratio} (thresh={self.dark_thresh})"
            )
            return True
        if bright_ratio >= self.backlit_bright_ratio:
            logger.info(
                f"[安全帽] 过曝(过亮)区域 | box={box.box} | 亮像素占比={bright_ratio:.3f} "
                f">= {self.backlit_bright_ratio} (thresh={self.bright_thresh})"
            )
            return True
        return False

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

    def detect_without_helmet(self, predictions: list[Box], fences=None, image_width: int = 0, image_height: int = 0, frame: np.ndarray = None) -> list[Box]:
        min_person_area = 10000
        min_head_area = 1200
        if image_width <= 1280 or image_height <= 720:
            min_person_area = 4000
            min_head_area = 300

        result = []
        person_boxes = label_filter(predictions, ['person'])
        person_boxes = [p for p in person_boxes if p.source == "SAM3"]
        person_boxes = area_filter(person_boxes, min_person_area)
        person_boxes = score_filter(person_boxes, 0.8)
        person_boxes = self.remove_duplicate_person(person_boxes)

        driver_boxes = label_filter(predictions, ['driver'])
        driver_boxes = [p for p in driver_boxes if p.source == "SAM3"]
        driver_boxes = area_filter(driver_boxes, min_person_area)
        driver_boxes = score_filter(driver_boxes, 0.8)
        driver_boxes = self.remove_duplicate_person(driver_boxes)

        helmet_boxes = label_filter(predictions, ['helmet', 'hard hat', 'hat'])
        # helmet_boxes = label_filter(predictions, ['helmet', 'hard hat'])
        helmet_boxes = self.remove_duplicate_helmets(helmet_boxes)

        head_boxes = label_filter(predictions, ['head'])
        head_boxes = area_filter(head_boxes, min_head_area)
        head_boxes = score_filter(head_boxes, 0.8)
        head_boxes = self.remove_duplicate_heads(head_boxes)
        head_boxes = edge_filter(head_boxes,image_width, image_height, 0.02)

        head_boxes = self.remove_duplicate_heads(head_boxes)
        helmet_boxes = self.remove_duplicate_helmets(helmet_boxes)

        umbrella_boxes = label_filter(predictions, ['umbrella'])
        umbrella_boxes = score_filter(umbrella_boxes ,0.4)

        person_except_driver_boxes = []
        for person in person_boxes:
            is_driver = False
            for driver in driver_boxes:
                if person.iom(driver) > 0.9:
                    is_driver = True

            if not is_driver:
                person_except_driver_boxes.appen(person)

        used_heads = set()
        for person in person_except_driver_boxes:
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
            has_umbrella = False
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

                for umbrella in umbrella_boxes:
                    if head.iom(umbrella) > 0.0001:
                        has_umbrella = True
                        break
                if has_umbrella:
                    break

            for head in matched_heads:
                used_heads.add(id(head))

            if not has_helmet and not has_umbrella:
                # 报违规：取面积最大的 matched_head 作为代表
                target_head = max(matched_heads, key=lambda h: h.area())
                # 逆光区域排除：传统算法判断头部区域是否过暗（逆光），是则跳过，避免误报
                if (
                    self.enable_backlight_filter
                    and frame is not None
                    and self._is_backlit_region(frame, target_head)
                ):
                    print(f"[安全帽] 逆光区域排除 | head={target_head.box}")
                    continue
                result.append(target_head)
        return result

    def detect(self, predictions: list[Box], fences=None, device_id: str = "", image_width: int = 0, image_height: int = 0, frame: np.ndarray = None) -> list[Box]:
        """
        判断人员是否佩戴安全帽；
        :param predictions: Box 对象列表，包含 label、score、box 坐标等信息
        :param fences: 区域坐标列表，当前安全帽检测不使用，预留以保持接口统一
        :param device_id: 设备ID，用于区分不同设备的检测历史；为空则不保存历史状态
        :param frame: 当前帧图像（BGR），用于逆光区域判断；为 None 时不做逆光排除
        :return: 违章目标 Box 列表，label 可能是 'head'（未戴安全帽）、'exposed_arm'（裸露小臂）、'exposed_leg'（裸露小腿）等
        """
        result = []
         
        # if device_id:
        #     if device_id not in self._history:
        #         self._history[device_id] = []
        #     self._history[device_id].append(predictions)
        
        result.extend(self.detect_without_helmet(
            predictions,
            fences=fences,
            image_width=image_width,
            image_height=image_height,
            frame=frame,
        ))

        return result
