### 反光衣/工服识别
import os
import time
from typing import List

import cv2
import numpy as np

from utils.obj import Box
from utils.filter import score_filter, label_filter, area_filter, aspect_ratio_filter, edge_filter
from utils.logger import setup_logger

logger = setup_logger("vest")


class VestDetector:
    """
    反光衣/工服缺失检测器（算法码 10）。

    逻辑：
      1. 过滤出有效 person（置信度、面积、宽高比）。
      2. 过滤出衣服框（pants / upper garment）。
      3. 过滤出反光衣/背心框（vest / orange upper garment，由 SAM3 检测）。
      4. 过滤出反光条框（high-visibility strip / high-vis strip / reflective strip，由 SAM3 检测）。
      5. 过滤出红色安全帽框（red hat / red helmet / red hard hat）。
      6. 逐人判定违规：
         - 先判定「合格的人」（有效目标）：人 与 任意衣服框（pants / upper garment）的
           IoM > 0.8（确认是穿有衣物的完整人员），不满足则跳过，避免对异常/截断目标误报。
         - 合格的人 满足任一豁免条件即不报警：
           a. 与 任意反光衣/背心框（vest / orange upper garment）的 IoM > 0.5；
           b. 与 任意反光条框（reflective strip 等）的 mask IoM > 0.5（按 mask 像素级相交计算）；
           c. 已获红色安全帽豁免（保留贪心匹配：一件红帽只豁免一个人）。

    返回的 Box 为违规「person」框。
    分类二次确认放在下游（classify_for_task，按 algorithm_classifiers 配置）进行，
    这里不做分类，以利用 SAM3 的 vest 检测先过滤掉一部分合规人员、减少分类压力。
    """

    # 衣服标签集合（pants / upper garment）
    CLOTHES_LABELS = {"pants", "upper garment"}
    # 反光衣/背心标签集合（SAM3 检测）
    VEST_LABELS = {"vest", "orange upper garment"}
    # 反光条标签集合（SAM3 检测；与上衣有交集时认为该上衣为反光工服，豁免）
    STRIP_LABELS = {"high-visibility strip", "high-vis strip", "reflective strip"}
    # 红色安全帽标签集合（佩戴时认为已做安全着装，排除误报）
    RED_HAT_LABELS = {"red hat", "red helmet", "red hard hat"}

    def __init__(
        self,
        person_min_score: float = 0.85,
        person_min_area: float = 10000.0,
        clothes_min_score: float = 0.7,
        clothes_min_area: float = 3000.0,
        io_min_thresh: float = 0.5,
        min_ratio: float = 0.15,
        max_ratio: float = 4.0,
        clothes_io_thresh: float = 0.8,
        vest_io_thresh: float = 0.5,
        strip_min_score: float = 0.35,
        debug_save_dir: str = "debug",
    ):
        """
        :param person_min_score: person 最低置信度
        :param person_min_area: person 最小面积（像素）
        :param clothes_min_score: 衣服框最低置信度
        :param clothes_min_area: 衣服框最小面积（像素）
        :param io_min_thresh: 红帽豁免 IoM（交集/最小面积）阈值，超过认为两框相关
        :param min_ratio: 最小宽高比（宽/高）
        :param max_ratio: 最大宽高比（宽/高）
        :param clothes_io_thresh: 人 与 衣服框（pants / upper garment）的 IoM 阈值（超过视为合格的人/有效目标）
        :param vest_io_thresh: 人 与 反光衣/背心/反光条的 IoM 阈值（超过视为合格）
        :param strip_min_score: 反光条最低置信度
        :param debug_save_dir: 调试用 mask 叠加图保存目录（相对项目根目录）；None 或空串表示不保存
        """
        self.person_min_score = person_min_score
        self.person_min_area = person_min_area
        self.clothes_min_score = clothes_min_score
        self.clothes_min_area = clothes_min_area
        self.io_min_thresh = io_min_thresh
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.clothes_io_thresh = clothes_io_thresh
        self.vest_io_thresh = vest_io_thresh
        self.strip_min_score = strip_min_score
        self.debug_save_dir = debug_save_dir

    @staticmethod
    def _greedy_exempt(
        sources: List[Box],
        boxes: List[Box],
        io_min_thresh: float,
        use_mask: bool = False,
        image_width: int = 0,
        image_height: int = 0,
    ) -> set:
        """
        贪心匹配豁免：每件豁免装备（反光衣/红帽/反光条）只豁免与之相交最大的一个源目标
        （person 或上衣），避免一件装备豁免多个目标。返回获得豁免的源目标 id 集合。
        use_mask=True 时按 mask 像素级相交（mask_iom）计算，否则按 bbox IoM。
        """
        exempt_ids = set()
        used_box_ids = set()
        pairs = []
        for src in sources:
            for box in boxes:
                if use_mask:
                    iom = src.mask_iom(box, image_width, image_height)
                else:
                    iom = src.iom(box)
                if iom > io_min_thresh:
                    pairs.append((iom, id(src), src, id(box), box))
        pairs.sort(key=lambda x: x[0], reverse=True)

        for _, sid, _, bid, _ in pairs:
            if sid in exempt_ids:
                continue  # 该目标已获豁免，不再占用其他装备，让给其他目标
            if bid in used_box_ids:
                continue  # 这件装备已匹配给其他目标
            exempt_ids.add(sid)
            used_box_ids.add(bid)
        return exempt_ids

    def _save_mask_debug_image(
        self,
        frame: np.ndarray,
        persons: list,
        clothes_boxes: list,
        vest_boxes: list,
        strip_boxes: list,
        red_hat_boxes: list,
        image_width: int,
        image_height: int,
        device_id: str = "",
    ) -> None:
        """
        调试用：把 person / 衣服 / 马甲 / 反光条 / 红帽 的 mask 用不同颜色叠加绘制到原图上并保存，
        用于排查「本地图片 vs 视频流」检测结果不一致时 mask 的具体分布。

        颜色约定：
          - 衣服（pants/upper garment）：橙
          - 红帽：紫
          - 马甲（vest/orange upper garment）：绿
          - 反光条（reflective strip 等）：黄
          - 人（person）：红
        """
        try:
            if frame is None or image_width <= 0 or image_height <= 0:
                return
            if not self.debug_save_dir:
                return

            overlay = frame.copy()
#
            def _paint(boxes, color):
                for box in boxes:
                    mask = box.compute_mask_array(image_width, image_height)
                    if mask is None:
                        continue
                    overlay[mask > 0] = color

            _paint(clothes_boxes, (255, 128, 0))    # 衣服：橙
            _paint(red_hat_boxes, (255, 0, 255))    # 红帽：紫
            _paint(vest_boxes, (0, 255, 0))         # 马甲：绿
            _paint(strip_boxes, (0, 255, 255))      # 反光条：黄
            #_paint(persons, (0, 0, 255))            # 人：红
            for strip in strip_boxes:
                x1, y1, x2, y2 = map(int, strip.box)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)
            img = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            save_dir = os.path.join(project_root, self.debug_save_dir)
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"vest_mask_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(path, img)
            logger.info(f"[device={device_id}] 已保存 vest mask 调试图: {path}")
        except Exception as e:
            logger.error(f"[device={device_id}] 保存 vest mask 调试图失败: {e}")

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
        frame: np.ndarray = None,
    ) -> list[Box]:
        """
        反光衣/工服缺失检测入口。

        :param predictions: 当前帧所有检测框
        :param fences: 区域坐标列表，当前检测不使用，预留以保持接口统一
        :param device_id: 设备ID
        :param image_width: 原图宽度
        :param image_height: 原图高度
        :param frame: 当前帧图像（BGR），用于保存 mask 叠加调试图（可选；analyze_for_task 会自动传入）
        :return: 违规 person Box 列表
        """
        # 1. 过滤有效 person
        persons = label_filter(predictions, ["person"])
        persons = score_filter(persons, self.person_min_score)
        persons = area_filter(persons, self.person_min_area)
        persons = aspect_ratio_filter(persons, self.min_ratio, self.max_ratio)

        # 2. 过滤衣服框（pants / upper garment）
        clothes_boxes = label_filter(predictions, list(self.CLOTHES_LABELS))
        clothes_boxes = score_filter(clothes_boxes, self.clothes_min_score)
        clothes_boxes = area_filter(clothes_boxes, self.clothes_min_area)
        clothes_boxes = aspect_ratio_filter(clothes_boxes, self.min_ratio, self.max_ratio)
        # 边缘过滤：排除紧贴图像边缘的截断/误检衣服框
        clothes_boxes = edge_filter(clothes_boxes, image_width, image_height, 0.02)

        # 3. 过滤反光衣/背心框（SAM3 检测）
        vest_boxes = label_filter(predictions, list(self.VEST_LABELS))
        vest_boxes = score_filter(vest_boxes, 0.45)

        # 4. 过滤反光条框（SAM3 检测；置信度阈值较低，默认 0.35）
        strip_boxes = label_filter(predictions, list(self.STRIP_LABELS))
        strip_boxes = score_filter(strip_boxes, self.strip_min_score)

        # 5. 过滤红色安全帽框
        red_hat_boxes = label_filter(predictions, list(self.RED_HAT_LABELS))
        red_hat_boxes = score_filter(red_hat_boxes, self.clothes_min_score)

        # 6. 红帽豁免：一件红帽只豁免一个人（避免一顶帽子豁免多人，保留原逻辑）
        red_hat_exempt_person_ids = self._greedy_exempt(persons, red_hat_boxes, self.io_min_thresh)

        # 7. 逐人判定违规（返回的是 person 框）
        violators = []
        for person in persons:
            # 合格的人（有效目标）：人 与 任意衣服框（pants / upper garment）的 IoM > 0.8，
            # 确认该人员是穿有衣物的完整目标；不满足则跳过（避免对异常/截断目标误报）
            if not any(
                person.iom(clothes) > self.clothes_io_thresh for clothes in clothes_boxes
            ):
                continue

            # 豁免 a：合格的人 与 任意反光衣/背心框（vest / orange upper garment）的 IoM > 0.5 → 不报警
            if any(
                person.iom(vest) > self.vest_io_thresh for vest in vest_boxes
            ):
                continue

            # 豁免 b：合格的人 与 任意反光条框 的 mask IoM > 0.5（按 mask 像素级相交计算）→ 不报警
            if any(
                person.mask_iom(strip, image_width, image_height) > self.vest_io_thresh
                for strip in strip_boxes
            ):
                continue

            # 豁免 c：合格的人 已获红色安全帽豁免 → 不报警
            if id(person) in red_hat_exempt_person_ids:
                continue

            logger.warning(
                f"[device={device_id}] 未穿反光衣/工服 | "
                f"box={person.box}, score={person.score:.3f}"
            )
            violators.append(person)
        if violators:
            # 8. 调试：保存 mask 叠加调试图（排查本地图片 vs 视频流结果差异）
            self._save_mask_debug_image(
                frame, persons, clothes_boxes, vest_boxes, strip_boxes, red_hat_boxes,
                image_width, image_height, device_id,
            )

        return violators
