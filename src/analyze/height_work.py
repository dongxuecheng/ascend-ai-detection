### 登高无安全带检测（专门用于算法码 58）
from utils.obj import Box
from utils.filter import score_filter, label_filter
from utils.logger import setup_logger

logger = setup_logger("height_work")


class HeightWorkDetector:
    """
    登高无安全带检测器（专门用于算法码 58）：
    检测到 person-on-ladder / person-on-scaffolding 与梯子/脚手架相交，
    且未佩戴 safety harness / harness 时，判定为违规。

    安全带置信度阈值单独设置（默认 0.3），便于在低置信度下也能识别到安全带。
    """

    def __init__(
        self,
        person_score_thresh: float = 0.6,
        equipment_score_thresh: float = 0.6,
        belt_score_thresh: float = 0.3,
        expand_pixel: int = 20,
    ):
        """
        :param person_score_thresh: 登高人员框最低置信度
        :param equipment_score_thresh: 梯子/脚手架框最低置信度
        :param belt_score_thresh: 安全带置信度阈值（设置较低，便于检出微弱安全带）
        :param expand_pixel: 报警框外扩像素数
        """
        self.person_score_thresh = person_score_thresh
        self.equipment_score_thresh = equipment_score_thresh
        self.belt_score_thresh = belt_score_thresh
        self.expand_pixel = expand_pixel

    @staticmethod
    def _intersection_area(box1: list, box2: list) -> float:
        """计算两个框的交集面积"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> list[Box]:
        """
        登高无安全带检测入口。

        逻辑：
        1. 过滤 person-on-ladder / person-on-scaffolding（登高人员）。
        2. 过滤 ladder / scaffolding（梯子/脚手架）。
        3. 过滤 safety harness / harness（安全带，使用较低阈值）。
        4. 对每个登高人员，判断是否与梯子/脚手架有交集。
        5. 对登高人员，检查是否有安全带与其有交集。
        6. 无安全带时，返回以人员+设备合并框外扩后的报警框，label 为 "alarm-no-safety-belt"。

        :param predictions: Box 对象列表
        :param fences: 区域坐标列表，当前登高检测不使用，预留以保持接口统一
        :param device_id: 设备ID
        :param image_width: 原图宽度，用于扩展框边界裁剪
        :param image_height: 原图高度，用于扩展框边界裁剪
        :return: 违规目标 Box 列表
        """
        person_boxes = label_filter(
            predictions,
            ["person-on-ladder", "person-on-scaffolding"],
        )
        person_boxes = score_filter(person_boxes, self.person_score_thresh)

        equipment_boxes = label_filter(
            predictions,
            ["ladder", "scaffolding"],
        )
        equipment_boxes = score_filter(equipment_boxes, self.equipment_score_thresh)

        belt_boxes = label_filter(
            predictions,
            ["safety harness", "harness"],
        )
        belt_boxes = score_filter(belt_boxes, self.belt_score_thresh)

        if not person_boxes or not equipment_boxes:
            return []

        # 图像边界，用于扩展框裁剪
        max_x = float(image_width - 1) if image_width > 0 else float("inf")
        max_y = float(image_height - 1) if image_height > 0 else float("inf")

        result = []
        used_persons = set()

        for person in person_boxes:
            pid = id(person)
            if pid in used_persons:
                continue

            # A. 判断登高人员是否与梯子/脚手架相交
            target_equipment = None
            for equipment in equipment_boxes:
                if self._intersection_area(person.box, equipment.box) > 0:
                    target_equipment = equipment
                    break  # 只要在一个设备上就算

            if target_equipment is None:
                continue

            # B. 判断该人员是否佩戴安全带（与人员框有交集即认为佩戴）
            has_belt = False
            for belt in belt_boxes:
                if self._intersection_area(person.box, belt.box) > 0:
                    has_belt = True
                    break

            # C. 在高处且没有安全带，触发报警
            if not has_belt:
                # 报警框为人员与设备的合并框，再外扩 expand_pixel
                u_x1 = min(person.box[0], target_equipment.box[0])
                u_y1 = min(person.box[1], target_equipment.box[1])
                u_x2 = max(person.box[2], target_equipment.box[2])
                u_y2 = max(person.box[3], target_equipment.box[3])

                ex_x1 = max(0.0, u_x1 - self.expand_pixel)
                ex_y1 = max(0.0, u_y1 - self.expand_pixel)
                ex_x2 = min(max_x, u_x2 + self.expand_pixel)
                ex_y2 = min(max_y, u_y2 + self.expand_pixel)

                violation = Box(
                    label="alarm-no-safety-belt",
                    score=person.score,
                    box=[ex_x1, ex_y1, ex_x2, ex_y2],
                    mask=person.mask,
                )
                logger.warning(
                    f"[device={device_id}] 登高无安全带违规触发 | "
                    f"box={violation.box}, score={violation.score:.3f}"
                )
                result.append(violation)
                used_persons.add(pid)

        return result
