### 煤流异物检测
from utils.obj import Box
from utils.filter import score_filter, label_filter, area_filter
from utils.logger import setup_logger

logger = setup_logger("coal_foreign_object")


class CoalForeignObjectDetector:
    """
    煤流异物检测器（算法码 50）。

    逻辑：
      1. 过滤出 conveyor belt（皮带/煤流区域）。
      2. 过滤出 YOLO 预检测到的异物候选框（如 dakuai）。
      3. 仅当异物候选框位于皮带区域内时，才将该皮带区域作为候选返回，
         由 VL 大模型进一步复核判断是否存在异物（石块、木头、金属、大块异物等）。

    返回的 Box label 为 "conveyor_belt"，会进入 VL 二次复核流程。
    """

    def __init__(
        self,
        belt_min_score: float = 0.7,
        belt_min_area: float = 1000.0,
        foreign_object_labels: list[str] = None,
        foreign_min_score: float = 0.5,
        foreign_min_area: float = 100.0,
        min_iou_with_belt: float = 0.95,
    ):
        """
        :param belt_min_score: conveyor belt 最低置信度
        :param belt_min_area: conveyor belt 最小面积（像素）
        :param foreign_object_labels: 视为异物候选的 YOLO 标签，默认 ["dakuai"]
        :param foreign_min_score: 异物候选框最低置信度
        :param foreign_min_area: 异物候选框最小面积（像素）
        :param min_iou_with_belt: 异物候选框与皮带区域的最小 IoU，低于此值认为不在皮带上
        """
        self.belt_min_score = belt_min_score
        self.belt_min_area = belt_min_area
        self.foreign_object_labels = foreign_object_labels or ["dakuai"]
        self.foreign_min_score = foreign_min_score
        self.foreign_min_area = foreign_min_area
        self.min_iou_with_belt = min_iou_with_belt

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> list[Box]:
        """
        煤流异物检测入口。

        :param predictions: 当前帧所有检测框（含 SAM3 的 conveyor belt 和 YOLO 的异物候选）
        :param fences: 区域坐标列表，当前煤流异物检测不使用，预留以保持接口统一
        :param device_id: 设备ID
        :param image_width: 原图宽度
        :param image_height: 原图高度
        :return: conveyor belt 候选区域 Box 列表，label 为 "conveyor_belt"
        """
        # 1. 过滤皮带区域
        belt_boxes = label_filter(predictions, ["conveyor belt"])
        belt_boxes = score_filter(belt_boxes, self.belt_min_score)
        belt_boxes = area_filter(belt_boxes, self.belt_min_area)
        if not belt_boxes:
            return []

        # 2. 过滤异物候选框
        foreign_boxes = label_filter(predictions, self.foreign_object_labels)
        foreign_boxes = score_filter(foreign_boxes, self.foreign_min_score)
        foreign_boxes = area_filter(foreign_boxes, self.foreign_min_area)
        if not foreign_boxes:
            logger.info(f"[device={device_id}] 煤流异物检测 | 未检测到异物候选框 {self.foreign_object_labels}，跳过 VL 复核")
            return []

        # 3. 仅保留有异物候选框落在其上的皮带区域
        result = []
        used_belt_ids = set()
        for belt in belt_boxes:
            bid = id(belt)
            if bid in used_belt_ids:
                continue

            matched_foreigns = []
            for foreign in foreign_boxes:
                # IoU 或中心点落在皮带内均可视为“在皮带上”
                iou = belt.iou(foreign)
                if iou >= self.min_iou_with_belt:
                    matched_foreigns.append(foreign)
                    continue
                cx, cy = foreign.center
                bx1, by1, bx2, by2 = belt.box
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    matched_foreigns.append(foreign)

            if not matched_foreigns:
                continue

            violation = Box(
                label="conveyor_belt",
                score=belt.score,
                box=list(belt.box),
                mask=belt.mask,
            )
            foreign_info = [
                f"{f.label}({f.score:.3f},{[round(v) for v in f.box]})"
                for f in matched_foreigns
            ]
            logger.info(
                f"[device={device_id}] 煤流异物候选区域 | "
                f"box={violation.box}, score={violation.score:.3f}, "
                f"匹配异物={foreign_info}"
            )
            result.append(violation)
            used_belt_ids.add(bid)

        return result
