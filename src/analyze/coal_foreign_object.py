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
      2. 按面积、置信度过滤，保留有效区域。
      3. 将 conveyor belt 框作为候选区域返回，由 VL 大模型进一步复核
         判断该区域是否存在异物（石块、木头、金属、大块异物等）。

    返回的 Box label 为 "conveyor_belt"，会进入 VL 二次复核流程。
    """

    def __init__(
        self,
        belt_min_score: float = 0.5,
        belt_min_area: float = 1000.0,
    ):
        """
        :param belt_min_score: conveyor belt 最低置信度
        :param belt_min_area: conveyor belt 最小面积（像素）
        """
        self.belt_min_score = belt_min_score
        self.belt_min_area = belt_min_area

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

        :param predictions: 当前帧所有检测框
        :param fences: 区域坐标列表，当前煤流异物检测不使用，预留以保持接口统一
        :param device_id: 设备ID
        :param image_width: 原图宽度
        :param image_height: 原图高度
        :return: conveyor belt 候选区域 Box 列表，label 为 "conveyor_belt"
        """
        belt_boxes = label_filter(predictions, ["conveyor belt"])
        belt_boxes = score_filter(belt_boxes, self.belt_min_score)
        belt_boxes = area_filter(belt_boxes, self.belt_min_area)

        if not belt_boxes:
            return []

        result = []
        used_ids = set()
        for belt in belt_boxes:
            bid = id(belt)
            if bid in used_ids:
                continue

            violation = Box(
                label="conveyor_belt",
                score=belt.score,
                box=list(belt.box),
                mask=belt.mask,
            )
            logger.info(
                f"[device={device_id}] 煤流异物候选区域 | "
                f"box={violation.box}, score={violation.score:.3f}"
            )
            result.append(violation)
            used_ids.add(bid)

        return result
