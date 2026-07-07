### 反光衣/工服识别
from utils.obj import Box
from utils.filter import score_filter, label_filter, area_filter, aspect_ratio_filter
from utils.logger import setup_logger

logger = setup_logger("vest")


class VestDetector:
    """
    反光衣缺失检测器（算法码 10）。

    逻辑：
      1. 过滤出有效 person（置信度、面积、宽高比）。
      2. 过滤出普通衣物框（clothes / work clothes 等）。
      3. 过滤出反光衣/安全背心框（reflective vest、vest、uniform 等）。
      4. 过滤出红色安全帽框（red hat / red helmet / red hard hat）。
      5. 当某个人与普通衣物有明显重叠（IoM > 0.5），但与反光衣/安全背心无重叠，
         且未佩戴红色安全帽时，判定为未穿反光衣/工服违规。

    返回的 Box 为违规 person 框。
    """

    # 普通衣物标签集合
    CLOTHES_LABELS = {"clothes", "work clothes", "clothing"}
    # 反光衣/安全背心标签集合
    VEST_LABELS = {
        "reflective vest",
        "reflective clothing",
        "high-visibility strip",
        "high-vis strip",
        "vest",
        "orange clothes",
        "reflective strip",
        "uniform",
    }
    # 红色安全帽标签集合（佩戴时认为已做安全着装，排除误报）
    RED_HAT_LABELS = {"red hat", "red helmet", "red hard hat"}

    def __init__(
        self,
        person_min_score: float = 0.7,
        person_min_area: float = 2500.0,
        clothes_min_score: float = 0.7,
        clothes_min_area: float = 1500.0,
        io_min_thresh: float = 0.5,
        min_ratio: float = 0.25,
        max_ratio: float = 4.0,
    ):
        """
        :param person_min_score: person 最低置信度
        :param person_min_area: person 最小面积（像素）
        :param clothes_min_score: 衣物框最低置信度
        :param clothes_min_area: 衣物框最小面积（像素）
        :param io_min_thresh: IoM（交集/最小面积）阈值，超过认为两框相关
        :param min_ratio: 最小宽高比（宽/高）
        :param max_ratio: 最大宽高比（宽/高）
        """
        self.person_min_score = person_min_score
        self.person_min_area = person_min_area
        self.clothes_min_score = clothes_min_score
        self.clothes_min_area = clothes_min_area
        self.io_min_thresh = io_min_thresh
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def _has_overlap(self, person: Box, boxes: list[Box]) -> bool:
        """判断 person 是否与给定框列表中任意一个的 IoM 超过阈值。"""
        for box in boxes:
            if person.iom(box) > self.io_min_thresh:
                return True
        return False

    def detect(
        self,
        predictions: list[Box],
        fences=None,
        device_id: str = "",
        image_width: int = 0,
        image_height: int = 0,
    ) -> list[Box]:
        """
        反光衣/工服缺失检测入口。

        :param predictions: 当前帧所有检测框
        :param fences: 区域坐标列表，当前检测不使用，预留以保持接口统一
        :param device_id: 设备ID
        :param image_width: 原图宽度
        :param image_height: 原图高度
        :return: 违规 person Box 列表
        """
        # 1. 过滤有效 person
        persons = label_filter(predictions, ["person"])
        persons = score_filter(persons, self.person_min_score)
        persons = area_filter(persons, self.person_min_area)
        persons = aspect_ratio_filter(persons, self.min_ratio, self.max_ratio)

        # 2. 过滤普通衣物框
        clothes_boxes = label_filter(predictions, list(self.CLOTHES_LABELS))
        clothes_boxes = score_filter(clothes_boxes, self.clothes_min_score)
        clothes_boxes = area_filter(clothes_boxes, self.clothes_min_area)
        clothes_boxes = aspect_ratio_filter(clothes_boxes, self.min_ratio, self.max_ratio)

        # 3. 过滤反光衣/安全背心框（不对面积做严格限制，避免漏掉小目标）
        vest_boxes = label_filter(predictions, list(self.VEST_LABELS))
        vest_boxes = score_filter(vest_boxes, self.clothes_min_score)

        # 4. 过滤红色安全帽框
        red_hat_boxes = label_filter(predictions, list(self.RED_HAT_LABELS))
        red_hat_boxes = score_filter(red_hat_boxes, self.clothes_min_score)

        # 5. 判定违规
        violators = []
        for person in persons:
            # 必须与普通衣物有交集
            if not self._has_overlap(person, clothes_boxes):
                continue

            # 不能与反光衣/安全背心有交集
            if self._has_overlap(person, vest_boxes):
                continue

            # 不能与红色安全帽有交集（已做安全着装时排除）
            if self._has_overlap(person, red_hat_boxes):
                continue

            logger.warning(
                f"[device={device_id}] 未穿反光衣/工服 | "
                f"box={person.box}, score={person.score:.3f}"
            )
            violators.append(person)

        return violators
